"""Process, output, disk, and elapsed-time limits for a run attempt."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable


class ResourceViolation(RuntimeError):
    """A configured resource limit was exceeded."""


# How many consecutive samples may miss a live work group before the run stops.
# At the 0.2s poll interval this is about two seconds.
GROUP_MISSING_TOLERANCE = 10


@dataclass(frozen=True)
class ResourceLimits:
    max_rss_bytes: int = 3 * 1024 * 1024 * 1024
    max_output_bytes: int = 12 * 1024 * 1024 * 1024
    initial_free_bytes: int = 14 * 1024 * 1024 * 1024
    min_free_bytes: int = 2 * 1024 * 1024 * 1024
    max_elapsed_seconds: float = 3 * 60 * 60
    poll_seconds: float = 0.2


def _ps_process_table() -> dict[int, tuple[int, int, int]] | None:
    """Read the process table with ``ps``; ``None`` when ``ps`` is unavailable."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss="],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # ps could not be executed at all; the caller may fall back to libproc.
        return None
    output = result.stdout or ""
    if not output.strip():
        # ps ran and said nothing.  That is an empty table, not a missing tool,
        # so it must not be papered over by another reader.
        raise ResourceViolation("process table unavailable: empty process table")
    table: dict[int, tuple[int, int, int]] = {}
    valid_rows = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            pid, parent, group, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        valid_rows += 1
        table[pid] = (parent, group, max(rss_kib, 0))
    if valid_rows == 0:
        return None
    return table or None


def _libproc_process_table() -> tuple[dict[int, tuple[int, int, int | None]], dict[str, object]] | None:
    """Read the same process facts through libproc when ``ps`` cannot be run.

    This is a Darwin-only compatibility reader for the same kernel facts: pid,
    parent pid, process group, and resident size.  It is never used on other
    platforms, it never fabricates a value, and any ABI mismatch, missing
    symbol, or unusable row set returns ``None`` so the caller fails closed.

    Identity enumeration and the RSS read are separate: a pid whose BSD record
    was read but whose task record was not stays in the table with
    ``rss=None`` and is reported in ``problems["rss_unreadable"]``.  A pid
    whose BSD record could not be read is not in the table at all and is
    counted under ``problems["identity_unreadable"]``; such a pid is treated as
    exited, and its identity can therefore not be used to confirm a group.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
    except Exception:  # pragma: no cover - ctypes ships with the standard library
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib", use_errno=True)
        word = ctypes.c_uint32

        class _bsd(ctypes.Structure):
            _fields_ = [
                ("flags", word), ("status", word), ("xstatus", word),
                ("pid", word), ("ppid", word),
                ("uid", word), ("gid", word), ("ruid", word), ("rgid", word),
                ("svuid", word), ("svgid", word), ("rfu1", word),
                ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32),
                ("nfiles", word), ("pgid", word), ("pjobc", word),
                ("e_tdev", word), ("e_tpgid", word), ("length", ctypes.c_int32),
                ("sel", ctypes.c_int32), ("start", ctypes.c_uint64),
            ]

        class _task(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
                ("threads_user", ctypes.c_int64), ("threads_system", ctypes.c_int64),
                ("policy", ctypes.c_int32), ("faults", ctypes.c_int32),
                ("pageins", ctypes.c_int32), ("cow_faults", ctypes.c_int32),
                ("messages_sent", ctypes.c_int32), ("messages_received", ctypes.c_int32),
                ("syscalls_mach", ctypes.c_int32), ("syscalls_unix", ctypes.c_int32),
                ("csw", ctypes.c_int32), ("threadnum", ctypes.c_int32),
                ("numrunning", ctypes.c_int32), ("priority", ctypes.c_int32),
            ]

        libc.proc_listallpids.restype = ctypes.c_int
        libc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libc.proc_pidinfo.restype = ctypes.c_int
        libc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]

        count = libc.proc_listallpids(None, 0)
        if count <= 0:
            return None
        buffer = (ctypes.c_int * (count + 256))()
        listed = libc.proc_listallpids(buffer, ctypes.sizeof(buffer))
        if listed <= 0:
            return None
        table: dict[int, tuple[int, int, int | None]] = {}
        identity_unknown: dict[int, str] = {}
        identity_exited: list[int] = []
        rss_unreadable: dict[int, str] = {}
        bsd, task = _bsd(), _task()
        for raw in buffer[:listed]:
            pid = int(raw)
            if pid <= 0:
                continue
            # A different structure size means a different ABI: fail closed
            # rather than reading fields at the wrong offsets.  The failure is
            # recorded against the PID: a missing identity is never proof that
            # the process exited (EPERM and ABI mismatch look identical), so
            # the PID stays unknown until a liveness probe says otherwise.
            ctypes.set_errno(0)
            bsd_rc = libc.proc_pidinfo(pid, 3, 0, ctypes.byref(bsd), ctypes.sizeof(bsd))
            if bsd_rc != ctypes.sizeof(bsd):
                bsd_errno = ctypes.get_errno()
                if bsd_errno == 3:  # ESRCH: the kernel no longer reports a
                    # process here.  A reaped pid and a zombie both look like
                    # this (measured: a zombie returns rc=0/errno=3 while
                    # kill(pid,0) still succeeds), so this is the one signal
                    # that means exited.
                    identity_exited.append(pid)
                else:
                    identity_unknown[pid] = (
                        f"proc_pidinfo(PROC_PIDTBSDINFO) returned {bsd_rc} with errno={bsd_errno}"
                    )
                continue
            task_size = libc.proc_pidinfo(pid, 4, 0, ctypes.byref(task), ctypes.sizeof(task))
            if task_size != ctypes.sizeof(task):
                # A process caught mid-exit can fail one task read and succeed
                # the next; retry once so a transient race is not recorded as a
                # hard unreadable.
                ctypes.set_errno(0)
                task_size = libc.proc_pidinfo(pid, 4, 0, ctypes.byref(task), ctypes.sizeof(task))
            if task_size == ctypes.sizeof(task):
                rss_kib: int | None = max(int(task.resident_size) // 1024, 0)
            else:
                # The process is alive and identified, but its resident size is
                # not readable.  Record it as unavailable instead of zero.
                rss_kib = None
                rss_unreadable[pid] = "task info unavailable"
            table[pid] = (int(bsd.ppid), int(bsd.pgid), rss_kib)
        if not table:
            return None
        for pid in (os.getpid(),):
            if pid not in table:
                # The reader cannot even see the process asking, so the table
                # is not trustworthy.
                return None
        problems: dict[str, object] = {
            "rss_unreadable": rss_unreadable,
            "identity_unknown": identity_unknown,
            "identity_exited": identity_exited,
        }
        return table, problems
    except Exception:
        return None


def _fixture_inject() -> str:
    return os.environ.get("REPRO_FIXTURE_INJECT", "")


FIXTURE_ENV_VARS = (
    "REPRO_FIXTURE_INJECT",
    "REPRO_FIXTURE_HANDSHAKE",
    "REPRO_FIXTURE_MASK_FILE",
    "REPRO_FIXTURE_FOREIGN_MODULE",
)


def fixture_control_state() -> dict[str, str]:
    """Return the fixture controls currently present in the environment."""
    return {name: os.environ[name] for name in FIXTURE_ENV_VARS if os.environ.get(name)}


def assert_fixture_controls_allowed(profile: object) -> dict[str, str]:
    """Refuse fixture controls unless the run uses a synthetic fixture profile.

    ``REPRO_FIXTURE_*`` variables are test controls.  A run with a real,
    missing, or unknown profile must not be able to carry them into a
    supervised work group, a fixture file write, or a fit, so this raises
    before any of those happen.  A synthetic fixture profile is accepted and
    the active controls are returned so they can be bound into the request.
    """
    active = fixture_control_state()
    if not active:
        return {}
    if isinstance(profile, str) and profile.startswith("synthetic_fixture"):
        return active
    raise ResourceViolation(
        f"fixture controls require a synthetic fixture profile: profile={profile!r}, "
        f"controls={sorted(active)}"
    )


def _fixture_override(
    snapshot: dict[str, int | float], limits: ResourceLimits
) -> dict[str, int | float]:
    """Fixture-only limit overrides so a real CLI run reaches each limit.

    A real run never sets ``REPRO_FIXTURE_INJECT``, so this returns the
    snapshot unchanged.  The overrides feed the real comparison logic: they do
    not bypass it, and they never lower a threshold.
    """
    inject = _fixture_inject()
    if not inject:
        return snapshot
    overrides = {
        "rss_overflow": {"rss_bytes": limits.max_rss_bytes + 1},
        "output_overflow": {"owned_bytes": limits.max_output_bytes + 1},
        "free_space_low": {"free_bytes": 0},
        "elapsed_overflow": {"elapsed_seconds": limits.max_elapsed_seconds + 1.0},
    }
    if inject in overrides:
        snapshot.update(overrides[inject])
    return snapshot


def _read_process_table() -> tuple[dict[int, tuple[int, int, int | None]], str, str | None, dict[str, object]]:
    """Return ``(table, reader, failure_reason, problems)`` for one attempt.

    ``reader`` is always reported so a manifest can state which source the
    numbers came from.  ``ps`` is preferred; the Darwin ``libproc`` reader is
    only a fallback for environments where ``ps`` cannot be executed, and on
    any other platform there is no fallback at all.  ``problems`` reports
    per-attempt reads that could not be completed; it is never used to fill in
    a value.
    """
    inject = _fixture_inject()
    if inject == "sampling_failed":
        return {}, "fixture", "fixture: process table read failed", {}
    if inject == "sampling_empty":
        raise ResourceViolation("process table unavailable: empty process table (fixture)")
    if inject == "sampling_invalid":
        return {}, "fixture", "fixture: process table contained no usable rows", {}
    try:
        table = _ps_process_table()
    except ResourceViolation:
        # ps ran and returned nothing; that is an empty table, not a missing
        # tool, so it must not be papered over by another reader.
        raise
    if table is not None:
        return table, "ps", None, {}
    ps_reason = "ps could not be executed"
    if sys.platform != "darwin":
        return {}, "none", f"{ps_reason}; no compatible fallback reader on {sys.platform}", {}
    read = _libproc_process_table()
    if read is not None:
        table, problems = read
        return table, "libproc", None, problems
    return {}, "none", f"{ps_reason}; libproc fallback unavailable or ABI mismatch", {}


def _process_table() -> dict[int, tuple[int, int, int | None]]:
    """Return pid → (parent pid, process group, RSS KiB) from the process table."""
    table, _reader, reason, _problems = _read_process_table()
    if not table:
        raise ResourceViolation(f"process table unavailable: {reason}")
    return table


def platform_facts() -> dict[str, object]:
    """Describe the platform that the process reader is running on."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "libproc_reader_supported": sys.platform == "darwin",
    }


def _probe_kill0(pid: int) -> str:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "eperm"
    except OSError:
        return "unknown"
    return "alive"


def classify_unknown_pid(pid: int, target_pgid: int | None = None) -> str:
    """Classify a pid whose BSD identity read failed.

    Grounded in the measured ABI behaviour (see the run diagnostics): a zombie
    answers ``proc_pidinfo(BSD)`` with rc=0/errno=ESRCH while ``kill(pid, 0)``
    still succeeds, so the kill probe alone would misclassify a zombie as
    alive.  The classification therefore retries the identity read first:

    * BSD read succeeds and the process group differs from ``target_pgid`` ->
      ``"other_group"`` (the pid can be excluded from this group);
    * BSD read fails with ESRCH, or the retry reports the process gone ->
      ``"exited"``;
    * anything else (EPERM, a live pid inside the target group, an unknown
      unit) stays ``"unresolved"``.
    """
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib", use_errno=True)
            w = ctypes.c_uint32

            class _bsd(ctypes.Structure):
                _fields_ = [
                    ("flags", w), ("status", w), ("xstatus", w), ("pid", w), ("ppid", w),
                    ("uid", w), ("gid", w), ("ruid", w), ("rgid", w), ("svuid", w),
                    ("svgid", w), ("rfu1", w), ("comm", ctypes.c_char * 16),
                    ("name", ctypes.c_char * 32), ("nfiles", w), ("pgid", w),
                    ("pjobc", w), ("e_tdev", w), ("e_tpgid", w), ("length", ctypes.c_int32),
                    ("sel", ctypes.c_int32), ("start", ctypes.c_uint64),
                ]

            libc.proc_pidinfo.restype = ctypes.c_int
            libc.proc_pidinfo.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
            ]
            bsd = _bsd()
            ctypes.set_errno(0)
            rc = libc.proc_pidinfo(pid, 3, 0, ctypes.byref(bsd), ctypes.sizeof(bsd))
            if rc == ctypes.sizeof(bsd):
                if target_pgid is not None and int(bsd.pgid) != target_pgid:
                    return "other_group"
                return "unresolved"
            bsd_errno = ctypes.get_errno()
            if bsd_errno == 3:  # ESRCH, including a zombie
                return "exited"
            # A permission error carries no UID or PGID evidence and proves
            # nothing about membership; fall through to the getpgid probe.
        except Exception:
            pass
    return _classify_by_getpgid(pid, target_pgid)


def _classify_by_getpgid(pid: int, target_pgid: int | None) -> str:
    """Ask the kernel directly which group the pid belongs to.

    ``os.getpgid`` is a different syscall from ``proc_pidinfo``: when it
    succeeds, its answer is the actual membership fact, recorded in the
    classification.  When it fails with EPERM the membership stays unknown.
    """
    import errno

    try:
        actual_pgid = os.getpgid(pid)
    except ProcessLookupError:
        return "exited"
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return "exited"
        return "unresolved"
    if target_pgid is not None:
        if actual_pgid != target_pgid:
            return "other_group"
        return "target_member"
    return "unresolved"


def inspect_group_identity(pgid: int) -> tuple[list[int] | None, list[int], str | None]:
    """Inspect a group and report which pids have unreadable identity.

    Returns ``(visible_members, unknown_identity_pids, error)``.  ``unknown``
    lists the pids the reader could list but not identify; they can never be
    excluded from the group by the caller, so a cleanup that finds an empty
    member list while ``unknown`` is non-empty must not report a clean group.
    """
    try:
        table, _reader, _reason, problems = _read_process_table()
    except ResourceViolation as exc:
        return None, [], str(exc)
    unknown = sorted(int(pid) for pid in problems.get("identity_unknown", {}))
    if not table:
        if unknown:
            # Every listed pid failed its identity read: the membership answer
            # is an empty list plus the unknowns, not a phantom clean group.
            return [], unknown, None
        return None, [], "process table unavailable"
    members = sorted(pid for pid, (_parent, group, _rss) in table.items() if group == pgid)
    return members, unknown, None


def process_group_members(pgid: int) -> tuple[list[int] | None, str | None]:
    """Return members of a process group; ``None`` means inspection failed.

    Unknown-identity pids are classified here: an empty member list is only
    returned as clean when every unknown pid can be excluded (exited or
    confirmed in another group).  A pid confirmed to belong to this group, and
    any unresolved pid, make the whole inspection unavailable, so a
    compatibility caller cannot mistake them for a clean, empty group.
    """
    members, unknown, error = inspect_group_identity(pgid)
    if members is None:
        return None, error
    blocking = [
        pid for pid in unknown
        if classify_unknown_pid(pid, pgid) in ("unresolved", "target_member")
    ]
    if blocking:
        return None, (
            f"cannot confirm the group is empty: unknown-identity pids {blocking} "
            "cannot be excluded"
        )
    return members, None


def _process_tree_rss_bytes(root_pid: int | None = None, *, include_descendants: bool = True) -> int:
    """Sum RSS over a parent-child tree, failing when any member is unreadable.

    Every process found by walking the PPID chain from ``root_pid`` belongs to
    this sampling range regardless of depth, so a descendant whose resident
    size cannot be read makes the whole measurement unavailable -- it is never
    excluded for being distant and never substituted with zero.  Callers that
    must sample only the supervisor itself pass
    ``include_descendants=False``; that is an explicit scope choice, not a way
    to drop descendants from a tree measurement.
    """
    root_pid = os.getpid() if root_pid is None else root_pid
    table, _reader, _reason, _problems = _read_process_table()
    if not table:
        raise ResourceViolation("process table unavailable")
    if root_pid not in table:
        raise ResourceViolation(f"process table unavailable: root pid {root_pid} is missing")
    if not include_descendants:
        if table[root_pid][2] is None:
            raise ResourceViolation(
                f"rss unavailable for the supervisor process {root_pid}; refusing to substitute zero"
            )
        return int(table[root_pid][2]) * 1024
    children: dict[int, list[int]] = {}
    for pid, (parent, _group, _rss) in table.items():
        children.setdefault(parent, []).append(pid)
    pids = {root_pid}
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        for child in children.get(parent, []):
            if child not in pids:
                pids.add(child)
                queue.append(child)
    unreadable = sorted(pid for pid in pids if table[pid][2] is None)
    if unreadable:
        raise ResourceViolation(
            f"rss unavailable for sampled tree members {unreadable}; refusing to substitute zero"
        )
    return sum(int(table[pid][2]) for pid in pids) * 1024


def _group_rss_bytes_from(
    table: dict[int, tuple[int, int, int | None]],
    group_pgid: int | None,
    extra_pids: tuple[int, ...] = (),
) -> tuple[int | None, list[int], list[int]]:
    """Sum RSS over a group and explicit pids using an already-read table.

    A sampled member whose resident size could not be read makes the whole
    reading unavailable: zero is never substituted.  Returns
    ``(rss_or_none, sampled_pids, unreadable_pids)``.
    """
    members: set[int] = set()
    if group_pgid is not None:
        members = {pid for pid, (_parent, group, _rss) in table.items() if group == group_pgid}
        if not members:
            raise ResourceViolation(
                f"process group sampling unavailable: no member visible for pgid {group_pgid}"
            )
    for pid in extra_pids:
        if pid not in table:
            raise ResourceViolation(f"process group sampling unavailable: pid {pid} is missing")
        members.add(pid)
    if not members:
        raise ResourceViolation("process group sampling unavailable: no sampled process")
    # Every member of the registered group is in sampling range, so an
    # unreadable resident size here makes the reading unavailable.  Processes
    # outside the group are never sampled and never block it.
    unreadable = sorted(pid for pid in members if table[pid][2] is None)
    if unreadable:
        raise ResourceViolation(
            f"rss unavailable for sampled group members {unreadable}; refusing to substitute zero"
        )
    return sum(int(table[pid][2]) for pid in members) * 1024, sorted(members), unreadable


def _group_rss_bytes(
    group_pgid: int | None, extra_pids: tuple[int, ...] = ()
) -> tuple[int | None, list[int], list[int]]:
    """Sum RSS over a registered process group plus explicit extra pids.

    Members are found by group id, not by following parent links, so a process
    that outlived its direct parent is still counted.  An empty member list for
    a registered group is an inspection failure, never a zero reading.
    """
    return _group_rss_bytes_from(_process_table(), group_pgid, extra_pids)


def _owned_bytes(path: Path, *, reject_symlinks: bool = False) -> int:
    total = 0
    if reject_symlinks and path.is_symlink():
        raise ResourceViolation(f'symlink in output budget: {path}')
    if not path.exists():
        return total
    if path.is_file(): return path.stat().st_size
    try:
        for item in path.rglob("*"):
            if reject_symlinks and item.is_symlink():
                raise ResourceViolation(f'symlink in output budget: {item}')
            if item.is_symlink() or not item.is_file():
                continue
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                # A transient partial file can disappear between listing and
                # stat.  That is a race, not an unusable output tree.
                continue
    except FileNotFoundError:
        return total
    except OSError as exc:
        raise ResourceViolation(f"output tree unavailable: {path}: {exc}") from exc
    return total


class ResourceGuard:
    """Poll limits in the parent while a stage process is running."""

    def __init__(
        self,
        root: Path,
        output: Path,
        limits: ResourceLimits | None = None,
        *,
        extra_outputs: tuple[Path, ...] = (),
        group_pgid: int | None = None,
        extra_pids: tuple[int, ...] = (),
        reject_output_symlinks: bool = False,
    ) -> None:
        self.root = root
        self.output = output
        self.outputs = tuple(dict.fromkeys((output, *extra_outputs)))
        self.reject_output_symlinks = reject_output_symlinks
        self.limits = limits or ResourceLimits()
        if self.limits.poll_seconds <= 0:
            raise ResourceViolation("resource guard poll interval must be positive")
        self.group_pgid = group_pgid
        self.extra_pids = tuple(extra_pids)
        self.sampled_pids: list[int] = []
        self.expected_members: tuple[int, ...] = ()
        # Recomputed on every sample: which expected members the most recent
        # sample actually saw.  Never a sticky "complete" flag.
        self.expected_members_observed: dict[int, bool] = {}
        self.membership_proven_sample: list[int] | None = None
        self.membership_proven_reader: str | None = None
        self.membership_proven_problems: dict[str, object] = {}
        self.reader: str = "none"
        self.reader_reason: str | None = None
        self.reader_problems: dict[str, object] = {}
        self.sample_count = 0
        self._group_missing_samples = 0
        self.sampling_failures = 0
        self.last_sampling_error: str | None = None
        self.cleanup_phase = False
        self.cleanup_phase_samples = 0
        self.cleanup_phase_last_members: list[int] = []
        self.observed_group_pids: set[int] = set()
        self.group_released = False
        self.started = time.monotonic()
        self.violation: str | None = None
        self.last_snapshot: dict[str, int | float] = {}
        self.peak_snapshot: dict[str, int | float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def register_group(self, pgid: int) -> None:
        with self._lock:
            self.group_pgid = pgid

    def prepare_next_group(self) -> None:
        """Keep budgets and elapsed time while starting a new verified-empty stage."""
        with self._lock:
            if not self.group_released:
                raise ResourceViolation('previous process group not confirmed cleaned')
            self.group_pgid = None
            self.group_released = False
            self.cleanup_phase = False
            self.expected_members = ()
            self.expected_members_observed = {}
            self.membership_proven_sample = None
            self.observed_group_pids = set()
            self._group_missing_samples = 0

    def register_pid(self, pid: int) -> None:
        with self._lock:
            if pid not in self.extra_pids:
                self.extra_pids = (*self.extra_pids, pid)

    def expect_member(self, pid: int) -> None:
        """Record a pid that sampling must actually observe in the group.

        Observation is recomputed on every sample; there is no sticky
        "complete" flag, because one sighting does not prove that every
        later sample would still see the member.
        """
        with self._lock:
            if pid not in self.expected_members:
                self.expected_members = (*self.expected_members, pid)

    def begin_cleanup(self) -> None:
        """Enter the bounded-cleanup phase.

        The PGID stays registered, so surviving members keep being counted
        after the leader exits.  An empty group is only accepted in this phase,
        and only because the caller is explicitly confirming the end of the
        group; a failed read is still never treated as an empty group.
        """
        with self._lock:
            self.cleanup_phase = True

    def release_group(self) -> None:
        """Stop sampling a work group whose end has been confirmed."""
        with self._lock:
            self.group_released = True

    def sampling_scope(self) -> dict[str, object]:
        with self._lock:
            return {
                "group_pgid": self.group_pgid,
                "extra_pids": list(self.extra_pids),
                "expected_members": list(self.expected_members),
                "poll_seconds": self.limits.poll_seconds,
                "sampled_pids": list(self.sampled_pids),
                "sample_count": self.sample_count,
                "sampling_failures": self.sampling_failures,
                "last_sampling_error": self.last_sampling_error,
                "note": "observed peaks over the poll interval, not instantaneous peaks",
            }

    @staticmethod
    def _json_stable_problems(problems: dict[str, object]) -> dict[str, object]:
        """Return a frozen, JSON-stable copy of a problems record.

        PID-keyed dicts are converted to string keys: JSON turns integer keys
        into strings, and the integer sort order and the string sort order
        differ once the pids have different digit counts, which would make a
        manifest self-hash depend on the serialisation round-trip.
        """
        frozen: dict[str, object] = {}
        for key, value in problems.items():
            if isinstance(value, dict):
                frozen[key] = {str(inner_key): inner for inner_key, inner in value.items()}
            else:
                frozen[key] = value
        return frozen

    def reader_facts(self) -> dict[str, object]:
        """Report which process reader produced the numbers, and why."""
        with self._lock:
            return {
                "reader": self.reader,
                "reader_reason": self.reader_reason,
                "reader_problems": self._json_stable_problems(self.reader_problems),
                "platform": platform_facts(),
                "expected_members": list(self.expected_members),
                "expected_members_observed": dict(self.expected_members_observed),
                "membership_proven_sample": list(self.membership_proven_sample)
                if self.membership_proven_sample is not None
                else None,
                "membership_proven_reader": self.membership_proven_reader,
                "membership_proven_problems": self._json_stable_problems(
                    self.membership_proven_problems
                ),
                "sampled_pids": list(self.sampled_pids),
                "sample_count": self.sample_count,
                "sampling_failures": self.sampling_failures,
                "cleanup_phase_samples": self.cleanup_phase_samples,
                "cleanup_phase_last_members": list(self.cleanup_phase_last_members),
                "observed_group_pids": sorted(self.observed_group_pids),
                "group_released": self.group_released,
            }

    def snapshot(self) -> dict[str, int | float]:
        try:
            free_bytes = shutil.disk_usage(self.root).free
        except OSError as exc:
            raise ResourceViolation(f"disk usage unavailable: {self.root}: {exc}") from exc
        with self._lock:
            group, extra = self.group_pgid, self.extra_pids
            last_rss = int(self.last_snapshot.get("rss_bytes", 0))
            expected = self.expected_members
            cleanup_phase = self.cleanup_phase
            released = self.group_released
        tree_excluded: list[int] = []
        tree_excluded: list[int] = []
        if group is None and not extra:
            table, reader, reason, problems = _read_process_table()
            if not table:
                raise ResourceViolation(f"process table unavailable: {reason}")
            self.reader, self.reader_reason, self.reader_problems = reader, reason, problems
            rss_bytes = _process_tree_rss_bytes()
            sampled = [os.getpid()]
            rss_sampled = True
            group_state = "no_group"
        elif released:
            # The end of the group has been confirmed by bounded cleanup.  Not
            # sampling after that point is not a reading of zero; the last
            # observed value is carried forward and flagged as unsampled.
            rss_bytes, sampled = last_rss, []
            rss_sampled = False
            group_state = "released_after_cleanup_confirmed"
        else:
            # The PGID stays registered through the cleanup phase, so members
            # that outlive the leader are still counted (PLAN §3.7).
            try:
                table, reader, reason, problems = _read_process_table()
                if not table:
                    raise ResourceViolation(f"process table unavailable: {reason}")
                self.reader, self.reader_reason, self.reader_problems = reader, reason, problems
                rss_bytes, sampled, _unreadable = _group_rss_bytes_from(table, group, extra)
                self._group_missing_samples = 0
                group_state = "cleanup_confirming" if cleanup_phase else "running"
                with self._lock:
                    self.sampled_pids = sampled
                    self.observed_group_pids.update(sampled)
                    self.expected_members_observed = {pid: pid in sampled for pid in expected}
                    if expected and all(pid in sampled for pid in expected):
                        # The most recent sample that saw every expected member.
                        # Recomputed on every sample; never a sticky flag.
                        # Unreadable identity records elsewhere in the table are
                        # reported separately and do not erase this sighting,
                        # because such a pid is treated as exited, not as a
                        # hidden member of the group.
                        self.membership_proven_sample = list(sampled)
                        self.membership_proven_reader = reader
                        self.membership_proven_problems = dict(problems)
                if cleanup_phase:
                    self.cleanup_phase_samples += 1
                    self.cleanup_phase_last_members = list(sampled)
                rss_sampled = True
            except ResourceViolation as exc:
                if cleanup_phase:
                    # During cleanup a vanished member is the expected outcome,
                    # so an empty group is a confirmed state here, not a zero
                    # that hides a failed read.  A read failure is not an empty
                    # group and still stops the run.
                    message = str(exc)
                    if "no member visible" in message:
                        rss_bytes, sampled = 0, []
                        rss_sampled = True
                        group_state = "empty_confirmed_by_cleanup"
                    else:
                        raise
                else:
                    # A live group that is momentarily not visible (exec window,
                    # table refresh) keeps the last observed reading and is
                    # marked unsampled.  Only a sustained absence stops the
                    # run, so a transient gap can never look like a clean,
                    # empty group.
                    self._group_missing_samples += 1
                    if self._group_missing_samples <= GROUP_MISSING_TOLERANCE:
                        rss_bytes, sampled = last_rss, []
                        rss_sampled = False
                        self.last_sampling_error = str(exc)
                        group_state = "temporarily_unobserved"
                    else:
                        raise
        snapshot: dict[str, int | float] = {
            "rss_bytes": rss_bytes,
            "owned_bytes": sum(_owned_bytes(path,reject_symlinks=self.reject_output_symlinks) for path in self.outputs),
            "free_bytes": free_bytes,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "sampled_processes": len(sampled),
            "rss_sampled": rss_sampled,
            "group_state": group_state,
        }
        snapshot = _fixture_override(snapshot, self.limits)
        with self._lock:
            self.sampled_pids = sampled
            self.last_snapshot = snapshot
            self.sample_count += 1
            if not self.peak_snapshot:
                self.peak_snapshot = {key: value for key, value in snapshot.items()}
            else:
                merged = {
                    "rss_bytes": max(int(self.peak_snapshot["rss_bytes"]), int(snapshot["rss_bytes"])),
                    "owned_bytes": max(int(self.peak_snapshot["owned_bytes"]), int(snapshot["owned_bytes"])),
                    "free_bytes": min(int(self.peak_snapshot["free_bytes"]), int(snapshot["free_bytes"])),
                    "elapsed_seconds": max(float(self.peak_snapshot["elapsed_seconds"]), float(snapshot["elapsed_seconds"])),
                    "sampled_processes": max(int(self.peak_snapshot.get("sampled_processes", 0)), int(snapshot["sampled_processes"])),
                }
                self.peak_snapshot = merged
        return snapshot

    def _record_sampling_failure(self, error: BaseException) -> None:
        with self._lock:
            self.sampling_failures += 1
            self.last_sampling_error = f"{type(error).__name__}: {error}"
            if self.violation is None:
                self.violation = f"resource sampling failed: {self.last_sampling_error}"
            self._stop.set()

    def check(self) -> dict[str, int | float]:
        snapshot = self.snapshot()
        reasons: list[str] = []
        if int(snapshot["rss_bytes"]) > self.limits.max_rss_bytes:
            reasons.append("rss")
        if int(snapshot["owned_bytes"]) > self.limits.max_output_bytes:
            reasons.append("output")
        if int(snapshot["free_bytes"]) < self.limits.min_free_bytes:
            reasons.append("free-space")
        if float(snapshot["elapsed_seconds"]) > self.limits.max_elapsed_seconds:
            reasons.append("elapsed")
        with self._lock:
            if reasons and self.violation is None:
                self.violation = ",".join(reasons) + f": {snapshot}"
                self._stop.set()
        return snapshot

    def check_or_raise(self) -> dict[str, int | float]:
        snapshot = self.check()
        if self.violation is not None:
            raise ResourceViolation(self.violation)
        return snapshot

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                self.check()
            except BaseException as exc:
                self.violation = f"resource guard failed: {type(exc).__name__}: {exc}"
                self._stop.set()
                return
            self._stop.wait(self.limits.poll_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise ResourceViolation("resource guard already started")
        self._thread = threading.Thread(target=self._poll, name="repro-resource-guard", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.limits.poll_seconds * 5))
            if self._thread.is_alive() and self.violation is None:
                self.violation = "resource guard failed: polling thread did not stop"
        try:
            return self.snapshot()
        except BaseException as exc:
            if self.violation is None:
                self.violation = f"resource guard failed: {type(exc).__name__}: {exc}"
            return dict(self.last_snapshot)


__all__ = ["ResourceGuard", "ResourceLimits", "ResourceViolation"]
