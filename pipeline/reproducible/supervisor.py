"""External supervisor for the train → score → evaluate workflow.

The supervisor stays outside the work group.  It starts one leader/launcher in
a fresh process group, receives the snapshot digest through a small readiness
record, and only then authorises the coordinator to run from that snapshot.
Publication happens at a single commit decision point: the final manifest is
serialised and fsynced beforehand, then published with one atomic link that
never replaces an existing target.
"""

from __future__ import annotations

from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from .artifacts import (
    ArtifactError,
    _canonical,
    artifact_facts,
    bound_path,
    sha256_file,
    write_json_exclusive,
    write_verified_manifest,
)
from .resource_guard import (
    ResourceGuard,
    ResourceLimits,
    ResourceViolation,
    assert_fixture_controls_allowed,
    classify_unknown_pid,
    inspect_group_identity,
    process_group_members,
)
from .runtime_context import code_root, project_root
from .source_snapshot import SourceSnapshotError, verify_source_snapshot


class SupervisorError(RuntimeError):
    """A supervised run was cancelled, exceeded a limit, or failed to publish."""


ROOT = project_root()
CODE_ROOT = code_root()
REQUEST_VERSION = "supervised-request-v2"
COORDINATOR_REQUEST_VERSION = "supervised-coordinator-request-v2"
LAUNCH_REQUEST_VERSION = "supervised-launch-request-v1"
BOOTSTRAP_RELATIVE = "pipeline/reproducible/supervisor_bootstrap.py"
STAGED_NAME = "isolated_run_manifest.staged.json"
FINAL_NAME = "isolated_run_manifest.json"
POLL_SECONDS = 0.05


def _reject_constant(value: str) -> object:
    raise SupervisorError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SupervisorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_small(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SupervisorError(f"{label} must be an object: {path}")
    return payload


def _handshake(stage: str) -> None:
    """Fixture-only synchronisation point; unused unless a fixture enables it."""
    import os as _os
    from pathlib import Path as _Path

    directory = _os.environ.get("REPRO_FIXTURE_HANDSHAKE")
    if not directory:
        return
    base = _Path(directory)
    (base / f"reached.{stage}").write_text(str(_os.getpid()), encoding="utf-8")
    release = base / f"release.{stage}"
    deadline = time.monotonic() + 60.0
    while not release.is_file() and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)


def _read_config_profile(config: Path | str) -> object:
    """Read the locked profile name from a configuration file."""
    try:
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload.get("profile")
    return None


def _inspect_group_identity(
    pgid: int, *, cleanup: bool = False
) -> tuple[list[int] | None, list[int], str | None]:
    """Inspect a group without raising.

    Returns ``(visible_members, unknown_identity_pids, error)``.  ``unknown``
    holds the pids the reader could list but not identify; the cleanup below
    must consume them instead of treating an empty member list as clean.
    """
    if cleanup and os.environ.get("REPRO_FIXTURE_INJECT") == "cleanup_query_fails":
        return None, [], "fixture: cleanup inspection failed"
    return inspect_group_identity(pgid)


def _resolve_unknown_identity(
    unknown: list[int], known_members: set[int], record: dict[str, object], target_pgid: int
) -> list[int]:
    """Split unreadable-identity pids into excludable and unresolved.

    A pid is excluded only on kernel evidence: ESRCH on the identity read (a
    zombie answers this way too), or ``os.getpgid`` confirming the pid sits in
    another process group.  A pid that ``getpgid`` places in the *target* group
    is a real member and stays unresolved; a permission error carries no
    membership evidence at all, so it also stays unresolved.  Historical
    sightings never decide the outcome.
    """
    unresolved: list[int] = []
    known = set(known_members or ())
    for pid in unknown:
        classification = classify_unknown_pid(pid, target_pgid)
        fact = {"pid": pid, "classification": classification, "previously_seen": pid in known}
        if classification in ("exited", "other_group"):
            record.setdefault("excluded_unknown_pids", []).append(fact)
            continue
        # "target_member" means the kernel placed this pid in our group while
        # its identity read failed: it is a real member that must not vanish,
        # and the run cannot claim a clean group while it remains.
        unresolved.append(fact)
    if unresolved:
        record["unknown_identity_pids"] = unresolved
    return unresolved


def _supports_sigmask() -> bool:
    return hasattr(signal, "pthread_sigmask") and hasattr(signal, "SIG_SETMASK")


def _current_mask() -> object | None:
    if not _supports_sigmask():
        return None
    return signal.pthread_sigmask(signal.SIG_BLOCK, set())


def _pending_cancel_signals() -> list[str]:
    if not hasattr(signal, "sigpending"):
        return []
    try:
        pending = signal.sigpending()
    except (AttributeError, OSError):
        return []
    return [signum.name for signum in (signal.SIGTERM, signal.SIGINT) if signum in pending]


def _inspect_group(pgid: int, *, cleanup: bool = False) -> tuple[list[int] | None, str | None]:
    """Inspect a group without raising; ``None`` means inspection failed."""
    if cleanup and os.environ.get("REPRO_FIXTURE_INJECT") == "cleanup_query_fails":
        return None, "fixture: cleanup inspection failed"
    try:
        return _process_group_members(pgid), None
    except SupervisorError as exc:
        return None, str(exc)


def _fixture_mask_record(stage: str, mask: object) -> None:
    """Write the signal mask to a fixture file; unused unless a fixture asks."""
    path = os.environ.get("REPRO_FIXTURE_MASK_FILE")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{stage}:{sorted(int(s) for s in mask) if mask is not None else None}\n")
    except OSError:
        pass


def _process_group_members(pgid: int) -> list[int]:
    """Return members of a process group, raising when inspection fails.

    An unreadable table is an error, never an empty group that would look
    clean.
    """
    members, error = process_group_members(pgid)
    if members is None:
        raise SupervisorError(f"cannot inspect supervised process group: {error}")
    return members


def _bounded_cleanup(
    pgid: int,
    process: subprocess.Popen[str] | None,
    known_members: set[int] | None = None,
) -> dict[str, object]:
    """Terminate the registered group with a bounded TERM → KILL → recheck.

    ``known_members`` is every pid the sampling ever saw inside the group.  A
    pid with an unreadable identity is only excluded from the group when it was
    never observed there and a liveness probe confirms it exited; otherwise the
    cleanup stays ``unknown`` instead of claiming a clean, empty group.
    """
    record: dict[str, object] = {
        "pgid": pgid,
        "state": "incomplete",
        "members": [],
        "errors": [],
        "unknown_identity_pids": [],
        "resolved_signal_races": [],
        "term_seconds_budget": 5.0,
        "kill_seconds_budget": 5.0,
    }
    known = set(known_members or ())

    def inspect_group():
        # Reap our direct child before enumerating the group. Otherwise ps can
        # keep reporting its zombie until the TERM budget expires.
        if process is not None:
            process.poll()
        return _inspect_group_identity(pgid, cleanup=True)

    def settle(state_hint: str) -> dict[str, object]:
        members, unknown, inspection_error = inspect_group()
        if members is None:
            record["state"] = "unknown"
            record["errors"].append(f"final inspection: {inspection_error}")
            return record
        if members:
            record["state"] = "incomplete"
            record["members"] = members
            return record
        unresolved = _resolve_unknown_identity(unknown, known, record, pgid)
        record["unknown_identity_pids"] = unresolved
        if unresolved:
            record["state"] = "unknown"
            record["errors"].append(
                f"unknown-identity pids {unresolved} cannot be excluded from the group"
            )
            return record
        record["state"] = "complete" if not record["errors"] else "unknown"
        return record

    def send_group(sig, label):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            # On Darwin an unreaped, exited group leader can produce EPERM.
            # The errno itself proves nothing: require both child exit and a
            # fresh empty-group query, with no unresolved identities.
            if exc.errno == errno.EPERM and process is not None and process.poll() is not None:
                current, unknown, error = inspect_group()
                if current == [] and error is None:
                    unresolved = _resolve_unknown_identity(unknown, known, record, pgid)
                    if not unresolved:
                        record["resolved_signal_races"].append({
                            "operation":label,"errno":exc.errno,"error":str(exc),
                            "child_returncode":process.returncode,"members_after_reap":[],
                        })
                        return
            record["errors"].append(f"{label}: {exc}")

    members, unknown, inspection_error = inspect_group()
    if members is None:
        # The group identity is ours, so terminate and force-kill are still
        # attempted; the result is recorded as unknown rather than as clean.
        record["state"] = "unknown"
        record["errors"].append(str(inspection_error))
    elif not members:
        unresolved = _resolve_unknown_identity(unknown, known, record, pgid)
        record["unknown_identity_pids"] = unresolved
        if unresolved:
            record["state"] = "unknown"
            record["errors"].append(
                f"unknown-identity pids {unresolved} cannot be excluded from the group"
            )
            # fall through: the bounded TERM/KILL below still runs, because the
            # kernel delivers it to whoever is actually inside our group.
        else:
            record["state"] = "complete"
            return record
    send_group(signal.SIGTERM, "terminate")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        members, unknown, inspection_error = inspect_group()
        if members is None:
            record["errors"].append(f"inspection during cleanup: {inspection_error}")
            break
        if not members:
            unresolved = _resolve_unknown_identity(unknown, known, record, pgid)
            record["unknown_identity_pids"] = unresolved
            if not unresolved:
                record["state"] = "complete" if not record["errors"] else "unknown"
                return record
            record["state"] = "unknown"
            record["errors"].append(
                f"unknown-identity pids {unresolved} cannot be excluded from the group"
            )
            return record
        time.sleep(0.05)
    send_group(signal.SIGKILL, "force-kill")
    if process is not None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            record["errors"].append("coordinator did not exit after SIGKILL")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        members, unknown, inspection_error = inspect_group()
        if members is None:
            record["errors"].append("inspection after kill: {0}".format(inspection_error))
            break
        if not members:
            unresolved = _resolve_unknown_identity(unknown, known, record, pgid)
            record["unknown_identity_pids"] = unresolved
            break
        time.sleep(0.05)
    return settle("after_kill")


def _terminate_group(pgid: int, process: subprocess.Popen[str] | None = None) -> list[int]:
    """Bounded TERM → KILL cleanup; returns the members that survived."""
    record = _bounded_cleanup(pgid, process)
    if record["state"] != "complete":
        raise SupervisorError(
            f"supervised work group cleanup is {record['state']}: {record['errors']} members={record['members']}"
        )
    return list(record["members"])


def _request_payload(
    *,
    output_dir: Path,
    train_panel: Path | str,
    eval_panel: Path | str,
    config: Path | str,
    train_start: str,
    train_end: str,
    train_cutoff: str,
    eval_start: str,
    eval_end: str,
    eval_cutoff: str,
    event_start: str,
    event_end: str,
    timeout_seconds: int,
    fit_timeout_seconds: float,
    limits: ResourceLimits,
) -> dict[str, object]:
    return {
        "version": REQUEST_VERSION,
        "project_root": str(ROOT),
        "train_panel": str(train_panel),
        "eval_panel": str(eval_panel),
        "config": str(config),
        "output_dir": str(output_dir),
        "train_start": train_start,
        "train_end": train_end,
        "train_cutoff": train_cutoff,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "eval_cutoff": eval_cutoff,
        "event_start": event_start,
        "event_end": event_end,
        "timeout_seconds": timeout_seconds,
        "fit_timeout_seconds": fit_timeout_seconds,
        "limits": asdict(limits),
    }


def _write_failure(output: Path, error: BaseException, *, reason: str, extra: dict[str, object] | None = None) -> None:
    payload: dict[str, object] = {
        "status": "failed",
        "scope": "isolated_worker_supervised_v1",
        "reason": reason,
        "error": f"{type(error).__name__}: {error}",
    }
    if extra:
        payload.update(extra)
    try:
        if not (output / "failure.json").exists():
            write_json_exclusive(output / "failure.json", payload)
        if not (output / "supervisor_failure.json").exists():
            write_json_exclusive(output / "supervisor_failure.json", payload)
    except (ArtifactError, OSError):
        pass


def run_supervised(
    *,
    train_panel: Path | str,
    eval_panel: Path | str,
    config: Path | str,
    output_dir: Path | str,
    train_start: str,
    train_end: str,
    train_cutoff: str,
    eval_start: str,
    eval_end: str,
    eval_cutoff: str,
    event_start: str,
    event_end: str,
    timeout_seconds: int = 900,
    fit_timeout_seconds: float = 900,
    limits: ResourceLimits | None = None,
) -> dict[str, object]:
    """Run the coordinator from a prepared snapshot under one supervised group."""
    if timeout_seconds <= 0 or fit_timeout_seconds <= 0:
        raise SupervisorError("supervised timeouts must be positive")
    active_limits = limits or ResourceLimits()
    if active_limits.max_elapsed_seconds <= 0:
        raise SupervisorError("supervised elapsed limit must be positive")
    try:
        output = bound_path(output_dir, "isolated run output")
        if output.exists():
            raise SupervisorError(f"isolated run output already exists: {output}")
        output.mkdir(parents=True)
    except (ArtifactError, OSError) as exc:
        raise SupervisorError(str(exc)) from exc

    guard = ResourceGuard(ROOT, output, active_limits)
    cancel = threading.Event()
    process: subprocess.Popen[str] | None = None
    pgid: int | None = None
    previous: dict[int, object] = {}
    original_mask = None
    cleaned = False
    cleanup_record: dict[str, object] = {"state": "not_started"}
    log_handle = None
    try:
        # Test controls must be refused before a work group, a fixture file,
        # or a fit can start, unless the configuration is a synthetic fixture.
        assert_fixture_controls_allowed(_read_config_profile(config))
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: cancel.set())
        original_mask = _current_mask()
        _fixture_mask_record("before", original_mask)
        guard.register_pid(os.getpid())
        guard.start()
        initial = guard.check_or_raise()
        if int(initial["free_bytes"]) < int(active_limits.initial_free_bytes):
            raise SupervisorError(
                f"initial free space is below supervised limit: {initial['free_bytes']} < {active_limits.initial_free_bytes}"
            )
        if cancel.is_set():
            raise SupervisorError("cancellation requested before the work group started")

        launcher_dir = output / "launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        supervisor_copy = launcher_dir / "supervisor.py"
        shutil.copyfile(Path(__file__).resolve(), supervisor_copy)
        launcher_facts = {
            "role": "launcher_code",
            "note": "outer supervisor; runs from the original checkout, not from the snapshot",
            "path": str(supervisor_copy.relative_to(output)),
            "bytes": supervisor_copy.stat().st_size,
            "sha256": sha256_file(supervisor_copy),
        }

        request = _request_payload(
            output_dir=output,
            train_panel=train_panel,
            eval_panel=eval_panel,
            config=config,
            train_start=train_start,
            train_end=train_end,
            train_cutoff=train_cutoff,
            eval_start=eval_start,
            eval_end=eval_end,
            eval_cutoff=eval_cutoff,
            event_start=event_start,
            event_end=event_end,
            timeout_seconds=timeout_seconds,
            fit_timeout_seconds=fit_timeout_seconds,
            limits=active_limits,
        )
        request_path = output / "supervisor_request.json"
        write_json_exclusive(request_path, request)
        request_facts = artifact_facts(request_path)

        snapshot_dir = output / "source_snapshot"
        launch_request = {
            "version": LAUNCH_REQUEST_VERSION,
            "project_root": str(ROOT),
            "code_root": str(CODE_ROOT),
            "output_dir": str(output),
            "snapshot_dir": str(snapshot_dir),
            "python_executable": str(Path(sys.executable)),
            "coordinator_timeout_seconds": float(timeout_seconds) * 3.0,
            "handshake_timeout_seconds": float(active_limits.max_elapsed_seconds),
            "supervisor_request": str(request_path.relative_to(output)),
            "config_profile": _read_config_profile(config),
        }
        launch_request_path = output / "launch_request.json"
        write_json_exclusive(launch_request_path, launch_request)

        log_path = output / "supervisor.log"
        env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
        env["PATH"] = os.environ.get("PATH", "")
        env["REPRO_PROJECT_ROOT"] = str(ROOT)
        env["REPRO_CODE_ROOT"] = str(CODE_ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[variable] = "1"
        command = [
            str(Path(sys.executable)),
            "-I",
            "-B",
            str(CODE_ROOT / BOOTSTRAP_RELATIVE),
            "--launch-request",
            str(launch_request_path),
        ]
        log_handle = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        pgid = process.pid
        # Wait until the leader is visible in the process table before the
        # guard starts sampling its group, so the start-up window is never
        # mistaken for a group that has already been cleaned up.
        visible_deadline = time.monotonic() + 5.0
        while True:
            members, _unknown, inspection_error = _inspect_group_identity(pgid)
            if members:
                break
            if process.poll() is not None and not members:
                break
            if time.monotonic() > visible_deadline:
                raise SupervisorError(
                    f"supervised work group never became visible: {inspection_error or pgid}"
                )
            time.sleep(POLL_SECONDS)
        guard.register_group(pgid)
        guard.expect_member(pgid)
        overall_deadline = time.monotonic() + min(float(timeout_seconds) * 3.0, float(active_limits.max_elapsed_seconds))

        def _poll_guard(reason_prefix: str) -> None:
            if cancel.is_set():
                raise SupervisorError(f"{reason_prefix}: cancellation requested")
            if guard.violation is not None:
                raise SupervisorError(f"{reason_prefix}: resource guard: {guard.violation}")
            guard.check_or_raise()
            if time.monotonic() > overall_deadline:
                raise SupervisorError(f"{reason_prefix}: supervised coordinator timed out")

        readiness_path = output / "snapshot_readiness.json"
        while not readiness_path.is_file():
            _poll_guard("before source snapshot readiness")
            if process.poll() is not None:
                raise SupervisorError(
                    f"work group leader exited with code {process.returncode} before the snapshot was ready; see {log_path}"
                )
            time.sleep(min(POLL_SECONDS, max(0.0, overall_deadline - time.monotonic())) or POLL_SECONDS)

        readiness = _load_small(readiness_path, "snapshot readiness record")
        snapshot_digest = readiness.get("source_manifest_sha256")
        if type(snapshot_digest) is not str or len(snapshot_digest) != 64:
            raise SupervisorError("snapshot readiness record has no usable digest")
        expectation = readiness.get("expectation")
        if not isinstance(expectation, dict):
            raise SupervisorError("snapshot readiness record has no external expectation")
        _poll_guard("after source snapshot readiness")

        coordinator_request = {
            "version": COORDINATOR_REQUEST_VERSION,
            "project_root": str(ROOT),
            "code_root": str(snapshot_dir),
            "source_manifest_sha256": snapshot_digest,
            "train_panel": str(train_panel),
            "eval_panel": str(eval_panel),
            "config": str(config),
            "output_dir": str(output),
            "train_start": train_start,
            "train_end": train_end,
            "train_cutoff": train_cutoff,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "eval_cutoff": eval_cutoff,
            "event_start": event_start,
            "event_end": event_end,
            "timeout_seconds": timeout_seconds,
            "fit_timeout_seconds": fit_timeout_seconds,
            "limits": asdict(active_limits),
        }
        write_json_exclusive(output / "coordinator_request.json", coordinator_request)

        while process.poll() is None:
            _poll_guard("during supervised execution")
            time.sleep(min(POLL_SECONDS, max(0.0, overall_deadline - time.monotonic())) or POLL_SECONDS)
        return_code = process.returncode
        # Sampling keeps counting the registered PGID after the leader exits
        # (PLAN §3.7): the group is only released once bounded cleanup has
        # confirmed its end.
        guard.begin_cleanup()
        cleanup_record = _bounded_cleanup(pgid, process, known_members=guard.observed_group_pids)
        cleaned = True
        cleanup_phase_snapshot = guard.stop()
        guard.release_group()
        if cleanup_record["state"] != "complete":
            raise SupervisorError(
                f"supervised work group was not clean after the leader exited: {cleanup_record}"
            )
        if cancel.is_set():
            raise SupervisorError("cancellation requested before publication")
        if guard.violation is not None:
            raise SupervisorError(f"resource guard: {guard.violation}")
        if return_code != 0:
            raise SupervisorError(f"work group leader exited with code {return_code}; see {log_path}")
        if artifact_facts(request_path) != request_facts:
            raise SupervisorError("supervisor request changed during execution")

        leader_result = _load_small(output / "leader_result.json", "leader result")
        if leader_result.get("verification_status") != "pass":
            raise SupervisorError(f"independent artifact verification did not pass: {leader_result.get('verification_status')}")
        verification = _load_small(output / "verification.json", "artifact verification")
        if verification.get("status") != "pass":
            raise SupervisorError(f"artifact verification reported failure: {verification.get('reasons')}")
        facts_index = verification.get("artifacts")
        if not isinstance(facts_index, dict) or not facts_index:
            raise SupervisorError("artifact verification recorded no checked files")

        reader_facts = guard.reader_facts()
        if reader_facts["membership_proven_sample"] is None:
            raise SupervisorError(
                "sampling never observed every expected member of the registered "
                f"work group with a clean enumeration: {reader_facts}"
            )

        verified_snapshot = verify_source_snapshot(
            snapshot_dir, project_root=ROOT, expectation=readiness.get("expectation")
        )
        if verified_snapshot["manifest_sha256"] != snapshot_digest:
            raise SupervisorError("source snapshot changed after the run")
        guard.check_or_raise()

        candidate = _load_small(output / "candidate_manifest.json", "candidate manifest")
        if candidate.get("manifest_hash") != _canonical(
            {key: value for key, value in candidate.items() if key != "manifest_hash"}
        ):
            raise SupervisorError("candidate manifest self-hash is invalid")
        if candidate.get("completion_state") != "candidate_pending_external_checks":
            raise SupervisorError("coordinator candidate is not pending external checks")

        # Files the independent verifier hashed inside the work group.
        declared = {
            "candidate_manifest": "candidate_manifest.json",
            "source_manifest": "source_snapshot/source_manifest.json",
            "supervisor_request": "supervisor_request.json",
            "launch_request": "launch_request.json",
            "coordinator_request": "coordinator_request.json",
            "group_identity": "group_identity.json",
            "source_expectation": "source_expectation.json",
            "environment": "environment.json",
            "train_manifest": "stages/train/train_manifest.json",
            "score_manifest": "stages/score/score_manifest.json",
            "selection_manifest": "stages/score/selection_manifest.json",
            "evaluation_manifest": "stages/evaluate/evaluation_manifest.json",
        }
        verified_artifacts: dict[str, object] = {}
        for name, relative in declared.items():
            fact = facts_index.get(relative)
            if not isinstance(fact, dict):
                raise SupervisorError(f"verifier did not check a declared artifact: {relative}")
            verified_artifacts[name] = fact
        # Small records produced after the verifier ran.  The supervisor binds
        # them directly; they are far below the size of the hashed artifacts.
        for name, relative in (("verification", "verification.json"), ("leader_result", "leader_result.json")):
            target = output / relative
            if not target.is_file():
                raise SupervisorError(f"missing post-verification record: {relative}")
            verified_artifacts[name] = {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }

        body = {
            **{key: value for key, value in candidate.items() if key not in {"manifest_hash", "verified_artifacts", "artifacts"}},
            "status": "complete",
            "scope": "isolated_worker_supervised_v1",
            "completion_state": "complete",
            "role": "external_supervisor",
            "source_snapshot": {
                "path": str(snapshot_dir.relative_to(ROOT)),
                "manifest_sha256": verified_snapshot["manifest_sha256"],
                "files": len(verified_snapshot["verified_files"]),
                "expectation": readiness.get("expectation"),
            },
            "launcher_code": {
                "supervisor": launcher_facts,
                "bootstrap": leader_result.get("launcher"),
            },
            "work_group": {
                "pgid": pgid,
                "identity": leader_result.get("identity"),
                "cleanup": cleanup_record,
            },
            "resource": {
                "sample_count": guard.sample_count,
                "peak_snapshot": guard.peak_snapshot,
                "last_snapshot": guard.last_snapshot,
                "sampling_scope": guard.sampling_scope(),
                "reader_facts": guard.reader_facts(),
                "sampled_until": "bounded_cleanup_confirmed_then_commit_decision_point",
            },
            "supervision": {
                "scope": "external_supervisor_one_process_group",
                "leader_return_code": return_code,
                "process_group_clean": cleanup_record["state"] == "complete",
                "cancelled": False,
                "candidate_name": "candidate_manifest.json",
                "final_name": FINAL_NAME,
                "staged_name": STAGED_NAME,
                "commit_point": "exclusive_final_manifest_publish",
                "coordinator_request_version": COORDINATOR_REQUEST_VERSION,
                "source_manifest_sha256": snapshot_digest,
                "request_facts": request_facts,
                "verification_status": verification.get("status"),
                "verification_checked_files": verification.get("checked_file_count"),
            },
        }

        # Serialise and fsync the manifest before the commit decision so the
        # decision point itself only performs one atomic link.
        staged_path = output / STAGED_NAME
        write_verified_manifest(staged_path, body, verified_artifacts=verified_artifacts, base=output)

        final_snapshot = guard.check_or_raise()
        guard.stop()
        if cancel.is_set():
            raise SupervisorError("cancellation requested before the commit decision point")
        if guard.violation is not None:
            raise SupervisorError(f"resource guard: {guard.violation} before the commit decision point")

        _handshake("before_commit")
        commit_mask = None
        if _supports_sigmask():
            commit_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            pending = _pending_cancel_signals()
            if cancel.is_set() or pending:
                raise SupervisorError(f"cancellation signal arrived at the commit decision point: {pending}")
            final_path = output / FINAL_NAME
            try:
                os.link(staged_path, final_path)
            except FileExistsError as exc:
                raise SupervisorError(f"refusing to overwrite an existing final manifest: {final_path}") from exc
            except OSError as exc:
                raise SupervisorError(f"final manifest publish failed: {final_path}: {exc}") from exc
            if os.environ.get("REPRO_FIXTURE_INJECT") == "signal_at_commit":
                # A signal after the commit decision must not retract a
                # published result, and must not create a competing failure.
                os.kill(os.getpid(), signal.SIGTERM)
        finally:
            if commit_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, commit_mask)

        published = _load_small(output / FINAL_NAME, "final manifest")
        try:
            (output / STAGED_NAME).unlink()
        except OSError:
            pass
        result = dict(published)
        result["verified_artifacts"] = dict(published.get("artifacts", {}))
        result["resource_sample_until"] = "commit_decision_point"
        result["resource_last_snapshot"] = final_snapshot
        return result
    except BaseException as caught:
        if isinstance(caught, ResourceViolation):
            reason = f"resource guard: {caught}"
        elif isinstance(caught, (SourceSnapshotError, SupervisorError)):
            reason = str(caught)
        else:
            reason = f"{type(caught).__name__}: {caught}"
        if pgid is not None and not cleaned:
            cleanup_record = _bounded_cleanup(pgid, process, known_members=guard.observed_group_pids)
            cleaned = True
        _write_failure(
            output,
            caught,
            reason=reason,
            extra={
                "cleanup": cleanup_record,
                "cancelled": cancel.is_set(),
                # Sampling facts survive a failure too, so a report never has
                # to claim "resource check passed" without naming the reader.
                "reader_facts": guard.reader_facts(),
            },
        )
        if isinstance(caught, SupervisorError):
            raise
        raise SupervisorError(reason) from caught
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError, TypeError):
                pass
        if original_mask is not None and _supports_sigmask():
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
                _fixture_mask_record("after", _current_mask())
            except (OSError, ValueError):
                pass
        try:
            guard.stop()
        except BaseException:
            pass


__all__ = [
    "COORDINATOR_REQUEST_VERSION",
    "REQUEST_VERSION",
    "SupervisorError",
    "run_supervised",
]
