"""Lightweight launch module and work-group leader for supervised runs.

Only the standard library is imported here.  The bound ``code_root`` from the
launch request is inserted into ``sys.path`` explicitly, so ``PYTHONPATH``,
the current directory, or a ``REPRO_*`` variable cannot make some other tree
become the project root.

This file is launcher code.  It runs from the original checkout, prepares the
source snapshot inside the supervised process group, and only then starts the
isolated coordinator from that snapshot with a fresh interpreter.  A run must
not claim that this leader itself executed from the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time


LAUNCH_REQUEST_VERSION = "supervised-launch-request-v1"
COORDINATOR_REQUEST_VERSION = "supervised-coordinator-request-v2"
READINESS_SCHEMA = "snapshot-readiness-v1"
LEADER_RESULT_SCHEMA = "supervised-leader-result-v1"
POLL_SECONDS = 0.05


class LeaderError(RuntimeError):
    """The work-group leader could not prepare, start, or verify a run."""


def _reject_constant(value: str) -> object:
    raise LeaderError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LeaderError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_no_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise LeaderError(f"launch request is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if type(value) is not str or not value:
        raise LeaderError(f"launch request field is invalid: {key}")
    return value


def _group_members(pgid: int) -> list[int] | None:
    """Return visible members of a process group, or None if inspection fails.

    Reading the table from inside the supervised group must not fork a helper
    process: a spawned reader would inherit this group and appear as a fresh
    leftover member on every check.  libproc is therefore tried first here.
    """
    from pipeline.reproducible.resource_guard import _libproc_process_table, _ps_process_table

    read = _libproc_process_table()
    if read is not None:
        table, _problems = read
    else:
        table = _ps_process_table()
    if not table:
        return None
    return sorted(pid for pid, (_parent, group, _rss) in table.items() if group == pgid)


def _reap_residual(pgid: int, pids: list[int]) -> dict[str, object]:
    """Bounded TERM → KILL for confirmed members of our own group.

    Only pids that were observed in this group are signalled, and the leader
    never signals itself, so an uncontrolled pid is never touched.
    """
    record: dict[str, object] = {"attempted": sorted(pids), "survived": [], "errors": []}
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            record["errors"].append(f"terminate {pid}: {exc}")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        members = _group_members(pgid)
        if members is None:
            record["errors"].append("member inspection failed during cleanup")
            break
        remaining = [pid for pid in members if pid != os.getpid()]
        if not remaining:
            return record
        time.sleep(0.05)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            record["errors"].append(f"force-kill {pid}: {exc}")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        members = _group_members(pgid)
        if members is None:
            record["errors"].append("member inspection failed after force-kill")
            break
        remaining = [pid for pid in members if pid != os.getpid()]
        if not remaining:
            break
        time.sleep(0.05)
    members = _group_members(pgid)
    record["survived"] = [] if members is None else [pid for pid in members if pid != os.getpid()]
    return record


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        written = handle.write(text)
        if written != len(text):
            raise LeaderError(f"short write for {path.name}")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _child_environment(project_root: str, code_root: str, snapshot_dir: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env["PATH"] = os.environ.get("PATH", "")
    env["REPRO_PROJECT_ROOT"] = project_root
    env["REPRO_CODE_ROOT"] = code_root
    env["REPRO_WORK_GROUP"] = "1"
    env["REPRO_SOURCE_MANIFEST"] = str(snapshot_dir / "source_manifest.json")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    return env


def _handshake(stage: str) -> None:
    """Fixture-only synchronisation point; unused unless the fixture enables it.

    A real run never sets ``REPRO_FIXTURE_HANDSHAKE``, so this returns
    immediately.  When a test fixture does set it, the leader announces the
    stage and waits to be released, which lets a cancellation test hit an exact
    point instead of racing a sleep.
    """
    directory = os.environ.get("REPRO_FIXTURE_HANDSHAKE")
    if not directory:
        return
    base = Path(directory)
    (base / f"reached.{stage}").write_text(str(os.getpid()), encoding="utf-8")
    release = base / f"release.{stage}"
    deadline = time.monotonic() + 60.0
    while not release.is_file() and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)


def _leader(request: dict[str, object]) -> int:
    from pipeline.reproducible.source_snapshot import SourceSnapshotError, create_source_snapshot, verify_source_snapshot

    if request.get("version") != LAUNCH_REQUEST_VERSION:
        raise LeaderError("launch request version is invalid")
    project_root = Path(_required_string(request, "project_root"))
    code_root = Path(_required_string(request, "code_root"))
    output_dir = Path(_required_string(request, "output_dir"))
    snapshot_dir = Path(_required_string(request, "snapshot_dir"))
    python_executable = _required_string(request, "python_executable")
    coordinator_timeout = float(request.get("coordinator_timeout_seconds") or 900)
    handshake_timeout = float(request.get("handshake_timeout_seconds") or 300)
    if not project_root.is_dir() or not code_root.is_dir():
        raise LeaderError("launch request roots are not directories")
    if not output_dir.is_dir():
        raise LeaderError("launch request output directory is missing")
    # The bound request carries the profile the supervisor already validated;
    # the leader re-checks it so a fixture control can never reach this entry
    # without a synthetic fixture profile.
    from pipeline.reproducible.resource_guard import assert_fixture_controls_allowed

    assert_fixture_controls_allowed(request.get("config_profile"))

    launcher_source = Path(__file__).resolve()
    launcher_copy_dir = output_dir / "launcher"
    launcher_copy_dir.mkdir(parents=True, exist_ok=True)
    launcher_copy = launcher_copy_dir / "supervisor_bootstrap.py"
    shutil.copyfile(launcher_source, launcher_copy)
    launcher_facts = {
        "role": "launcher_code",
        "note": "runs from the original checkout before the snapshot exists",
        "path": str(launcher_copy.relative_to(output_dir)),
        "bytes": launcher_copy.stat().st_size,
        "sha256": _sha256(launcher_copy),
    }

    identity = {
        "schema": "work-group-identity-v1",
        "role": "work_group_leader_launcher",
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "stage": "source_preparation",
        "launcher": launcher_facts,
    }
    _write_json(output_dir / "group_identity.json", identity)

    _handshake("source_preparation")
    try:
        snapshot = create_source_snapshot(snapshot_dir, project_root=project_root)
    except SourceSnapshotError:
        raise
    expectation = dict(snapshot["expectation"])
    _write_json(output_dir / "source_expectation.json", expectation)
    verified = verify_source_snapshot(snapshot_dir, project_root=project_root, expectation=expectation)
    readiness = {
        "schema": READINESS_SCHEMA,
        "snapshot_root": str(snapshot_dir),
        "source_manifest_sha256": verified["manifest_sha256"],
        "file_count": len(verified["verified_files"]),
        "expectation": expectation,
        "identity": identity,
    }
    _write_json(output_dir / "snapshot_readiness.json", readiness)

    coordinator_request_path = output_dir / "coordinator_request.json"
    deadline = time.monotonic() + handshake_timeout
    while not coordinator_request_path.is_file():
        if time.monotonic() > deadline:
            raise LeaderError("supervisor did not publish a coordinator request before the handshake deadline")
        time.sleep(POLL_SECONDS)

    coordinator = _load(coordinator_request_path)
    if coordinator.get("version") != COORDINATOR_REQUEST_VERSION:
        raise LeaderError("coordinator request version is invalid")
    if Path(str(coordinator.get("code_root"))) != snapshot_dir:
        raise LeaderError("coordinator request is not bound to the prepared snapshot")
    if coordinator.get("source_manifest_sha256") != readiness["source_manifest_sha256"]:
        raise LeaderError("coordinator request is not bound to the prepared snapshot digest")
    if Path(str(coordinator.get("output_dir"))) != output_dir or Path(str(coordinator.get("project_root"))) != project_root:
        raise LeaderError("coordinator request roots differ from the launch request")

    _handshake("before_coordinator")
    if os.environ.get("REPRO_FIXTURE_INJECT") == "leader_exit_early":
        # Fixture: a descendant installs its SIGTERM-ignoring handler first,
        # announces its real pid through the handshake directory, and only then
        # does the leader exit.  This removes the start-up race the review
        # flagged: the supervisor always observes a descendant that is already
        # immune to SIGTERM.
        handshake = os.environ.get("REPRO_FIXTURE_HANDSHAKE", "")
        if not handshake:
            raise LeaderError("leader_exit_early requires REPRO_FIXTURE_HANDSHAKE")
        descendant = subprocess.Popen(
            [
                str(Path(request["python_executable"])),
                "-c",
                "import os, signal, time;"
                " signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                " open(os.environ['REPRO_DESCENDANT_READY'], 'w').write(str(os.getpid()));"
                " time.sleep(300)",
            ],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "REPRO_DESCENDANT_READY": os.path.join(handshake, "ready.descendant")},
        )
        ready_path = os.path.join(handshake, "ready.descendant")
        deadline = time.monotonic() + 30.0
        while not os.path.isfile(ready_path) and time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
        if not os.path.isfile(ready_path):
            descendant.kill()
            raise LeaderError("fixture descendant never announced readiness")
        try:
            descendant_pid = int(Path(ready_path).read_text(encoding="utf-8").strip())
        except ValueError:
            descendant_pid = descendant.pid
        _write_json(
            output_dir / "leader_result.json",
            {
                "schema": LEADER_RESULT_SCHEMA,
                "coordinator_return_code": None,
                "group_pgid": os.getpgrp(),
                "residual_members": [descendant_pid],
                "cleanup_state": "leader_exited_early_fixture",
                "verification_status": "not_run",
                "verification_path": None,
                "fixture_descendant_pid": descendant_pid,
                "launcher": launcher_facts,
                "identity": identity,
            },
        )
        return 0
    env = _child_environment(str(project_root), str(snapshot_dir), snapshot_dir)
    log_path = output_dir / "coordinator.log"
    command = [
        python_executable,
        "-I",
        "-B",
        str(snapshot_dir / "tools/run_research.py"),
        "isolated-coordinator",
        "--request",
        str(coordinator_request_path),
    ]
    verification_status = "not_run"
    verification_path = output_dir / "verification.json"
    return_code = -1
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            return_code = process.wait(timeout=coordinator_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = -9
    identity["stage"] = "coordinator_finished"
    members = _group_members(os.getpgrp())
    residual = sorted(pid for pid in (members or []) if pid != os.getpid())
    # A leftover member is cleaned up here, within the group, before the leader
    # reports.  Not cleaning first would report a normal finish as a failure.
    reaped: dict[str, object] = {}
    if residual:
        reaped = _reap_residual(os.getpgrp(), residual)
        members = _group_members(os.getpgrp())
        residual = sorted(pid for pid in (members or []) if pid != os.getpid())
    cleanup_state = "complete" if members is not None and not residual else ("unknown" if members is None else "incomplete")

    if return_code == 0 and members is not None and not residual:
        _handshake("before_verification")
        verification_request = {
            "version": "verification-request-v1",
            "config_profile": request.get("config_profile"),
            "project_root": str(project_root),
            "code_root": str(snapshot_dir),
            "output_dir": str(output_dir),
            "snapshot_dir": str(snapshot_dir),
            "source_manifest_sha256": readiness["source_manifest_sha256"],
            "expectation": expectation,
            "candidate": "candidate_manifest.json",
            "output": "verification.json",
        }
        verification_request_path = output_dir / "verification_request.json"
        _write_json(verification_request_path, verification_request)
        completed = subprocess.run(
            [
                python_executable,
                "-I",
                "-B",
                str(snapshot_dir / "tools/run_research.py"),
                "verify-run",
                "--request",
                str(verification_request_path),
            ],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(60.0, coordinator_timeout),
        )
        with (output_dir / "verification.log").open("w", encoding="utf-8") as log:
            log.write(completed.stdout or "")
            log.write(completed.stderr or "")
        if completed.returncode == 0 and verification_path.is_file():
            payload = _load(verification_path)
            verification_status = str(payload.get("status"))
        else:
            verification_status = f"verifier_exit_{completed.returncode}"

    result = {
        "schema": LEADER_RESULT_SCHEMA,
        "coordinator_return_code": return_code,
        "group_pgid": os.getpgrp(),
        "residual_members": residual,
        "cleanup_state": cleanup_state,
        "residual_cleanup": reaped,
        "verification_status": verification_status,
        "verification_path": str(verification_path.relative_to(output_dir)) if verification_path.is_file() else None,
        "launcher": launcher_facts,
        "identity": identity,
    }
    _write_json(output_dir / "leader_result.json", result)
    if return_code != 0:
        return 2
    if cleanup_state != "complete":
        return 2
    if verification_status != "pass":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = _load(args.launch_request)
        sys.path.insert(0, str(Path(str(request.get("code_root")))))
        return _leader(request)
    except (LeaderError, OSError, ValueError, json.JSONDecodeError) as exc:
        message = f"STOPPED: {type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        try:
            output = Path(str(request.get("output_dir"))) if isinstance(request, dict) else None
            if output is not None and output.is_dir():
                _write_json(
                    output / "leader_failure.json",
                    {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"},
                )
        except (OSError, ValueError, KeyError, UnboundLocalError):
            pass
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    signal.signal(signal.SIGINT, lambda _signum, _frame: sys.exit(130))
    raise SystemExit(main())
