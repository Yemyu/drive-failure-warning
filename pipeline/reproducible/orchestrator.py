"""Parent process for the isolated synthetic stage workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Mapping

from .artifacts import artifact_facts, bound_path, verify_manifest, write_json_exclusive, write_manifest
from .contract import ContractError, config_facts, load_config_for_bindings, require_panel_contract, validate_dates
from .input_binding import BindingError, create_binding_file
from .process_contract import authorize_inputs
from .resource_guard import ResourceGuard, ResourceLimits, ResourceViolation, assert_fixture_controls_allowed
from .environment import write_snapshot
from .runtime_context import code_root, project_root


class OrchestratorError(RuntimeError):
    """The parent could not start, verify, or publish a stage."""


ROOT = project_root()
CODE_ROOT = code_root()


def _load_config(path: Path | str, bindings: Mapping[str, Path]) -> tuple[Path, Mapping[str, object], bool]:
    try:
        return load_config_for_bindings(path, bindings)
    except ContractError as exc:
        raise OrchestratorError(str(exc)) from exc


def _validate_parent_contract(
    *,
    train_path: Path,
    eval_path: Path,
    config_path: Path | str,
    dates: Mapping[str, str],
    limits: ResourceLimits,
    fit_timeout_seconds: float,
) -> tuple[Mapping[str, object], bool, dict[str, object], dict[str, object]]:
    try:
        config_path, config, strict = _load_config(
            config_path,
            {"train_panel": train_path, "eval_panel": eval_path},
        )
        validate_dates(dates, role="parent", config=config, strict=strict)
        train_contract = require_panel_contract(train_path, role="training")
        eval_contract = require_panel_contract(eval_path, role="evaluation")
        if strict:
            expected_limits = config["limits"]
            checks = {
                "max_rss_bytes": limits.max_rss_bytes,
                "max_output_bytes": limits.max_output_bytes,
                "initial_free_bytes": limits.initial_free_bytes,
                "min_free_bytes": limits.min_free_bytes,
                "max_elapsed_seconds": limits.max_elapsed_seconds,
                "poll_seconds": limits.poll_seconds,
            }
            for key, actual in checks.items():
                if float(actual) != float(expected_limits[key]):
                    raise ContractError(f"parent resource limit differs from locked configuration: {key}")
            available = shutil.disk_usage(ROOT).free
            if available < int(expected_limits["initial_free_bytes"]):
                raise ContractError(
                    "initial free space is below the locked minimum: "
                    f"{available} < {int(expected_limits['initial_free_bytes'])}"
                )
            if float(fit_timeout_seconds) != float(expected_limits["fit_timeout_seconds"]):
                raise ContractError("parent fit timeout differs from locked configuration")
        return config, strict, train_contract, eval_contract
    except ContractError as exc:
        raise OrchestratorError(str(exc)) from exc


def _run_child(
    role: str,
    spec_path: Path,
    stage_output: Path,
    log_path: Path,
    timeout_seconds: int,
    guard: ResourceGuard,
) -> None:
    command = [
        sys.executable,
        str(CODE_ROOT / "tools/run_research.py"),
        "worker",
        "--role",
        role,
        "--spec",
        str(spec_path),
        "--output",
        str(stage_output),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            # A supervised coordinator creates one process group for all
            # workers.  Direct use retains the older per-stage group boundary.
            start_new_session=not bool(os.environ.get("REPRO_WORK_GROUP")),
        )
        try:
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                if guard.violation is not None:
                    raise ResourceViolation(guard.violation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    process.wait(timeout=min(guard.limits.poll_seconds, remaining))
                except subprocess.TimeoutExpired:
                    continue
            return_code = process.returncode
        except (subprocess.TimeoutExpired, ResourceViolation) as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait()
            if isinstance(exc, ResourceViolation):
                raise OrchestratorError(f"{role} worker stopped by resource guard: {exc}") from exc
            raise OrchestratorError(f"{role} worker timed out after {timeout_seconds}s") from exc
    if return_code != 0:
        raise OrchestratorError(f"{role} worker exited with code {return_code}; see {log_path}")


def _write_stage_spec(path: Path, payload: Mapping[str, object]) -> None:
    write_json_exclusive(path, dict(payload))


def _assert_access_stable(role: str, bindings: Mapping[str, Path], expected: Mapping[str, object]) -> None:
    actual = authorize_inputs(role, bindings)
    if actual != expected:
        raise OrchestratorError(f"{role} input changed while worker was running")


def _isolated_run_inner(
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
    _precreated: bool = False,
    _supervised: bool = False,
) -> dict[str, object]:
    """Run train → score → evaluate workers and publish one parent manifest."""
    if timeout_seconds <= 0:
        raise OrchestratorError("worker timeout must be positive")
    if fit_timeout_seconds <= 0:
        raise OrchestratorError("fit timeout must be positive")
    train_path = bound_path(train_panel, "training panel", must_exist=True)
    eval_path = bound_path(eval_panel, "evaluation panel", must_exist=True)
    config_path = bound_path(config, "orchestrator config", must_exist=True)
    parent_dates = {
        "train_start": train_start,
        "train_end": train_end,
        "train_cutoff": train_cutoff,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "eval_cutoff": eval_cutoff,
        "event_start": event_start,
        "event_end": event_end,
    }
    active_limits = limits or ResourceLimits()
    config_payload, strict, train_contract, eval_contract = _validate_parent_contract(
        train_path=train_path,
        eval_path=eval_path,
        config_path=config_path,
        dates=parent_dates,
        limits=active_limits,
        fit_timeout_seconds=fit_timeout_seconds,
    )
    # A fit must never start while test controls are set for a real profile.
    assert_fixture_controls_allowed(config_payload.get("profile"))
    output = bound_path(output_dir, "isolated run output")
    if output.exists() and not _precreated:
        raise OrchestratorError(f"isolated run output already exists: {output}")
    if not output.exists():
        output.mkdir(parents=True)
    guard = ResourceGuard(ROOT, output, active_limits)
    guard.start()
    stages = output / "stages"
    specs = output / "specs"
    stages.mkdir()
    specs.mkdir()
    train_output = stages / "train"
    score_output = stages / "score"
    evaluate_output = stages / "evaluate"
    logs = output / "logs"
    logs.mkdir()
    environment_path = output / "environment.json"
    try:
        write_snapshot(environment_path)
        guard.check_or_raise()
        train_bindings = {"train_panel": train_path, "config": config_path}
        train_access = authorize_inputs("train", train_bindings)
        train_binding_path = specs / "train.binding.json"
        try:
            create_binding_file(train_binding_path, "train", config_path, train_bindings)
        except BindingError as exc:
            raise OrchestratorError(str(exc)) from exc
        train_spec = {
            "run_id": "isolated_train_h7_v1",
            "bindings": {key: str(value) for key, value in train_bindings.items()},
            "binding_file": str(train_binding_path),
            "train_start": train_start, "train_end": train_end, "train_cutoff": train_cutoff,
            "fit_timeout_seconds": fit_timeout_seconds,
            "config_profile": config_payload.get("profile"),
        }
        train_spec_path = specs / "train.json"
        _write_stage_spec(train_spec_path, train_spec)
        _run_child("train", train_spec_path, train_output, logs / "train.log", timeout_seconds, guard)
        guard.check_or_raise()
        _assert_access_stable("train", train_bindings, train_access)
        train_manifest = verify_manifest(train_output / "train_manifest.json", expected_status="complete")
        if train_manifest.get("input_contract") != train_access:
            raise OrchestratorError("train worker input contract differs from parent preflight")

        score_bindings = {
            "eval_panel": eval_path,
            "history_panel": train_path,
            "model": train_output / "model.json",
            "config": config_path,
            "train_manifest": train_output / "train_manifest.json",
        }
        score_access = authorize_inputs("score", score_bindings)
        score_binding_path = specs / "score.binding.json"
        try:
            create_binding_file(score_binding_path, "score", config_path, score_bindings)
        except BindingError as exc:
            raise OrchestratorError(str(exc)) from exc
        score_spec = {
            "bindings": {key: str(value) for key, value in score_bindings.items()},
            "binding_file": str(score_binding_path),
            "eval_start": eval_start, "eval_end": eval_end, "eval_cutoff": eval_cutoff,
            "config_profile": config_payload.get("profile"),
        }
        score_spec_path = specs / "score.json"
        _write_stage_spec(score_spec_path, score_spec)
        _run_child("score", score_spec_path, score_output, logs / "score.log", timeout_seconds, guard)
        guard.check_or_raise()
        _assert_access_stable("score", score_bindings, score_access)
        score_manifest = verify_manifest(score_output / "score_manifest.json", expected_status="complete")
        if score_manifest.get("input_contract") != score_access:
            raise OrchestratorError("score worker input contract differs from parent preflight")
        selection_manifest = verify_manifest(score_output / "selection_manifest.json", expected_status="closed")
        if score_manifest.get("selection_manifest_hash") != selection_manifest.get("manifest_hash"):
            raise OrchestratorError("score manifest does not bind its closed selection manifest")

        evaluate_bindings = {
            "eval_panel": eval_path,
            "history_panel": train_path,
            "selection_manifest": score_output / "selection_manifest.json",
            "score_manifest": score_output / "score_manifest.json",
            "config": config_path,
        }
        evaluate_access = authorize_inputs("evaluate", evaluate_bindings)
        evaluate_binding_path = specs / "evaluate.binding.json"
        try:
            create_binding_file(evaluate_binding_path, "evaluate", config_path, evaluate_bindings)
        except BindingError as exc:
            raise OrchestratorError(str(exc)) from exc
        evaluate_spec = {
            "run_id": "isolated_eval_h7_v1",
            "bindings": {key: str(value) for key, value in evaluate_bindings.items()},
            "binding_file": str(evaluate_binding_path),
            "eval_start": eval_start, "eval_end": eval_end, "eval_cutoff": eval_cutoff,
            "event_start": event_start, "event_end": event_end,
            "config_profile": config_payload.get("profile"),
        }
        evaluate_spec_path = specs / "evaluate.json"
        _write_stage_spec(evaluate_spec_path, evaluate_spec)
        _run_child("evaluate", evaluate_spec_path, evaluate_output, logs / "evaluate.log", timeout_seconds, guard)
        guard.check_or_raise()
        _assert_access_stable("evaluate", evaluate_bindings, evaluate_access)
        evaluate_manifest = verify_manifest(evaluate_output / "evaluation_manifest.json", expected_status="complete")
        if evaluate_manifest.get("input_contract") != evaluate_access:
            raise OrchestratorError("evaluation worker input contract differs from parent preflight")

        guard.check_or_raise()
        candidate = write_manifest(
            output / "candidate_manifest.json",
            {
                "status": "complete",
                "role": "parent",
                "scope": "isolated_worker_prototype",
                "specification_acceptance": "incomplete",
                "completion_state": "candidate_pending_external_checks",
                "config": {**config_facts(config_path), "strict_locked": strict},
                "train_panel_contract": train_contract,
                "eval_panel_contract": eval_contract,
                "input_contracts": {"train": train_access, "score": score_access, "evaluate": evaluate_access},
                "stages": {
                    "train": train_manifest["manifest_hash"],
                    "score": score_manifest["manifest_hash"],
                    "evaluate": evaluate_manifest["manifest_hash"],
                },
                "selection_manifest_hash": selection_manifest["manifest_hash"],
                "environment": {
                    "path": str(environment_path.relative_to(ROOT)),
                    "sha256": artifact_facts(environment_path)["sha256"],
                },
                "timeout_seconds": timeout_seconds,
                "fit_timeout_seconds": fit_timeout_seconds,
                "resource_limits": {
                    "max_rss_bytes": active_limits.max_rss_bytes,
                    "max_output_bytes": active_limits.max_output_bytes,
                    "initial_free_bytes": active_limits.initial_free_bytes,
                    "min_free_bytes": active_limits.min_free_bytes,
                    "max_elapsed_seconds": active_limits.max_elapsed_seconds,
                    "poll_seconds": active_limits.poll_seconds,
                },
                "resource_snapshot_before_publish": guard.last_snapshot,
                "supervision": {
                    "scope": "parent_guard_through_candidate_verification",
                    "candidate_name": "candidate_manifest.json",
                    "final_name": "isolated_run_manifest.json",
                    "final_publish_after": ["child_exit_codes", "input_fingerprints", "resource_guard", "candidate_manifest"],
                },
            },
            artifacts={
                "train_panel": train_path,
                "eval_panel": eval_path,
                "config": config_path,
                "environment": environment_path,
                "train_spec": train_spec_path,
                "score_spec": score_spec_path,
                "evaluate_spec": evaluate_spec_path,
                "train_binding": train_binding_path,
                "score_binding": score_binding_path,
                "evaluate_binding": evaluate_binding_path,
                "train_manifest": train_output / "train_manifest.json",
                "score_manifest": score_output / "score_manifest.json",
                "selection_manifest": score_output / "selection_manifest.json",
                "evaluation_manifest": evaluate_output / "evaluation_manifest.json",
                "train_log": logs / "train.log",
                "score_log": logs / "score.log",
                "evaluate_log": logs / "evaluate.log",
            },
        )
        guard.check_or_raise()
        verify_manifest(output / "candidate_manifest.json", expected_status="complete")
        guard.check_or_raise()
        final_path = output / "isolated_run_manifest.json"
        if _supervised:
            return verify_manifest(output / "candidate_manifest.json", expected_status="complete")
        if final_path.exists():
            raise OrchestratorError("isolated run final manifest already exists")
        candidate_path = output / "candidate_manifest.json"
        try:
            os.link(candidate_path, final_path)
        except FileExistsError as exc:
            raise OrchestratorError("isolated run final manifest already exists") from exc
        except OSError as exc:
            raise OrchestratorError(f"cannot publish isolated run final manifest exclusively: {exc}") from exc
        candidate_path.unlink()
        return verify_manifest(final_path, expected_status="complete")
    except BaseException as caught:
        exc = caught
        if isinstance(caught, ResourceViolation):
            exc = OrchestratorError(f"parent stopped by resource guard: {caught}")
        write_json_exclusive(
            output / "failure.json",
            {"status": "failed", "scope": "isolated_worker_prototype", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise exc
    finally:
        guard.stop()


def isolated_run(
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
    """Run the workflow under the external supervisor."""
    from .supervisor import run_supervised

    return run_supervised(
        train_panel=train_panel,
        eval_panel=eval_panel,
        config=config,
        output_dir=output_dir,
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
        limits=limits,
    )


__all__ = ["OrchestratorError", "_isolated_run_inner", "isolated_run"]
