"""Parent process for bounded, sequential real Q1/Q2 and Q3 panel builds."""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import replace
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Mapping, Sequence

from ..member_io import manifest_entries
from . import current
from .artifacts import (
    artifact_facts,
    bound_path,
    verify_manifest,
    write_json_exclusive,
    write_manifest,
)
from .environment import write_snapshot
from .cancellation import CancellationToken
from .resource_guard import ResourceGuard, ResourceLimits, ResourceViolation
from .source_streams import ArchiveSource
from .runtime_context import project_root


ROOT = project_root()


class PanelOrchestratorError(RuntimeError):
    """The parent could not safely build or publish the panel pair."""


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PanelOrchestratorError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise PanelOrchestratorError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_child(
    command: Sequence[str],
    *,
    stage: str,
    log_path: Path,
    guard: ResourceGuard,
    timeout_seconds: float,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise PanelOrchestratorError("panel stage timeout must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=False,
        )
        try:
            deadline = started + timeout_seconds
            while process.poll() is None:
                if guard.violation is not None:
                    raise ResourceViolation(guard.violation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(list(command), timeout_seconds)
                try:
                    process.wait(timeout=min(guard.limits.poll_seconds, remaining))
                except subprocess.TimeoutExpired:
                    continue
            return_code = process.returncode
        except BaseException as exc:
            _kill_process_group(process)
            if isinstance(exc, ResourceViolation):
                raise PanelOrchestratorError(
                    f"{stage} stopped by resource guard: {exc}"
                ) from exc
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            raise PanelOrchestratorError(
                f"{stage} timed out after {timeout_seconds:g}s"
            ) from exc
    if return_code != 0:
        raise PanelOrchestratorError(
            f"{stage} exited with code {return_code}; see {log_path}"
        )
    return {
        "status": "complete",
        "stage": stage,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path.relative_to(ROOT)),
    }


def _expected_q1q2_totals(manifest: Mapping[str, object]) -> dict[str, int]:
    entries = manifest_entries(manifest)
    return {
        "source_rows": sum(int(entry["expected_counts"]["source_rows"]) for entry in entries),
        "selected_rows": sum(int(entry["expected_counts"]["selected_rows"]) for entry in entries),
        "failure_rows": sum(int(entry["expected_counts"]["failure_rows"]) for entry in entries),
    }


def _expected_q3_totals(receipt: Mapping[str, object]) -> dict[str, int]:
    raw = receipt.get("totals")
    if isinstance(raw, Mapping):
        return {
            "source_rows": int(raw["source_rows"]),
            "selected_rows": int(raw["selected_rows"]),
            "failure_rows": int(raw["failure_rows"]),
        }
    members = receipt.get("members")
    if not isinstance(members, list):
        raise PanelOrchestratorError("Q3 receipt has no totals or members")
    return {
        "source_rows": sum(int(member["source_rows"]) for member in members),
        "selected_rows": sum(int(member["selected_rows"]) for member in members),
        "failure_rows": sum(int(member["failure_rows"]) for member in members),
    }


def _panel_contract(path: Path, expected: Mapping[str, int], *, label: str) -> dict[str, object]:
    connection = current._open_readonly(path)
    try:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
        required = {
            "build_status": "panel_complete",
            "serial_model_registry_scope": "full_source_rows",
        }
        for key, value in required.items():
            if metadata.get(key) != value:
                raise PanelOrchestratorError(
                    f"{label} metadata {key}={metadata.get(key)!r}, expected {value!r}"
                )
        member_counts = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(source_rows),0), "
            "COALESCE(SUM(selected_rows),0), COALESCE(SUM(failure_rows),0) "
            "FROM member_counts"
        ).fetchone()
        daily_rows = int(connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0])
        registry_rows = int(connection.execute(
            "SELECT COUNT(*) FROM serial_model_registry"
        ).fetchone()[0])
        facts = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": current._sha256_file(path),
            "metadata": {
                key: metadata.get(key)
                for key in (
                    "build_status",
                    "serial_model_registry_scope",
                    "selected_model",
                    "reproducible_source_manifest_sha256",
                    "source_receipt_sha256",
                    "cross_quarter_identity",
                )
                if metadata.get(key) is not None
            },
            "member_count": int(member_counts[0]),
            "source_rows": int(member_counts[1]),
            "selected_rows": int(member_counts[2]),
            "failure_rows": int(member_counts[3]),
            "daily_rows": daily_rows,
            "registry_rows": registry_rows,
            "expected": dict(expected),
        }
        for key in ("source_rows", "selected_rows", "failure_rows"):
            if facts[key] != int(expected[key]):
                raise PanelOrchestratorError(
                    f"{label} {key}={facts[key]} does not match expected {expected[key]}"
                )
        if daily_rows != int(expected["selected_rows"]):
            raise PanelOrchestratorError(
                f"{label} daily rows={daily_rows} does not match selected rows={expected['selected_rows']}"
            )
        return facts
    finally:
        connection.close()


def _stage_manifest(
    path: Path,
    *,
    stage: str,
    panel: Path,
    evidence: Path,
    source: Path,
    environment: Path,
    log: Path,
    panel_facts: Mapping[str, object],
    resources: Mapping[str, object],
    extra_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    artifacts = {
        "panel": panel,
        "stage_evidence": evidence,
        "source": source,
        "environment": environment,
        "log": log,
    }
    if extra_artifacts:
        artifacts.update(extra_artifacts)
    return write_manifest(
        path,
        {
            "status": "complete",
            "scope": "real_panel_rebuild_stage_v1",
            "stage": stage,
            "panel": dict(panel_facts),
            "resource_snapshot": dict(resources),
            "real_training_executed": False,
            "labels_read": False,
            "model_fit": False,
            "scores_generated": False,
            "q4_access": False,
            "remote_setup": False,
            "code_sha256": {
                "pipeline/reproducible/panel_orchestrator.py": current._sha256_file(Path(__file__)),
                "pipeline/reproducible/current.py": current._sha256_file(ROOT / "pipeline/reproducible/current.py"),
                "pipeline/reproducible/source_streams.py": current._sha256_file(ROOT / "pipeline/reproducible/source_streams.py"),
                "pipeline/panel.py": current._sha256_file(ROOT / "pipeline/panel.py"),
            },
        },
        artifacts=artifacts,
    )


def _build_panel_pair(
    *,
    q1q2_manifest: Path | str,
    q3_receipt: Path | str,
    output_root: Path | str,
    evidence_dir: Path | str,
    model: str = current.MODEL,
    limits: ResourceLimits | None = None,
) -> dict[str, object]:
    """Build Q1/Q2, then Q3 with cross-quarter identity verification.

    The parent never reads labels or trains a model.  Each builder is a child
    process, and a stopped child leaves its partial output and log in place.
    """
    q1_path = bound_path(q1q2_manifest, "Q1/Q2 source manifest", must_exist=True)
    q3_path = bound_path(q3_receipt, "Q3 source receipt", must_exist=True)
    output = bound_path(output_root, "real panel output")
    evidence = bound_path(evidence_dir, "real panel evidence")
    if output.exists() or evidence.exists():
        raise PanelOrchestratorError("real panel output and evidence directories must be new")
    if output == evidence or output.is_relative_to(evidence) or evidence.is_relative_to(output):
        raise PanelOrchestratorError("panel output and evidence directories must not overlap")
    output.mkdir(parents=True)
    evidence.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    environment = output / "environment.json"
    write_snapshot(environment)
    active_limits = limits or ResourceLimits(max_elapsed_seconds=4 * 60 * 60)
    if active_limits.max_elapsed_seconds <= 0:
        raise PanelOrchestratorError("panel stage timeout must be positive")

    q1_manifest = _load_object(q1_path, "Q1/Q2 source manifest")
    q3_receipt_obj = _load_object(q3_path, "Q3 source receipt")
    q1_source = current.verify_source_manifest(q1_path, verify_members=False)
    archive_path = current._path(q3_receipt_obj["archive_path"], "Q3 archive", must_exist=True)
    archive_facts = {
        "path": str(archive_path.relative_to(ROOT)),
        "bytes": archive_path.stat().st_size,
        "sha256": current._sha256_file(archive_path),
    }
    if archive_facts["bytes"] != int(q3_receipt_obj["archive_bytes"]):
        raise PanelOrchestratorError("Q3 archive byte count changed before build")
    if archive_facts["sha256"] != q3_receipt_obj["archive_sha256"]:
        raise PanelOrchestratorError("Q3 archive SHA256 changed before build")
    with ArchiveSource(q3_receipt_obj, ROOT):
        q3_inventory = "pass"

    q1q2_panel = output / "q1q2.sqlite"
    q1q2_evidence = evidence / "q1q2_progress.json"
    q1q2_log = logs / "q1q2.log"
    q3_output = output / "q3"
    q3_panel = q3_output / "panel.sqlite"
    q3_log = logs / "q3.log"
    q1q2_totals = _expected_q1q2_totals(q1_manifest)
    q3_totals = _expected_q3_totals(q3_receipt_obj)
    stage_records: dict[str, object] = {}
    q1q2_stage_manifest = evidence / "q1q2_stage_manifest.json"
    q3_stage_manifest = evidence / "q3_stage_manifest.json"
    parent_manifest = evidence / "candidate_manifest.json"

    def run_stage(stage: str, command: Sequence[str], log: Path) -> dict[str, object]:
        guard = ResourceGuard(ROOT, output, active_limits)
        guard.start()
        try:
            record = _run_child(
                command,
                stage=stage,
                log_path=log,
                guard=guard,
                timeout_seconds=active_limits.max_elapsed_seconds,
            )
            guard.check_or_raise()
            record["resource_snapshot"] = guard.stop()
            stage_records[stage] = record
            return record
        except BaseException:
            record = {"status": "failed", "resource_snapshot": guard.stop()}
            stage_records[stage] = record
            raise

    try:
        run_stage(
            "q1q2",
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/run_research.py"),
                "build-panel",
                str(q1_path),
                "--output",
                str(q1q2_panel),
                "--evidence",
                str(q1q2_evidence),
                "--model",
                model,
            ],
            q1q2_log,
        )
        q1q2_progress = _load_object(q1q2_evidence, "Q1/Q2 build evidence")
        if q1q2_progress.get("status") != "complete":
            raise PanelOrchestratorError("Q1/Q2 builder did not publish complete evidence")
        q1q2_panel_facts = _panel_contract(q1q2_panel, q1q2_totals, label="Q1/Q2 panel")
        if q1q2_panel_facts["metadata"].get("reproducible_source_manifest_sha256") != q1_source["manifest_sha256"]:
            raise PanelOrchestratorError("Q1/Q2 panel is bound to a different source manifest")
        q1q2_stage = _stage_manifest(
            q1q2_stage_manifest,
            stage="q1q2",
            panel=q1q2_panel,
            evidence=q1q2_evidence,
            source=q1_path,
            environment=environment,
            log=q1q2_log,
            panel_facts=q1q2_panel_facts,
            resources=stage_records["q1q2"]["resource_snapshot"],
        )

        run_stage(
            "q3",
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/run_research.py"),
                "build-zip-panel",
                str(q3_path),
                "--output",
                str(q3_output),
                "--prior-panel",
                str(q1q2_panel),
            ],
            q3_log,
        )
        q3_manifest_path = q3_output / "panel_manifest.json"
        q3_published = verify_manifest(q3_manifest_path, expected_status="complete")
        q3_panel_facts = _panel_contract(q3_panel, q3_totals, label="Q3 panel")
        if q3_panel_facts["metadata"].get("source_receipt_sha256") != current._sha256_file(q3_path):
            raise PanelOrchestratorError("Q3 panel is bound to a different source receipt")
        identity = current.verify_cross_quarter_identity(q1q2_panel, q3_panel)
        if identity.get("status") != "pass":
            raise PanelOrchestratorError("cross-quarter identity did not pass")
        q3_stage = _stage_manifest(
            q3_stage_manifest,
            stage="q3",
            panel=q3_panel,
            evidence=q3_manifest_path,
            source=q3_path,
            environment=environment,
            log=q3_log,
            panel_facts=q3_panel_facts,
            resources=stage_records["q3"]["resource_snapshot"],
            extra_artifacts={"prior_panel": q1q2_panel},
        )

        final = {
            "status": "complete",
            "role": "parent",
            "scope": "real_panel_rebuild_v1",
            "specification_acceptance": "incomplete",
            "source_bindings": {
                "q1q2_manifest": q1_source,
                "q3_receipt": {
                    "path": str(q3_path.relative_to(ROOT)),
                    "sha256": current._sha256_file(q3_path),
                    "member_count": q3_receipt_obj.get("member_count"),
                    "archive": archive_facts,
                    "member_inventory_status": q3_inventory,
                },
            },
            "stages": {
                "q1q2": q1q2_stage["manifest_hash"],
                "q3": q3_stage["manifest_hash"],
            },
            "panels": {"q1q2": q1q2_panel_facts, "q3": q3_panel_facts},
            "cross_quarter_identity": identity,
            "q3_panel_manifest_hash": q3_published["manifest_hash"],
            "environment": {
                "path": str(environment.relative_to(ROOT)),
                "sha256": artifact_facts(environment)["sha256"],
            },
            "resource_limits": {
                "max_rss_bytes": active_limits.max_rss_bytes,
                "max_output_bytes": active_limits.max_output_bytes,
                "min_free_bytes": active_limits.min_free_bytes,
                "max_elapsed_seconds_per_panel": active_limits.max_elapsed_seconds,
                "poll_seconds": active_limits.poll_seconds,
            },
            "supervision": {
                "scope": "preflight through final publication",
                "owned_directories": [str(output.relative_to(ROOT)), str(evidence.relative_to(ROOT))],
                "total_timeout_seconds": 2 * active_limits.max_elapsed_seconds,
                "completion_publisher": "outer supervisor",
            },
            "stage_records": stage_records,
            "real_training_executed": False,
            "labels_read": False,
            "model_fit": False,
            "scores_generated": False,
            "q4_access": False,
            "remote_setup": False,
            "code_sha256": {
                "pipeline/reproducible/panel_orchestrator.py": current._sha256_file(Path(__file__)),
                "pipeline/reproducible/current.py": current._sha256_file(ROOT / "pipeline/reproducible/current.py"),
                "pipeline/reproducible/source_streams.py": current._sha256_file(ROOT / "pipeline/reproducible/source_streams.py"),
                "pipeline/panel.py": current._sha256_file(ROOT / "pipeline/panel.py"),
            },
        }
        return write_manifest(
            parent_manifest,
            final,
            artifacts={
                "q1q2_panel": q1q2_panel,
                "q1q2_evidence": q1q2_evidence,
                "q1q2_stage_manifest": q1q2_stage_manifest,
                "q3_panel": q3_panel,
                "q3_panel_manifest": q3_manifest_path,
                "q3_stage_manifest": q3_stage_manifest,
                "q1q2_manifest": q1_path,
                "q3_receipt": q3_path,
                "environment": environment,
                "q1q2_log": q1q2_log,
                "q3_log": q3_log,
            },
        )
    except BaseException as exc:
        try:
            write_json_exclusive(
                evidence / "failure.json",
                {
                    "status": "failed",
                    "scope": "real_panel_rebuild_v1",
                    "error": f"{type(exc).__name__}: {exc}",
                    "stage_records": stage_records,
                    "q1q2_panel": str(q1q2_panel.relative_to(ROOT)),
                    "q3_panel": str(q3_panel.relative_to(ROOT)),
                    "real_training_executed": False,
                    "labels_read": False,
                    "model_fit": False,
                    "scores_generated": False,
                    "q4_access": False,
                    "remote_setup": False,
                },
            )
        except BaseException:
            pass
        if isinstance(exc, PanelOrchestratorError):
            raise
        if isinstance(exc, (current.ReproducibleStopped, ResourceViolation)):
            raise PanelOrchestratorError(str(exc)) from exc
        raise


__all__ = ["PanelOrchestratorError", "rebuild_panels"]


def _panel_job(kwargs):
    """Own one process group so the supervisor can stop all builder children."""
    os.setsid()
    evidence = Path(kwargs["evidence_dir"])
    try:
        _build_panel_pair(**kwargs)
        verify_manifest(evidence / "candidate_manifest.json", expected_status="complete")
    except BaseException as exc:
        evidence.mkdir(parents=True, exist_ok=True)
        if not (evidence / "failure.json").exists():
            write_json_exclusive(evidence / "failure.json", {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "scope": "panel worker including preflight and verification",
            })
        raise


def rebuild_panels(*, q1q2_manifest, q3_receipt, output_root, evidence_dir,
                   model=current.MODEL, limits=None):
    """Supervise preflight, both builders and verification before publication."""
    output = bound_path(output_root, "panel output")
    evidence = bound_path(evidence_dir, "panel evidence")
    if output.exists() or evidence.exists():
        raise PanelOrchestratorError("panel output and evidence must be new")
    if output == evidence or output.is_relative_to(evidence) or evidence.is_relative_to(output):
        raise PanelOrchestratorError("panel output and evidence must not overlap")
    active = limits or ResourceLimits(max_elapsed_seconds=4 * 60 * 60)
    if active.max_elapsed_seconds <= 0:
        raise PanelOrchestratorError("panel timeout must be positive")
    guard = ResourceGuard(ROOT, output, replace(active, max_elapsed_seconds=2 * active.max_elapsed_seconds),
                          extra_outputs=(evidence,))
    process = multiprocessing.get_context("spawn").Process(target=_panel_job, args=({
        "q1q2_manifest": str(q1q2_manifest), "q3_receipt": str(q3_receipt),
        "output_root": str(output), "evidence_dir": str(evidence),
        "model": model, "limits": active,
    },))
    started = False
    final = evidence / "real_panel_rebuild_manifest.json"
    token = CancellationToken()
    token.install()
    guard.start()
    try:
        guard.check_or_raise()
        process.start()
        started = True
        while process.is_alive():
            token.check()
            guard.check_or_raise()
            process.join(timeout=active.poll_seconds)
        if process.exitcode != 0:
            raise PanelOrchestratorError(f"panel worker exited with code {process.exitcode}; see failure.json")
        guard.check_or_raise()
        candidate = evidence / "candidate_manifest.json"
        result = _load_object(candidate, "verified candidate manifest")
        token.check()
        guard.check_or_raise()
        # Only the supervisor can publish the authoritative completion name.
        os.link(candidate, final)
        try:
            token.check()
            guard.check_or_raise()
        except BaseException:
            final.unlink()
            raise
        return result
    except BaseException as exc:
        if started:
            # Builders inherit the worker's group; kill remaining descendants
            # even if the worker itself has already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        evidence.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(evidence / "supervisor_failure.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "resource_snapshot": guard.last_snapshot,
            "scope": "whole panel rebuild", "real_training_executed": False,
        })
        raise PanelOrchestratorError(str(exc)) from exc
    finally:
        guard.stop()
        token.restore()
        if started and not process.is_alive():
            process.close()
