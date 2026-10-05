#!/usr/bin/env python3
"""Reproducible panel, source, and isolated-worker entry points.

The legacy ``run`` command remains a panel-input prototype; full resource and
real-data acceptance is still pending. See
reports/REPRODUCIBLE_PIPELINE_REVIEW.md before using real data.

The subcommand is parsed before any stage implementation is imported, so a
supervised run never loads the numerical training modules in the process that
decides whether to start it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("REPRO_PROJECT_ROOT", str(CODE_ROOT))).expanduser().resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline.reproducible.artifacts import ArtifactError  # noqa: E402
from pipeline.reproducible.process_contract import ProcessAccessError, authorize_bindings_file, ROLE_INPUTS  # noqa: E402
from pipeline.reproducible.resource_guard import ResourceLimits  # noqa: E402
from pipeline.reproducible.runtime_context import validate_context  # noqa: E402
from pipeline.reproducible.contract import ContractError, load_json  # noqa: E402
from pipeline.reproducible.supervisor import SupervisorError  # noqa: E402


def _check_code_root_binding() -> None:
    """Reject a project root that came from PYTHONPATH, cwd, or an env alias.

    The bound code root is the directory of this entry point.  A second
    ``pipeline`` package visible from another ``sys.path`` entry would let an
    uncontrolled tree supply the modules that run, so it is an error rather
    than something to skip.
    """
    declared = os.environ.get("REPRO_CODE_ROOT")
    if declared:
        try:
            resolved = Path(declared).expanduser().resolve()
        except OSError as exc:
            raise SupervisorError(f"REPRO_CODE_ROOT is not usable: {exc}") from exc
        if resolved != CODE_ROOT:
            raise SupervisorError(
                f"REPRO_CODE_ROOT does not match the launched entry point: {resolved} != {CODE_ROOT}"
            )
    # A decoy package that PYTHONPATH or the current directory puts on the path
    # is only ignored if it never becomes the pipeline that actually runs.
    module = sys.modules.get("pipeline")
    origin = getattr(module, "__file__", None)
    if not origin:
        return
    resolved = Path(origin).resolve()
    if not resolved.is_relative_to(CODE_ROOT):
        raise SupervisorError(f"the imported pipeline package is outside the bound code root: {resolved}")


_CONTROLLED_STOP_NAMES = frozenset({
    "BindingError",
    "CancellationRequested",
    "ComparisonError",
    "EvaluationSummaryError",
    "ModuleIdentityError",
    "OrchestratorError",
    "PanelOrchestratorError",
    "PanelReferenceError",
    "ReproducibleStopped",
    "ResourceViolation",
    "SourceError",
    "SourceSnapshotError",
    "TrainingCheckError",
    "VerificationError",
    "WorkerError",
})

COORDINATOR_BASE_KEYS = {
    "version", "project_root", "code_root", "train_panel", "eval_panel", "config",
    "output_dir", "train_start", "train_end", "train_cutoff", "eval_start", "eval_end",
    "eval_cutoff", "event_start", "event_end", "timeout_seconds", "fit_timeout_seconds", "limits",
}
COORDINATOR_V2_KEYS = COORDINATOR_BASE_KEYS | {"source_manifest_sha256"}


def _coordinator(request: dict[str, object]) -> dict[str, object]:
    from pipeline.reproducible.artifacts import sha256_file
    from pipeline.reproducible.orchestrator import _isolated_run_inner

    version = request.get("version")
    expected = COORDINATOR_V2_KEYS if version == "supervised-coordinator-request-v2" else COORDINATOR_BASE_KEYS
    if set(request) != expected:
        raise SupervisorError("supervisor request keys are invalid")
    if version not in ("supervised-request-v1", "supervised-coordinator-request-v2"):
        raise SupervisorError("supervisor request version is invalid")
    if type(request.get("project_root")) is not str or type(request.get("code_root")) is not str:
        raise SupervisorError("supervisor request roots are invalid")
    try:
        validate_context(
            expected_code_root=request["code_root"],
            expected_project_root=request["project_root"],
        )
    except RuntimeError as exc:
        raise SupervisorError(str(exc)) from exc
    string_fields = {
        "train_panel", "eval_panel", "config", "output_dir", "train_start", "train_end",
        "train_cutoff", "eval_start", "eval_end", "eval_cutoff", "event_start", "event_end",
    }
    if any(type(request.get(field)) is not str or not request[field] for field in string_fields):
        raise SupervisorError("supervisor request path or date fields are invalid")
    if type(request.get("timeout_seconds")) is not int or request["timeout_seconds"] <= 0:
        raise SupervisorError("supervisor request timeout is invalid")
    if type(request.get("fit_timeout_seconds")) not in (int, float) or request["fit_timeout_seconds"] <= 0:
        raise SupervisorError("supervisor request fit timeout is invalid")
    if version == "supervised-coordinator-request-v2":
        declared = request.get("source_manifest_sha256")
        if type(declared) is not str or len(declared) != 64:
            raise SupervisorError("supervisor request snapshot digest is invalid")
        bound = os.environ.get("REPRO_SOURCE_MANIFEST")
        if not bound:
            raise SupervisorError("supervised coordinator was started without a bound source manifest")
        if sha256_file(Path(bound)) != declared:
            raise SupervisorError("bound source manifest does not match the supervisor request digest")
    raw_limits = request.get("limits")
    if not isinstance(raw_limits, dict):
        raise SupervisorError("supervisor request limits are invalid")
    limit_keys = {
        "max_rss_bytes", "max_output_bytes", "initial_free_bytes", "min_free_bytes",
        "max_elapsed_seconds", "poll_seconds",
    }
    if set(raw_limits) != limit_keys:
        raise SupervisorError("supervisor request limit keys are invalid")
    if any(type(raw_limits[key]) is not int or raw_limits[key] < 0 for key in limit_keys - {"max_elapsed_seconds", "poll_seconds"}):
        raise SupervisorError("supervisor request byte limits are invalid")
    if type(raw_limits["max_elapsed_seconds"]) not in (int, float) or raw_limits["max_elapsed_seconds"] <= 0:
        raise SupervisorError("supervisor request elapsed limit is invalid")
    if type(raw_limits["poll_seconds"]) not in (int, float) or raw_limits["poll_seconds"] <= 0:
        raise SupervisorError("supervisor request poll limit is invalid")
    return _isolated_run_inner(
        train_panel=request["train_panel"],
        eval_panel=request["eval_panel"],
        config=request["config"],
        output_dir=request["output_dir"],
        train_start=request["train_start"],
        train_end=request["train_end"],
        train_cutoff=request["train_cutoff"],
        eval_start=request["eval_start"],
        eval_end=request["eval_end"],
        eval_cutoff=request["eval_cutoff"],
        event_start=request["event_start"],
        event_end=request["event_end"],
        timeout_seconds=request["timeout_seconds"],
        fit_timeout_seconds=float(request["fit_timeout_seconds"]),
        limits=ResourceLimits(**raw_limits),
        _precreated=True,
        _supervised=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-source", help="verify retained source-member hashes")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--scan-members", action="store_true", help="also decode and count each member")

    build = sub.add_parser("build-panel", help="build a new Q1/Q2 panel from a verified member manifest")
    build.add_argument("manifest", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--evidence", type=Path, required=True)
    build.add_argument("--model", default="ST4000DM000")

    archive = sub.add_parser("build-zip-panel", help="stream a Q3 ZIP receipt into a new raw panel")
    archive.add_argument("receipt", type=Path)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--prior-panel", type=Path, help="completed Q1/Q2 panel for cross-quarter identity verification")

    access = sub.add_parser("check-access", help="validate the semantic inputs for one stage process")
    access.add_argument("--role", choices=sorted(ROLE_INPUTS), required=True)
    access.add_argument("--bindings", type=Path, required=True, help="JSON object mapping semantic input names to paths")

    worker = sub.add_parser("worker", help="run one isolated reproducible stage")
    worker.add_argument("--role", choices=("train", "score", "evaluate"), required=True)
    worker.add_argument("--spec", type=Path, required=True, help="JSON stage specification")
    worker.add_argument("--output", type=Path, required=True)

    isolated = sub.add_parser("isolated-run", help="run train, score, and evaluate workers in sequence")
    isolated.add_argument("--train-panel", type=Path, required=True)
    isolated.add_argument("--eval-panel", type=Path, required=True)
    isolated.add_argument("--config", type=Path, required=True)
    isolated.add_argument("--output", type=Path, required=True)
    for prefix in ("train", "eval"):
        isolated.add_argument(f"--{prefix}-start", required=True)
        isolated.add_argument(f"--{prefix}-end", required=True)
        isolated.add_argument(f"--{prefix}-cutoff", required=True)
    isolated.add_argument("--event-start", required=True)
    isolated.add_argument("--event-end", required=True)
    isolated.add_argument("--timeout-seconds", type=int, default=900)
    isolated.add_argument("--fit-timeout-seconds", type=float, default=900)
    isolated.add_argument("--max-rss-mib", type=int, default=3072)
    isolated.add_argument("--max-output-mib", type=int, default=12288)
    isolated.add_argument("--min-free-mib", type=int, default=2048)
    isolated.add_argument("--max-elapsed-seconds", type=float, default=10800)
    isolated.add_argument("--poll-seconds", type=float, default=0.2)

    coordinator = sub.add_parser(
        "isolated-coordinator",
        help="run the isolated workflow from a supervisor-created source snapshot",
    )
    coordinator.add_argument("--request", type=Path, required=True)

    verify_run = sub.add_parser(
        "verify-run",
        help="independently re-hash a supervised output tree and write verification.json",
    )
    verify_run.add_argument("--request", type=Path, required=True)

    rebuild = sub.add_parser("rebuild-panels", help="build real Q1/Q2 then Q3 panels under a guarded parent")
    rebuild.add_argument("--q1q2-manifest", type=Path, required=True)
    rebuild.add_argument("--q3-receipt", type=Path, required=True)
    rebuild.add_argument("--output-root", type=Path, required=True)
    rebuild.add_argument("--evidence", type=Path, required=True)
    rebuild.add_argument("--model", default="ST4000DM000")
    rebuild.add_argument("--max-panel-seconds", type=float, default=4 * 60 * 60)
    rebuild.add_argument("--max-rss-mib", type=int, default=3072)
    rebuild.add_argument("--max-output-mib", type=int, default=12288)
    rebuild.add_argument("--min-free-mib", type=int, default=2048)
    rebuild.add_argument("--poll-seconds", type=float, default=0.2)

    run = sub.add_parser("run", help="fit current_lr and evaluate current_lr/smart_nonzero")
    run.add_argument("--train-panel", type=Path, required=True)
    run.add_argument("--eval-panel", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    for prefix in ("train", "eval"):
        run.add_argument(f"--{prefix}-start", required=True)
        run.add_argument(f"--{prefix}-end", required=True)
        run.add_argument(f"--{prefix}-cutoff", required=True)
    run.add_argument("--event-start", required=True)
    run.add_argument("--event-end", required=True)
    args = parser.parse_args(argv)
    try:
        _check_code_root_binding()
        if args.command == "verify-source":
            from pipeline.reproducible.current import verify_source_manifest
            result = verify_source_manifest(args.manifest, verify_members=args.scan_members)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-panel":
            from pipeline.reproducible.current import build_panel_from_manifest
            result = build_panel_from_manifest(args.manifest, args.output, args.evidence, model=args.model)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-zip-panel":
            from pipeline.reproducible.current import build_archive_panel
            result = build_archive_panel(args.receipt, args.output, prior_panel=args.prior_panel)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "check-access":
            result = authorize_bindings_file(args.role, args.bindings)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "worker":
            from pipeline.reproducible.worker import run_worker
            result = run_worker(args.role, args.spec, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "isolated-run":
            from pipeline.reproducible.orchestrator import isolated_run
            result = isolated_run(
                train_panel=args.train_panel,
                eval_panel=args.eval_panel,
                config=args.config,
                output_dir=args.output,
                train_start=args.train_start,
                train_end=args.train_end,
                train_cutoff=args.train_cutoff,
                eval_start=args.eval_start,
                eval_end=args.eval_end,
                eval_cutoff=args.eval_cutoff,
                event_start=args.event_start,
                event_end=args.event_end,
                timeout_seconds=args.timeout_seconds,
                fit_timeout_seconds=args.fit_timeout_seconds,
                limits=ResourceLimits(
                    max_rss_bytes=args.max_rss_mib * 1024 * 1024,
                    max_output_bytes=args.max_output_mib * 1024 * 1024,
                    min_free_bytes=args.min_free_mib * 1024 * 1024,
                    max_elapsed_seconds=args.max_elapsed_seconds,
                    poll_seconds=args.poll_seconds,
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "isolated-coordinator":
            request = load_json(args.request, "supervisor request")
            if not isinstance(request, dict):
                raise SupervisorError("supervisor request must be an object")
            result = _coordinator(request)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "verify-run":
            from pipeline.reproducible.artifact_verifier import verify_run as _verify_run
            result = _verify_run(args.request)
            print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, ensure_ascii=False, indent=2))
            return 0 if result.get("status") == "pass" else 2
        if args.command == "rebuild-panels":
            from pipeline.reproducible.panel_orchestrator import rebuild_panels
            result = rebuild_panels(
                q1q2_manifest=args.q1q2_manifest,
                q3_receipt=args.q3_receipt,
                output_root=args.output_root,
                evidence_dir=args.evidence,
                model=args.model,
                limits=ResourceLimits(
                    max_rss_bytes=args.max_rss_mib * 1024 * 1024,
                    max_output_bytes=args.max_output_mib * 1024 * 1024,
                    min_free_bytes=args.min_free_mib * 1024 * 1024,
                    max_elapsed_seconds=args.max_panel_seconds,
                    poll_seconds=args.poll_seconds,
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        from pipeline.reproducible.current import ReproducibleStopped, run_current_research
        result = run_current_research(
            train_panel=args.train_panel,
            eval_panel=args.eval_panel,
            output_dir=args.output,
            train_start=args.train_start,
            train_end=args.train_end,
            train_cutoff=args.train_cutoff,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            eval_cutoff=args.eval_cutoff,
            event_start=args.event_start,
            event_end=args.event_end,
        )
        print(result["status"], "scope=" + result["scope"], "specification_acceptance=" + result["specification_acceptance"], result["provenance"])
        for method, summary in result["evaluation"]["summaries"].items():
            print(method, summary["metrics"])
        return 0
    except (ArtifactError, ProcessAccessError, ContractError, SupervisorError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - controlled stops from lazily imported stages
        if type(exc).__name__ in _CONTROLLED_STOP_NAMES and type(exc).__module__.startswith("pipeline."):
            print(f"STOPPED: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
