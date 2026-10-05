#!/usr/bin/env python
"""Narrow entrypoint for the locked 2024 Q1 validation branch.

``fixture`` exercises the full 91-day Q1-shaped chain with synthetic rows.
``run`` is the real entry and is release-gated; it must refuse before opening
the Q1 archive until an approved lock-bound release exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.reproducible.artifacts import ArtifactError, bound_path, sha256_file
from pipeline.r_validation.binding import approve_parameters
from pipeline.r_validation.q1_fixture import build_q1_inputs, q1_spec, synthetic_contract
from pipeline.r_validation.q1_binding import (validate_q1_bound_zip, _verify_lock, _canonical,
                                               HISTORICAL_COMPLETION_SPECS,
                                               historical_source_proof)
from pipeline.r_validation.q1_directory import verify_directory_proof
from pipeline.r_validation.release import ReleaseError, require_release
from tools.run_r_validation import _implementation_binding

CONFIG = ROOT / "configs/q1_2024_validation_v1.json"
PROTOCOL = ROOT / "evidence/protocols/q1_2024_validation_v1.md"
RELEASE = ROOT / "evidence/q1_2024_validation_v1/release.json"
MODEL = ROOT / "examples/small_replay/current_lr.json"
HISTORICAL = {
    "q1q2_panel": ROOT / "data/derived/panel_q1q2_verified_v1.sqlite",
    "q3_panel": ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite",
    "q4_candidate_panel": ROOT / "evidence/r_validation_v1/attempt_001/continuation_001/work/candidate/candidate.sqlite",
}
MANIFEST = ROOT / "evidence/q3/scoring_amended_v2/attempt_001/q3_scoring_amended_v2_complete_manifest_v1.json"


def _config() -> dict:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value.get("version") != "q1_2024_validation_v1":
        raise SystemExit("Q1 configuration version mismatch")
    return value


def _historical_bindings() -> dict:
    proof = historical_source_proof()
    return {role: entry["source"] for role, entry in proof["entries"].items()}


def _lock(output: Path, acceptance: Path | None, directory: Path) -> dict:
    config = _config()
    approved = approve_parameters(
        config["model_binding"]["source_database"],
        config["model_binding"]["full_parameters"],
        expected_source_sha256=config["model_binding"]["source_database_sha256"],
        expected_model="current_lr",
    )
    output = bound_path(output, "Q1 preflight output")
    if output.exists():
        raise SystemExit(f"refusing to overwrite Q1 preflight output: {output}")
    output.mkdir(parents=True)
    directory = bound_path(directory, "Q1 directory preflight", must_exist=True)
    directory_proof = json.loads(directory.read_text(encoding="utf-8"))
    verify_directory_proof(directory_proof, config["source_zip"],
                           start=config["dates"]["quarter_raw"][0],
                           end=config["dates"]["quarter_raw"][1])
    source_proof = historical_source_proof()
    source_proof_path = output / "historical_source_bindings.json"
    source_proof_path.write_text(json.dumps(source_proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    approved_path = output / "approved_parameters.json"
    approved_path.write_text(json.dumps(approved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    panels = {}
    for role, path in HISTORICAL.items():
        if not path.is_file():
            raise SystemExit(f"Q1 historical input is missing: {path}")
        with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            row = db.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM daily WHERE model=?", (config["model"],)).fetchone()
            registry = db.execute("SELECT COUNT(*) FROM serial_model_registry WHERE model=?", (config["model"],)).fetchone()[0]
        if not row[0] or not row[2] or not registry:
            raise SystemExit(f"Q1 historical input has no usable coverage: {role}")
        panels[role] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                        "model": config["model"], "daily_date_min": row[0],
                        "daily_date_max": row[1], "daily_rows": int(row[2]),
                        "registry_rows": int(registry)}
    body = {
        "schema": "q1-lock-v1",
        "protocol": config["protocol"],
        "protocol_source_sha256": sha256_file(PROTOCOL),
        "config_sha256": sha256_file(CONFIG),
        "config_digest": _canonical(config),
        "approved_parameters": {"path": str(approved_path.relative_to(ROOT)),
                                "sha256": sha256_file(approved_path),
                                "source_database_sha256": config["model_binding"]["source_database_sha256"],
                                "checked_features": approved["approval"]["checked_features"]},
        "implementation_sha256": _implementation_binding(),
        "historical_inputs": {"roles": list(HISTORICAL),
                              "source_proof": {"path": str(source_proof_path.relative_to(ROOT)),
                                                "sha256": sha256_file(source_proof_path)},
                              "completion_manifests": {
                                  role: {"path": spec["manifest"],
                                         "sha256": sha256_file(ROOT / spec["manifest"])}
                                  for role, spec in HISTORICAL_COMPLETION_SPECS.items()
                              },
                              "panels": panels},
        "directory_preflight": {"path": str(directory.relative_to(ROOT)),
                                 "sha256": sha256_file(directory)},
        "acceptance_evidence": None,
        "pending_release": {"source_object_id": config["source_zip"]["bz_file_id"],
                             "note": "An approved release must bind this lock before real Q1 access"},
    }
    if acceptance is not None:
        marker = acceptance / "bound_synthetic_complete.json"
        if not marker.is_file():
            raise SystemExit("Q1 acceptance directory has no bound_synthetic_complete.json")
        body["acceptance_evidence"] = {"status": "pass", "attempt": str(acceptance.relative_to(ROOT)),
                                        "files": {str(p.relative_to(ROOT)): sha256_file(p)
                                                  for p in acceptance.rglob("*") if p.is_file()}}
    body["lock_digest"] = _canonical(body)
    target = output / "q1_lock.json"
    target.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "pass", "lock_path": str(target), "lock_digest": body["lock_digest"],
            "acceptance_bound": acceptance is not None}


def cmd_fixture(args) -> int:
    from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
    output = bound_path(args.output, "Q1 fixture output")
    if output.exists():
        raise SystemExit(f"refusing to overwrite fixture output: {output}")
    output.mkdir(parents=True)
    inputs_a = build_q1_inputs(ROOT, output / "inputs_a")
    inputs_b = build_q1_inputs(ROOT, output / "inputs_b", future_variant=True)
    model = MODEL
    contract_a = synthetic_contract(inputs_a, model)
    contract_b = synthetic_contract(inputs_b, model)
    result_a = supervise_zip_fixture(output / "run_a", calendar="quarter", bound_contract=contract_a)
    result_b = supervise_zip_fixture(output / "run_b", calendar="quarter", bound_contract=contract_b)
    selection_a = sha256_file(output / "run_a/work/evaluation/selection.sqlite")
    selection_b = sha256_file(output / "run_b/work/evaluation/selection.sqlite")
    panel_a = sha256_file(output / "run_a/work/prepared/panel.sqlite")
    panel_b = sha256_file(output / "run_b/work/prepared/panel.sqlite")
    import sqlite3
    metadata_connection = sqlite3.connect(output / "run_a/work/evaluation/selection.sqlite")
    metadata = dict(metadata_connection.execute("SELECT key,value FROM metadata"))
    bridge_first = metadata_connection.execute(
        "SELECT MIN(decision_date) FROM features WHERE serial_number=?", ("q1-synthetic-01",)
    ).fetchone()[0]
    old_failure_rows = metadata_connection.execute(
        "SELECT COUNT(*) FROM features WHERE serial_number=?", ("q1-synthetic-00",)
    ).fetchone()[0]
    metadata_connection.close()
    evaluation = json.loads((output / "run_a/work/evaluation/evaluation.json").read_text(encoding="utf-8"))
    event_rows = evaluation["methods"]["current_lr"]["event_summary"]
    outside = evaluation["protocol_metrics"]["methods"]["current_lr"]["outside_main_event_alerts"]
    import zipfile
    with zipfile.ZipFile(inputs_a["archive"]) as archive:
        leap_day_member = any("2024-02-29.csv" in name for name in archive.namelist())
    checks = {
        "calendar_91_84_78": result_a.get("calendar") == "quarter",
        "leap_day_member_present": leap_day_member,
        "bridge_history_used": bridge_first == "2024-01-01",
        "prior_failure_not_reactivated": old_failure_rows == 0,
        "main_and_outside_event_boundaries": (
            any(row.get("first_failure_date") == "2024-02-29" and row.get("opportunity") == 1 for row in event_rows)
            and any(row.get("first_failure_date") == "2024-03-26" for row in outside)
        ),
        "selection_closed_before_evaluation": (
            evaluation.get("evaluation_opened_after_selection_close") is True
            and evaluation.get("selection_sha256_before_evaluation") == evaluation.get("selection_sha256_after_evaluation")
        ),
        "metadata_matches_protocol": (metadata.get("start") == "2024-01-01" and metadata.get("end") == "2024-03-24"
                                       and metadata.get("outcome_cutoff") == "2024-03-31"),
        "bound_completion_a": (output / "run_a/bound_synthetic_complete.json").is_file(),
        "bound_completion_b": (output / "run_b/bound_synthetic_complete.json").is_file(),
        "future_only_change_keeps_selection": selection_a == selection_b,
        "future_only_change_changes_panel": panel_a != panel_b,
        "three_history_roles_bound": set(inputs_a["history"]) == {"q1q2_panel", "q3_panel", "q4_candidate_panel"},
        "both_supervised_pass": result_a.get("status") == "synthetic_complete" and result_b.get("status") == "synthetic_complete",
    }
    payload = {"status": "pass" if all(checks.values()) else "fail", "scope": "q1_full_calendar_synthetic_bound_chain",
               "protocol": q1_spec(), "checks": checks, "run_a": result_a, "run_b": result_b}
    (output / "q1_fixture_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


def cmd_run(args) -> int:
    config = _config()
    try:
        lock = bound_path(args.lock, "Q1 lock", must_exist=True)
        lock_payload = _verify_lock(lock)
        release = require_release(RELEASE, r0_lock_digest=lock_payload.get("lock_digest", ""),
                                  source_object_id=config["source_zip"]["bz_file_id"], stage=args.stage)
    except (ArtifactError, ReleaseError, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "stage": args.stage}, ensure_ascii=False))
        return 3
    if not args.output or not args.source_zip or not args.source_receipt:
        raise SystemExit("release accepted but --output, --source-zip and --source-receipt are required")
    output = bound_path(args.output, "Q1 production output")
    expected_output = (ROOT / config["paths"]["evidence"] / "attempt_001").resolve()
    if output != expected_output:
        raise SystemExit(f"Q1 production output must be {expected_output}")
    if output.exists():
        raise SystemExit("refusing to reuse a Q1 production attempt")
    try:
        receipt_path = bound_path(args.source_receipt, "Q1 source receipt", must_exist=True)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        archive = bound_path(args.source_zip, "Q1 source ZIP", must_exist=True)
        model_path = bound_path(args.model_json or lock_payload["approved_parameters"]["path"],
                                "Q1 approved model", must_exist=True)
        contract = {
            "schema": "r-bound-zip-v1", "profile": "released_q1_zip",
            "source": {"archive": {"path": str(archive), "sha256": receipt.get("archive_sha256")},
                       "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
                       "prefix": config["source_zip"]["prefix"],
                       "start": config["dates"]["quarter_raw"][0],
                       "end": config["dates"]["quarter_raw"][1]},
            "model": {"path": str(model_path), "sha256": lock_payload["approved_parameters"]["sha256"]},
            "history": {role: {"path": str(ROOT / item["path"]), "sha256": item["sha256"]}
                        for role, item in lock_payload["historical_inputs"]["panels"].items()},
            "spec": q1_spec(),
            "authorization": {"lock": {"path": str(lock), "sha256": sha256_file(lock)},
                              "release": {"path": str(RELEASE), "sha256": sha256_file(RELEASE)}},
        }
        validate_q1_bound_zip(contract)
    except (ArtifactError, ReleaseError, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "stage": args.stage}, ensure_ascii=False))
        return 3
    from pipeline.reproducible.resource_guard import ResourceLimits
    from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
    budget = config["budget"]
    limits = ResourceLimits(max_rss_bytes=budget["process_rss_bytes"],
                            max_output_bytes=budget["retention_bytes"],
                            initial_free_bytes=budget["min_free_at_start_bytes"],
                            min_free_bytes=budget["min_free_running_bytes"],
                            max_elapsed_seconds=budget["real_chain_total_seconds"],
                            poll_seconds=budget["sampling_seconds"])
    result = supervise_zip_fixture(output, limits=limits,
                                   worker_timeout_seconds=budget["real_chain_total_seconds"],
                                   calendar="quarter", bound_contract=contract)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_download(args) -> int:
    """Acquire the approved Q1 ZIP through the release-gated supervisor."""
    from pipeline.r_validation.download_supervisor import supervise_download
    from pipeline.r_validation.release import ReleaseError

    try:
        result = supervise_download(lock_path=args.lock, profile="released_q1_download")
    except (ArtifactError, ReleaseError, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_directory(args) -> int:
    """Fetch only the official Q1 central-directory range."""
    from pipeline.r_validation.q1_directory import write_directory_proof
    config = _config()
    output = bound_path(args.output, "Q1 directory proof")
    if output.exists():
        raise SystemExit(f"refusing to overwrite Q1 directory proof: {output}")
    proof = write_directory_proof(
        output, config["source_zip"],
        start=config["dates"]["quarter_raw"][0],
        end=config["dates"]["quarter_raw"][1],
    )
    print(json.dumps({"status": proof["status"], "output": str(output),
                      "entry_count": proof["central_directory"]["entry_count"],
                      "csv_member_count": proof["quarter"]["csv_member_count"],
                      "range": proof["range"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    from pipeline.r_validation.post_worker_audit import audit_worker_output, inventory
    attempt = bound_path(args.attempt, "Q1 synthetic attempt")
    request = json.loads((attempt / "request.json").read_text(encoding="utf-8"))
    contract = request.get("source_contract")
    if not isinstance(contract, dict) or contract.get("profile") != "synthetic_q1_bound_zip":
        raise SystemExit("Q1 audit requires a synthetic_q1_bound_zip attempt")
    result = audit_worker_output(
        attempt / "work",
        inventory(attempt / "work"),
        expected_model_sha256=contract["model"]["sha256"],
        calendar="quarter",
        bound_contract=contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "pass" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--output", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--acceptance", default=None)
    preflight.add_argument("--directory", required=True,
                           help="passing q1-directory-proof-v1 JSON")
    directory = sub.add_parser("directory", help="bounded official ZIP directory preflight")
    directory.add_argument("--output", required=True)
    download = sub.add_parser("download", help="release-gated supervised Q1 ZIP acquisition")
    download.add_argument("--lock", required=True)
    run = sub.add_parser("run")
    run.add_argument("--stage", required=True)
    run.add_argument("--lock", required=True)
    run.add_argument("--source-zip")
    run.add_argument("--source-receipt")
    run.add_argument("--output")
    run.add_argument("--model-json")
    audit = sub.add_parser("audit")
    audit.add_argument("--attempt", required=True)
    args = parser.parse_args(argv)
    if args.command == "fixture":
        return cmd_fixture(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "directory":
        return cmd_directory(args)
    if args.command == "preflight":
        result = _lock(Path(args.output), Path(args.acceptance).resolve() if args.acceptance else None,
                       Path(args.directory))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit":
        return cmd_audit(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
