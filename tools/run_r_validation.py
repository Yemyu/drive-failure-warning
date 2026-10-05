#!/usr/bin/env python
"""R0 CLI: preflight / fixture / run / audit for the R validation protocol.

``preflight`` builds the parameter approval, historical-input binding and
``r0_lock.json``.  ``fixture`` runs the synthetic acceptance pipeline (two
content-different inputs, perturbation, boundary and gate checks) without any
Q4 content.  ``run`` is the real Q4 entry — it requires an approved release
bound to the protocol lock and therefore must refuse during R0.  ``audit`` runs the
independent hand-computed checks over a completed attempt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from pipeline.reproducible.runtime_context import project_root
ROOT = project_root()

from pipeline.r_validation.binding import BindingError, approve_parameters
from pipeline.r_validation.release import ReleaseError, require_release

CONFIG = ROOT / "configs/r_validation_v1.json"
DEFAULT_SOURCE_DB = ROOT / "data/derived/simple_baseline_v2/model_results.sqlite"
DEFAULT_PARAMETERS = ROOT / "examples/small_replay/current_lr.json"
RELEASE_PATH = ROOT / "evidence/r_validation_v1/release.json"
HISTORICAL_PANELS = {
    "prior_panel": ROOT / "data/derived/panel_q1q2_verified_v1.sqlite",
    "q3_panel": ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite",
}
IMPLEMENTATION_FILES = (
    "pipeline/r_validation/zip_execution.py",
    "pipeline/r_validation/budget_scope.py",
    "pipeline/r_validation/synthetic_publication.py",
    "pipeline/r_validation/snapshot_verifier.py",
    "pipeline/r_validation/post_worker_audit.py",
    "pipeline/r_validation/__init__.py",
    "pipeline/r_validation/binding.py",
    "pipeline/r_validation/engine.py",
    "pipeline/r_validation/release.py",
    "pipeline/r_validation/synthetic_suite.py",
    "pipeline/r_validation/cli_fixture.py",
    "pipeline/r_validation/independent_audit.py",
    "pipeline/r_validation/stage_runner.py",
    "pipeline/r_validation/history.py",
    "pipeline/r_validation/zip_source.py",
    "pipeline/r_validation/zip_panel.py",
    "pipeline/r_validation/zip_audit.py",
    "pipeline/r_validation/zip_history.py",
    "pipeline/r_validation/zip_chain.py",
    "pipeline/r_validation/snapshot_worker.py",
    "pipeline/r_validation/outer_supervisor.py",
    "pipeline/r_validation/protocol_fixture.py",
    "tools/run_small_replay.py",
    "pipeline/build_feature_replay.py",
    "pipeline/replay_selection.py",
    "pipeline/labeling.py",
    "examples/small_replay/manifest.json",
    "tools/run_r_validation.py",
)
ACCEPTANCE_FILES = ("tests/test_r_validation.py",)


def implementation_files():
    """Bind the complete snapshot Python closure, not a hand-maintained subset."""
    from pipeline.reproducible.source_snapshot import _files
    return tuple(sorted(set(IMPLEMENTATION_FILES) | {p.relative_to(CODE_ROOT).as_posix() for p in _files(CODE_ROOT)}))


def _load_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("version") != "r_validation_v1":
        raise SystemExit("r_validation config version mismatch")
    return config


def _canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_file(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"lock field {field} is missing")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ReleaseError(f"lock field {field} escapes the project")
    if not resolved.is_file():
        raise ReleaseError(f"lock field {field} points to a missing file: {value}")
    return resolved


def _verify_lock_binding(lock_path: Path) -> dict:
    """Recompute every lock binding immediately before a real-stage gate."""

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"lock is unreadable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema") != "r0-lock-v1":
        raise ReleaseError("lock schema is invalid")
    claimed = lock.get("lock_digest")
    body = {key: value for key, value in lock.items() if key != "lock_digest"}
    if not isinstance(claimed, str) or _canonical_digest(body) != claimed:
        raise ReleaseError("lock self-digest does not match its current content")
    protocol_sha = _sha256_file(ROOT / "evidence/protocols/r_validation_v1.md")
    if lock.get("protocol_source_sha256") != protocol_sha:
        raise ReleaseError("protocol source changed after lock creation")
    if lock.get("config_sha256") != _sha256_file(CONFIG):
        raise ReleaseError("configuration changed after lock creation")
    if lock.get("config_digest") != _canonical_digest(_load_config()):
        raise ReleaseError("configuration digest changed after lock creation")
    implementation = lock.get("implementation_sha256")
    if not isinstance(implementation, dict):
        raise ReleaseError("lock has no implementation binding")
    if set(implementation)!=set(implementation_files()):
        raise ReleaseError('implementation dependency closure changed')
    for relative in implementation_files():
        path = CODE_ROOT / relative if relative.endswith('.py') else _project_file(relative, "implementation")
        if implementation.get(relative) != _sha256_file(path):
            raise ReleaseError(f"implementation changed after lock creation: {relative}")
    approved = lock.get("approved_parameters")
    if not isinstance(approved, dict):
        raise ReleaseError("lock has no approved parameter binding")
    approved_path = _project_file(approved.get("path"), "approved_parameters.path")
    if approved.get("sha256") != _sha256_file(approved_path):
        raise ReleaseError("approved parameter package changed after lock creation")
    historical = lock.get("historical_inputs")
    if not isinstance(historical, dict):
        raise ReleaseError("lock has no historical input binding")
    completion = historical.get("completion_manifest", {})
    completion_path = _project_file(completion.get("path"), "completion_manifest.path")
    if completion.get("sha256") != _sha256_file(completion_path):
        raise ReleaseError("completion manifest changed after lock creation")
    database_path = _project_file(historical.get("database"), "historical database")
    if historical.get("database_sha256") != _sha256_file(database_path):
        raise ReleaseError("historical database changed after lock creation")
    panels = historical.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ReleaseError("lock has no historical panel binding")
    for role, item in panels.items():
        if not isinstance(item, dict):
            raise ReleaseError(f"historical panel binding is invalid: {role}")
        panel_path = _project_file(item.get("path"), f"historical panel {role}")
        if item.get("sha256") != _sha256_file(panel_path):
            raise ReleaseError(f"historical panel changed after lock creation: {role}")
    acceptance = lock.get("acceptance_evidence")
    if not isinstance(acceptance, dict) or acceptance.get("status") != "pass":
        raise ReleaseError("lock has no passing synthetic acceptance binding")
    evidence_files = acceptance.get("files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise ReleaseError("lock has no synthetic evidence file binding")
    for relative, expected in evidence_files.items():
        evidence_path = _project_file(relative, "acceptance_evidence.files")
        if expected != _sha256_file(evidence_path):
            raise ReleaseError(f"synthetic evidence changed after lock creation: {relative}")
    return lock


def _acceptance_binding(attempt: Path) -> dict[str, object]:
    attempt = attempt.resolve()
    if not attempt.is_relative_to(ROOT):
        raise SystemExit("synthetic attempt must stay inside the project")
    complete=attempt/'bound_synthetic_complete.json'
    bound_mode=complete.exists()
    if not bound_mode: complete=attempt/'synthetic_complete.json'
    if complete.exists():
        from pipeline.reproducible.artifacts import verify_manifest
        value=verify_manifest(complete,expected_status='synthetic_complete')
        if value.get('calendar')!='quarter' or value.get('real_data_release') is not False:
            raise SystemExit('full-quarter synthetic completion required')
        from pipeline.reproducible.source_snapshot import _files
        tested = {p.relative_to(attempt/'snapshot').as_posix(): _sha256_file(p)
                  for p in _files(attempt/'snapshot')}
        current = {p.relative_to(CODE_ROOT).as_posix(): _sha256_file(p) for p in _files(CODE_ROOT)}
        if tested != current:
            raise SystemExit('synthetic completion tested different source bytes')
        paths = list(attempt.rglob('*'))
        if any(p.is_symlink() for p in paths):
            raise SystemExit('synthetic acceptance contains symlink')
        bound_files={p.relative_to(ROOT).as_posix():_sha256_file(p) for p in paths if p.is_file()}
        if bound_mode:
            from pipeline.r_validation.bound_zip import validate_bound_zip
            contract=value.get('source_contract',{})
            if contract.get('profile')!='synthetic_bound_zip': raise SystemExit('real source cannot authorise R0')
            validate_bound_zip(contract)
            bindings=[contract['source']['archive'],contract['model'],*contract['history'].values()]
            for binding in bindings:
                external=_project_file(binding['path'],'acceptance external input')
                bound_files[external.relative_to(ROOT).as_posix()]=_sha256_file(external)
        for p in (ROOT/'tests').glob('test_*.py'): bound_files[p.relative_to(ROOT).as_posix()]=_sha256_file(p)
        return {'status':'pass','attempt':attempt.relative_to(ROOT).as_posix(),'files':bound_files,
                'chain_stages':['source','panel','score','seal','evaluate','audit','publish'],
                'independent_audit':True,'calendar':'quarter','completion_manifest_hash':value['manifest_hash']}
    suite = attempt / "synthetic_suite.json"
    chain = attempt / "cli_chain/chain_results.json"
    audit = attempt / "cli_chain/audit.json"
    protocol = attempt / "protocol_chain.json"
    for path in (suite, chain, audit, protocol):
        if not path.is_file():
            raise SystemExit(f"synthetic acceptance file is missing: {path}")
    suite_payload = json.loads(suite.read_text(encoding="utf-8"))
    chain_payload = json.loads(chain.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    if (suite_payload.get("status") != "pass" or chain_payload.get("status") != "pass"
            or audit_payload.get("status") != "pass" or protocol_payload.get("status") != "pass"):
        raise SystemExit("synthetic acceptance is not complete and passing")
    files: dict[str, str] = {}
    # Bind every small fixture input and stage artifact, not just the summary
    # files.  The directory is intentionally bounded and contains no Q4 data.
    for path in sorted(attempt.rglob("*")):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = _sha256_file(path)
    tests_path = ROOT / "tests/test_r_validation.py"
    files[tests_path.relative_to(ROOT).as_posix()] = _sha256_file(tests_path)
    return {
        "status": "pass",
        "attempt": attempt.relative_to(ROOT).as_posix(),
        "files": files,
        "chain_stages": chain_payload.get("stages"),
        "independent_audit": True,
    }


def _historical_panel_binding() -> dict[str, object]:
    """Bind the full historical panels needed for Q4 eligibility.

    The Q3 scoring database contains only scores and alert summaries.  It is
    not a substitute for the Q1/Q2 and Q3 daily panels or their identity
    registries, so the lock records both immutable panel files and a few
    independently queried coverage facts.
    """
    bound: dict[str, object] = {}
    for role, path in HISTORICAL_PANELS.items():
        if not path.is_file():
            raise SystemExit(f"historical panel is missing: {path}")
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            required = {"daily", "serial_model_registry"}
            if not required.issubset(tables):
                raise SystemExit(f"historical panel {role} misses {sorted(required - tables)}")
            daily = connection.execute(
                "SELECT MIN(date), MAX(date), COUNT(*), SUM(CASE WHEN failure=1 THEN 1 ELSE 0 END) "
                "FROM daily WHERE model=?", ("ST4000DM000",)
            ).fetchone()
            registry = connection.execute(
                "SELECT COUNT(*) FROM serial_model_registry WHERE model=?", ("ST4000DM000",)
            ).fetchone()[0]
            if not daily[0] or not daily[1] or not daily[2] or registry <= 0:
                raise SystemExit(f"historical panel {role} has no usable ST4000DM000 coverage")
            bound[role] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256_file(path),
                "model": "ST4000DM000",
                "daily_date_min": daily[0],
                "daily_date_max": daily[1],
                "daily_rows": int(daily[2]),
                "failure_rows": int(daily[3] or 0),
                "registry_rows": int(registry),
            }
        finally:
            connection.close()
    return bound


def _implementation_binding() -> dict[str, str]:
    result = {}
    for relative in implementation_files():
        path = CODE_ROOT / relative if relative.endswith('.py') else ROOT / relative
        if not path.is_file():
            raise SystemExit(f"implementation file is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def cmd_preflight(args) -> int:
    config = _load_config()
    binding = config["model_binding"]
    approved = approve_parameters(
        args.source_database or binding["source_database"],
        args.parameters or binding["full_parameters"],
        expected_source_sha256=binding["source_database_sha256"],
        expected_model="current_lr",
    )
    # copy the approved package into the R binding area (new file, no overwrite)
    if args.output is None:
        raise SystemExit("--output is required for preflight")
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("output must stay inside the project")
    if output.exists() and not args.allow_existing:
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=args.allow_existing)
    package_path = output / "approved_parameters.json"
    package_path.write_text(
        json.dumps(approved, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # verify the copied package independently
    from pipeline.r_validation.binding import verify_approved_package

    verify_approved_package(package_path, source_database=binding["source_database"])

    # historical inputs: the accepted completion manifest and its bound files
    manifest_path = ROOT / "evidence/q3/scoring_amended_v2/attempt_001/q3_scoring_amended_v2_complete_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("attempt_id"):
        raise SystemExit("historical completion manifest is not complete")
    scoring_database = ROOT / str(manifest.get("database", ""))
    declared_database_sha = str(manifest.get("database_sha256", ""))
    if not scoring_database.is_file() or _sha256_file(scoring_database) != declared_database_sha:
        raise SystemExit("historical scoring database does not match its completion manifest")
    bound = {
        "completion_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "status": manifest.get("status"),
            "attempt_id": manifest.get("attempt_id"),
        },
        "database": str(scoring_database.relative_to(ROOT)),
        "database_sha256": declared_database_sha,
        "coverage": {
            "prior_failures_through": config["dates"]["quarter_raw"][0],
            "daily_history": [config["dates"]["bridge_history_start"], "2023-09-30"],
            "note": "覆盖 2023-01-01—09-30 既往 failure 与 09-18—09-30 每日历史；不能以 126 行评价账本代替全部既往 failure",
        },
        "panels": _historical_panel_binding(),
    }
    lock = {
        "schema": "r0-lock-v1",
        "protocol": config["protocol"],
        "protocol_source_sha256": hashlib.sha256(
            (ROOT / "evidence/protocols/r_validation_v1.md").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "config_digest": _canonical_digest(config),
        "approved_parameters": {
            "path": str(package_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
            "source_database_sha256": approved["approval"]["source_database_sha256"],
            "checked_features": approved["approval"]["checked_features"],
        },
        "implementation_sha256": _implementation_binding(),
        "historical_inputs": bound,
        "pending_release": {
            "note": "审批通过后填写 release.json，并绑定本 lock 摘要",
            "source_object_id": config["source_zip"]["bz_file_id"],
        },
    }
    if args.synthetic_attempt:
        lock["acceptance_evidence"] = _acceptance_binding(
            _project_file(args.synthetic_attempt, "synthetic_attempt")
            if Path(args.synthetic_attempt).is_file() else
            (ROOT / args.synthetic_attempt if not Path(args.synthetic_attempt).is_absolute() else Path(args.synthetic_attempt)).resolve()
        )
    lock["lock_digest"] = _canonical_digest(lock)
    (output / "r0_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": output.as_posix(), "lock_digest": lock["lock_digest"],
                      "checked_features": approved["approval"]["checked_features"]},
                     ensure_ascii=False))
    return 0


def cmd_fixture(args) -> int:
    """Run the synthetic acceptance pipeline (no Q4 content)."""
    from pipeline.r_validation.synthetic_suite import run_synthetic_suite
    from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
    from pipeline.r_validation.protocol_fixture import run_protocol_fixture

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if not output.exists() and not args.allow_existing:
        output.mkdir(parents=True)
    # The stage runner requires a new output root.  Run file-backed chains
    # before the pure-function suite writes its summary into that root.
    results: dict[str, object] = {}
    results["cli_chain"] = run_cli_fixture_chain(output, root=ROOT)
    results["protocol_chain"] = run_protocol_fixture(output, root=ROOT)
    suite = run_synthetic_suite(output, root=ROOT)
    results = {**suite, **results}
    (output / "synthetic_suite.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results.get("status") == "pass" else 1


def cmd_run(args) -> int:
    """Real ZIP entry: release-bound, snapshot-isolated and externally verified."""
    config = _load_config()
    lock_path = Path(args.lock) if args.lock else None
    lock_digest = None
    lock = None
    if lock_path and lock_path.is_file():
        lock = _verify_lock_binding(lock_path.resolve())
        lock_digest = lock["lock_digest"]
    source_object_id = config["source_zip"]["bz_file_id"]
    try:
        release = require_release(
            RELEASE_PATH,
            r0_lock_digest=lock_digest or "",
            source_object_id=source_object_id,
            stage=args.stage,
        )
    except ReleaseError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc),
                          "stage": args.stage}, ensure_ascii=False))
        return 3
    if args.source_csv:
        raise SystemExit('CSV production adapter retired; use the approved ZIP and download receipt')
    if not args.output or not args.source_zip or not args.source_receipt:
        raise SystemExit('release accepted but --output, --source-zip and --source-receipt are required')
    if lock is None:
        raise SystemExit('a verified R0 lock is required')
    from pipeline.reproducible.artifacts import bound_path
    from pipeline.reproducible.resource_guard import ResourceLimits
    from pipeline.reproducible.supervisor import _load_small
    from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
    output = bound_path(args.output, 'production output')
    expected_output = (ROOT / config['paths']['evidence']).resolve()
    from pipeline.r_validation.continuation import validate_continuation
    continuation=validate_continuation(release,lock,config['source_zip'])
    if continuation is not None:expected_output=continuation['output']
    if output != expected_output:
        raise SystemExit(f'production output must be {expected_output}')
    if output.exists(): raise SystemExit('refusing to reuse a production attempt')
    receipt_path = bound_path(args.source_receipt, 'download receipt', must_exist=True)
    receipt = _load_small(receipt_path, 'download receipt')
    archive = bound_path(args.source_zip, 'production ZIP', must_exist=True)
    model_path = bound_path(args.model_json or lock['approved_parameters']['path'], 'approved model', must_exist=True)
    spec = {key: config['dates'][key] for key in
            ('score_start', 'score_end', 'event_start', 'event_end', 'outcome_cutoff')}
    spec.update(horizon_days=config['labels']['horizon_days'], allow_smart_decreases=True)
    contract = {'schema':'r-bound-zip-v1','profile':'released_zip',
        'source': {'archive': {'path':str(archive),'sha256':receipt.get('archive_sha256')},
                   'receipt': {'path':str(receipt_path),'sha256':_sha256_file(receipt_path)},
                   'prefix':'data_Q4_2023','start':config['dates']['quarter_raw'][0],
                   'end':config['dates']['quarter_raw'][1]},
        'model': {'path':str(model_path),'sha256':lock['approved_parameters']['sha256']},
        'history': {role:{'path':str(_project_file(item['path'],role)),'sha256':item['sha256']}
                    for role,item in lock['historical_inputs']['panels'].items()},
        'spec':spec, 'authorization':{
            'lock':{'path':str(lock_path.resolve()),'sha256':_sha256_file(lock_path)},
            'release':{'path':str(RELEASE_PATH),'sha256':_sha256_file(RELEASE_PATH)}}}
    budget=config['budget']
    limits=ResourceLimits(max_rss_bytes=budget['process_rss_bytes'],max_output_bytes=budget['retention_bytes'],
        initial_free_bytes=budget['min_free_at_start_bytes'],min_free_bytes=budget['min_free_running_bytes'],
        max_elapsed_seconds=budget['real_chain_total_seconds'],poll_seconds=budget['sampling_seconds'])
    result=supervise_zip_fixture(output,limits=limits,worker_timeout_seconds=budget['real_chain_total_seconds'],
                                 calendar='quarter',bound_contract=contract)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_audit(args) -> int:
    """Independent audit over a completed (synthetic) attempt directory."""
    from pipeline.r_validation.synthetic_suite import run_independent_audit

    attempt = Path(args.attempt)
    if not attempt.is_absolute():
        attempt = ROOT / attempt
    results = run_independent_audit(attempt, root=ROOT)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results.get("status") == "pass" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser('download', help='one release-gated supervised source download')
    download.add_argument('--lock', required=True)
    download_worker = sub.add_parser('download-worker', help='internal snapshot-bound download worker')
    download_worker.add_argument('--request', required=True)
    download_worker.add_argument('--request-sha256', required=True)

    bound_fixture = sub.add_parser('bound-zip-fixture', help='self-generated ZIP through external-input supervision')
    bound_fixture.add_argument('--output', required=True)
    pre = sub.add_parser("preflight", help="parameter approval + r0_lock.json")
    pre.add_argument("--output", required=True)
    pre.add_argument("--source-database", default=None)
    pre.add_argument("--parameters", default=None)
    pre.add_argument("--allow-existing", action="store_true")
    pre.add_argument("--synthetic-attempt", default=None,
                     help="passing fixture attempt to bind into the lock")

    fix = sub.add_parser("fixture", help="synthetic acceptance suite (no Q4)")
    fix.add_argument("--output", required=True)
    fix.add_argument("--allow-existing", action="store_true")

    zip_fix = sub.add_parser("zip-fixture", help="self-generated ZIP through scoring and audit; no real source accepted")
    zip_fix.add_argument("--output", required=True)
    zip_fix.add_argument("--calendar", choices=("short","quarter"), default="short")
    zip_fix.add_argument("--devices", type=int, default=10)

    worker = sub.add_parser("snapshot-worker", help="internal fixed synthetic worker; requires bound snapshot request")
    worker.add_argument("--request", required=True)
    worker.add_argument("--request-sha256", required=True)

    verifier = sub.add_parser("snapshot-verifier", help="internal bound synthetic verification process")
    verifier.add_argument("--request", required=True)
    verifier.add_argument("--request-sha256", required=True)

    supervised = sub.add_parser("supervised-zip-fixture", help="supervise and complete fixed synthetic ZIP run; no real-data release")
    supervised.add_argument("--output", required=True)
    supervised.add_argument("--calendar", choices=("short","quarter"), default="short")

    post = sub.add_parser("audit-zip-worker", help="read-only fresh audit of fixed synthetic worker output")
    post.add_argument("--attempt", required=True)
    post.add_argument("--inventory", required=True)
    post.add_argument("--inventory-sha256", required=True)
    post.add_argument("--model-sha256", required=True)
    post.add_argument("--output", required=True)
    post.add_argument("--calendar", choices=("short","quarter"), default="short")

    run = sub.add_parser("run", help="real Q4 entry (release-gated)")
    run.add_argument("--stage", required=True)
    run.add_argument("--lock", default=None)
    run.add_argument("--source-zip", help="approved local quarter ZIP")
    run.add_argument("--source-csv", default=None, help="retired CSV adapter; rejected for real runs")
    run.add_argument("--source-receipt", default=None, help="source-object/member receipt bound after release")
    run.add_argument("--model-json", default=None, help="approved frozen model payload")
    run.add_argument("--output", default=None, help="new output directory")

    aud = sub.add_parser("audit", help="independent audit of an attempt")
    aud.add_argument("--attempt", required=True)

    args = parser.parse_args(argv)
    if args.command == 'download':
        from pipeline.r_validation.download_supervisor import supervise_download
        try:
            result=supervise_download(lock_path=args.lock)
        except ReleaseError as exc:
            print(json.dumps({'status':'refused','reason':str(exc)}));return 3
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command == 'download-worker':
        from pipeline.r_validation.download_supervisor import run_download_worker
        result=run_download_worker(args.request,args.request_sha256)
        print(json.dumps({'status':result['status']}));return 0
    if args.command == 'bound-zip-fixture':
        from pipeline.r_validation.zip_chain import run_bound_fixture
        print(json.dumps(run_bound_fixture(args.output),ensure_ascii=False,indent=2))
        return 0
    if args.command == "snapshot-verifier":
        from pipeline.r_validation.snapshot_verifier import run_verifier
        result=run_verifier(args.request,args.request_sha256)
        print(json.dumps({"status":result["status"]}))
        return 0
    if args.command == "audit-zip-worker":
        from pipeline.r_validation.post_worker_audit import audit_worker_output
        from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
        from pipeline.reproducible.supervisor import _load_small
        inventory_path=bound_path(args.inventory,"audit inventory",must_exist=True)
        if sha256_file(inventory_path)!=args.inventory_sha256:
            raise ValueError("audit inventory SHA mismatch")
        target=bound_path(args.output,"audit result")
        attempt=bound_path(args.attempt,"audit attempt")
        if target.exists() or target.is_relative_to(attempt):
            raise ValueError("audit result must be new and outside audited input")
        result=audit_worker_output(attempt,_load_small(inventory_path,"audit inventory"),
                                   expected_model_sha256=args.model_sha256,calendar=args.calendar)
        if sha256_file(inventory_path)!=args.inventory_sha256:
            raise ValueError("audit inventory changed")
        write_json_exclusive(target,result)
        print(json.dumps({"status":result["status"]}))
        return 0
    if args.command == "supervised-zip-fixture":
        from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
        result = supervise_zip_fixture(args.output,calendar=args.calendar)
        print(json.dumps({"status": result["status"],"synthetic_completion":result["synthetic_completion"]}))
        return 0
    if args.command == "snapshot-worker":
        from pipeline.r_validation.snapshot_worker import run_worker
        result = run_worker(args.request, args.request_sha256)
        print(json.dumps({"status": result["status"]}))
        return 0
    if args.command == "zip-fixture":
        from pipeline.r_validation.zip_chain import run_zip_fixture
        result = run_zip_fixture(args.output,calendar=args.calendar,device_count=args.devices)
        print(json.dumps({"status": result["status"], "profile": result["profile"]}))
        return 0
    return {"preflight": cmd_preflight, "fixture": cmd_fixture,
            "run": cmd_run, "audit": cmd_audit}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
