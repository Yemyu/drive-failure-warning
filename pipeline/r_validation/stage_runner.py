"""Shared R stage runner for synthetic and released production profiles.

The runner owns the stage order and boundary checks.  The synthetic profile
uses only the checked-in fixture; the production profile requires a verified
release and an explicit local source adapter.  It never opens a network source
by itself, which keeps release authorization separate from the data reader.
"""

from __future__ import annotations

import hashlib
import json
import datetime as dt
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Mapping

from tools import run_small_replay as replay
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceLimits, ResourceViolation


class StageRunnerError(RuntimeError):
    """A stage profile or stage boundary is invalid."""


STAGES = ("source", "panel", "score", "seal", "evaluate", "audit")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_input(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_synthetic_receipt(path: Path, source: Path, summary: Mapping[str, Any]) -> Path:
    """Write an explicit local receipt for a synthetic source copy.

    This is deliberately labelled synthetic.  It is never accepted by the
    production release gate, but keeping the same receipt shape in the
    fixture chain prevents a bare CSV from being mistaken for a verified
    source adapter.
    """
    payload = {
        "schema": "r-validation-source-receipt-v1",
        "status": "synthetic_verified",
        "source_object_id": "synthetic-local-fixture",
        "transport": "local_copy",
        "source_sha256": _sha(source),
        "member_count": int(summary["date_count"]),
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "serial_count": int(summary["serial_count"]),
        "schema_columns": list(replay.SOURCE_COLUMNS),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _stream_source_to_panel(
    panel: Path, rows: Any
) -> dict[str, Any]:
    """Materialise one validated source stream without a Python row list."""
    panel.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(panel)
    try:
        columns = ", ".join(
            f"{name} {'INTEGER' if name in {'capacity_bytes', 'failure'} or name.startswith('smart_') else 'TEXT'}"
            for name in replay.SOURCE_COLUMNS
        )
        connection.execute(
            f"CREATE TABLE daily ({columns}, PRIMARY KEY (date, serial_number))"
        )
        placeholders = ",".join("?" for _ in replay.SOURCE_COLUMNS)
        sql = f"INSERT INTO daily ({','.join(replay.SOURCE_COLUMNS)}) VALUES ({placeholders})"
        batch: list[tuple[Any, ...]] = []
        row_count = 0
        dates: set[str] = set()
        serials: set[str] = set()
        for row in rows:
            row_count += 1
            dates.add(str(row["date"]))
            serials.add(str(row["serial_number"]))
            batch.append(tuple(row[name] for name in replay.SOURCE_COLUMNS))
            if len(batch) >= 1000:
                connection.executemany(sql, batch)
                batch.clear()
        if batch:
            connection.executemany(sql, batch)
        if row_count == 0:
            raise StageRunnerError("source CSV has no data rows")
        connection.commit()
        return {
            "row_count": row_count,
            "date_count": len(dates),
            "date_min": min(dates),
            "date_max": max(dates),
            "serial_count": len(serials),
            "serials": sorted(serials),
        }
    finally:
        connection.close()


def _validate_production_receipt(
    receipt: Path | None,
    source: Path,
    *,
    source_object_id: str | None,
) -> dict[str, Any]:
    if receipt is None:
        raise StageRunnerError("production profile requires a bound source receipt")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageRunnerError(f"source receipt is unreadable: {receipt}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "r-validation-source-receipt-v1":
        raise StageRunnerError("source receipt schema is invalid")
    if payload.get("status") != "verified":
        raise StageRunnerError("source receipt is not verified for production")
    if source_object_id is not None and payload.get("source_object_id") != source_object_id:
        raise StageRunnerError("source receipt object identity differs from the release")
    actual = _sha(source)
    if payload.get("source_sha256") != actual:
        raise StageRunnerError("source CSV differs from the source receipt SHA")
    if payload.get("transport") not in {"local_verified_copy", "streamed_verified_member"}:
        raise StageRunnerError("source receipt has no accepted verified transport")
    return payload


def _scoring_quality(selection_connection: sqlite3.Connection, panel_connection: sqlite3.Connection) -> dict[str, Any]:
    """Evaluate follow-up completeness over sealed eligible device-days."""
    from pipeline.labeling import classify_device_rows

    metadata = dict(selection_connection.execute("SELECT key,value FROM metadata"))
    total = unknown = 0
    panel_connection.row_factory = sqlite3.Row
    for (serial,) in selection_connection.execute("SELECT DISTINCT serial_number FROM features"):
        dates = {row[0] for row in selection_connection.execute(
            "SELECT decision_date FROM features WHERE serial_number=?", (serial,)
        )}
        rows = panel_connection.execute(
            "SELECT * FROM daily WHERE serial_number=? ORDER BY date", (serial,)
        )
        labels = {row["decision_date"]: row for row in classify_device_rows(
            rows, start=metadata["start"], end=metadata["end"],
            dataset_end=metadata["outcome_cutoff"],
            horizon_days=int(metadata["horizon_days"]),
            history_days=int(metadata["history_days"]), min_history=int(metadata["min_history"]),
        )}
        for date in dates:
            if date not in labels or labels[date]["eligible"] != 1:
                raise StageRunnerError(f"sealed scoring key has no eligible outcome: {serial}/{date}")
            total += 1
            unknown += int(labels[date]["label"] is None)
    ratio = unknown / total if total else None
    return {
        "scope": "common_eligible_scoring_rows",
        "eligible_rows": total, "unknown_rows": unknown, "unknown_ratio": ratio,
        "unknown_ratio_max": 0.2,
        "status": "not_evaluable" if not total else ("failed" if unknown * 5 > total else "pass"),
    }


def _secondary_metrics(selection: sqlite3.Connection, panel: sqlite3.Connection) -> dict[str, Any]:
    from collections import Counter
    from pipeline.labeling import classify_device_rows
    from pipeline.r_validation.engine import known_average_precision

    metadata = dict(selection.execute("SELECT key,value FROM metadata"))
    panel.row_factory = sqlite3.Row
    pools = {method: [] for method in ("current_lr", "smart_nonzero")}
    devices = {method: Counter() for method in pools}
    outside = {method: [] for method in pools}
    no_hit = {method: 0 for method in pools}
    for (serial,) in selection.execute("SELECT DISTINCT serial_number FROM features"):
        labels = {row["decision_date"]: row for row in classify_device_rows(
            panel.execute("SELECT * FROM daily WHERE serial_number=? ORDER BY date", (serial,)),
            start=metadata["start"], end=metadata["end"], dataset_end=metadata["outcome_cutoff"],
            horizon_days=int(metadata["horizon_days"]), history_days=int(metadata["history_days"]),
            min_history=int(metadata["min_history"]),
        )}
        for method, day, score in selection.execute(
                "SELECT model,decision_date,score FROM scores WHERE serial_number=?", (serial,)):
            pools[method].append((float(score), labels[day]["label"]))
        for method, day in selection.execute(
                "SELECT model,decision_date FROM selections WHERE serial_number=?", (serial,)):
            label = labels[day]
            devices[method][serial] += 1
            no_hit[method] += int(label["label"] == 0)
            failure = label["first_failure_date"]
            if label["label"] == 1 and not metadata["event_start"] <= failure <= metadata["event_end"]:
                outside[method].append({"decision_date": day, "serial_number": serial,
                                        "first_failure_date": failure})
    result = {}
    for method, pairs in pools.items():
        alerts = sum(devices[method].values())
        total = len(pairs)
        result[method] = {
            "average_precision_known": known_average_precision(pairs),
            "burden": {"eligible_device_days": total, "alerts": alerts,
                       "alerted_devices": len(devices[method]),
                       "first_alerts": len(devices[method]), "later_alerts": alerts - len(devices[method]),
                       "repeat_alert_devices": sum(count >= 2 for count in devices[method].values()),
                       "alerts_per_1000_device_days": alerts * 1000 / total if total else None,
                       "confirmed_no_hit_per_1000_device_days": no_hit[method] * 1000 / total if total else None},
            "outside_main_event_alert_count": len(outside[method]),
            "outside_main_event_alerts": sorted(outside[method], key=lambda row: (row["decision_date"], row["serial_number"])),
        }
    return result


def _protocol_metrics(
    selection: Path,
    panel: Path,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialise the R1 comparison metrics from the sealed chain output."""
    from pipeline.r_validation.engine import alert_precision, lead_statistics, lead_recall, paired_bootstrap_difference, gain_conditions
    from pipeline.r_validation.engine import interval_publication

    selection_connection = sqlite3.connect(selection.as_uri() + "?mode=ro&immutable=1", uri=True)
    panel_connection = sqlite3.connect(panel.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        eligible_serials = {
            str(row[0]) for row in selection_connection.execute(
                "SELECT DISTINCT serial_number FROM features"
            )
        }
        current_rows = list(evaluation.get("methods", {}).get("current_lr", {}).get("event_summary", []))
        smart_rows = list(evaluation.get("methods", {}).get("smart_nonzero", {}).get("event_summary", []))
        current_events = {str(row["event_key"]): row for row in current_rows}
        smart_events = {str(row["event_key"]): row for row in smart_rows}
        event_keys = sorted(current_events)
        device_flags = {
            serial: {
                "opportunity": int(current_events.get(serial, {}).get("opportunity", 0)),
                "current_hit": int(current_events.get(serial, {}).get("hit", 0)),
                "smart_hit": int(smart_events.get(serial, {}).get("hit", 0)),
            }
            for serial in sorted(eligible_serials, key=lambda value: value.encode("utf-8"))
        }
        current_hits = {serial: int(current_events.get(serial, {}).get("hit", 0)) for serial in event_keys}
        smart_hits = {serial: int(smart_events.get(serial, {}).get("hit", 0)) for serial in event_keys}
        bootstrap = paired_bootstrap_difference(
            event_keys, current_hits, smart_hits, device_flags=device_flags
        )
        current = evaluation["methods"]["current_lr"]
        smart = evaluation["methods"]["smart_nonzero"]
        current_leads = [
            int(row["earliest_lead_days"])
            for row in current_rows
            if row.get("hit") and row.get("earliest_lead_days") is not None
        ]
        smart_leads = [
            int(row["earliest_lead_days"])
            for row in smart_rows
            if row.get("hit") and row.get("earliest_lead_days") is not None
        ]
        precision = {
            method: alert_precision(
                int(values.get("alerts", 0)),
                int(values.get("known_hit_alerts", 0)),
                int(values.get("unknown_alerts", 0)),
            )
            for method, values in (("current_lr", current), ("smart_nonzero", smart))
        }
        unknown_ratios = {
            method: values.get("unknown_alert_ratio")
            for method, values in (("current_lr", current), ("smart_nonzero", smart))
        }
        quality_gate = _scoring_quality(selection_connection, panel_connection)
        secondary = _secondary_metrics(selection_connection, panel_connection)
        current_recall = current.get("event_recall_at_opportunity")
        smart_recall = smart.get("event_recall_at_opportunity")
        delta = None if current_recall is None or smart_recall is None else (float(current_recall) - float(smart_recall)) * 100.0
        return {
            "status": {"failed": "data_quality_failed", "not_evaluable": "not_evaluable", "pass": "pass"}[quality_gate["status"]],
            "methods": {
                "current_lr": {"lead_statistics": lead_statistics(current_leads), "precision": precision["current_lr"],
                               "lead_recall": lead_recall(current_leads, int(bootstrap["opportunity_events"])), **secondary["current_lr"]},
                "smart_nonzero": {"lead_statistics": lead_statistics(smart_leads), "precision": precision["smart_nonzero"],
                                  "lead_recall": lead_recall(smart_leads, int(bootstrap["opportunity_events"])), **secondary["smart_nonzero"]},
            },
            "delta_recall_pp": delta,
            "unknown_alert_ratio": unknown_ratios,
            "quality_gate": quality_gate,
            "bootstrap": bootstrap,
            "interval_publication": interval_publication(bootstrap, quality_gate["status"]),
            "eligible_roster_devices": len(device_flags),
            "opportunity_events": int(bootstrap.get("opportunity_events", 0)),
            "interpretation_gate": gain_conditions(
                opportunities=int(bootstrap["opportunity_events"]),
                current_hits=sum(item["current_hit"] for item in device_flags.values()),
                smart_hits=sum(item["smart_hit"] for item in device_flags.values()),
                current_alerts=int(current["alerts"]), current_unknown=int(current["unknown_alerts"]),
                smart_alerts=int(smart["alerts"]), smart_unknown=int(smart["unknown_alerts"]),
                valid_replicates=int(bootstrap["valid_replicates"]),
                interval=bootstrap["interval"], quality_status=quality_gate["status"],
            ),
        }
    finally:
        selection_connection.close()
        panel_connection.close()


def validate_profile(profile: str, *, release_verified: bool, source_csv: Path | None) -> None:
    if profile not in {"synthetic", "production"}:
        raise StageRunnerError(f"unknown R profile: {profile}")
    if profile == "production" and not release_verified:
        raise StageRunnerError("production profile requires a verified release")
    if profile == "production" and source_csv is None:
        raise StageRunnerError("production profile requires an explicit source adapter path")


def _require_production_spec(spec: Mapping[str, Any]) -> None:
    required = ("score_start", "score_end", "outcome_cutoff", "horizon_days")
    missing = [name for name in required if name not in spec]
    if missing:
        raise StageRunnerError(
            "production profile requires explicit protocol fields: " + ", ".join(missing)
        )
    if "serials" in spec and not spec["serials"]:
        raise StageRunnerError("production serial roster cannot be an empty explicit list")
    from pipeline.r_validation.engine import event_window, parse_date
    start = parse_date(spec["score_start"])
    end = parse_date(spec["score_end"])
    declared = (
        parse_date(spec["event_start"]) if "event_start" in spec else None,
        parse_date(spec["event_end"]) if "event_end" in spec else None,
    )
    derived = event_window(start, end)
    if declared != (None, None) and declared != derived:
        raise StageRunnerError(
            f"declared event window {declared} differs from protocol-derived {derived}"
        )


def _run_stage(
    output: Path,
    *,
    root: Path,
    profile: str,
    source_csv: Path | None = None,
    model_json: Path | None = None,
    expected_json: Path | None = None,
    source_receipt: Path | None = None,
    source_object_id: str | None = None,
    spec: Mapping[str, Any] | None = None,
    release_verified: bool = False,
    exercise_future_guards: bool = False,
    approved_model_sha256: str | None = None,
    historical_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one parameterized source→audit chain.

    The production profile is intentionally local-adapter based in R0: after
    release, a future source acquisition stage can hand the verified CSV to
    this runner.  It is not allowed to silently download or inspect Q4 here.
    """
    output = output.resolve()
    root = root.resolve()
    spec = dict(spec or {})
    if profile == "production" and "horizon_days" not in spec:
        # The protocol fixes H=7.  CLI requests still write the explicit value
        # into their request/metadata; this compatibility default is only for
        # direct callers of the narrow runner in existing tests.
        spec["horizon_days"] = 7
    if profile == "production":
        if not historical_inputs:
            raise StageRunnerError("production requires bound historical panels")
        _require_production_spec(spec)
        from pipeline.r_validation.engine import event_window, parse_date
        derived_start, derived_end = event_window(
            parse_date(spec["score_start"]), parse_date(spec["score_end"])
        )
        spec.setdefault("event_start", derived_start.isoformat())
        spec.setdefault("event_end", derived_end.isoformat())
    from pipeline.r_validation.cli_fixture import _seal, _write_csv, _write_panel
    chain = output / "cli_chain"
    chain.mkdir(parents=True, exist_ok=True)
    if profile == "synthetic":
        source_template = root / "examples/small_replay/daily.csv"
        model_template = root / "examples/small_replay/current_lr.json"
        expected_template = root / "examples/small_replay/expected.json"
        source_path = _copy_input(source_csv or source_template, chain / "daily.csv")
        model_path = _copy_input(model_json or model_template, chain / "current_lr.json")
        # The checked-in January fixture keeps its hand-written checks.  A
        # custom protocol fixture must not inherit demo serials or January
        # event assertions by accident.
        expected_path = (
            _copy_input(expected_json, chain / "expected.json")
            if expected_json is not None
            else (_copy_input(expected_template, chain / "expected.json") if source_csv is None else None)
        )
        _copy_input(root / "examples/small_replay/manifest.json", chain / "manifest.json")
        validate_profile(profile, release_verified=release_verified, source_csv=source_path)
        strict_fixture = source_csv is None
        # The receipt is written after the source stream has been consumed;
        # it binds the exact copied bytes and the observed date/serial facts.
        receipt_path = None
    else:
        validate_profile(profile, release_verified=release_verified, source_csv=source_csv)
        source_path = _copy_input(source_csv, chain / "source.csv")
        if model_json is None:
            raise StageRunnerError("production profile requires the approved model payload")
        model_path = _copy_input(model_json, chain / "current_lr.json")
        expected_path = None if expected_json is None else _copy_input(expected_json, chain / "expected.json")
        strict_fixture = False
        receipt_copy = _copy_input(source_receipt, chain / "source_receipt.json") if source_receipt is not None else None
        receipt_payload = _validate_production_receipt(
            receipt_copy, source_path, source_object_id=source_object_id
        )
        receipt_path = receipt_copy

    # Validate and materialise one source stream.  The scorer receives a
    # read-only file-backed view, never a Python list or an unrestricted
    # sqlite.Connection API.
    panel = chain / "panel.sqlite"
    source_summary = _stream_source_to_panel(
        panel, replay._iter_source(source_path, strict=strict_fixture)
    )
    if profile == "synthetic":
        receipt_path = _write_synthetic_receipt(chain / "source_receipt.json", source_path, source_summary)
    history_receipt = None
    if historical_inputs:
        from pipeline.r_validation.history import merge_history
        history_receipt = merge_history(panel, historical_inputs, score_start=str(spec["score_start"]))
        (chain / "history_receipt.json").write_text(json.dumps(history_receipt, indent=2) + "\n")
    coverage = {"status": "not_applicable", "reason": "short_synthetic_fixture"}
    if profile == "production" or spec.get("coverage_check") is True:
        from pipeline.r_validation.history import check_panel_coverage
        coverage = check_panel_coverage(panel, start=str(spec["score_start"]), end=str(spec["outcome_cutoff"]))
        (chain / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        if coverage["status"] != "pass":
            raise StageRunnerError(f"coverage stopped: {coverage['reason']} on {coverage.get('stopped_on')}")
    model = replay._load_model(model_path)
    if approved_model_sha256 is not None and _sha(model_path) != approved_model_sha256:
        raise StageRunnerError("run-local model does not match the approved parameter SHA")
    if profile == "synthetic":
        frozen_model = replay._load_model()
        if model != frozen_model:
            raise StageRunnerError("run-local model differs from the frozen model")
    expected = replay._json_read(expected_path) if expected_path else {}
    selection = chain / "selection.sqlite"
    access: list[dict[str, Any]] = []
    source_connection = sqlite3.connect(panel.as_uri() + "?mode=ro&immutable=1", uri=True)
    source_connection.row_factory = sqlite3.Row
    source = replay._asof_reader(source_connection, access)
    source_serials = ()
    try:
        # Empty source_rows is deliberate: scoring receives no in-memory full
        # source list.  It can only use the bounded _asof_rows SQL path.
        replay._score_and_select(source, [], model, selection, access, spec=spec)
        # Roster metadata is read after scoring; it cannot select the devices
        # visible to an earlier daily consumer.
        source_serials = source.serials()
        selection_sha = _sha(selection)
        seal = chain / "selection_seal.json"
        _seal(selection, seal)
        panel_connection = replay.sqlite3.connect(panel)
        panel_connection.row_factory = replay.sqlite3.Row
        try:
            evaluation = replay._evaluate(selection, panel_connection, selection_sha, expected, access, spec=spec)
            evaluation["protocol_metrics"] = _protocol_metrics(selection, panel, evaluation)
        finally:
            panel_connection.close()
    finally:
        source_connection.close()
    evaluation_path = chain / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from pipeline.r_validation.independent_audit import audit_selection

    audit_path = chain / "audit.json"
    audit = audit_selection(
        selection, panel, evaluation, audit_path, model=model, source_path=source_path,
        historical_inputs=historical_inputs,
    )
    if audit.get("status") != "pass":
        raise StageRunnerError("independent audit failed")

    result: dict[str, Any] = {
        "status": "pass",
        "profile": profile,
        "historical_inputs": history_receipt,
        "coverage": coverage,
        "stages": list(STAGES),
        "source_sha256": _sha(source_path),
        "panel_sha256": _sha(panel),
        "selection_sha256": selection_sha,
        "evaluation_sha256": _sha(evaluation_path),
        "audit_sha256": _sha(audit_path),
        "model_binding": {"model": "current_lr", "parameters": str(model_path.relative_to(output)), "sha256": _sha(model_path), "feature_count": len(model["features"])},
        "protocol": {
            "score_start": spec.get("score_start"),
            "score_end": spec.get("score_end"),
            "event_start": spec.get("event_start"),
            "event_end": spec.get("event_end"),
            "outcome_cutoff": spec.get("outcome_cutoff"),
            "horizon_days": spec.get("horizon_days"),
            "serials": list(spec.get("serials")) if spec.get("serials") is not None else list(source_serials),
        },
        "source_receipt": {
            "path": str(receipt_path.relative_to(output)),
            "sha256": _sha(receipt_path),
            "status": json.loads(receipt_path.read_text(encoding="utf-8")).get("status"),
        },
        "source_access": {
            "score_receives_full_source_rows": False,
            "score_query_mode": "as_of_sql_only",
            "reader_interface": "DecisionDayReader",
            "score_access_count": len([item for item in access if item.get("phase") == "score"]),
            "evaluation_after_seal": True,
        },
        "outcome_access_after_seal": True,
        "audit": audit,
        "independent_audit": audit,
    }

    if exercise_future_guards and profile == "synthetic":
        # Reuse the runner's actual parser and scorer with changed/deleted
        # future input copies; selection must remain byte-identical.
        rows, _ = replay._parse_source(source_path, strict=strict_fixture)
        changed = False
        perturbed = [dict(row) for row in rows]
        score_end = dt.date.fromisoformat(str(spec.get("score_end", replay.END.isoformat())))
        future_rows = [row for row in perturbed if dt.date.fromisoformat(str(row["date"])) > score_end]
        if future_rows:
            target = future_rows[0]
            target["smart_5_raw"] = int(target["smart_5_raw"] or 0) + 999
            changed = True
        if not changed:
            raise StageRunnerError("future perturbation target is absent")
        perturbed_path = chain / "daily_perturbed.csv"
        _write_csv(perturbed_path, perturbed)
        perturbed_panel = chain / "panel_perturbed.sqlite"
        _write_panel(perturbed_panel, perturbed)
        altered_selection = chain / "selection_perturbed.sqlite"
        altered_access: list[dict[str, Any]] = []
        altered_connection = replay._source_connection(perturbed, altered_access)
        altered_source = replay._asof_reader(altered_connection, altered_access)
        try:
            replay._score_and_select(altered_source, [], model, altered_selection, altered_access, spec=spec)
        finally:
            altered_connection.close()
        if _sha(altered_selection) != selection_sha:
            raise StageRunnerError("future value perturbation changed selection")
        future_rows = [row for row in rows if dt.date.fromisoformat(str(row["date"])) > score_end]
        if not future_rows:
            raise StageRunnerError("future deletion target is absent")
        delete_target = future_rows[-1]
        deleted = [row for row in rows if row is not delete_target]
        deleted_path = chain / "panel_future_row_deleted.sqlite"
        _write_panel(deleted_path, deleted)
        deleted_selection = chain / "selection_future_row_deleted.sqlite"
        deleted_access: list[dict[str, Any]] = []
        deleted_connection = replay._source_connection(deleted, deleted_access)
        deleted_source = replay._asof_reader(deleted_connection, deleted_access)
        try:
            replay._score_and_select(deleted_source, [], model, deleted_selection, deleted_access, spec=spec)
        finally:
            deleted_connection.close()
        if _sha(deleted_selection) != selection_sha:
            raise StageRunnerError("future row deletion changed selection")
        result["future_perturbation_preserved_selection"] = True
        result["future_deletion_preserved_selection"] = True
    (chain / "chain_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_stage(
    output: Path,
    *,
    root: Path,
    profile: str,
    source_csv: Path | None = None,
    model_json: Path | None = None,
    expected_json: Path | None = None,
    source_receipt: Path | None = None,
    source_object_id: str | None = None,
    spec: Mapping[str, Any] | None = None,
    release_verified: bool = False,
    exercise_future_guards: bool = False,
    approved_model_sha256: str | None = None,
    resource_limits: ResourceLimits | None = None,
    historical_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one R chain with the shared resource reader around every stage.

    R0 is deliberately in-process: the existing train/evaluate supervisor has
    a fixed training request and cannot be safely called with fake fit
    arguments.  This wrapper therefore reuses its ``ResourceGuard`` and
    records that limitation explicitly; the release-time R1 launcher still
    needs a process-group supervisor before real Q4 access.
    """
    output = Path(output).resolve()
    root = Path(root).resolve()
    if output.exists() and any(output.iterdir()):
        raise StageRunnerError(f"R output directory must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    limits = resource_limits or ResourceLimits(
        max_rss_bytes=3 * 1024 * 1024 * 1024,
        max_output_bytes=12 * 1024 * 1024 * 1024,
        initial_free_bytes=14 * 1024 * 1024 * 1024,
        min_free_bytes=2 * 1024 * 1024 * 1024,
        max_elapsed_seconds=3 * 60 * 60,
        poll_seconds=0.2,
    )
    guard = ResourceGuard(root, output, limits, extra_pids=(os.getpid(),))
    started = False
    try:
        guard.start()
        started = True
        try:
            guard.check_or_raise()
        except ResourceViolation as exc:
            raise StageRunnerError(f"resource guard: {exc}") from exc
        result = _run_stage(
            output,
            root=root,
            profile=profile,
            source_csv=source_csv,
            model_json=model_json,
            expected_json=expected_json,
            source_receipt=source_receipt,
            source_object_id=source_object_id,
            spec=spec,
            release_verified=release_verified,
            exercise_future_guards=exercise_future_guards,
            approved_model_sha256=approved_model_sha256,
            historical_inputs=historical_inputs,
        )
        try:
            guard.check_or_raise()
        except ResourceViolation as exc:
            raise StageRunnerError(f"resource guard: {exc}") from exc
        if guard.violation is not None:
            raise StageRunnerError(f"resource guard: {guard.violation}")
        result["resource"] = {
            "mode": "in_process_resource_guard",
            "process_group_supervised": False,
            "limits": {
                "max_rss_bytes": limits.max_rss_bytes,
                "max_output_bytes": limits.max_output_bytes,
                "min_free_bytes": limits.min_free_bytes,
                "max_elapsed_seconds": limits.max_elapsed_seconds,
                "poll_seconds": limits.poll_seconds,
            },
            "peak_snapshot": dict(guard.peak_snapshot),
            "last_snapshot": dict(guard.last_snapshot),
            "reader_facts": guard.reader_facts(),
            "sampling_scope": guard.sampling_scope(),
        }
        # Refresh the chain summary so the resource record is part of the
        # same artifact consumed by the acceptance binder.
        (output / "cli_chain/chain_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if started:
            guard.stop()
