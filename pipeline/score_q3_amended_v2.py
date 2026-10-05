"""Run the single, locked Stage45 Q3 amended scoring attempt.

This entry point is deliberately separate from the failed v1 attempt.  It
keeps the reviewed raw SMART declines, enforces an exact source whitelist and
typed values, writes scores and alert lists before opening the outcome tables,
and publishes only after the SQLite file has been closed and re-opened
read-only.  The frozen v1 input and code files are content-bound in the v2
configuration and are never modified by this module.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import score_q3_amended as legacy  # noqa: E402


MODEL = "ST4000DM000"
SCORE_START = dt.date(2023, 7, 1)
SCORE_END = dt.date(2023, 9, 23)
LABEL_CUTOFF = dt.date(2023, 9, 30)
SOURCE_START = dt.date(2023, 1, 1)
EVENT_START = dt.date(2023, 7, 8)
EVENT_END = dt.date(2023, 9, 24)
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
BUDGET_DENOMINATOR = 1000
COOLDOWN_DAYS = 7
LABEL_RUN_ID = "validation_q3_h7_v1"
LABEL_SPLIT = "validation"
LABEL_HORIZON = 7
METHODS = ("random", "smart_nonzero", "smart187", "age_lr", "current_lr", "history_lr")
LR_METHODS = ("age_lr", "current_lr", "history_lr")
SMART_FIELDS = (5, 9, 187, 188, 197, 198)
NONZERO_FIELDS = (5, 187, 188, 197, 198)
WINDOWS = (7, 14)
MONOTONIC_FIELDS = (5, 9, 187)
AUDIT_COLUMNS = (
    "decrease_seen_asof_t_5",
    "decrease_seen_asof_t_9",
    "decrease_seen_asof_t_187",
    "window_crosses_decrease_w7_5",
    "window_crosses_decrease_w7_9",
    "window_crosses_decrease_w7_187",
    "window_crosses_decrease_w14_5",
    "window_crosses_decrease_w14_9",
    "window_crosses_decrease_w14_187",
)
AUDIT_ANY_COLUMNS = ("decrease_seen_any", "window_crosses_decrease_w7_any", "window_crosses_decrease_w14_any")
SOURCE_COLUMNS = tuple(["date", "serial_number", "model", "failure"] + [f"smart_{field}_raw" for field in SMART_FIELDS])
SOURCE_COLUMN_SET = frozenset(SOURCE_COLUMNS)

Q3_DB = ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite"
PRIOR_DB = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
MODEL_DB = ROOT / "data/derived/simple_baseline_v2/model_results.sqlite"
ZIP_PATH = ROOT / "data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip"
CONFIG_PATH = ROOT / "configs/q3_scoring_amended_v2.json"
SOURCE_REVIEW = ROOT / "evidence/q3/source_review_v1/review.json"
ANOMALY_SOURCE = ROOT / "evidence/q3/source_review_v1/anomaly_source_check.json"
DIAGNOSIS = ROOT / "evidence/q3/smart_decision_v1/local_diagnosis.json"
STAGE45_REVIEW = ROOT / "reports/Q3_SCORING_STAGE45_REVIEW.md"
STAGE45_EVIDENCE = ROOT / "evidence/q3/scoring_review_v2/review.json"
OUTPUT_ROOT = ROOT / "data/derived/q3_scoring_amended_v2/attempt_001"
OUTPUT_DB = OUTPUT_ROOT / "q3_scoring_amended_v2.sqlite"
PARTIAL_DB = OUTPUT_ROOT / "q3_scoring_amended_v2.partial.sqlite"
EVIDENCE_ROOT = ROOT / "evidence/q3/scoring_amended_v2/attempt_001"
TMP_ROOT = ROOT / ".tmp/q3_scoring_amended_v2/attempt_001"
OLD_FAILED_ROOT = ROOT / "data/derived/q3_scoring_amended_v1"

EXPECTED_ELIGIBLE = 1_497_209
EXPECTED_EVENTS = 126
EXPECTED_OPPORTUNITIES = 126
MAX_RSS = 2 * 1024**3
MAX_NEW_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 2 * 1024**3
MAX_STAGE_SECONDS = 2 * 60 * 60
CHECK_ROWS = 10_000


class ScoringStopped(RuntimeError):
    """A frozen-input, leakage, resource, or QA boundary stopped the attempt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def owned_bytes() -> int:
    """Count the current attempt and the preserved failed v1 output."""
    roots = (OLD_FAILED_ROOT, OUTPUT_ROOT, EVIDENCE_ROOT, TMP_ROOT)
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def resource_snapshot(start: float) -> dict[str, int | float]:
    return {
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "rss_bytes": rss_bytes(),
        "owned_bytes": owned_bytes(),
        "free_bytes": shutil.disk_usage(ROOT).free,
    }


def check_resources(start: float, label: str) -> dict[str, int | float]:
    snapshot = resource_snapshot(start)
    if snapshot["rss_bytes"] >= MAX_RSS:
        raise ScoringStopped(f"RSS limit at {label}: {snapshot}")
    if snapshot["owned_bytes"] >= MAX_NEW_BYTES:
        raise ScoringStopped(f"project byte limit at {label}: {snapshot}")
    if snapshot["free_bytes"] < MIN_FREE_BYTES:
        raise ScoringStopped(f"free-space floor at {label}: {snapshot}")
    if snapshot["elapsed_seconds"] >= MAX_STAGE_SECONDS:
        raise ScoringStopped(f"stage timeout at {label}: {snapshot}")
    return snapshot


def connect_immutable(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ScoringStopped(f"missing immutable input: {path}")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ScoringStopped(f"non-empty sidecar for immutable input: {sidecar}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key,value FROM metadata")}


def load_config() -> dict:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringStopped(f"cannot read v2 config: {exc}") from exc
    if config.get("status") != "execution_locked" or config.get("attempt_id") != "attempt_001":
        raise ScoringStopped("v2 config is not the locked attempt_001 contract")
    for key in ("fit_allowed", "q4_access", "remote_setup", "model_adoption_allowed"):
        if config.get(key) is not False:
            raise ScoringStopped(f"v2 config unexpectedly permits {key}")
    if config.get("scoring_allowed_after_acceptance") is not True:
        raise ScoringStopped("v2 config does not permit scoring after acceptance")
    scope = config.get("scope", {})
    checks = {
        "model": MODEL,
        "history_days": HISTORY_DAYS,
        "minimum_history_observations": MIN_HISTORY,
        "horizon_days": HORIZON,
        "score_dates": [SCORE_START.isoformat(), SCORE_END.isoformat()],
        "source_start": SOURCE_START.isoformat(),
        "label_cutoff": LABEL_CUTOFF.isoformat(),
        "event_dates": [EVENT_START.isoformat(), EVENT_END.isoformat()],
        "label_run_id": LABEL_RUN_ID,
        "split": LABEL_SPLIT,
        "methods": list(METHODS),
        "budget_denominator": BUDGET_DENOMINATOR,
        "cooldown_days": COOLDOWN_DAYS,
        "tie_break_prefix": "drive-v1|20260912",
        "bootstrap_seed": 20260913,
        "bootstrap_replicates": 2000,
        "bootstrap_min_valid": 1900,
        "minimum_opportunity_events": 100,
    }
    for key, expected in checks.items():
        if scope.get(key) != expected:
            raise ScoringStopped(f"v2 scope mismatch for {key}: {scope.get(key)!r}")
    if config.get("expected_eligible_rows") != EXPECTED_ELIGIBLE:
        raise ScoringStopped("v2 expected eligible count differs")
    transitions = config.get("reviewed_transitions")
    if not isinstance(transitions, list) or len(transitions) != 3:
        raise ScoringStopped("v2 reviewed transition ledger must contain exactly three entries")
    return config


def verify_bindings(config: dict) -> dict[str, object]:
    bindings = config.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ScoringStopped("v2 config has no content bindings")
    observed: dict[str, str] = {}
    for relative, expected in bindings.items():
        path = ROOT / str(relative)
        if not path.is_file():
            raise ScoringStopped(f"bound file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ScoringStopped(f"bound file changed: {relative}")
        observed[str(relative)] = actual
    frozen_paths = config.get("frozen_paths")
    expected_paths = {
        "q3": "data/derived/q3_validation_v1/panel_q3.sqlite",
        "prior": "data/derived/panel_q1q2_verified_v1.sqlite",
        "model": "data/derived/simple_baseline_v2/model_results.sqlite",
        "zip": "data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip",
    }
    if frozen_paths != expected_paths:
        raise ScoringStopped("v2 frozen path map differs")
    frozen_observed: dict[str, str] = {}
    for key, relative in expected_paths.items():
        path = ROOT / relative
        if not path.is_file():
            raise ScoringStopped(f"frozen input is missing: {relative}")
        frozen_observed[key] = sha256_file(path)
        observed[relative] = frozen_observed[key]
    if config.get("frozen_sha256") != frozen_observed:
        raise ScoringStopped("v2 frozen input hash map differs")
    old_failed = config.get("old_failed_attempt", {})
    old_path = ROOT / str(old_failed.get("path", ""))
    if old_failed.get("publishable") is not False or not old_path.is_file() or sha256_file(old_path) != old_failed.get("sha256"):
        raise ScoringStopped("preserved v1 failed attempt changed or was made publishable")
    return {"bindings": observed, "config_sha256": sha256_file(CONFIG_PATH)}


def _deny_outcome_reads(action: int, arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
    if action == sqlite3.SQLITE_READ and arg1 in {"label_flow", "label_runs"}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def attach_prior(q3: sqlite3.Connection) -> None:
    q3.execute("ATTACH DATABASE ? AS prior", (PRIOR_DB.as_uri() + "?mode=ro&immutable=1",))


def _date_ord(value: str | dt.date) -> int:
    return value.toordinal() if isinstance(value, dt.date) else dt.date.fromisoformat(value).toordinal()


def _strict_date(value: object) -> str:
    if not isinstance(value, str):
        raise ScoringStopped("date must be a string")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ScoringStopped(f"invalid ISO date: {value!r}") from exc
    if parsed.isoformat() != value:
        raise ScoringStopped(f"date is not canonical ISO format: {value!r}")
    return value


def validate_source_row(row: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact daily source contract before any feature formula."""
    actual = set(row.keys())
    if actual != SOURCE_COLUMN_SET:
        extra = sorted(actual - SOURCE_COLUMN_SET)
        missing = sorted(SOURCE_COLUMN_SET - actual)
        raise ScoringStopped(f"daily whitelist mismatch; extra={extra}, missing={missing}")
    normalized = {key: row[key] for key in SOURCE_COLUMNS}
    normalized["date"] = _strict_date(normalized["date"])
    for key in ("serial_number", "model"):
        value = normalized[key]
        if not isinstance(value, str) or not value:
            raise ScoringStopped(f"{key} must be a non-empty string")
    if normalized["model"] != MODEL:
        raise ScoringStopped(f"unexpected model: {normalized['model']!r}")
    failure = normalized["failure"]
    if isinstance(failure, bool) or type(failure) is not int or failure not in (0, 1):
        raise ScoringStopped(f"failure must be an integer 0/1, got {failure!r}")
    normalized["failure"] = failure
    for field in SMART_FIELDS:
        key = f"smart_{field}_raw"
        value = normalized[key]
        if value is not None and (isinstance(value, bool) or type(value) is not int or value < 0):
            raise ScoringStopped(f"{key} must be NULL or a non-negative integer, got {value!r}")
    return normalized


def transition_key(serial: str, field: int, previous_date: str, previous: int, date: str, value: int) -> tuple[object, ...]:
    return (serial, f"smart_{field}_raw", previous_date, previous, date, value)


def ledger_map(config: Mapping[str, object]) -> dict[tuple[object, ...], dict[str, object]]:
    result: dict[tuple[object, ...], dict[str, object]] = {}
    for item in config["reviewed_transitions"]:  # type: ignore[index]
        full = {
            "serial": str(item["serial"]), "field": str(item["field"]),
            "previous_date": str(item["previous_date"]), "previous": int(item["previous"]),
            "date": str(item["date"]), "value": int(item["value"]),
        }
        field = int(full["field"].removeprefix("smart_").removesuffix("_raw"))
        key = transition_key(str(full["serial"]), field, str(full["previous_date"]), int(full["previous"]), str(full["date"]), int(full["value"]))
        result[key] = full
    return result


def validate_decline_transition(
    serial: str,
    field: int,
    previous: tuple[int, int] | None,
    current_date: str,
    current_value: int | None,
    expected: Mapping[tuple[object, ...], Mapping[str, object]],
    seen: set[tuple[object, ...]],
) -> dict[str, object] | None:
    if previous is None or current_value is None or current_value >= previous[1]:
        return None
    previous_date = dt.date.fromordinal(previous[0]).isoformat()
    key = transition_key(serial, field, previous_date, previous[1], current_date, current_value)
    item = expected.get(key)
    if item is None:
        raise ScoringStopped(f"unreviewed SMART decline: {key}")
    if key in seen:
        raise ScoringStopped(f"duplicate reviewed SMART decline: {key}")
    seen.add(key)
    return dict(item)


def _feature_schema_sql() -> str:
    definitions = [f'"{name}" {definition}' for name, definition in legacy.FEATURE_COLUMNS]
    definitions.extend(f'"{name}" INTEGER NOT NULL' for name in AUDIT_COLUMNS + AUDIT_ANY_COLUMNS)
    return ",\n            ".join(definitions)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE feature_rows(
            {_feature_schema_sql()},
            PRIMARY KEY(decision_date, serial_number)
        );
        CREATE INDEX feature_date ON feature_rows(decision_date);
        CREATE TABLE smart_audit(
            decision_date TEXT NOT NULL, serial_number TEXT NOT NULL,
            decrease_seen_asof_t_5 INTEGER NOT NULL, decrease_seen_asof_t_9 INTEGER NOT NULL,
            decrease_seen_asof_t_187 INTEGER NOT NULL,
            window_crosses_decrease_w7_5 INTEGER NOT NULL, window_crosses_decrease_w7_9 INTEGER NOT NULL,
            window_crosses_decrease_w7_187 INTEGER NOT NULL,
            window_crosses_decrease_w14_5 INTEGER NOT NULL, window_crosses_decrease_w14_9 INTEGER NOT NULL,
            window_crosses_decrease_w14_187 INTEGER NOT NULL,
            decrease_seen_any INTEGER NOT NULL, window_crosses_decrease_w7_any INTEGER NOT NULL,
            window_crosses_decrease_w14_any INTEGER NOT NULL, transition_count INTEGER NOT NULL,
            PRIMARY KEY(decision_date, serial_number)
        );
        CREATE TABLE smart_transitions(
            serial_number TEXT NOT NULL, field TEXT NOT NULL, previous_date TEXT NOT NULL,
            previous INTEGER NOT NULL, date TEXT NOT NULL, value INTEGER NOT NULL,
            reviewed INTEGER NOT NULL, PRIMARY KEY(serial_number,field,previous_date,date)
        );
        CREATE TABLE model_scores(
            decision_date TEXT NOT NULL, serial_number TEXT NOT NULL,
            tie_break_sha256 TEXT NOT NULL, age_lr_score REAL NOT NULL,
            current_lr_score REAL NOT NULL, history_lr_score REAL NOT NULL,
            history_observations_14 INTEGER NOT NULL, observed_days_7 INTEGER NOT NULL,
            current_missing_any INTEGER NOT NULL, decrease_seen_any INTEGER NOT NULL,
            window_crosses_decrease_w7_any INTEGER NOT NULL,
            window_crosses_decrease_w14_any INTEGER NOT NULL,
            smart_nonzero_signal_count INTEGER NOT NULL, smart_187_signal INTEGER NOT NULL,
            PRIMARY KEY(decision_date, serial_number)
        );
        CREATE INDEX score_date ON model_scores(decision_date);
        CREATE TABLE model_alerts(
            model TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL,
            score REAL NOT NULL, tie_break_sha256 TEXT NOT NULL, status TEXT NOT NULL,
            label INTEGER, first_failure_date TEXT, event_key TEXT, event_hit INTEGER NOT NULL,
            lead_days INTEGER, PRIMARY KEY(model, decision_date, serial_number)
        );
        CREATE TABLE model_daily(
            model TEXT NOT NULL, decision_date TEXT NOT NULL, eligible_count INTEGER NOT NULL,
            budget_k INTEGER NOT NULL, cooldown_excluded INTEGER NOT NULL, alerts_count INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL,
            unknown_alerts INTEGER NOT NULL, PRIMARY KEY(model, decision_date)
        );
        CREATE TABLE model_event_summary(
            model TEXT NOT NULL, event_key TEXT NOT NULL, first_failure_date TEXT NOT NULL,
            opportunity INTEGER NOT NULL, hit INTEGER NOT NULL, earliest_alert_date TEXT,
            earliest_lead_days INTEGER, PRIMARY KEY(model,event_key)
        );
        CREATE TABLE model_strata(
            model TEXT NOT NULL, stratum_type TEXT NOT NULL, stratum TEXT NOT NULL,
            eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL,
            unknown_alerts INTEGER NOT NULL, PRIMARY KEY(model,stratum_type,stratum)
        );
        CREATE TABLE week_summary(
            model TEXT NOT NULL, iso_year INTEGER NOT NULL, iso_week INTEGER NOT NULL,
            eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL,
            unknown_alerts INTEGER NOT NULL, PRIMARY KEY(model,iso_year,iso_week)
        );
        CREATE TABLE model_metrics(
            model TEXT PRIMARY KEY, eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL, unknown_alerts INTEGER NOT NULL,
            unknown_alert_ratio REAL, precision_lower_bound REAL, precision_upper_bound REAL,
            known_outcome_precision REAL, confirmed_no_hit_per_1000_device_days REAL NOT NULL,
            event_total INTEGER NOT NULL, event_opportunity_total INTEGER NOT NULL, event_hits INTEGER NOT NULL,
            event_recall_at_opportunity REAL, all_event_capture REAL,
            early_event_hits_ge2_days INTEGER NOT NULL, early_event_hits_ge3_days INTEGER NOT NULL,
            early_recall_at_opportunity_ge2_days REAL, early_recall_at_opportunity_ge3_days REAL,
            hit_only_ratio_ge2_days REAL, hit_only_ratio_ge3_days REAL,
            earliest_lead_count INTEGER NOT NULL, earliest_lead_median REAL,
            earliest_lead_q25 REAL, earliest_lead_q75 REAL,
            repeated_alert_devices INTEGER NOT NULL, max_alerts_per_device INTEGER NOT NULL,
            minimum_alert_gap_days INTEGER, average_precision_known REAL, metric_scope TEXT NOT NULL
        );
        CREATE TABLE resource_log(
            event TEXT NOT NULL, label TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
            rss_bytes INTEGER NOT NULL, owned_bytes INTEGER NOT NULL, free_bytes INTEGER NOT NULL
        );
        """
    )


def _feature_insert_sql() -> str:
    names = list(legacy.FEATURE_COLUMN_NAMES) + list(AUDIT_COLUMNS) + list(AUDIT_ANY_COLUMNS)
    return f'INSERT INTO feature_rows ({", ".join(names)}) VALUES ({", ".join("?" for _ in names)})'


def _daily_rows_for_date(connection: sqlite3.Connection, date_text: str) -> sqlite3.Cursor:
    fields = ",".join(SOURCE_COLUMNS)
    return connection.execute(
        f"SELECT {fields} FROM prior.daily WHERE date=? UNION ALL SELECT {fields} FROM main.daily WHERE date=? ORDER BY serial_number",
        (date_text, date_text),
    )


def _record_resource(connection: sqlite3.Connection, start: float, event: str, label: str) -> None:
    snapshot = resource_snapshot(start)
    connection.execute("INSERT INTO resource_log VALUES (?,?,?,?,?,?)", (event, label, snapshot["elapsed_seconds"], snapshot["rss_bytes"], snapshot["owned_bytes"], snapshot["free_bytes"]))


def build_features(source: sqlite3.Connection, output: sqlite3.Connection, config: Mapping[str, object], start: float) -> dict[str, object]:
    expected = ledger_map(config)
    seen_transitions: set[tuple[object, ...]] = set()
    transition_records: list[dict[str, object]] = []
    states: dict[str, dict[str, object]] = {}
    insert_sql = _feature_insert_sql()
    audit_sql = "INSERT INTO smart_audit VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    feature_batch: list[tuple[object, ...]] = []
    audit_batch: list[tuple[object, ...]] = []
    per_day: collections.Counter[str] = collections.Counter()
    count = 0
    total_days = (SCORE_END - SOURCE_START).days + 1
    for day_offset in range(total_days):
        current_date = SOURCE_START + dt.timedelta(days=day_offset)
        date_text = current_date.isoformat()
        decision_ord = current_date.toordinal()
        seen_today: set[str] = set()
        for source_row in _daily_rows_for_date(source, date_text):
            row = validate_source_row(dict(source_row))
            serial = str(row["serial_number"])
            if serial in seen_today:
                raise ScoringStopped(f"duplicate device/date row: {serial}/{date_text}")
            seen_today.add(serial)
            state = states.setdefault(
                serial,
                {
                    "last_ord": None,
                    "history": collections.deque(maxlen=HISTORY_DAYS),
                    "last_values": {},
                    "decrease_seen": {field: 0 for field in MONOTONIC_FIELDS},
                    "transitions": {field: [] for field in MONOTONIC_FIELDS},
                    "failed_seen": False,
                    "transition_count": 0,
                },
            )
            previous_ord = state["last_ord"]
            if previous_ord is not None and decision_ord <= int(previous_ord):
                raise ScoringStopped(f"device/date order violation: {serial}/{date_text}")
            state["last_ord"] = decision_ord
            last_values = state["last_values"]
            transitions = state["transitions"]
            decrease_seen = state["decrease_seen"]
            for field in MONOTONIC_FIELDS:
                value = row[f"smart_{field}_raw"]
                previous = last_values.get(field)
                record = validate_decline_transition(serial, field, previous, date_text, value, expected, seen_transitions)
                if record is not None:
                    transition_records.append(record)
                    state["transition_count"] = int(state["transition_count"]) + 1
                    decrease_seen[field] = 1
                    transitions[field].append((int(previous[0]), decision_ord))  # type: ignore[index]
                if value is not None:
                    last_values[field] = (decision_ord, int(value))
                    transitions[field][:] = [item for item in transitions[field] if item[1] >= decision_ord - HISTORY_DAYS + 1]
            if bool(state["failed_seen"]):
                continue
            if int(row["failure"]) == 1:
                state["failed_seen"] = True
                continue
            history: collections.deque[dict[str, object]] = state["history"]
            history.append(row)
            if not SCORE_START <= current_date <= SCORE_END:
                continue
            result = legacy.feature_row_asof(list(history), date_text)
            if result is None:
                continue
            values, _legacy_audit = result
            audit: dict[str, int] = {}
            for field in MONOTONIC_FIELDS:
                audit[f"decrease_seen_asof_t_{field}"] = int(decrease_seen[field])
                for window in WINDOWS:
                    audit[f"window_crosses_decrease_w{window}_{field}"] = int(any(decision_ord - window + 1 <= begin <= end <= decision_ord for begin, end in transitions[field]))
            values["decrease_seen_any"] = int(any(audit[f"decrease_seen_asof_t_{field}"] for field in MONOTONIC_FIELDS))
            values["window_crosses_decrease_w7_any"] = int(any(audit[f"window_crosses_decrease_w7_{field}"] for field in MONOTONIC_FIELDS))
            values["window_crosses_decrease_w14_any"] = int(any(audit[f"window_crosses_decrease_w14_{field}"] for field in MONOTONIC_FIELDS))
            feature_batch.append(tuple(values.get(name) for name in legacy.FEATURE_COLUMN_NAMES) + tuple(audit[name] for name in AUDIT_COLUMNS) + tuple(int(values[name]) for name in AUDIT_ANY_COLUMNS))
            audit_batch.append((date_text, serial, *(audit[name] for name in AUDIT_COLUMNS), *(int(values[name]) for name in AUDIT_ANY_COLUMNS), int(state["transition_count"])))
            count += 1
            per_day[date_text] += 1
            if len(feature_batch) >= 5000:
                output.executemany(insert_sql, feature_batch)
                output.executemany(audit_sql, audit_batch)
                output.commit()
                feature_batch.clear(); audit_batch.clear()
                if count % CHECK_ROWS < 5000:
                    _record_resource(output, start, "feature_build", f"rows={count}")
                    output.commit()
                    check_resources(start, f"feature rows={count}")
        if day_offset % 7 == 0:
            check_resources(start, f"feature day={date_text}")
    if feature_batch:
        output.executemany(insert_sql, feature_batch)
        output.executemany(audit_sql, audit_batch)
        output.commit()
    if count != EXPECTED_ELIGIBLE:
        raise ScoringStopped(f"Q3 feature count {count} != expected {EXPECTED_ELIGIBLE}")
    if seen_transitions != set(expected):
        raise ScoringStopped(f"reviewed transition ledger mismatch: seen={len(seen_transitions)} expected={len(expected)}")
    output.executemany(
        "INSERT INTO smart_transitions VALUES (?,?,?,?,?,?,?)",
        [(r["serial"], r["field"], r["previous_date"], r["previous"], r["date"], r["value"], 1) for r in transition_records],
    )
    output.execute("INSERT OR REPLACE INTO metadata VALUES ('audit_status','complete')")
    output.execute("INSERT OR REPLACE INTO metadata VALUES ('audit_transition_count',?)", (str(len(transition_records)),))
    output.commit()
    return {
        "status": "complete", "feature_rows": count,
        "day_counts": [{"date": date, "eligible_count": value} for date, value in sorted(per_day.items())],
        "source_whitelist": list(SOURCE_COLUMNS), "source_types": "strict_sqlite_integer_or_null",
        "reviewed_transitions": transition_records, "future_outcome_tables_read": False,
    }


def _score_matrix(rows: Sequence[sqlite3.Row], payloads: Mapping[str, Mapping[str, object]]) -> dict[str, np.ndarray]:
    return legacy._score_matrix(rows, payloads)


def score_features(connection: sqlite3.Connection, model_connection: sqlite3.Connection, start: float) -> dict[str, object]:
    payloads = legacy._model_payloads(model_connection)
    insert_sql = "INSERT INTO model_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    total = 0
    day_counts: list[dict[str, object]] = []
    dates = [str(row[0]) for row in connection.execute("SELECT DISTINCT decision_date FROM feature_rows ORDER BY decision_date")]
    expected_dates = (SCORE_END - SCORE_START).days + 1
    if len(dates) != expected_dates:
        raise ScoringStopped("Q3 feature dates are incomplete")
    for date_text in dates:
        rows = connection.execute("SELECT * FROM feature_rows WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
        scores = _score_matrix(rows, payloads)
        records = []
        for index, row in enumerate(rows):
            current_missing = int(any(int(row[f"smart_{field}_current_missing"]) for field in SMART_FIELDS))
            records.append((date_text, str(row["serial_number"]), str(row["tie_break_sha256"]), float(scores["age_lr"][index]), float(scores["current_lr"][index]), float(scores["history_lr"][index]), int(row["history_observations_14"]), int(row["observed_days_7"]), current_missing, int(row["decrease_seen_any"]), int(row["window_crosses_decrease_w7_any"]), int(row["window_crosses_decrease_w14_any"]), int(row["smart_nonzero_signal_count"]), int(row["smart_187_signal"])))
        connection.executemany(insert_sql, records)
        connection.commit()
        total += len(records)
        day_counts.append({"date": date_text, "eligible_count": len(records)})
        if total % CHECK_ROWS < len(records):
            _record_resource(connection, start, "score_build", f"rows={total}")
            connection.commit()
            check_resources(start, f"score rows={total}")
    if total != EXPECTED_ELIGIBLE:
        raise ScoringStopped(f"Q3 score count {total} != expected {EXPECTED_ELIGIBLE}")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('score_status','complete')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('score_rows',?)", (str(total),))
    connection.commit()
    return {"status": "complete", "score_rows": total, "day_counts": day_counts, "models": list(LR_METHODS), "outcome_tables_read": False}


def _rank_rows(rows: Sequence[sqlite3.Row], indices: Sequence[int], scores: np.ndarray) -> list[int]:
    return sorted(indices, key=lambda index: (-float(scores[index]), str(rows[index]["tie_break_sha256"]), str(rows[index]["serial_number"])))


def select_alert_indices_v2(rows: Sequence[sqlite3.Row], scores: np.ndarray, method: str, decision_date: dt.date, last_alert: dict[str, dt.date]) -> tuple[list[int], int, int]:
    if method == "smart_nonzero":
        candidates = [index for index, row in enumerate(rows) if int(row["smart_nonzero_signal_count"]) > 0]
    elif method == "smart187":
        candidates = [index for index, row in enumerate(rows) if int(row["smart_187_signal"]) > 0]
    else:
        candidates = list(range(len(rows)))
    ranked = _rank_rows(rows, candidates, scores)
    cooldown_excluded = 0
    available: list[int] = []
    for index in ranked:
        serial = str(rows[index]["serial_number"])
        previous = last_alert.get(serial)
        if previous is not None and (decision_date - previous).days <= COOLDOWN_DAYS:
            cooldown_excluded += 1
        else:
            available.append(index)
    budget = (len(rows) + BUDGET_DENOMINATOR - 1) // BUDGET_DENOMINATOR if rows else 0
    selected = available[:budget]
    for index in selected:
        last_alert[str(rows[index]["serial_number"])] = decision_date
    return selected, cooldown_excluded, len(candidates)


def persist_selections(connection: sqlite3.Connection, start: float) -> dict[str, object]:
    if metadata(connection).get("score_status") != "complete":
        raise ScoringStopped("score pass is not complete before selection")
    dates = [str(row[0]) for row in connection.execute("SELECT DISTINCT decision_date FROM model_scores ORDER BY decision_date")]
    selection_summary: dict[str, object] = {"status": "complete", "outcome_tables_read": False, "models": {}}
    for method in METHODS:
        last_alert: dict[str, dt.date] = {}
        pending: list[tuple[object, ...]] = []
        daily: list[tuple[object, ...]] = []
        selected_keys: list[dict[str, object]] = []
        total_alerts = 0
        for date_text in dates:
            rows = connection.execute("SELECT * FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
            if method == "random": scores = np.zeros(len(rows), dtype=float)
            elif method == "smart_nonzero": scores = np.asarray([float(row["smart_nonzero_signal_count"]) for row in rows])
            elif method == "smart187": scores = np.asarray([float(row["smart_187_signal"]) for row in rows])
            else: scores = np.asarray([float(row[f"{method}_score"]) for row in rows])
            selected, cooldown_excluded, candidate_count = select_alert_indices_v2(rows, scores, method, dt.date.fromisoformat(date_text), last_alert)
            budget = (len(rows) + BUDGET_DENOMINATOR - 1) // BUDGET_DENOMINATOR if rows else 0
            daily.append((method, date_text, len(rows), budget, cooldown_excluded, len(selected), 0, 0, 0))
            for index in selected:
                row = rows[index]
                serial = str(row["serial_number"])
                score = float(scores[index])
                tie = str(row["tie_break_sha256"])
                pending.append((method, date_text, serial, score, tie, "pending_evaluation", None, None, None, 0, None))
                selected_keys.append({"date": date_text, "serial": serial, "score": score, "tie_break_sha256": tie})
            total_alerts += len(selected)
        connection.executemany("INSERT INTO model_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?)", pending)
        connection.executemany("INSERT INTO model_daily VALUES (?,?,?,?,?,?,?,?,?)", daily)
        connection.commit()
        content_hash = canonical_hash(selected_keys)
        selection_summary["models"][method] = {"dates": len(dates), "alerts": total_alerts, "content_sha256": content_hash, "candidate_filter": method in {"smart_nonzero", "smart187"}, "pending_status": "pending_evaluation"}
        connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (f"selection_{method}_sha256", content_hash))
        check_resources(start, f"selection {method}")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('selection_status','complete')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('selection_alert_rows',?)", (str(connection.execute("SELECT COUNT(*) FROM model_alerts").fetchone()[0]),))
    connection.commit()
    return selection_summary


def event_opportunities(panel: sqlite3.Connection) -> dict[str, dict[str, object]]:
    # The attached prior schema is read-only and contains only daily rows here.
    fields = ",".join(SOURCE_COLUMNS)
    cursor = panel.execute(f"SELECT {fields} FROM prior.daily WHERE date <= ? UNION ALL SELECT {fields} FROM main.daily WHERE date <= ? ORDER BY serial_number,date", (LABEL_CUTOFF.isoformat(), LABEL_CUTOFF.isoformat()))
    result: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[sqlite3.Row]] = collections.defaultdict(list)
    for row in cursor:
        grouped[str(row["serial_number"])].append(row)
    for serial, rows in grouped.items():
        dates = {_date_ord(str(row["date"])) for row in rows}
        failures = sorted(_date_ord(str(row["date"])) for row in rows if int(row["failure"]) == 1)
        if not failures:
            continue
        first = dt.date.fromordinal(failures[0])
        if not EVENT_START <= first <= EVENT_END:
            continue
        candidates: list[str] = []
        for offset in range(1, HORIZON + 1):
            decision = first - dt.timedelta(days=offset)
            if not SCORE_START <= decision <= SCORE_END or decision.toordinal() not in dates:
                continue
            history = sum(decision.toordinal() - i in dates for i in range(HISTORY_DAYS))
            if history >= MIN_HISTORY:
                candidates.append(decision.isoformat())
        result[serial] = {"event_key": serial, "first_failure_date": first.isoformat(), "opportunity": int(bool(candidates)), "opportunity_dates": candidates}
    return result


def _metric_add(strata: dict[tuple[str, str], dict[str, int]], typ: str, value: str, key: str, amount: int = 1) -> None:
    item = strata.setdefault((typ, value), {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0})
    item[key] += amount


def evaluate_scores(connection: sqlite3.Connection, panel: sqlite3.Connection, start: float) -> dict[str, object]:
    current_meta = metadata(connection)
    if current_meta.get("score_status") != "complete" or current_meta.get("selection_status") != "complete":
        raise ScoringStopped("evaluation opened before scores and alert lists were persisted")
    events = event_opportunities(panel)
    opportunity_total = sum(int(item["opportunity"]) for item in events.values())
    if (len(events), opportunity_total) != (EXPECTED_EVENTS, EXPECTED_OPPORTUNITIES):
        raise ScoringStopped(f"Q3 event denominator differs: {len(events)}/{opportunity_total}")
    dates = [str(row[0]) for row in connection.execute("SELECT DISTINCT decision_date FROM model_scores ORDER BY decision_date")]
    known_scores: dict[str, list[float]] = {method: [] for method in LR_METHODS}
    known_labels: list[int] = []
    labels_by_date: dict[str, dict[str, sqlite3.Row]] = {}
    for date_text in dates:
        label_rows = panel.execute("SELECT serial_number,status,label,first_failure_date FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? AND eligible=1", (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON, date_text)).fetchall()
        labels = {str(row["serial_number"]): row for row in label_rows}
        score_rows = connection.execute("SELECT * FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
        if set(labels) != {str(row["serial_number"]) for row in score_rows}:
            raise ScoringStopped(f"label key set differs on {date_text}")
        labels_by_date[date_text] = labels
        for row in score_rows:
            label = labels[str(row["serial_number"])] ["label"]
            if label is None:
                continue
            for method in LR_METHODS:
                known_scores[method].append(float(row[f"{method}_score"]))
            known_labels.append(int(label))
    metrics: dict[str, dict[str, object]] = {}
    event_hits_by_model: dict[str, dict[str, int]] = {}
    serials = sorted({str(row[0]) for row in connection.execute("SELECT DISTINCT serial_number FROM model_scores")}, key=lambda value: value.encode("utf-8"))
    for method in METHODS:
        strata: dict[tuple[str, str], dict[str, int]] = {}
        week_data: dict[tuple[int, int], dict[str, int]] = {}
        event_alerts: dict[str, list[tuple[dt.date, int]]] = collections.defaultdict(list)
        alert_dates: dict[str, list[dt.date]] = collections.defaultdict(list)
        last_leads: list[int] = []
        event_hit_map: dict[str, int] = {}
        known_hits = known_no_hit = unknown = alerts = eligible_total = 0
        for date_text in dates:
            score_rows = connection.execute("SELECT * FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
            selected = connection.execute("SELECT rowid,* FROM model_alerts WHERE model=? AND decision_date=? ORDER BY serial_number", (method, date_text)).fetchall()
            labels = labels_by_date[date_text]
            alert_key_set = {str(row["serial_number"]) for row in selected}
            if not alert_key_set.issubset(labels):
                raise ScoringStopped(f"selected alert key absent from label set on {method}/{date_text}")
            day_known_hit = day_known_no_hit = day_unknown = 0
            for row in score_rows:
                values = (
                    ("month", date_text[:7]),
                    ("current_missing", str(int(row["current_missing_any"]))),
                    ("history_observations_14", str(int(row["history_observations_14"]))),
                    ("decrease_seen_asof_t", str(int(row["decrease_seen_any"]))),
                    ("window_crosses_decrease_w7", str(int(row["window_crosses_decrease_w7_any"]))),
                    ("window_crosses_decrease_w14", str(int(row["window_crosses_decrease_w14_any"]))),
                )
                for typ, value in values:
                    _metric_add(strata, typ, value, "eligible")
                iso = dt.date.fromisoformat(date_text).isocalendar()
                item = week_data.setdefault((iso.year, iso.week), {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0})
                item["eligible"] += 1
            for alert in selected:
                serial = str(alert["serial_number"])
                label_row = labels[serial]
                label = label_row["label"]
                failure_date = label_row["first_failure_date"]
                if label == 1:
                    known_hits += 1; day_known_hit += 1
                elif label == 0:
                    known_no_hit += 1; day_known_no_hit += 1
                else:
                    unknown += 1; day_unknown += 1
                event_key = None; event_hit = 0; lead_days = None
                info = events.get(serial)
                if info and failure_date == info["first_failure_date"] and date_text in info["opportunity_dates"]:
                    event_key = serial
                    failure = dt.date.fromisoformat(str(failure_date))
                    lead_days = (failure - dt.date.fromisoformat(date_text)).days
                    event_hit = 1
                    event_alerts[serial].append((dt.date.fromisoformat(date_text), int(lead_days)))
                    last_leads.append(int(lead_days))
                connection.execute("UPDATE model_alerts SET status=?,label=?,first_failure_date=?,event_key=?,event_hit=?,lead_days=? WHERE model=? AND decision_date=? AND serial_number=?", (str(label_row["status"]), None if label is None else int(label), failure_date, event_key, event_hit, lead_days, method, date_text, serial))
                alert_date = dt.date.fromisoformat(date_text)
                alert_dates[serial].append(alert_date)
                for typ, value in (("month", date_text[:7]), ("current_missing", str(int(alert["serial_number"] in labels and connection.execute("SELECT current_missing_any FROM model_scores WHERE decision_date=? AND serial_number=?", (date_text, serial)).fetchone()[0]))),):
                    _metric_add(strata, typ, value, "alerts")
                    if label == 1: _metric_add(strata, typ, value, "known_hit")
                    elif label == 0: _metric_add(strata, typ, value, "known_no_hit")
                    else: _metric_add(strata, typ, value, "unknown")
                score_row = connection.execute("SELECT * FROM model_scores WHERE decision_date=? AND serial_number=?", (date_text, serial)).fetchone()
                for typ, value in (("history_observations_14", str(int(score_row["history_observations_14"]))), ("decrease_seen_asof_t", str(int(score_row["decrease_seen_any"]))), ("window_crosses_decrease_w7", str(int(score_row["window_crosses_decrease_w7_any"]))), ("window_crosses_decrease_w14", str(int(score_row["window_crosses_decrease_w14_any"])) )):
                    _metric_add(strata, typ, value, "alerts")
                    if label == 1: _metric_add(strata, typ, value, "known_hit")
                    elif label == 0: _metric_add(strata, typ, value, "known_no_hit")
                    else: _metric_add(strata, typ, value, "unknown")
                iso = alert_date.isocalendar()
                week_item = week_data.setdefault((iso.year, iso.week), {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0})
                week_item["alerts"] += 1
                if label == 1: week_item["known_hit"] += 1
                elif label == 0: week_item["known_no_hit"] += 1
                else: week_item["unknown"] += 1
            eligible_total += len(score_rows)
            alerts += len(selected); day_counts = (day_known_hit, day_known_no_hit, day_unknown)
            connection.execute("UPDATE model_daily SET known_hit_alerts=?,known_no_hit_alerts=?,unknown_alerts=? WHERE model=? AND decision_date=?", (*day_counts, method, date_text))
        connection.commit()
        event_rows = []
        early2 = early3 = 0
        for serial, info in events.items():
            choices = sorted(event_alerts.get(serial, []))
            hit = int(bool(choices) and bool(info["opportunity"]))
            event_hit_map[serial] = hit
            earliest_date = choices[0][0].isoformat() if choices else None
            earliest_lead = choices[0][1] if choices else None
            if hit:
                early2 += int(int(earliest_lead) >= 2)
                early3 += int(int(earliest_lead) >= 3)
            event_rows.append((method, serial, info["first_failure_date"], int(info["opportunity"]), hit, earliest_date, earliest_lead))
        connection.executemany("INSERT INTO model_event_summary VALUES (?,?,?,?,?,?,?)", event_rows)
        for (typ, value), item in strata.items():
            connection.execute("INSERT INTO model_strata VALUES (?,?,?,?,?,?,?,?)", (method, typ, value, item["eligible"], item["alerts"], item["known_hit"], item["known_no_hit"], item["unknown"]))
        for (iso_year, iso_week), item in week_data.items():
            connection.execute("INSERT INTO week_summary VALUES (?,?,?,?,?,?,?,?)", (method, iso_year, iso_week, item["eligible"], item["alerts"], item["known_hit"], item["known_no_hit"], item["unknown"]))
            connection.execute("INSERT INTO model_strata VALUES (?,?,?,?,?,?,?,?)", (method, "iso_week", f"{iso_year}-W{iso_week:02d}", item["eligible"], item["alerts"], item["known_hit"], item["known_no_hit"], item["unknown"]))
        ap = legacy.average_precision_grouped(known_scores[method], known_labels) if method in LR_METHODS else None
        leads = sorted(last_leads)
        gaps = [min((right - left).days for left, right in zip(sorted(days), sorted(days)[1:])) for days in alert_dates.values() if len(days) >= 2]
        event_hits = sum(event_hit_map.values())
        summary: dict[str, object] = {
            "model": method, "eligible_device_days": eligible_total, "alerts": alerts,
            "known_hit_alerts": known_hits, "known_no_hit_alerts": known_no_hit, "unknown_alerts": unknown,
            "unknown_alert_ratio": unknown / alerts if alerts else None,
            "precision_lower_bound": known_hits / alerts if alerts else None,
            "precision_upper_bound": (known_hits + unknown) / alerts if alerts else None,
            "known_outcome_precision": known_hits / (known_hits + known_no_hit) if known_hits + known_no_hit else None,
            "confirmed_no_hit_per_1000_device_days": known_no_hit / eligible_total * 1000 if eligible_total else 0.0,
            "event_total": len(events), "event_opportunity_total": opportunity_total, "event_hits": event_hits,
            "event_recall_at_opportunity": event_hits / opportunity_total if opportunity_total else None,
            "all_event_capture": event_hits / len(events) if events else None,
            "early_event_hits_ge2_days": early2, "early_event_hits_ge3_days": early3,
            "early_recall_at_opportunity_ge2_days": early2 / opportunity_total if opportunity_total else None,
            "early_recall_at_opportunity_ge3_days": early3 / opportunity_total if opportunity_total else None,
            "hit_only_ratio_ge2_days": early2 / event_hits if event_hits else None,
            "hit_only_ratio_ge3_days": early3 / event_hits if event_hits else None,
            "earliest_lead_count": len(leads), "earliest_lead_median": legacy.linear_quantile(leads, 0.5),
            "earliest_lead_q25": legacy.linear_quantile(leads, 0.25), "earliest_lead_q75": legacy.linear_quantile(leads, 0.75),
            "repeated_alert_devices": sum(len(days) > 1 for days in alert_dates.values()),
            "max_alerts_per_device": max((len(days) for days in alert_dates.values()), default=0),
            "minimum_alert_gap_days": min(gaps) if gaps else None, "average_precision_known": ap,
            "metric_scope": "Q3 amended E2-R v2; unknown outcomes excluded from AP; precision bounds are unknown-outcome bounds",
        }
        metric_values = (
            method, summary["eligible_device_days"], summary["alerts"], summary["known_hit_alerts"], summary["known_no_hit_alerts"], summary["unknown_alerts"], summary["unknown_alert_ratio"], summary["precision_lower_bound"], summary["precision_upper_bound"], summary["known_outcome_precision"], summary["confirmed_no_hit_per_1000_device_days"], summary["event_total"], summary["event_opportunity_total"], summary["event_hits"], summary["event_recall_at_opportunity"], summary["all_event_capture"], summary["early_event_hits_ge2_days"], summary["early_event_hits_ge3_days"], summary["early_recall_at_opportunity_ge2_days"], summary["early_recall_at_opportunity_ge3_days"], summary["hit_only_ratio_ge2_days"], summary["hit_only_ratio_ge3_days"], summary["earliest_lead_count"], summary["earliest_lead_median"], summary["earliest_lead_q25"], summary["earliest_lead_q75"], summary["repeated_alert_devices"], summary["max_alerts_per_device"], summary["minimum_alert_gap_days"], summary["average_precision_known"], summary["metric_scope"],
        )
        connection.execute("INSERT INTO model_metrics VALUES (" + ",".join("?" for _ in metric_values) + ")", metric_values)
        connection.commit()
        metrics[method] = summary
        event_hits_by_model[method] = event_hit_map
        check_resources(start, f"evaluation {method}")
    opportunities = {serial: int(events.get(serial, {}).get("opportunity", 0)) for serial in serials}
    bootstrap = legacy.paired_device_bootstrap(serials, opportunities, event_hits_by_model["current_lr"], event_hits_by_model["history_lr"])
    recall_gain = float(metrics["history_lr"]["event_recall_at_opportunity"] - metrics["current_lr"]["event_recall_at_opportunity"])
    unknown_change = float(metrics["history_lr"]["unknown_alert_ratio"] - metrics["current_lr"]["unknown_alert_ratio"])
    numeric_gate = {
        "recall_gain_at_least_0_05": recall_gain >= 0.05,
        "bootstrap_valid_at_least_1900": bootstrap["valid_replicates"] >= 1900,
        "bootstrap_lower_strictly_positive": bootstrap["quantile_025"] is not None and bootstrap["quantile_025"] > 0,
        "unknown_alert_ratio_increase_at_most_0_02": unknown_change <= 0.02,
        "opportunity_at_least_100": opportunity_total >= 100,
    }
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('evaluation_status','complete')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('outcome_access_during_score_pass','false')")
    connection.commit()
    return {"status": "complete", "metrics": metrics, "bootstrap_history_minus_current": bootstrap, "comparison": {"recall_gain": recall_gain, "unknown_alert_ratio_change": unknown_change, "numeric_gate": numeric_gate, "model_adoption": "pending_review" if all(numeric_gate.values()) else "retain_current_simple_control"}, "events": {"event_total": len(events), "opportunity_total": opportunity_total}}


def write_model_payload_evidence(model_connection: sqlite3.Connection, output: Path) -> None:
    payload: dict[str, object] = {"model_db_sha256": sha256_file(MODEL_DB), "models": {}}
    for model in LR_METHODS:
        row = model_connection.execute("SELECT model,status,feature_count,training_rows,positive_training_rows,negative_training_rows,n_iter,intercept,params_json FROM model_runs WHERE model=?", (model,)).fetchone()
        payload["models"][model] = dict(row)
    atomic_json(output, payload)


def verify_output_database(path: Path) -> dict[str, object]:
    connection = connect_immutable(path)
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick != "ok":
            raise ScoringStopped(f"published output quick_check={quick}")
        meta = metadata(connection)
        if meta.get("run_status") != "complete" or meta.get("evaluation_status") != "complete":
            raise ScoringStopped("published output metadata is incomplete")
        counts = {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in ("feature_rows", "model_scores", "model_alerts", "model_daily", "model_event_summary", "model_metrics", "smart_transitions")}
        return {"quick_check": quick, "metadata": meta, "counts": counts}
    finally:
        connection.close()


def verify_complete_reentry() -> dict[str, object]:
    manifest_path = EVIDENCE_ROOT / "q3_scoring_amended_v2_complete_manifest_v1.json"
    if not OUTPUT_DB.is_file() or not manifest_path.is_file():
        raise ScoringStopped("complete v2 output or manifest is missing")
    config = load_config()
    binding_facts = verify_bindings(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_hash") != canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise ScoringStopped("v2 manifest self-hash differs")
    if sha256_file(OUTPUT_DB) != manifest.get("database_sha256"):
        raise ScoringStopped("v2 output database changed")
    for key, value in manifest.items():
        if key.endswith("_file_sha256"):
            path = ROOT / str(manifest.get(key.removesuffix("_file_sha256"), ""))
            if not path.is_file() or sha256_file(path) != value:
                raise ScoringStopped(f"v2 evidence changed: {path}")
    if manifest.get("input_bindings") != binding_facts["bindings"]:
        raise ScoringStopped("v2 frozen input binding map differs")
    database_meta = manifest.get("database_verification", {}).get("metadata", {})
    for key, relative in {
        "source_q3_sha256": "data/derived/q3_validation_v1/panel_q3.sqlite",
        "source_prior_sha256": "data/derived/panel_q1q2_verified_v1.sqlite",
        "model_db_sha256": "data/derived/simple_baseline_v2/model_results.sqlite",
        "zip_sha256": "data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip",
    }.items():
        if database_meta.get(key) != binding_facts["bindings"][relative]:
            raise ScoringStopped(f"v2 database provenance differs for {key}")
    if database_meta.get("config_sha256") != binding_facts["config_sha256"]:
        raise ScoringStopped("v2 database config provenance differs")
    facts = verify_output_database(OUTPUT_DB)
    return {"status": "already_complete", "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": sha256_file(OUTPUT_DB), "counts": facts["counts"]}


def publish_output(connection: sqlite3.Connection, start: float, binding_facts: Mapping[str, object], evidence_paths: Mapping[str, Path], fixture_path: Path) -> dict[str, object]:
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('run_status','complete')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('completed_at_utc',?)", (dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),))
    connection.commit()
    connection.close()
    facts = verify_output_database(PARTIAL_DB)
    os.replace(PARTIAL_DB, OUTPUT_DB)
    database_sha = sha256_file(OUTPUT_DB)
    manifest: dict[str, object] = {
        "manifest_version": "q3-scoring-amended-v2-complete-v1", "status": "complete", "protocol": "E2-R", "attempt_id": "attempt_001",
        "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": database_sha,
        "scoring_code": str(Path(__file__).relative_to(ROOT)), "scoring_code_file_sha256": sha256_file(Path(__file__)),
        "config": str(CONFIG_PATH.relative_to(ROOT)), "config_file_sha256": sha256_file(CONFIG_PATH),
        "fixture_acceptance": str(fixture_path.relative_to(ROOT)), "fixture_acceptance_file_sha256": sha256_file(fixture_path),
        "training_approval": False, "fit_allowed": False, "q4_access": False, "remote_setup": False,
        "model_adoption": "pending_review", "published_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "input_bindings": binding_facts["bindings"], "database_verification": facts,
    }
    for name, path in evidence_paths.items():
        manifest[name] = str(path.relative_to(ROOT))
        manifest[f"{name}_file_sha256"] = sha256_file(path)
    manifest["source_review"] = str(SOURCE_REVIEW.relative_to(ROOT)); manifest["source_review_file_sha256"] = sha256_file(SOURCE_REVIEW)
    manifest["anomaly_source"] = str(ANOMALY_SOURCE.relative_to(ROOT)); manifest["anomaly_source_file_sha256"] = sha256_file(ANOMALY_SOURCE)
    manifest["diagnosis"] = str(DIAGNOSIS.relative_to(ROOT)); manifest["diagnosis_file_sha256"] = sha256_file(DIAGNOSIS)
    manifest["stage45_review"] = str(STAGE45_REVIEW.relative_to(ROOT)); manifest["stage45_review_file_sha256"] = sha256_file(STAGE45_REVIEW)
    manifest["stage45_evidence"] = str(STAGE45_EVIDENCE.relative_to(ROOT)); manifest["stage45_evidence_file_sha256"] = sha256_file(STAGE45_EVIDENCE)
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(EVIDENCE_ROOT / "q3_scoring_amended_v2_complete_manifest_v1.json", manifest)
    return {"status": "complete", "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": database_sha, "database_counts": facts["counts"], "elapsed_seconds": round(time.monotonic() - start, 3)}


def run(fixture_path: Path | None = None) -> dict[str, object]:
    if OUTPUT_DB.exists():
        return verify_complete_reentry()
    if PARTIAL_DB.exists():
        raise ScoringStopped(f"preserving existing v2 failed attempt: {PARTIAL_DB}")
    config = load_config()
    binding_facts = verify_bindings(config)
    if fixture_path is None:
        fixture_path = EVIDENCE_ROOT / "fixture_acceptance_v1.json"
    if not fixture_path.is_file():
        raise ScoringStopped(f"fixture acceptance evidence is required before real run: {fixture_path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True); EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True); TMP_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    q3 = connect_immutable(Q3_DB)
    model_connection = connect_immutable(MODEL_DB)
    connection: sqlite3.Connection | None = None
    try:
        q3_meta = metadata(q3); model_meta = metadata(model_connection)
        if q3_meta.get("panel_status") != "complete" or q3_meta.get("training_approval") != "false":
            raise ScoringStopped("Q3 panel metadata changed")
        if model_meta.get("run_status") != "complete" or model_meta.get("training_approval") != "false":
            raise ScoringStopped("frozen model metadata changed")
        attach_prior(q3)
        q3.set_authorizer(_deny_outcome_reads)
        connection = sqlite3.connect(PARTIAL_DB)
        connection.row_factory = sqlite3.Row
        create_schema(connection)
        metadata_values = {
            "run_status": "running", "protocol": "E2-R", "attempt_id": "attempt_001", "config_sha256": binding_facts["config_sha256"], "scoring_code_sha256": sha256_file(Path(__file__)),
            "source_q3_sha256": binding_facts["bindings"]["data/derived/q3_validation_v1/panel_q3.sqlite"], "source_prior_sha256": binding_facts["bindings"]["data/derived/panel_q1q2_verified_v1.sqlite"], "model_db_sha256": binding_facts["bindings"]["data/derived/simple_baseline_v2/model_results.sqlite"], "zip_sha256": binding_facts["bindings"]["data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip"],
            "feature_dictionary_hash": legacy.feature_dictionary_hash(), "model": MODEL, "score_start": SCORE_START.isoformat(), "score_end": SCORE_END.isoformat(), "source_start": SOURCE_START.isoformat(), "label_cutoff": LABEL_CUTOFF.isoformat(), "horizon_days": str(HORIZON), "fit_allowed": "false", "q4_access": "false", "remote_setup": "false", "training_approval": "false", "outcome_access_during_score_pass": "false", "feature_access_mode": "one_day_whitelist_with_bounded_state", "decrease_policy": "retain_raw_audit_asof_t_and_reviewed_ledger",
        }
        connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata_values.items()); connection.commit()
        feature_result = build_features(q3, connection, config, start); atomic_json(EVIDENCE_ROOT / "features_v2.json", feature_result)
        score_result = score_features(connection, model_connection, start); atomic_json(EVIDENCE_ROOT / "scores_v2.json", score_result)
        selection_result = persist_selections(connection, start); atomic_json(EVIDENCE_ROOT / "selection_v2.json", selection_result)
        # Outcome access begins only after selection_status is durable.
        q3.set_authorizer(None)
        evaluation_result = evaluate_scores(connection, q3, start); atomic_json(EVIDENCE_ROOT / "evaluation_v2.json", evaluation_result)
        write_model_payload_evidence(model_connection, EVIDENCE_ROOT / "model_payload_v2.json")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('run_status','ready_to_publish')"); connection.commit()
        evidence_paths = {"features": EVIDENCE_ROOT / "features_v2.json", "scores": EVIDENCE_ROOT / "scores_v2.json", "selection": EVIDENCE_ROOT / "selection_v2.json", "evaluation": EVIDENCE_ROOT / "evaluation_v2.json", "model_payload": EVIDENCE_ROOT / "model_payload_v2.json"}
        return publish_output(connection, start, binding_facts, evidence_paths, fixture_path)
    except BaseException as exc:
        if connection is not None:
            try:
                connection.execute("INSERT OR REPLACE INTO metadata VALUES ('run_status','failed')"); connection.execute("INSERT OR REPLACE INTO metadata VALUES ('failure',?)", (f"{type(exc).__name__}: {exc}",)); connection.commit(); connection.close()
            except Exception:
                try: connection.close()
                except Exception: pass
        atomic_json(EVIDENCE_ROOT / "failed_attempt_v2.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "partial_database": str(PARTIAL_DB.relative_to(ROOT)), "scoring_code": str(Path(__file__).relative_to(ROOT)), "scoring_code_sha256": sha256_file(Path(__file__)), "elapsed_seconds": round(time.monotonic() - start, 3)})
        raise
    finally:
        if connection is not None:
            connection.close()
        q3.close(); model_connection.close()


def probe() -> dict[str, object]:
    base = {"date": "2023-07-01", "serial_number": "fixture", "model": MODEL, "failure": 0}
    rows = []
    for offset in range(15):
        row = dict(base); row["date"] = (SCORE_START + dt.timedelta(days=offset)).isoformat()
        for field in SMART_FIELDS: row[f"smart_{field}_raw"] = 10 + offset
        rows.append(row)
    try:
        validate_source_row({**rows[0], "first_failure_date": None})
    except ScoringStopped:
        extra_rejected = True
    else:
        extra_rejected = False
    try:
        validate_source_row({**rows[0], "failure": 0.5})
    except ScoringStopped:
        fractional_rejected = True
    else:
        fractional_rejected = False
    expected = {transition_key("fixture", 5, "2023-07-10", 20, "2023-07-11", 2): {}}
    seen: set[tuple[object, ...]] = set()
    try:
        validate_decline_transition("fixture", 5, (dt.date(2023, 7, 10).toordinal(), 20), "2023-07-11", 2, expected, seen)
        ledger_accept = True
    except ScoringStopped:
        ledger_accept = False
    try:
        validate_decline_transition("fixture", 5, (dt.date(2023, 7, 10).toordinal(), 20), "2023-07-11", 3, expected, set())
    except ScoringStopped:
        unknown_decline_rejected = True
    else:
        unknown_decline_rejected = False
    return {"status": "pass" if extra_rejected and fractional_rejected and ledger_accept and unknown_decline_rejected else "fail", "extra_column_rejected": extra_rejected, "fractional_failure_rejected": fractional_rejected, "reviewed_decline_accepted": ledger_accept, "unknown_decline_rejected": unknown_decline_rejected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    try:
        result = probe() if args.probe else run(args.fixture)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ScoringStopped, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "stopped", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
