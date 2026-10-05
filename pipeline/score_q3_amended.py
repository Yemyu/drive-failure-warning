"""Score the fixed models on Q3 under the amended E2-R protocol.

The input databases, the retained Q3 archive, and the v2 model payload are
immutable.  This entry point writes one new attempt database.  Feature and
score creation never reads Q3 outcomes; the evaluation pass starts only after
all scores have been committed.  The three source-confirmed SMART declines
are retained and represented by audit flags, never corrected or filtered.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import itertools
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

from pipeline.build_feature_replay import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_COLUMN_NAMES,
    NONZERO_FIELDS,
    RANDOM_SALT,
    SMART_FIELDS,
    WINDOWS,
    feature_dictionary_hash,
)


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
MONOTONIC_FIELDS = (5, 9, 187)

Q3_DB = ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite"
PRIOR_DB = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
MODEL_DB = ROOT / "data/derived/simple_baseline_v2/model_results.sqlite"
ZIP_PATH = ROOT / "data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip"
CONFIG_PATH = ROOT / "configs/q3_scoring_amended_v1.json"
SOURCE_REVIEW = ROOT / "evidence/q3/source_review_v1/review.json"
ANOMALY_SOURCE = ROOT / "evidence/q3/source_review_v1/anomaly_source_check.json"
DIAGNOSIS = ROOT / "evidence/q3/smart_decision_v1/local_diagnosis.json"
OUTPUT_ROOT = ROOT / "data/derived/q3_scoring_amended_v1"
OUTPUT_DB = OUTPUT_ROOT / "q3_scoring_amended_v1.sqlite"
EVIDENCE_ROOT = ROOT / "evidence/q3/scoring_amended_v1"
PARTIAL_DB = OUTPUT_ROOT / "q3_scoring_amended_v1.partial.sqlite"

MAX_RSS = 2 * 1024**3
MAX_NEW_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 2 * 1024**3
MAX_STAGE_SECONDS = 2 * 60 * 60
CHECK_ROWS = 10_000
SQL_PROGRESS_OPS = 20_000
EXPECTED_ELIGIBLE = 1_497_209
EXPECTED_EVENTS = 126
EXPECTED_OPPORTUNITIES = 126
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


class ScoringStopped(RuntimeError):
    """A frozen-input, leakage, resource, or QA boundary stopped the attempt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def owned_bytes() -> int:
    total = 0
    for base in (OUTPUT_ROOT, EVIDENCE_ROOT):
        if not base.exists():
            continue
        for path in base.rglob("*"):
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
        raise ScoringStopped(f"new project byte limit at {label}: {snapshot}")
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
        value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringStopped(f"cannot read amended config: {exc}") from exc
    if value.get("status") != "design_locked_not_executed":
        raise ScoringStopped("amended config must remain design_locked_not_executed")
    if value.get("fit_allowed") or value.get("q4_access") or value.get("remote_setup"):
        raise ScoringStopped("amended config unexpectedly permits fit, Q4, or remote")
    if value.get("scoring_allowed_after_acceptance") is not True:
        raise ScoringStopped("amended config does not declare conditional scoring permission")
    if value.get("model_adoption_allowed") is not False:
        raise ScoringStopped("amended config must forbid automatic model adoption")
    if value.get("scope", {}).get("horizon_days") != HORIZON:
        raise ScoringStopped("amended config horizon differs")
    if value.get("expected_eligible_rows") != EXPECTED_ELIGIBLE:
        raise ScoringStopped("amended config expected eligible count differs")
    return value


def verify_bindings(config: dict) -> dict[str, object]:
    bindings = config.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ScoringStopped("amended config has no file bindings")
    observed: dict[str, str] = {}
    for relative, expected in bindings.items():
        path = ROOT / str(relative)
        if not path.is_file():
            raise ScoringStopped(f"bound file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ScoringStopped(f"bound file changed: {relative}")
        observed[str(relative)] = actual
    expected_paths = {
        "q3": "data/derived/q3_validation_v1/panel_q3.sqlite",
        "prior": "data/derived/panel_q1q2_verified_v1.sqlite",
        "model": "data/derived/simple_baseline_v2/model_results.sqlite",
        "zip": "data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip",
    }
    if config.get("frozen_paths") != expected_paths:
        raise ScoringStopped("amended frozen path map differs from locked scope")
    frozen_observed: dict[str, str] = {}
    for key, relative in expected_paths.items():
        path = ROOT / relative
        if not path.is_file():
            raise ScoringStopped(f"frozen input is missing: {relative}")
        frozen_observed[key] = sha256_file(path)
        # Keep one canonical observed map for callers that record provenance.
        observed[relative] = frozen_observed[key]
    if config.get("frozen_sha256") != frozen_observed:
        raise ScoringStopped("frozen input hash map differs from bound files")
    return {"bindings": observed, "config_sha256": sha256_file(CONFIG_PATH)}


def _date_ord(value: str | dt.date) -> int:
    return value.toordinal() if isinstance(value, dt.date) else dt.date.fromisoformat(value).toordinal()


def _transform_delta(delta: int | float) -> float:
    if delta == 0:
        return 0.0
    return math.copysign(math.log1p(abs(delta)), delta)


def _smart_value(value: object, field: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ScoringStopped(f"SMART{field} boolean value")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ScoringStopped(f"SMART{field} is not an integer") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != number):
        raise ScoringStopped(f"SMART{field} is not a finite integer")
    if number < 0:
        raise ScoringStopped(f"SMART{field} is negative")
    return number


def _validate_prefix(rows: Sequence[Mapping[str, object]], decision_date: str) -> None:
    decision_ord = _date_ord(decision_date)
    previous: int | None = None
    serial: str | None = None
    model: str | None = None
    seen: set[int] = set()
    for row in rows:
        current_ord = _date_ord(str(row["date"]))
        if current_ord > decision_ord:
            raise ScoringStopped(f"future row supplied to feature function: {row['date']} > {decision_date}")
        if previous is not None and current_ord <= previous:
            raise ScoringStopped("rows for one device must be strictly date ordered")
        previous = current_ord
        if current_ord in seen:
            raise ScoringStopped("duplicate device/date row")
        seen.add(current_ord)
        current_serial = str(row["serial_number"])
        current_model = str(row["model"])
        if serial is None:
            serial, model = current_serial, current_model
        elif (current_serial, current_model) != (serial, model):
            raise ScoringStopped("feature prefix mixes serials or models")
        if current_model != MODEL:
            raise ScoringStopped(f"feature model differs from locked model: {current_model}")
        if int(row["failure"] or 0) not in (0, 1):
            raise ScoringStopped("failure must be 0 or 1")
        for field in SMART_FIELDS:
            _smart_value(row.get(f"smart_{field}_raw"), field)


def feature_row_asof(rows: Sequence[Mapping[str, object]], decision_date: str) -> tuple[dict[str, object], dict[str, int]] | None:
    """Build an eligible feature row from a prefix ending at ``decision_date``.

    Unlike the original training feature function this version permits signed
    decreases in SMART 5/9/187.  It still rejects malformed values and any
    future row, and it derives audit flags only from observations present by
    the decision date.
    """
    _validate_prefix(rows, decision_date)
    if not rows:
        return None
    decision_ord = _date_ord(decision_date)
    by_ord = {_date_ord(str(row["date"])): row for row in rows}
    current = by_ord.get(decision_ord)
    if current is None or int(current["failure"] or 0):
        return None
    if any(int(row["failure"] or 0) for row in rows):
        return None
    history = [by_ord[ordinal] for ordinal in range(decision_ord - HISTORY_DAYS + 1, decision_ord + 1) if ordinal in by_ord]
    if len(history) < MIN_HISTORY:
        return None
    values: dict[str, object] = {
        "decision_date": decision_date,
        "serial_number": str(current["serial_number"]),
        "model": str(current["model"]),
        "history_observations_14": len(history),
        "observed_days_7": sum(1 for ordinal in range(decision_ord - 6, decision_ord + 1) if ordinal in by_ord),
        "observed_days_14": len(history),
    }
    for field_number in SMART_FIELDS:
        source = f"smart_{field_number}_raw"
        prefix = f"smart_{field_number}"
        current_value = _smart_value(current.get(source), field_number)
        missing = int(current_value is None)
        if field_number == 188:
            values[f"{prefix}_current_nonzero"] = None if missing else int(current_value > 0)
            values[f"{prefix}_current_missing"] = missing
        else:
            values[f"{prefix}_current_log1p"] = math.log1p(current_value) if current_value is not None else None
            values[f"{prefix}_current_missing"] = missing
            if field_number != 9:
                values[f"{prefix}_current_nonzero"] = None if missing else int(current_value > 0)
        for window in WINDOWS:
            valid: list[tuple[int, int]] = []
            for ordinal in range(decision_ord - window + 1, decision_ord + 1):
                row = by_ord.get(ordinal)
                if row is None:
                    continue
                value = _smart_value(row.get(source), field_number)
                if value is not None:
                    valid.append((ordinal, value))
            if field_number == 188:
                indicators = [int(value > 0) for _, value in valid]
                values[f"{prefix}_w{window}_mean_nonzero"] = sum(indicators) / len(indicators) if indicators else None
                values[f"{prefix}_w{window}_missing_days"] = window - len(valid)
                if not missing and len(valid) >= 2:
                    values[f"{prefix}_w{window}_delta_nonzero"] = int(current_value > 0) - indicators[0]
                    values[f"{prefix}_w{window}_delta_span_days"] = decision_ord - valid[0][0]
                else:
                    values[f"{prefix}_w{window}_delta_nonzero"] = None
                    values[f"{prefix}_w{window}_delta_span_days"] = None
            else:
                logs = [math.log1p(value) for _, value in valid]
                values[f"{prefix}_w{window}_mean_log1p"] = sum(logs) / len(logs) if logs else None
                values[f"{prefix}_w{window}_max_log1p"] = max(logs) if logs else None
                values[f"{prefix}_w{window}_missing_days"] = window - len(valid)
                if not missing and len(valid) >= 2:
                    values[f"{prefix}_w{window}_delta_log1p"] = _transform_delta(current_value - valid[0][1])
                    values[f"{prefix}_w{window}_delta_span_days"] = decision_ord - valid[0][0]
                else:
                    values[f"{prefix}_w{window}_delta_log1p"] = None
                    values[f"{prefix}_w{window}_delta_span_days"] = None
    values["smart_nonzero_signal_count"] = sum(
        int(current.get(f"smart_{field}_raw") is not None and int(current[f"smart_{field}_raw"]) > 0)
        for field in NONZERO_FIELDS
    )
    values["smart_187_signal"] = int(current.get("smart_187_raw") is not None and int(current["smart_187_raw"]) > 0)
    values["tie_break_sha256"] = hashlib.sha256(f"{RANDOM_SALT}|{decision_date}|{current['serial_number']}".encode()).hexdigest()

    last: dict[int, tuple[int, int]] = {}
    transitions: dict[int, list[tuple[int, int]]] = {field: [] for field in MONOTONIC_FIELDS}
    for row in rows:
        ordinal = _date_ord(str(row["date"]))
        for field in MONOTONIC_FIELDS:
            value = _smart_value(row.get(f"smart_{field}_raw"), field)
            if value is None:
                continue
            previous = last.get(field)
            if previous is not None and value < previous[1]:
                transitions[field].append((previous[0], ordinal))
            last[field] = (ordinal, value)
    audit: dict[str, int] = {}
    for field in MONOTONIC_FIELDS:
        field_transitions = transitions[field]
        audit[f"decrease_seen_asof_t_{field}"] = int(any(end <= decision_ord for _start, end in field_transitions))
        audit[f"window_crosses_decrease_w7_{field}"] = int(any(decision_ord - 6 <= begin and end <= decision_ord for begin, end in field_transitions))
        audit[f"window_crosses_decrease_w14_{field}"] = int(any(decision_ord - 13 <= begin and end <= decision_ord for begin, end in field_transitions))
    values["decrease_seen_any"] = int(any(audit[f"decrease_seen_asof_t_{field}"] for field in MONOTONIC_FIELDS))
    values["window_crosses_decrease_w7_any"] = int(any(audit[f"window_crosses_decrease_w7_{field}"] for field in MONOTONIC_FIELDS))
    values["window_crosses_decrease_w14_any"] = int(any(audit[f"window_crosses_decrease_w14_{field}"] for field in MONOTONIC_FIELDS))
    return values, audit


def feature_schema_sql() -> str:
    definitions = [f'"{name}" {definition}' for name, definition in FEATURE_COLUMNS]
    definitions.extend(f'"{name}" INTEGER NOT NULL' for name in AUDIT_COLUMNS + AUDIT_ANY_COLUMNS)
    return ",\n        ".join(definitions)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE feature_rows(
            {feature_schema_sql()},
            PRIMARY KEY(decision_date, serial_number)
        );
        CREATE INDEX feature_date ON feature_rows(decision_date);
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
            unknown_alerts INTEGER NOT NULL,
            PRIMARY KEY(model, decision_date)
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
    names = list(FEATURE_COLUMN_NAMES) + list(AUDIT_COLUMNS) + list(AUDIT_ANY_COLUMNS)
    return f'INSERT INTO feature_rows ({", ".join(names)}) VALUES ({", ".join("?" for _ in names)})'


def _combined_daily(connection: sqlite3.Connection) -> sqlite3.Cursor:
    fields = ",".join(["date", "serial_number", "model", "failure"] + [f"smart_{field}_raw" for field in SMART_FIELDS])
    return connection.execute(
        f"SELECT {fields} FROM prior.daily WHERE date <= ? UNION ALL SELECT {fields} FROM main.daily WHERE date <= ? ORDER BY serial_number,date",
        ("2023-06-30", LABEL_CUTOFF.isoformat()),
    )


def _daily_rows_for_date(connection: sqlite3.Connection, date_text: str) -> sqlite3.Cursor:
    """Read one as-of day from the two immutable panel databases.

    The feature pass deliberately does not issue a ``date <= t`` query or
    materialize a whole serial's quarter.  Each iteration can therefore see
    only the whitelist fields for the current day, while the compact state in
    :func:`build_features` supplies the already observed history.
    """
    fields = ",".join(["date", "serial_number", "model", "failure"] + [f"smart_{field}_raw" for field in SMART_FIELDS])
    return connection.execute(
        f"SELECT {fields} FROM prior.daily WHERE date=? UNION ALL SELECT {fields} FROM main.daily WHERE date=? ORDER BY serial_number",
        (date_text, date_text),
    )


def _attach_prior(q3: sqlite3.Connection) -> None:
    q3.execute("ATTACH DATABASE ? AS prior", (PRIOR_DB.as_uri() + "?mode=ro&immutable=1",))


def build_features(source_connection: sqlite3.Connection, output_connection: sqlite3.Connection, start: float) -> tuple[int, list[dict[str, int | float]]]:
    """Build as-of features from the immutable source into the new output DB.

    Keeping the source and destination connections explicit prevents an easy
    provenance mistake: the cursor may read only the attached immutable panel,
    while every write (including resource checkpoints) must go to the partial
    output database.
    """
    insert_sql = _feature_insert_sql()
    batch: list[tuple[object, ...]] = []
    count = 0
    per_day: collections.Counter[str] = collections.Counter()
    # The prior panel begins on 2023-01-01.  Scanning it day by day is needed
    # for the cumulative as-of decline flag; retaining only the last 14 rows
    # would miss a pre-window transition that still has to be reported.
    history_start = SOURCE_START
    # State is bounded per serial: recent rows support feature windows, while
    # cumulative flags and recent transition pairs support the amended audit.
    states: dict[str, dict[str, object]] = {}
    total_days = (SCORE_END - history_start).days + 1
    for day_offset in range(total_days):
        current_date = history_start + dt.timedelta(days=day_offset)
        date_text = current_date.isoformat()
        decision_ord = current_date.toordinal()
        seen_today: set[str] = set()
        for source_row in _daily_rows_for_date(source_connection, date_text):
            row = dict(source_row)
            serial_value = row.get("serial_number")
            if serial_value is None:
                raise ScoringStopped(f"NULL serial_number on {date_text}")
            serial = str(serial_value)
            if serial in seen_today:
                raise ScoringStopped(f"duplicate device/date row: {serial}/{date_text}")
            seen_today.add(serial)
            if str(row.get("model")) != MODEL:
                raise ScoringStopped(f"feature model differs from locked model: {row.get('model')}")
            try:
                failure = int(row.get("failure") or 0)
            except (TypeError, ValueError) as exc:
                raise ScoringStopped(f"invalid failure value on {serial}/{date_text}") from exc
            if failure not in (0, 1):
                raise ScoringStopped(f"failure must be 0 or 1 on {serial}/{date_text}")
            for field in SMART_FIELDS:
                _smart_value(row.get(f"smart_{field}_raw"), field)
            state = states.setdefault(
                serial,
                {
                    "last_ord": None,
                    "history": collections.deque(maxlen=HISTORY_DAYS),
                    "last_values": {},
                    "decrease_seen": {field: 0 for field in MONOTONIC_FIELDS},
                    "transitions": {field: [] for field in MONOTONIC_FIELDS},
                    "failed_seen": False,
                },
            )
            previous_ord = state["last_ord"]
            if previous_ord is not None and decision_ord <= int(previous_ord):
                raise ScoringStopped(f"device/date order violation: {serial}/{date_text}")
            state["last_ord"] = decision_ord
            if bool(state["failed_seen"]):
                continue
            if failure:
                state["failed_seen"] = True
                continue
            history = state["history"]
            history.append(row)
            last_values = state["last_values"]
            transitions = state["transitions"]
            decrease_seen = state["decrease_seen"]
            for field in MONOTONIC_FIELDS:
                value = _smart_value(row.get(f"smart_{field}_raw"), field)
                if value is None:
                    continue
                previous = last_values.get(field)
                if previous is not None and value < int(previous[1]):
                    decrease_seen[field] = 1
                    transitions[field].append((int(previous[0]), decision_ord))
                last_values[field] = (decision_ord, value)
                transitions[field][:] = [item for item in transitions[field] if item[1] >= decision_ord - HISTORY_DAYS + 1]
            if current_date < SCORE_START or current_date > SCORE_END:
                continue
            result = feature_row_asof(list(history), date_text)
            if result is None:
                continue
            values, audit = result
            for field in MONOTONIC_FIELDS:
                audit[f"decrease_seen_asof_t_{field}"] = int(decrease_seen[field])
                for window in WINDOWS:
                    audit[f"window_crosses_decrease_w{window}_{field}"] = int(
                        any(decision_ord - window + 1 <= begin <= end <= decision_ord for begin, end in transitions[field])
                    )
            values["decrease_seen_any"] = int(any(audit[f"decrease_seen_asof_t_{field}"] for field in MONOTONIC_FIELDS))
            values["window_crosses_decrease_w7_any"] = int(any(audit[f"window_crosses_decrease_w7_{field}"] for field in MONOTONIC_FIELDS))
            values["window_crosses_decrease_w14_any"] = int(any(audit[f"window_crosses_decrease_w14_{field}"] for field in MONOTONIC_FIELDS))
            batch.append(
                tuple(values.get(name) for name in FEATURE_COLUMN_NAMES)
                + tuple(audit[name] for name in AUDIT_COLUMNS)
                + tuple(int(values[name]) for name in AUDIT_ANY_COLUMNS)
            )
            count += 1
            per_day[date_text] += 1
            if len(batch) >= 5_000:
                output_connection.executemany(insert_sql, batch)
                output_connection.commit()
                batch.clear()
                if count % CHECK_ROWS < 5_000:
                    snapshot = resource_snapshot(start)
                    output_connection.execute("INSERT INTO resource_log VALUES (?,?,?,?,?,?)", ("feature_build", f"rows={count}", snapshot["elapsed_seconds"], snapshot["rss_bytes"], snapshot["owned_bytes"], snapshot["free_bytes"]))
                    output_connection.commit()
                    check_resources(start, f"feature build rows={count}")
        if day_offset % 7 == 0:
            check_resources(start, f"feature day {date_text}")
    if batch:
        output_connection.executemany(insert_sql, batch)
        output_connection.commit()
    if count != EXPECTED_ELIGIBLE:
        raise ScoringStopped(f"Q3 feature count {count} != expected {EXPECTED_ELIGIBLE}")
    return count, [{"date": date, "eligible_count": value} for date, value in sorted(per_day.items())]


def _model_payloads(model_connection: sqlite3.Connection) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    expected_dims = {"age_lr": 2, "current_lr": 16, "history_lr": 122}
    for model in LR_METHODS:
        mapping = list(model_connection.execute("SELECT * FROM model_feature_map WHERE model=? ORDER BY position", (model,)))
        coefficients = {str(row["feature_name"]): float(row["coefficient"]) for row in model_connection.execute("SELECT * FROM model_coefficients WHERE model=?", (model,))}
        prep = {str(row["feature_name"]): dict(row) for row in model_connection.execute("SELECT * FROM preprocessing_stats WHERE model=?", (model,))}
        run = model_connection.execute("SELECT * FROM model_runs WHERE model=?", (model,)).fetchone()
        if run is None or run["status"] not in {"reused_v1", "fitted"}:
            raise ScoringStopped(f"frozen model run missing: {model}")
        if len(mapping) != expected_dims[model] or len(coefficients) != len(mapping) or len(prep) != len(mapping):
            raise ScoringStopped(f"frozen model payload dimension mismatch: {model}")
        names = [str(row["feature_name"]) for row in mapping]
        if set(names) != set(coefficients) or set(names) != set(prep):
            raise ScoringStopped(f"frozen model payload names mismatch: {model}")
        if not all(math.isfinite(coefficients[name]) for name in names):
            raise ScoringStopped(f"non-finite frozen coefficient: {model}")
        for name in names:
            if not all(math.isfinite(float(prep[name][key])) for key in ("imputation_mean", "standardization_mean", "standardization_scale")):
                raise ScoringStopped(f"non-finite frozen preprocessing value: {model}/{name}")
            if float(prep[name]["standardization_scale"]) <= 0:
                raise ScoringStopped(f"invalid frozen preprocessing scale: {model}/{name}")
        result[model] = {"mapping": mapping, "names": names, "coefficients": np.asarray([coefficients[name] for name in names]), "prep": prep, "intercept": float(run["intercept"])}
    return result


def _mapped_value(row: sqlite3.Row, mapping: sqlite3.Row) -> float:
    name = str(mapping["feature_name"])
    source = str(mapping["source_name"])
    if int(mapping["is_missing_indicator"]):
        if name == source:
            value = row[source]
            return float(value) if value is not None else 1.0
        return float(row[source] is None)
    value = row[source]
    return float(value) if value is not None else float("nan")


def _score_matrix(rows: Sequence[sqlite3.Row], payloads: Mapping[str, Mapping[str, object]]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for model, payload in payloads.items():
        mapping = payload["mapping"]
        matrix = np.asarray([[_mapped_value(row, mapping_item) for mapping_item in mapping] for row in rows], dtype=np.float64)
        names = payload["names"]
        means = np.asarray([float(payload["prep"][name]["imputation_mean"]) for name in names])
        centers = np.asarray([float(payload["prep"][name]["standardization_mean"]) for name in names])
        scales = np.asarray([float(payload["prep"][name]["standardization_scale"]) for name in names])
        transformed = (np.where(np.isfinite(matrix), matrix, means) - centers) / scales
        score = transformed @ payload["coefficients"] + float(payload["intercept"])
        if not np.isfinite(score).all():
            raise ScoringStopped(f"non-finite Q3 score for {model}")
        scores[model] = score
    scores["random"] = np.zeros(len(rows), dtype=np.float64)
    scores["smart_nonzero"] = np.asarray([float(row["smart_nonzero_signal_count"]) for row in rows])
    scores["smart187"] = np.asarray([float(row["smart_187_signal"]) for row in rows])
    return scores


def score_features(connection: sqlite3.Connection, model_connection: sqlite3.Connection, start: float) -> dict[str, object]:
    payloads = _model_payloads(model_connection)
    insert_sql = "INSERT INTO model_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    total = 0
    day_counts: list[dict[str, int | str]] = []
    dates = [str(row[0]) for row in connection.execute("SELECT DISTINCT decision_date FROM feature_rows ORDER BY decision_date")]
    if len(dates) != (SCORE_END - SCORE_START).days + 1:
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
            connection.execute("INSERT INTO resource_log VALUES (?,?,?,?,?,?)", ("score_build", f"rows={total}", resource_snapshot(start)["elapsed_seconds"], rss_bytes(), owned_bytes(), shutil.disk_usage(ROOT).free))
            connection.commit()
            check_resources(start, f"score build rows={total}")
    if total != EXPECTED_ELIGIBLE:
        raise ScoringStopped(f"Q3 score count {total} != expected {EXPECTED_ELIGIBLE}")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('score_status','complete')")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('score_rows',?)", (str(total),))
    connection.commit()
    return {"status": "complete", "score_rows": total, "day_counts": day_counts, "models": list(LR_METHODS)}


def event_opportunities(connection: sqlite3.Connection) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    cursor = _combined_daily(connection)
    for serial, group in itertools.groupby(cursor, key=lambda row: str(row["serial_number"])):
        rows = [dict(row) for row in group]
        dates = {_date_ord(str(row["date"])) for row in rows}
        failures = sorted(_date_ord(str(row["date"])) for row in rows if int(row["failure"] or 0))
        if not failures:
            continue
        first = failures[0]
        first_date = dt.date.fromordinal(first)
        if not EVENT_START <= first_date <= EVENT_END:
            continue
        candidates = []
        for offset in range(1, HORIZON + 1):
            decision = first - offset
            if not SCORE_START.toordinal() <= decision <= SCORE_END.toordinal() or decision not in dates:
                continue
            history = sum(decision - i in dates for i in range(HISTORY_DAYS))
            if history >= MIN_HISTORY:
                candidates.append(dt.date.fromordinal(decision).isoformat())
        result[serial] = {"event_key": serial, "first_failure_date": first_date.isoformat(), "opportunity": int(bool(candidates)), "opportunity_dates": candidates}
    return result


def _rank_indices(rows: Sequence[sqlite3.Row], scores: np.ndarray) -> list[int]:
    return sorted(range(len(rows)), key=lambda index: (-float(scores[index]), str(rows[index]["tie_break_sha256"]), str(rows[index]["serial_number"])))


def select_alert_indices(rows: Sequence[sqlite3.Row], scores: np.ndarray, decision_date: dt.date, last_alert: dict[str, dt.date]) -> tuple[list[int], int]:
    ranked = _rank_indices(rows, scores)
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
    return selected, cooldown_excluded


def average_precision_grouped(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels):
        raise ScoringStopped("AP inputs differ in length")
    positives = int(sum(labels))
    if positives == 0:
        return None
    pairs = sorted(zip((float(x) for x in scores), (int(y) for y in labels)), key=lambda item: -item[0])
    ap = 0.0
    seen = true = position = 0
    while position < len(pairs):
        end = position + 1
        while end < len(pairs) and pairs[end][0] == pairs[position][0]:
            end += 1
        group_positive = sum(label for _score, label in pairs[position:end])
        seen += end - position
        true += group_positive
        ap += (true / seen) * (group_positive / positives)
        position = end
    return float(ap)


def linear_quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_device_bootstrap(serials: Sequence[str], opportunities: Mapping[str, int], current_hits: Mapping[str, int], history_hits: Mapping[str, int], *, seed: int = 20260913, replicates: int = 2000) -> dict[str, object]:
    ordered = sorted({str(serial) for serial in serials}, key=lambda value: value.encode("utf-8"))
    if not ordered:
        raise ScoringStopped("bootstrap has no eligible devices")
    opportunity = np.asarray([int(opportunities.get(serial, 0)) for serial in ordered], dtype=np.int64)
    current = np.asarray([int(current_hits.get(serial, 0)) for serial in ordered], dtype=np.int64)
    history = np.asarray([int(history_hits.get(serial, 0)) for serial in ordered], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    differences: list[float] = []
    zero_opportunity = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(ordered), size=len(ordered), dtype=np.int64)
        denom = int(opportunity[indices].sum())
        if denom == 0:
            zero_opportunity += 1
            continue
        differences.append(float(history[indices].sum() / denom - current[indices].sum() / denom))
    return {"seed": seed, "replicates": replicates, "device_count": len(ordered), "valid_replicates": len(differences), "zero_opportunity_replicates": zero_opportunity, "quantile_025": linear_quantile(differences, 0.025), "quantile_975": linear_quantile(differences, 0.975), "mean_difference": float(np.mean(differences)) if differences else None, "differences": differences}


def _metric_add(strata: dict[tuple[str, str], dict[str, int]], typ: str, value: str, key: str, amount: int = 1) -> None:
    item = strata.setdefault((typ, value), {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0})
    item[key] += amount


def evaluate_scores(connection: sqlite3.Connection, panel: sqlite3.Connection, start: float) -> dict[str, object]:
    if metadata(connection).get("score_status") != "complete":
        raise ScoringStopped("score pass is not complete before evaluation")
    events = event_opportunities(panel)
    if (len(events), sum(int(item["opportunity"]) for item in events.values())) != (EXPECTED_EVENTS, EXPECTED_OPPORTUNITIES):
        raise ScoringStopped("Q3 event denominator differs from locked scope")
    metrics: dict[str, dict[str, object]] = {}
    event_hits_by_model: dict[str, dict[str, int]] = {}
    eligible_serials = {str(row[0]) for row in connection.execute("SELECT DISTINCT serial_number FROM model_scores")}
    dates = [str(row[0]) for row in connection.execute("SELECT DISTINCT decision_date FROM model_scores ORDER BY decision_date")]
    if len(dates) != (SCORE_END - SCORE_START).days + 1:
        raise ScoringStopped("Q3 score dates are incomplete before evaluation")
    # Build the known-outcome score vectors once.  The output DB intentionally
    # has no label table, so this cross-DB join is performed in Python after
    # the score pass has completed and never during feature construction.
    known_scores = {name: [] for name in LR_METHODS}
    known_labels: list[int] = []
    for date_text in dates:
        score_rows = connection.execute(
            "SELECT serial_number,age_lr_score,current_lr_score,history_lr_score FROM model_scores WHERE decision_date=? ORDER BY serial_number",
            (date_text,),
        ).fetchall()
        label_rows = panel.execute(
            "SELECT serial_number,label FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? AND eligible=1 AND label IS NOT NULL",
            (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON, date_text),
        ).fetchall()
        labels_by_serial = {str(row["serial_number"]): int(row["label"]) for row in label_rows}
        for row in score_rows:
            serial = str(row["serial_number"])
            if serial not in labels_by_serial:
                continue
            for name in LR_METHODS:
                known_scores[name].append(float(row[f"{name}_score"]))
            known_labels.append(labels_by_serial[serial])
    for model in METHODS:
        last_alert: dict[str, dt.date] = {}
        event_alerts: dict[str, list[tuple[dt.date, int]]] = collections.defaultdict(list)
        strata: dict[tuple[str, str], dict[str, int]] = {}
        alert_dates: dict[str, list[dt.date]] = collections.defaultdict(list)
        known_hits = known_no_hit = unknown = alerts = eligible_total = cooldown_total = 0
        all_leads: list[int] = []
        daily_rows: list[tuple[object, ...]] = []
        alert_rows: list[tuple[object, ...]] = []
        for date_text in dates:
            rows = connection.execute("SELECT * FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
            if model == "random": score_array = np.zeros(len(rows), dtype=float)
            elif model == "smart_nonzero": score_array = np.asarray([float(row["smart_nonzero_signal_count"]) for row in rows])
            elif model == "smart187": score_array = np.asarray([float(row["smart_187_signal"]) for row in rows])
            else: score_array = np.asarray([float(row[f"{model}_score"]) for row in rows])
            if model == "smart_nonzero":
                eligible_rows = [row for row in rows if int(row["smart_nonzero_signal_count"]) > 0]
                signal_count = len(eligible_rows)
                rank_rows = rows
            elif model == "smart187":
                signal_count = sum(int(row["smart_187_signal"]) > 0 for row in rows)
                rank_rows = rows
            else:
                signal_count = len(rows)
                rank_rows = rows
            labels = {str(row["serial_number"]): row for row in panel.execute("SELECT serial_number,status,label,first_failure_date FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? AND eligible=1", (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON, date_text))}
            if set(labels) != {str(row["serial_number"]) for row in rows}:
                raise ScoringStopped(f"label key set differs on {date_text}")
            selected, cooldown_excluded = select_alert_indices(rank_rows, score_array, dt.date.fromisoformat(date_text), last_alert)
            eligible_total += len(rows); cooldown_total += cooldown_excluded
            day_known_hit = day_known_no_hit = day_unknown = 0
            for index in selected:
                row = rows[index]
                serial = str(row["serial_number"])
                label_row = labels[serial]
                status = str(label_row["status"]); label = label_row["label"]
                failure_date = label_row["first_failure_date"]
                if label == 1: known_hits += 1; day_known_hit += 1
                elif label == 0: known_no_hit += 1; day_known_no_hit += 1
                else: unknown += 1; day_unknown += 1
                event_key = None; event_hit = 0; lead = None
                if serial in events and failure_date == events[serial]["first_failure_date"]:
                    failure = dt.date.fromisoformat(str(failure_date)); days = (failure - dt.date.fromisoformat(date_text)).days
                    if 1 <= days <= HORIZON and events[serial]["opportunity"]:
                        event_key = serial; event_alerts[serial].append((dt.date.fromisoformat(date_text), days))
                        event_hit = 1; lead = days
                alert_rows.append((model, date_text, serial, float(score_array[index]), str(row["tie_break_sha256"]), status, label, failure_date, event_key, event_hit, lead))
                alert_dates[serial].append(dt.date.fromisoformat(date_text))
                if event_hit:
                    all_leads.append(int(lead))
                month = date_text[:7]
                current_missing = str(int(row["current_missing_any"]))
                history_obs = str(int(row["history_observations_14"]))
                decrease = str(int(row["decrease_seen_any"]))
                cross7 = str(int(row["window_crosses_decrease_w7_any"]))
                cross14 = str(int(row["window_crosses_decrease_w14_any"]))
                for typ, value in (("month", month), ("current_missing", current_missing), ("history_observations_14", history_obs), ("decrease_seen_asof_t", decrease), ("window_crosses_decrease_w7", cross7), ("window_crosses_decrease_w14", cross14)):
                    _metric_add(strata, typ, value, "alerts")
                    if label == 1: _metric_add(strata, typ, value, "known_hit")
                    elif label == 0: _metric_add(strata, typ, value, "known_no_hit")
                    else: _metric_add(strata, typ, value, "unknown")
            for row in rows:
                for typ, value in (("month", date_text[:7]), ("current_missing", str(int(row["current_missing_any"]))), ("history_observations_14", str(int(row["history_observations_14"]))), ("decrease_seen_asof_t", str(int(row["decrease_seen_any"]))), ("window_crosses_decrease_w7", str(int(row["window_crosses_decrease_w7_any"]))), ("window_crosses_decrease_w14", str(int(row["window_crosses_decrease_w14_any"])) )):
                    _metric_add(strata, typ, value, "eligible")
            alerts += len(selected)
            day_rows = (model, date_text, len(rows), (len(rows) + 999) // 1000 if rows else 0, cooldown_excluded, len(selected), day_known_hit, day_known_no_hit, day_unknown)
            daily_rows.append(day_rows)
        connection.executemany("INSERT INTO model_daily VALUES (?,?,?,?,?,?,?,?,?)", daily_rows)
        connection.executemany("INSERT INTO model_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?)", alert_rows)
        event_rows = []
        event_hit_map: dict[str, int] = {}
        early2 = early3 = 0
        for serial, info in events.items():
            choices = sorted(event_alerts.get(serial, []))
            hit = int(bool(choices) and bool(info["opportunity"]))
            earliest_date = choices[0][0].isoformat() if choices else None
            earliest_lead = choices[0][1] if choices else None
            if hit:
                event_hit_map[serial] = 1
                early2 += int(earliest_lead >= 2)
                early3 += int(earliest_lead >= 3)
            else: event_hit_map[serial] = 0
            event_rows.append((model, serial, info["first_failure_date"], int(info["opportunity"]), hit, earliest_date, earliest_lead))
        connection.executemany("INSERT INTO model_event_summary VALUES (?,?,?,?,?,?,?)", event_rows)
        for (typ, value), item in strata.items():
            connection.execute("INSERT INTO model_strata VALUES (?,?,?,?,?,?,?,?)", (model, typ, value, item["eligible"], item["alerts"], item["known_hit"], item["known_no_hit"], item["unknown"]))
        ap = average_precision_grouped(known_scores[model], known_labels) if model in LR_METHODS else None
        leads = sorted(all_leads)
        gaps = [min((right - left).days for left, right in zip(sorted(days), sorted(days)[1:])) for days in alert_dates.values() if len(days) >= 2]
        summary = {
            "model": model, "eligible_device_days": eligible_total, "alerts": alerts,
            "known_hit_alerts": known_hits, "known_no_hit_alerts": known_no_hit, "unknown_alerts": unknown,
            "unknown_alert_ratio": unknown / alerts if alerts else None,
            "precision_lower_bound": known_hits / alerts if alerts else None,
            "precision_upper_bound": (known_hits + unknown) / alerts if alerts else None,
            "known_outcome_precision": known_hits / (known_hits + known_no_hit) if known_hits + known_no_hit else None,
            "confirmed_no_hit_per_1000_device_days": known_no_hit / eligible_total * 1000 if eligible_total else 0.0,
            "event_total": len(events), "event_opportunity_total": sum(int(info["opportunity"]) for info in events.values()),
            "event_hits": sum(event_hit_map.values()), "event_recall_at_opportunity": sum(event_hit_map.values()) / sum(int(info["opportunity"]) for info in events.values()),
            "all_event_capture": sum(event_hit_map.values()) / len(events), "early_event_hits_ge2_days": early2, "early_event_hits_ge3_days": early3,
            "early_recall_at_opportunity_ge2_days": early2 / sum(int(info["opportunity"]) for info in events.values()), "early_recall_at_opportunity_ge3_days": early3 / sum(int(info["opportunity"]) for info in events.values()),
            "hit_only_ratio_ge2_days": early2 / sum(event_hit_map.values()) if sum(event_hit_map.values()) else None, "hit_only_ratio_ge3_days": early3 / sum(event_hit_map.values()) if sum(event_hit_map.values()) else None,
            "earliest_lead_count": len(leads), "earliest_lead_median": linear_quantile(leads, 0.5), "earliest_lead_q25": linear_quantile(leads, 0.25), "earliest_lead_q75": linear_quantile(leads, 0.75),
            "repeated_alert_devices": sum(len(days) > 1 for days in alert_dates.values()), "max_alerts_per_device": max((len(days) for days in alert_dates.values()), default=0), "minimum_alert_gap_days": min(gaps) if gaps else None,
            "average_precision_known": ap, "metric_scope": "Q3 amended E2-R; unknown outcomes excluded from AP; precision bounds are unknown-outcome bounds, not confidence intervals",
        }
        metric_values = (
            model, summary["eligible_device_days"], summary["alerts"], summary["known_hit_alerts"],
            summary["known_no_hit_alerts"], summary["unknown_alerts"], summary["unknown_alert_ratio"],
            summary["precision_lower_bound"], summary["precision_upper_bound"], summary["known_outcome_precision"],
            summary["confirmed_no_hit_per_1000_device_days"], summary["event_total"],
            summary["event_opportunity_total"], summary["event_hits"], summary["event_recall_at_opportunity"],
            summary["all_event_capture"], summary["early_event_hits_ge2_days"], summary["early_event_hits_ge3_days"],
            summary["early_recall_at_opportunity_ge2_days"], summary["early_recall_at_opportunity_ge3_days"],
            summary["hit_only_ratio_ge2_days"], summary["hit_only_ratio_ge3_days"], summary["earliest_lead_count"],
            summary["earliest_lead_median"], summary["earliest_lead_q25"], summary["earliest_lead_q75"],
            summary["repeated_alert_devices"], summary["max_alerts_per_device"], summary["minimum_alert_gap_days"],
            summary["average_precision_known"], summary["metric_scope"],
        )
        connection.execute(f"INSERT INTO model_metrics VALUES ({','.join('?' for _ in metric_values)})", metric_values)
        connection.commit()
        metrics[model] = summary
        event_hits_by_model[model] = event_hit_map
        check_resources(start, f"evaluation {model}")
    serials = sorted(eligible_serials, key=lambda value: value.encode("utf-8"))
    opportunities = {serial: int(events.get(serial, {}).get("opportunity", 0)) for serial in serials}
    bootstrap = paired_device_bootstrap(serials, opportunities, event_hits_by_model["current_lr"], event_hits_by_model["history_lr"])
    current = metrics["current_lr"]; history = metrics["history_lr"]
    recall_gain = float(history["event_recall_at_opportunity"] - current["event_recall_at_opportunity"])
    unknown_change = float(history["unknown_alert_ratio"] - current["unknown_alert_ratio"])
    numeric_gate = {
        "recall_gain_at_least_0_05": recall_gain >= 0.05,
        "bootstrap_valid_at_least_1900": bootstrap["valid_replicates"] >= 1900,
        "bootstrap_lower_strictly_positive": bootstrap["quantile_025"] is not None and bootstrap["quantile_025"] > 0,
        "unknown_alert_ratio_increase_at_most_0_02": unknown_change <= 0.02,
        "opportunity_at_least_100": EXPECTED_OPPORTUNITIES >= 100,
    }
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('evaluation_status','complete')")
    connection.commit()
    return {"status": "complete", "metrics": metrics, "bootstrap_history_minus_current": bootstrap, "comparison": {"recall_gain": recall_gain, "unknown_alert_ratio_change": unknown_change, "numeric_gate": numeric_gate, "model_adoption": "pending_review" if all(numeric_gate.values()) else "retain_current_simple_control"}, "events": {"event_total": len(events), "opportunity_total": sum(int(info["opportunity"]) for info in events.values())}}


def write_model_payload_evidence(model_connection: sqlite3.Connection, output: Path) -> None:
    payload = {"model_db_sha256": sha256_file(MODEL_DB), "models": {}}
    for model in LR_METHODS:
        run = model_connection.execute("SELECT model,status,feature_count,training_rows,positive_training_rows,negative_training_rows,n_iter,intercept,params_json FROM model_runs WHERE model=?", (model,)).fetchone()
        payload["models"][model] = dict(run)
    atomic_json(output, payload)


def verify_complete_reentry() -> dict[str, object]:
    manifest_path = EVIDENCE_ROOT / "q3_scoring_complete_manifest_v1.json"
    if not OUTPUT_DB.is_file() or not manifest_path.is_file():
        raise ScoringStopped("complete amended output or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_hash")
    if expected_hash != canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"}):
        raise ScoringStopped("amended complete manifest self hash differs")
    if sha256_file(OUTPUT_DB) != manifest.get("database_sha256"):
        raise ScoringStopped("amended output database changed")
    for key, value in manifest.items():
        if key.endswith("_file_sha256"):
            path_key = key.removesuffix("_file_sha256")
            path = ROOT / str(manifest.get(path_key, ""))
            if not path.is_file() or sha256_file(path) != value:
                raise ScoringStopped(f"amended evidence changed: {path_key}")
    connection = connect_immutable(OUTPUT_DB)
    try:
        if metadata(connection).get("run_status") != "complete":
            raise ScoringStopped("amended output metadata is not complete")
        return {"status": "already_complete", "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": sha256_file(OUTPUT_DB)}
    finally:
        connection.close()


def run() -> dict[str, object]:
    if OUTPUT_DB.exists():
        return verify_complete_reentry()
    if PARTIAL_DB.exists():
        raise ScoringStopped(f"preserving existing failed amended attempt: {PARTIAL_DB}")
    config = load_config()
    binding_facts = verify_bindings(config)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    q3 = connect_immutable(Q3_DB)
    model_connection = connect_immutable(MODEL_DB)
    connection: sqlite3.Connection | None = None
    try:
        q3_meta = metadata(q3)
        if q3_meta.get("panel_status") != "complete" or q3_meta.get("training_approval") != "false":
            raise ScoringStopped("Q3 panel is not complete or training approval changed")
        model_meta = metadata(model_connection)
        if model_meta.get("run_status") != "complete" or model_meta.get("training_approval") != "false":
            raise ScoringStopped("frozen model output is not complete or training approval changed")
        _attach_prior(q3)
        connection = sqlite3.connect(PARTIAL_DB)
        connection.row_factory = sqlite3.Row
        create_schema(connection)
        for key, value in {
            "run_status": "running", "protocol": "E2-R", "config_sha256": binding_facts["config_sha256"],
            "scoring_code_sha256": sha256_file(Path(__file__)),
            "source_q3_sha256": binding_facts["bindings"]["data/derived/q3_validation_v1/panel_q3.sqlite"],
            "source_prior_sha256": binding_facts["bindings"]["data/derived/panel_q1q2_verified_v1.sqlite"],
            "model_db_sha256": binding_facts["bindings"]["data/derived/simple_baseline_v2/model_results.sqlite"],
            "zip_sha256": binding_facts["bindings"]["data/raw/q3_validation_v1/attempt_001/data_Q3_2023.zip"],
            "feature_dictionary_hash": feature_dictionary_hash(), "model": MODEL,
            "score_start": SCORE_START.isoformat(), "score_end": SCORE_END.isoformat(),
            "source_start": SOURCE_START.isoformat(), "label_cutoff": LABEL_CUTOFF.isoformat(), "horizon_days": str(HORIZON),
            "fit_allowed": "false", "q4_access": "false", "remote_setup": "false", "training_approval": "false",
            "outcome_access_during_score_pass": "false", "feature_access_mode": "one_day_whitelist_with_bounded_state",
            "decrease_policy": "retain_raw_audit_asof_t",
        }.items():
            connection.execute("INSERT INTO metadata VALUES (?,?)", (key, str(value)))
        connection.commit()
        count, day_counts = build_features(q3, connection, start)
        feature_result = {"status": "complete", "feature_rows": count, "day_counts": day_counts, "future_outcome_tables_read": False}
        atomic_json(EVIDENCE_ROOT / "features_v1.json", feature_result)
        score_result = score_features(connection, model_connection, start)
        atomic_json(EVIDENCE_ROOT / "scores_v1.json", score_result)
        evaluation = evaluate_scores(connection, q3, start)
        atomic_json(EVIDENCE_ROOT / "evaluation_v1.json", evaluation)
        write_model_payload_evidence(model_connection, EVIDENCE_ROOT / "model_payload_v1.json")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('run_status','complete')")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('completed_at_utc',?)", (dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),))
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
        connection.close(); connection = None
        os.replace(PARTIAL_DB, OUTPUT_DB)
        database_sha = sha256_file(OUTPUT_DB)
        manifest = {
            "manifest_version": "q3-scoring-amended-complete-v1", "status": "complete", "protocol": "E2-R",
            "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": database_sha,
            "features": str((EVIDENCE_ROOT / "features_v1.json").relative_to(ROOT)),
            "scores": str((EVIDENCE_ROOT / "scores_v1.json").relative_to(ROOT)),
            "evaluation": str((EVIDENCE_ROOT / "evaluation_v1.json").relative_to(ROOT)),
            "model_payload": str((EVIDENCE_ROOT / "model_payload_v1.json").relative_to(ROOT)),
            "scoring_code": str(Path(__file__).relative_to(ROOT)), "scoring_code_file_sha256": sha256_file(Path(__file__)),
            "config": str(CONFIG_PATH.relative_to(ROOT)), "config_file_sha256": sha256_file(CONFIG_PATH),
            "source_review": str(SOURCE_REVIEW.relative_to(ROOT)), "source_review_file_sha256": sha256_file(SOURCE_REVIEW),
            "anomaly_source": str(ANOMALY_SOURCE.relative_to(ROOT)), "anomaly_source_file_sha256": sha256_file(ANOMALY_SOURCE),
            "diagnosis": str(DIAGNOSIS.relative_to(ROOT)), "diagnosis_file_sha256": sha256_file(DIAGNOSIS),
            "features_file_sha256": sha256_file(EVIDENCE_ROOT / "features_v1.json"),
            "scores_file_sha256": sha256_file(EVIDENCE_ROOT / "scores_v1.json"),
            "evaluation_file_sha256": sha256_file(EVIDENCE_ROOT / "evaluation_v1.json"),
            "model_payload_file_sha256": sha256_file(EVIDENCE_ROOT / "model_payload_v1.json"),
            "training_approval": False, "fit_allowed": False, "q4_access": False, "remote_setup": False,
            "model_adoption": "pending_review", "published_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        atomic_json(EVIDENCE_ROOT / "q3_scoring_complete_manifest_v1.json", manifest)
        return {"status": "complete", "database": str(OUTPUT_DB.relative_to(ROOT)), "database_sha256": database_sha, "evaluation": evaluation, "elapsed_seconds": round(time.monotonic() - start, 3)}
    except BaseException as exc:
        if connection is not None:
            try:
                connection.execute("INSERT OR REPLACE INTO metadata VALUES ('run_status','failed')")
                connection.execute("INSERT OR REPLACE INTO metadata VALUES ('failure',?)", (f"{type(exc).__name__}: {exc}",))
                connection.commit(); connection.close()
            except sqlite3.Error:
                try: connection.close()
                except Exception: pass
        atomic_json(EVIDENCE_ROOT / "failed_attempt_v1.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "partial_database": str(PARTIAL_DB.relative_to(ROOT)), "scoring_code": str(Path(__file__).relative_to(ROOT)), "scoring_code_sha256": sha256_file(Path(__file__)), "elapsed_seconds": round(time.monotonic() - start, 3)})
        raise
    finally:
        if connection is not None:
            connection.close()
        q3.close(); model_connection.close()


def probe() -> dict[str, object]:
    rows = []
    base = {"date": "2023-07-01", "serial_number": "fixture", "model": MODEL, "failure": 0}
    for offset in range(15):
        row = dict(base); row["date"] = (SCORE_START + dt.timedelta(days=offset)).isoformat()
        for field in SMART_FIELDS: row[f"smart_{field}_raw"] = 10 + offset
        rows.append(row)
    changed = dict(rows[13]); changed["smart_5_raw"] = 2; rows[13] = changed
    feature, audit = feature_row_asof(rows[:14], rows[13]["date"])
    assert feature is not None and feature["smart_5_w7_delta_log1p"] < 0 and audit["decrease_seen_asof_t_5"] == 1
    future = dict(rows[14])
    try: feature_row_asof(rows[:14] + [future], rows[13]["date"])
    except ScoringStopped: pass
    else: raise AssertionError("future prefix was accepted")
    fixture_rows = [{"serial_number": f"s{i}", "tie_break_sha256": f"{i:064x}"} for i in range(4)]
    selected, excluded = select_alert_indices([sqlite3.Row(sqlite3.Cursor())] if False else fixture_rows, np.asarray([0.0, 0.0, 0.0, 0.0]), dt.date(2023, 7, 1), {})
    assert len(selected) == 1 and excluded == 0
    bootstrap = paired_device_bootstrap(["b", "a"], {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 1}, replicates=20)
    assert bootstrap["valid_replicates"] == 20
    return {"status": "pass", "negative_delta": feature["smart_5_w7_delta_log1p"], "decline_audit": audit, "bootstrap_valid": bootstrap["valid_replicates"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        result = probe() if args.probe else run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ScoringStopped, OSError, sqlite3.Error, ValueError) as exc:
        print(json.dumps({"status": "stopped", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
