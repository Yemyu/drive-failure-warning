"""Build leakage-aware historical features and deterministic rule replays.

This module deliberately keeps feature construction separate from label access:
the feature pass reads only the verified panel's daily table.  The replay pass
may join the already-reviewed label run only after ranking and selecting alerts.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL = "ST4000DM000"
SCORE_START = dt.date(2023, 1, 15)
SCORE_END = dt.date(2023, 6, 23)
DATASET_END = dt.date(2023, 6, 30)
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
EVENT_START = SCORE_START + dt.timedelta(days=HORIZON)
EVENT_END = SCORE_END + dt.timedelta(days=1)
EXPECTED_FEATURE_ROWS = 2_860_459
BUDGET_DENOMINATOR = 1000
COOLDOWN_DAYS = 7
MAX_OWNED_BYTES = 4 * 1024 * 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
SOFT_TIMEOUT_SECONDS = 60 * 60
FEATURE_RUN_ID = "features_train_v1"
REPLAY_RUN_ID = "rule_replay_train_v1"
RANDOM_SALT = "drive-v1|20260912"

SMART_FIELDS = (5, 9, 187, 188, 197, 198)
NONZERO_FIELDS = (5, 187, 188, 197, 198)
WINDOWS = (7, 14)
PANEL_COLUMNS = tuple(
    ["date", "serial_number", "model", "failure"]
    + [f"smart_{field}_raw" for field in SMART_FIELDS]
)
ALLOWED_INPUT_COLUMNS = frozenset(PANEL_COLUMNS)
MONOTONIC_RAW_FIELDS = (5, 9, 187)


class BuildStopped(RuntimeError):
    """A declared QA, resource, or provenance gate stopped the stage."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _rss_max_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _owned_bytes(database: Path, evidence_root: Path) -> int:
    paths = [database, database.with_name(database.name + "-wal"), database.with_name(database.name + "-shm")]
    if evidence_root.exists():
        paths.extend(item for item in evidence_root.rglob("*") if item.is_file())
    total = 0
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            total += resolved.stat().st_size
        except OSError:
            continue
    return total


def _resource_snapshot(database: Path, evidence_root: Path) -> dict[str, int]:
    return {
        "rss_max_bytes": _rss_max_bytes(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "owned_bytes": _owned_bytes(database, evidence_root),
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _relative_project_path(path: Path, field: str) -> Path:
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise BuildStopped(f"{field} escapes project root: {path}")
    return resolved


def _feature_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("decision_date", "TEXT NOT NULL"),
        ("serial_number", "TEXT NOT NULL"),
        ("model", "TEXT NOT NULL"),
        ("history_observations_14", "INTEGER NOT NULL"),
        ("observed_days_7", "INTEGER NOT NULL"),
        ("observed_days_14", "INTEGER NOT NULL"),
    ]
    for field in SMART_FIELDS:
        prefix = f"smart_{field}"
        if field == 188:
            columns.extend(
                [
                    (f"{prefix}_current_nonzero", "INTEGER"),
                    (f"{prefix}_current_missing", "INTEGER NOT NULL"),
                ]
            )
            for window in WINDOWS:
                columns.extend(
                    [
                        (f"{prefix}_w{window}_mean_nonzero", "REAL"),
                        (f"{prefix}_w{window}_missing_days", "INTEGER NOT NULL"),
                        (f"{prefix}_w{window}_delta_nonzero", "REAL"),
                        (f"{prefix}_w{window}_delta_span_days", "INTEGER"),
                    ]
                )
            continue
        columns.extend(
            [
                (f"{prefix}_current_log1p", "REAL"),
                (f"{prefix}_current_missing", "INTEGER NOT NULL"),
            ]
        )
        if field != 9:
            columns.append((f"{prefix}_current_nonzero", "INTEGER"))
        for window in WINDOWS:
            columns.extend(
                [
                    (f"{prefix}_w{window}_mean_log1p", "REAL"),
                    (f"{prefix}_w{window}_max_log1p", "REAL"),
                    (f"{prefix}_w{window}_missing_days", "INTEGER NOT NULL"),
                    (f"{prefix}_w{window}_delta_log1p", "REAL"),
                    (f"{prefix}_w{window}_delta_span_days", "INTEGER"),
                ]
            )
    columns.extend(
        [
            ("smart_nonzero_signal_count", "INTEGER NOT NULL"),
            ("smart_187_signal", "INTEGER NOT NULL"),
            ("tie_break_sha256", "TEXT NOT NULL"),
        ]
    )
    return columns


FEATURE_COLUMNS = _feature_columns()
FEATURE_COLUMN_NAMES = [name for name, _ in FEATURE_COLUMNS]


def feature_dictionary() -> list[dict[str, str]]:
    """Return a human-readable, hashable description of every output column."""
    result = [
        {"name": "decision_date", "definition": "评分日，输入截止到该自然日"},
        {"name": "serial_number", "definition": "连接键；禁止作为拟合输入"},
        {"name": "model", "definition": "主型号审计字段；研究范围固定为ST4000DM000"},
        {"name": "history_observations_14", "definition": "[t-13,t]内有设备日记录的自然日数量"},
        {"name": "observed_days_7", "definition": "[t-6,t]内有设备日记录的自然日数量"},
        {"name": "observed_days_14", "definition": "与history_observations_14相同，便于回放审计"},
    ]
    for field in SMART_FIELDS:
        prefix = f"smart_{field}"
        if field == 188:
            result.extend(
                [
                    {"name": f"{prefix}_current_nonzero", "definition": "当天raw>0的指示；空值为NULL"},
                    {"name": f"{prefix}_current_missing", "definition": "当天raw是否缺失"},
                ]
            )
            for window in WINDOWS:
                result.extend(
                    [
                        {"name": f"{prefix}_w{window}_mean_nonzero", "definition": f"过去{window}个自然日有效188非零指示均值"},
                        {"name": f"{prefix}_w{window}_missing_days", "definition": f"过去{window}个自然日中缺记录或188缺失的天数"},
                        {"name": f"{prefix}_w{window}_delta_nonzero", "definition": "当天与窗口内最早有效非零指示之差"},
                        {"name": f"{prefix}_w{window}_delta_span_days", "definition": "上述差分两端实际自然日间隔"},
                    ]
                )
            continue
        result.extend(
            [
                {"name": f"{prefix}_current_log1p", "definition": "当天非负raw的log1p；空值为NULL"},
                {"name": f"{prefix}_current_missing", "definition": "当天raw是否缺失"},
            ]
        )
        if field != 9:
            result.append({"name": f"{prefix}_current_nonzero", "definition": "当天raw>0的指示；空值为0以外的缺失由current_missing保留"})
        for window in WINDOWS:
            result.extend(
                [
                    {"name": f"{prefix}_w{window}_mean_log1p", "definition": f"过去{window}个自然日有效raw的log1p均值"},
                    {"name": f"{prefix}_w{window}_max_log1p", "definition": f"过去{window}个自然日有效raw的log1p最大值"},
                    {"name": f"{prefix}_w{window}_missing_days", "definition": f"过去{window}个自然日中缺记录或raw缺失的天数"},
                    {"name": f"{prefix}_w{window}_delta_log1p", "definition": "当天与窗口内最早有效raw之差的sign(delta)*log1p(abs(delta))"},
                    {"name": f"{prefix}_w{window}_delta_span_days", "definition": "上述差分两端实际自然日间隔"},
                ]
            )
    result.extend(
        [
            {"name": "smart_nonzero_signal_count", "definition": "当天SMART5/187/188/197/198中raw>0的数量"},
            {"name": "smart_187_signal", "definition": "当天SMART187 raw>0的指示"},
            {"name": "tie_break_sha256", "definition": "固定日期/序列号SHA256，仅用于稳定排序，不作模型输入"},
        ]
    )
    if [item["name"] for item in result] != FEATURE_COLUMN_NAMES:
        raise RuntimeError("feature dictionary does not cover feature schema exactly")
    return result


def feature_dictionary_hash() -> str:
    payload = json.dumps(feature_dictionary(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_tie_break(date: str, serial: str) -> str:
    return _sha256_text(f"{RANDOM_SALT}|{date}|{serial}")


def _date_ord(value: str | dt.date) -> int:
    return value.toordinal() if isinstance(value, dt.date) else dt.date.fromisoformat(value).toordinal()


def _row_keys(row: sqlite3.Row | dict) -> set[str]:
    try:
        return set(row.keys())
    except AttributeError as exc:
        raise BuildStopped("feature input row must expose named columns") from exc


def _validate_asof_rows(
    rows: list[sqlite3.Row | dict],
    decision_date: str,
    *,
    allow_smart_decreases: bool = False,
) -> list[sqlite3.Row | dict]:
    """Validate the complete, ordered as-of input consumed by one feature row."""
    decision_ord = _date_ord(decision_date)
    if not rows:
        return []
    expected_serial: str | None = None
    expected_model: str | None = None
    previous_ord: int | None = None
    last_nonmissing: dict[int, int] = {}
    for row in rows:
        keys = _row_keys(row)
        extra = sorted(keys - ALLOWED_INPUT_COLUMNS)
        missing = sorted(ALLOWED_INPUT_COLUMNS - keys)
        if extra or missing:
            raise BuildStopped(
                f"feature input schema mismatch; extra={extra}, missing={missing}"
            )
        try:
            row_ord = _date_ord(row["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BuildStopped("feature input has an invalid date") from exc
        if row_ord > decision_ord:
            raise BuildStopped(
                f"future row supplied to feature function: {row['date']} > {decision_date}"
            )
        if previous_ord is not None and row_ord <= previous_ord:
            raise BuildStopped("panel rows for one serial must be strictly date ordered")
        previous_ord = row_ord
        serial = row["serial_number"]
        model = row["model"]
        if serial is None or model is None:
            raise BuildStopped("feature input serial_number and model cannot be NULL")
        serial = str(serial)
        model = str(model)
        if expected_serial is None:
            expected_serial = serial
            expected_model = model
        elif serial != expected_serial or model != expected_model:
            raise BuildStopped("feature input must contain one serial and one model")
        if model != MODEL:
            raise BuildStopped(f"feature input model differs from locked model: {model}")
        if row["failure"] not in (0, 1):
            raise BuildStopped("feature input failure must be 0 or 1")
        for field_number in SMART_FIELDS:
            value = row[f"smart_{field_number}_raw"]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BuildStopped(f"SMART{field_number} raw value is not numeric")
            if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
                raise BuildStopped(f"SMART{field_number} raw value is not an integer")
            integer_value = int(value)
            if integer_value < 0:
                raise BuildStopped(f"negative SMART value in smart_{field_number}_raw")
            if field_number in MONOTONIC_RAW_FIELDS and not allow_smart_decreases:
                previous_value = last_nonmissing.get(field_number)
                if previous_value is not None and integer_value < previous_value:
                    raise BuildStopped(
                        f"SMART{field_number} raw value decreases in ordered input"
                    )
                last_nonmissing[field_number] = integer_value
    return rows


def _transform_delta(delta: int | float) -> float:
    if delta == 0:
        return 0.0
    return math.copysign(math.log1p(abs(delta)), delta)


def _window_values(row_by_ord: dict[int, sqlite3.Row | dict], current_ord: int, field: str, window: int) -> list[tuple[int, int | float]]:
    values = []
    for ordinal in range(current_ord - window + 1, current_ord + 1):
        row = row_by_ord.get(ordinal)
        if row is None:
            continue
        value = row[field]
        if value is None:
            continue
        value = int(value)
        if value < 0:
            raise BuildStopped(f"negative SMART value in {field} at ordinal {ordinal}")
        values.append((ordinal, value))
    return values


def historical_rows_until(
    rows: list[sqlite3.Row | dict],
    decision_date: str,
    *,
    allow_smart_decreases: bool = False,
) -> list[sqlite3.Row | dict]:
    """Return the ordered panel prefix that is available at ``decision_date``.

    The panel query used by the builder is ordered by serial number and date.
    Keeping this prefix operation explicit makes the as-of boundary visible to
    callers and lets the feature function reject an accidentally supplied
    future row instead of silently ignoring it.
    """
    return _validate_asof_rows(
        rows, decision_date, allow_smart_decreases=allow_smart_decreases
    )


def feature_row_for_serial(
    rows: list[sqlite3.Row | dict],
    decision_date: str,
    *,
    allow_smart_decreases: bool = False,
) -> tuple | None:
    """Build one eligible feature row from a single serial's panel rows.

    This pure function is also used by synthetic tests. It never receives or
    consults label_flow, future labels, or a device's last observed date.
    """
    rows = _validate_asof_rows(
        rows, decision_date, allow_smart_decreases=allow_smart_decreases
    )
    decision_ord = _date_ord(decision_date)
    row_by_ord = {_date_ord(row["date"]): row for row in rows}
    current = row_by_ord.get(decision_ord)
    if current is None or int(current["failure"]):
        return None
    first_failure = min(
        (_date_ord(row["date"]) for row in rows if int(row["failure"])),
        default=None,
    )
    if first_failure is not None and first_failure <= decision_ord:
        return None
    history = [row_by_ord[ordinal] for ordinal in range(decision_ord - HISTORY_DAYS + 1, decision_ord + 1) if ordinal in row_by_ord]
    if len(history) < MIN_HISTORY:
        return None

    values: dict[str, int | float | str | None] = {
        "decision_date": decision_date,
        "serial_number": current["serial_number"],
        "model": current["model"],
        "history_observations_14": len(history),
        "observed_days_7": sum(1 for ordinal in range(decision_ord - 6, decision_ord + 1) if ordinal in row_by_ord),
        "observed_days_14": len(history),
    }
    for field_number in SMART_FIELDS:
        field = f"smart_{field_number}_raw"
        prefix = f"smart_{field_number}"
        current_value = current[field]
        if current_value is not None:
            current_value = int(current_value)
            if current_value < 0:
                raise BuildStopped(f"negative SMART value in {field} on {decision_date}")
        current_missing = int(current_value is None)
        if field_number == 188:
            values[f"{prefix}_current_nonzero"] = None if current_missing else int(current_value > 0)
            values[f"{prefix}_current_missing"] = current_missing
            for window in WINDOWS:
                valid = _window_values(row_by_ord, decision_ord, field, window)
                indicators = [int(value > 0) for _, value in valid]
                values[f"{prefix}_w{window}_mean_nonzero"] = sum(indicators) / len(indicators) if indicators else None
                values[f"{prefix}_w{window}_missing_days"] = window - len(valid)
                if not current_missing and len(valid) >= 2:
                    values[f"{prefix}_w{window}_delta_nonzero"] = int(current_value > 0) - indicators[0]
                    values[f"{prefix}_w{window}_delta_span_days"] = decision_ord - valid[0][0]
                else:
                    values[f"{prefix}_w{window}_delta_nonzero"] = None
                    values[f"{prefix}_w{window}_delta_span_days"] = None
            continue
        values[f"{prefix}_current_log1p"] = math.log1p(current_value) if current_value is not None else None
        values[f"{prefix}_current_missing"] = current_missing
        if field_number != 9:
            values[f"{prefix}_current_nonzero"] = None if current_missing else int(current_value > 0)
        for window in WINDOWS:
            valid = _window_values(row_by_ord, decision_ord, field, window)
            logs = [math.log1p(value) for _, value in valid]
            values[f"{prefix}_w{window}_mean_log1p"] = sum(logs) / len(logs) if logs else None
            values[f"{prefix}_w{window}_max_log1p"] = max(logs) if logs else None
            values[f"{prefix}_w{window}_missing_days"] = window - len(valid)
            if not current_missing and len(valid) >= 2:
                values[f"{prefix}_w{window}_delta_log1p"] = _transform_delta(current_value - valid[0][1])
                values[f"{prefix}_w{window}_delta_span_days"] = decision_ord - valid[0][0]
            else:
                values[f"{prefix}_w{window}_delta_log1p"] = None
                values[f"{prefix}_w{window}_delta_span_days"] = None
    values["smart_nonzero_signal_count"] = sum(
        int(current[f"smart_{field}_raw"] is not None and int(current[f"smart_{field}_raw"]) > 0)
        for field in NONZERO_FIELDS
    )
    values["smart_187_signal"] = int(current["smart_187_raw"] is not None and int(current["smart_187_raw"]) > 0)
    values["tie_break_sha256"] = stable_tie_break(decision_date, str(current["serial_number"]))
    return tuple(values[name] for name in FEATURE_COLUMN_NAMES)


def _panel_connect(path: Path) -> sqlite3.Connection:
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise BuildStopped(f"refusing panel with non-empty WAL: {wal}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _feature_ddl(connection: sqlite3.Connection) -> None:
    fields = ",\n            ".join(f'"{name}" {definition}' for name, definition in FEATURE_COLUMNS)
    connection.executescript(
        f"""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS feature_rows (
            {fields},
            PRIMARY KEY (decision_date, serial_number)
        );
        CREATE INDEX IF NOT EXISTS feature_date ON feature_rows(decision_date);
        CREATE TABLE IF NOT EXISTS replay_runs (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            feature_dictionary_hash TEXT NOT NULL,
            source_panel_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            PRIMARY KEY(run_id, method)
        );
        CREATE TABLE IF NOT EXISTS replay_daily (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            eligible_count INTEGER NOT NULL,
            budget_k INTEGER NOT NULL,
            signal_count INTEGER NOT NULL,
            cooldown_excluded INTEGER NOT NULL,
            alerts_count INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL,
            known_no_hit_alerts INTEGER NOT NULL,
            unknown_alerts INTEGER NOT NULL,
            event_hits INTEGER NOT NULL,
            early_event_hits INTEGER NOT NULL,
            PRIMARY KEY(run_id, method, decision_date)
        );
        CREATE TABLE IF NOT EXISTS replay_alerts (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            decision_date TEXT NOT NULL,
            serial_number TEXT NOT NULL,
            score REAL NOT NULL,
            tie_break_sha256 TEXT NOT NULL,
            status TEXT,
            label INTEGER,
            first_failure_date TEXT,
            event_key TEXT,
            event_hit INTEGER NOT NULL,
            lead_days INTEGER,
            PRIMARY KEY(run_id, method, decision_date, serial_number)
        );
        CREATE TABLE IF NOT EXISTS replay_event_summary (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            event_key TEXT NOT NULL,
            first_failure_date TEXT NOT NULL,
            opportunity INTEGER NOT NULL,
            hit INTEGER NOT NULL,
            earliest_alert_date TEXT,
            earliest_lead_days INTEGER,
            PRIMARY KEY(run_id, method, event_key)
        );
        """
    )


def _set_metadata(connection: sqlite3.Connection, values: dict[str, str]) -> None:
    connection.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", values.items())
    connection.commit()


def _panel_meta(connection: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata")}


def _config_hash(panel_hash: str, code_hash: str) -> str:
    payload = {
        "model": MODEL,
        "score_start": SCORE_START.isoformat(),
        "score_end": SCORE_END.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "horizon": HORIZON,
        "history_days": HISTORY_DAYS,
        "min_history": MIN_HISTORY,
        "windows": WINDOWS,
        "panel_hash": panel_hash,
        "code_hash": code_hash,
        "feature_dictionary_hash": feature_dictionary_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _feature_qa(connection: sqlite3.Connection) -> dict:
    failures: list[str] = []
    count = connection.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
    dates = connection.execute("SELECT MIN(decision_date), MAX(decision_date), COUNT(DISTINCT decision_date) FROM feature_rows").fetchone()
    serials = connection.execute("SELECT COUNT(DISTINCT serial_number) FROM feature_rows").fetchone()[0]
    if count != EXPECTED_FEATURE_ROWS:
        failures.append(f"feature row count {count} != {EXPECTED_FEATURE_ROWS}")
    if tuple(dates) != (SCORE_START.isoformat(), SCORE_END.isoformat(), (SCORE_END - SCORE_START).days + 1):
        failures.append(f"feature date range/count mismatch: {tuple(dates)}")
    if connection.execute("SELECT COUNT(*) FROM feature_rows WHERE model != ?", (MODEL,)).fetchone()[0]:
        failures.append("feature model differs from locked model")
    if connection.execute("SELECT COUNT(*) FROM feature_rows WHERE history_observations_14 < 12 OR history_observations_14 > 14").fetchone()[0]:
        failures.append("history observations outside 12..14")
    if connection.execute("SELECT COUNT(*) FROM feature_rows WHERE observed_days_7 < 0 OR observed_days_7 > 7 OR observed_days_14 < 0 OR observed_days_14 > 14").fetchone()[0]:
        failures.append("observed day counts outside declared windows")
    for field in SMART_FIELDS:
        for column in (f"smart_{field}_current_missing",) if field == 188 else (f"smart_{field}_current_missing",):
            if connection.execute(f'SELECT COUNT(*) FROM feature_rows WHERE "{column}" NOT IN (0,1)').fetchone()[0]:
                failures.append(f"invalid missing flag {column}")
        for window in WINDOWS:
            missing = f"smart_{field}_w{window}_missing_days"
            if connection.execute(f'SELECT COUNT(*) FROM feature_rows WHERE "{missing}" < 0 OR "{missing}" > ?', (window,)).fetchone()[0]:
                failures.append(f"invalid missing days {missing}")
    if connection.execute("SELECT COUNT(*) FROM feature_rows WHERE length(tie_break_sha256) != 64").fetchone()[0]:
        failures.append("invalid tie-break digest")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        failures.append(f"feature sqlite integrity={integrity}")
    return {
        "status": "pass" if not failures else "fail",
        "feature_rows": count,
        "date_min": dates[0],
        "date_max": dates[1],
        "distinct_dates": dates[2],
        "distinct_serials": serials,
        "integrity_check": integrity,
        "failures": failures,
    }


def build_features(panel_db: Path, output_db: Path, evidence_root: Path, *, force: bool = False) -> dict:
    panel_db = _relative_project_path(panel_db, "panel database")
    output_db = _relative_project_path(output_db, "feature database")
    evidence_root = _relative_project_path(evidence_root, "feature evidence")
    if not panel_db.is_file():
        raise BuildStopped(f"panel database missing: {panel_db}")
    panel = _panel_connect(panel_db)
    meta = _panel_meta(panel)
    if meta.get("build_status") != "panel_complete":
        panel.close()
        raise BuildStopped("panel is not complete")
    if meta.get("training_approval") != "false":
        panel.close()
        raise BuildStopped("panel training_approval must remain false")
    panel_hash = meta.get("build_manifest_hash", "")
    if not panel_hash:
        panel.close()
        raise BuildStopped("panel source manifest hash missing")
    code_hash = _sha256_file(Path(__file__))
    config_hash = _config_hash(panel_hash, code_hash)
    evidence_root.mkdir(parents=True, exist_ok=True)
    qa_path = evidence_root / "feature_qa_v1.json"
    dictionary_path = evidence_root / "feature_dictionary_v1.json"
    progress_path = evidence_root / "feature_progress_v1.json"
    if output_db.exists() and not force:
        existing = sqlite3.connect(output_db)
        existing.row_factory = sqlite3.Row
        existing_meta = _panel_meta(existing)
        if existing_meta.get("feature_status") == "complete" and existing_meta.get("feature_config_hash") == config_hash:
            qa = _feature_qa(existing)
            existing.close(); panel.close()
            _atomic_json(qa_path, qa)
            return {"status": "already_complete", "database": str(output_db), "qa": qa}
        existing.close()
        panel.close()
        raise BuildStopped(f"feature database exists; use --force only after preserving it: {output_db}")
    partial_db = output_db.with_name(output_db.name + ".partial")
    if partial_db.exists() and not force:
        panel.close()
        raise BuildStopped(f"partial feature database exists; preserve and inspect before retry: {partial_db}")
    if partial_db.exists() and force:
        archive = evidence_root / "failed_feature_attempts" / f"{partial_db.name}.{int(time.time())}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_db, archive)
    connection = sqlite3.connect(partial_db)
    connection.row_factory = sqlite3.Row
    _feature_ddl(connection)
    _set_metadata(
        connection,
        {
            "feature_status": "running",
            "feature_version": "features_train_v1",
            "feature_config_hash": config_hash,
            "feature_code_sha256": code_hash,
            "source_panel_manifest_hash": panel_hash,
            "source_panel_database": str(panel_db.relative_to(ROOT)),
            "model": MODEL,
            "score_start": SCORE_START.isoformat(),
            "score_end": SCORE_END.isoformat(),
            "dataset_end": DATASET_END.isoformat(),
            "feature_dictionary_hash": feature_dictionary_hash(),
            "training_approval": "false",
        },
    )
    _atomic_json(dictionary_path, {"status": "locked", "hash": feature_dictionary_hash(), "columns": feature_dictionary()})
    progress = {"status": "running", "feature_rows": 0, "serials_completed": 0, "started_at_utc": _utc_now(), "resources": []}
    _atomic_json(progress_path, progress)
    start = time.monotonic()
    insert_sql = f'INSERT INTO feature_rows ({", ".join(FEATURE_COLUMN_NAMES)}) VALUES ({", ".join("?" for _ in FEATURE_COLUMN_NAMES)})'
    cursor = panel.execute(
        "SELECT date,serial_number,model,failure," + ",".join(f"smart_{field}_raw" for field in SMART_FIELDS) + " FROM daily WHERE model=? AND date <= ? ORDER BY serial_number,date",
        (MODEL, DATASET_END.isoformat()),
    )
    completed = 0
    feature_count = 0
    try:
        for serial, grouped in itertools.groupby(cursor, key=lambda row: row["serial_number"]):
            rows = list(grouped)
            available_rows: list[sqlite3.Row | dict] = []
            row_index = 0
            batch = []
            for offset in range((SCORE_END - SCORE_START).days + 1):
                date = SCORE_START + dt.timedelta(days=offset)
                while row_index < len(rows) and _date_ord(rows[row_index]["date"]) <= date.toordinal():
                    available_rows.append(rows[row_index])
                    row_index += 1
                item = feature_row_for_serial(available_rows, date.isoformat())
                if item is None:
                    continue
                batch.append(item)
                if len(batch) >= 5000:
                    connection.executemany(insert_sql, batch)
                    connection.commit()
                    feature_count += len(batch)
                    batch.clear()
                    snapshot = _resource_snapshot(partial_db, evidence_root)
                    progress["feature_rows"] = feature_count
                    progress["resources"].append({"feature_rows": feature_count, **snapshot})
                    _atomic_json(progress_path, progress)
                    if snapshot["rss_max_bytes"] >= MAX_RSS_BYTES or snapshot["owned_bytes"] >= MAX_OWNED_BYTES or snapshot["free_bytes"] < MIN_FREE_BYTES:
                        raise BuildStopped(f"feature resource gate reached at {feature_count}: {snapshot}")
                    if time.monotonic() - start >= SOFT_TIMEOUT_SECONDS:
                        raise BuildStopped("feature soft timeout reached")
            if batch:
                connection.executemany(insert_sql, batch)
                connection.commit()
                feature_count += len(batch)
            completed += 1
            if completed % 100 == 0:
                snapshot = _resource_snapshot(partial_db, evidence_root)
                progress.update({"feature_rows": feature_count, "serials_completed": completed, "last_completed_at_utc": _utc_now()})
                progress["resources"].append({"feature_rows": feature_count, **snapshot})
                _atomic_json(progress_path, progress)
                if snapshot["rss_max_bytes"] >= MAX_RSS_BYTES or snapshot["owned_bytes"] >= MAX_OWNED_BYTES or snapshot["free_bytes"] < MIN_FREE_BYTES:
                    raise BuildStopped(f"feature resource gate reached after serial {serial}: {snapshot}")
                if time.monotonic() - start >= SOFT_TIMEOUT_SECONDS:
                    raise BuildStopped("feature soft timeout reached")
        qa = _feature_qa(connection)
        _atomic_json(qa_path, qa)
        if qa["status"] != "pass":
            raise BuildStopped("feature QA failed: " + "; ".join(qa["failures"]))
        _set_metadata(connection, {"feature_status": "complete", "feature_completed_at_utc": _utc_now(), "training_approval": "false"})
        progress.update({"status": "complete", "feature_rows": feature_count, "serials_completed": completed, "completed_at_utc": _utc_now()})
        _atomic_json(progress_path, progress)
        connection.close(); panel.close()
        os.replace(partial_db, output_db)
        return {"status": "complete", "database": str(output_db), "qa": qa, "feature_rows": feature_count, "serials_completed": completed}
    except BaseException as exc:
        progress.update({"status": "failed", "feature_rows": feature_count, "serials_completed": completed, "error": f"{type(exc).__name__}: {exc}", "failed_at_utc": _utc_now()})
        _atomic_json(progress_path, progress)
        try:
            connection.rollback(); connection.close()
        finally:
            panel.close()
        raise


def _load_feature_rows(connection: sqlite3.Connection, date: str) -> list[sqlite3.Row]:
    return connection.execute("SELECT * FROM feature_rows WHERE decision_date=? ORDER BY tie_break_sha256,serial_number", (date,)).fetchall()


def _event_opportunities(panel: sqlite3.Connection) -> dict[str, dict]:
    opportunities: dict[str, dict] = {}
    serials = panel.execute("SELECT DISTINCT serial_number FROM daily WHERE model=? ORDER BY serial_number", (MODEL,))
    for item in serials:
        serial = item[0]
        rows = panel.execute("SELECT date,failure FROM daily WHERE model=? AND serial_number=? ORDER BY date", (MODEL, serial)).fetchall()
        failure_dates = [dt.date.fromisoformat(row["date"]) for row in rows if row["failure"]]
        if not failure_dates:
            continue
        failure_date = failure_dates[0]
        if not EVENT_START <= failure_date <= EVENT_END:
            continue
        days = {dt.date.fromisoformat(row["date"]) for row in rows}
        candidate = []
        for offset in range(1, HORIZON + 1):
            decision = failure_date - dt.timedelta(days=offset)
            if not SCORE_START <= decision <= SCORE_END or decision not in days:
                continue
            history = sum(decision - dt.timedelta(days=i) in days for i in range(HISTORY_DAYS))
            if history >= MIN_HISTORY:
                candidate.append(decision)
        opportunities[serial] = {
            "event_key": serial,
            "first_failure_date": failure_date.isoformat(),
            "opportunity": int(bool(candidate)),
            "opportunity_dates": [day.isoformat() for day in sorted(candidate)],
        }
    return opportunities


def _label_for_alert(panel: sqlite3.Connection, date: str, serial: str) -> sqlite3.Row | None:
    return panel.execute(
        "SELECT status,label,first_failure_date FROM label_flow WHERE run_id='train_q1q2_verified_h7_v1' AND split='train' AND horizon_days=7 AND decision_date=? AND serial_number=?",
        (date, serial),
    ).fetchone()


def _rank_rows(rows: list[sqlite3.Row], method: str) -> list[sqlite3.Row]:
    if method == "random":
        return sorted(rows, key=lambda row: (row["tie_break_sha256"], row["serial_number"]))
    if method == "smart_nonzero":
        return sorted(
            (row for row in rows if row["smart_nonzero_signal_count"] > 0),
            key=lambda row: (-row["smart_nonzero_signal_count"], row["tie_break_sha256"], row["serial_number"]),
        )
    if method == "smart187":
        return sorted(
            (row for row in rows if row["smart_187_signal"] > 0),
            key=lambda row: (row["tie_break_sha256"], row["serial_number"]),
        )
    raise ValueError(f"unknown replay method: {method}")


def replay_methods(panel_db: Path, feature_db: Path, evidence_root: Path) -> dict:
    panel_db = _relative_project_path(panel_db, "panel database")
    feature_db = _relative_project_path(feature_db, "feature database")
    evidence_root = _relative_project_path(evidence_root, "replay evidence")
    if not feature_db.is_file():
        raise BuildStopped(f"feature database missing: {feature_db}")
    panel = _panel_connect(panel_db)
    feature = sqlite3.connect(feature_db)
    feature.row_factory = sqlite3.Row
    meta = _panel_meta(feature)
    if meta.get("feature_status") != "complete":
        feature.close(); panel.close(); raise BuildStopped("feature database is not complete")
    if meta.get("training_approval") != "false":
        feature.close(); panel.close(); raise BuildStopped("feature database training_approval must remain false")
    qa = _feature_qa(feature)
    if qa["status"] != "pass":
        feature.close(); panel.close(); raise BuildStopped("feature QA is not pass")
    _feature_ddl(feature)
    evidence_root.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps({"run_id": REPLAY_RUN_ID, "methods": ["random", "smart_nonzero", "smart187"], "budget_denominator": BUDGET_DENOMINATOR, "cooldown_days": COOLDOWN_DAYS, "event_start": EVENT_START.isoformat(), "event_end": EVENT_END.isoformat(), "feature_config_hash": meta["feature_config_hash"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    opportunities = _event_opportunities(panel)
    opportunity_total = sum(int(v["opportunity"]) for v in opportunities.values())
    event_total = len(opportunities)
    if (event_total, opportunity_total) != (289, 286):
        feature.close(); panel.close(); raise BuildStopped(f"event opportunity denominator changed: {(event_total, opportunity_total)}")
    methods = ("random", "smart_nonzero", "smart187")
    summary_by_method: dict[str, dict] = {}
    for method in methods:
        if feature.execute("SELECT 1 FROM replay_runs WHERE run_id=? AND method=?", (REPLAY_RUN_ID, method)).fetchone():
            feature.close(); panel.close(); raise BuildStopped(f"replay already exists for {method}; preserve it")
        cooldown_until: dict[str, dt.date] = {}
        event_alerts: dict[str, list[tuple[dt.date, int]]] = collections.defaultdict(list)
        daily_rows = []
        alert_rows = []
        known_hits = known_no_hit = unknown = alerts = event_hits = early_hits = 0
        total_eligible = 0
        for offset in range((SCORE_END - SCORE_START).days + 1):
            date = SCORE_START + dt.timedelta(days=offset)
            date_text = date.isoformat()
            rows = _load_feature_rows(feature, date_text)
            n = len(rows); k = (n + BUDGET_DENOMINATOR - 1) // BUDGET_DENOMINATOR if n else 0
            total_eligible += n
            ranked = _rank_rows(rows, method)
            signal_count = sum(1 for row in rows if (method == "random" or (row["smart_nonzero_signal_count"] > 0 if method == "smart_nonzero" else row["smart_187_signal"] > 0)))
            selected = []
            cooldown_excluded = 0
            for row in ranked:
                if date <= cooldown_until.get(row["serial_number"], date - dt.timedelta(days=COOLDOWN_DAYS + 1)):
                    cooldown_excluded += 1
                    continue
                selected.append(row)
                if len(selected) >= k:
                    break
            day_hits = day_early = day_known_hits = day_known_no_hit = day_unknown = 0
            for row in selected:
                serial = row["serial_number"]
                cooldown_until[serial] = date + dt.timedelta(days=COOLDOWN_DAYS)
                label = _label_for_alert(panel, date_text, serial)
                if label is None:
                    feature.close(); panel.close(); raise BuildStopped(f"missing label for alert key {date_text}/{serial}")
                status = label["status"]; label_value = label["label"]; failure_date = label["first_failure_date"]
                if label_value == 1:
                    day_known_hits += 1; known_hits += 1
                elif label_value == 0:
                    day_known_no_hit += 1; known_no_hit += 1
                else:
                    day_unknown += 1; unknown += 1
                event_key = None; event_hit = 0; lead_days = None
                if failure_date:
                    failure = dt.date.fromisoformat(failure_date)
                    lead = (failure - date).days
                    if EVENT_START <= failure <= EVENT_END and 1 <= lead <= HORIZON and serial in opportunities:
                        event_key = serial
                        event_alerts[serial].append((date, lead))
                        if opportunities[serial]["opportunity"]:
                            event_hit = 1; lead_days = lead; day_hits += 1; event_hits += 1
                            if lead >= 2:
                                day_early += 1; early_hits += 1
                alert_rows.append((REPLAY_RUN_ID, method, date_text, serial, float(row["smart_nonzero_signal_count"] if method == "smart_nonzero" else row["smart_187_signal"] if method == "smart187" else 0.0), row["tie_break_sha256"], status, label_value, failure_date, event_key, event_hit, lead_days))
            alerts += len(selected)
            daily_rows.append((REPLAY_RUN_ID, method, date_text, n, k, signal_count, cooldown_excluded, len(selected), day_known_hits, day_known_no_hit, day_unknown, day_hits, day_early))
        event_rows = []
        event_hit_count = early_event_count = 0
        lead_values = []
        lead_2 = lead_3 = 0
        for event_key, info in opportunities.items():
            choices = sorted(event_alerts.get(event_key, []))
            hit = int(bool(choices) and bool(info["opportunity"]))
            earliest_date = choices[0][0].isoformat() if choices else None
            earliest_lead = choices[0][1] if choices else None
            if hit:
                event_hit_count += 1; lead_values.append(earliest_lead)
                if earliest_lead >= 2: early_event_count += 1; lead_2 += 1
                if earliest_lead >= 3: lead_3 += 1
            event_rows.append((REPLAY_RUN_ID, method, event_key, info["first_failure_date"], info["opportunity"], hit, earliest_date, earliest_lead))
        lower = known_hits / alerts if alerts else None
        upper = (known_hits + unknown) / alerts if alerts else None
        known_precision = known_hits / (known_hits + known_no_hit) if known_hits + known_no_hit else None
        summary = {
            "status": "pass",
            "method": method,
            "eligible_device_days": total_eligible,
            "alerts": alerts,
            "known_hit_alerts": known_hits,
            "known_no_hit_alerts": known_no_hit,
            "unknown_alerts": unknown,
            "unknown_alert_ratio": unknown / alerts if alerts else None,
            "precision_lower_bound": lower,
            "precision_upper_bound": upper,
            "known_outcome_precision": known_precision,
            "event_total": event_total,
            "event_opportunity_total": opportunity_total,
            "event_hits": event_hit_count,
            "event_recall_at_opportunity": event_hit_count / opportunity_total if opportunity_total else None,
            "all_event_capture": event_hit_count / event_total if event_total else None,
            "early_event_hits_ge2_days": early_event_count,
            "early_recall_at_opportunity": early_event_count / opportunity_total if opportunity_total else None,
            "earliest_lead_days": {"median": sorted(lead_values)[len(lead_values)//2] if lead_values else None, "values_count": len(lead_values), "ge2": lead_2, "ge3": lead_3},
            "budget_denominator": BUDGET_DENOMINATOR,
            "cooldown_days": COOLDOWN_DAYS,
            "training_approval": False,
        }
        feature.executemany("INSERT INTO replay_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", daily_rows)
        feature.executemany("INSERT INTO replay_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", alert_rows)
        feature.executemany("INSERT INTO replay_event_summary VALUES (?,?,?,?,?,?,?,?)", event_rows)
        feature.execute("INSERT INTO replay_runs VALUES (?,?,?,?,?,?,?,?)", (REPLAY_RUN_ID, method, config_hash, meta["feature_dictionary_hash"], meta["source_panel_manifest_hash"], "complete", json.dumps(summary, ensure_ascii=False, sort_keys=True), _utc_now()))
        feature.commit()
        summary_by_method[method] = summary
    replay_qa = {"status": "pass", "run_id": REPLAY_RUN_ID, "config_hash": config_hash, "event_denominators": {"all": event_total, "with_opportunity": opportunity_total}, "methods": summary_by_method, "training_approval": False, "limitations": ["训练期开发诊断；无验证/测试泛化结论。", "未知告警保留并给精度上下界；未把未知填成负。", "未拟合年龄、逻辑回归或树模型；未做bootstrap/敏感性预算。"]}
    _atomic_json(evidence_root / "replay_qa_v1.json", replay_qa)
    _set_metadata(feature, {"replay_status": "complete", "replay_config_hash": config_hash, "training_approval": "false"})
    feature.close(); panel.close()
    return replay_qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=ROOT / "data/derived/panel_q1q2_verified_v1.sqlite")
    parser.add_argument("--database", type=Path, default=ROOT / "data/derived/features_train_v1.sqlite")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "evidence/q2/feature_replay_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if not args.replay_only:
        result = build_features(args.panel, args.database, args.evidence_root, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] == "already_complete":
            return 0
    result = replay_methods(args.panel, _relative_project_path(args.database, "feature database"), args.evidence_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
