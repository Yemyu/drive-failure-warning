"""Independent replay of the raw panel, labels, and current features.

The production builders are deliberately not imported here.  This module is a
small reference implementation used at the training boundary: it reads the
source panel with the same cutoff, recomputes the locked label flow and
current-as-of features, and compares both results with the private stage
databases.  A mismatch stops the run before a sample can be fitted.
"""

from __future__ import annotations

from collections import Counter, deque
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .artifacts import ArtifactError, artifact_facts, bound_path
from .contract import LOCKED_FEATURES
from .runtime_context import project_root


ROOT = project_root()
MODEL = "ST4000DM000"
HORIZON_DAYS = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
RAW_FIELDS = (5, 9, 187, 188, 197, 198)
NONZERO_FIELDS = (5, 187, 188, 197, 198)
ABS_TOL = 1e-12
REL_TOL = 1e-10
REFERENCE_VERSION = "panel-reference-v1"

LABEL_COLUMNS = (
    "run_id", "decision_date", "serial_number", "model", "capacity_bytes",
    "first_failure_date", "label", "status", "eligible",
    "history_observations", "future_observations",
)
FEATURE_COLUMNS = (
    "decision_date", "serial_number", "model", *LOCKED_FEATURES,
    "smart_nonzero_signal_count", "tie_break_sha256",
)
FEATURE_FLAG_COLUMNS = {
    name for name in LOCKED_FEATURES
    if name.endswith("_current_missing") or name.endswith("_current_nonzero")
}
CALENDAR_COLUMNS = (
    "decision_date", "observed_rows", "model_rows", "failed_model_rows",
    "history_ready_rows", "eligible_rows",
)
KNOWN_LABEL_STATUSES = {
    "positive_observed", "positive_with_gap", "negative_observed",
    "end_censored", "gap_or_exit_censored", "history_insufficient",
    "same_day_failure", "post_failure",
}


class PanelReferenceError(RuntimeError):
    """The raw panel and one of its derived training tables disagree."""


def _fail(message: str) -> None:
    raise PanelReferenceError(message)


def _bound(path: Path | str, label: str, *, must_exist: bool = True) -> Path:
    try:
        return bound_path(path, label, must_exist=must_exist)
    except ArtifactError as exc:
        raise PanelReferenceError(str(exc)) from exc


def _check_progress(callback: Callable[[], int] | None) -> None:
    if callback is not None and callback():
        _fail("panel reference check cancelled")


def _parse_date(value: object, label: str) -> dt.date:
    if type(value) is not str:
        _fail(f"{label} must be an ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise PanelReferenceError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        _fail(f"{label} must use canonical YYYY-MM-DD")
    return parsed


def _validate_arguments(start: str, end: str, cutoff: str) -> tuple[dt.date, dt.date, dt.date]:
    lo, hi, source_end = _parse_date(start, "start"), _parse_date(end, "end"), _parse_date(cutoff, "cutoff")
    if lo > hi or hi > source_end:
        _fail("panel reference date range is invalid")
    return lo, hi, source_end


def _open_readonly(path: Path, label: str) -> sqlite3.Connection:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        try:
            if sidecar.exists() and sidecar.stat().st_size:
                _fail(f"non-empty SQLite sidecar for {label}: {sidecar}")
        except OSError as exc:
            raise PanelReferenceError(f"cannot inspect SQLite sidecar: {sidecar}") from exc
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as exc:
        raise PanelReferenceError(f"cannot open {label}: {path}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
    except sqlite3.Error as exc:
        connection.close()
        raise PanelReferenceError(f"cannot set read-only mode for {label}") from exc
    return connection


def _table_columns(connection: sqlite3.Connection, table: str, label: str) -> set[str]:
    try:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error as exc:
        raise PanelReferenceError(f"cannot inspect {label} schema") from exc
    if not columns:
        _fail(f"{label} table is missing")
    return columns


def _metadata(connection: sqlite3.Connection, label: str) -> dict[str, str]:
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "metadata" not in tables:
        _fail(f"{label} has no metadata table")
    try:
        rows = connection.execute("SELECT key,value FROM metadata").fetchall()
    except sqlite3.Error as exc:
        raise PanelReferenceError(f"cannot read {label} metadata") from exc
    result: dict[str, str] = {}
    for key, value in rows:
        if type(key) is not str or not key or type(value) is not str or key in result:
            _fail(f"{label} metadata contains an invalid or duplicate key")
        result[key] = value
    return result


def _validate_raw_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(row)
    required = {"date", "serial_number", "model", "capacity_bytes", "failure", *(f"smart_{field}_raw" for field in RAW_FIELDS)}
    if not required.issubset(row):
        _fail("panel daily row is missing a required field")
    date_text = row["date"]
    _parse_date(date_text, "panel daily date")
    serial = row["serial_number"]
    model = row["model"]
    if type(serial) is not str or not serial or type(model) is not str or not model:
        _fail("panel daily identity is invalid")
    capacity = row["capacity_bytes"]
    if capacity is not None and (type(capacity) is not int or isinstance(capacity, bool) or capacity < 0):
        _fail(f"invalid capacity at {date_text}/{serial}")
    failure = row["failure"]
    if type(failure) is not int or isinstance(failure, bool) or failure not in (0, 1):
        _fail(f"invalid failure flag at {date_text}/{serial}")
    item = dict(row)
    for field in RAW_FIELDS:
        value = item[f"smart_{field}_raw"]
        if value is not None and (type(value) is not int or isinstance(value, bool) or value < 0):
            _fail(f"invalid integer SMART {field} at {date_text}/{serial}")
    return item


def _panel_columns(connection: sqlite3.Connection) -> None:
    required = {"date", "serial_number", "model", "capacity_bytes", "failure", *(f"smart_{field}_raw" for field in RAW_FIELDS)}
    if not required.issubset(_table_columns(connection, "daily", "panel daily")):
        _fail("panel daily table is missing locked raw fields")


def _serial_rows(path: Path, cutoff: dt.date, callback: Callable[[], int] | None) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    connection = _open_readonly(path, "source panel")
    try:
        _panel_columns(connection)
        query = (
            "SELECT date,serial_number,model,capacity_bytes,failure," +
            ",".join(f"smart_{field}_raw" for field in RAW_FIELDS) +
            " FROM daily WHERE date<=? ORDER BY serial_number,date"
        )
        try:
            rows = connection.execute(query, (cutoff.isoformat(),))
        except sqlite3.Error as exc:
            raise PanelReferenceError("cannot stream source panel rows") from exc
        for serial, grouped in itertools.groupby(rows, key=lambda row: str(row["serial_number"])):
            by_date: dict[str, dict[str, Any]] = {}
            for raw in grouped:
                _check_progress(callback)
                item = _validate_raw_row(raw)
                date_text = str(item["date"])
                if date_text in by_date:
                    _fail(f"duplicate panel serial/date: {date_text}/{serial}")
                by_date[date_text] = item
            yield serial, [by_date[key] for key in sorted(by_date)]
    finally:
        connection.close()


def _classify_rows(rows: list[dict[str, Any]], lo: dt.date, hi: dt.date, source_end: dt.date) -> Iterator[dict[str, Any]]:
    by_date = {str(row["date"]): row for row in rows}
    first_failure = next((row for row in rows if row["failure"] == 1), None)
    first_failure_date = str(first_failure["date"]) if first_failure is not None else None
    for row in rows:
        decision = _parse_date(row["date"], "decision date")
        if decision < lo or decision > hi:
            continue
        if first_failure_date is not None and row["date"] > first_failure_date:
            status, label, history_count, future_count = "post_failure", None, 0, 0
        elif row["failure"] == 1:
            status, label, history_count, future_count = "same_day_failure", None, 0, 0
        else:
            history_count = sum(
                (decision - dt.timedelta(days=offset)).isoformat() in by_date
                for offset in range(HISTORY_DAYS)
            )
            future_seen = [
                (decision + dt.timedelta(days=offset)).isoformat() in by_date
                for offset in range(1, HORIZON_DAYS + 1)
            ]
            future_count = sum(future_seen)
            if history_count < MIN_HISTORY:
                status, label = "history_insufficient", None
            elif first_failure_date is not None and decision < _parse_date(first_failure_date, "first failure date") <= decision + dt.timedelta(days=HORIZON_DAYS):
                failure_offset = (_parse_date(first_failure_date, "first failure date") - decision).days
                status = "positive_observed" if all(future_seen[: max(0, failure_offset - 1)]) else "positive_with_gap"
                label = 1
            elif all(future_seen):
                status, label = "negative_observed", 0
            elif decision + dt.timedelta(days=HORIZON_DAYS) > source_end:
                status, label = "end_censored", None
            else:
                status, label = "gap_or_exit_censored", None
        if status not in KNOWN_LABEL_STATUSES:
            _fail(f"unknown label status: {status}")
        yield {
            "run_id": None,
            "decision_date": row["date"],
            "serial_number": row["serial_number"],
            "model": row["model"],
            "capacity_bytes": row["capacity_bytes"],
            "first_failure_date": first_failure_date,
            "label": label,
            "status": status,
            "eligible": int(status in {"positive_observed", "positive_with_gap", "negative_observed", "end_censored", "gap_or_exit_censored"}),
            "history_observations": history_count,
            "future_observations": future_count,
        }


def _label_db_rows(path: Path, run_id: str) -> Iterator[dict[str, Any]]:
    connection = _open_readonly(path, "label database")
    try:
        columns = _table_columns(connection, "label_flow", "label flow")
        if columns != set(LABEL_COLUMNS):
            _fail(f"label flow columns differ from the locked schema: {sorted(columns)}")
        try:
            cursor = connection.execute("SELECT * FROM label_flow WHERE run_id=? ORDER BY serial_number,decision_date", (run_id,))
            for row in cursor:
                yield dict(row)
        except sqlite3.Error as exc:
            raise PanelReferenceError("cannot stream label flow") from exc
    finally:
        connection.close()


def _compare_values(actual: object, expected: object, label: str) -> None:
    if actual is None or expected is None:
        if actual != expected:
            _fail(f"{label} nullness differs: actual={actual!r}, expected={expected!r}")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(actual) is not type(expected) or actual != expected:
            _fail(f"{label} differs: actual={actual!r}, expected={expected!r}")
        return
    if type(expected) is int or type(actual) is int:
        if type(actual) is not int or type(expected) is not int or actual != expected:
            _fail(f"{label} differs: actual={actual!r}, expected={expected!r}")
        return
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            actual_number, expected_number = float(actual), float(expected)
        except (TypeError, ValueError) as exc:
            raise PanelReferenceError(f"{label} is not numeric") from exc
        if not math.isfinite(actual_number) or not math.isfinite(expected_number) or not math.isclose(actual_number, expected_number, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            _fail(f"{label} differs: actual={actual_number!r}, expected={expected_number!r}")
    elif type(actual) is not type(expected) or actual != expected:
        _fail(f"{label} differs: actual={actual!r}, expected={expected!r}")


def _compare_label_row(actual: Mapping[str, Any], expected: Mapping[str, Any], run_id: str) -> None:
    expected = dict(expected)
    expected["run_id"] = run_id
    for name in LABEL_COLUMNS:
        _compare_values(actual.get(name), expected.get(name), f"label_flow {expected['serial_number']}/{expected['decision_date']} {name}")


def _compare_label_flow(path: Path, source: Path, run_id: str, lo: dt.date, hi: dt.date, cutoff: dt.date, callback: Callable[[], int] | None) -> dict[str, int]:
    expected_count = 0
    actual_iter = _label_db_rows(path, run_id)
    actual = iter(actual_iter)
    try:
        for serial, rows in _serial_rows(source, cutoff, callback):
            for expected in _classify_rows(rows, lo, hi, cutoff):
                _check_progress(callback)
                try:
                    row = next(actual)
                except StopIteration:
                    _fail(f"label flow is missing {serial}/{expected['decision_date']}")
                _compare_label_row(row, expected, run_id)
                expected_count += 1
        try:
            extra = next(actual)
        except StopIteration:
            extra = None
        if extra is not None:
            _fail(f"label flow has an unexpected row: {extra.get('serial_number')}/{extra.get('decision_date')}")
    finally:
        close = getattr(actual_iter, "close", None)
        if close is not None:
            close()
    return {"expected_rows": expected_count, "actual_rows": expected_count}


def _feature_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "decision_date": row["date"],
        "serial_number": row["serial_number"],
        "model": row["model"],
    }
    for field in RAW_FIELDS:
        raw = row[f"smart_{field}_raw"]
        missing = int(raw is None)
        prefix = f"smart_{field}"
        if field == 188:
            values[f"{prefix}_current_nonzero"] = None if missing else int(raw > 0)
            values[f"{prefix}_current_missing"] = missing
        else:
            values[f"{prefix}_current_missing"] = missing
            values[f"{prefix}_current_log1p"] = None if missing else math.log1p(raw)
            if field != 9:
                values[f"{prefix}_current_nonzero"] = None if missing else int(raw > 0)
    values["smart_nonzero_signal_count"] = sum(
        int(row[f"smart_{field}_raw"] is not None and row[f"smart_{field}_raw"] > 0)
        for field in NONZERO_FIELDS
    )
    values["tie_break_sha256"] = hashlib.sha256(
        f"drive-v1|20260912|{row['date']}|{row['serial_number']}".encode("utf-8")
    ).hexdigest()
    return values


def _source_days(path: Path, first: dt.date, end: dt.date, callback: Callable[[], int] | None) -> Iterator[tuple[dt.date, dict[str, dict[str, Any]]]]:
    connection = _open_readonly(path, "source panel")
    try:
        _panel_columns(connection)
        selected = "date,serial_number,model,capacity_bytes,failure," + ",".join(f"smart_{field}_raw" for field in RAW_FIELDS)
        day = first
        while day <= end:
            _check_progress(callback)
            try:
                rows = connection.execute(f"SELECT {selected} FROM daily WHERE date=? ORDER BY serial_number", (day.isoformat(),))
            except sqlite3.Error as exc:
                raise PanelReferenceError("cannot stream source panel day") from exc
            current: dict[str, dict[str, Any]] = {}
            for raw in rows:
                _check_progress(callback)
                item = _validate_raw_row(raw)
                serial = str(item["serial_number"])
                if serial in current:
                    _fail(f"duplicate panel serial/date: {day.isoformat()}/{serial}")
                current[serial] = item
            yield day, current
            day += dt.timedelta(days=1)
    finally:
        connection.close()


def _expected_features_and_calendar(path: Path, lo: dt.date, hi: dt.date, callback: Callable[[], int] | None) -> tuple[Iterator[dict[str, Any]], list[dict[str, int | str]]]:
    first = lo - dt.timedelta(days=HISTORY_DAYS - 1)
    source_connection = _open_readonly(path, "source panel")
    try:
        _panel_columns(source_connection)
        failed = {
            str(row[0]) for row in source_connection.execute("SELECT DISTINCT serial_number FROM daily WHERE failure=1 AND date<?", (first.isoformat(),))
        }
    except sqlite3.Error as exc:
        source_connection.close()
        raise PanelReferenceError("cannot read source failure history") from exc
    finally:
        source_connection.close()

    def generate() -> Iterator[dict[str, Any]]:
        window: deque[set[str]] = deque()
        counts: Counter[str] = Counter()
        for day, current in _source_days(path, first, hi, callback):
            observed = {serial for serial, row in current.items() if row["model"] == MODEL}
            counts.update(observed)
            window.append(observed)
            if len(window) > HISTORY_DAYS:
                counts.subtract(window.popleft())
                counts += Counter()
            failed.update(serial for serial, row in current.items() if row["failure"] == 1)
            if day < lo:
                continue
            eligible = [serial for serial in sorted(observed) if serial not in failed and counts[serial] >= MIN_HISTORY]
            for serial in eligible:
                yield _feature_from_row(current[serial])

    calendar: list[dict[str, int | str]] = []
    window: deque[set[str]] = deque()
    counts: Counter[str] = Counter()
    failed_for_calendar: set[str] = set()
    initial_connection = _open_readonly(path, "source panel")
    try:
        _panel_columns(initial_connection)
        failed_for_calendar.update(
            str(row[0]) for row in initial_connection.execute(
                "SELECT DISTINCT serial_number FROM daily WHERE failure=1 AND date<?",
                (first.isoformat(),),
            )
        )
    except sqlite3.Error as exc:
        raise PanelReferenceError("cannot read source failure history") from exc
    finally:
        initial_connection.close()
    # The separate iterator below keeps the calendar construction explicit and
    # deterministic; the feature generator is consumed independently so that
    # each comparison still uses a bounded serial/day stream.
    connection = _open_readonly(path, "source panel")
    try:
        _panel_columns(connection)
        selected = "date,serial_number,model,capacity_bytes,failure," + ",".join(f"smart_{field}_raw" for field in RAW_FIELDS)
        day = first
        while day <= hi:
            _check_progress(callback)
            current: dict[str, dict[str, Any]] = {}
            for raw in connection.execute(f"SELECT {selected} FROM daily WHERE date=? ORDER BY serial_number", (day.isoformat(),)):
                item = _validate_raw_row(raw)
                serial = str(item["serial_number"])
                if serial in current:
                    _fail(f"duplicate panel serial/date: {day.isoformat()}/{serial}")
                current[serial] = item
            observed = {serial for serial, row in current.items() if row["model"] == MODEL}
            counts.update(observed)
            window.append(observed)
            if len(window) > HISTORY_DAYS:
                counts.subtract(window.popleft())
                counts += Counter()
            failed_for_calendar.update(serial for serial, row in current.items() if row["failure"] == 1)
            if day >= lo:
                history_ready = sum(counts[serial] >= MIN_HISTORY for serial in observed)
                eligible = sum(serial not in failed_for_calendar and counts[serial] >= MIN_HISTORY for serial in observed)
                calendar.append({
                    "decision_date": day.isoformat(),
                    "observed_rows": len(current),
                    "model_rows": len(observed),
                    "failed_model_rows": sum(serial in failed_for_calendar for serial in observed),
                    "history_ready_rows": history_ready,
                    "eligible_rows": eligible,
                })
            day += dt.timedelta(days=1)
    finally:
        connection.close()
    return generate(), calendar


def _feature_db_rows(path: Path) -> Iterator[dict[str, Any]]:
    connection = _open_readonly(path, "feature database")
    try:
        columns = _table_columns(connection, "feature_rows", "feature rows")
        if columns != set(FEATURE_COLUMNS):
            _fail(f"feature rows columns differ from locked schema: {sorted(columns)}")
        try:
            for row in connection.execute("SELECT * FROM feature_rows ORDER BY decision_date,serial_number"):
                yield dict(row)
        except sqlite3.Error as exc:
            raise PanelReferenceError("cannot stream feature rows") from exc
    finally:
        connection.close()


def _compare_features(path: Path, source: Path, lo: dt.date, hi: dt.date, callback: Callable[[], int] | None) -> dict[str, Any]:
    expected_iter, calendar = _expected_features_and_calendar(source, lo, hi, callback)
    actual_iter = _feature_db_rows(path)
    actual = iter(actual_iter)
    expected_count = 0
    try:
        for expected in expected_iter:
            _check_progress(callback)
            try:
                row = next(actual)
            except StopIteration:
                _fail(f"feature rows are missing {expected['serial_number']}/{expected['decision_date']}")
            for name in FEATURE_COLUMNS:
                actual_value = row.get(name)
                expected_value = expected.get(name)
                # Feature flags are declared REAL in the production table, so
                # SQLite returns exact 0/1 values as floats.  Preserve strict
                # integer semantics everywhere else while normalising this
                # storage-only representation.
                if name in FEATURE_FLAG_COLUMNS and type(expected_value) is int and type(actual_value) is float and actual_value.is_integer():
                    actual_value = int(actual_value)
                _compare_values(actual_value, expected_value, f"feature_rows {expected['serial_number']}/{expected['decision_date']} {name}")
            expected_count += 1
        try:
            extra = next(actual)
        except StopIteration:
            extra = None
        if extra is not None:
            _fail(f"feature rows have an unexpected row: {extra.get('serial_number')}/{extra.get('decision_date')}")
    finally:
        close = getattr(actual_iter, "close", None)
        if close is not None:
            close()

    connection = _open_readonly(path, "feature database")
    try:
        columns = _table_columns(connection, "qualification_calendar", "qualification calendar")
        if columns != set(CALENDAR_COLUMNS):
            _fail(f"qualification calendar columns differ from locked schema: {sorted(columns)}")
        rows = [dict(row) for row in connection.execute("SELECT * FROM qualification_calendar ORDER BY decision_date")]
    except sqlite3.Error as exc:
        raise PanelReferenceError("cannot read qualification calendar") from exc
    finally:
        connection.close()
    if rows != calendar:
        _fail("qualification calendar differs from independent source replay")
    return {"expected_rows": expected_count, "actual_rows": expected_count, "calendar_days": len(calendar), "calendar": calendar}


def _check_derived_metadata(path: Path, label: str, source_sha: str, *, feature: bool) -> dict[str, str]:
    connection = _open_readonly(path, label)
    try:
        metadata = _metadata(connection, label)
    finally:
        connection.close()
    if metadata.get("status") != "complete":
        _fail(f"{label} is not complete")
    try:
        hashes = json.loads(metadata.get("source_panel_sha256", ""))
    except json.JSONDecodeError as exc:
        raise PanelReferenceError(f"{label} source hash metadata is invalid") from exc
    if hashes != [source_sha]:
        _fail(f"{label} is bound to a different source panel")
    if feature:
        try:
            columns = json.loads(metadata.get("feature_columns", ""))
        except json.JSONDecodeError as exc:
            raise PanelReferenceError("feature database column metadata is invalid") from exc
        if columns != list(LOCKED_FEATURES):
            _fail("feature database column metadata differs from locked whitelist")
    return metadata


def verify_training_reference(
    panel_path: Path | str,
    label_db: Path | str,
    feature_db: Path | str,
    *,
    run_id: str,
    start: str,
    end: str,
    cutoff: str,
    progress_callback: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Replay a training slice and require exact derived-table agreement."""
    if type(run_id) is not str or not run_id:
        _fail("panel reference run_id is invalid")
    lo, hi, source_end = _validate_arguments(start, end, cutoff)
    source = _bound(panel_path, "reference source panel")
    labels = _bound(label_db, "reference label database")
    features = _bound(feature_db, "reference feature database")
    _check_progress(progress_callback)
    source_facts = artifact_facts(source)
    source_sha = str(source_facts["sha256"])
    _check_derived_metadata(labels, "label database", source_sha, feature=False)
    _check_derived_metadata(features, "feature database", source_sha, feature=True)
    label_result = _compare_label_flow(labels, source, run_id, lo, hi, source_end, progress_callback)
    feature_result = _compare_features(features, source, lo, hi, progress_callback)
    _check_progress(progress_callback)
    current_source_facts = artifact_facts(source)
    if current_source_facts != source_facts:
        _fail("source panel changed during reference replay")
    return {
        "status": "pass",
        "version": REFERENCE_VERSION,
        "run_id": run_id,
        "date_range": {"start": start, "end": end, "cutoff": cutoff},
        "source_panel": source_facts,
        "labels": label_result,
        "features": {
            "expected_rows": feature_result["expected_rows"],
            "actual_rows": feature_result["actual_rows"],
            "calendar_days": feature_result["calendar_days"],
        },
        "checks": {
            "source_schema_and_values": True,
            "label_flow_replay": True,
            "feature_replay": True,
            "qualification_calendar": True,
            "source_unchanged": True,
        },
    }


__all__ = ["ABS_TOL", "REL_TOL", "PanelReferenceError", "REFERENCE_VERSION", "verify_training_reference"]
