"""Run the small, offline end-to-end replay fixture."""

from __future__ import annotations

import argparse
import csv
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from pipeline.reproducible.runtime_context import project_root
ROOT = project_root()
FIXTURE_ROOT = ROOT / "examples/small_replay"
SOURCE_CSV = FIXTURE_ROOT / "daily.csv"
MODEL_JSON = FIXTURE_ROOT / "current_lr.json"
FIXTURE_MANIFEST = FIXTURE_ROOT / "manifest.json"
EXPECTED_JSON = FIXTURE_ROOT / "expected.json"
MODEL_DATABASE_SHA256 = "f112323774527887686a6471296af036415e45e751416f4a2e4a3574cf16d3c4"
MODEL = "ST4000DM000"
START = dt.date(2023, 1, 15)
END = dt.date(2023, 1, 28)
DATASET_END = dt.date(2023, 2, 4)
EVENT_START = dt.date(2023, 1, 22)
EVENT_END = dt.date(2023, 1, 29)
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
BUDGET_DENOMINATOR = 1000
COOLDOWN_DAYS = 7
MAX_INPUT_ROWS = 1000
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_RSS_BYTES = 256 * 1024 * 1024
SERIALS = tuple(f"demo-{letter}" for letter in "ABCDEFGHIJKL")
SMART_FIELDS = (5, 9, 187, 188, 197, 198)
SOURCE_COLUMNS = ("date", "serial_number", "model", "capacity_bytes", "failure") + tuple(
    f"smart_{field}_raw" for field in SMART_FIELDS
)
FEATURE_SOURCE_COLUMNS = ("date", "serial_number", "model", "failure") + tuple(
    f"smart_{field}_raw" for field in SMART_FIELDS
)
METHODS = ("current_lr", "smart_nonzero")


class ReplayError(RuntimeError):
    """The fixture or one of its execution boundaries is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_input_sha256() -> dict[str, str]:
    """Return the files whose bytes define this replay."""

    return {
        "daily": sha256(SOURCE_CSV),
        "model": sha256(MODEL_JSON),
        "expected": sha256(EXPECTED_JSON),
        "fixture_manifest": sha256(FIXTURE_MANIFEST),
    }


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@contextlib.contextmanager
def _time_limit(seconds: float | None):
    """Install a process-local wall-clock guard for the public CLI."""

    if seconds is None:
        yield
        return
    if not math.isfinite(seconds) or seconds <= 0:
        raise ReplayError("timeout-seconds must be a positive finite number")
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _timeout_handler(_signum: int, _frame: Any) -> None:
        raise ReplayError(f"replay exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON must be an object: {path}")
    return value


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"non-numeric {field}") from exc
    if not math.isfinite(number):
        raise ReplayError(f"non-finite {field}")
    return number


def _load_fixture_manifest() -> dict[str, Any]:
    if not FIXTURE_MANIFEST.is_file() or not EXPECTED_JSON.is_file():
        raise ReplayError("fixture manifest or expected checks are missing")
    manifest = _json_read(FIXTURE_MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("scope") != "small_replay_fixture_v1":
        raise ReplayError("fixture manifest scope differs")
    bound_files = {
        "daily": SOURCE_CSV,
        "model": MODEL_JSON,
        "expected": EXPECTED_JSON,
    }
    bound_sha = manifest.get("sha256")
    if not isinstance(bound_sha, dict):
        raise ReplayError("fixture manifest has no file SHA map")
    for key, path in bound_files.items():
        if not path.is_file():
            raise ReplayError(f"fixture file missing: {path}")
        actual = sha256(path)
        if actual != bound_sha.get(key):
            raise ReplayError(f"fixture SHA mismatch for {key}: {actual}")
    source_code = manifest.get("code_sha256")
    code_files = {
        "run_small_replay": CODE_ROOT / "tools/run_small_replay.py",
        "build_feature_replay": CODE_ROOT / "pipeline/build_feature_replay.py",
        "labeling": CODE_ROOT / "pipeline/labeling.py",
        "replay_selection": CODE_ROOT / "pipeline/replay_selection.py",
    }
    if not isinstance(source_code, dict):
        raise ReplayError("fixture manifest has no code SHA map")
    for key, path in code_files.items():
        if not path.is_file() or sha256(path) != source_code.get(key):
            actual = sha256(path) if path.is_file() else "missing"
            raise ReplayError(f"code SHA mismatch for {key}: {actual}")
    return manifest


def _load_model(path: Path | None = None) -> dict[str, Any]:
    """Load a frozen current-LR payload from the caller's bound path."""
    model_path = MODEL_JSON if path is None else Path(path)
    model = _json_read(model_path)
    if model.get("schema_version") != 1 or model.get("model") != "current_lr":
        raise ReplayError("small replay model is not current_lr v1")
    if model.get("source_database_sha256") != MODEL_DATABASE_SHA256:
        raise ReplayError("small replay model source database SHA differs")
    features = model.get("features")
    if not isinstance(features, list) or len(features) != 16:
        raise ReplayError("small replay model must contain 16 features")
    positions = [item.get("position") for item in features if isinstance(item, dict)]
    if positions != list(range(16)):
        raise ReplayError("small replay model feature positions are not contiguous")
    names = []
    for item in features:
        if not isinstance(item, dict):
            raise ReplayError("small replay model feature row is not an object")
        name = item.get("feature_name")
        if not isinstance(name, str) or name in names:
            raise ReplayError("small replay model feature names are not unique")
        names.append(name)
        scale = _finite(item.get("standardization_scale"), f"scale {name}")
        if scale <= 0:
            raise ReplayError(f"small replay model scale is not positive: {name}")
        for field in ("imputation_mean", "standardization_mean", "coefficient"):
            _finite(item.get(field), f"{field} {name}")
    _finite(model.get("intercept"), "intercept")
    return model


def _iter_source(path: Path, *, strict: bool = True) -> Iterable[dict[str, Any]]:
    """Yield validated source rows without retaining the whole CSV in memory."""
    row_count = 0
    serials: set[str] = set()
    first_date: str | None = None
    last_date: str | None = None
    previous_key: tuple[str, str] | None = None
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ReplayError(f"cannot open fixture CSV: {exc}") from exc
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ReplayError("fixture CSV columns or order differ")
        for line_number, raw in enumerate(reader, start=2):
            if strict and row_count >= MAX_INPUT_ROWS:
                raise ReplayError(f"fixture has more than {MAX_INPUT_ROWS} data rows")
            try:
                day = dt.date.fromisoformat(str(raw["date"]))
            except ValueError as exc:
                raise ReplayError(f"invalid date on CSV line {line_number}") from exc
            serial = str(raw["serial_number"])
            if strict and serial not in SERIALS:
                raise ReplayError(f"unexpected serial on CSV line {line_number}: {serial}")
            if str(raw["model"]) != MODEL:
                raise ReplayError(f"unexpected model on CSV line {line_number}")
            key = (day.isoformat(), serial)
            if previous_key is not None and key <= previous_key:
                raise ReplayError(f"fixture rows are not strictly date/serial ordered on line {line_number}")
            previous_key = key
            try:
                capacity = int(str(raw["capacity_bytes"]))
                failure = int(str(raw["failure"]))
            except ValueError as exc:
                raise ReplayError(f"invalid integer on CSV line {line_number}") from exc
            if capacity <= 0 or failure not in (0, 1):
                raise ReplayError(f"invalid capacity or failure on CSV line {line_number}")
            item: dict[str, Any] = {
                "date": day.isoformat(),
                "serial_number": serial,
                "model": MODEL,
                "capacity_bytes": capacity,
                "failure": failure,
            }
            for field in SMART_FIELDS:
                raw_value = raw[f"smart_{field}_raw"]
                if raw_value == "":
                    item[f"smart_{field}_raw"] = None
                else:
                    try:
                        number = int(str(raw_value))
                    except ValueError as exc:
                        raise ReplayError(f"invalid SMART{field} on CSV line {line_number}") from exc
                    if number < 0:
                        raise ReplayError(f"negative SMART{field} on CSV line {line_number}")
                    item[f"smart_{field}_raw"] = number
            row_count += 1
            serials.add(serial)
            first_date = item["date"] if first_date is None else min(first_date, item["date"])
            last_date = item["date"] if last_date is None else max(last_date, item["date"])
            yield item
    if strict and (row_count != 395 or serials != set(SERIALS)):
        raise ReplayError(f"fixture row or serial count differs: rows={row_count}, serials={len(serials)}")
    if row_count == 0:
        raise ReplayError("source CSV has no data rows")
    if strict and (first_date != "2023-01-01" or last_date != "2023-02-04"):
        raise ReplayError("fixture date range differs")


def _parse_source(path: Path, *, strict: bool = True) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = list(_iter_source(path, strict=strict))
    by_serial: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_serial.setdefault(str(row["serial_number"]), []).append(row)
    return rows, by_serial


class DecisionDayReader:
    """A fixed-day view; consumers cannot request a later date or a new view.

    This is an application interface boundary, not a Python security sandbox.
    """
    __slots__ = ("__connection", "__access", "__day")

    def __init__(self, connection, access, day):
        self.__day = dt.date.fromisoformat(day).isoformat()
        if self.__day != day:
            raise ReplayError("decision date must be canonical YYYY-MM-DD")
        self.__connection = connection
        self.__access = access

    def serials(self) -> tuple[str, ...]:
        # Only devices visible on this decision day can generate features.
        return tuple(row[0] for row in self.__connection.execute(
            "SELECT serial_number FROM daily WHERE date=? ORDER BY serial_number", (self.__day,)
        ))

    def rows_until(self, serial: str, date_text: str) -> list[dict[str, Any]]:
        requested = dt.date.fromisoformat(date_text).isoformat()
        if requested != date_text or requested > self.__day:
            raise ReplayError(f"read exceeds bound decision day {self.__day}: {date_text}")
        return _asof_rows(self.__connection, serial, requested, self.__access)


class AsOfReader:
    """Coordinator-owned factory; daily scorers receive DecisionDayReader."""

    def __init__(self, connection: sqlite3.Connection, access: list[dict[str, Any]]):
        self._connection = connection
        self._access = access

    def serials(self) -> tuple[str, ...]:
        return tuple(
            str(row[0]) for row in self._connection.execute(
                "SELECT DISTINCT serial_number FROM daily ORDER BY serial_number"
            ).fetchall()
        )

    def rows_until(self, serial: str, date_text: str) -> list[dict[str, Any]]:
        return _asof_rows(self._connection, serial, date_text, self._access)

    def for_day(self, day: str) -> DecisionDayReader:
        return DecisionDayReader(self._connection, self._access, day)


def _source_connection(rows: Iterable[Mapping[str, Any]], access: list[dict[str, Any]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    columns = ", ".join(f"{name} {'INTEGER' if name in {'capacity_bytes', 'failure'} or name.startswith('smart_') else 'TEXT'}" for name in SOURCE_COLUMNS)
    connection.execute(f"CREATE TABLE daily ({columns}, PRIMARY KEY (date, serial_number))")
    placeholders = ",".join("?" for _ in SOURCE_COLUMNS)
    connection.executemany(
        f"INSERT INTO daily ({','.join(SOURCE_COLUMNS)}) VALUES ({placeholders})",
        [tuple(row[name] for name in SOURCE_COLUMNS) for row in rows],
    )
    connection.commit()
    connection.set_trace_callback(lambda statement: access.append({"phase": "sql", "statement": statement}))
    return connection


def _asof_reader(connection: sqlite3.Connection, access: list[dict[str, Any]]) -> AsOfReader:
    return AsOfReader(connection, access)


def _asof_rows(connection: sqlite3.Connection, serial: str, date_text: str, access: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = (
        "SELECT date,serial_number,model,failure,"
        + ",".join(f"smart_{field}_raw" for field in SMART_FIELDS)
        + " FROM daily WHERE serial_number=? AND date<=? ORDER BY date"
    )
    result = [dict(row) for row in connection.execute(query, (serial, date_text)).fetchall()]
    access.append(
        {
            "phase": "score",
            "serial_number": serial,
            "decision_date": date_text,
            "max_date": max((row["date"] for row in result), default=None),
            "row_count": len(result),
            "sql": query,
        }
    )
    if any(str(row["date"]) > date_text for row in result):
        raise ReplayError(f"as-of query returned a future row: {serial}/{date_text}")
    return result


def _score(values: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    total = _finite(model["intercept"], "intercept")
    for item in model["features"]:
        name = str(item["feature_name"])
        value = values.get(name)
        if value is None:
            value = item["imputation_mean"]
        number = _finite(value, name)
        transformed = (number - float(item["standardization_mean"])) / float(item["standardization_scale"])
        total += transformed * float(item["coefficient"])
    if not math.isfinite(total):
        raise ReplayError("non-finite score")
    return total


def _feature_values(
    rows: list[dict[str, Any]],
    date_text: str,
    *,
    allow_smart_decreases: bool = False,
) -> dict[str, Any] | None:
    from pipeline.build_feature_replay import FEATURE_COLUMN_NAMES, feature_row_for_serial

    feature_row = feature_row_for_serial(
        rows, date_text, allow_smart_decreases=allow_smart_decreases
    )
    if feature_row is None:
        return None
    return dict(zip(FEATURE_COLUMN_NAMES, feature_row))


def _create_selection_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE features (decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (decision_date, serial_number));
        CREATE TABLE scores (model TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, score REAL NOT NULL, tie_break_sha256 TEXT NOT NULL, smart_nonzero_signal_count INTEGER NOT NULL, smart_187_signal INTEGER NOT NULL, PRIMARY KEY (model, decision_date, serial_number));
        CREATE TABLE daily (model TEXT NOT NULL, decision_date TEXT NOT NULL, eligible_count INTEGER NOT NULL, budget_k INTEGER NOT NULL, signal_count INTEGER NOT NULL, cooldown_excluded INTEGER NOT NULL, alerts_count INTEGER NOT NULL, PRIMARY KEY (model, decision_date));
        CREATE TABLE selections (model TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, score REAL NOT NULL, tie_break_sha256 TEXT NOT NULL, selected_rank INTEGER NOT NULL, PRIMARY KEY (model, decision_date, serial_number));
        CREATE INDEX features_serial_date ON features(serial_number,decision_date);
        CREATE INDEX scores_serial_date ON scores(serial_number,decision_date);
        CREATE INDEX selections_serial_date ON selections(serial_number,decision_date);
        CREATE TABLE access_log (phase TEXT NOT NULL, serial_number TEXT, decision_date TEXT, max_date TEXT, query_type TEXT NOT NULL);
        """
    )
    return connection


def _load_selection_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _json_payload(values: Mapping[str, Any]) -> str:
    return json.dumps({key: values[key] for key in values}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _score_day(view: DecisionDayReader, date_text: str, model: Mapping[str, Any],
               *, allow_smart_decreases: bool, requested_serials=None):
    """Daily scoring has no factory or unrestricted connection argument."""
    from pipeline.build_feature_replay import stable_tie_break
    requested = None if requested_serials is None else set(requested_serials)
    for serial in view.serials():
        if requested is not None and serial not in requested:
            continue
        features = _feature_values(view.rows_until(serial, date_text), date_text,
                                   allow_smart_decreases=allow_smart_decreases)
        if features is not None:
            features["score"] = _score(features, model)
            features["tie_break_sha256"] = stable_tie_break(date_text, serial)
            yield serial, features


def _score_and_select(source: sqlite3.Connection | AsOfReader, source_rows: list[dict[str, Any]], model: Mapping[str, Any], output_db: Path, access: list[dict[str, Any]], *, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from pipeline.build_feature_replay import stable_tie_break
    from pipeline.replay_selection import select_alerts

    connection = _create_selection_database(output_db)
    spec = dict(spec or {})
    start = dt.date.fromisoformat(str(spec.get("score_start", START.isoformat())))
    end = dt.date.fromisoformat(str(spec.get("score_end", END.isoformat())))
    horizon = int(spec.get("horizon_days", HORIZON))
    requested_serials = spec.get("serials")
    if requested_serials is None:
        serials = () if isinstance(source, AsOfReader) else tuple(
            sorted({str(row["serial_number"]) for row in source_rows})
        )
    else:
        serials = tuple(str(item) for item in requested_serials)
    if not serials and not isinstance(source, AsOfReader):
        raise ReplayError("source contains no serials")
    allow_smart_decreases = bool(spec.get("allow_smart_decreases", False))
    outcome_cutoff = dt.date.fromisoformat(str(spec.get("outcome_cutoff", (end + dt.timedelta(days=horizon)).isoformat())))
    event_start = dt.date.fromisoformat(str(spec.get("event_start", (start + dt.timedelta(days=horizon)).isoformat())))
    event_end = dt.date.fromisoformat(str(spec.get("event_end", (end + dt.timedelta(days=1)).isoformat())))
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", [("phase", "scoring"), ("model", "current_lr"), ("methods", json.dumps(METHODS)), ("start", start.isoformat()), ("end", end.isoformat()), ("outcome_cutoff", outcome_cutoff.isoformat()), ("event_start", event_start.isoformat()), ("event_end", event_end.isoformat()), ("horizon_days", str(horizon)), ("history_days", str(HISTORY_DAYS)), ("min_history", str(MIN_HISTORY)), ("budget_denominator", str(BUDGET_DENOMINATOR)), ("cooldown_days", str(COOLDOWN_DAYS)), ("allow_smart_decreases", str(bool(spec.get("allow_smart_decreases", False))).lower())])
    last_alerts = {method: {} for method in METHODS}
    compact_features = spec.get("feature_storage") == "current"
    model_names = {str(item["feature_name"]) for item in model["features"]}
    daily_counts: dict[str, int] = {}
    for offset in range((end - start).days + 1):
        date = start + dt.timedelta(days=offset)
        date_text = date.isoformat()
        count = 0
        if isinstance(source, AsOfReader):
            daily_features = _score_day(source.for_day(date_text), date_text, model,
                                        allow_smart_decreases=allow_smart_decreases,
                                        requested_serials=requested_serials)
        else:
            # Preserve the fixed demonstration's recorded query contract.
            legacy_features = []
            for serial in serials:
                features = _feature_values(_asof_rows(source, serial, date_text, access), date_text,
                                           allow_smart_decreases=allow_smart_decreases)
                if features is not None:
                    features["score"] = _score(features, model)
                    features["tie_break_sha256"] = stable_tie_break(date_text, serial)
                    legacy_features.append((serial, features))
            daily_features = legacy_features
        rows = []
        for serial, features in daily_features:
            if compact_features:
                features = {key: value for key, value in features.items()
                            if key in model_names or "current_" in key or "decrease" in key
                            or key in {"serial_number", "decision_date", "score", "tie_break_sha256",
                                       "smart_nonzero_signal_count", "smart_187_signal"}}
            rows.append(features)
            connection.execute("INSERT INTO features VALUES (?, ?, ?)", (date_text, serial, _json_payload(features)))
            count += 1
        daily_counts[date_text] = count
        for method in METHODS:
            last_alert = last_alerts[method]
            for row in rows:
                if method == "current_lr":
                    method_score = row["score"]
                elif method == "smart_nonzero":
                    method_score = row["smart_nonzero_signal_count"]
                else:
                    raise ReplayError(f"unregistered small replay method: {method}")
                connection.execute("INSERT OR IGNORE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)", (method, date_text, row["serial_number"], float(method_score), row["tie_break_sha256"], int(row["smart_nonzero_signal_count"]), int(row["smart_187_signal"])))
            selected, stats = select_alerts(rows, method, date, last_alert, denominator=BUDGET_DENOMINATOR, cooldown_days=COOLDOWN_DAYS)
            connection.execute("INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?, ?)", (method, date_text, stats["eligible_count"], stats["budget_k"], stats["signal_count"], stats["cooldown_excluded"], stats["alerts_count"]))
            for rank, row in enumerate(selected, start=1):
                score = row["score"] if method == "current_lr" else row["smart_nonzero_signal_count"]
                connection.execute("INSERT INTO selections VALUES (?, ?, ?, ?, ?, ?)", (method, date_text, row["serial_number"], float(score), row["tie_break_sha256"], rank))
    if not sum(daily_counts.values()) and not isinstance(source, AsOfReader):
        raise ReplayError("no eligible feature rows in small replay")
    for item in access:
        if item.get("phase") == "score" and item.get("serial_number") is not None:
            connection.execute("INSERT INTO access_log VALUES (?, ?, ?, ?, ?)", ("score", item["serial_number"], item["decision_date"], item["max_date"], "asof_daily"))
    connection.execute("UPDATE metadata SET value='complete' WHERE key='phase'")
    connection.commit()
    connection.close()
    access.append({"phase": "selection_closed", "selection_sha256": sha256(output_db)})
    return {"eligible_device_days": sum(daily_counts.values()), "decision_days": len(daily_counts)}


def _labels_after_close(source: sqlite3.Connection, serials: Iterable[str] | None, access: list[dict[str, Any]], *, spec: Mapping[str, Any] | None = None, retain_keys: set[tuple[str, str]] | None = None) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    from pipeline.labeling import classify_device_rows

    spec = dict(spec or {})
    start = dt.date.fromisoformat(str(spec.get("score_start", START.isoformat())))
    end = dt.date.fromisoformat(str(spec.get("score_end", END.isoformat())))
    dataset_end = dt.date.fromisoformat(str(spec.get("outcome_cutoff", DATASET_END.isoformat())))
    horizon = int(spec.get("horizon_days", HORIZON))
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    events: dict[str, dict[str, Any]] = {}
    if serials is None:
        serials = tuple(
            str(row[0]) for row in source.execute(
                "SELECT DISTINCT serial_number FROM daily ORDER BY serial_number"
            ).fetchall()
        )
    select_all = "SELECT date,serial_number,model,capacity_bytes,failure," + ",".join(f"smart_{field}_raw" for field in SMART_FIELDS) + " FROM daily WHERE serial_number=? ORDER BY date"
    for serial in serials:
        rows = [dict(row) for row in source.execute(select_all, (serial,)).fetchall()]
        access.append({"phase": "evaluation", "serial_number": serial, "query_type": "full_daily_after_selection_close", "row_count": len(rows), "sql": select_all})
        classified = classify_device_rows(rows, start=start.isoformat(), end=end.isoformat(), dataset_end=dataset_end.isoformat(), horizon_days=horizon, history_days=HISTORY_DAYS, min_history=MIN_HISTORY)
        failure_dates = sorted(dt.date.fromisoformat(row["date"]) for row in rows if int(row["failure"]) == 1)
        event_keys = set() if not failure_dates else {
            ((failure_dates[0] - dt.timedelta(days=lead)).isoformat(), serial)
            for lead in range(1, horizon + 1)
        }
        for row in classified:
            key = (row["decision_date"], serial)
            if retain_keys is None or key in retain_keys or key in event_keys:
                labels[key] = row
        event_start = dt.date.fromisoformat(str(spec.get("event_start", EVENT_START.isoformat())))
        event_end = dt.date.fromisoformat(str(spec.get("event_end", EVENT_END.isoformat())))
        if failure_dates and event_start <= failure_dates[0] <= event_end:
            events[serial] = {"event_key": serial, "first_failure_date": failure_dates[0].isoformat(), "opportunity": 0}
    return labels, events


def _validate_expected(expected: Mapping[str, Any], labels: Mapping[tuple[str, str], Mapping[str, Any]], features: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    checks = expected.get("decision_checks")
    if not isinstance(checks, list):
        raise ReplayError("expected checks have no decision_checks")
    checked: list[str] = []
    for item in checks:
        if not isinstance(item, dict):
            raise ReplayError("expected decision check is not an object")
        key = (str(item.get("date")), str(item.get("serial_number")))
        actual = labels.get(key)
        if actual is None or actual.get("status") != item.get("status") or actual.get("label") != item.get("label"):
            raise ReplayError(f"expected label check differs: {key}")
        checked.append(f"{key[1]}/{key[0]}")
    for item in expected.get("feature_checks", []):
        key = (str(item.get("date")), str(item.get("serial_number")))
        actual = features.get(key)
        if actual is None:
            raise ReplayError(f"expected feature row missing: {key}")
        for name, value in dict(item.get("values", {})).items():
            if actual.get(name) != value:
                raise ReplayError(f"expected feature value differs: {key}/{name}")
        checked.append(f"features:{key[1]}/{key[0]}")
    for item in expected.get("score_checks", []):
        key = (str(item.get("date")), str(item.get("serial_number")))
        actual = features.get(key)
        if actual is None or not math.isclose(float(actual.get("score")), float(item.get("score")), rel_tol=1e-12, abs_tol=1e-12):
            raise ReplayError(f"expected score differs: {key}")
        checked.append(f"score:{key[1]}/{key[0]}")
    expected_events = expected.get("main_event_serials")
    if sorted(str(item) for item in (expected_events or [])) != ["demo-A", "demo-B"]:
        raise ReplayError("expected main event set differs")
    return {"checks": checked, "main_event_serials": ["demo-A", "demo-B"]}


def _evaluate(output_db: Path, source: sqlite3.Connection, selection_sha: str, expected: Mapping[str, Any], access: list[dict[str, Any]], *, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    connection = _load_selection_readonly(output_db)
    try:
        if connection.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()[0] != "complete":
            raise ReplayError("selection database is not complete before evaluation")
        selection_rows = [dict(row) for row in connection.execute("SELECT * FROM selections ORDER BY model,decision_date,selected_rank")]
        feature_payloads = {}
        if expected.get("decision_checks") is not None:
            feature_payloads = {(str(row["decision_date"]), str(row["serial_number"])): json.loads(row["payload_json"]) for row in connection.execute("SELECT decision_date,serial_number,payload_json FROM features")}
            feature_keys = set(feature_payloads)
        else:
            feature_keys = {(str(row[0]), str(row[1])) for row in connection.execute("SELECT decision_date,serial_number FROM features")}
        eligible_by_method = {method: int(connection.execute("SELECT COALESCE(SUM(eligible_count),0) FROM daily WHERE model=?", (method,)).fetchone()[0]) for method in METHODS}
    finally:
        connection.close()
    access.append({"phase": "evaluation_selection_closed", "selection_sha256": selection_sha})
    spec = dict(spec or {})
    start = dt.date.fromisoformat(str(spec.get("score_start", START.isoformat())))
    end = dt.date.fromisoformat(str(spec.get("score_end", END.isoformat())))
    horizon = int(spec.get("horizon_days", HORIZON))
    event_start = dt.date.fromisoformat(str(spec.get("event_start", EVENT_START.isoformat())))
    event_end = dt.date.fromisoformat(str(spec.get("event_end", EVENT_END.isoformat())))
    serials = tuple(str(item) for item in spec.get("serials", ())) or None
    retain_keys = None if expected.get("decision_checks") is not None else {
        (str(row["decision_date"]), str(row["serial_number"])) for row in selection_rows
    }
    labels, events = _labels_after_close(source, serials, access, spec=spec, retain_keys=retain_keys)
    validation = (
        _validate_expected(expected, labels, feature_payloads)
        if expected.get("decision_checks") is not None
        else {"checks": [], "main_event_serials": sorted(events)}
    )
    for event in events.values():
        failure_day = dt.date.fromisoformat(str(event["first_failure_date"]))
        serial = str(event["event_key"])
        event["opportunity"] = int(
            any(
                key in feature_keys and labels.get(key, {}).get("eligible") == 1
                for key in (((failure_day - dt.timedelta(days=lead)).isoformat(), serial)
                            for lead in range(1, horizon + 1))
            )
        )
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in selection_rows:
        by_method[str(row["model"])].append(row)
    results: dict[str, Any] = {"schema_version": 1, "status": "pass", "scope": "synthetic_small_replay", "selection_sha256_before_evaluation": selection_sha, "methods": {}}
    for method in METHODS:
        chosen = by_method[method]
        known_hit = known_no_hit = unknown = 0
        hits_by_event: dict[str, list[tuple[dt.date, int]]] = {}
        for row in chosen:
            key = (str(row["decision_date"]), str(row["serial_number"]))
            label = labels.get(key)
            if label is None:
                raise ReplayError(f"missing label after selection close: {key}")
            if label["label"] == 1:
                known_hit += 1
            elif label["label"] == 0:
                known_no_hit += 1
            else:
                unknown += 1
            failure = label.get("first_failure_date")
            if failure and str(row["serial_number"]) in events:
                failure_day = dt.date.fromisoformat(failure)
                alert_day = dt.date.fromisoformat(str(row["decision_date"]))
                lead = (failure_day - alert_day).days
                if event_start <= failure_day <= event_end and 1 <= lead <= horizon:
                    hits_by_event.setdefault(str(row["serial_number"]), []).append((alert_day, lead))
        event_rows = []
        event_hits = 0
        for event_key in sorted(events):
            choices = sorted(hits_by_event.get(event_key, []))
            opportunity = int(events[event_key]["opportunity"])
            eligible_choices = choices if opportunity else []
            hit = int(bool(eligible_choices))
            event_hits += hit
            event_rows.append({"event_key": event_key, "first_failure_date": events[event_key]["first_failure_date"], "opportunity": opportunity, "hit": hit, "earliest_alert_date": eligible_choices[0][0].isoformat() if eligible_choices else None, "earliest_lead_days": eligible_choices[0][1] if eligible_choices else None})
        devices = {}
        for row in chosen:
            devices.setdefault(str(row["serial_number"]), 0)
            devices[str(row["serial_number"])] += 1
        first_alerts = sum(1 for count in devices.values() if count >= 1)
        later_alerts = sum(max(0, count - 1) for count in devices.values())
        event_total = len(events)
        event_opportunity_total = sum(int(event["opportunity"]) for event in events.values())
        results["methods"][method] = {
            "status": "pass",
            "eligible_device_days": eligible_by_method[method],
            "alerts": len(chosen),
            "known_hit_alerts": known_hit,
            "known_no_hit_alerts": known_no_hit,
            "unknown_alerts": unknown,
            "unknown_alert_ratio": unknown / len(chosen) if chosen else None,
            "precision_lower_bound": known_hit / len(chosen) if chosen else None,
            "precision_upper_bound": (known_hit + unknown) / len(chosen) if chosen else None,
            "event_total": event_total,
            "event_opportunity_total": event_opportunity_total,
            "event_hits": event_hits,
            "event_recall_at_opportunity": event_hits / event_opportunity_total if event_opportunity_total else None,
            "first_alerts": first_alerts,
            "later_alerts": later_alerts,
            "repeat_alert_devices": sum(1 for count in devices.values() if count >= 2),
            "event_summary": event_rows,
        }
    results["selection_sha256_after_evaluation"] = sha256(output_db)
    if results["selection_sha256_after_evaluation"] != selection_sha:
        raise ReplayError("selection database changed during evaluation")
    selection_close_positions = [index for index, item in enumerate(access) if item.get("phase") == "selection_closed"]
    evaluation_selection_close_positions = [index for index, item in enumerate(access) if item.get("phase") == "evaluation_selection_closed"]
    evaluation_positions = [index for index, item in enumerate(access) if item.get("phase") == "evaluation"]
    selection_closed_before_evaluation = bool(selection_close_positions and evaluation_selection_close_positions and evaluation_positions and max(selection_close_positions) < min(evaluation_selection_close_positions) < min(evaluation_positions))
    results["evaluation_opened_after_selection_close"] = selection_closed_before_evaluation
    results["evaluation_source_query_count"] = len(
        tuple(source.execute("SELECT DISTINCT serial_number FROM daily").fetchall())
        if serials is None else serials
    )
    results["expected_checks"] = validation
    score_access = [item for item in access if item.get("phase") == "score"]
    score_sql_bounded = bool(score_access) and all("date<=?" in str(item.get("sql", "")).replace(" ", "").lower() for item in score_access)
    results["access_checks"] = {
        "score_queries_have_date_upper_bound": score_sql_bounded and all(item.get("max_date") is None or item["max_date"] <= item["decision_date"] for item in score_access),
        "score_query_count": len(score_access),
        "evaluation_after_selection_close": selection_closed_before_evaluation,
        "selection_close_marker_count": len(selection_close_positions),
        "evaluation_selection_close_marker_count": len(evaluation_selection_close_positions),
        "evaluation_query_count": len(evaluation_positions),
    }
    return results


def _summary(results: Mapping[str, Any]) -> str:
    lines = [
        "# 小规模预测与评价复现",
        "",
        "这是 12 台虚构设备的离线机制演示，不是 Backblaze 子集，也不代表 Q3 研究成绩。评分使用冻结的 current_lr 参数；smart_nonzero 是登记的规则对照。",
        "",
        "| 方法 | 告警 | 已知命中 | 已知未命中 | 未知 | 主事件捕获 | 机会分母 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = results["methods"][method]
        lines.append(f"| {method} | {row['alerts']} | {row['known_hit_alerts']} | {row['known_no_hit_alerts']} | {row['unknown_alerts']} | {row['event_hits']} | {row['event_opportunity_total']} |")
    lines.extend(
        [
            "",
            "评分阶段只查询 date≤t 的记录；名单写入并关闭后才连接未来结局。未知结局没有填成负例。输出中的命中率仅用于检查这组手工数据的执行路径，不能与真实季度的事件召回比较。",
            "",
            "边界检查：A/B 是评价期内的两个主事件；C 的故障在主事件窗口外；D 有一天缺记录；E 提前退出；F 在一个决策日的六项 SMART 全缺失；G 在历史不足后才达到 12 次观测；H 在评分期前已经故障。",
            "",
        ]
    )
    return "\n".join(lines)


def _output_directory(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ReplayError("output directory must be inside the project")
    if os.path.lexists(str(candidate)):
        raise ReplayError(f"output directory already exists: {candidate}")
    try:
        resolved.mkdir(parents=False)
    except FileExistsError as exc:
        raise ReplayError(f"output directory was created concurrently: {resolved}") from exc
    return resolved


def _output_bytes(directory: Path) -> int:
    total = 0
    for item in directory.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError as exc:
            raise ReplayError(f"cannot inspect output size: {item}: {exc}") from exc
    return total


def _enforce_output_budget(directory: Path, additional_bytes: int = 0) -> None:
    total = _output_bytes(directory) + additional_bytes
    if total > MAX_OUTPUT_BYTES:
        raise ReplayError(f"output exceeds {MAX_OUTPUT_BYTES} bytes: {total}")


def _stage_output(fd: int, value: bytes) -> None:
    """Write and flush a staged output file before it is published."""

    with os.fdopen(fd, "wb") as stream:
        written = stream.write(value)
        if written != len(value):
            raise OSError(f"short write while staging output: {written}/{len(value)} bytes")
        stream.flush()
        os.fsync(stream.fileno())


def _write_new(path: Path, value: bytes) -> None:
    """Publish a new file without exposing a partially written final name."""

    if len(value) > MAX_OUTPUT_BYTES:
        raise ReplayError(f"output exceeds {MAX_OUTPUT_BYTES} bytes")
    _enforce_output_budget(path.parent, len(value))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        _stage_output(fd, value)
        try:
            # A hard link publishes the complete staged inode and fails if a
            # competing writer has already claimed the final name.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReplayError(f"output file already exists: {path}") from exc
        except OSError as exc:
            raise ReplayError(f"cannot publish output file: {path}: {exc}") from exc
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    else:
        temporary.unlink(missing_ok=True)


def _validate_completion_gates(evaluation: Mapping[str, Any]) -> None:
    """Reject a run unless the recorded access boundary is complete."""

    access_checks = evaluation.get("access_checks")
    expected_score_queries = len(SERIALS) * ((END - START).days + 1)
    selection_sha_before = evaluation.get("selection_sha256_before_evaluation")
    selection_sha_after = evaluation.get("selection_sha256_after_evaluation")
    required = (
        evaluation.get("status") == "pass",
        evaluation.get("evaluation_opened_after_selection_close") is True,
        isinstance(selection_sha_before, str)
        and len(selection_sha_before) == 64
        and selection_sha_before == selection_sha_after,
        isinstance(access_checks, Mapping),
        access_checks.get("score_queries_have_date_upper_bound") is True if isinstance(access_checks, Mapping) else False,
        access_checks.get("evaluation_after_selection_close") is True if isinstance(access_checks, Mapping) else False,
        access_checks.get("score_query_count") == expected_score_queries if isinstance(access_checks, Mapping) else False,
        access_checks.get("evaluation_query_count") == len(SERIALS) if isinstance(access_checks, Mapping) else False,
        access_checks.get("selection_close_marker_count") == 1 if isinstance(access_checks, Mapping) else False,
        access_checks.get("evaluation_selection_close_marker_count") == 1 if isinstance(access_checks, Mapping) else False,
    )
    if not all(required):
        raise ReplayError("completion access gates failed")


def run(output: Path, *, timeout_seconds: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    manifest = _load_fixture_manifest()
    input_sha_before = _bound_input_sha256()
    model = _load_model()
    expected = _json_read(EXPECTED_JSON)
    source_rows, by_serial = _parse_source(SOURCE_CSV)
    access: list[dict[str, Any]] = []
    source = _source_connection(source_rows, access)
    selection_db = output / "selection.sqlite"
    try:
        _score_and_select(source, source_rows, model, selection_db, access)
        _enforce_output_budget(output)
        selection_sha = sha256(selection_db)
        evaluation = _evaluate(selection_db, source, selection_sha, expected, access)
        input_sha_after = _bound_input_sha256()
        if input_sha_after != input_sha_before:
            raise ReplayError("bound input changed during replay")
        _validate_completion_gates(evaluation)
        evaluation["expected_checks_version"] = expected.get("schema_version")
        _write_new(output / "evaluation.json", (json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        _write_new(output / "summary.md", _summary(evaluation).encode("utf-8"))
        run_manifest = {
            "schema_version": 1,
            "status": "complete",
            "scope": "synthetic_small_replay",
            "fixture_manifest_sha256": input_sha_before["fixture_manifest"],
            "fixture_manifest": FIXTURE_MANIFEST.relative_to(ROOT).as_posix(),
            "input_sha256_before": input_sha_before,
            "input_sha256_after": input_sha_after,
            "selection_sha256_before_evaluation": selection_sha,
            "selection_sha256_after_evaluation": evaluation["selection_sha256_after_evaluation"],
            "selection_closed_before_evaluation": evaluation["evaluation_opened_after_selection_close"],
            "source_query_access": evaluation["access_checks"],
            "outputs": {name: sha256(output / name) for name in ("selection.sqlite", "evaluation.json", "summary.md")},
            "limits": {"max_input_rows": MAX_INPUT_ROWS, "max_output_bytes": MAX_OUTPUT_BYTES, "observed_rss_limit_bytes": MAX_RSS_BYTES, "rss_is_observation": True, "timeout_seconds": timeout_seconds, "network": False, "training": False},
            "expected_checks": evaluation["expected_checks"],
            "runtime_rss_bytes": rss_bytes(),
            "elapsed_seconds": time.monotonic() - started,
        }
        if run_manifest["runtime_rss_bytes"] > MAX_RSS_BYTES:
            raise ReplayError(f"observed RSS exceeds {MAX_RSS_BYTES} bytes")
        _enforce_output_budget(output)
        output_sizes = {name: (output / name).stat().st_size for name in ("selection.sqlite", "evaluation.json", "summary.md")}
        run_manifest["output_sizes_before_manifest"] = output_sizes
        run_manifest["output_bytes_before_manifest"] = sum(output_sizes.values())
        _write_new(output / "run_manifest.json", (json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        return run_manifest
    finally:
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline small HDD warning replay")
    parser.add_argument("--output", required=True, help="new project-local output directory")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="wall-clock limit for the replay")
    args = parser.parse_args(argv)
    output: Path | None = None
    try:
        output = _output_directory(args.output)
        with _time_limit(args.timeout_seconds):
            run(output, timeout_seconds=args.timeout_seconds)
        print(f"wrote {output.relative_to(ROOT)}")
    except (ReplayError, OSError, sqlite3.Error, ValueError) as exc:
        if output is not None:
            error_path = output / "error.json"
            try:
                with error_path.open("x", encoding="utf-8") as stream:
                    json.dump({"status": "failed", "error": str(exc)}, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except OSError:
                pass
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
