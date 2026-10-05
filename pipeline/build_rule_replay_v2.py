"""Build an immutable, independently versioned rule-replay result.

The v1 feature database is opened read-only.  This stage writes only
``rule_replay_train_v2.sqlite`` and keeps ranking/selection in the pure
``pipeline.replay_selection`` module so synthetic boundary tests can exercise
the budget and cooldown contract without SQLite or labels.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.replay_selection import METHODS, median, minimum_gap, quantile, select_alerts


MODEL = "ST4000DM000"
REPLAY_ID = "rule_replay_train_v2"
SCORE_START = dt.date(2023, 1, 15)
SCORE_END = dt.date(2023, 6, 23)
DATASET_END = dt.date(2023, 6, 30)
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
EVENT_START = SCORE_START + dt.timedelta(days=HORIZON)
EVENT_END = SCORE_END + dt.timedelta(days=1)
BUDGET_DENOMINATOR = 1000
COOLDOWN_DAYS = 7
LABEL_RUN_ID = "train_q1q2_verified_h7_v1"
LABEL_SPLIT = "train"
LABEL_HORIZON = 7
FEATURE_DB_DEFAULT = ROOT / "data/derived/features_train_v1.sqlite"
PANEL_DB_DEFAULT = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
OUTPUT_DB_DEFAULT = ROOT / "data/derived/rule_replay_train_v2.sqlite"
EVIDENCE_DEFAULT = ROOT / "evidence/q2/feature_replay_closeout_v1"
EXPECTED_FEATURE_ROWS = 2_860_459
EXPECTED_EVENT_TOTAL = 289
EXPECTED_EVENT_OPPORTUNITY = 286
MAX_OWNED_BYTES = 4 * 1024 * 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
SOFT_TIMEOUT_SECONDS = 60 * 60
SMART_FIELDS = (5, 9, 187, 188, 197, 198)
RESOURCE_CHECK_ROWS = 10_000
SQL_PROGRESS_OPS = 10_000
HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")


class BuildStopped(RuntimeError):
    """A preflight, provenance, resource, or QA gate stopped the stage."""


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
    seen: set[Path] = set()
    total = 0
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


def _require_hex64(value: object, name: str) -> str:
    text = str(value or "")
    if not HEX64.fullmatch(text):
        raise BuildStopped(f"{name} must be a non-empty SHA256 hex digest")
    return text


def _install_sql_progress_guard(
    connection: sqlite3.Connection,
    start_time: float,
) -> dict[str, str | None]:
    """Interrupt long SQLite statements when a hard stage gate is reached."""
    state: dict[str, str | None] = {"reason": None}

    def callback() -> int:
        if time.monotonic() - start_time >= SOFT_TIMEOUT_SECONDS:
            state["reason"] = "replay soft timeout reached in SQLite progress callback"
            return 1
        if _rss_max_bytes() >= MAX_RSS_BYTES:
            state["reason"] = "replay RSS limit reached in SQLite progress callback"
            return 1
        if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
            state["reason"] = "replay free-space limit reached in SQLite progress callback"
            return 1
        return 0

    connection.set_progress_handler(callback, SQL_PROGRESS_OPS)
    return state


def _clear_sql_progress_guard(connection: sqlite3.Connection) -> None:
    connection.set_progress_handler(None, 0)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _project_path(path: Path, field: str) -> Path:
    resolved = (ROOT / path if not path.is_absolute() else path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise BuildStopped(f"{field} escapes project root: {path}")
    return resolved


def _connect_immutable(path: Path) -> sqlite3.Connection:
    wal = path.with_name(path.name + "-wal")
    shm = path.with_name(path.name + "-shm")
    if wal.exists() and wal.stat().st_size:
        raise BuildStopped(f"refusing immutable read with non-empty WAL: {wal}")
    if shm.exists() and shm.stat().st_size:
        raise BuildStopped(f"refusing immutable read with non-empty SHM: {shm}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _meta(connection: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata")}


_REPLAY_TABLES = frozenset(
    {"metadata", "replay_runs", "replay_daily", "replay_alerts", "replay_event_summary", "replay_strata"}
)


def _read_replay_metadata(path: Path) -> dict[str, str]:
    """Read a replay file without running DDL or changing its bytes."""
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect_immutable(path)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = sorted(_REPLAY_TABLES - tables)
        if missing:
            raise BuildStopped(f"replay database is missing tables: {missing}")
        metadata = _meta(connection)
        if not metadata.get("replay_config_hash"):
            raise BuildStopped("replay database has no replay_config_hash")
        return metadata
    except sqlite3.DatabaseError as exc:
        raise BuildStopped(f"cannot inspect replay database safely: {path}") from exc
    finally:
        if connection is not None:
            connection.close()


def _validate_partial_config(path: Path, config_hash: str) -> None:
    """Validate every persisted partial-run binding before opening it for write."""
    metadata = _read_replay_metadata(path)
    if metadata.get("replay_config_hash") != config_hash:
        raise BuildStopped("partial v2 replay config does not match current preflight")
    connection: sqlite3.Connection | None = None
    try:
        connection = _connect_immutable(path)
        for row in connection.execute(
            "SELECT method,status,config_hash FROM replay_runs "
            "WHERE run_id=? ORDER BY method",
            (REPLAY_ID,),
        ):
            if row["config_hash"] != config_hash:
                raise BuildStopped(
                    "partial v2 replay contains a method with a mismatched config: "
                    f"{row['method']} ({row['status']})"
                )
    except sqlite3.DatabaseError as exc:
        raise BuildStopped(f"cannot inspect partial replay safely: {path}") from exc
    finally:
        if connection is not None:
            connection.close()


def _event_opportunities(panel: sqlite3.Connection) -> dict[str, dict]:
    """Stream one device at a time and independently derive event opportunities."""
    opportunities: dict[str, dict] = {}
    cursor = panel.execute(
        "SELECT serial_number,date,failure FROM daily WHERE model=? ORDER BY serial_number,date",
        (MODEL,),
    )
    current_serial: str | None = None
    rows: list[tuple[str, int]] = []

    def flush(serial: str | None, device_rows: list[tuple[str, int]]) -> None:
        if serial is None:
            return
        dates = [dt.date.fromisoformat(day) for day, _ in device_rows]
        failures = [day for (day, failed) in zip(dates, (flag for _, flag in device_rows)) if failed]
        if not failures:
            return
        first_failure = failures[0]
        if not EVENT_START <= first_failure <= EVENT_END:
            return
        observed = set(dates)
        candidate: list[dt.date] = []
        for offset in range(1, HORIZON + 1):
            decision = first_failure - dt.timedelta(days=offset)
            if not SCORE_START <= decision <= SCORE_END or decision not in observed:
                continue
            history = sum(decision - dt.timedelta(days=i) in observed for i in range(HISTORY_DAYS))
            if history >= MIN_HISTORY:
                candidate.append(decision)
        opportunities[serial] = {
            "event_key": serial,
            "first_failure_date": first_failure.isoformat(),
            "opportunity": int(bool(candidate)),
            "opportunity_dates": [day.isoformat() for day in sorted(candidate)],
        }

    for item in cursor:
        serial = str(item["serial_number"])
        if current_serial is None:
            current_serial = serial
        if serial != current_serial:
            flush(current_serial, rows)
            rows = []
            current_serial = serial
        rows.append((item["date"], int(item["failure"] or 0)))
    flush(current_serial, rows)
    return opportunities


def _label_for_alert(panel: sqlite3.Connection, date_text: str, serial: str) -> sqlite3.Row | None:
    return panel.execute(
        "SELECT status,label,first_failure_date FROM label_flow "
        "WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? AND serial_number=?",
        (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON, date_text, serial),
    ).fetchone()


def _feature_rows(feature: sqlite3.Connection, date_text: str) -> list[sqlite3.Row]:
    return feature.execute(
        "SELECT * FROM feature_rows WHERE decision_date=? ORDER BY tie_break_sha256,serial_number",
        (date_text,),
    ).fetchall()


def _feature_row_missing_bucket(row: sqlite3.Row) -> str:
    missing = any(int(row[f"smart_{field}_current_missing"]) for field in SMART_FIELDS)
    return "any_current_missing" if missing else "all_current_present"


def _strata_add(
    strata: dict[tuple[str, str], dict[str, int]],
    stratum_type: str,
    stratum: str,
    key: str,
    amount: int = 1,
) -> None:
    item = strata.setdefault(
        (stratum_type, stratum),
        {"eligible_device_days": 0, "alerts": 0, "known_hit_alerts": 0, "unknown_alerts": 0, "event_hits": 0},
    )
    item[key] += amount


def _ddl(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS replay_runs (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            feature_dictionary_hash TEXT NOT NULL,
            source_panel_hash TEXT NOT NULL,
            label_run_id TEXT NOT NULL,
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
            cooldown_signal_excluded INTEGER NOT NULL,
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
        CREATE TABLE IF NOT EXISTS replay_strata (
            run_id TEXT NOT NULL,
            method TEXT NOT NULL,
            stratum_type TEXT NOT NULL,
            stratum TEXT NOT NULL,
            eligible_device_days INTEGER NOT NULL,
            alerts INTEGER NOT NULL,
            known_hit_alerts INTEGER NOT NULL,
            unknown_alerts INTEGER NOT NULL,
            event_hits INTEGER NOT NULL,
            PRIMARY KEY(run_id, method, stratum_type, stratum)
        );
        """
    )


def _config_hash(
    feature_meta: dict[str, str],
    panel_meta: dict[str, str],
    replay_code_sha256: str,
    label_run: dict[str, str],
    selection_code_sha256: str,
) -> str:
    payload = {
        "replay_run_id": REPLAY_ID,
        "model": MODEL,
        "score_start": SCORE_START.isoformat(),
        "score_end": SCORE_END.isoformat(),
        "dataset_end": DATASET_END.isoformat(),
        "horizon": HORIZON,
        "history_days": HISTORY_DAYS,
        "min_history": MIN_HISTORY,
        "budget_denominator": BUDGET_DENOMINATOR,
        "cooldown_days": COOLDOWN_DAYS,
        "event_start": EVENT_START.isoformat(),
        "event_end": EVENT_END.isoformat(),
        "feature_dictionary_hash": feature_meta["feature_dictionary_hash"],
        "feature_config_hash": feature_meta["feature_config_hash"],
        "feature_code_at_build": feature_meta.get("feature_code_sha256"),
        "panel_manifest_hash": panel_meta.get("build_manifest_hash"),
        "label_run_id": LABEL_RUN_ID,
        "label_config_hash": label_run.get("config_hash"),
        "label_code_sha256": label_run.get("code_hash"),
        "label_source_manifest_hash": label_run.get("manifest_hash"),
        "replay_code_sha256": replay_code_sha256,
        "selection_code_sha256": selection_code_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _preflight(
    panel_db: Path,
    feature_db: Path,
    output_db: Path,
    evidence_root: Path,
    *,
    allow_existing_output: bool = False,
) -> tuple[sqlite3.Connection, sqlite3.Connection, dict, dict, str, dict]:
    if not panel_db.is_file() or not feature_db.is_file():
        raise BuildStopped("panel or feature database is missing")
    panel = _connect_immutable(panel_db)
    feature = _connect_immutable(feature_db)
    panel_meta = _meta(panel)
    feature_meta = _meta(feature)
    if panel_meta.get("build_status") != "panel_complete":
        raise BuildStopped("panel is not complete")
    if feature_meta.get("feature_status") != "complete":
        raise BuildStopped("feature database is not complete")
    if panel_meta.get("training_approval") != "false" or feature_meta.get("training_approval") != "false":
        raise BuildStopped("training_approval must remain false")
    if feature_meta.get("source_panel_manifest_hash") != panel_meta.get("build_manifest_hash"):
        raise BuildStopped("feature source panel hash does not match panel metadata")
    # Every provenance edge used by this replay must be an actual SHA256
    # digest.  Truthy placeholders or an empty legacy field are not bindings.
    for name, value in (
        ("panel.build_manifest_hash", panel_meta.get("build_manifest_hash")),
        ("panel.label_build_manifest_hash", panel_meta.get("label_build_manifest_hash")),
        ("panel.label_source_manifest_hash", panel_meta.get("label_source_manifest_hash")),
        ("feature.source_panel_manifest_hash", feature_meta.get("source_panel_manifest_hash")),
        ("feature.feature_dictionary_hash", feature_meta.get("feature_dictionary_hash")),
        ("feature.feature_config_hash", feature_meta.get("feature_config_hash")),
        ("feature.feature_code_sha256", feature_meta.get("feature_code_sha256")),
    ):
        _require_hex64(value, name)
    label_run = panel.execute(
        "SELECT run_id,split,horizon_days,start_date,end_date,dataset_end,status,config_hash,code_hash,manifest_hash "
        "FROM label_runs WHERE run_id=? AND split=? AND horizon_days=?",
        (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON),
    ).fetchone()
    if label_run is None or label_run["status"] != "complete":
        raise BuildStopped("locked H=7 label run missing or incomplete")
    if tuple(label_run[key] for key in ("start_date", "end_date", "dataset_end")) != (
        SCORE_START.isoformat(), SCORE_END.isoformat(), DATASET_END.isoformat()
    ):
        raise BuildStopped("label run dates do not match the locked training range")
    expected_label_source_hash = panel_meta.get("label_source_manifest_hash")
    _require_hex64(label_run["config_hash"], "label_run.config_hash")
    _require_hex64(label_run["code_hash"], "label_run.code_hash")
    _require_hex64(label_run["manifest_hash"], "label_run.manifest_hash")
    if label_run["manifest_hash"] != expected_label_source_hash:
        raise BuildStopped("label run does not bind to the verified label source manifest")
    if panel_meta.get("label_build_manifest_hash") != panel_meta.get("build_manifest_hash"):
        raise BuildStopped("label build manifest does not match panel build manifest")
    feature_rows = feature.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0]
    if feature_rows != EXPECTED_FEATURE_ROWS:
        raise BuildStopped(f"feature row count changed: {feature_rows}")
    if output_db.exists() and not allow_existing_output:
        raise BuildStopped(f"refusing to overwrite existing v2 output: {output_db}")
    evidence_root.mkdir(parents=True, exist_ok=True)
    replay_code_sha256 = _sha256_file(Path(__file__))
    selection_code_sha256 = _sha256_file(ROOT / "pipeline/replay_selection.py")
    config_hash = _config_hash(
        feature_meta,
        panel_meta,
        replay_code_sha256,
        dict(label_run),
        selection_code_sha256,
    )
    snapshot = _resource_snapshot(output_db.with_name(output_db.name + ".partial"), evidence_root)
    if snapshot["free_bytes"] < MIN_FREE_BYTES:
        raise BuildStopped(f"free-space gate reached before replay: {snapshot}")
    preflight = {
        "status": "pass",
        "panel_database": str(panel_db),
        "feature_database": str(feature_db),
        "output_database": str(output_db),
        "panel_meta": panel_meta,
        "feature_meta": feature_meta,
        "label_run": dict(label_run),
        "feature_rows": feature_rows,
        "config_hash": config_hash,
        "feature_code_at_build_sha256": feature_meta.get("feature_code_sha256"),
        "feature_code_current_sha256": _sha256_file(ROOT / "pipeline/build_feature_replay.py"),
        "replay_code_sha256": replay_code_sha256,
        "selection_code_sha256": selection_code_sha256,
        "resources_before": snapshot,
        "training_approval": False,
    }
    return panel, feature, panel_meta, feature_meta, config_hash, preflight


def _label_status_strata(panel: sqlite3.Connection, method: str, strata: dict[tuple[str, str], dict[str, int]]) -> None:
    rows = panel.execute(
        "SELECT status,COUNT(*) AS n FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? GROUP BY status",
        (LABEL_RUN_ID, LABEL_SPLIT, LABEL_HORIZON),
    )
    for row in rows:
        _strata_add(strata, "label_status", str(row["status"]), "eligible_device_days", int(row["n"]))


def _run_method(
    panel: sqlite3.Connection,
    feature: sqlite3.Connection,
    output: sqlite3.Connection,
    method: str,
    config_hash: str,
    feature_meta: dict[str, str],
    panel_meta: dict[str, str],
    opportunities: dict[str, dict],
    output_db: Path,
    evidence_root: Path,
    start_time: float,
) -> dict:
    # A partial method may have date rows from an interrupted attempt.  It is
    # safe to remove only that method in the new v2 database and replay it.
    output.execute("DELETE FROM replay_daily WHERE run_id=? AND method=?", (REPLAY_ID, method))
    output.execute("DELETE FROM replay_alerts WHERE run_id=? AND method=?", (REPLAY_ID, method))
    output.execute("DELETE FROM replay_event_summary WHERE run_id=? AND method=?", (REPLAY_ID, method))
    output.execute("DELETE FROM replay_strata WHERE run_id=? AND method=?", (REPLAY_ID, method))
    output.execute("DELETE FROM replay_runs WHERE run_id=? AND method=?", (REPLAY_ID, method))
    output.commit()
    last_alert: dict[str, dt.date] = {}
    event_alerts: dict[str, list[tuple[dt.date, int]]] = collections.defaultdict(list)
    strata: dict[tuple[str, str], dict[str, int]] = {}
    daily_rows: list[tuple] = []
    alert_rows: list[tuple] = []
    total_eligible = alerts = known_hits = known_no_hit = unknown = event_hits = early_hits = 0
    outside_range_positive_alerts = 0
    alert_dates_by_serial: dict[str, list[dt.date]] = collections.defaultdict(list)
    processed_feature_rows = 0
    next_row_gate = RESOURCE_CHECK_ROWS
    for offset in range((SCORE_END - SCORE_START).days + 1):
        date = SCORE_START + dt.timedelta(days=offset)
        date_text = date.isoformat()
        rows = _feature_rows(feature, date_text)
        processed_feature_rows += len(rows)
        selected, selection_stats = select_alerts(
            rows,
            method,
            date,
            last_alert,
            denominator=BUDGET_DENOMINATOR,
            cooldown_days=COOLDOWN_DAYS,
        )
        total_eligible += len(rows)
        month = date_text[:7]
        for row in rows:
            _strata_add(strata, "month", month, "eligible_device_days")
            _strata_add(strata, "current_smart_missing", _feature_row_missing_bucket(row), "eligible_device_days")
            _strata_add(strata, "history_observations_14", str(row["history_observations_14"]), "eligible_device_days")
        day_known_hits = day_known_no_hit = day_unknown = day_hits = day_early = 0
        for row in selected:
            serial = str(row["serial_number"])
            label = _label_for_alert(panel, date_text, serial)
            if label is None:
                raise BuildStopped(f"missing label for alert key {date_text}/{serial}")
            status = label["status"]
            label_value = label["label"]
            failure_date = label["first_failure_date"]
            if label_value == 1:
                known_hits += 1; day_known_hits += 1
            elif label_value == 0:
                known_no_hit += 1; day_known_no_hit += 1
            elif label_value is None:
                unknown += 1; day_unknown += 1
            else:
                raise BuildStopped(f"invalid label value for {date_text}/{serial}: {label_value}")
            event_key = None
            event_hit = 0
            lead_days = None
            if failure_date:
                failure = dt.date.fromisoformat(failure_date)
                if label_value == 1 and not EVENT_START <= failure <= EVENT_END:
                    outside_range_positive_alerts += 1
                lead = (failure - date).days
                if EVENT_START <= failure <= EVENT_END and 1 <= lead <= HORIZON and serial in opportunities:
                    event_key = serial
                    event_alerts[serial].append((date, lead))
                    if opportunities[serial]["opportunity"]:
                        event_hit = 1
                        lead_days = lead
                        day_hits += 1; event_hits += 1
                        if lead >= 2:
                            day_early += 1; early_hits += 1
            score = 0.0
            if method == "smart_nonzero":
                score = float(row["smart_nonzero_signal_count"])
            elif method == "smart187":
                score = float(row["smart_187_signal"])
            alert_rows.append(
                (
                    REPLAY_ID, method, date_text, serial, score,
                    row["tie_break_sha256"], status, label_value, failure_date,
                    event_key, event_hit, lead_days,
                )
            )
            alert_dates_by_serial[serial].append(date)
            _strata_add(strata, "month", month, "alerts")
            _strata_add(strata, "current_smart_missing", _feature_row_missing_bucket(row), "alerts")
            _strata_add(strata, "history_observations_14", str(row["history_observations_14"]), "alerts")
            if label_value == 1:
                _strata_add(strata, "month", month, "known_hit_alerts")
                _strata_add(strata, "current_smart_missing", _feature_row_missing_bucket(row), "known_hit_alerts")
                _strata_add(strata, "history_observations_14", str(row["history_observations_14"]), "known_hit_alerts")
            elif label_value is None:
                _strata_add(strata, "month", month, "unknown_alerts")
                _strata_add(strata, "current_smart_missing", _feature_row_missing_bucket(row), "unknown_alerts")
                _strata_add(strata, "history_observations_14", str(row["history_observations_14"]), "unknown_alerts")
            if event_hit:
                _strata_add(strata, "month", month, "event_hits")
                _strata_add(strata, "current_smart_missing", _feature_row_missing_bucket(row), "event_hits")
                _strata_add(strata, "history_observations_14", str(row["history_observations_14"]), "event_hits")
        alerts += len(selected)
        daily_rows.append(
            (
                REPLAY_ID, method, date_text,
                selection_stats["eligible_count"], selection_stats["budget_k"],
                selection_stats["signal_count"], selection_stats["cooldown_excluded"],
                selection_stats["alerts_count"], day_known_hits, day_known_no_hit,
                day_unknown, day_hits, day_early,
            )
        )
        # Commit every date so an interrupted run has a bounded, inspectable
        # prefix.  The method summary is committed only after all dates pass.
        output.executemany("INSERT INTO replay_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [daily_rows[-1]])
        if alert_rows:
            output.executemany("INSERT INTO replay_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", alert_rows)
            alert_rows.clear()
        output.commit()
        # Check after every date and at least every 10,000 feature rows.  The
        # per-date check makes an empty date observable; the row threshold
        # keeps a very large date from running unchecked.
        snapshot = _resource_snapshot(output_db, evidence_root)
        if snapshot["rss_max_bytes"] >= MAX_RSS_BYTES or snapshot["owned_bytes"] >= MAX_OWNED_BYTES or snapshot["free_bytes"] < MIN_FREE_BYTES:
            raise BuildStopped(f"replay resource gate reached at {date_text}: {snapshot}")
        if time.monotonic() - start_time >= SOFT_TIMEOUT_SECONDS:
            raise BuildStopped(f"replay soft timeout reached at {date_text}: {snapshot}")
        while processed_feature_rows >= next_row_gate:
            next_row_gate += RESOURCE_CHECK_ROWS
    # The list was flushed at each date, so no alert rows remain pending.
    event_rows: list[tuple] = []
    lead_values: list[int] = []
    event_hit_count = early_event_count = 0
    lead_2 = lead_3 = 0
    for event_key, info in opportunities.items():
        choices = sorted(event_alerts.get(event_key, []))
        hit = int(bool(choices) and bool(info["opportunity"]))
        earliest_date = choices[0][0].isoformat() if choices else None
        earliest_lead = choices[0][1] if choices else None
        if hit:
            event_hit_count += 1
            lead_values.append(earliest_lead)
            if earliest_lead >= 2:
                early_event_count += 1; lead_2 += 1
            if earliest_lead >= 3:
                lead_3 += 1
        event_rows.append(
            (
                REPLAY_ID, method, event_key, info["first_failure_date"],
                info["opportunity"], hit, earliest_date, earliest_lead,
            )
        )
    output.executemany("INSERT INTO replay_event_summary VALUES (?,?,?,?,?,?,?,?)", event_rows)
    strata_rows = []
    _label_status_strata(panel, method, strata)
    for (stratum_type, stratum), item in sorted(strata.items()):
        strata_rows.append(
            (
                REPLAY_ID, method, stratum_type, stratum,
                item["eligible_device_days"], item["alerts"], item["known_hit_alerts"],
                item["unknown_alerts"], item["event_hits"],
            )
        )
    output.executemany("INSERT INTO replay_strata VALUES (?,?,?,?,?,?,?,?,?)", strata_rows)
    repeat_counts = [len(values) for values in alert_dates_by_serial.values() if len(values) > 1]
    min_gap = minimum_gap([date for values in alert_dates_by_serial.values() for date in values])
    # ``minimum_gap`` over all devices is not valid; recompute within device.
    gaps = [minimum_gap(values) for values in alert_dates_by_serial.values() if len(values) > 1]
    min_gap = min((gap for gap in gaps if gap is not None), default=None)
    lower = known_hits / alerts if alerts else None
    upper = (known_hits + unknown) / alerts if alerts else None
    known_precision = known_hits / (known_hits + known_no_hit) if known_hits + known_no_hit else None
    method_summary = {
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
        "confirmed_no_hit_per_1000_device_days": 1000 * known_no_hit / total_eligible if total_eligible else None,
        "positive_alerts_outside_main_event_range": outside_range_positive_alerts,
        "event_total": len(opportunities),
        "event_opportunity_total": sum(int(value["opportunity"]) for value in opportunities.values()),
        "event_hits": event_hit_count,
        "event_recall_at_opportunity": event_hit_count / len([v for v in opportunities.values() if v["opportunity"]]) if opportunities else None,
        "all_event_capture": event_hit_count / len(opportunities) if opportunities else None,
        "early_event_hits_ge2_days": early_event_count,
        "early_event_hits_ge3_days": lead_3,
        "early_recall_at_opportunity_ge2_days": early_event_count / sum(int(v["opportunity"]) for v in opportunities.values()) if opportunities else None,
        "early_recall_at_opportunity_ge3_days": lead_3 / sum(int(v["opportunity"]) for v in opportunities.values()) if opportunities else None,
        "hit_only_ratio_ge2_days": early_event_count / event_hit_count if event_hit_count else None,
        "hit_only_ratio_ge3_days": lead_3 / event_hit_count if event_hit_count else None,
        "earliest_lead_days": {
            "count": len(lead_values),
            "median": median(lead_values),
            "q25": quantile(lead_values, 0.25),
            "q75": quantile(lead_values, 0.75),
            "ge2": lead_2,
            "ge3": lead_3,
        },
        "repeated_alert_devices": len(repeat_counts),
        "max_alerts_per_device": max((len(values) for values in alert_dates_by_serial.values()), default=0),
        "minimum_alert_gap_days": min_gap,
        "budget_denominator": BUDGET_DENOMINATOR,
        "cooldown_days": COOLDOWN_DAYS,
        "training_approval": False,
    }
    output.execute(
        "INSERT INTO replay_runs VALUES (?,?,?,?,?,?,?,?,?)",
        (
            REPLAY_ID, method, config_hash,
            feature_meta["feature_dictionary_hash"], panel_meta["build_manifest_hash"],
            LABEL_RUN_ID, "complete", json.dumps(method_summary, ensure_ascii=False, sort_keys=True), _utc_now(),
        ),
    )
    output.commit()
    return method_summary


def _completed_replay_result(path: Path, config_hash: str) -> dict:
    """Return an idempotent result only for a fully matching replay file."""
    metadata = _read_replay_metadata(path)
    if metadata.get("replay_status") != "complete":
        raise BuildStopped(f"existing replay database is not complete: {path}")
    if metadata.get("replay_config_hash") != config_hash:
        raise BuildStopped("existing replay database config does not match current preflight")
    connection = _connect_immutable(path)
    try:
        rows = connection.execute(
            "SELECT method,status,config_hash,summary_json FROM replay_runs "
            "WHERE run_id=? ORDER BY method",
            (REPLAY_ID,),
        ).fetchall()
        if {row["method"] for row in rows} != set(METHODS):
            raise BuildStopped("existing replay database is missing a completed method")
        for row in rows:
            if row["status"] != "complete" or row["config_hash"] != config_hash:
                raise BuildStopped("existing replay method is incomplete or mismatched")
        summaries = {row["method"]: json.loads(row["summary_json"]) for row in rows}
    finally:
        connection.close()
    return {
        "status": "already_complete",
        "database": str(path),
        "run_id": REPLAY_ID,
        "config_hash": config_hash,
        "methods": summaries,
        "training_approval": False,
    }


def build_replay(panel_db: Path, feature_db: Path, output_db: Path, evidence_root: Path, *, resume: bool = False) -> dict:
    panel_db = _project_path(panel_db, "panel database")
    feature_db = _project_path(feature_db, "feature database")
    output_db = _project_path(output_db, "replay database")
    evidence_root = _project_path(evidence_root, "replay evidence")
    partial_db = output_db.with_name(output_db.name + ".partial")
    if partial_db.exists() and not resume:
        raise BuildStopped(f"partial v2 replay exists; use --resume after inspecting: {partial_db}")
    panel, feature, panel_meta, feature_meta, config_hash, preflight = _preflight(
        panel_db,
        feature_db,
        output_db,
        evidence_root,
        allow_existing_output=True,
    )
    if output_db.exists():
        try:
            result = _completed_replay_result(output_db, config_hash)
        finally:
            panel.close()
            feature.close()
        return result
    if partial_db.exists():
        # This read-only check must happen before opening the partial database
        # for writes.  A rejected resume must leave both bytes and metadata
        # untouched.
        try:
            _validate_partial_config(partial_db, config_hash)
        except BaseException:
            panel.close()
            feature.close()
            raise
        # The helper above is deliberately called while the source inputs are
        # still immutable and before the shared report or writable connection
        # is touched.
    _atomic_json(evidence_root / "preflight_v2.json", preflight)
    start_time = time.monotonic()
    output: sqlite3.Connection | None = None
    progress_guards: list[tuple[sqlite3.Connection, dict[str, str | None]]] = []
    try:
        if partial_db.exists():
            output = sqlite3.connect(partial_db)
            output.row_factory = sqlite3.Row
        else:
            output = sqlite3.connect(partial_db)
            output.row_factory = sqlite3.Row
            _ddl(output)
            output.executemany(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                [
                    ("replay_status", "running"),
                    ("replay_version", REPLAY_ID),
                    ("replay_config_hash", config_hash),
                    ("replay_code_sha256", preflight["replay_code_sha256"]),
                    ("feature_dictionary_hash", feature_meta["feature_dictionary_hash"]),
                    ("feature_config_hash", feature_meta["feature_config_hash"]),
                    ("feature_code_at_build_sha256", feature_meta.get("feature_code_sha256", "")),
                    ("feature_code_current_sha256", preflight["feature_code_current_sha256"]),
                    ("source_panel_manifest_hash", panel_meta["build_manifest_hash"]),
                    ("label_run_id", LABEL_RUN_ID),
                    ("label_config_hash", preflight["label_run"]["config_hash"]),
                    ("label_code_sha256", preflight["label_run"]["code_hash"]),
                    ("label_source_manifest_hash", preflight["label_run"]["manifest_hash"]),
                    ("selection_code_sha256", preflight["selection_code_sha256"]),
                    ("model", MODEL),
                    ("score_start", SCORE_START.isoformat()),
                    ("score_end", SCORE_END.isoformat()),
                    ("dataset_end", DATASET_END.isoformat()),
                    ("horizon_days", str(HORIZON)),
                    ("budget_denominator", str(BUDGET_DENOMINATOR)),
                    ("cooldown_days", str(COOLDOWN_DAYS)),
                    ("training_approval", "false"),
                ],
            )
            output.commit()
        for connection in (panel, feature):
            progress_guards.append((connection, _install_sql_progress_guard(connection, start_time)))
        opportunities = _event_opportunities(panel)
        if (len(opportunities), sum(int(value["opportunity"]) for value in opportunities.values())) != (EXPECTED_EVENT_TOTAL, EXPECTED_EVENT_OPPORTUNITY):
            raise BuildStopped(f"event opportunity denominator changed: {(len(opportunities), sum(int(value['opportunity']) for value in opportunities.values()))}")
        summaries: dict[str, dict] = {}
        for method in METHODS:
            existing = output.execute(
                "SELECT status,config_hash FROM replay_runs WHERE run_id=? AND method=?",
                (REPLAY_ID, method),
            ).fetchone()
            if existing is not None and existing["status"] == "complete":
                if existing["config_hash"] != config_hash:
                    raise BuildStopped(f"completed method config mismatch: {method}")
                summaries[method] = json.loads(
                    output.execute("SELECT summary_json FROM replay_runs WHERE run_id=? AND method=?", (REPLAY_ID, method)).fetchone()[0]
                )
                continue
            summaries[method] = _run_method(
                panel, feature, output, method, config_hash, feature_meta, panel_meta,
                opportunities, partial_db, evidence_root, start_time,
            )
        output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("replay_status", "complete"))
        output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("training_approval", "false"))
        output.commit()
        snapshot = _resource_snapshot(partial_db, evidence_root)
        qa = {
            "status": "pass",
            "run_id": REPLAY_ID,
            "config_hash": config_hash,
            "event_denominators": {"all": len(opportunities), "with_opportunity": sum(int(v["opportunity"]) for v in opportunities.values())},
            "methods": summaries,
            "training_approval": False,
            "resources_after": snapshot,
            "limitations": [
                "训练期开发诊断；没有验证/测试泛化结论。",
                "未知告警保留并提供精度上下界；没有把未知填成负例。",
                "没有拟合年龄、逻辑回归或树模型，也没有bootstrap区间或预算敏感性。",
                "低覆盖与缺失分层只作事后诊断，不进入排名。",
            ],
        }
        _atomic_json(evidence_root / "replay_qa_v2.json", qa)
        for connection, _state in progress_guards:
            _clear_sql_progress_guard(connection)
        output.close(); panel.close(); feature.close()
        os.replace(partial_db, output_db)
        return qa
    except sqlite3.OperationalError as exc:
        reasons = [state.get("reason") for _connection, state in progress_guards if state.get("reason")]
        if reasons:
            raise BuildStopped(reasons[0]) from exc
        raise
    except BaseException as exc:
        try:
            for connection, _state in progress_guards:
                _clear_sql_progress_guard(connection)
            if output is not None:
                output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("replay_status", "failed"))
                output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("last_error", f"{type(exc).__name__}: {exc}"))
                output.commit(); output.close()
        finally:
            panel.close(); feature.close()
        attempt = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "config_hash": config_hash,
            "resources": _resource_snapshot(partial_db, evidence_root),
            "partial_database": str(partial_db),
        }
        _atomic_json(evidence_root / f"replay_attempt_failed_{int(time.time())}.json", attempt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL_DB_DEFAULT)
    parser.add_argument("--feature", type=Path, default=FEATURE_DB_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DB_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build_replay(args.panel, args.feature, args.output, args.evidence_root, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
