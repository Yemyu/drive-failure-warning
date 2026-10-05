"""SQLite panel storage and validated member ingestion."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pathlib
import sqlite3
from typing import Callable, Iterable

from .member_io import REQUIRED_FIELDS, SMART_FIELDS, iter_member_rows


SCHEMA_VERSION = "panel-v3"

# Exact ordered header contracts for the two source families used here.
# Unknown or changed headers are rejected before any selected rows are written.
DECLARED_SCHEMAS = {
    "data_Q1_2023": {
        "count": 179,
        "sha256": "326bb31d6eb53fa3ad12f0fa90e66c8f26be4724e492cd3fea8f4e1d50055e11",
    },
    "data_Q2_2023": {
        "count": 186,
        "sha256": "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2",
    },
}


def _execute_ddl(connection: sqlite3.Connection, script: str) -> None:
    """Execute our static DDL without executescript's implicit COMMIT."""
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip():
        raise RuntimeError("incomplete schema statement")


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _ensure_member_count_columns(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, "member_counts")
    additions = {
        "schema_sha256": "TEXT NOT NULL DEFAULT ''",
        "schema_json": "TEXT NOT NULL DEFAULT '[]'",
        "audit_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE member_counts ADD COLUMN {name} {definition}")


def _ensure_label_run_columns(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, "label_runs")
    if "code_hash" not in columns:
        connection.execute("ALTER TABLE label_runs ADD COLUMN code_hash TEXT NOT NULL DEFAULT ''")


def _ensure_daily_columns(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, "daily")
    if "capacity_clean_bytes" not in columns:
        connection.execute("ALTER TABLE daily ADD COLUMN capacity_clean_bytes INTEGER")
    if "capacity_missing" not in columns:
        connection.execute(
            "ALTER TABLE daily ADD COLUMN capacity_missing INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        "UPDATE daily SET capacity_clean_bytes = CASE "
        "WHEN capacity_bytes IS NULL OR capacity_bytes = -1 THEN NULL "
        "WHEN capacity_bytes >= 0 THEN capacity_bytes ELSE NULL END, "
        "capacity_missing = CASE WHEN capacity_bytes IS NULL OR capacity_bytes = -1 THEN 1 ELSE 0 END"
    )


def _backfill_identity_registry(connection: sqlite3.Connection) -> None:
    """Seed legacy panels while recording that their scope is incomplete."""
    registry_exists = connection.execute(
        "SELECT 1 FROM serial_model_registry LIMIT 1"
    ).fetchone()
    daily_exists = connection.execute("SELECT 1 FROM daily LIMIT 1").fetchone()
    if registry_exists is None and daily_exists:
        conflict = connection.execute(
            "SELECT serial_number FROM daily GROUP BY serial_number "
            "HAVING COUNT(DISTINCT model) > 1 LIMIT 1"
        ).fetchone()
        if conflict:
            raise ValueError(f"legacy serial/model identity conflict: {conflict[0]}")
        connection.execute(
            """
            INSERT INTO serial_model_registry(
                serial_number, model, first_date, last_date,
                first_source_member, source_scope
            )
            SELECT serial_number, model, MIN(date), MAX(date),
                   MIN(source_member), 'selected_daily_backfill'
            FROM daily GROUP BY serial_number, model
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES "
            "('serial_model_registry_scope', 'selected_daily_backfill')"
        )


def _create_label_flow(connection: sqlite3.Connection) -> None:
    _execute_ddl(connection,
        """
        CREATE TABLE IF NOT EXISTS label_flow (
            run_id TEXT NOT NULL,
            split TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            decision_date TEXT NOT NULL,
            serial_number TEXT NOT NULL,
            model TEXT NOT NULL,
            capacity_bytes INTEGER,
            first_failure_date TEXT,
            label INTEGER,
            status TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            history_observations INTEGER NOT NULL,
            future_observations INTEGER NOT NULL,
            PRIMARY KEY (run_id, split, horizon_days, decision_date, serial_number)
        );
        CREATE INDEX IF NOT EXISTS label_flow_date
            ON label_flow (run_id, split, horizon_days, decision_date);
        CREATE INDEX IF NOT EXISTS label_flow_label
            ON label_flow (run_id, split, horizon_days, label);
        """
    )


def _upgrade_label_flow(connection: sqlite3.Connection) -> None:
    required = {"run_id", "split", "config_hash"}
    columns = _table_columns(connection, "label_flow")
    if not columns:
        _create_label_flow(connection)
        return
    if required.issubset(columns):
        return
    connection.execute("DROP INDEX IF EXISTS label_flow_date")
    connection.execute("DROP INDEX IF EXISTS label_flow_label")
    legacy = "label_flow_legacy"
    suffix = 2
    existing = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    while legacy in existing:
        legacy = f"label_flow_legacy_{suffix}"
        suffix += 1
    connection.execute(f"ALTER TABLE label_flow RENAME TO {legacy}")
    _create_label_flow(connection)
    connection.execute(
        f"""
        INSERT INTO label_flow(
            run_id, split, config_hash, horizon_days, decision_date,
            serial_number, model, capacity_bytes, first_failure_date, label,
            status, eligible, history_observations, future_observations
        )
        SELECT 'legacy', 'legacy', 'legacy', horizon_days, decision_date,
               serial_number, model, capacity_bytes, first_failure_date, label,
               status, eligible, history_observations, future_observations
        FROM {legacy}
        """
    )


def _int_or_none(value: str, field: str, source: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"non-integer {field}={value!r} in {source}") from exc


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def create_schema(connection: sqlite3.Connection) -> None:
    # Connection settings live outside the migration. Every schema/data change
    # below belongs to one savepoint, including legacy rename/create/copy.
    if not connection.in_transaction:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("SAVEPOINT panel_schema_migration")
    try:
        _initialize_schema(connection)
    except BaseException:
        connection.execute("ROLLBACK TO panel_schema_migration")
        connection.execute("RELEASE panel_schema_migration")
        raise
    connection.execute("RELEASE panel_schema_migration")


def _initialize_schema(connection: sqlite3.Connection) -> None:
    _execute_ddl(connection,
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily (
            date TEXT NOT NULL,
            serial_number TEXT NOT NULL,
            model TEXT NOT NULL,
            capacity_bytes INTEGER,
            capacity_clean_bytes INTEGER,
            capacity_missing INTEGER NOT NULL DEFAULT 0,
            failure INTEGER NOT NULL CHECK (failure IN (0, 1)),
            smart_5_raw INTEGER,
            smart_9_raw INTEGER,
            smart_187_raw INTEGER,
            smart_188_raw INTEGER,
            smart_197_raw INTEGER,
            smart_198_raw INTEGER,
            smart_5_missing INTEGER NOT NULL,
            smart_9_missing INTEGER NOT NULL,
            smart_187_missing INTEGER NOT NULL,
            smart_188_missing INTEGER NOT NULL,
            smart_197_missing INTEGER NOT NULL,
            smart_198_missing INTEGER NOT NULL,
            source_member TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            PRIMARY KEY (date, serial_number)
        );
        CREATE INDEX IF NOT EXISTS daily_serial_date ON daily (serial_number, date);
        CREATE INDEX IF NOT EXISTS daily_date ON daily (date);
        CREATE TABLE IF NOT EXISTS member_counts (
            date TEXT PRIMARY KEY,
            source_member TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_rows INTEGER NOT NULL,
            selected_rows INTEGER NOT NULL,
            failure_rows INTEGER NOT NULL,
            schema_columns INTEGER NOT NULL,
            schema_sha256 TEXT NOT NULL DEFAULT '',
            schema_json TEXT NOT NULL DEFAULT '[]',
            audit_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS label_runs (
            run_id TEXT NOT NULL,
            split TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            dataset_end TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            code_hash TEXT NOT NULL DEFAULT '',
            manifest_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            PRIMARY KEY (run_id, split, horizon_days)
        );
        CREATE TABLE IF NOT EXISTS serial_model_registry (
            serial_number TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            first_date TEXT NOT NULL,
            last_date TEXT NOT NULL,
            first_source_member TEXT NOT NULL,
            source_scope TEXT NOT NULL
        );
        """
    )
    _ensure_member_count_columns(connection)
    _ensure_label_run_columns(connection)
    _ensure_daily_columns(connection)
    _upgrade_label_flow(connection)
    _backfill_identity_registry(connection)


class PanelWriter:
    def __init__(
        self,
        path: pathlib.Path,
        *,
        reset: bool = False,
        schema_declarations: dict | None = None,
    ) -> None:
        if reset and path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        create_schema(self.connection)
        self.path = path
        self.schema_declarations = schema_declarations or DECLARED_SCHEMAS
        self._seen_dates = {
            row[0] for row in self.connection.execute("SELECT date FROM member_counts")
        }
        self.stats = collections.Counter()

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()

    def set_metadata(self, values: dict[str, str]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            values.items(),
        )
        self.connection.commit()

    @property
    def seen_dates(self) -> set[str]:
        return set(self._seen_dates)

    def append_member(
        self,
        entry: dict,
        root: pathlib.Path,
        model: str,
        *,
        min_selected_rows: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        source_data: tuple | None = None,
        expected_counts: dict | None = None,
        strict_smart: bool = False,
    ) -> dict:
        date, sha256, columns, rows, diagnostics = source_data if source_data is not None else iter_member_rows(entry, root)
        if date in self._seen_dates:
            raise RuntimeError(f"date already exists in panel: {date}")
        expected = set(REQUIRED_FIELDS)
        if not expected.issubset(columns):
            raise RuntimeError(f"schema missing required fields for {date}")
        index = {column: columns.index(column) for column in columns}
        selected = []
        selected_failures = 0
        source_rows = 0
        serials = set()
        all_serials = set()
        all_serial_models: dict[str, str] = {}
        capacity_missing = 0
        capacity_invalid = 0
        smart_negative = collections.Counter()
        group_fields = tuple(
            field
            for field in ("vault_id", "pod_id", "datacenter", "cluster_id", "is_legacy_format")
            if field in index
        )
        all_groups = {field: collections.Counter() for field in group_fields}
        selected_groups = {field: collections.Counter() for field in group_fields}
        for source_row, values in rows:
            source_rows += 1
            if progress_callback is not None and source_rows % 10_000 == 0:
                progress_callback(
                    {
                        "date": date,
                        "source_row": source_row,
                        "source_rows": source_rows,
                        "selected_rows": len(selected),
                    }
                )
            if len(values) != len(columns):
                raise ValueError(f"row length mismatch in {entry['name']} at row {source_row}")
            row_date = values[index["date"]]
            if row_date != date:
                raise ValueError(f"member/date mismatch: {entry['name']} has {row_date}")
            serial = values[index["serial_number"]]
            if not serial:
                raise ValueError(f"empty serial number in {entry['name']} at row {source_row}")
            if serial in all_serials:
                raise ValueError(f"duplicate source serial on {date}: {serial}")
            all_serials.add(serial)
            source_model = values[index["model"]]
            if not source_model:
                raise ValueError(f"empty model in {entry['name']} at row {source_row}")
            all_serial_models[serial] = source_model
            failure = _int_or_none(values[index["failure"]], "failure", entry["name"])
            if failure not in (0, 1):
                raise ValueError(f"invalid failure={failure!r} in {entry['name']} at row {source_row}")
            source_capacity = _int_or_none(
                values[index["capacity_bytes"]], "capacity_bytes", entry["name"]
            )
            if source_capacity is None or source_capacity == -1:
                capacity_missing += 1
            elif source_capacity < 0:
                capacity_invalid += 1
                raise ValueError(
                    f"unexpected negative capacity={source_capacity} in "
                    f"{entry['name']} at row {source_row}"
                )
            for field in group_fields:
                all_groups[field][values[index[field]]] += 1
            if source_model != model:
                continue
            if serial in serials:
                raise ValueError(f"duplicate selected serial on {date}: {serial}")
            serials.add(serial)
            for field in group_fields:
                selected_groups[field][values[index[field]]] += 1
            capacity = source_capacity
            smart = []
            missing = []
            for field in SMART_FIELDS:
                value = _int_or_none(values[index[field]], field, entry["name"])
                smart.append(value)
                missing.append(int(value is None))
                if value is not None and value < 0:
                    smart_negative[field] += 1
                    if strict_smart:
                        raise ValueError(f"negative {field} in {entry['name']} at row {source_row}")
            selected.append(
                (
                    date,
                    serial,
                    model,
                    capacity,
                    None if capacity is None or capacity == -1 else capacity,
                    int(capacity is None or capacity == -1),
                    failure,
                    *smart,
                    *missing,
                    entry["name"],
                    sha256,
                    source_row,
                )
            )
            selected_failures += failure
        existing_models = {
            row["serial_number"]: row["model"]
            for row in self.connection.execute(
                "SELECT serial_number, model FROM serial_model_registry"
            )
        }
        for serial, source_model in all_serial_models.items():
            known_model = existing_models.get(serial)
            if known_model is not None and known_model != source_model:
                raise ValueError(
                    f"serial/model identity conflict for {serial}: "
                    f"previous={known_model!r}, current={source_model!r}, date={date}"
                )
        if len(columns) != len(set(columns)):
            raise ValueError(f"duplicate column names in {entry['name']}")
        schema_hash = hashlib.sha256(
            json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        family = entry["name"].split("/", 1)[0]
        declared = self.schema_declarations.get(family)
        if declared is None:
            raise ValueError(f"source family {family!r} has no declared schema")
        if len(columns) != declared["count"] or schema_hash != declared["sha256"]:
            raise ValueError(
                f"schema does not match declared {family}: "
                f"count={len(columns)} hash={schema_hash}"
            )
        schema_key = f"schema_registry:{family}"
        registered = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (schema_key,)
        ).fetchone()
        if registered is None:
            prior_hashes = [
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT schema_sha256 FROM member_counts "
                    "WHERE source_member LIKE ? AND schema_sha256 <> ''",
                    (f"{family}/%",),
                )
            ]
            if len(prior_hashes) > 1 or (prior_hashes and prior_hashes[0] != schema_hash):
                raise ValueError(f"schema changed within source family {family!r}")
        else:
            registered_value = json.loads(registered["value"])
            if registered_value.get("sha256") != schema_hash or registered_value.get("columns") != columns:
                raise ValueError(f"schema changed within source family {family!r}")
        if not selected:
            # A source day can legitimately contain no member of the primary model,
            # but it is still recorded so the observation calendar remains explicit.
            pass
        if min_selected_rows is not None and len(selected) < min_selected_rows:
            raise RuntimeError(
                f"selected-row coverage below gate for {date}: "
                f"{len(selected)} < {min_selected_rows}"
            )
        if expected_counts is not None:
            actual = dict(source_rows=source_rows, selected_rows=len(selected), failure_rows=selected_failures,
                          schema_columns=len(columns), schema_sha256=schema_hash)
            for field, value in actual.items():
                if field not in expected_counts or expected_counts[field] != value:
                    raise ValueError(f"source contract {field} mismatch for {date}: {value} != {expected_counts.get(field)}")
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO daily(
                    date, serial_number, model, capacity_bytes, capacity_clean_bytes,
                    capacity_missing, failure,
                    smart_5_raw, smart_9_raw, smart_187_raw, smart_188_raw,
                    smart_197_raw, smart_198_raw,
                    smart_5_missing, smart_9_missing, smart_187_missing,
                    smart_188_missing, smart_197_missing, smart_198_missing,
                    source_member, source_sha256, source_row
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                selected,
            )
            self.connection.execute(
                """
                INSERT INTO member_counts(
                    date, source_member, source_sha256, source_rows, selected_rows,
                    failure_rows, schema_columns, schema_sha256, schema_json, audit_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date,
                    entry["name"],
                    sha256,
                    source_rows,
                    len(selected),
                    selected_failures,
                    len(columns),
                    hashlib.sha256(
                        json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()
                    ).hexdigest(),
                    json.dumps(columns, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(
                        {
                            "group_fields": list(group_fields),
                            "capacity_missing_or_sentinel": capacity_missing,
                            "capacity_invalid_negative": capacity_invalid,
                            "smart_negative_values": dict(smart_negative),
                            "all": {field: dict(counter) for field, counter in all_groups.items()},
                            "selected": {field: dict(counter) for field, counter in selected_groups.items()},
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO serial_model_registry(
                    serial_number, model, first_date, last_date,
                    first_source_member, source_scope
                ) VALUES (?, ?, ?, ?, ?, 'full_source_member')
                ON CONFLICT(serial_number) DO UPDATE SET
                    last_date = CASE WHEN excluded.last_date > last_date
                                     THEN excluded.last_date ELSE last_date END
                """,
                [
                    (serial, source_model, date, date, entry["name"])
                    for serial, source_model in all_serial_models.items()
                ],
            )
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (
                    schema_key,
                    json.dumps(
                        {"sha256": schema_hash, "columns": columns},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            if self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'serial_model_registry_scope'"
            ).fetchone() is None:
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES "
                    "('serial_model_registry_scope', 'full_source_rows')"
                )
        self._seen_dates.add(date)
        self.stats.update(
            source_rows=source_rows,
            selected_rows=len(selected),
            failure_rows=selected_failures,
            days=1,
        )
        return {
            "date": date,
            "source_member": entry["name"],
            "source_sha256": sha256,
            "source_rows": source_rows,
            "selected_rows": len(selected),
            "failure_rows": selected_failures,
            "schema_columns": len(columns),
            "schema_sha256": schema_hash,
            "capacity_missing_or_sentinel": capacity_missing,
            "capacity_invalid_negative": capacity_invalid,
            "smart_negative_values": dict(smart_negative),
            "group_audit_summary": {
                field: {
                    "all_unique": len(all_groups[field]),
                    "selected_unique": len(selected_groups[field]),
                    "all_rows": sum(all_groups[field].values()),
                    "selected_rows": sum(selected_groups[field].values()),
                }
                for field in group_fields
            },
            **diagnostics,
        }

    def summary(self) -> dict:
        values = dict(self.stats)
        values["panel_rows"] = self.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
        values["unique_serials"] = self.connection.execute("SELECT COUNT(DISTINCT serial_number) FROM daily").fetchone()[0]
        values["database"] = str(self.path)
        return values
