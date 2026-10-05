"""Build a run-scoped, leakage-aware label flow in the panel database."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import re
import sqlite3
import sys

from .labeling import classify_device_rows
from .panel import SCHEMA_VERSION, create_schema


HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _rows_for_serial(connection: sqlite3.Connection, serial: str, dataset_end: str):
    return connection.execute(
        """
        SELECT date, serial_number, model, capacity_bytes, failure
        FROM daily
        WHERE serial_number = ? AND date <= ?
        ORDER BY date
        """,
        (serial, dataset_end),
    ).fetchall()


def _validate_args(args: argparse.Namespace) -> None:
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dataset_end = dt.date.fromisoformat(args.dataset_end)
    if start > end:
        raise ValueError("start must be on or before end")
    if end > dataset_end:
        raise ValueError("end must not be after dataset_end")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.history_days <= 0:
        raise ValueError("history_days must be positive")
    if not 1 <= args.min_history <= args.history_days:
        raise ValueError("min_history must be between 1 and history_days")
    if not args.allow_end_censoring and end + dt.timedelta(days=args.horizon) > dataset_end:
        raise ValueError(
            "scoring end plus horizon exceeds dataset_end; "
            "use --allow-end-censoring only for diagnostics"
        )


def _code_hash() -> str:
    files = [
        pathlib.Path(__file__),
        pathlib.Path(__file__).with_name("labeling.py"),
        pathlib.Path(__file__).with_name("panel.py"),
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(args: argparse.Namespace, manifest_hash: str, code_hash: str) -> str:
    config = {
        "allow_end_censoring": args.allow_end_censoring,
        "dataset_end": args.dataset_end,
        "end": args.end,
        "horizon": args.horizon,
        "history_days": args.history_days,
        "manifest_hash": manifest_hash,
        "code_hash": code_hash,
        "min_history": args.min_history,
        "run_id": args.run_id,
        "split": args.split,
        "start": args.start,
        "schema_version": SCHEMA_VERSION,
    }
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _existing_run(connection: sqlite3.Connection, run_id: str, split: str, horizon: int):
    return connection.execute(
        """
        SELECT run_id, split, horizon_days, config_hash, manifest_hash, status
        FROM label_runs
        WHERE run_id = ? AND split = ? AND horizon_days = ?
        """,
        (run_id, split, horizon),
    ).fetchone()


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _source_manifest_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return validated member facts used by a formal label run.

    A digest of an empty table, a blank SHA, or a partially populated legacy
    table is not evidence of a stable input.  This check intentionally runs
    before a label transaction starts and is also used for day-by-day matching
    against the observed panel.
    """
    columns = (
        "date", "source_member", "source_sha256", "source_rows", "selected_rows",
        "failure_rows", "schema_columns", "schema_sha256", "schema_json", "audit_json",
    )
    available = {row[1] for row in connection.execute("PRAGMA table_info(member_counts)")}
    required = {
        "date", "source_member", "source_sha256", "source_rows", "selected_rows",
        "failure_rows", "schema_columns", "schema_sha256",
    }
    if not required.issubset(available):
        raise RuntimeError(
            "member_counts lacks required source facts: "
            + ", ".join(sorted(required - available))
        )
    selected = [column for column in columns if column in available]
    rows = connection.execute(
        f"SELECT {', '.join(selected)} FROM member_counts ORDER BY date, source_member"
    ).fetchall()
    if not rows:
        raise RuntimeError("member_counts is empty; source verification is unavailable")
    seen_members: set[str] = set()
    seen_dates: set[str] = set()
    for row in rows:
        date = row["date"]
        member = row["source_member"]
        source_sha = row["source_sha256"]
        if not date or not member or member in seen_members or date in seen_dates:
            raise RuntimeError(f"source member facts are not unique/complete for date {date!r}")
        if not HEX64.fullmatch(str(source_sha or "")):
            raise RuntimeError(f"source_sha256 is not a 64-hex digest for {member!r}")
        if not HEX64.fullmatch(str(row["schema_sha256"] or "")):
            raise RuntimeError(f"schema_sha256 is not a 64-hex digest for {member!r}")
        for field in ("source_rows", "selected_rows", "failure_rows", "schema_columns"):
            if row[field] is None or int(row[field]) < 0:
                raise RuntimeError(f"invalid {field} for {member!r}")
        if int(row["schema_columns"]) == 0:
            raise RuntimeError(f"schema_columns is zero for {member!r}")
        seen_members.add(member)
        seen_dates.add(date)
    return rows


def _validate_source_facts(connection: sqlite3.Connection, dataset_end: str) -> None:
    """Validate source facts and their exact relationship to observed daily rows."""
    if not _table_exists(connection, "member_counts"):
        raise RuntimeError("member_counts table is missing; source verification is unavailable")
    rows = _source_manifest_rows(connection)
    facts = {row["date"]: row for row in rows}
    if _table_exists(connection, "daily"):
        daily_dates = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT date FROM daily WHERE date <= ? ORDER BY date", (dataset_end,)
            )
        ]
        missing = sorted(set(daily_dates) - set(facts))
        if missing:
            raise RuntimeError(
                "member_counts is missing source facts for observed dates: "
                + ", ".join(missing[:5])
            )
        absent = sorted(
            date for date, fact in facts.items()
            if date <= dataset_end and int(fact['selected_rows']) > 0
            and date not in set(daily_dates)
        )
        if absent:
            raise RuntimeError(
                'daily is missing dates with positive selected source counts: '
                + ', '.join(absent[:5])
            )
        for row in connection.execute(
            """
            SELECT date, COUNT(*) AS selected_rows, COALESCE(SUM(failure), 0) AS failure_rows,
                   COUNT(DISTINCT source_member) AS members,
                   COUNT(DISTINCT source_sha256) AS hashes,
                   MIN(source_member) AS source_member,
                   MIN(source_sha256) AS source_sha256
            FROM daily WHERE date <= ? GROUP BY date
            """,
            (dataset_end,),
        ):
            fact = facts[row["date"]]
            if row["members"] != 1 or row["hashes"] != 1:
                raise RuntimeError(f"daily source identity is not unique for {row['date']}")
            if row["source_member"] != fact["source_member"] or row["source_sha256"] != fact["source_sha256"]:
                raise RuntimeError(f"daily/member_counts source mismatch for {row['date']}")
            if row["selected_rows"] != int(fact["selected_rows"]):
                raise RuntimeError(f"selected row count mismatch for {row['date']}")
            if row["failure_rows"] != int(fact["failure_rows"]):
                raise RuntimeError(f"failure row count mismatch for {row['date']}")


def _source_manifest_hash(connection: sqlite3.Connection) -> str:
    """Hash validated member content facts, excluding retrieval timestamps/status."""
    _source_manifest_rows(connection)
    columns = (
        "date", "source_member", "source_sha256", "source_rows", "selected_rows",
        "failure_rows", "schema_columns", "schema_sha256", "schema_json", "audit_json",
    )
    available = {row[1] for row in connection.execute("PRAGMA table_info(member_counts)")}
    selected = [column for column in columns if column in available]
    if not selected:
        raise RuntimeError("member_counts has no stable source columns")
    rows = connection.execute(
        f"SELECT {', '.join(selected)} FROM member_counts ORDER BY date, source_member"
    ).fetchall()
    payload = [list(row) for row in rows]
    return hashlib.sha256(
        json.dumps({"columns": selected, "rows": payload}, ensure_ascii=False,
                   sort_keys=False, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve_manifest_hash(
    connection: sqlite3.Connection,
    supplied: str,
    allow_unverified: bool,
    *,
    dataset_end: str | None = None,
) -> str:
    if allow_unverified:
        return "UNVERIFIED"
    if dataset_end is not None:
        _validate_source_facts(connection, dataset_end)
    actual = _source_manifest_hash(connection)
    if supplied and supplied != actual:
        raise RuntimeError(
            "manifest hash does not match stable member_counts content; "
            "rebuild the manifest from the declared input"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=pathlib.Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--dataset-end", required=True)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--history-days", type=int, default=14)
    parser.add_argument("--min-history", type=int, default=12)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--run-id")
    parser.add_argument("--manifest-hash", default="")
    parser.add_argument(
        "--allow-unverified-manifest",
        action="store_true",
        help="diagnostic-only escape hatch; labels carry the literal UNVERIFIED marker",
    )
    parser.add_argument("--config-hash")
    parser.add_argument("--allow-end-censoring", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = f"{args.split}-{args.start}-{args.end}-h{args.horizon}"
    _validate_args(args)

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    # Migrations are protected by the same outer savepoint as source/run
    # preflight.  A rejected source or changed run scope must leave even legacy
    # schema tables exactly as they were.
    connection.execute("SAVEPOINT label_run_preflight")
    try:
        create_schema(connection)
        args.manifest_hash = _resolve_manifest_hash(
            connection,
            args.manifest_hash,
            args.allow_unverified_manifest,
            dataset_end=args.dataset_end,
        )
        source_verification_status = (
            "unverified" if args.manifest_hash == "UNVERIFIED" else "verified"
        )
        code_hash = _code_hash()
        computed_config_hash = _config_hash(args, args.manifest_hash, code_hash)
        if args.config_hash and args.config_hash != computed_config_hash:
            raise ValueError(
                "--config-hash is an expected value and does not match the supplied run configuration"
            )
        config_hash = computed_config_hash
        existing = _existing_run(connection, args.run_id, args.split, args.horizon)
        if existing is not None and existing["config_hash"] != config_hash and not args.replace:
            raise RuntimeError(
                "run scope already exists with a different config_hash; "
                "use a new run_id or explicit --replace"
            )
        connection.execute("RELEASE label_run_preflight")
    except BaseException:
        connection.execute("ROLLBACK TO label_run_preflight")
        connection.execute("RELEASE label_run_preflight")
        connection.close()
        raise

    counters = collections.Counter()
    serials = [
        row[0]
        for row in connection.execute("SELECT DISTINCT serial_number FROM daily ORDER BY serial_number")
    ]
    # Rebuild exactly this run scope on every accepted invocation. This makes
    # retries idempotent and prevents rows outside the declared date range from
    # surviving a changed implementation. The flow and its metadata commit as
    # one transaction; a failed classifier leaves the previous run untouched.
    connection.execute("BEGIN")
    try:
        connection.execute(
            "DELETE FROM label_flow WHERE run_id = ? AND split = ? AND horizon_days = ?",
            (args.run_id, args.split, args.horizon),
        )
        connection.execute(
            "DELETE FROM label_runs WHERE run_id = ? AND split = ? AND horizon_days = ?",
            (args.run_id, args.split, args.horizon),
        )
        connection.execute(
            "DELETE FROM metadata WHERE key = ?",
            (f"label_summary:{args.run_id}:{args.split}:{args.horizon}",),
        )
        for serial in serials:
            rows = _rows_for_serial(connection, serial, args.dataset_end)
            flow = classify_device_rows(
                rows,
                start=args.start,
                end=args.end,
                dataset_end=args.dataset_end,
                horizon_days=args.horizon,
                history_days=args.history_days,
                min_history=args.min_history,
            )
            connection.executemany(
                """
                INSERT INTO label_flow(
                    run_id, split, config_hash, horizon_days, decision_date,
                    serial_number, model, capacity_bytes, first_failure_date,
                    label, status, eligible, history_observations,
                    future_observations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        args.run_id,
                        args.split,
                        config_hash,
                        args.horizon,
                        item["decision_date"],
                        item["serial_number"],
                        item["model"],
                        item["capacity_bytes"],
                        item["first_failure_date"],
                        item["label"],
                        item["status"],
                        item["eligible"],
                        item["history_observations"],
                        item["future_observations"],
                    )
                    for item in flow
                ],
            )
            counters.update(item["status"] for item in flow)

        summary = {
            "database": str(args.database),
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            "split": args.split,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "manifest_hash": args.manifest_hash,
            "horizon_days": args.horizon,
            "start": args.start,
            "end": args.end,
            "dataset_end": args.dataset_end,
            "serials": len(serials),
            "flow_rows": sum(counters.values()),
            "status_counts": dict(sorted(counters.items())),
        }
        summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        connection.execute(
            """
            INSERT INTO label_runs(
                run_id, split, horizon_days, start_date, end_date, dataset_end,
                config_hash, code_hash, manifest_hash, status, summary_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.run_id, args.split, args.horizon, args.start, args.end,
                args.dataset_end, config_hash, code_hash, args.manifest_hash, "complete",
                summary_json,
                dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (f"label_summary:{args.run_id}:{args.split}:{args.horizon}", summary_json),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        connection.close()
        raise
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
