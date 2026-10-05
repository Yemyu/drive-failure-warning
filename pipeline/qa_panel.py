"""Regression checks for the Q1 panel and its leakage-aware label flow."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sqlite3


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--split")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    failures: list[str] = []
    label_columns = {row[1] for row in connection.execute("PRAGMA table_info(label_flow)")}
    if {"run_id", "split"}.issubset(label_columns):
        run_id = args.run_id or "legacy"
        split = args.split or "legacy"
        label_predicate = f"run_id = {_sql_literal(run_id)} AND split = {_sql_literal(split)} AND "
        qualified_label_predicate = f"l.run_id = {_sql_literal(run_id)} AND l.split = {_sql_literal(split)} AND "
    else:
        label_predicate = ""
        qualified_label_predicate = ""

    def check(name: str, actual, expected) -> None:
        if actual != expected:
            failures.append(f"{name}: {actual!r} != {expected!r}")

    summary = {
        "database": str(args.database),
        "label_run_id": args.run_id or ("legacy" if label_columns else None),
        "label_split": args.split or ("legacy" if label_columns else None),
        "daily_rows": connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0],
        "dates": connection.execute("SELECT COUNT(DISTINCT date) FROM daily").fetchone()[0],
        "serials": connection.execute("SELECT COUNT(DISTINCT serial_number) FROM daily").fetchone()[0],
        "failure_rows": connection.execute("SELECT SUM(failure) FROM daily").fetchone()[0],
        "source_rows": connection.execute("SELECT SUM(source_rows) FROM member_counts").fetchone()[0],
        "selected_rows": connection.execute("SELECT SUM(selected_rows) FROM member_counts").fetchone()[0],
        "member_count_rows": connection.execute("SELECT COUNT(*) FROM member_counts").fetchone()[0],
    }
    check("daily_rows", summary["daily_rows"], 1_633_962)
    check("dates", summary["dates"], 90)
    check("serials", summary["serials"], 18_246)
    check("failure_rows", summary["failure_rows"], 170)
    check("source_rows", summary["source_rows"], 21_454_992)
    check("selected_rows", summary["selected_rows"], 1_633_962)
    check("member_count_rows", summary["member_count_rows"], 90)

    bad_capacity = connection.execute(
        "SELECT COUNT(*) FROM daily WHERE capacity_bytes NOT IN (-1, 4000787030016)"
    ).fetchone()[0]
    missing_sentinel = connection.execute(
        """
        SELECT COUNT(*) FROM daily
        WHERE capacity_bytes = -1
          AND NOT (
            smart_5_missing = 1 AND smart_9_missing = 1 AND smart_187_missing = 1
            AND smart_188_missing = 1 AND smart_197_missing = 1 AND smart_198_missing = 1
          )
        """
    ).fetchone()[0]
    check("unexpected_capacity_values", bad_capacity, 0)
    check("capacity_sentinel_without_all_smart_missing", missing_sentinel, 0)

    decreases = {}
    for field in ("smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"):
        decreases[field] = connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {field}, LAG({field}) OVER (PARTITION BY serial_number ORDER BY date) AS previous
                FROM daily WHERE {field} IS NOT NULL
            ) WHERE previous IS NOT NULL AND {field} < previous
            """
        ).fetchone()[0]
    summary["adjacent_decreases"] = decreases
    check("smart_197_decreases", decreases["smart_197_raw"], 275)
    check("smart_198_decreases", decreases["smart_198_raw"], 275)
    check("smart_5_decreases", decreases["smart_5_raw"], 0)
    check("smart_9_decreases", decreases["smart_9_raw"], 0)
    check("smart_187_decreases", decreases["smart_187_raw"], 0)
    check("smart_188_decreases", decreases["smart_188_raw"], 0)

    return_rows = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT serial_number, MIN(date) AS first_failure
            FROM daily WHERE failure=1 GROUP BY serial_number
        ) AS f
        WHERE EXISTS (
            SELECT 1 FROM daily d
            WHERE d.serial_number=f.serial_number AND d.date > f.first_failure
        )
        """
    ).fetchone()[0]
    check("failure_then_return_devices", return_rows, 3)

    status_rows = connection.execute(
        f"SELECT status, COUNT(*) AS n FROM label_flow WHERE {label_predicate}horizon_days=7 GROUP BY status"
    ).fetchall()
    status_counts = {row["status"]: row["n"] for row in status_rows}
    summary["label_status_counts"] = status_counts
    expected_status = {
        "gap_or_exit_censored": 916,
        "history_insufficient": 29,
        "negative_observed": 1_250_221,
        "positive_observed": 865,
        "positive_with_gap": 40,
        "post_failure": 2,
        "same_day_failure": 138,
    }
    check("label_status_counts", status_counts, expected_status)
    check(
        "eligible_label_rows",
        connection.execute(f"SELECT COUNT(*) FROM label_flow WHERE {label_predicate}horizon_days=7 AND eligible=1").fetchone()[0],
        1_252_042,
    )
    check(
        "positive_rows",
        connection.execute(f"SELECT COUNT(*) FROM label_flow WHERE {label_predicate}horizon_days=7 AND label=1").fetchone()[0],
        905,
    )
    check(
        "negative_rows",
        connection.execute(f"SELECT COUNT(*) FROM label_flow WHERE {label_predicate}horizon_days=7 AND label=0").fetchone()[0],
        1_250_221,
    )
    check(
        "same_day_or_post_failure_eligible",
        connection.execute(
            f"SELECT COUNT(*) FROM label_flow WHERE {label_predicate}horizon_days=7 AND status IN ('same_day_failure','post_failure') AND eligible=1"
        ).fetchone()[0],
        0,
    )
    future_failure_in_negative = connection.execute(
        f"""
        SELECT COUNT(*) FROM label_flow l
        WHERE {qualified_label_predicate}l.horizon_days=7 AND l.label=0
          AND EXISTS (
              SELECT 1 FROM daily d
              WHERE d.serial_number=l.serial_number AND d.failure=1
                AND d.date > l.decision_date
                AND d.date <= date(l.decision_date, '+7 day')
          )
        """
    ).fetchone()[0]
    check("negative_rows_with_future_failure", future_failure_in_negative, 0)
    label_partition = connection.execute(
        f"""
        SELECT
          SUM(status='positive_observed') + SUM(status='positive_with_gap') AS positives,
          SUM(status='negative_observed') AS negatives,
          SUM(status IN ('gap_or_exit_censored','end_censored')) AS unknowns,
          SUM(eligible) AS eligible
        FROM label_flow WHERE {label_predicate}horizon_days=7
        """
    ).fetchone()
    summary["label_partition"] = dict(label_partition)
    check("label_partition_eligible", label_partition["eligible"], 1_252_042)

    summary["passed"] = not failures
    summary["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Q1 panel QA failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    main()
