"""Create and audit the locked H=7 training label run on the verified panel."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import build_labels
from pipeline.labeling import classify_device_rows
from pipeline.rebuild_q1q2_local import _load_manifest, _relative_project_path, _sha256_file


DATABASE_DEFAULT = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
MANIFEST_DEFAULT = ROOT / "evidence/q2/input_review/q1q2_build_manifest_v1.json"
EVIDENCE_DEFAULT = ROOT / "evidence/q2/rebuild_v1/labels_v1.json"
RUN_ID = "train_q1q2_verified_h7_v1"
SPLIT = "train"
START = "2023-01-15"
END = "2023-06-23"
DATASET_END = "2023-06-30"
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _boundary_checks() -> dict:
    def make_rows(dates: list[str], failures: set[str] = set()) -> list[dict]:
        return [
            {
                "date": date,
                "serial_number": "boundary-disk",
                "model": "ST4000DM000",
                "capacity_bytes": 4_000_787_030_016,
                "failure": int(date in failures),
            }
            for date in dates
        ]

    full_dates = [
        (dt.date(2023, 6, 1) + dt.timedelta(days=offset)).isoformat()
        for offset in range(30)
    ]
    checks = {}
    for horizon in (3, 7, 14):
        output = classify_device_rows(
            make_rows(full_dates, {"2023-06-20"}),
            start="2023-06-15",
            end="2023-06-15",
            dataset_end="2023-06-30",
            horizon_days=horizon,
            history_days=14,
            min_history=12,
        )
        item = output[0]
        checks[f"h{horizon}_future_failure"] = {
            "status": item["status"],
            "label": item["label"],
            "expected": "positive_observed" if horizon >= 7 else "negative_observed",
        }
        if horizon >= 7 and (item["status"], item["label"]) != ("positive_observed", 1):
            raise RuntimeError(f"H={horizon} boundary check failed")
        if horizon == 3 and (item["status"], item["label"]) != ("negative_observed", 0):
            raise RuntimeError("H=3 boundary check failed")

    same_day = classify_device_rows(
        make_rows(full_dates, {"2023-06-15"}),
        start="2023-06-15",
        end="2023-06-15",
        dataset_end="2023-06-30",
    )[0]
    if (same_day["status"], same_day["eligible"]) != ("same_day_failure", 0):
        raise RuntimeError("same-day failure boundary check failed")
    checks["same_day_failure"] = {"status": same_day["status"], "eligible": same_day["eligible"]}

    gap_dates = [date for date in full_dates if date != "2023-06-18"]
    gap = classify_device_rows(
        make_rows(gap_dates),
        start="2023-06-16",
        end="2023-06-16",
        dataset_end="2023-06-30",
    )[0]
    if (gap["status"], gap["label"]) != ("gap_or_exit_censored", None):
        raise RuntimeError("gap boundary check failed")
    checks["future_gap"] = {"status": gap["status"], "label": gap["label"]}

    cutoff_dates = [
        (dt.date(2023, 6, 16) + dt.timedelta(days=offset)).isoformat()
        for offset in range(20)
    ]
    clipped = classify_device_rows(
        make_rows([date for date in cutoff_dates if date <= DATASET_END]),
        start="2023-06-29",
        end="2023-06-29",
        dataset_end=DATASET_END,
    )[0]
    if (clipped["status"], clipped["label"]) != ("end_censored", None):
        raise RuntimeError("dataset cutoff boundary check failed")
    checks["dataset_cutoff"] = {"status": clipped["status"], "label": clipped["label"]}
    return checks


def _run_label_builder(database: Path, source_manifest_hash: str) -> None:
    argv = [
        "build_labels",
        "--database", str(database),
        "--start", START,
        "--end", END,
        "--dataset-end", DATASET_END,
        "--horizon", str(HORIZON),
        "--history-days", str(HISTORY_DAYS),
        "--min-history", str(MIN_HISTORY),
        "--split", SPLIT,
        "--run-id", RUN_ID,
        "--manifest-hash", source_manifest_hash,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        build_labels.main()
    finally:
        sys.argv = old_argv


def _qa_labels(connection: sqlite3.Connection, database: Path, output: Path, source_manifest_hash: str, build_manifest_hash: str, build_manifest_file_sha256: str) -> dict:
    predicate = "run_id = ? AND split = ? AND horizon_days = ?"
    params = (RUN_ID, SPLIT, HORIZON)
    run = connection.execute(
        "SELECT run_id, split, horizon_days, start_date, end_date, dataset_end, config_hash, code_hash, manifest_hash, status, summary_json "
        "FROM label_runs WHERE " + predicate,
        params,
    ).fetchone()
    if run is None or run["status"] != "complete":
        raise RuntimeError("verified H=7 label run is missing or incomplete")
    status_counts = {
        row["status"]: row["n"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS n FROM label_flow WHERE " + predicate + " GROUP BY status ORDER BY status",
            params,
        )
    }
    label_counts = {
        str(row["label"]): row["n"]
        for row in connection.execute(
            "SELECT label, COUNT(*) AS n FROM label_flow WHERE " + predicate + " GROUP BY label ORDER BY label",
            params,
        )
    }
    totals = connection.execute(
        "SELECT COUNT(*) AS flow_rows, SUM(eligible) AS eligible, "
        "SUM(label=1) AS positives, SUM(label=0) AS negatives, "
        "SUM(eligible=1 AND label IS NULL) AS unknowns FROM label_flow WHERE " + predicate,
        params,
    ).fetchone()
    q2 = connection.execute(
        "SELECT COUNT(*) AS flow_rows, SUM(eligible) AS eligible, "
        "SUM(eligible=1 AND label IS NULL) AS unknowns, SUM(label=1) AS positives, SUM(label=0) AS negatives "
        "FROM label_flow WHERE " + predicate + " AND decision_date BETWEEN '2023-04-01' AND '2023-06-23'",
        params,
    ).fetchone()
    q2_eligible = int(q2["eligible"] or 0)
    q2_unknown = int(q2["unknowns"] or 0)
    q2_unknown_ratio = q2_unknown / q2_eligible if q2_eligible else 1.0

    failures: list[str] = []
    if run["manifest_hash"] != source_manifest_hash:
        failures.append("label run source manifest hash mismatch")
    if int(totals["flow_rows"] or 0) != connection.execute(
        "SELECT COUNT(*) FROM daily WHERE date BETWEEN ? AND ?", (START, END)
    ).fetchone()[0]:
        failures.append("label flow does not cover the scoring panel rows")
    if q2_unknown_ratio > 0.20:
        failures.append(f"Q2 unknown ratio exceeds 20%: {q2_unknown_ratio:.4f}")
    # SQLite's ``NULL != 1`` is NULL, not TRUE.  Use explicit predicates for
    # every status so a malformed NULL label or eligibility flag cannot pass
    # this gate merely because SQL's three-valued logic hides the mismatch.
    bad_status_labels = connection.execute(
        "SELECT COUNT(*) FROM label_flow WHERE " + predicate + " AND ("
        "eligible NOT IN (0, 1) OR "
        "(label IS NOT NULL AND label NOT IN (0, 1)) OR "
        "status NOT IN ('positive_observed','positive_with_gap','negative_observed',"
        "'end_censored','gap_or_exit_censored','history_insufficient','same_day_failure','post_failure') OR "
        "(status IN ('positive_observed','positive_with_gap') AND (eligible != 1 OR label IS NULL OR label != 1)) OR "
        "(status = 'negative_observed' AND (eligible != 1 OR label IS NULL OR label != 0)) OR "
        "(status IN ('end_censored','gap_or_exit_censored') AND (eligible != 1 OR label IS NOT NULL)) OR "
        "(status IN ('same_day_failure','post_failure','history_insufficient') AND (eligible != 0 OR label IS NOT NULL))"
        ")",
        params,
    ).fetchone()[0]
    if bad_status_labels:
        failures.append(f"status/label consistency failures={bad_status_labels}")
    negative_future_failure = connection.execute(
        "SELECT COUNT(*) FROM label_flow l WHERE " + predicate.replace("run_id", "l.run_id").replace("split", "l.split").replace("horizon_days", "l.horizon_days") + " AND l.label=0 "
        "AND EXISTS (SELECT 1 FROM daily d WHERE d.serial_number=l.serial_number AND d.failure=1 "
        "AND d.date > l.decision_date AND d.date <= date(l.decision_date, '+7 day'))",
        params,
    ).fetchone()[0]
    if negative_future_failure:
        failures.append(f"negative labels with future failure={negative_future_failure}")
    invalid_windows = connection.execute(
        "SELECT COUNT(*) FROM label_flow WHERE " + predicate + " AND ("
        "history_observations IS NULL OR history_observations < 0 OR history_observations > 14 OR "
        "future_observations IS NULL OR future_observations < 0 OR future_observations > 7"
        ")",
        params,
    ).fetchone()[0]
    if invalid_windows:
        failures.append(f"invalid history/future observation counts={invalid_windows}")
    eligible_partition = connection.execute(
        "SELECT SUM(eligible) AS eligible, SUM(status IN ('positive_observed','positive_with_gap','negative_observed','end_censored','gap_or_exit_censored')) AS expected FROM label_flow WHERE " + predicate,
        params,
    ).fetchone()
    if int(eligible_partition["eligible"] or 0) != int(eligible_partition["expected"] or 0):
        failures.append("eligible partition does not match locked statuses")

    monthly = [dict(row) for row in connection.execute(
        "SELECT substr(decision_date, 1, 7) AS month, status, label, COUNT(*) AS n FROM label_flow WHERE " + predicate + " GROUP BY month, status, label ORDER BY month, status, label",
        params,
    )]
    daily = [dict(row) for row in connection.execute(
        "SELECT decision_date, status, label, COUNT(*) AS n FROM label_flow WHERE " + predicate + " GROUP BY decision_date, status, label ORDER BY decision_date, status, label",
        params,
    )]
    summary = {
        "status": "pass" if not failures else "fail",
        "database": str(database),
        "run": dict(run),
        "source_manifest_hash": source_manifest_hash,
        "build_manifest_hash": build_manifest_hash,
        "build_manifest_file_sha256": build_manifest_file_sha256,
        "run_id": RUN_ID,
        "split": SPLIT,
        "horizon_days": HORIZON,
        "history_days": HISTORY_DAYS,
        "min_history": MIN_HISTORY,
        "status_counts": status_counts,
        "label_counts": label_counts,
        "totals": dict(totals),
        "q2_scoring_period": {
            "start": "2023-04-01",
            "end": "2023-06-23",
            "flow_rows": q2["flow_rows"],
            "eligible": q2_eligible,
            "unknown": q2_unknown,
            "unknown_ratio": q2_unknown_ratio,
            "positives": q2["positives"],
            "negatives": q2["negatives"],
            "acceptance_threshold": 0.20,
        },
        "monthly_counts": monthly,
        "daily_counts": daily,
        "checks": {
            "status_label_consistency_failures": bad_status_labels,
            "negative_future_failure": negative_future_failure,
            "invalid_window_counts": invalid_windows,
            "eligible_partition": dict(eligible_partition),
            "boundary_tests": _boundary_checks(),
        },
        "failures": failures,
        "training_approval": False,
    }
    _atomic_json(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--output", type=Path, default=EVIDENCE_DEFAULT)
    args = parser.parse_args()
    database = _relative_project_path(args.database, "database")
    manifest_path = _relative_project_path(args.manifest, "manifest")
    output = _relative_project_path(args.output, "label evidence")
    if not database.is_file():
        raise RuntimeError(f"verified panel is missing: {database}")
    manifest, _, manifest_file_sha256, build_manifest_hash = _load_manifest(manifest_path)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    metadata = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")}
    if metadata.get("build_status") != "panel_complete":
        connection.close()
        raise RuntimeError("panel must pass completion QA before labels are generated")
    if metadata.get("build_manifest_hash") != build_manifest_hash:
        connection.close()
        raise RuntimeError("panel was built from a different source manifest")
    build_labels._validate_source_facts(connection, DATASET_END)
    source_manifest_hash = build_labels._source_manifest_hash(connection)
    connection.close()

    _run_label_builder(database, source_manifest_hash)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        summary = _qa_labels(connection, database, output, source_manifest_hash, build_manifest_hash, manifest_file_sha256)
        status = summary["status"]
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("label_build_manifest_hash", build_manifest_hash),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("label_build_manifest_file_sha256", manifest_file_sha256),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("label_source_manifest_hash", source_manifest_hash),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("labels_status", "qa_pass" if status == "pass" else "qa_failed"),
        )
        connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("training_approval", "false"))
        connection.commit()
    finally:
        connection.close()
    print(json.dumps({"status": summary["status"], "run_id": RUN_ID, "q2_scoring_period": summary["q2_scoring_period"]}, ensure_ascii=False, indent=2))
    if summary["status"] != "pass":
        raise RuntimeError("verified label QA failed: " + "; ".join(summary["failures"][:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
