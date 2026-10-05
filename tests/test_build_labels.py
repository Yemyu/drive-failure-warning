import pathlib
import contextlib
import io
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline import build_labels
from pipeline import build_verified_labels


PROJECT_TMP = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"


def _seed_panel(database: pathlib.Path) -> None:
    connection = sqlite3.connect(database)
    from pipeline.panel import create_schema

    create_schema(connection)
    values = []
    source_hash = "a" * 64
    schema_hash = "b" * 64
    for day in range(1, 23):
        date_text = f"2023-01-{day:02d}"
        values.append(
            (
                date_text, "disk-1", "ST4000DM000", 4000787030016, 0,
                0, 100 + day, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                f"fixture/{date_text}.csv", source_hash, day,
            )
        )
    columns = (
        "date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_9_raw,"
        "smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw,smart_5_missing,"
        "smart_9_missing,smart_187_missing,smart_188_missing,smart_197_missing,"
        "smart_198_missing,source_member,source_sha256,source_row"
    )
    connection.executemany(
        f"INSERT INTO daily({columns}) VALUES ({','.join('?' for _ in range(20))})",
        values,
    )
    connection.executemany(
        """
        INSERT INTO member_counts(
            date, source_member, source_sha256, source_rows, selected_rows,
            failure_rows, schema_columns, schema_sha256, schema_json, audit_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                f"2023-01-{day:02d}", f"fixture/2023-01-{day:02d}.csv", source_hash,
                1, 1, 0, 11, schema_hash, "[]", "{}",
            )
            for day in range(1, 23)
        ],
    )
    connection.commit()
    connection.close()


class BuildLabelsTest(unittest.TestCase):
    def test_verified_label_qa_rejects_null_label_for_positive_status(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE daily (date TEXT, serial_number TEXT, failure INTEGER);
            CREATE TABLE label_runs (
                run_id TEXT, split TEXT, horizon_days INTEGER, start_date TEXT,
                end_date TEXT, dataset_end TEXT, config_hash TEXT, code_hash TEXT,
                manifest_hash TEXT, status TEXT, summary_json TEXT
            );
            CREATE TABLE label_flow (
                run_id TEXT, split TEXT, horizon_days INTEGER, decision_date TEXT,
                serial_number TEXT, status TEXT, label INTEGER, eligible INTEGER,
                history_observations INTEGER, future_observations INTEGER,
                first_failure_date TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO daily VALUES ('2023-04-01', 'disk-1', 0)"
        )
        connection.execute(
            "INSERT INTO label_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                build_verified_labels.RUN_ID,
                build_verified_labels.SPLIT,
                build_verified_labels.HORIZON,
                build_verified_labels.START,
                build_verified_labels.END,
                build_verified_labels.DATASET_END,
                "config",
                "code",
                "source",
                "complete",
                "{}",
            ),
        )
        connection.execute(
            "INSERT INTO label_flow VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                build_verified_labels.RUN_ID,
                build_verified_labels.SPLIT,
                build_verified_labels.HORIZON,
                "2023-04-01",
                "disk-1",
                "positive_observed",
                None,
                1,
                12,
                7,
                "2023-04-03",
            ),
        )
        try:
            with patch.object(build_verified_labels, "_atomic_json"):
                result = build_verified_labels._qa_labels(
                    connection,
                    pathlib.Path(":memory:"),
                    pathlib.Path("unused.json"),
                    "source",
                    "build",
                    "file",
                )
            self.assertEqual(result["status"], "fail")
            self.assertGreater(result["checks"]["status_label_consistency_failures"], 0)
            self.assertTrue(any("status/label consistency" in item for item in result["failures"]))
        finally:
            connection.close()

    def test_source_date_with_positive_count_cannot_disappear_from_daily(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / 'panel.sqlite'
            _seed_panel(database)
            connection = sqlite3.connect(database)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("DELETE FROM daily WHERE date='2023-01-10'")
                with self.assertRaisesRegex(RuntimeError, 'daily is missing dates'):
                    build_labels._validate_source_facts(connection, '2023-01-22')
                # A genuinely zero-selected source day is allowed and remains
                # available to observation/coverage accounting.
                connection.execute("UPDATE member_counts SET selected_rows=0 WHERE date='2023-01-10'")
                build_labels._validate_source_facts(connection, '2023-01-22')
            finally:
                connection.close()

    def test_source_rejection_rolls_back_legacy_schema_migration(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / "panel.sqlite"
            _seed_panel(database)
            connection = sqlite3.connect(database)
            connection.execute("ALTER TABLE daily DROP COLUMN capacity_clean_bytes")
            connection.execute("ALTER TABLE daily DROP COLUMN capacity_missing")
            connection.execute("DELETE FROM member_counts")
            connection.commit()
            before_columns = [row[1] for row in connection.execute("PRAGMA table_info(daily)")]
            before_master = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            connection.close()
            argv = [
                "build_labels", "--database", str(database), "--start", "2023-01-15",
                "--end", "2023-01-16", "--dataset-end", "2023-01-23",
                "--run-id", "reject-migration", "--split", "train",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    build_labels.main()
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    [row[1] for row in connection.execute("PRAGMA table_info(daily)")],
                    before_columns,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall(),
                    before_master,
                )
            finally:
                connection.close()

    def test_empty_source_facts_rejected_and_diagnostic_escape_is_explicit(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / "panel.sqlite"
            _seed_panel(database)
            connection = sqlite3.connect(database)
            connection.execute("DELETE FROM member_counts")
            connection.commit()
            before = {
                table: connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
                for table in ("label_flow", "label_runs", "metadata")
            }
            connection.close()
            argv = [
                "build_labels", "--database", str(database), "--start", "2023-01-15",
                "--end", "2023-01-16", "--dataset-end", "2023-01-23",
                "--run-id", "unverified", "--split", "train",
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(RuntimeError):
                    build_labels.main()
            connection = sqlite3.connect(database)
            try:
                after = {
                    table: connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
                    for table in ("label_flow", "label_runs", "metadata")
                }
            finally:
                connection.close()
            self.assertEqual(after, before)
            argv.append("--allow-unverified-manifest")
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(build_labels.main(), 0)
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT manifest_hash FROM label_runs WHERE run_id='unverified'"
                    ).fetchone()[0],
                    "UNVERIFIED",
                )
            finally:
                connection.close()

    def test_repeat_replace_and_midrun_failure_are_atomic(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / "panel.sqlite"
            _seed_panel(database)

            def run(run_id="a", start="2023-01-15", extra=()):
                argv = ["build_labels", "--database", str(database), "--start", start,
                        "--end", "2023-01-16", "--dataset-end", "2023-01-23",
                        "--run-id", run_id,
                        "--split", "train", *extra]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    build_labels.main()

            def snapshot():
                connection = sqlite3.connect(database)
                try:
                    return {
                        table: connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
                        for table in ("label_flow", "label_runs", "metadata")
                    }
                finally:
                    connection.close()

            run()
            before = snapshot()
            run()
            after = snapshot()
            self.assertEqual(before["label_flow"], after["label_flow"])
            self.assertEqual(before["metadata"], after["metadata"])
            # A retry may update its execution timestamp, but not its declared inputs/results.
            self.assertEqual([r[:-1] for r in before["label_runs"]],
                             [r[:-1] for r in after["label_runs"]])
            run("b")
            before = snapshot()
            original = build_labels.classify_device_rows

            def fail_during_insert(*args, **kwargs):
                rows = original(*args, **kwargs)
                # First insert succeeds; the duplicate primary key fails the second.
                return [rows[0], rows[0]]

            with patch.object(build_labels, "classify_device_rows", side_effect=fail_during_insert):
                with self.assertRaises(sqlite3.IntegrityError):
                    run(start="2023-01-16", extra=("--replace",))
            self.assertEqual(before, snapshot())
            run(start="2023-01-16", extra=("--replace",))
            after = snapshot()
            for table in ("label_flow", "label_runs"):
                self.assertEqual([r for r in before[table] if r[0] == "b"],
                                 [r for r in after[table] if r[0] == "b"])
            self.assertEqual([r[4] for r in after["label_flow"] if r[0] == "a"], ["2023-01-16"])

    def test_runs_are_isolated_by_split_and_run_id(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / "panel.sqlite"
            _seed_panel(database)
            common = [
                "--database", str(database), "--start", "2023-01-15", "--end", "2023-01-15",
                "--dataset-end", "2023-01-22", "--horizon", "7", "--history-days", "14",
                "--min-history", "12",
            ]
            with patch.object(sys, "argv", ["build_labels", *common, "--split", "train", "--run-id", "train-a"]):
                self.assertEqual(build_labels.main(), 0)
            with patch.object(sys, "argv", ["build_labels", *common, "--split", "validation", "--run-id", "val-a"]):
                self.assertEqual(build_labels.main(), 0)
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT run_id, split, COUNT(*) FROM label_flow GROUP BY run_id, split ORDER BY run_id"
                    ).fetchall(),
                    [("train-a", "train", 1), ("val-a", "validation", 1)],
                )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM label_runs").fetchone()[0], 2)
            finally:
                connection.close()

    def test_changed_scope_is_rejected_without_mutating_existing_run(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            database = pathlib.Path(directory) / "panel.sqlite"
            _seed_panel(database)
            common = [
                "--database", str(database), "--dataset-end", "2023-01-23",
                "--horizon", "7", "--history-days", "14", "--min-history", "12",
                "--split", "train", "--run-id", "same-run",
            ]
            with patch.object(sys, "argv", ["build_labels", *common, "--start", "2023-01-15", "--end", "2023-01-15"]):
                build_labels.main()
            connection = sqlite3.connect(database)
            try:
                before = connection.execute(
                    "SELECT decision_date, config_hash FROM label_flow ORDER BY decision_date"
                ).fetchall()
            finally:
                connection.close()
            with patch.object(sys, "argv", ["build_labels", *common, "--start", "2023-01-16", "--end", "2023-01-16"]):
                with self.assertRaises(RuntimeError):
                    build_labels.main()
            connection = sqlite3.connect(database)
            try:
                after = connection.execute(
                    "SELECT decision_date, config_hash FROM label_flow ORDER BY decision_date"
                ).fetchall()
                declared = connection.execute(
                    "SELECT start_date, end_date FROM label_runs WHERE run_id='same-run'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(after, before)
            self.assertEqual(declared, ("2023-01-15", "2023-01-15"))


if __name__ == "__main__":
    unittest.main()
