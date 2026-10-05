"""Failure-injection checks for legacy migrations, using project-local fixtures."""
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from pipeline import panel


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MigrationAcceptanceTest(unittest.TestCase):
    def setUp(self):
        directory = ROOT / ".tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.connection = sqlite3.connect(pathlib.Path(self.temp.name) / "fixture.sqlite")

    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def seed_legacy(self):
        self.connection.execute("""CREATE TABLE label_flow (
            horizon_days INTEGER, decision_date TEXT, serial_number TEXT,
            model TEXT, capacity_bytes INTEGER, first_failure_date TEXT,
            label INTEGER, status TEXT, eligible INTEGER,
            history_observations INTEGER, future_observations INTEGER)""")
        self.connection.execute("""INSERT INTO label_flow VALUES
            (7,'2023-01-15','a','ST4000DM000',4000787030016,NULL,
             0,'negative',1,14,7)""")
        self.connection.commit()

    def test_interrupted_migration_preserves_old_table_and_can_retry(self):
        self.seed_legacy()
        original = panel._create_label_flow

        def fail_after_create(connection):
            original(connection)
            raise RuntimeError("injected interruption before legacy copy")

        with patch.object(panel, "_create_label_flow", side_effect=fail_after_create):
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                panel.create_schema(self.connection)
        self.connection.rollback()
        self.assertNotIn("run_id", panel._table_columns(self.connection, "label_flow"))
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM label_flow").fetchone()[0], 1)
        panel.create_schema(self.connection)
        self.assertEqual(self.connection.execute(
            "SELECT run_id, split, serial_number FROM label_flow"
        ).fetchall(), [("legacy", "legacy", "a")])
        panel.create_schema(self.connection)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM label_flow").fetchone()[0], 1)

    def test_legacy_identity_conflict_is_not_silently_collapsed(self):
        panel.create_schema(self.connection)
        columns = ("date,serial_number,model,failure,smart_5_missing,smart_9_missing,"
                   "smart_187_missing,smart_188_missing,smart_197_missing,smart_198_missing,"
                   "source_member,source_sha256,source_row")
        for date, model in [("2023-01-01", "A"), ("2023-01-02", "B")]:
            self.connection.execute(
                f"INSERT INTO daily({columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (date, "same", model, 0, 1, 1, 1, 1, 1, 1, "fixture", "fixture", 2),
            )
        self.connection.commit()
        with self.assertRaisesRegex(ValueError, "legacy serial/model identity conflict"):
            panel.create_schema(self.connection)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0], 2)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM serial_model_registry").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
