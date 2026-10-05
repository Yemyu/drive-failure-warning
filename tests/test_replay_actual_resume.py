"""Exercise real SQLite preflight, replay, interruption and re-entry."""
import datetime as dt
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import build_rule_replay_v2 as replay

ROOT = Path(__file__).resolve().parents[1]


def rows(path, sql):
    connection = replay._connect_immutable(path)
    try:
        return [tuple(row) for row in connection.execute(sql)]
    finally:
        connection.close()


def fixture(base):
    panel, feature = base / "panel.sqlite", base / "features.sqlite"
    connection = sqlite3.connect(panel)
    connection.executescript("""
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE daily(date TEXT,serial_number TEXT,model TEXT,failure INTEGER);
        CREATE TABLE label_runs(run_id TEXT,split TEXT,horizon_days INTEGER,
            start_date TEXT,end_date TEXT,dataset_end TEXT,status TEXT,
            config_hash TEXT,code_hash TEXT,manifest_hash TEXT);
        CREATE TABLE label_flow(run_id TEXT,split TEXT,horizon_days INTEGER,
            decision_date TEXT,serial_number TEXT,status TEXT,label INTEGER,first_failure_date TEXT);
    """)
    connection.executemany("INSERT INTO metadata VALUES (?,?)", {
        "build_status": "panel_complete", "training_approval": "false",
        "build_manifest_hash": "a" * 64, "label_build_manifest_hash": "a" * 64,
        "label_source_manifest_hash": "b" * 64,
    }.items())
    connection.execute("INSERT INTO label_runs VALUES (?,?,?,?,?,?,?,?,?,?)", (
        replay.LABEL_RUN_ID, "train", 7, "2023-01-15", "2023-01-21",
        replay.DATASET_END.isoformat(), "complete", "c" * 64, "d" * 64, "b" * 64))
    for day in range(1, 23):
        connection.execute("INSERT INTO daily VALUES (?,?,?,?)", (
            f"2023-01-{day:02}", "device", replay.MODEL, int(day == 22)))
    for day in range(15, 22):
        connection.execute("INSERT INTO label_flow VALUES (?,?,?,?,?,?,?,?)", (
            replay.LABEL_RUN_ID, "train", 7, f"2023-01-{day:02}",
            "device", "positive_observed", 1, "2023-01-22"))
    connection.commit()
    connection.close()
    connection = sqlite3.connect(feature)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
    connection.executemany("INSERT INTO metadata VALUES (?,?)", {
        "feature_status": "complete", "training_approval": "false",
        "source_panel_manifest_hash": "a" * 64, "feature_dictionary_hash": "e" * 64,
        "feature_config_hash": "f" * 64, "feature_code_sha256": "a" * 64,
    }.items())
    missing = ",".join(f"smart_{field}_current_missing INTEGER" for field in replay.SMART_FIELDS)
    connection.execute("CREATE TABLE feature_rows(decision_date TEXT,serial_number TEXT,"
                       "tie_break_sha256 TEXT,smart_nonzero_signal_count INTEGER,"
                       "smart_187_signal INTEGER,history_observations_14 INTEGER," + missing + ")")
    for day in range(15, 22):
        connection.execute("INSERT INTO feature_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            f"2023-01-{day:02}", "device", "1" * 64, 5, 1, 14, 0, 0, 0, 0, 0, 0))
    connection.commit()
    connection.close()
    return panel, feature


class ActualResumeTest(unittest.TestCase):
    def test_actual_interruption_resume_and_completed_reentry(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evidence/q2/closeout_evidence_review_v1",
                                         prefix="actual-resume-") as folder, patch.multiple(
            replay, SCORE_END=dt.date(2023, 1, 21), EVENT_END=dt.date(2023, 1, 22),
            EXPECTED_FEATURE_ROWS=7, EXPECTED_EVENT_TOTAL=1, EXPECTED_EVENT_OPPORTUNITY=1,
        ):
            base = Path(folder)
            panel, feature = fixture(base)
            clean, resumed = base / "clean.sqlite", base / "resumed.sqlite"
            evidence = base / "resumed-evidence"
            baseline = replay.build_replay(panel, feature, clean, base / "clean-evidence")
            original = replay._run_method

            def interrupt(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("injected after first method commit")

            with patch.object(replay, "_run_method", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    replay.build_replay(panel, feature, resumed, evidence)
            partial = resumed.with_name(resumed.name + ".partial")
            committed = rows(partial, "SELECT * FROM replay_runs")
            self.assertEqual(len(committed), 1)
            self.assertEqual(committed[0][1], replay.METHODS[0])
            # A method row is part of the resume contract too.  Corrupt it in
            # place and verify that the real entry point rejects before any
            # report or database mutation, then restore the fixture for the
            # successful resume below.
            partial_before = hashlib.sha256(partial.read_bytes()).hexdigest()
            connection = sqlite3.connect(partial)
            config = connection.execute(
                "SELECT value FROM metadata WHERE key='replay_config_hash'"
            ).fetchone()[0]
            connection.execute(
                "UPDATE replay_runs SET config_hash=? WHERE method=?",
                ("9" * 64, replay.METHODS[0]),
            )
            connection.commit(); connection.close()
            mismatch_before = hashlib.sha256(partial.read_bytes()).hexdigest()
            self.assertNotEqual(partial_before, mismatch_before)
            with self.assertRaises(replay.BuildStopped):
                replay.build_replay(panel, feature, resumed, evidence, resume=True)
            self.assertEqual(mismatch_before, hashlib.sha256(partial.read_bytes()).hexdigest())
            connection = sqlite3.connect(partial)
            connection.execute(
                "UPDATE replay_runs SET config_hash=? WHERE method=?",
                (config, replay.METHODS[0]),
            )
            connection.commit(); connection.close()
            with patch.object(replay, "_run_method", wraps=original) as spy:
                result = replay.build_replay(panel, feature, resumed, evidence, resume=True)
                self.assertEqual([call.args[3] for call in spy.call_args_list], list(replay.METHODS[1:]))
            self.assertEqual(result["methods"], baseline["methods"])
            self.assertEqual(committed, rows(resumed, "SELECT * FROM replay_runs WHERE method='random'"))
            for table in ("replay_daily", "replay_alerts", "replay_event_summary", "replay_strata"):
                self.assertEqual(rows(clean, f"SELECT * FROM {table} ORDER BY 1,2,3,4"),
                                 rows(resumed, f"SELECT * FROM {table} ORDER BY 1,2,3,4"))
            protected = (resumed, evidence / "preflight_v2.json", evidence / "replay_qa_v2.json")
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected]
            with patch.object(replay, "_run_method", wraps=original) as spy:
                result = replay.build_replay(panel, feature, resumed, evidence)
                self.assertEqual(result["status"], "already_complete")
                spy.assert_not_called()
            sha = replay._sha256_file
            def changed_selection(path):
                return "0" * 64 if path.name == "replay_selection.py" else sha(path)
            with patch.object(replay, "_sha256_file", side_effect=changed_selection):
                with self.assertRaises(replay.BuildStopped):
                    replay.build_replay(panel, feature, resumed, evidence)
            connection = sqlite3.connect(panel)
            connection.execute("UPDATE label_runs SET config_hash=?", ("9" * 64,))
            connection.commit()
            connection.close()
            with self.assertRaises(replay.BuildStopped):
                replay.build_replay(panel, feature, resumed, evidence)
            self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in protected])

    def test_real_entry_stops_from_sql_progress_gate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "evidence/q2/closeout_evidence_review_v1",
                                         prefix="actual-resource-") as folder, patch.multiple(
            replay, SCORE_END=dt.date(2023, 1, 21), EVENT_END=dt.date(2023, 1, 22),
            EXPECTED_FEATURE_ROWS=7, EXPECTED_EVENT_TOTAL=1, EXPECTED_EVENT_OPPORTUNITY=1,
            SOFT_TIMEOUT_SECONDS=-1,
        ):
            base = Path(folder)
            panel, feature = fixture(base)
            output = base / "resource.sqlite"
            evidence = base / "resource-evidence"
            with self.assertRaises(replay.BuildStopped):
                replay.build_replay(panel, feature, output, evidence)
            partial = output.with_name(output.name + ".partial")
            self.assertTrue(partial.exists())
            metadata = dict(rows(partial, "SELECT key,value FROM metadata"))
            self.assertEqual(metadata["replay_status"], "failed")
            self.assertTrue(list(evidence.glob("replay_attempt_failed_*.json")))


if __name__ == "__main__":
    unittest.main()
