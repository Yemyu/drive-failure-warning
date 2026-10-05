import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from investigation.feature_replay_closeout_v1 import values_equal
from pipeline import build_rule_replay_v2 as replay


ROOT = Path(__file__).resolve().parents[1]


class ReplaySafetyTest(unittest.TestCase):
    def test_strict_float_tolerance_near_zero(self):
        self.assertFalse(values_equal(0.0, 5e-11))
        self.assertTrue(values_equal(1.0, 1.0 + 5e-11))
        self.assertTrue(values_equal(None, None))
        self.assertFalse(values_equal(None, 0.0))

    def test_integer_values_are_exact_even_at_large_magnitudes(self):
        self.assertFalse(values_equal(10**12, 10**12 + 1))
        self.assertTrue(values_equal(10**12, float(10**12)))
        self.assertFalse(values_equal(10**12, float(10**12) + 0.5))

    def test_resume_config_mismatch_preserves_partial_bytes(self):
        evidence = ROOT / "evidence/q2/closeout_evidence_review_v1"
        with tempfile.TemporaryDirectory(prefix="resume-test-", dir=evidence) as folder:
            base = Path(folder)
            target = base / "fixture.sqlite"
            partial = base / "fixture.sqlite.partial"
            connection = sqlite3.connect(partial)
            replay._ddl(connection)
            connection.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [("replay_config_hash", "old"), ("replay_status", "running")],
            )
            connection.commit()
            connection.close()
            before = hashlib.sha256(partial.read_bytes()).digest()
            panel = sqlite3.connect(":memory:")
            feature = sqlite3.connect(":memory:")
            try:
                with patch.object(
                    replay,
                    "_preflight",
                    return_value=(panel, feature, {}, {}, "different", {}),
                ):
                    with self.assertRaises(replay.BuildStopped):
                        replay.build_replay(
                            base / "panel.sqlite",
                            base / "feature.sqlite",
                            target,
                            base,
                            resume=True,
                        )
            finally:
                panel.close()
                feature.close()
            self.assertEqual(before, hashlib.sha256(partial.read_bytes()).digest())
            metadata = dict(sqlite3.connect(partial).execute("SELECT key,value FROM metadata"))
            self.assertEqual(metadata, {"replay_config_hash": "old", "replay_status": "running"})

    def test_completed_replay_is_idempotent_when_config_matches(self):
        with tempfile.TemporaryDirectory(prefix="complete-test-", dir=ROOT / "evidence/q2/closeout_evidence_review_v1") as folder:
            path = Path(folder) / "complete.sqlite"
            connection = sqlite3.connect(path)
            replay._ddl(connection)
            config = "config"
            connection.execute("INSERT INTO metadata VALUES (?,?)", ("replay_config_hash", config))
            connection.execute("INSERT INTO metadata VALUES (?,?)", ("replay_status", "complete"))
            for method in replay.METHODS:
                summary = {"method": method, "status": "pass"}
                connection.execute(
                    "INSERT INTO replay_runs VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        "rule_replay_train_v2", method, config, "dictionary", "panel",
                        "train_q1q2_verified_h7_v1", "complete", json.dumps(summary), "now",
                    ),
                )
            connection.commit()
            before = hashlib.sha256(path.read_bytes()).digest()
            result = replay._completed_replay_result(path, config)
            self.assertEqual(result["status"], "already_complete")
            self.assertEqual(set(result["methods"]), set(replay.METHODS))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
