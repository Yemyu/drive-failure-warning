"""Small contract tests for the independent v2 closeout consumer."""

import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from investigation import audit_simple_baseline_v2_closeout as audit


ROOT = Path(__file__).resolve().parents[1]


class SimpleBaselineV2CloseoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = audit.load_json(audit.CONFIG_PATH)

    def test_locked_config_rejects_unknown_field(self):
        candidate = copy.deepcopy(self.config)
        candidate["unexpected"] = True
        with self.assertRaises(audit.AuditFailure):
            audit.strict_config(candidate)

    def test_locked_config_rejects_horizon_change(self):
        candidate = copy.deepcopy(self.config)
        candidate["horizon_days"] = 14
        with self.assertRaises(audit.AuditFailure):
            audit.strict_config(candidate)

    def test_sampling_hash_is_deterministic_and_tie_safe(self):
        selected, digest = audit.sample_negative("2023-01-15", ["B", "A", "C"])
        selected_again, digest_again = audit.sample_negative("2023-01-15", ["C", "A", "B"])
        self.assertEqual(selected, selected_again)
        self.assertEqual(digest, digest_again)

    def test_rank_alerts_keeps_model_cooldowns_separate(self):
        rows = [
            {"decision_date": "2023-01-15", "serial_number": "A", "tie_break_sha256": "a", "history_observations_14": 14, "smart_5_current_missing": 0, "smart_9_current_missing": 0, "smart_187_current_missing": 0, "smart_188_current_missing": 0, "smart_197_current_missing": 0, "smart_198_current_missing": 0},
            {"decision_date": "2023-01-15", "serial_number": "B", "tie_break_sha256": "b", "history_observations_14": 14, "smart_5_current_missing": 0, "smart_9_current_missing": 0, "smart_187_current_missing": 0, "smart_188_current_missing": 0, "smart_197_current_missing": 0, "smart_198_current_missing": 0},
        ]
        scores = {model: np.asarray([2.0, 1.0]) for model in audit.MODELS}
        last = {model: {} for model in audit.MODELS}
        selected, _cooldown = audit.rank_alerts(rows, scores, last)
        self.assertEqual(len(selected["age_lr"]), 1)
        self.assertEqual(len(selected["current_lr"]), 1)
        self.assertEqual(len(selected["history_lr"]), 1)

    def test_linear_quantile_matches_locked_rule(self):
        self.assertEqual(audit.linear_quantile([1, 2, 4, 7], 0.5), 3.0)
        self.assertEqual(audit.linear_quantile([], 0.5), None)

    def test_reentry_rejects_bound_file_change(self):
        current_root = audit.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit.ROOT = root
            try:
                for relative, content in {
                    "artifact.bin": b"original\n",
                    "evidence/simple_baseline_v2/input_freeze_v1.json": b"freeze\n",
                    "evidence/q2/fit_review_v1/review_v1.json": b"source\n",
                    "evidence/simple_baseline_v2/post_stage_checks_v1.json": b"checks\n",
                    "evidence/simple_baseline_v2/run_v1.json": b"run\n",
                }.items():
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                report_path = root / "audit_report.json"
                report = {
                    "status": "pass",
                    "files": {"artifact": {"path": "artifact.bin", "bytes": 9, "sha256": hashlib.sha256(b"original\n").hexdigest()}},
                    "input_snapshots": {
                        "v2_freeze": audit.file_sha256(root / "evidence/simple_baseline_v2/input_freeze_v1.json"),
                        "source_freeze": audit.file_sha256(root / "evidence/q2/fit_review_v1/review_v1.json"),
                        "post_checks": audit.file_sha256(root / "evidence/simple_baseline_v2/post_stage_checks_v1.json"),
                        "run_evidence": audit.file_sha256(root / "evidence/simple_baseline_v2/run_v1.json"),
                    },
                }
                report_path.write_text(json.dumps(report), encoding="utf-8")
                manifest_path = root / "complete_manifest.json"
                manifest_path.write_text(json.dumps({"status": "complete", "audit_report": "audit_report.json", "audit_report_sha256": audit.file_sha256(report_path), "audit_code_sha256": audit.file_sha256(Path(audit.__file__))}), encoding="utf-8")
                audit.validate_complete_manifest(manifest_path)
                (root / "artifact.bin").write_bytes(b"tampered\n")
                with self.assertRaises(audit.AuditFailure):
                    audit.validate_complete_manifest(manifest_path)
            finally:
                audit.ROOT = current_root


if __name__ == "__main__":
    unittest.main()
