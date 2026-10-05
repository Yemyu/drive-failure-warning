"""Regression checks for the finite v2 baseline repair contract."""

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from pipeline import build_simple_baselines_v2 as baseline


ROOT = Path(__file__).resolve().parents[1]


class SimpleBaselineV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/simple_baseline_v2.json").read_text(encoding="utf-8"))

    def test_average_precision_groups_exact_ties(self):
        first = baseline.average_precision(np.asarray([1.0, 1.0, 0.0]), np.asarray([1, 0, 1], dtype=np.int8))
        reversed_tie = baseline.average_precision(np.asarray([1.0, 1.0, 0.0]), np.asarray([0, 1, 1], dtype=np.int8))
        self.assertAlmostEqual(first, (0.5 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(reversed_tie, first)
        self.assertIsNone(baseline.average_precision(np.asarray([1.0, 0.0]), np.zeros(2, dtype=np.int8)))
        with self.assertRaises(baseline.BaselineStopped):
            baseline.average_precision(np.asarray([np.inf]), np.asarray([1], dtype=np.int8))

    def test_missing_indicator_is_materialized_from_source_null(self):
        row = {"nullable": None, "observed": 4.0}
        self.assertEqual(baseline.row_value(row, "nullable__missing", {"nullable__missing": "nullable"}), 1.0)
        row["nullable"] = 4.0
        self.assertEqual(baseline.row_value(row, "nullable__missing", {"nullable__missing": "nullable"}), 0.0)
        self.assertEqual(baseline.row_value(row, "observed"), 4.0)

    def test_real_feature_spec_has_locked_dimensions(self):
        connection = baseline.connect_immutable(baseline.FEATURE_DB)
        try:
            columns = [row["name"] for row in connection.execute("PRAGMA table_info(feature_rows)")]
            spec = baseline.resolve_feature_spec(self.config, columns)
        finally:
            connection.close()
        self.assertEqual(spec["dimensions"], {"age_lr": 2, "current_lr": 16, "history_lr": 122})
        self.assertEqual(len(spec["synthetic_indicators"]), 46)
        self.assertEqual(len(spec["missing_indicator_names"]), 52)

    def test_spec_rejects_missing_indicator_mapping(self):
        malformed = copy.deepcopy(self.config)
        malformed["missing_indicator_sources"].pop(next(iter(malformed["missing_indicator_sources"])))
        connection = baseline.connect_immutable(baseline.FEATURE_DB)
        try:
            columns = [row["name"] for row in connection.execute("PRAGMA table_info(feature_rows)")]
        finally:
            connection.close()
        with self.assertRaises(baseline.BaselineStopped):
            baseline.resolve_feature_spec(malformed, columns)

    def test_v1_reuse_payload_is_read_only_and_dimension_matched(self):
        spec = self.config["base_models"]
        payload = baseline.load_v1_payload("age_lr", spec["age_lr"], self.config["models"]["age_lr"])
        self.assertEqual(payload["fit_mode"], "reused_v1")
        self.assertEqual(payload["feature_count"], 2)
        self.assertEqual(len(payload["coef"]), 2)
        self.assertEqual(payload["source_code_sha256"], baseline.V1_CODE_SHA256)

    def test_child_rss_helper_is_nonnegative(self):
        self.assertGreaterEqual(baseline.process_rss_bytes(baseline.os.getpid()), 0)


if __name__ == "__main__":
    unittest.main()
