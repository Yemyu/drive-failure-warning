"""Small deterministic checks for the locked simple-baseline helpers."""

import datetime as dt
import unittest

import numpy as np

from pipeline import build_simple_baselines_v1 as baseline


class SimpleBaselineHelpersTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_uses_declared_count(self):
        selected_a, digest_a = baseline.sample_negative_indices("2023-01-15", [f"d{i}" for i in range(1000)], 1000)
        selected_b, digest_b = baseline.sample_negative_indices("2023-01-15", [f"d{i}" for i in range(1000)], 1000)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(len(selected_a), 50)
        selected_small, _ = baseline.sample_negative_indices("2023-01-15", [f"d{i}" for i in range(19)], 19)
        self.assertEqual(len(selected_small), 1)

    def test_preprocessing_handles_missing_and_constant_columns(self):
        raw = np.asarray([[1.0, np.nan, 3.0], [2.0, 4.0, 3.0], [3.0, 6.0, 3.0]])
        transformed, params = baseline.weighted_preprocess(raw, np.asarray([1.0, 2.0, 1.0]))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(params["observed_count"], [3, 2, 3])
        self.assertEqual(params["standardization_scale"][2], 1.0)

    def test_budget_and_cooldown_boundaries(self):
        rows = [{"serial_number": f"d{i}", "tie_break_sha256": f"{i:064x}"} for i in range(1001)]
        scores = np.linspace(1.0, 0.0, len(rows))
        last = {}
        selected, _ = baseline.select_alert_indices(scores, rows, dt.date(2023, 1, 1), last)
        self.assertEqual(len(selected), 2)
        one = [{"serial_number": "disk", "tie_break_sha256": "0" * 64}]
        state = {}
        baseline.select_alert_indices(np.asarray([1.0]), one, dt.date(2023, 1, 1), state)
        self.assertEqual(baseline.select_alert_indices(np.asarray([1.0]), one, dt.date(2023, 1, 8), state)[0], [])
        self.assertEqual(baseline.select_alert_indices(np.asarray([1.0]), one, dt.date(2023, 1, 9), state)[0], [0])

    def test_average_precision_has_known_outcome_scope(self):
        scores = np.asarray([0.9, 0.8, 0.1, 0.0])
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        self.assertAlmostEqual(baseline.average_precision(scores, labels), (1.0 + 2.0 / 3.0) / 2.0)
        self.assertIsNone(baseline.average_precision(scores, np.zeros(4, dtype=np.int8)))


if __name__ == "__main__":
    unittest.main()
