import datetime as dt
import unittest

import numpy as np

from pipeline.build_nonlinear_baseline_v1 import (
    ExperimentStopped,
    average_precision_grouped,
    paired_bootstrap,
    rank_indices,
    row_matrix,
    select_indices,
)


class NonlinearBaselinePureTests(unittest.TestCase):
    def test_tie_break_and_cooldown_boundary(self):
        rows = [
            {"serial_number": "B", "tie_break_sha256": "b"},
            {"serial_number": "A", "tie_break_sha256": "a"},
        ]
        self.assertEqual(rank_indices(rows, [1.0, 1.0]), [1, 0])
        last = {}
        self.assertEqual(select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 1), last, 2, 7)[0], [0])
        self.assertEqual(select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 8), last, 2, 7)[0], [1])
        self.assertEqual(select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 9), last, 2, 7)[0], [0])

    def test_average_precision_groups_equal_scores(self):
        self.assertAlmostEqual(average_precision_grouped([0.9, 0.9, 0.1], [1, 0, 0]), 0.5)

    def test_device_bootstrap_is_deterministic(self):
        first = paired_bootstrap(["b", "a"], {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 1}, 20260913, 20)
        second = paired_bootstrap(["a", "b"], {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 1}, 20260913, 20)
        self.assertEqual(first, second)
        self.assertEqual(first["valid_replicates"], 20)

    def test_missing_mapping_preserves_physical_flag_and_derives_history_flag(self):
        mapping = [
            {"feature_name": "smart_5_current_missing", "source_name": "smart_5_current_missing", "is_missing_indicator": 1},
            {"feature_name": "smart_5_w7_mean_log1p__missing", "source_name": "smart_5_w7_mean_log1p", "is_missing_indicator": 1},
        ]
        actual = row_matrix([{"smart_5_current_missing": 1, "smart_5_w7_mean_log1p": None}], mapping)
        np.testing.assert_array_equal(actual, [[1.0, 1.0]])

    def test_missing_mapping_rejects_invalid_physical_flag(self):
        mapping = [{"feature_name": "smart_5_current_missing", "source_name": "smart_5_current_missing", "is_missing_indicator": 1}]
        with self.assertRaises(ExperimentStopped):
            row_matrix([{"smart_5_current_missing": 2}], mapping)


if __name__ == "__main__":
    unittest.main()
