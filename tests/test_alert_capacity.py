import datetime as dt
import unittest

from investigation.build_alert_capacity_comparison import choose_alerts, ceil_budget


class CapacityReplayTests(unittest.TestCase):
    def test_budget_boundaries(self):
        self.assertEqual([ceil_budget(n, 1000) for n in (0, 1, 1000, 1001)], [0, 1, 1, 2])

    def test_cooldown_t_plus_seven_and_t_plus_eight(self):
        rows = [
            {"serial_number": "a", "tie_break_sha256": "a", "current_lr_score": 2, "history_lr_score": 2, "history_hgb_score": 2},
            {"serial_number": "b", "tie_break_sha256": "b", "current_lr_score": 1, "history_lr_score": 1, "history_hgb_score": 1},
        ]
        last = {"a": dt.date(2023, 7, 1)}
        selected, stats = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 8), last)
        self.assertEqual([row["serial_number"] for row in selected], ["b"])
        self.assertEqual(stats["cooldown_excluded"], 1)
        selected, _ = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 9), last)
        self.assertEqual([row["serial_number"] for row in selected], ["a"])

    def test_capacity_paths_keep_cooldown_state_separate(self):
        rows = [{"serial_number": "a", "tie_break_sha256": "a", "current_lr_score": 2, "history_lr_score": 2, "history_hgb_score": 2}]
        first, _ = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 1), {})
        path_a = {"a": dt.date(2023, 7, 1)}
        path_b = {}
        second_a, _ = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 2), path_a)
        second_b, _ = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 2), path_b)
        self.assertEqual([row["serial_number"] for row in first], ["a"])
        self.assertEqual(second_a, [])
        self.assertEqual([row["serial_number"] for row in second_b], ["a"])

    def test_same_score_uses_tie_break_then_serial(self):
        rows = [
            {"serial_number": "b", "tie_break_sha256": "2", "current_lr_score": 1, "history_lr_score": 1, "history_hgb_score": 1},
            {"serial_number": "a", "tie_break_sha256": "1", "current_lr_score": 1, "history_lr_score": 1, "history_hgb_score": 1},
        ]
        selected, _ = choose_alerts(rows, "current_lr", 1000, dt.date(2023, 7, 1), {})
        self.assertEqual([row["serial_number"] for row in selected], ["a"])


if __name__ == "__main__":
    unittest.main()
