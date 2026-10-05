import datetime as dt
import unittest

from pipeline.replay_selection import (
    ceil_budget,
    median,
    minimum_gap,
    quantile,
    rank_candidates,
    select_alerts,
)


def row(serial, tie, *, nonzero=0, smart187=0, score=0.0):
    return {
        "serial_number": serial,
        "tie_break_sha256": tie,
        "smart_nonzero_signal_count": nonzero,
        "smart_187_signal": smart187,
        "score": score,
    }


class ReplaySelectionTest(unittest.TestCase):
    def test_budget_boundaries(self):
        self.assertEqual([ceil_budget(n) for n in (0, 1, 1000, 1001)], [0, 1, 1, 2])

    def test_signal_shortage_is_not_padded(self):
        rows = [row("a", "a", nonzero=1), row("b", "b")]
        selected, stats = select_alerts(rows, "smart_nonzero", dt.date(2023, 1, 1), {})
        self.assertEqual([item["serial_number"] for item in selected], ["a"])
        self.assertEqual(stats["signal_count"], 1)
        self.assertEqual(stats["alerts_count"], 1)

    def test_cooldown_edges_and_full_candidate_count(self):
        rows = [row("a", "a", nonzero=1), row("b", "b", nonzero=1)]
        last = {"a": dt.date(2023, 1, 1)}
        selected, stats = select_alerts(rows, "smart_nonzero", dt.date(2023, 1, 8), last)
        self.assertEqual({item["serial_number"] for item in selected}, {"b"})
        self.assertEqual(stats["cooldown_excluded"], 1)
        # t+7 is still excluded; t+8 is available.
        last = {"a": dt.date(2023, 1, 1)}
        selected, _ = select_alerts(rows, "smart_nonzero", dt.date(2023, 1, 9), last)
        self.assertEqual({item["serial_number"] for item in selected}, {"a"})

    def test_rankings_and_no_signal(self):
        rows = [row("b", "2", nonzero=1), row("a", "1", nonzero=0, smart187=1), row("c", "3", nonzero=2)]
        self.assertEqual([r["serial_number"] for r in rank_candidates(rows, "random")], ["a", "b", "c"])
        self.assertEqual([r["serial_number"] for r in rank_candidates(rows, "smart_nonzero")], ["c", "b"])
        self.assertEqual([r["serial_number"] for r in rank_candidates(rows, "smart187")], ["a"])

    def test_current_lr_ranking_is_score_then_tie(self):
        rows = [row("serial-b", "b", score=1.0), row("serial-a", "a", score=1.0), row("serial-c", "c", score=2.0)]
        self.assertEqual([item["serial_number"] for item in rank_candidates(rows, "current_lr")], ["serial-c", "serial-a", "serial-b"])

    def test_summary_statistics(self):
        self.assertEqual(median([]), None)
        self.assertEqual(median([3]), 3)
        self.assertEqual(median([3, 4]), 3.5)
        self.assertEqual(quantile([], 0.75), None)
        self.assertEqual(quantile([1], 0.75), 1)
        self.assertEqual(quantile([1, 2, 4, 8], 0.25), 1.75)
        self.assertEqual(minimum_gap([dt.date(2023, 1, 1)]), None)
        self.assertEqual(minimum_gap([dt.date(2023, 1, 1), dt.date(2023, 1, 4), dt.date(2023, 1, 7)]), 3)


if __name__ == "__main__":
    unittest.main()
