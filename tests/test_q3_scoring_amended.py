import datetime as dt
import sqlite3
import unittest

import numpy as np

from pipeline.score_q3_amended import (
    MODEL,
    SMART_FIELDS,
    ScoringStopped,
    average_precision_grouped,
    create_schema,
    feature_row_asof,
    paired_device_bootstrap,
    select_alert_indices,
)


def fixture_rows(days: int = 15) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset in range(days):
        row: dict[str, object] = {
            "date": (dt.date(2023, 7, 1) + dt.timedelta(days=offset)).isoformat(),
            "serial_number": "fixture",
            "model": MODEL,
            "failure": 0,
        }
        row.update({f"smart_{field}_raw": 10 + offset for field in SMART_FIELDS})
        rows.append(row)
    return rows


class AmendedScoringTests(unittest.TestCase):
    def test_signed_smart_decline_is_retained_and_audited_as_of_decision(self) -> None:
        rows = fixture_rows()
        rows[13] = dict(rows[13], smart_5_raw=2)
        result = feature_row_asof(rows[:14], rows[13]["date"])
        self.assertIsNotNone(result)
        values, audit = result  # type: ignore[misc]
        self.assertLess(float(values["smart_5_w7_delta_log1p"]), 0.0)
        self.assertEqual(audit["decrease_seen_asof_t_5"], 1)
        self.assertEqual(audit["window_crosses_decrease_w7_5"], 1)

    def test_future_observation_is_rejected(self) -> None:
        rows = fixture_rows()
        with self.assertRaises(ScoringStopped):
            feature_row_asof(rows[:14] + [rows[14]], rows[13]["date"])

    def test_alert_budget_and_cooldown_are_deterministic(self) -> None:
        rows = [{"serial_number": f"s{i}", "tie_break_sha256": f"{i:064x}"} for i in range(1500)]
        selected, excluded = select_alert_indices(rows, np.zeros(len(rows)), dt.date(2023, 7, 1), {})
        self.assertEqual(len(selected), 2)
        self.assertEqual(excluded, 0)
        last = {"s0": dt.date(2023, 7, 1)}
        selected, excluded = select_alert_indices(rows[:2], np.zeros(2), dt.date(2023, 7, 8), last)
        self.assertEqual(excluded, 1)
        self.assertEqual(len(selected), 1)

    def test_average_precision_groups_score_ties(self) -> None:
        self.assertAlmostEqual(average_precision_grouped([1.0, 1.0, 0.0], [1, 0, 1]), 7 / 12)
        self.assertIsNone(average_precision_grouped([0.0, 0.0], [0, 0]))

    def test_bootstrap_is_reproducible_and_paired(self) -> None:
        args = (["b", "a"], {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 1})
        first = paired_device_bootstrap(*args, replicates=100)
        second = paired_device_bootstrap(*args, replicates=100)
        self.assertEqual(first, second)
        self.assertEqual(first["valid_replicates"], 100)
        self.assertGreater(first["quantile_025"], 0.0)

    def test_output_schema_has_explicit_audit_and_score_columns(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            create_schema(connection)
            score_columns = connection.execute("PRAGMA table_info(model_scores)").fetchall()
            metric_columns = connection.execute("PRAGMA table_info(model_metrics)").fetchall()
            self.assertEqual(len(score_columns), 14)
            self.assertEqual(len(metric_columns), 31)
            self.assertIn("decrease_seen_any", {row[1] for row in connection.execute("PRAGMA table_info(feature_rows)")})
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
