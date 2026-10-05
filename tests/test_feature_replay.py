import datetime as dt
import math
import unittest

from pipeline import build_feature_replay


def _rows(days=14, *, missing_day=None, failure_day=None):
    rows = []
    for offset in range(days):
        day = dt.date(2023, 1, 1) + dt.timedelta(days=offset)
        if day == missing_day:
            continue
        row = {
            "date": day.isoformat(),
            "serial_number": "disk-1",
            "model": "ST4000DM000",
            "failure": int(day == failure_day),
        }
        for field in build_feature_replay.SMART_FIELDS:
            row[f"smart_{field}_raw"] = offset + 1
        rows.append(row)
    return rows


class FeatureReplayTest(unittest.TestCase):
    def test_feature_row_is_historical_and_schema_complete(self):
        decision = "2023-01-14"
        base = _rows()
        future = _rows() + [
            {
                "date": "2023-01-15",
                "serial_number": "disk-1",
                "model": "ST4000DM000",
                "failure": 0,
                **{f"smart_{field}_raw": 999999 for field in build_feature_replay.SMART_FIELDS},
            }
        ]
        first = build_feature_replay.feature_row_for_serial(base, decision)
        self.assertIsNotNone(first)
        with self.assertRaises(build_feature_replay.BuildStopped):
            build_feature_replay.feature_row_for_serial(future, decision)
        self.assertEqual(
            build_feature_replay.historical_rows_until(base, decision),
            base,
        )
        self.assertEqual(len(first), len(build_feature_replay.FEATURE_COLUMN_NAMES))
        values = dict(zip(build_feature_replay.FEATURE_COLUMN_NAMES, first))
        self.assertEqual(values["history_observations_14"], 14)
        self.assertEqual(values["observed_days_7"], 7)
        self.assertEqual(values["smart_5_current_missing"], 0)
        self.assertEqual(values["smart_5_w7_missing_days"], 0)
        self.assertAlmostEqual(values["smart_5_w7_delta_log1p"], math.log1p(6), places=12)
        self.assertEqual(values["smart_nonzero_signal_count"], 5)
        self.assertEqual(values["smart_187_signal"], 1)
        self.assertEqual(len(values["tie_break_sha256"]), 64)

    def test_historical_prefix_rejects_future_or_unsorted_rows(self):
        decision = "2023-01-14"
        future = _rows() + [
            {
                "date": "2023-01-15",
                "serial_number": "disk-1",
                "model": "ST4000DM000",
                "failure": 0,
                **{f"smart_{field}_raw": 999 for field in build_feature_replay.SMART_FIELDS},
            }
        ]
        with self.assertRaises(build_feature_replay.BuildStopped):
            build_feature_replay.historical_rows_until(future, decision)
        unsorted = _rows()
        unsorted[1], unsorted[2] = unsorted[2], unsorted[1]
        with self.assertRaises(build_feature_replay.BuildStopped):
            build_feature_replay.historical_rows_until(unsorted, decision)

    def test_feature_input_rejects_middle_future_extra_columns_and_counter_decrease(self):
        decision = "2023-01-14"
        rows = _rows()
        future = dict(rows[-1], date="2023-01-15")
        with self.assertRaises(build_feature_replay.BuildStopped):
            build_feature_replay.feature_row_for_serial(rows[:3] + [future] + rows[3:], decision)
        with self.assertRaises(build_feature_replay.BuildStopped):
            build_feature_replay.feature_row_for_serial(
                [dict(row, label=0) for row in rows], decision
            )
        for field in (5, 9, 187):
            decreased = [dict(row) for row in rows]
            decreased[-1][f"smart_{field}_raw"] = 0
            with self.assertRaises(build_feature_replay.BuildStopped):
                build_feature_replay.feature_row_for_serial(decreased, decision)

    def test_missing_natural_day_is_counted_and_does_not_borrow_old_value(self):
        decision = "2023-01-14"
        rows = _rows(missing_day=dt.date(2023, 1, 5))
        result = build_feature_replay.feature_row_for_serial(rows, decision)
        self.assertIsNotNone(result)
        values = dict(zip(build_feature_replay.FEATURE_COLUMN_NAMES, result))
        self.assertEqual(values["history_observations_14"], 13)
        self.assertEqual(values["observed_days_7"], 7)
        self.assertEqual(values["smart_5_w14_missing_days"], 1)
        self.assertEqual(values["smart_5_w14_delta_span_days"], 13)

    def test_history_and_failure_gates(self):
        decision = "2023-01-14"
        short = _rows()
        short = [row for row in short if row["date"] not in {"2023-01-02", "2023-01-03", "2023-01-04"}]
        self.assertIsNone(build_feature_replay.feature_row_for_serial(short, decision))
        self.assertIsNone(
            build_feature_replay.feature_row_for_serial(_rows(failure_day=dt.date(2023, 1, 14)), decision)
        )
        self.assertIsNone(
            build_feature_replay.feature_row_for_serial(_rows(failure_day=dt.date(2023, 1, 13)), decision)
        )

    def test_rankings_are_deterministic_and_signal_scoped(self):
        rows = [
            {"serial_number": "b", "tie_break_sha256": "2", "smart_nonzero_signal_count": 1, "smart_187_signal": 0},
            {"serial_number": "a", "tie_break_sha256": "1", "smart_nonzero_signal_count": 0, "smart_187_signal": 1},
            {"serial_number": "c", "tie_break_sha256": "3", "smart_nonzero_signal_count": 2, "smart_187_signal": 0},
        ]
        self.assertEqual(
            [row["serial_number"] for row in build_feature_replay._rank_rows(rows, "random")],
            ["a", "b", "c"],
        )
        self.assertEqual(
            [row["serial_number"] for row in build_feature_replay._rank_rows(rows, "smart_nonzero")],
            ["c", "b"],
        )
        self.assertEqual(
            [row["serial_number"] for row in build_feature_replay._rank_rows(rows, "smart187")],
            ["a"],
        )

    def test_feature_dictionary_covers_schema(self):
        dictionary = build_feature_replay.feature_dictionary()
        self.assertEqual(
            [item["name"] for item in dictionary],
            build_feature_replay.FEATURE_COLUMN_NAMES,
        )
        self.assertEqual(len(build_feature_replay.feature_dictionary_hash()), 64)


if __name__ == "__main__":
    unittest.main()
