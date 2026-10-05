import datetime as dt
import unittest

from pipeline.labeling import classify_device_rows


def rows(dates, failures=()):
    failure_set = set(failures)
    return [
        {
            "date": date,
            "serial_number": "disk-1",
            "model": "ST4000DM000",
            "capacity_bytes": 4000787030016,
            "failure": int(date in failure_set),
        }
        for date in dates
    ]


class LabelingTest(unittest.TestCase):
    def test_same_day_failure_is_not_an_eligible_decision(self):
        dates = [f"2023-01-{day:02d}" for day in range(1, 22)]
        output = classify_device_rows(
            rows(dates, failures=("2023-01-16",)),
            start="2023-01-15",
            end="2023-01-20",
            dataset_end="2023-01-31",
        )
        same_day = next(item for item in output if item["decision_date"] == "2023-01-16")
        self.assertEqual(same_day["status"], "same_day_failure")
        self.assertEqual(same_day["eligible"], 0)

    def test_future_failure_is_positive_and_post_failure_is_excluded(self):
        dates = [f"2023-01-{day:02d}" for day in range(1, 25)]
        output = classify_device_rows(
            rows(dates, failures=("2023-01-20",)),
            start="2023-01-15",
            end="2023-01-23",
            dataset_end="2023-01-31",
        )
        by_date = {item["decision_date"]: item for item in output}
        self.assertEqual(by_date["2023-01-15"]["label"], 1)
        self.assertEqual(by_date["2023-01-15"]["status"], "positive_observed")
        self.assertEqual(by_date["2023-01-21"]["status"], "post_failure")
        self.assertEqual(by_date["2023-01-21"]["eligible"], 0)

    def test_gap_is_unknown_and_never_filled_as_negative(self):
        dates = [f"2023-01-{day:02d}" for day in range(1, 25) if day != 18]
        output = classify_device_rows(
            rows(dates),
            start="2023-01-15",
            end="2023-01-16",
            dataset_end="2023-01-31",
        )
        item = next(item for item in output if item["decision_date"] == "2023-01-16")
        self.assertEqual(item["label"], None)
        self.assertEqual(item["status"], "gap_or_exit_censored")
        self.assertEqual(item["eligible"], 1)

    def test_history_shortage_is_explicit(self):
        dates = [f"2023-01-{day:02d}" for day in range(1, 12)]
        output = classify_device_rows(
            rows(dates),
            start="2023-01-01",
            end="2023-01-14",
            dataset_end="2023-01-31",
        )
        self.assertTrue(all(item["status"] == "history_insufficient" for item in output))
        self.assertTrue(all(item["eligible"] == 0 for item in output))

    def test_cutoff_excludes_future_failure(self):
        dates = [
            (dt.date(2023, 6, 16) + dt.timedelta(days=offset)).isoformat()
            for offset in range(20)
        ]
        full = classify_device_rows(
            rows(dates, failures=("2023-07-03",)),
            start="2023-06-29",
            end="2023-06-29",
            dataset_end="2023-06-30",
        )
        clipped = classify_device_rows(
            rows([date for date in dates if date <= "2023-06-30"]),
            start="2023-06-29",
            end="2023-06-29",
            dataset_end="2023-06-30",
        )
        self.assertEqual(full, clipped)
        self.assertEqual(full[0]["status"], "end_censored")
        self.assertIsNone(full[0]["label"])

    def test_invalid_configuration_is_rejected(self):
        dates = [f"2023-01-{day:02d}" for day in range(1, 22)]
        with self.assertRaises(ValueError):
            classify_device_rows(
                rows(dates),
                start="2023-01-15",
                end="2023-01-14",
                dataset_end="2023-01-31",
            )
        with self.assertRaises(ValueError):
            classify_device_rows(
                rows(dates),
                start="2023-01-15",
                end="2023-01-20",
                dataset_end="2023-01-19",
            )
        with self.assertRaises(ValueError):
            classify_device_rows(
                rows(dates),
                start="2023-01-15",
                end="2023-01-20",
                dataset_end="2023-01-31",
                min_history=0,
            )


if __name__ == "__main__":
    unittest.main()
