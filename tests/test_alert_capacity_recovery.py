import datetime as dt
import unittest

from investigation.evaluate_alert_capacity_recovery import event_for_alert, efficiency, summarize_devices, validate_label


class CapacityRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.events = {"A": {"first_failure_date": "2023-07-12", "opportunity": 1}}

    def test_event_window_boundaries(self):
        self.assertEqual(event_for_alert("A", "2023-07-11", "2023-07-12", self.events), ("A", 1, 1))
        self.assertEqual(event_for_alert("A", "2023-07-04", "2023-07-12", self.events), (None, 0, None))
        self.assertEqual(event_for_alert("A", "2023-07-19", "2023-07-12", self.events), (None, 0, None))
        self.assertEqual(event_for_alert("A", "2023-07-04", "2023-07-12", {"A": {"first_failure_date": "2023-07-12", "opportunity": 0}}), (None, 0, None))

    def test_label_status_and_type_are_checked(self):
        self.assertEqual(validate_label({"status": "positive_observed", "label": 1}), ("positive_observed", 1))
        self.assertEqual(validate_label({"status": "gap_or_exit_censored", "label": None}), ("gap_or_exit_censored", None))
        with self.assertRaises(RuntimeError):
            validate_label({"status": "positive_observed", "label": 1.0})
        with self.assertRaises(RuntimeError):
            validate_label({"status": "negative_observed", "label": 1})

    def test_repeat_devices_count_is_based_on_devices(self):
        repeated, maximum, minimum_gap = summarize_devices({"a": [dt.date(2023, 7, 1)], "b": [dt.date(2023, 7, 1), dt.date(2023, 7, 9)], "c": [dt.date(2023, 7, 1), dt.date(2023, 7, 4), dt.date(2023, 7, 10)]})
        self.assertEqual((repeated, maximum, minimum_gap), (2, 3, 3))

    def test_efficiency_is_blank_when_no_positive_net_tradeoff(self):
        self.assertEqual(efficiency(0, 0)[0], None)
        self.assertEqual(efficiency(100, -1)[0], None)
        self.assertEqual(efficiency(200, 4), (50.0, "positive net events and positive alert increment"))


if __name__ == "__main__":
    unittest.main()
