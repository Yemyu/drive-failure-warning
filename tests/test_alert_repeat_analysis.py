import unittest

from investigation.review_tree_and_alert_repeats import summarize_alerts


class AlertRepeatAnalysisTests(unittest.TestCase):
    def test_later_reminder_can_be_first_event_capture(self):
        # A's first reminder is outside H7; its later reminder captures the event.
        rows = [
            dict(serial_number='A', label=0, event_hit=0, event_key=None),
            dict(serial_number='B', label=None, event_hit=0, event_key=None),
            dict(serial_number='A', label=1, event_hit=1, event_key='A-failure'),
            # A duplicate hit must not become a second captured event.
            dict(serial_number='A', label=1, event_hit=1, event_key='A-failure'),
            # Known positive outside the primary event dates is not a main hit.
            dict(serial_number='C', label=1, event_hit=0, event_key='outside'),
        ]
        result = summarize_alerts(rows, {'A-failure': {'hit': 1}, 'D-failure': {'hit': 0}})
        self.assertEqual(result['alerts'], 5)
        self.assertEqual(result['devices'], 3)
        self.assertEqual(result['captured_events'], 1)
        self.assertEqual(result['by_occurrence']['1']['unknown'], 1)
        self.assertEqual(result['by_occurrence']['2']['first_captured_events'], 1)
        self.assertEqual(result['by_occurrence']['3'].get('first_captured_events', 0), 0)

    def test_different_event_identity_is_rejected_even_when_counts_match(self):
        rows = [dict(serial_number='A', label=1, event_hit=1, event_key='wrong')]
        with self.assertRaises(AssertionError):
            summarize_alerts(rows, {'expected': {'hit': 1}})


if __name__ == '__main__':
    unittest.main()
