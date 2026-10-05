import datetime as dt
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.reproducible.current import build_current_features, build_labels
from pipeline.reproducible.evaluation_summary import summarize_lead_metrics
from pipeline.reproducible.panel_reference import PanelReferenceError, verify_training_reference
from tests.test_reproducible_pipeline import make_panel


ROOT = Path(__file__).resolve().parents[1]


class PanelReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.panel = self.root / "panel.sqlite"
        self.labels = self.root / "labels.sqlite"
        self.features = self.root / "features.sqlite"
        make_panel(self.panel)
        build_labels(
            [self.panel], self.labels, start="2023-01-15", end="2023-01-28",
            dataset_end="2023-02-04", run_id="reference-run",
        )
        build_current_features(
            self.panel, self.features, start="2023-01-15", end="2023-01-28",
            dataset_end="2023-02-04",
        )

    def _verify(self):
        return verify_training_reference(
            self.panel, self.labels, self.features, run_id="reference-run",
            start="2023-01-15", end="2023-01-28", cutoff="2023-02-04",
        )

    def test_reference_replays_fixture_labels_features_and_calendar(self):
        result = self._verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["version"], "panel-reference-v1")
        self.assertEqual(result["labels"], {"expected_rows": 42, "actual_rows": 42})
        self.assertEqual(result["features"]["expected_rows"], 42)
        self.assertEqual(result["features"]["calendar_days"], 14)

    def test_reference_rejects_tampered_label_flow(self):
        with sqlite3.connect(self.labels) as connection:
            connection.execute(
                "UPDATE label_flow SET label=1,status='positive_observed' "
                "WHERE run_id='reference-run' AND serial_number='disk-a' AND decision_date='2023-01-15'"
            )
        with self.assertRaisesRegex(PanelReferenceError, "label_flow"):
            self._verify()

    def test_reference_rejects_tampered_feature_row(self):
        with sqlite3.connect(self.features) as connection:
            connection.execute(
                "UPDATE feature_rows SET smart_5_current_log1p=123.0 "
                "WHERE serial_number='disk-b' AND decision_date='2023-01-15'"
            )
        with self.assertRaisesRegex(PanelReferenceError, "feature_rows"):
            self._verify()

    def test_reference_rejects_source_change_after_derivation(self):
        with sqlite3.connect(self.panel) as connection:
            connection.execute(
                "UPDATE daily SET smart_5_raw=smart_5_raw+1 "
                "WHERE serial_number='disk-b' AND date='2023-01-15'"
            )
        with self.assertRaisesRegex(PanelReferenceError, "different source panel|changed"):
            self._verify()


class EvaluationSummaryTests(unittest.TestCase):
    def test_lead_summary_uses_all_opportunities_and_keeps_outside_details(self):
        events = {
            "a": {"opportunity": 1, "hit": 1, "earliest_lead_days": 1, "earliest_alert_date": "2023-02-12", "first_failure_date": "2023-02-13", "opportunity_dates": ["2023-02-12"]},
            "b": {"opportunity": 1, "hit": 1, "earliest_lead_days": 3, "earliest_alert_date": "2023-02-10", "first_failure_date": "2023-02-13", "opportunity_dates": ["2023-02-10"]},
            "c": {"opportunity": 1, "hit": 1, "earliest_lead_days": 7, "earliest_alert_date": "2023-02-06", "first_failure_date": "2023-02-13", "opportunity_dates": ["2023-02-06"]},
            "d": {"opportunity": 1, "hit": 0},
        }
        alerts = [
            {"method": "current_lr", "decision_date": "2023-02-12", "serial_number": "disk-a", "first_failure_date": "2023-02-13"},
            {"method": "current_lr", "decision_date": "2023-02-10", "serial_number": "disk-b", "first_failure_date": "2023-02-13"},
            {"method": "current_lr", "decision_date": "2023-02-06", "serial_number": "disk-c", "first_failure_date": "2023-02-13"},
            {"method": "current_lr", "decision_date": "2023-02-03", "serial_number": "disk-x", "first_failure_date": "2023-02-04"},
        ]
        result = summarize_lead_metrics(
            alerts, events, event_start="2023-02-05", event_end="2023-02-13",
        )
        self.assertEqual(result["lead_days_median"], 3.0)
        self.assertEqual(result["lead_days_q25"], 2.0)
        self.assertEqual(result["lead_days_q75"], 5.0)
        self.assertEqual(result["events_lead_ge_2"], 2)
        self.assertEqual(result["events_lead_ge_3"], 2)
        self.assertEqual(result["opportunity_recall_lead_ge_2"], 0.5)
        self.assertEqual(result["opportunity_recall_lead_ge_3"], 0.5)
        self.assertEqual(result["outside_main_event_alerts"][0]["serial_number"], "disk-x")
        self.assertEqual(result["outside_main_event_alerts"][0]["lead_days"], 1)

    def test_lead_summary_returns_null_ratio_without_opportunities(self):
        result = summarize_lead_metrics(
            [], {"a": {"opportunity": 0, "hit": 0}},
            event_start=dt.date(2023, 2, 5), event_end=dt.date(2023, 2, 13),
        )
        self.assertIsNone(result["lead_days_median"])
        self.assertIsNone(result["opportunity_recall_lead_ge_2"])
        self.assertIsNone(result["opportunity_recall_lead_ge_3"])
        self.assertEqual(result["events_lead_ge_2"], 0)
        self.assertEqual(result["outside_main_event_alerts"], [])


if __name__ == "__main__":
    unittest.main()
