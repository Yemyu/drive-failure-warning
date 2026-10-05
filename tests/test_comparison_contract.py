import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from investigation import compare_reproducible_run as comparator
from pipeline.reproducible.artifacts import (
    ArtifactError,
    _canonical,
    artifact_facts,
    verify_manifest,
    write_json_exclusive,
)
from pipeline.reproducible.comparison import ComparisonError, numeric_summary, scalar_equal


ROOT = Path(__file__).resolve().parents[1]


def _model_payload(*, intercept: float = 0.0) -> dict[str, object]:
    return {
        "model": "current_lr",
        "fit_mode": "fitted",
        "columns": ["x"],
        "coef": [1.0],
        "intercept": intercept,
        "preprocessing": {
            "imputation_mean": [0.0],
            "standardization_mean": [0.0],
            "standardization_scale": [1.0],
        },
        "training_rows": 2,
        "positive_rows": 1,
        "negative_rows": 1,
        "params": {
            "penalty": "l2",
            "solver": "lbfgs",
            "C": 1.0,
            "tol": 1e-6,
            "max_iter": 1000,
            "fit_intercept": True,
            "class_weight": None,
            "random_state": 20260913,
        },
    }


class ScalarComparisonTests(unittest.TestCase):
    def test_integers_are_exact_and_floats_use_tolerance(self):
        self.assertFalse(scalar_equal(10**12, 10**12 + 1))
        self.assertTrue(scalar_equal(10**12, 10**12))
        self.assertTrue(scalar_equal(1.0, 1.0 + 1e-13))
        self.assertFalse(scalar_equal(1, 1.0))
        summary = numeric_summary([(10**12, 10**12 + 1)], "capacity")
        self.assertEqual(summary["outside_tolerance"], 1)
        self.assertEqual(summary["max_abs"], 1)

    def test_empty_numeric_comparison_is_not_silently_passed(self):
        with self.assertRaisesRegex(ComparisonError, "no comparable"):
            numeric_summary([], "coefficients")


class ArtifactParsingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_manifest_rejects_duplicate_or_nonfinite_json(self):
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"status":"complete","status":"failed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "duplicate JSON key"):
            verify_manifest(duplicate)
        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text('{"status":"complete","value":NaN}\n', encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "JSON constant"):
            verify_manifest(nonfinite)

    def test_manifest_rejects_float_byte_count(self):
        result = self.root / "result.txt"
        result.write_text("stable\n", encoding="utf-8")
        facts = artifact_facts(result)
        body = {"status": "complete", "artifacts": {"result": facts}}
        body["artifacts"]["result"]["bytes"] = float(facts["bytes"])
        body["manifest_hash"] = _canonical(body)
        manifest_path = self.root / "bad_manifest.json"
        manifest_path.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "invalid artifact facts"):
            verify_manifest(manifest_path)

    def test_json_writer_rejects_nonfinite_payload(self):
        with self.assertRaises(ValueError):
            write_json_exclusive(self.root / "bad.json", {"value": float("nan")})
        self.assertFalse((self.root / "bad.json").exists())


class FrozenComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _old_model(self) -> Path:
        old = self.root / "old.sqlite"
        with sqlite3.connect(old) as connection:
            connection.executescript(
                "CREATE TABLE model_runs(model TEXT PRIMARY KEY,status TEXT,feature_count INTEGER,"
                "training_rows INTEGER,positive_training_rows INTEGER,negative_training_rows INTEGER,"
                "intercept REAL);"
                "CREATE TABLE model_coefficients(model TEXT,feature_name TEXT,coefficient REAL,"
                "PRIMARY KEY(model,feature_name));"
                "CREATE TABLE preprocessing_stats(model TEXT,feature_name TEXT,imputation_mean REAL,"
                "standardization_mean REAL,standardization_scale REAL,PRIMARY KEY(model,feature_name));"
            )
            connection.execute("INSERT INTO model_runs VALUES ('current_lr','fitted',1,2,1,1,1.0)")
            connection.execute("INSERT INTO model_coefficients VALUES ('current_lr','x',1.0)")
            connection.execute("INSERT INTO preprocessing_stats VALUES ('current_lr','x',0.0,0.0,1.0)")
        return old

    def _old_evaluation(self) -> Path:
        old = self.root / "old_evaluation.sqlite"
        with sqlite3.connect(old) as connection:
            connection.execute(
                "CREATE TABLE model_metrics(model TEXT PRIMARY KEY,eligible_device_days INTEGER,alerts INTEGER,"
                "known_hit_alerts INTEGER,known_no_hit_alerts INTEGER,unknown_alerts INTEGER,event_total INTEGER,"
                "event_opportunity_total INTEGER,event_hits INTEGER,event_recall_at_opportunity REAL,"
                "unknown_alert_ratio REAL,precision_lower_bound REAL,precision_upper_bound REAL,"
                "known_outcome_precision REAL,confirmed_no_hit_per_1000_device_days REAL,average_precision_known REAL)"
            )
            values = (2, 1, 1, 0, 0, 1, 1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0)
            connection.executemany(
                "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(method, *values) for method in ("current_lr", "smart_nonzero")],
            )
        return old

    def _v2_metrics_fixture(self) -> tuple[dict[str, object], dict[str, object]]:
        event = {
            "event_key": "disk-a",
            "first_failure_date": "2023-02-13",
            "opportunity": 1,
            "hit": 1,
            "opportunity_dates": ["2023-02-12"],
            "earliest_alert_date": "2023-02-12",
            "earliest_lead_days": 1,
        }
        alert = {
            "method": "current_lr",
            "decision_date": "2023-02-12",
            "serial_number": "disk-a",
            "first_failure_date": "2023-02-13",
            "label": 1,
        }
        metrics = {
            "method": "current_lr",
            "eligible_rows": 2,
            "alerts": 1,
            "known_hit_alerts": 1,
            "known_no_hit_alerts": 0,
            "unknown_alerts": 0,
            "event_count": 1,
            "opportunity_events": 1,
            "event_hits": 1,
            "event_recall": 1.0,
            "known_precision": 1.0,
            "precision_lower_bound": 1.0,
            "precision_upper_bound": 1.0,
            "unknown_ratio": 0.0,
            "known_misses_per_1000_eligible_days": 0.0,
            "lead_days_median": 1.0,
            "lead_days_q25": 1.0,
            "lead_days_q75": 1.0,
            "events_lead_ge_2": 0,
            "events_lead_ge_3": 0,
            "opportunity_recall_lead_ge_2": 0.0,
            "opportunity_recall_lead_ge_3": 0.0,
            "outside_main_event_alerts": [],
            "outside_main_event_alert_count": 0,
            "lead_days_captured_events": 1,
            "lead_days_values": [1],
            "known_pool_rows": 2,
            "positive_pool_rows": 1,
            "full_pool_average_precision": 1.0,
        }
        methods = {"current_lr": metrics, "smart_nonzero": {**metrics, "method": "smart_nonzero"}}
        payload = {
            "status": "complete",
            "scope": "evaluation_metrics_v2",
            "event_window": {"start": "2023-02-05", "end": "2023-02-13"},
            "methods": methods,
            "event_opportunities": {"current_lr": {"disk-a": event}, "smart_nonzero": {"disk-a": event}},
        }
        summaries = {
            method: {
                "events": {"disk-a": event},
                "daily": [{"date": "2023-02-12", "eligible": 2, "budget": 1, "cooldown_excluded": 0, "alerts": 1}],
                "metrics": methods[method],
            }
            for method in ("current_lr", "smart_nonzero")
        }
        return payload, summaries

    def test_intercept_difference_is_reported(self):
        new = self.root / "new_model.json"
        write_json_exclusive(new, _model_payload(intercept=2.0))
        differences, checks = comparator._compare_model(new, self._old_model())
        self.assertTrue(any(item["kind"] == "intercept_numeric_difference" for item in differences))
        self.assertEqual(checks["intercept_numeric"]["outside_tolerance"], 1)

    def test_missing_model_array_or_intercept_is_rejected(self):
        missing_coef = _model_payload()
        missing_coef.pop("coef")
        with self.assertRaisesRegex(ComparisonError, "missing required fields"):
            comparator._validate_new_model(missing_coef)
        missing_intercept = _model_payload()
        missing_intercept.pop("intercept")
        with self.assertRaisesRegex(ComparisonError, "missing required fields"):
            comparator._validate_new_model(missing_intercept)

    def test_metrics_must_contain_both_methods(self):
        metrics = self.root / "metrics.json"
        write_json_exclusive(metrics, {"methods": {"current_lr": {}}})
        manifest = {"verified_artifacts": {"metrics": {"path": str(metrics)}}}
        with self.assertRaisesRegex(ComparisonError, "current_lr and smart_nonzero"):
            comparator._compare_metrics(manifest, self._old_model())

    def test_v2_metrics_are_recomputed_from_events_and_alerts(self):
        payload, summaries = self._v2_metrics_fixture()
        metrics_path = self.root / "metrics_v2.json"
        write_json_exclusive(metrics_path, payload)
        manifest = {"summaries": summaries, "verified_artifacts": {"metrics": {"path": str(metrics_path)}}}
        for method in ("current_lr", "smart_nonzero"):
            alert = dict(payload["methods"][method])
            write_json_exclusive(
                self.root / f"{method}_alerts.json",
                {"status": "complete", "method": method, "alerts": [{
                    "method": method,
                    "decision_date": "2023-02-12",
                    "serial_number": "disk-a",
                    "first_failure_date": "2023-02-13",
                    "label": 1,
                }]},
            )
            manifest["verified_artifacts"][f"{method}_alerts"] = {"path": str(self.root / f"{method}_alerts.json")}
        self.assertEqual(comparator._compare_metrics(manifest, self._old_evaluation()), [])

    def test_v2_metrics_missing_lead_field_or_unknown_scope_is_rejected(self):
        payload, summaries = self._v2_metrics_fixture()
        old_evaluation = self._old_evaluation()
        payload["methods"]["current_lr"].pop("lead_days_q25")
        metrics_path = self.root / "missing_v2_field.json"
        write_json_exclusive(metrics_path, payload)
        manifest = {"summaries": summaries, "verified_artifacts": {"metrics": {"path": str(metrics_path)}}}
        with self.assertRaisesRegex(ComparisonError, "missing required fields"):
            comparator._compare_metrics(manifest, old_evaluation)
        payload, _ = self._v2_metrics_fixture()
        payload["scope"] = "evaluation_metrics_v9"
        unknown_path = self.root / "unknown_v2_scope.json"
        write_json_exclusive(unknown_path, payload)
        manifest["verified_artifacts"]["metrics"] = {"path": str(unknown_path)}
        with self.assertRaisesRegex(ComparisonError, "unknown evaluation metrics scope"):
            comparator._compare_metrics(manifest, old_evaluation)


if __name__ == "__main__":
    unittest.main()
