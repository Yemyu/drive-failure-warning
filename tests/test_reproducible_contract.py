import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.reproducible.contract import (
    ContractError,
    LOCKED_FEATURES,
    load_json,
    validate_locked_config,
)
from pipeline.reproducible.worker import WorkerError, _selection_from_score_db, _validate_model_payload


ROOT = Path(__file__).resolve().parents[1]


class LockedContractTests(unittest.TestCase):
    def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
            with self.assertRaises(ContractError):
                load_json(duplicate, "duplicate probe")
            with self.assertRaises(ContractError):
                load_json(nonfinite, "nonfinite probe")

    def test_locked_config_has_exact_profile_and_rejects_mutation(self):
        payload = load_json(ROOT / "configs/reproducible_current_v1.json", "locked config")
        validated = validate_locked_config(payload)
        self.assertEqual(validated["features"], list(LOCKED_FEATURES))
        mutated = json.loads(json.dumps(payload))
        mutated["task"]["horizon_days"] = 14
        with self.assertRaisesRegex(ContractError, "H7"):
            validate_locked_config(mutated)

    def test_strict_model_payload_requires_locked_estimator_parameters(self):
        config = validate_locked_config(load_json(ROOT / "configs/reproducible_current_v1.json", "locked config"))
        model = {
            "model": "current_lr",
            "fit_mode": "fitted",
            "columns": list(LOCKED_FEATURES),
            "coef": [0.0] * len(LOCKED_FEATURES),
            "intercept": 0.0,
            "preprocessing": {
                "imputation_mean": [0.0] * len(LOCKED_FEATURES),
                "standardization_mean": [0.0] * len(LOCKED_FEATURES),
                "standardization_scale": [1.0] * len(LOCKED_FEATURES),
            },
            "params": dict(config["estimator"]),
        }
        _validate_model_payload(model, strict=True, config=config)
        model["params"]["solver"] = "liblinear"
        with self.assertRaisesRegex(WorkerError, "solver"):
            _validate_model_payload(model, strict=True, config=config)


class SelectionCalendarContractTests(unittest.TestCase):
    def test_selection_emits_zero_eligible_days_and_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            score_db = Path(directory) / "scores.sqlite"
            with sqlite3.connect(score_db) as connection:
                connection.execute(
                    "CREATE TABLE scores(decision_date TEXT, serial_number TEXT, score REAL, "
                    "rule_score INTEGER, tie_break_sha256 TEXT)"
                )
            calendar = [
                {"date": "2023-07-01", "eligible_rows": 0},
                {"date": "2023-07-02", "eligible_rows": 0},
            ]
            selection = _selection_from_score_db(score_db, "current_lr", calendar_rows=calendar)
            self.assertEqual([row["date"] for row in selection["daily"]], ["2023-07-01", "2023-07-02"])
            self.assertEqual([row["eligible"] for row in selection["daily"]], [0, 0])
            with self.assertRaisesRegex(WorkerError, "non-empty qualification day"):
                _selection_from_score_db(
                    score_db, "current_lr", calendar_rows=[{"date": "2023-07-01", "eligible_rows": 1}]
                )
