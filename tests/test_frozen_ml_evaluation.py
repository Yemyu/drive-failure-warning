"""Small, independent checks for the frozen E1.0 diagnostics."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_frozen_ml as ml  # noqa: E402





def test_fixture_contract_passes() -> None:
    assert ml.fixture()["status"] == "pass"


def test_tied_ap_uses_whole_threshold_group() -> None:
    result = ml.average_precision_rows([(2.0, 1, "a"), (2.0, 0, "b"), (0.0, 0, "c")])
    assert result["value"] == 0.5
    assert result["known_rows"] == 3


def test_unknown_is_not_a_negative_for_ap() -> None:
    result = ml.average_precision_rows([(2.0, 1, "a"), (1.0, None, "b"), (0.0, 0, "c")])
    assert result["known_rows"] == 2
    assert result["unknown_rows"] == 1
    assert result["value"] == 1.0


def test_zero_positive_ap_is_not_zero() -> None:
    result = ml.average_precision_rows([(2.0, 0, "a"), (1.0, None, "b")])
    assert result["value"] is None
    assert result["positive_rows"] == 0


def test_extreme_logits_have_finite_log_loss() -> None:
    assert math.isfinite(ml.log_loss_from_logit(1000.0, 1))
    assert math.isfinite(ml.log_loss_from_logit(-1000.0, 0))
    assert ml.sigmoid(1000.0) == 1.0
    assert ml.sigmoid(-1000.0) == 0.0


def test_fixed_bins_are_left_closed_and_last_right_closed() -> None:
    result = ml.calibration_rows([
        (math.log(0.0002 / (1 - 0.0002)), 0, "a"),
        (math.log(1 / 3), 1, "b"),
    ])
    nonempty = [row for row in result["bins"] if row["all_rows"]]
    assert len(nonempty) == 2
    assert nonempty[0]["lower"] == 0.0001
    assert nonempty[0]["upper"] == 0.0003


def test_brier_and_log_loss_match_independent_formula() -> None:
    rows = [(0.0, 0, "a"), (math.log(3.0), 1, "b"), (math.log(0.25), 0, "c")]
    result = ml.calibration_rows(rows)
    probabilities = [ml.sigmoid(score) for score, _, _ in rows]
    labels = [label for _, label, _ in rows]
    expected_brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    expected_logloss = sum(max(0.0, score) - score * y + math.log1p(math.exp(-abs(score)))
                           for (score, y, _), p in zip(rows, probabilities)) / len(labels)
    assert abs(result["brier"] - expected_brier) < 1e-12
    assert abs(result["log_loss"] - expected_logloss) < 1e-12


def test_burden_separates_unknown_and_repeats() -> None:
    result = ml.burden([
        {"serial_number": "a", "label": 1, "event_hit": 1, "event_key": "a|2023-01-03"},
        {"serial_number": "a", "label": None, "event_hit": 0},
        {"serial_number": "b", "label": 0, "event_hit": 0},
    ], event_start="2023-01-01", event_end="2023-01-07", eligible_device_days=100, opportunity_events=2)
    assert result["alerts"] == 3
    assert result["event_hits"] == 1
    assert result["event_recall"] == 0.5
    assert result["unknown_alerts"] == 1
    assert result["alerted_devices"] == 2
    assert result["later_alerts"] == 1


def test_burden_deduplicates_repeated_alerts_for_one_event() -> None:
    result = ml.burden([
        {"serial_number": "a", "label": 1, "event_hit": 1, "event_key": "a|2023-01-03"},
        {"serial_number": "a", "label": 1, "event_hit": 1, "event_key": "a|2023-01-03"},
    ], event_start="2023-01-01", event_end="2023-01-07", eligible_device_days=100, opportunity_events=1)
    assert result["alerts"] == 2
    assert result["event_hits"] == 1
    assert result["alerts_per_event"] == 2


def test_sparse_gate_uses_distinct_positive_devices() -> None:
    repeated = [(0.1, 1, "a")] * 20 + [(0.1, 0, "b")]
    result = ml.calibration_rows(repeated)
    item = next(row for row in result["bins"] if row["all_rows"])
    assert item["known_rows"] == 21
    assert item["positive_devices"] == 1
    assert item["sparse_positive_devices"] is True
    assert item["sparse_known_rows"] is False


def test_attempt_id_is_safe() -> None:
    assert ml.safe_attempt_id("attempt_009") == "attempt_009"
    for value in ("../escape", "a/b", "", "attempt.1"):
        try:
            ml.safe_attempt_id(value)
        except ml.AnalysisError:
            pass
        else:
            raise AssertionError(value)










def load_tests(loader, tests, pattern):
    """Expose all synthetic checks to the project's unittest discovery."""
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))
