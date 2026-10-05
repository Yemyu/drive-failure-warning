"""Locked gain thresholds and independent rejection of unsupported claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.r_validation.engine import gain_conditions, ProtocolError
from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError

ROOT = Path(__file__).resolve().parents[1]


class GainGateTests(unittest.TestCase):
    def gate(self, **changes):
        values = dict(opportunities=100, current_hits=55, smart_hits=50,
                      current_alerts=100, current_unknown=12,
                      smart_alerts=100, smart_unknown=10,
                      valid_replicates=1900, interval=[0.01, 10.0], quality_status="pass")
        values.update(changes)
        return gain_conditions(**values)

    def test_exact_inclusive_boundaries_do_not_self_accept(self):
        gate = self.gate()
        self.assertTrue(gate["numerical_conditions_met"])
        self.assertFalse(gate["support_gain"])
        self.assertEqual(gate["complete_acceptance"], "pending")

    def test_each_failed_condition_blocks_numerical_evidence(self):
        cases = [(dict(opportunities=99), "minimum_events"),
                 (dict(current_hits=54), "delta_at_least_5pp"),
                 (dict(valid_replicates=1899), "minimum_valid_replicates"),
                 (dict(interval=[0, 10]), "interval_lower_above_zero"),
                 (dict(current_unknown=13), "unknown_increase_at_most_2pp"),
                 (dict(quality_status="failed"), "quality_passed")]
        for change, key in cases:
            with self.subTest(key=key):
                gate = self.gate(**change)
                self.assertFalse(gate["numerical_conditions_met"])
                self.assertIn(key, gate["unmet_conditions"])

    def test_missing_ratios_and_interval_are_unavailable_not_zero(self):
        for changes in (dict(current_alerts=0, current_unknown=0), dict(smart_alerts=0, smart_unknown=0)):
            gate = self.gate(**changes, interval=None)
            self.assertEqual(gate["unavailable_conditions"],
                             ["interval_lower_above_zero", "unknown_increase_at_most_2pp"])
            self.assertFalse(gate["numerical_conditions_met"])

    def test_zero_events_and_no_hits_are_distinct(self):
        zero = self.gate(opportunities=0, current_hits=0, smart_hits=0, interval=None)
        self.assertIsNone(zero["conditions"]["delta_at_least_5pp"])
        no_hits = self.gate(current_hits=0, smart_hits=0, interval=[0, 0])
        self.assertIs(no_hits["conditions"]["delta_at_least_5pp"], False)

    def test_invalid_counts_or_nonfinite_intervals_refused(self):
        for changes in (dict(current_hits=101), dict(current_unknown=101),
                        dict(valid_replicates=2001), dict(interval=[float("nan"), 10])):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                self.gate(**changes)

    def test_real_chain_and_independent_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            output = Path(directory)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            evaluation = json.loads((chain / "evaluation.json").read_text())
            model = json.loads((chain / "current_lr.json").read_text())
            self.assertFalse(evaluation["protocol_metrics"]["interpretation_gate"]["numerical_conditions_met"])
            for key, value in (("support_gain", True), ("complete_acceptance", "pass"),
                               ("numerical_conditions_met", True), ("unavailable_conditions", [])):
                with self.subTest(key=key):
                    changed = copy.deepcopy(evaluation)
                    changed["protocol_metrics"]["interpretation_gate"][key] = value
                    with self.assertRaisesRegex(IndependentAuditError, "interpretation gate"):
                        audit_selection(chain / "selection.sqlite", chain / "panel.sqlite", changed,
                                        None, model=model)


if __name__ == "__main__":
    unittest.main()
