"""Lead recall includes missed opportunities in its denominator."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.r_validation.engine import lead_recall, ProtocolError
from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
from tests.test_r_zero import zero_chain
from tools import run_small_replay as replay

ROOT = Path(__file__).resolve().parents[1]


class LeadRecallTests(unittest.TestCase):
    def test_denominator_includes_misses_and_thresholds_are_inclusive(self):
        self.assertEqual(lead_recall([1, 2, 3, 7], 5), {
            "opportunity_events": 5, "ge2_hits": 3, "ge3_hits": 2,
            "ge2_recall": 0.6, "ge3_recall": 0.4})

    def test_no_hits_differs_from_no_opportunities(self):
        self.assertEqual(lead_recall([], 5)["ge2_recall"], 0)
        self.assertIsNone(lead_recall([], 0)["ge2_recall"])

    def test_invalid_denominator_or_leads_refused(self):
        for leads, denominator in (([1], 0), ([0], 1), ([8], 1), ([True], 1), ([], -1)):
            with self.subTest(leads=leads, denominator=denominator), self.assertRaises(ProtocolError):
                lead_recall(leads, denominator)

    def test_actual_chain_rejects_each_corrupted_field(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            output = Path(directory)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            evaluation = json.loads((chain / "evaluation.json").read_text())
            model = json.loads((chain / "current_lr.json").read_text())
            for method in ("current_lr", "smart_nonzero"):
                for field in ("opportunity_events", "ge2_hits", "ge3_hits", "ge2_recall", "ge3_recall"):
                    changed = copy.deepcopy(evaluation)
                    changed["protocol_metrics"]["methods"][method]["lead_recall"][field] = 999
                    with self.subTest(method=method, field=field), self.assertRaisesRegex(IndependentAuditError, "lead recall"):
                        audit_selection(chain / "selection.sqlite", chain / "panel.sqlite", changed, None, model=model)

    def test_zero_chain_preserves_null_recalls(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            output = Path(directory)
            result = zero_chain(output)
            self.assertEqual(result["audit"]["status"], "pass")
            evaluation = json.loads((output / "attempt/cli_chain/evaluation.json").read_text())
            for method in evaluation["protocol_metrics"]["methods"].values():
                self.assertEqual(method["lead_recall"], {"opportunity_events": 0, "ge2_hits": 0,
                                                       "ge3_hits": 0, "ge2_recall": None, "ge3_recall": None})


if __name__ == "__main__":
    unittest.main()
