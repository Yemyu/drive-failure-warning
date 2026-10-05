"""AP uses full known score pool; alert burden retains every eligible day."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.r_validation.engine import known_average_precision
from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
from tests.test_r_zero import zero_chain

ROOT = Path(__file__).resolve().parents[1]


class SecondaryTests(unittest.TestCase):
    def test_ties_unknown_and_hand_computed_ap(self):
        # Known thresholds: score 2 contains one positive and one negative;
        # score 1 adds one positive. AP = (1/2)*(1/2)+(1/2)*(2/3).
        pairs = [(3.0, None), (2.0, 1), (2.0, 0), (1.0, 1)]
        result = known_average_precision(pairs)
        self.assertAlmostEqual(result['value'], 7/12)
        self.assertEqual((result['known_rows'],result['positive_rows'],result['unknown_rows']), (3,2,1))
        self.assertEqual(result, known_average_precision(list(reversed(pairs))))

    def test_empty_and_no_positive_ap_are_null(self):
        for pairs in ([], [(1, None)], [(1, 0)]):
            self.assertIsNone(known_average_precision(pairs)['value'])
        self.assertEqual(known_average_precision([(1,1),(1,1)])['value'], 1)

    def test_actual_chain_and_all_secondary_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)
            run_cli_fixture_chain(output, root=ROOT)
            chain=output/'cli_chain'
            evaluation=json.loads((chain/'evaluation.json').read_text())
            model=json.loads((chain/'current_lr.json').read_text())
            smart=evaluation['protocol_metrics']['methods']['smart_nonzero']
            self.assertEqual(smart['outside_main_event_alerts'], [
                {'decision_date':'2023-01-18','serial_number':'demo-C','first_failure_date':'2023-01-21'}])
            self.assertEqual(smart['outside_main_event_alert_count'], 1)
            self.assertEqual(evaluation['methods']['smart_nonzero']['event_opportunity_total'], 2)
            self.assertEqual(smart['average_precision_known']['known_rows'],112)
            for method in ('current_lr','smart_nonzero'):
                metrics=evaluation['protocol_metrics']['methods'][method]
                self.assertGreater(metrics['average_precision_known']['known_rows'], metrics['burden']['alerts'])
                self.assertEqual(metrics['burden']['first_alerts']+metrics['burden']['later_alerts'], metrics['burden']['alerts'])
                for group in ('average_precision_known','burden'):
                    for field in metrics[group]:
                        changed=copy.deepcopy(evaluation)
                        changed['protocol_metrics']['methods'][method][group][field]=999
                        with self.subTest(method=method,group=group,field=field), self.assertRaises(IndependentAuditError):
                            audit_selection(chain/'selection.sqlite',chain/'panel.sqlite',changed,None,model=model)
                changed=copy.deepcopy(evaluation)
                changed['protocol_metrics']['methods'][method]['outside_main_event_alert_count']=999
                with self.assertRaisesRegex(IndependentAuditError,'outside event'):
                    audit_selection(chain/'selection.sqlite',chain/'panel.sqlite',changed,None,model=model)
            changed=copy.deepcopy(evaluation)
            changed['protocol_metrics']['methods']['smart_nonzero']['outside_main_event_alerts'][0]['first_failure_date']='2023-01-22'
            with self.assertRaisesRegex(IndependentAuditError,'outside event'):
                audit_selection(chain/'selection.sqlite',chain/'panel.sqlite',changed,None,model=model)

    def test_zero_chain_null_ap_and_burden(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)
            self.assertEqual(zero_chain(output)['audit']['status'],'pass')
            evaluation=json.loads((output/'attempt/cli_chain/evaluation.json').read_text())
            for metrics in evaluation['protocol_metrics']['methods'].values():
                self.assertIsNone(metrics['average_precision_known']['value'])
                self.assertIsNone(metrics['burden']['alerts_per_1000_device_days'])
                self.assertEqual(metrics['outside_main_event_alerts'],[])


if __name__ == '__main__':
    unittest.main()
