"""Publication branches with explicit mocked authorization, no real release."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from pipeline.r_validation.engine import gain_conditions, interval_publication
from pipeline.r_validation.synthetic_publication import publish_bound
from pipeline.reproducible.artifacts import verify_manifest

ROOT=Path(__file__).resolve().parents[1]


class Guard:
    violation=None
    def check_or_raise(self):pass
    def stop(self):pass


class CompletionGateTests(unittest.TestCase):
    def publish(self, events=100, valid=1900, quality='pass', interval=None, alerts=100, delta=5, synthetic=False):
        interval=[1.,9.] if interval is None and events>=100 and valid>=1900 else interval
        bootstrap={'opportunity_events':events,'valid_replicates':valid,'interval':interval}
        gate=gain_conditions(opportunities=events,current_hits=min(events,50+delta),smart_hits=min(events,50),
            current_alerts=alerts,current_unknown=0,smart_alerts=alerts,smart_unknown=0,
            valid_replicates=valid,interval=interval,quality_status=quality)
        evaluation={'protocol_metrics':{'bootstrap':bootstrap,'interval_publication':interval_publication(bootstrap,quality),
                                       'interpretation_gate':gate}}
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            output=Path(tmp);artifact=output/'audit.txt';artifact.write_text('unit test fixture')
            with patch('pipeline.r_validation.bound_zip.validate_bound_zip',return_value={}):
                result=publish_bound(output,artifacts={'audit.txt':artifact},guard=Guard(),cancel=threading.Event(),
                    validate=lambda:None,contract={'profile':'synthetic_bound_zip' if synthetic else 'released_zip'},evaluation=evaluation)
            name='bound_synthetic_complete' if synthetic else 'r_complete'
            verify_manifest(output/(name+'.json'),expected_status=result['status'])
            return result

    def test_zero_and_99_events_withhold_interval_and_gain(self):
        for events in (0,99):
            with self.subTest(events=events):
                result=self.publish(events=events)
                self.assertIsNone(result['interval_publication']['interval_pp'])
                self.assertIn('insufficient_events',result['interval_publication']['blocking_reasons'])
                self.assertFalse(result['gain_evidence']['support_gain'])

    def test_exact_100_events_and_1900_valid_can_publish(self):
        result=self.publish()
        self.assertEqual(result['interval_publication']['status'],'published')
        self.assertTrue(result['gain_evidence']['support_gain'])
        self.assertEqual(result['gain_evidence']['final_interpretation'],'pending_review')
        self.assertFalse(result['model_adoption'])

    def test_1899_valid_withholds(self):
        result=self.publish(valid=1899)
        self.assertIsNone(result['interval_publication']['interval_pp'])
        self.assertFalse(result['gain_evidence']['support_gain'])

    def test_quality_failed_or_not_evaluable_withholds(self):
        for quality in ('failed','not_evaluable'):
            result=self.publish(quality=quality)
            self.assertIsNone(result['interval_publication']['interval_pp'])
            self.assertFalse(result['gain_evidence']['support_gain'])

    def test_undefined_unknown_ratio_prevents_gain_not_interval(self):
        result=self.publish(alerts=0)
        self.assertEqual(result['interval_publication']['status'],'published')
        self.assertFalse(result['gain_evidence']['support_gain'])

    def test_zero_lower_bound_or_small_delta_do_not_support_gain(self):
        for parameters in ({'interval':[0.,10.]},{'interval':[-1.,10.]},{'delta':4}):
            result=self.publish(**parameters)
            self.assertEqual(result['interval_publication']['status'],'published')
            self.assertFalse(result['gain_evidence']['support_gain'])

    def test_synthetic_cannot_publish_confirmatory_interval_or_gain(self):
        result=self.publish(synthetic=True)
        self.assertIsNone(result['interval_publication']['interval_pp'])
        self.assertFalse(result['gain_evidence']['support_gain'])
