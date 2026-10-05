import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import outer_supervisor as supervisor
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceViolation

ROOT=Path(__file__).resolve().parents[1]


class RSupervisedVerifierTests(unittest.TestCase):
    def test_normal_verification_keeps_one_budget_clock(self):
        original=ResourceGuard.prepare_next_group
        transitions=[]
        def transition(guard):
            before=(guard.started,dict(guard.peak_snapshot),guard.sample_count)
            original(guard)
            transitions.append(before)
            self.assertEqual(guard.started,before[0])
            self.assertEqual(guard.peak_snapshot,before[1])
            self.assertEqual(guard.sample_count,before[2])
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(ResourceGuard,'prepare_next_group',transition):
                result=supervisor.supervise_zip_fixture(output)
            self.assertEqual(len(transitions),1)
            self.assertEqual(result['independent_post_worker_verification'],'pass')
            self.assertEqual(result['worker_stage']['cleanup']['state'],'complete')
            self.assertEqual(result['cleanup']['state'],'complete')
            verification=json.loads((output/'verification.json').read_text())
            self.assertEqual(verification['modules']['pipeline']['violations'],[])
            self.assertEqual(result['publication'],'not_authorized')

    def test_changed_input_after_request_refused_by_verifier(self):
        original=supervisor.write_json_exclusive
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def write(path,value):
                result=original(path,value)
                if path.name=='verification_request.json':
                    (output/'work/extra.txt').write_text('unexpected')
                return result
            with patch.object(supervisor,'write_json_exclusive',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'verifier exited'):
                    supervisor.supervise_zip_fixture(output)
            self.assertIn('inventory changed',(output/'verifier.log').read_text())
            self.assertFalse((output/'verification.json').exists())
            self.assertFalse((output/'supervision.json').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')

    def test_cancel_before_verifier_permission(self):
        original=supervisor._load_small
        def load(path,label):
            result=original(path,label)
            if label=='verifier ready': os.kill(os.getpid(),signal.SIGTERM)
            return result
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'_load_small',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    supervisor.supervise_zip_fixture(output)
            self.assertFalse((output/'verification_go.json').exists())
            self.assertFalse((output/'supervision.json').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')

    def test_cannot_replace_unclean_group(self):
        guard=ResourceGuard(ROOT,ROOT/'.tmp')
        with self.assertRaisesRegex(ResourceViolation,'not confirmed cleaned'): guard.prepare_next_group()


if __name__=='__main__': unittest.main()
