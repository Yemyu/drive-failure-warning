import json
from pathlib import Path
import tempfile
import unittest
import os
import signal
from unittest.mock import patch
from pipeline.r_validation import outer_supervisor as supervisor

from pipeline.r_validation.outer_supervisor import supervise_zip_fixture

ROOT=Path(__file__).resolve().parents[1]


class ROuterSupervisorTests(unittest.TestCase):
    def test_cancel_during_actual_computation(self):
        original=supervisor.ResourceGuard.check_or_raise
        sent=[]
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def check(guard):
                result=original(guard)
                if (output/'work').exists() and not sent:
                    sent.append(True)
                    os.kill(os.getpid(),signal.SIGTERM)
                return result
            with patch.object(supervisor.ResourceGuard,'check_or_raise',check):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    supervise_zip_fixture(output)
            self.assertEqual(sent,[True])
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'complete',json.dumps(failure['cleanup']))
            self.assertEqual(failure['cleanup']['members'],[])
            self.assertFalse((output/'supervision.json').exists())

    def test_wrong_ready_identity_refused_before_permission(self):
        original=supervisor._load_small
        def load(path,label):
            result=original(path,label)
            if label=='worker ready': result['pgid']=-1
            return result
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'_load_small',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'ready identity mismatch'):
                    supervise_zip_fixture(output)
            self.assertFalse((output/'go.json').exists())
            self.assertFalse((output/'work').exists())

    def test_worker_rejects_wrong_permission_before_work(self):
        original=supervisor.write_json_exclusive
        def write(path,value):
            if path.name=='go.json': value={**value,'request_sha256':'0'*64}
            return original(path,value)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'write_json_exclusive',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'worker exited'):
                    supervise_zip_fixture(output)
            self.assertIn('worker permission mismatch',(output/'worker.log').read_text())
            self.assertFalse((output/'work').exists())
            self.assertFalse((output/'supervision.json').exists())

    def test_cancel_before_permission_prevents_work(self):
        original=supervisor._load_small
        def load(path,label):
            result=original(path,label)
            if label=='worker ready': os.kill(os.getpid(),signal.SIGTERM)
            return result
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'_load_small',side_effect=load):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    supervise_zip_fixture(output)
            self.assertFalse((output/'go.json').exists())
            self.assertFalse((output/'work').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')

    def test_cancel_after_permission_cleans_worker_and_restores_handler(self):
        original=supervisor.write_json_exclusive
        handler=signal.getsignal(signal.SIGTERM)
        def write(path,value):
            result=original(path,value)
            if path.name=='go.json': os.kill(os.getpid(),signal.SIGTERM)
            return result
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'write_json_exclusive',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    supervise_zip_fixture(output)
            self.assertTrue((output/'go.json').exists())
            self.assertFalse((output/'supervision.json').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')
            self.assertEqual(signal.getsignal(signal.SIGTERM),handler)

    def test_normal_group_observed_and_cleaned_without_final_publish(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            result=supervise_zip_fixture(output)
            self.assertEqual(result['status'],'supervised_candidate')
            self.assertEqual(result['cleanup']['state'],'complete')
            self.assertTrue(result['reader']['membership_proven_sample'])
            self.assertEqual(result['publication'],'not_authorized')

    def test_actual_worker_timeout_keeps_failure_and_cleans_group(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with self.assertRaisesRegex(RuntimeError,'worker timeout'):
                supervise_zip_fixture(output,worker_timeout_seconds=0.001)
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'complete')
            self.assertEqual(failure['cleanup']['members'],[])
            self.assertFalse((output/'supervision.json').exists())


if __name__=='__main__': unittest.main()
