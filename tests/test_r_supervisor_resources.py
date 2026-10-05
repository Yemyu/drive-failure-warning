"""Bounded failures through the actual R supervisor and isolated worker."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import outer_supervisor as supervisor
from pipeline.reproducible.resource_guard import ResourceViolation

ROOT=Path(__file__).resolve().parents[1]


def run_resource_case(output, reason, phase='ready'):
    """Lower a limit after worker ready; do not allocate dangerous resources."""
    original=supervisor.ResourceGuard.check
    armed=[]
    def check(guard):
        trigger=output/('ready.json' if phase=='ready' else 'work')
        if guard.group_pgid is not None and trigger.exists() and not armed:
            changes={'rss':{'max_rss_bytes':1},'output':{'max_output_bytes':1},
                     'free-space':{'min_free_bytes':10**20},'elapsed':{'max_elapsed_seconds':0}}
            guard.limits=replace(guard.limits,**changes[reason])
            armed.append(True)
        return original(guard)
    with patch.object(supervisor.ResourceGuard,'check',check):
        try:
            supervisor.supervise_zip_fixture(output)
        except ResourceViolation as exc:
            assert reason in str(exc),str(exc)
        else:
            raise AssertionError('resource failure did not stop supervisor')
    assert armed
    failure=json.loads((output/'failure.json').read_text())
    assert failure['cleanup']['state']=='complete',failure
    assert failure['cleanup']['members']==[]
    assert not (output/'supervision.json').exists()
    if phase=='ready':
        assert not (output/'go.json').exists()
        assert not (output/'work').exists()
    else:
        assert (output/'go.json').exists()
        assert (output/'work').exists()
    return failure


class RSupervisorResourceTests(unittest.TestCase):
    def case(self,reason,phase='ready'):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            run_resource_case(Path(directory)/'run',reason,phase)

    def test_rss_limit_before_permission(self): self.case('rss')
    def test_output_limit_before_permission(self): self.case('output')
    def test_free_space_limit_before_permission(self): self.case('free-space')
    def test_elapsed_limit_before_permission(self): self.case('elapsed')

    def test_rss_limit_during_work(self): self.case('rss','work')
    def test_output_limit_during_work(self): self.case('output','work')
    def test_free_space_limit_during_work(self): self.case('free-space','work')
    def test_elapsed_limit_during_work(self): self.case('elapsed','work')

    def test_initial_space_refuses_worker_launch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            limits=replace(supervisor.ResourceLimits(),initial_free_bytes=10**20)
            with self.assertRaisesRegex(RuntimeError,'initial free space'):
                supervisor.supervise_zip_fixture(output,limits=limits)
            self.assertFalse((output/'snapshot').exists())
            self.assertFalse((output/'request.json').exists())
            self.assertFalse((output/'supervision.json').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'not_started')

    def test_sampling_exception_is_not_zero(self):
        original=supervisor.ResourceGuard.snapshot
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def sample(guard):
                if guard.group_pgid is not None and (output/'ready.json').exists():
                    raise ResourceViolation('injected process reader unavailable')
                return original(guard)
            with patch.object(supervisor.ResourceGuard,'snapshot',sample):
                with self.assertRaisesRegex(ResourceViolation,'reader unavailable'):
                    supervisor.supervise_zip_fixture(output)
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'complete')
            self.assertFalse((output/'go.json').exists())
            self.assertFalse((output/'supervision.json').exists())

    def test_unconfirmed_cleanup_refuses_success(self):
        original=supervisor._bounded_cleanup
        actual=[]
        def cleanup(*args,**kwargs):
            result=original(*args,**kwargs)
            actual.append(result)
            # Actual cleanup still runs. Inject only the unconfirmed receipt.
            return {**result,'state':'unknown','errors':['injected inspection unavailable']}
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor,'_bounded_cleanup',side_effect=cleanup):
                with self.assertRaisesRegex(RuntimeError,'cleanup unconfirmed'):
                    supervisor.supervise_zip_fixture(output)
            self.assertTrue(actual)
            self.assertTrue(all(item['state']=='complete' for item in actual))
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'unknown')
            self.assertFalse(failure['reader']['group_released'])
            self.assertFalse((output/'supervision.json').exists())


if __name__=='__main__': unittest.main()
