import errno
import unittest
from unittest.mock import Mock,patch
from pipeline.reproducible import supervisor


class CleanupExitRaceTests(unittest.TestCase):
    def check_case(self,exit_code,inspection):
        process=Mock(); process.poll.return_value=exit_code; process.returncode=exit_code
        calls=iter([([77],{},None),inspection,([],{},None)])
        def inspect(*args,**kwargs): return next(calls,([],{},None))
        with patch.object(supervisor,'_inspect_group_identity',side_effect=inspect),patch.object(supervisor.os,'killpg',side_effect=PermissionError(errno.EPERM,'denied')):
            return supervisor._bounded_cleanup(77,process,known_members={77})

    def test_exited_child_and_fresh_empty_group_resolve_race(self):
        result=self.check_case(0,([],{},None))
        self.assertEqual(result['state'],'complete')
        self.assertEqual(result['errors'],[])
        self.assertEqual(result['resolved_signal_races'][0]['errno'],errno.EPERM)

    def test_live_child_does_not_resolve_permission_error(self):
        result=self.check_case(None,([],{},None))
        self.assertEqual(result['state'],'unknown')
        self.assertTrue(result['errors'])
        self.assertEqual(result['resolved_signal_races'],[])

    def test_failed_query_does_not_resolve_permission_error(self):
        result=self.check_case(0,(None,{},'unavailable'))
        self.assertEqual(result['state'],'unknown')
        self.assertTrue(result['errors'])
        self.assertEqual(result['resolved_signal_races'],[])


if __name__=='__main__': unittest.main()
