import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import outer_supervisor as supervisor

ROOT=Path(__file__).resolve().parents[1]


class RProcessBoundaryTests(unittest.TestCase):
    def test_signal_between_popen_return_and_registration(self):
        original=subprocess.Popen
        children=[]
        def launch(args,*pos,**kwargs):
            child=original(args,*pos,**kwargs)
            if 'snapshot-worker' in args:
                children.append(child)
                os.kill(os.getpid(),signal.SIGTERM)
            return child
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor.subprocess,'Popen',side_effect=launch):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):
                    supervisor.supervise_zip_fixture(output)
            self.assertEqual(len(children),1)
            self.assertIsNotNone(children[0].poll())
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'complete')
            self.assertFalse((output/'go.json').exists())
            self.assertFalse((output/'supervision.json').exists())

    def test_popen_failure_restores_handlers_without_success(self):
        original=subprocess.Popen
        handler=signal.getsignal(signal.SIGTERM)
        def launch(args,*pos,**kwargs):
            if 'snapshot-worker' in args: raise OSError('injected launch failure')
            return original(args,*pos,**kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(supervisor.subprocess,'Popen',side_effect=launch):
                with self.assertRaisesRegex(OSError,'launch failure'):
                    supervisor.supervise_zip_fixture(output)
            self.assertEqual(signal.getsignal(signal.SIGTERM),handler)
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'not_started')
            self.assertFalse((output/'supervision.json').exists())

    def test_leader_exit_with_descendant_ignoring_term(self):
        original=subprocess.Popen
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            ready=Path(directory)/'descendant.json'
            child_code=('import signal,time,os,json; from pathlib import Path; '
                        'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                        f'Path({str(ready)!r}).write_text(json.dumps({{"pid":os.getpid(),"pgid":os.getpgrp()}})); '
                        'time.sleep(60)')
            leader_code=('import subprocess,sys,time; from pathlib import Path; '
                         f'subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
                         f'p=Path({str(ready)!r}); '
                         '\nwhile not p.exists(): time.sleep(.01)\nsys.exit(7)\n')
            def launch(args,*pos,**kwargs):
                if 'snapshot-worker' in args:
                    args=[sys.executable,'-I','-B','-c',leader_code]
                return original(args,*pos,**kwargs)
            with patch.object(supervisor.subprocess,'Popen',side_effect=launch):
                with self.assertRaises(RuntimeError):
                    supervisor.supervise_zip_fixture(output)
            self.assertTrue(ready.exists())
            failure=json.loads((output/'failure.json').read_text())
            self.assertEqual(failure['cleanup']['state'],'complete',failure)
            self.assertEqual(failure['cleanup']['members'],[])
            self.assertFalse((output/'supervision.json').exists())


if __name__=='__main__': unittest.main()
