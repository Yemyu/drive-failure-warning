from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import outer_supervisor as supervisor
from pipeline.reproducible.artifacts import verify_manifest

ROOT=Path(__file__).resolve().parents[1]


class RQuarterSupervisionTests(unittest.TestCase):
    def test_worker_wrong_calendar_refuses_verification(self):
        original=supervisor.write_json_exclusive
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def write(path,value):
                if path.name=='request.json': value={**value,'calendar':'short'}
                return original(path,value)
            with patch.object(supervisor,'write_json_exclusive',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'candidate binding mismatch'):
                    supervisor.supervise_zip_fixture(output,calendar='quarter')
            self.assertFalse((output/'verification_request.json').exists())
            self.assertFalse((output/'synthetic_complete.json').exists())

    def test_quarter_cli_completes_with_bound_calendar(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'tools/run_r_validation.py'),
                'supervised-zip-fixture','--calendar','quarter','--output',str(output)],
                cwd='/',capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            complete=verify_manifest(output/'synthetic_complete.json',expected_status='synthetic_complete')
            self.assertEqual(complete['calendar'],'quarter')
            self.assertFalse(complete['real_data_release'])
            for name in ('request.json','verification_request.json','verification.json','work/worker_candidate.json'):
                self.assertEqual(json.loads((output/name).read_text())['calendar'],'quarter')
            config=json.loads((ROOT/'configs/r_validation_v1.json').read_text())
            with closing(sqlite3.connect(output/'work/evaluation/selection.sqlite')) as db:
                metadata=dict(db.execute('SELECT key,value FROM metadata'))
                for key in ('event_start','event_end','outcome_cutoff'):
                    self.assertEqual(metadata[key],config['dates'][key])
                self.assertEqual(metadata['end'],config['dates']['score_end'])
                self.assertEqual(db.execute('SELECT COUNT(*) FROM features').fetchone()[0],694)

    def test_verifier_wrong_calendar_refuses_completion(self):
        original=supervisor.write_json_exclusive
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def write(path,value):
                if path.name=='verification_request.json': value={**value,'calendar':'short'}
                return original(path,value)
            with patch.object(supervisor,'write_json_exclusive',side_effect=write):
                with self.assertRaisesRegex(RuntimeError,'verifier exited'):
                    supervisor.supervise_zip_fixture(output,calendar='quarter')
            self.assertIn('contract mismatch',(output/'verifier.log').read_text())
            self.assertFalse((output/'synthetic_complete.json').exists())
            self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')

    def test_unknown_calendar_refused_before_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with self.assertRaisesRegex(ValueError,'unknown synthetic calendar'):
                supervisor.supervise_zip_fixture(output,calendar='custom')
            self.assertFalse(output.exists())


if __name__=='__main__': unittest.main()
