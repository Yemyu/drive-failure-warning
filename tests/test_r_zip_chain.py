from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation.zip_chain import run_zip_fixture

ROOT=Path(__file__).resolve().parents[1]


class ZipChainTests(unittest.TestCase):
    def test_external_cwd_cli_completes_all_three_audits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'attempt'; env=dict(os.environ); env.pop('PYTHONPATH',None)
            completed=subprocess.run([sys.executable,'-B',str(ROOT/'tools/run_r_validation.py'),
                                      'zip-fixture','--output',str(output)],cwd='/',env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result=json.loads((output/'zip_chain_result.json').read_text())
            for key in ('source_audit','merge_audit','score_evaluation_audit'): self.assertEqual(result[key]['status'],'pass')
            self.assertTrue(result['evaluation_after_seal'])
            with closing(sqlite3.connect(output/'evaluation/selection.sqlite')) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM features WHERE serial_number='zip-fixture-1'").fetchone()[0],0)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM features').fetchone()[0],126)
            evaluation=json.loads((output/'evaluation/evaluation.json').read_text())
            self.assertEqual(evaluation['protocol_metrics']['opportunity_events'],1)

    def test_audit_failure_never_writes_success_summary(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'attempt'
            with patch('pipeline.r_validation.zip_execution.audit_selection',side_effect=ValueError('injected audit failure')):
                with self.assertRaisesRegex(ValueError,'injected audit'): run_zip_fixture(output)
            self.assertFalse((output/'zip_chain_result.json').exists())
            self.assertTrue((output/'failure.json').exists())


if __name__=='__main__': unittest.main()
