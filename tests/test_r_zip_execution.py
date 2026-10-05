from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from pipeline.r_validation.zip_chain import run_zip_fixture
from pipeline.r_validation.zip_execution import run_bound_zip_chain
from pipeline.r_validation.history import file_sha
from pipeline.r_validation.stage_runner import StageRunnerError

ROOT=Path(__file__).resolve().parents[1]


class RZipExecutionTests(unittest.TestCase):
    def test_full_quarter_uses_92_members_and_85_scoring_days(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            result=run_zip_fixture(output,calendar='quarter')
            with zipfile.ZipFile(output/'synthetic.zip') as archive:
                names=archive.namelist()
                self.assertEqual(len([name for name in names if name.endswith('.csv')]),92)
                self.assertEqual(set(name for name in names if not name.endswith('.csv')),
                                 {'fixture/.DS_Store','__MACOSX/fixture/._.DS_Store'})
            with closing(sqlite3.connect(output/'evaluation/selection.sqlite')) as db:
                metadata=dict(db.execute('SELECT key,value FROM metadata'))
                self.assertEqual(metadata['end'],'2023-12-24')
                self.assertEqual(metadata['event_end'],'2023-12-25')
                self.assertEqual(metadata['outcome_cutoff'],'2023-12-31')
                self.assertEqual(db.execute('SELECT COUNT(DISTINCT decision_date) FROM scores').fetchone()[0],85)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM features').fetchone()[0],694)
            self.assertEqual(result['score_evaluation_audit']['status'],'pass')
            self.assertEqual(result['publication'],'not_authorized')

    def refused_before_source_read(self,wrong_model=False):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            root=Path(directory); archive=root/'dummy.zip'; archive.write_bytes(b'not to be read')
            model=ROOT/'examples/small_replay/current_lr.json'
            spec={'score_start':'2023-10-01','score_end':'2023-12-24','event_start':'2023-10-08',
                  'event_end':'2023-12-25','outcome_cutoff':'2023-12-31','horizon_days':7,'allow_smart_decreases':True}
            if not wrong_model: spec['event_end']='2023-12-24'
            with patch('pipeline.r_validation.zip_execution.build_zip_candidate') as build:
                with self.assertRaises(StageRunnerError):
                    run_bound_zip_chain(root/'output',archive,
                        source_args={'start':'2023-10-01','end':'2023-12-31'},model_source=model,
                        expected_model_sha256='0'*64 if wrong_model else file_sha(model),spec=spec)
                build.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_model_binding_checked_before_zip_content(self): self.refused_before_source_read(True)
    def test_wrong_event_end_refused_before_zip_content(self): self.refused_before_source_read(False)


if __name__=='__main__': unittest.main()
