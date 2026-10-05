from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tests.test_r_zip_panel import fixture
from tests.test_r_history import row
from pipeline.r_validation.history import file_sha, HistoryError
from pipeline.r_validation.zip_panel import build_zip_candidate
from pipeline.r_validation.zip_history import prepare_zip_scoring_panel, verify_merged_panel
from pipeline.r_validation.zip_audit import ZipAuditError
from tools import run_small_replay as replay

ROOT=Path(__file__).resolve().parents[1]


class ZipHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.archive,self.history=fixture(self.path/'input')
        history=Path(self.history['history']['path'])
        with closing(sqlite3.connect(history)) as db:
            old=row('2023-02-01','disk-0',1)
            db.execute(f"INSERT INTO daily VALUES ({','.join('?' for _ in replay.SOURCE_COLUMNS)})",tuple(old[k] for k in replay.SOURCE_COLUMNS))
            db.commit()
        self.history['history']['sha256']=file_sha(history)
        self.args=dict(expected_sha256=file_sha(self.archive),start='2023-10-01',end='2023-10-04',prefix='fixture',historical_inputs=self.history)
        self.candidate=self.path/'candidate'
        build_zip_candidate(self.archive,self.candidate,**self.args)

    def prepare(self): return prepare_zip_scoring_panel(self.archive,self.candidate,self.path/'prepared',**self.args)

    def test_actual_scorer_excludes_old_failure_and_keeps_history(self):
        before=file_sha(self.candidate/'candidate.sqlite')
        result=self.prepare(); panel=self.path/'prepared/panel.sqlite'
        self.assertEqual(result['merge_audit']['rows'],171)
        self.assertEqual(result['merge_audit']['full_source_identities'],11)
        access=[]
        with closing(sqlite3.connect(panel)) as db:
            db.row_factory=sqlite3.Row
            replay._score_and_select(replay._asof_reader(db,access),[],replay._load_model(),self.path/'selection.sqlite',access,
                                     spec={'score_start':'2023-10-01','score_end':'2023-10-04','outcome_cutoff':'2023-10-11','horizon_days':7,'allow_smart_decreases':True})
        with closing(sqlite3.connect(self.path/'selection.sqlite')) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM features WHERE serial_number='disk-0'").fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM features').fetchone()[0],36)
        self.assertEqual(file_sha(self.candidate/'candidate.sqlite'),before)

    def test_deleted_old_failure_rejected_by_independent_merge_audit(self):
        self.prepare(); panel=self.path/'prepared/panel.sqlite'
        with closing(sqlite3.connect(panel)) as db:
            db.execute("DELETE FROM daily WHERE date='2023-02-01'"); db.commit()
        with self.assertRaisesRegex(HistoryError,'merged row differs'):
            verify_merged_panel(panel,self.candidate/'candidate.sqlite',self.history,start='2023-10-01')

    def test_corrupted_candidate_rehashed_still_refused_before_output(self):
        panel=self.candidate/'candidate.sqlite'
        with closing(sqlite3.connect(panel)) as db:
            db.execute('UPDATE daily SET failure=1'); db.commit()
        p=self.candidate/'candidate_receipt.json'; r=json.loads(p.read_text()); r['panel_sha256']=file_sha(panel); p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ZipAuditError,'panel row'): self.prepare()
        self.assertFalse((self.path/'prepared').exists())


if __name__=='__main__': unittest.main()
