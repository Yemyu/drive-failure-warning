from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from pipeline.r_validation.zip_chain import run_zip_fixture
from pipeline.r_validation.post_worker_audit import audit_worker_output, inventory
from pipeline.r_validation.independent_audit import IndependentAuditError
from pipeline.reproducible.artifacts import sha256_file

ROOT=Path(__file__).resolve().parents[1]


class RPostWorkerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.work=self.base/'work'
        run_zip_fixture(self.work)
        self.model_sha=sha256_file(ROOT/'examples/small_replay/current_lr.json')

    def audit(self,expected=None):
        return audit_worker_output(self.work,inventory(self.work) if expected is None else expected,
                                   expected_model_sha256=self.model_sha)

    def test_separate_cli_recomputes_read_only(self):
        before=inventory(self.work); binding=self.base/'inventory.json'
        binding.write_text(json.dumps(before))
        result_path=self.base/'audit.json'
        result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'tools/run_r_validation.py'),
            'audit-zip-worker','--attempt',str(self.work),'--inventory',str(binding),
            '--inventory-sha256',sha256_file(binding),'--model-sha256',self.model_sha,
            '--output',str(result_path)],cwd='/',capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result_path.read_text())['status'],'pass')
        self.assertEqual(inventory(self.work),before)

    def test_changed_inventory_refused(self):
        before=inventory(self.work)
        (self.work/'extra.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'inventory changed'): self.audit(before)

    def test_changed_model_with_new_inventory_refused(self):
        model=self.work/'evaluation/current_lr.json'; model.write_text(model.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'model binding'): self.audit()

    def test_wrong_calendar_with_updated_seal_and_hash_refused(self):
        selection=self.work/'evaluation/selection.sqlite'
        with closing(sqlite3.connect(selection)) as db:
            db.execute("UPDATE metadata SET value='2023-10-13' WHERE key='end'"); db.commit()
        self.rebind_selection(selection)
        with self.assertRaisesRegex(ValueError,'contract mismatch'): self.audit()

    def rebind_selection(self,selection):
        digest=sha256_file(selection)
        seal=self.work/'evaluation/selection_seal.json'; value=json.loads(seal.read_text())
        value['selection_sha256']=digest; seal.write_text(json.dumps(value))
        chain=self.work/'zip_chain_result.json'; value=json.loads(chain.read_text())
        value['selection_sha256']=digest; chain.write_text(json.dumps(value))

    def test_wrong_score_with_updated_hashes_recomputed(self):
        selection=self.work/'evaluation/selection.sqlite'
        with closing(sqlite3.connect(selection)) as db:
            db.execute('UPDATE scores SET score=score+0.25'); db.commit()
        self.rebind_selection(selection)
        with self.assertRaises(IndependentAuditError): self.audit()


if __name__=='__main__': unittest.main()
