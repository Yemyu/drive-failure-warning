from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tests.test_r_zip_panel import fixture
from pipeline.r_validation.zip_panel import build_zip_candidate
from pipeline.r_validation.zip_audit import audit_zip_candidate, ZipAuditError
from pipeline.r_validation.history import file_sha

ROOT=Path(__file__).resolve().parents[1]


class ZipAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name); self.output=self.path/'candidate'
        self.archive,self.history=fixture(self.path/'input')
        self.args=dict(expected_sha256=file_sha(self.archive),start='2023-10-01',end='2023-10-04',prefix='fixture',historical_inputs=self.history)
        build_zip_candidate(self.archive,self.output,**self.args)

    def audit(self): return audit_zip_candidate(self.archive,self.output,**self.args)

    def update_panel_hash(self):
        p=self.output/'candidate_receipt.json'; r=json.loads(p.read_text())
        r['panel_sha256']=file_sha(self.output/'candidate.sqlite'); p.write_text(json.dumps(r))

    def test_independent_audit_is_read_only(self):
        files=[p for p in self.path.rglob('*') if p.is_file()]
        before={str(p):file_sha(p) for p in files}
        self.assertEqual(self.audit()['selected_rows'],40)
        self.assertEqual(before,{str(p):file_sha(p) for p in files})

    def test_numeric_tamper_rejected_after_rehash(self):
        with closing(sqlite3.connect(self.output/'candidate.sqlite')) as db:
            db.execute('UPDATE daily SET smart_5_raw=99'); db.commit()
        self.update_panel_hash()
        with self.assertRaisesRegex(ZipAuditError,'panel row'): self.audit()

    def test_non_target_registry_tamper_rejected(self):
        with closing(sqlite3.connect(self.output/'candidate.sqlite')) as db:
            db.execute("DELETE FROM serial_model_registry WHERE serial_number='outside'"); db.commit()
        self.update_panel_hash()
        with self.assertRaisesRegex(ZipAuditError,'registry'): self.audit()

    def test_member_counts_tamper_rejected(self):
        with closing(sqlite3.connect(self.output/'candidate.sqlite')) as db:
            db.execute('UPDATE member_counts SET source_rows=999'); db.commit()
        self.update_panel_hash()
        with self.assertRaisesRegex(ZipAuditError,'member count'): self.audit()

    def test_both_coverage_copies_tampered_still_rejected(self):
        p=self.output/'candidate_receipt.json'; r=json.loads(p.read_text())
        r['coverage']['days'][0]['previous_7_day_median']=999
        p.write_text(json.dumps(r))
        (self.output/'coverage.json').write_text(json.dumps(r['coverage']))
        with self.assertRaisesRegex(ZipAuditError,'coverage'): self.audit()


if __name__=='__main__': unittest.main()
