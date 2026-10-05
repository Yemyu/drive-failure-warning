from contextlib import closing
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from tests.test_r_history import row
from tests.test_source_streams import csv_bytes, COLUMNS
from pipeline.r_validation.cli_fixture import _write_panel
from pipeline.r_validation.history import file_sha, HistoryError
from pipeline.r_validation.zip_panel import build_zip_candidate

ROOT=Path(__file__).resolve().parents[1]


def fixture(path, *, counts=(10,10,10,10), conflict=False, invalid=False):
    path.mkdir()
    history=path/'history.sqlite'
    _write_panel(history,[row((dt.date(2023,9,18)+dt.timedelta(days=i)).isoformat(),f'disk-{j}')
                          for i in range(13) for j in range(10)])
    with closing(sqlite3.connect(history)) as db:
        db.execute('CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT)')
        db.executemany('INSERT INTO serial_model_registry VALUES (?,?)',[(f'disk-{j}','ST4000DM000') for j in range(10)]+[('outside','OLD')])
        db.commit()
    archive=path/'synthetic.zip'
    columns=COLUMNS+[f'extra_{i}' for i in range(193-len(COLUMNS))]
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for i,count in enumerate(counts):
            day=(dt.date(2023,10,1)+dt.timedelta(days=i)).isoformat()
            rows=[{'serial_number':f'disk-{j}','smart_5_raw':'-1' if invalid else '0'} for j in range(count)]
            rows.append({'serial_number':'outside','model':'NEW' if conflict else 'OLD'})
            z.writestr(f'fixture/{day}.csv',csv_bytes(day,rows,columns))
    return archive,{'history':{'path':str(history),'sha256':file_sha(history)}}


class ZipPanelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)

    def build(self, **kwargs):
        archive,history=fixture(self.path/'input',**kwargs)
        return build_zip_candidate(archive,self.path/'output',expected_sha256=file_sha(archive),
                                   start='2023-10-01',end='2023-10-04',prefix='fixture',historical_inputs=history)

    def test_candidate_has_main_panel_full_registry_and_real_member_hashes(self):
        receipt=self.build()
        self.assertEqual(receipt['status'],'candidate_ready')
        self.assertEqual(len(receipt['source']['schema_columns']),193)
        self.assertFalse(receipt['historical_daily_rows_merged'])
        with closing(sqlite3.connect(self.path/'output/candidate.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM daily').fetchone()[0],40)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM serial_model_registry').fetchone()[0],11)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM daily WHERE source_sha256='pending'").fetchone()[0],0)
            self.assertEqual(db.execute('SELECT date,source_sha256 FROM member_counts ORDER BY date').fetchall(),
                             [(r['date'],r['sha256']) for r in receipt['source']['members']])

    def test_non_target_cross_quarter_conflict_stops(self):
        with self.assertRaisesRegex(HistoryError,'cross-quarter'):
            self.build(conflict=True)
        self.assertTrue((self.path/'output/failure.json').exists())
        self.assertFalse((self.path/'output/candidate_receipt.json').exists())

    def test_full_92_day_synthetic_archive(self):
        archive,history=fixture(self.path/'input',counts=(10,)*92)
        receipt=build_zip_candidate(archive,self.path/'output',expected_sha256=file_sha(archive),
                                    start='2023-10-01',end='2023-12-31',prefix='fixture',historical_inputs=history)
        self.assertEqual(len(receipt['source']['members']),92)
        self.assertEqual(len(receipt['coverage']['days']),92)
        self.assertEqual(receipt['source']['members'][-1]['date'],'2023-12-31')

    def test_third_low_day_stops_before_fourth_member(self):
        with self.assertRaisesRegex(HistoryError,'coverage stopped'):
            self.build(counts=(7,7,7,10))
        with closing(sqlite3.connect(self.path/'output/candidate.sqlite')) as db:
            self.assertEqual(db.execute('SELECT MAX(date) FROM member_counts').fetchone()[0],'2023-10-03')
        self.assertFalse((self.path/'output/candidate_receipt.json').exists())

    def test_true_zero_target_day_is_recorded_not_missing(self):
        receipt=self.build(counts=(0,10,10,10))
        first=receipt['coverage']['days'][0]
        self.assertEqual(first['observed'],0)
        self.assertTrue(first['low'])
        self.assertEqual(receipt['status'],'candidate_ready')

    def test_invalid_smart_preserves_failure_without_receipt(self):
        with self.assertRaisesRegex(ValueError,'negative smart'):
            self.build(invalid=True)
        self.assertFalse((self.path/'output/candidate_receipt.json').exists())


if __name__=='__main__': unittest.main()
