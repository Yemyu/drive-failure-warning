"""Input diagnostics must not inherit eligibility or alert filters."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.r_validation.decrease_ledger import build_decrease_ledger, audit_decrease_ledger
from pipeline.r_validation.history import file_sha, HistoryError

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'ST4000DM000'


class DecreaseLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = self.root/'history.sqlite'; self.candidate = self.root/'candidate.sqlite'
        self.ledger = self.root/'ledger.sqlite'
        for path in (self.history, self.candidate):
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE daily(date TEXT,serial_number TEXT,model TEXT,failure INTEGER,smart_5_raw INTEGER,smart_9_raw INTEGER,smart_187_raw INTEGER)')
        self.insert(self.history, [('2023-08-01','OLD',MODEL,1,10,100,20),
                                  ('2023-09-30','OLD',MODEL,0,None,None,None)])
        self.insert(self.candidate, [('2023-10-01','OLD',MODEL,0,9,99,19),
                                    ('2023-10-01','NEW',MODEL,0,8,80,18),
                                    ('2023-10-02','OLD',MODEL,0,None,None,None),
                                    ('2023-10-03','OLD',MODEL,1,8,101,0),
                                    ('2023-10-03','NEW',MODEL,0,7,79,17),
                                    ('2023-10-04','OLD',MODEL,0,7,102,0),
                                    ('2023-10-04','OTHER','other-model',0,0,0,0)])
        self.binding = {'history':{'path':str(self.history),'sha256':file_sha(self.history)}}
        self.dates = dict(start='2023-10-01',end='2023-10-04')

    def insert(self, path, rows):
        with closing(sqlite3.connect(path)) as db:
            db.executemany('INSERT INTO daily VALUES (?,?,?,?,?,?,?)',rows);db.commit()

    def build(self):
        return build_decrease_ledger(self.candidate,self.binding,self.ledger,**self.dates)

    def audit(self):
        return audit_decrease_ledger(self.candidate,self.binding,self.ledger,**self.dates)

    def test_cross_quarter_null_gap_new_and_previously_failed_all_count(self):
        report=self.build();self.assertEqual(report['decreases'],9)
        self.assertEqual(self.audit()['decreases'],9)
        with closing(sqlite3.connect(self.ledger)) as db:
            self.assertEqual(db.execute("SELECT previous_date,previous_value,current_value FROM decreases WHERE date='2023-10-01' AND serial_number='OLD' AND field='smart_5_raw'").fetchone(),('2023-08-01',10,9))
            self.assertEqual(db.execute("SELECT previous_date FROM decreases WHERE date='2023-10-03' AND serial_number='OLD' AND field='smart_5_raw'").fetchone(),('2023-10-01',))
            self.assertEqual(db.execute('SELECT SUM(observations) FROM days').fetchone()[0],6)

    def test_independent_audit_rejects_deleted_row(self):
        self.build()
        with closing(sqlite3.connect(self.ledger)) as db:
            db.execute('DELETE FROM decreases WHERE rowid=(SELECT MIN(rowid) FROM decreases)');db.commit()
        with self.assertRaisesRegex(HistoryError,'row differs'):self.audit()

    def test_independent_audit_rejects_changed_value(self):
        self.build()
        with closing(sqlite3.connect(self.ledger)) as db:
            db.execute('UPDATE decreases SET previous_value=999');db.commit()
        with self.assertRaisesRegex(HistoryError,'row differs'):self.audit()

    def test_independent_audit_rejects_missing_day(self):
        self.build()
        with closing(sqlite3.connect(self.ledger)) as db:
            db.execute("DELETE FROM days WHERE date='2023-10-02'");db.commit()
        with self.assertRaisesRegex(HistoryError,'coverage differs'):self.audit()

    def test_history_sha_mismatch(self):
        self.binding['history']['sha256']='0'*64
        with self.assertRaisesRegex(HistoryError,'SHA mismatch'):self.build()

    def test_refuses_existing_ledger(self):
        self.build();before=file_sha(self.ledger)
        with self.assertRaises(FileExistsError):self.build()
        self.assertEqual(file_sha(self.ledger),before)

    def test_overlapping_history_is_not_silently_deduplicated(self):
        other=self.root/'other.sqlite';other.write_bytes(self.history.read_bytes())
        self.binding['other']={'path':str(other),'sha256':file_sha(other)}
        with self.assertRaisesRegex(HistoryError,'overlapping'):self.build()

    def test_historical_future_is_rejected(self):
        self.insert(self.history,[('2023-10-01','FUTURE',MODEL,0,1,2,3)])
        self.binding['history']['sha256']=file_sha(self.history)
        with self.assertRaisesRegex(HistoryError,'historical future'):self.build()
