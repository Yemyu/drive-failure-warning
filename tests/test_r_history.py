"""Cross-quarter qualification using local synthetic history and source."""
from contextlib import closing
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.r_validation.cli_fixture import _write_csv, _write_panel
from pipeline.r_validation.history import file_sha, HistoryError
from pipeline.r_validation.stage_runner import run_stage
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
from tools import run_small_replay as replay

ROOT = Path(__file__).resolve().parents[1]


def row(day, serial, failure=0):
    return {'date': day, 'serial_number': serial, 'model': replay.MODEL,
            'capacity_bytes': 4000787030016, 'failure': failure,
            **{f'smart_{field}_raw': 100 if field == 9 else 0 for field in replay.SMART_FIELDS}}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        history = [row('2023-02-01', 'old-failure', 1)]
        history += [row((dt.date(2023,9,18)+dt.timedelta(days=i)).isoformat(), 'resident') for i in range(13)]
        self.history = self.path/'history.sqlite'
        _write_panel(self.history, history)
        with closing(sqlite3.connect(self.history)) as db:
            db.execute('CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT)')
            db.executemany('INSERT INTO serial_model_registry VALUES (?,?)', [(s,replay.MODEL) for s in ['old-failure','resident']])
            db.commit()
        source = [row((dt.date(2023,10,1)+dt.timedelta(days=i)).isoformat(), s)
                  for i in range(21) for s in ['new-device','old-failure','resident']]
        self.source = self.path/'source.csv'
        _write_csv(self.source, source)
        self.bindings = {'history': {'path': str(self.history), 'sha256': file_sha(self.history)}}
        self.spec = {'score_start':'2023-10-01','score_end':'2023-10-14',
                     'event_start':'2023-10-08','event_end':'2023-10-15',
                     'outcome_cutoff':'2023-10-21','horizon_days':7,'allow_smart_decreases':True}

    def run_chain(self):
        return run_stage(self.path/'attempt', root=ROOT, profile='synthetic',
                         source_csv=self.source, spec=self.spec, historical_inputs=self.bindings)

    def test_history_failure_and_new_device_eligibility(self):
        result = self.run_chain()
        self.assertEqual(result['audit']['status'], 'pass')
        self.assertEqual(len(result['historical_inputs']['coverage_seed']), 7)
        chain = self.path/'attempt/cli_chain'
        with closing(sqlite3.connect(chain/'selection.sqlite')) as db:
            first = dict(db.execute('SELECT serial_number,MIN(decision_date) FROM features GROUP BY serial_number'))
        self.assertEqual(first, {'resident':'2023-10-01','new-device':'2023-10-12'})
        # Delete an old failure from the merged panel: independent input union must catch it.
        with closing(sqlite3.connect(chain/'panel.sqlite')) as db:
            db.execute("DELETE FROM daily WHERE date='2023-02-01'")
            db.commit()
        with self.assertRaisesRegex(IndependentAuditError, 'source CSV and panel rows differ'):
            audit_selection(chain/'selection.sqlite',chain/'panel.sqlite',
                            json.loads((chain/'evaluation.json').read_text()),self.path/'bad_audit.json',
                            model=replay._load_model(), source_path=self.source, historical_inputs=self.bindings)

    def test_historical_hash_change_refused(self):
        self.bindings['history']['sha256'] = '0'*64
        with self.assertRaisesRegex(HistoryError, 'SHA mismatch'):
            self.run_chain()

    def test_production_branch_consumes_bound_history(self):
        receipt = self.path/'receipt.json'
        receipt.write_text(json.dumps({
            'schema':'r-validation-source-receipt-v1','status':'verified',
            'source_object_id':'local-synthetic-test','transport':'local_verified_copy',
            'source_sha256':file_sha(self.source),
        }))
        model = ROOT/'examples/small_replay/current_lr.json'
        result = run_stage(self.path/'production_fixture',root=ROOT,profile='production',
                           source_csv=self.source,model_json=model,source_receipt=receipt,
                           source_object_id='local-synthetic-test',approved_model_sha256=file_sha(model),
                           release_verified=True,spec={**self.spec,'coverage_check':False},historical_inputs=self.bindings)
        self.assertEqual(result['profile'],'production')
        self.assertEqual(result['historical_inputs']['inputs'][0]['rows_added'],14)
        self.assertEqual(result['independent_audit']['status'],'pass')
        self.assertEqual(result['coverage']['status'],'pass')
        self.assertEqual(len(result['coverage']['days']),21)

    def test_registry_conflict_refused_even_without_daily_history(self):
        with closing(sqlite3.connect(self.history)) as db:
            db.execute("INSERT INTO serial_model_registry VALUES ('new-device','OTHER-MODEL')")
            db.commit()
        self.bindings['history']['sha256'] = file_sha(self.history)
        with self.assertRaisesRegex(HistoryError, 'serial/model conflict'):
            self.run_chain()

    def test_missing_coverage_seed_refused(self):
        with closing(sqlite3.connect(self.history)) as db:
            db.execute("DELETE FROM daily WHERE date='2023-09-27'")
            db.commit()
        self.bindings['history']['sha256'] = file_sha(self.history)
        with self.assertRaisesRegex(HistoryError, 'coverage seed'):
            self.run_chain()

    def test_future_history_refused(self):
        with closing(sqlite3.connect(self.history)) as db:
            db.execute("UPDATE daily SET date='2023-10-01' WHERE date='2023-09-30'")
            db.commit()
        self.bindings['history']['sha256'] = file_sha(self.history)
        with self.assertRaisesRegex(HistoryError, 'future records'):
            self.run_chain()


if __name__ == '__main__':
    unittest.main()
