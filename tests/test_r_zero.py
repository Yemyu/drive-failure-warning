"""No eligible device-days is a reportable result, not zero effectiveness."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.r_validation.cli_fixture import _write_csv
from pipeline.r_validation.stage_runner import run_stage
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
from tools import run_small_replay as replay

ROOT = Path(__file__).resolve().parents[1]


def zero_chain(path):
    rows, _ = replay._parse_source(replay.SOURCE_CSV)
    # One observation per device, so none can reach 12 observations.
    source = path/'source.csv'
    _write_csv(source, [r for r in rows if r['date'] == '2023-01-15'])
    return run_stage(path/'attempt', root=ROOT, profile='synthetic', source_csv=source,
                     spec={'score_start':'2023-01-15','score_end':'2023-01-28',
                           'event_start':'2023-01-22','event_end':'2023-01-29',
                           'outcome_cutoff':'2023-02-04','horizon_days':7})


class ZeroQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.result = zero_chain(self.path)
        self.chain = self.path/'attempt/cli_chain'

    def test_empty_chain_completes_with_null_effectiveness(self):
        self.assertEqual(self.result['audit']['status'], 'pass')
        evaluation = json.loads((self.chain/'evaluation.json').read_text())
        for method in evaluation['methods'].values():
            self.assertEqual(method['alerts'], 0)
            self.assertEqual(method['event_opportunity_total'], 0)
            self.assertIsNone(method['event_recall_at_opportunity'])
            self.assertIsNone(method['precision_lower_bound'])
        protocol = evaluation['protocol_metrics']
        self.assertEqual(protocol['quality_gate']['status'], 'not_evaluable')
        self.assertEqual(protocol['bootstrap']['roster_devices'], 0)
        self.assertIsNone(protocol['bootstrap']['interval'])
        self.assertEqual(protocol['bootstrap']['differences'], [])
        self.assertFalse(protocol['interpretation_gate']['support_gain'])
        with closing(sqlite3.connect(self.chain/'selection.sqlite')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM daily').fetchone()[0], 28)
            self.assertEqual(db.execute('SELECT SUM(eligible_count),SUM(budget_k),SUM(alerts_count) FROM daily').fetchone(), (0,0,0))

    def test_deleted_zero_budget_day_is_rejected(self):
        with closing(sqlite3.connect(self.chain/'selection.sqlite')) as db:
            db.execute('DELETE FROM daily WHERE rowid=(SELECT rowid FROM daily LIMIT 1)')
            db.commit()
        with self.assertRaisesRegex(IndependentAuditError, 'daily key set'):
            audit_selection(self.chain/'selection.sqlite', self.chain/'panel.sqlite',
                            json.loads((self.chain/'evaluation.json').read_text()),None,
                            model=replay._load_model())


if __name__ == '__main__':
    unittest.main()
