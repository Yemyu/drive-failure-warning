"""99/100 opportunities reach the actual source-to-independent-audit pipeline."""
from contextlib import closing
import copy
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from tests.test_r_history import row
from pipeline.r_validation.cli_fixture import _write_csv, _write_panel
from pipeline.r_validation.history import file_sha
from pipeline.r_validation.stage_runner import run_stage
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
from tools import run_small_replay as replay

ROOT = Path(__file__).resolve().parents[1]


def event_chain(output, events, disappearances=0):
    output.mkdir()
    serials = [f"event-{i:03d}" for i in range(events)] + [f"exit-{i:03d}" for i in range(disappearances)]
    history = output/'history.sqlite'
    _write_panel(history, [row((dt.date(2023,9,18)+dt.timedelta(days=i)).isoformat(), serial)
                           for i in range(13) for serial in serials])
    with closing(sqlite3.connect(history)) as db:
        db.execute('CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT)')
        db.executemany('INSERT INTO serial_model_registry VALUES (?,?)', [(s,replay.MODEL) for s in serials])
        db.commit()
    bindings={'history':{'path':str(history),'sha256':file_sha(history)}}
    source=output/'source.csv'
    rows=[row((dt.date(2023,10,1)+dt.timedelta(days=i)).isoformat(), serial, int(i==7))
          for i in range(8) for serial in serials if i < 7 or serial.startswith('event-')]
    _write_csv(source, rows)
    start=time.monotonic()
    result=run_stage(output/'attempt',root=ROOT,profile='synthetic',source_csv=source,historical_inputs=bindings,
                     spec={'score_start':'2023-10-01','score_end':'2023-10-07',
                           'event_start':'2023-10-08','event_end':'2023-10-08',
                           'outcome_cutoff':'2023-10-14','horizon_days':7,'allow_smart_decreases':True})
    elapsed=time.monotonic()-start
    (output/'elapsed.json').write_text(json.dumps({'seconds':elapsed})+'\n')
    return result, bindings, elapsed


class EventBoundaryTests(unittest.TestCase):
    def check_chain(self, events, exits=0):
        temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(temp.cleanup)
        output=Path(temp.name)/'case'
        result,bindings,elapsed=event_chain(output,events,exits)
        self.assertLess(elapsed,30)
        self.assertEqual(result['audit']['status'],'pass')
        chain=output/'attempt/cli_chain'
        evaluation=json.loads((chain/'evaluation.json').read_text())
        metrics=evaluation['protocol_metrics']
        self.assertEqual(metrics['opportunity_events'],events)
        self.assertEqual(metrics['bootstrap']['valid_replicates'],2000)
        self.assertEqual(metrics['eligible_roster_devices'],events+exits)
        self.assertIsNone(metrics['interval_publication']['interval_pp'])
        self.assertFalse(metrics['interpretation_gate']['support_gain'])
        return output,chain,evaluation,bindings

    def test_99_events_never_publish_confirmatory_interval(self):
        _,_,evaluation,_=self.check_chain(99)
        metrics=evaluation['protocol_metrics']
        self.assertIsNone(metrics['bootstrap']['interval'])
        self.assertIn('insufficient_events',metrics['interval_publication']['blocking_reasons'])

    def test_100_events_compute_interval_but_do_not_self_accept(self):
        output,chain,evaluation,bindings=self.check_chain(100)
        metrics=evaluation['protocol_metrics']
        self.assertIsNotNone(metrics['bootstrap']['interval'])
        self.assertEqual(metrics['quality_gate']['status'],'pass')
        self.assertEqual(metrics['interval_publication']['blocking_reasons'],['complete_acceptance_pending'])
        changed=copy.deepcopy(evaluation)
        changed['protocol_metrics']['interval_publication']['interval_pp']=metrics['bootstrap']['interval']
        with self.assertRaisesRegex(IndependentAuditError,'interval publication'):
            audit_selection(chain/'selection.sqlite',chain/'panel.sqlite',changed,None,
                            model=replay._load_model(),source_path=output/'source.csv',historical_inputs=bindings)

    def test_100_events_quality_failure_is_still_withheld(self):
        _,_,evaluation,_=self.check_chain(100,100)
        metrics=evaluation['protocol_metrics']
        self.assertEqual(metrics['status'],'data_quality_failed')
        self.assertEqual(metrics['quality_gate']['unknown_ratio'],0.5)
        self.assertIsNotNone(metrics['bootstrap']['interval'])
        self.assertIn('data_quality_not_passed',metrics['interval_publication']['blocking_reasons'])
        self.assertFalse(metrics['interpretation_gate']['numerical_conditions_met'])


if __name__ == '__main__':
    unittest.main()
