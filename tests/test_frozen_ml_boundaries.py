"""Synthetic E1 audit and publication regressions; no real attempt required."""
import csv
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import frozen_ml_checks as checks
import analyze_frozen_ml as ml
import run_frozen_ml_supervised as runner


def setUpModule():
    (ROOT / '.tmp').mkdir(exist_ok=True)

class NumericTests(unittest.TestCase):
    def test_equal_counts_different_keys_and_duplicates_are_rejected(self):
        import sqlite3
        db=sqlite3.connect(':memory:')
        try:
            db.execute('CREATE TABLE scores (model TEXT,decision_date TEXT,serial_number TEXT)')
            db.executemany('INSERT INTO scores VALUES (?,?,?)',[('current_lr','2023-10-01','a'),('smart_nonzero','2023-10-01','b')])
            with self.assertRaisesRegex(ml.AnalysisError,'key sets'):ml.assert_q4_score_keys(db)
            db.execute("UPDATE scores SET serial_number='a'")
            ml.assert_q4_score_keys(db)
            db.execute("INSERT INTO scores VALUES ('current_lr','2023-10-01','a')")
            with self.assertRaisesRegex(ml.AnalysisError,'duplicate'):ml.assert_q4_score_keys(db)
        finally:db.close()

    def test_packed_rows_preserve_unknown_and_intern_devices(self):
        source=[(0.,None,'a'),(1.,1,'a'),(-1.,0,'b')]
        rows=ml.PackedRows()
        for row in source:rows.append(row)
        self.assertEqual(list(rows),source)
        self.assertEqual(len(rows.names),2)
        checks.require_equal(ml.calibration_rows(rows),checks.calibration(source))
        checks.require_equal(ml.average_precision_rows(rows)['value'],checks.ranking(source)['ap'])

    def test_batches_are_bounded_and_complete(self):
        import sqlite3
        db=sqlite3.connect(':memory:')
        try:
            db.execute('CREATE TABLE fixture (n INTEGER)')
            db.executemany('INSERT INTO fixture VALUES (?)',((i,) for i in range(100003)))
            sizes=[len(batch) for batch in ml.batches(db.execute('SELECT n FROM fixture ORDER BY n'))]
            self.assertEqual(sizes,[50000,50000,3])
        finally:db.close()

    def test_pr_compaction_keeps_first_threshold_and_recall_changes(self):
        rows=[(float(i),1 if i in (3,7) else 0,str(i)) for i in range(20)]
        ranked=ml.average_precision_rows(rows)
        self.assertEqual(len(ranked['curve']),3)
        checks.require_equal(ml.fixed_recall_points(ranked['curve']),checks.ranking(rows)['points'])

    def test_probability_totals_and_no_capture_status(self):
        rows=[(0.,1,'a'),(0.,0,'b'),(0.,None,'c')]
        value=ml.calibration_rows(rows)
        self.assertEqual(value['predicted_positive_known'],1.)
        self.assertEqual(value['observed_positive_known'],1)
        self.assertEqual(value['predicted_observed_ratio_known'],1.)
        self.assertEqual(value['unknown_ratio'],1/3)
        checks.require_equal(value,checks.calibration(rows))
        facts=dict(event_start='2023-07-08',event_end='2023-09-24',eligible_device_days=100,opportunity_events=2)
        result=ml.burden([],**facts)
        self.assertEqual(result['capture_status'],'no_captured_event')
        self.assertIsNone(result['alerts_per_event'])
        checks.require_equal(result,checks.alert_burden([],**facts))

    def test_independent_tied_ranking_and_pr_tamper(self):
        rows=[(2.,1,'a'),(2.,0,'b'),(3.,None,'c'),(0.,0,'d')]
        result=checks.ranking(rows)
        self.assertEqual(result['ap'],.5)
        self.assertEqual(result['unknown'],1)
        expected=ml.fixed_recall_points(ml.average_precision_rows(rows)['curve'])
        checks.require_equal(result['points'],expected)
        exported=[{k:'' if v is None else str(v) for k,v in p.items()} for p in expected]
        checks.compare_csv(exported,expected,'PR')
        exported[0]['precision']='0.99'
        with self.assertRaises(ValueError):checks.compare_csv(exported,expected,'PR')
        self.assertIsNone(checks.ranking([(0.,0,'a')])['ap'])

    def test_fixed_linear_score_uses_all_columns_and_imputation(self):
        parameters={'intercept':-2.,'features':[dict(position=i,source_name=f'x{i}',imputation_mean=2.,standardization_mean=1.,standardization_scale=2.,coefficient=i+1.) for i in range(16)]}
        values={f'x{i}':3. for i in range(16)}
        self.assertEqual(checks.linear_score(parameters,values),134.)
        values['x15']=None
        self.assertEqual(checks.linear_score(parameters,values),126.)
        values.pop('x0')
        with self.assertRaises(KeyError):checks.linear_score(parameters,values)

    def test_independent_burden_counts_and_csv_mutation(self):
        rows=[dict(serial_number='a',label=1,event_hit=1,first_failure_date='2023-07-10',event_key='a|2023-07-10')]*2
        rows += [dict(serial_number='b',label=None,event_hit=0)]
        facts=dict(eligible_device_days=100,opportunity_events=2,event_start='2023-07-08',event_end='2023-09-24')
        expected=checks.alert_burden(rows,**facts)
        checks.require_equal(ml.burden(rows,**facts),expected)
        self.assertEqual(expected['event_hits'],1)
        self.assertEqual(expected['alerts_per_1000_device_days'],30.)
        bad=dict(expected);bad['known_precision']=0.1
        with self.assertRaises(ValueError):checks.require_equal(bad,expected)

    def test_sparse_zero_nineteen_twenty(self):
        for n in (0,19,20):
            rows = [(0.,1,str(i)) for i in range(n)]+[(0.,0,'negative')]
            prod = ml.calibration_rows(rows)
            other = checks.calibration(rows)
            checks.require_equal(prod,other)
            self.assertEqual(prod['bins'][8]['sparse_positive_devices'], n<20)

    def test_empty_unknown_extremes(self):
        for rows in ([],[(0.,None,'a')],[(1000.,0,'a'),(-1000.,1,'b')]):
            self.assertEqual(ml.calibration_rows(rows),checks.calibration(rows))

    def test_mutations_individually_rejected(self):
        expected=checks.calibration([(0.,1,'a'),(0.,None,'b')])
        for key,value in [('brier',999.),('log_loss',float('nan')),('known_rows',99),('positive_devices',42)]:
            changed=dict(expected);changed[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checks.require_equal(changed,expected)
        for key in expected['bins'][8]:
            changed=dict(expected['bins'][8]);changed[key]='wrong'
            with self.subTest(bin_field=key),self.assertRaises(ValueError):
                checks.require_equal(changed,expected['bins'][8])

    def test_dates_gap_positive_negative_unknown(self):
        daily=[dict(date=f'2023-10-{i:02}',failure=0) for i in range(1,9)]
        self.assertEqual(checks.outcome_from_dates(daily,'2023-10-01'),0)
        self.assertIsNone(checks.outcome_from_dates(daily[:-1],'2023-10-01'))
        daily[-1]['failure']=1
        self.assertEqual(checks.outcome_from_dates(daily[::7],'2023-10-01'),1)
        with self.assertRaises(ValueError):checks.outcome_from_dates(daily,'2023-10-08')

    def test_q3_end(self):
        self.assertEqual(ml.QUARTER_FACTS['q3']['event_end'],'2023-09-24')

class ArtifactTests(unittest.TestCase):
    def test_synthetic_writer_emits_complete_artifact_contract(self):
        import sqlite3
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp);source=root/'source';source.write_text('synthetic')
            inputs={str(source.relative_to(ROOT)):checks.digest(source)}
            db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
            db.execute('CREATE TABLE metrics (model TEXT, denominator INTEGER)')
            db.execute('CREATE TABLE capacity_deltas (model TEXT, denominator INTEGER)')
            rows=[(-2.,1,'a'),(0.,0,'b'),(1.,None,'c')]
            stage=root/f'.staging_fixture_{os.getppid()}'
            with patch.object(ml,'OUT_ROOT',root),patch.object(ml,'preflight',return_value={'input_sha256':inputs}),patch.object(ml,'load_q3_rows',return_value=rows),patch.object(ml,'read_q3_alerts',return_value={m:[] for m in ml.METHODS}),patch.object(ml,'load_q4_rows',return_value=({m:rows for m in ml.METHODS},{m:[] for m in ml.METHODS})),patch.object(ml,'ro_db',return_value=db):
                out=ml.run('fixture',stage)
            summary=json.loads((out/'summary.json').read_text())
            facts=json.loads((out/'input_manifest.json').read_text())
            checks.check_artifacts(out,summary,facts,inputs,ROOT)
            self.assertEqual(summary['resources']['output_bytes'],sum(p.stat().st_size for p in out.iterdir()))
            count=json.loads((out/'CHECKS.json').read_text())['row_counts']
            self.assertEqual(count,{q:{m:3 for m in ml.METHODS} for q in ('q3','q4')})
            self.assertIn('numpy',facts['environment'])

    def test_each_contract_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            (root/'source').write_text('source')
            inputs={'source':checks.digest(root/'source')}
            for name in checks.OUTPUTS: (out/name).write_text('{}')
            (out/'summary.json').write_text('{}')
            summary={'outputs':{n:checks.digest(out/n) for n in checks.OUTPUTS}}
            for alias,name in [('metrics_sha256','metrics.json'),('checks_sha256','CHECKS.json'),('input_manifest_sha256','input_manifest.json')]:
                summary[alias]=summary['outputs'][name]
            facts={'input_sha256':inputs}
            checks.check_artifacts(out,summary,facts,inputs,root)
            for alias in ('outputs','metrics_sha256','checks_sha256','input_manifest_sha256'):
                bad=dict(summary);bad[alias]={} if alias=='outputs' else 'false'
                with self.subTest(alias=alias),self.assertRaises(ValueError):checks.check_artifacts(out,bad,facts,inputs,root)
            for bad in ({},{'source':'forged'}):
                with self.assertRaises(ValueError):checks.check_artifacts(out,summary,{'input_sha256':bad},inputs,root)

class SupervisorTests(unittest.TestCase):
    def test_descendant_ignoring_term_is_removed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp);ready=root/'descendant.json';records=[]
            code='''import os,signal,time,json,sys
pid=os.fork()
if pid==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 with open(sys.argv[1],'w') as f:json.dump({'pid':os.getpid()},f)
 time.sleep(60)
else:
 time.sleep(60)
'''
            began=runner.time.monotonic()
            def budget(started,stage,pgid=None):
                if ready.exists():raise RuntimeError('controlled output limit')
                if runner.time.monotonic()-began>8:raise RuntimeError('fixture handshake timeout')
                return {'members':[]}
            with patch.object(runner,'check_budget',side_effect=budget):
                with self.assertRaisesRegex(RuntimeError,'controlled output limit'):
                    runner.run_phase([sys.executable,'-B','-c',code,str(ready)],began,root,'descendant',records)
            self.assertTrue(ready.exists())
            self.assertEqual(records[0]['cleanup']['state'],'complete')
            self.assertEqual(records[0]['cleanup']['members'],[])

    def test_signals_before_and_after_commit(self):
        script = '''
import sys,os,signal,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import run_frozen_ml_supervised as r
root=Path(sys.argv[2]);when=sys.argv[3]
old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1})
def phase(command,started,stage_root,name,records):
    if name=='compute':
        stage=stage_root/'fixture';stage.mkdir()
        (stage/'summary.json').write_text('{"status":"complete"}')
    elif when=='before':os.kill(os.getpid(),signal.SIGTERM)
original=r.publish
def publish(stage,final):
    original(stage,final)
    os.kill(os.getpid(),signal.SIGTERM)
with patch.object(r,'OUT_ROOT',root),patch.object(r,'check_budget',return_value={}),patch.object(r,'run_phase',side_effect=phase),patch.object(r,'publish',side_effect=publish):
    try:result=r.supervise('fixture')
    except RuntimeError:result={'status':'failed'}
mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
assert signal.SIGUSR1 in mask
signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
assert (root/'fixture'/'summary.json').exists()==(when=='after')
assert result['status']==('pass' if when=='after' else 'failed')
'''
        for when in ('before','after'):
            with self.subTest(when=when),tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
                result=subprocess.run([sys.executable,'-B','-c',script,str(ROOT/'tools'),tmp,when],capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_live_child_cumulative_timeout_is_cleaned(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            records=[]
            original=runner.check_budget
            calls=0
            def budget(started,stage,pgid=None):
                nonlocal calls
                calls+=1
                if calls==1:return {}
                return original(runner.time.monotonic()-1801,stage,pgid)
            with patch.object(runner,'check_budget',side_effect=budget):
                with self.assertRaisesRegex(RuntimeError,'elapsed limit'):
                    runner.run_phase([sys.executable,'-B','-c','import time;time.sleep(60)'],runner.time.monotonic(),Path(tmp),'timeout',records)
            self.assertEqual(records[0]['cleanup']['state'],'complete')
            self.assertIsNotNone(records[0]['returncode'])

    def test_failed_audit_keeps_candidate_without_success_marker(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp)
            def phase(command, started, stage_root, name, records):
                if name=='compute':
                    stage=stage_root/'fixture';stage.mkdir()
                    (stage/'summary.json').write_text('{"status":"complete"}')
                else:
                    raise RuntimeError('audit rejected')
            with patch.object(runner,'OUT_ROOT',root),patch.object(runner,'check_budget',return_value={}),patch.object(runner,'run_phase',side_effect=phase):
                with self.assertRaisesRegex(RuntimeError,'audit rejected'):runner.supervise('fixture')
            stage=root/f'.staging_fixture_{os.getpid()}'/'fixture'
            self.assertFalse((stage/'summary.json').exists())
            self.assertFalse((root/'fixture'/'summary.json').exists())
            self.assertEqual(json.loads((stage/'candidate_summary.failed.json').read_text())['status'],'complete')
            self.assertEqual(json.loads((stage.parent/'failure.json').read_text())['status'],'failed')

    def test_initial_budget_does_not_count_historical_attempts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp);(root/'historical').write_text('retained evidence')
            table={os.getpid():(0,1,1)}
            with patch.object(runner.resource_guard,'_read_process_table',return_value=(table,'mock',None,{})),patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=3*1024**3)),patch.object(runner.resource_guard,'_owned_bytes',side_effect=AssertionError('historical scope read')):
                sample=runner.check_budget(runner.time.monotonic(),root/'.new-stage')
            self.assertEqual(sample['owned_bytes'],0)

    def test_live_child_is_cleaned_when_sampling_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            records=[]
            with patch.object(runner,'check_budget',side_effect=[{},RuntimeError('sample unavailable')]):
                with self.assertRaisesRegex(RuntimeError,'sample unavailable'):
                    runner.run_phase([sys.executable,'-B','-c','import time; time.sleep(60)'],runner.time.monotonic(),Path(tmp),'fixture',records)
            self.assertEqual(records[0]['cleanup']['state'],'complete')
            self.assertEqual(records[0]['cleanup']['members'],[])

    def test_resource_failure_and_budget_boundaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            table={os.getpid():(0,1,1)}
            with patch.object(runner.resource_guard,'_read_process_table',return_value=(table,'mock',None,{})),patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=3*1024**3)):
                self.assertGreater(runner.check_budget(runner.time.monotonic(),Path(tmp))['rss_bytes'],0)
                with patch.object(runner.resource_guard,'_read_process_table',return_value=({},'none','unavailable',{})),self.assertRaises(RuntimeError):runner.check_budget(runner.time.monotonic(),Path(tmp))
                with patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=1536*1024**2)),self.assertRaises(RuntimeError):runner.check_budget(runner.time.monotonic(),Path(tmp))
                with self.assertRaises(RuntimeError):runner.check_budget(runner.time.monotonic()-1801,Path(tmp))
                with patch.object(runner.resource_guard,'_owned_bytes',return_value=runner.MAX_OUTPUT_BYTES+1),self.assertRaises(RuntimeError):runner.check_budget(runner.time.monotonic(),Path(tmp))

    def test_exclusive_publication_and_write_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as tmp:
            root=Path(tmp);stage=root/'stage';stage.mkdir()
            (stage/'metrics.json').write_text('{}');(stage/'summary.json').write_text('{}')
            final=root/'final';runner.publish(stage,final)
            with self.assertRaises(FileExistsError):runner.publish(stage,final)
            failed=root/'failed'
            with patch.object(runner.os,'link',side_effect=OSError('disk failure')),self.assertRaises(OSError):runner.publish(stage,failed)
            self.assertFalse((failed/'summary.json').exists())

    def test_worker_cannot_choose_external_output(self):
        with self.assertRaises(ml.AnalysisError):ml.run('fixture',ROOT/'.tmp')

if __name__=='__main__':unittest.main()
