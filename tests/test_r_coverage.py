"""Coverage uses past raw observations, never future labels or filtered dates."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation.history import coverage_report
from pipeline.r_validation.stage_runner import run_stage, StageRunnerError
from pipeline.r_validation.cli_fixture import _write_csv
from tests.test_r_history import row

ROOT=Path(__file__).resolve().parents[1]


class CoverageTests(unittest.TestCase):
    def counts(self,n=30):
        return {(dt.date(2023,9,24)+dt.timedelta(days=i)).isoformat():100 for i in range(7+n)}

    def test_exact_eighty_percent_is_not_low(self):
        counts=self.counts(); counts['2023-10-01']=80
        report=coverage_report(counts,start='2023-10-01',end='2023-10-01')
        self.assertEqual(report['status'],'pass')
        self.assertFalse(report['days'][0]['low'])

    def test_third_consecutive_low_stops(self):
        counts=self.counts()
        for day in range(1,4): counts[f'2023-10-{day:02d}']=79
        report=coverage_report(counts,start='2023-10-01',end='2023-10-30')
        self.assertEqual((report['reason'],report['stopped_on']),('consecutive_low','2023-10-03'))
        self.assertEqual(len(report['days']),3)

    def test_tenth_nonconsecutive_low_stops(self):
        counts=self.counts()
        for day in range(1,29,3): counts[f'2023-10-{day:02d}']=79
        report=coverage_report(counts,start='2023-10-01',end='2023-10-30')
        self.assertEqual((report['reason'],report['stopped_on']),('cumulative_low','2023-10-28'))
        self.assertEqual(report['days'][-1]['cumulative_low'],10)

    def test_missing_seed_day_or_zero_baseline_is_not_pass(self):
        for mode in ('missing_seed','missing_current','zero'):
            counts=self.counts()
            if mode=='missing_seed': del counts['2023-09-24']
            elif mode=='missing_current': del counts['2023-10-01']
            else:
                for day in range(24,31): counts[f'2023-09-{day}']=0
            self.assertEqual(coverage_report(counts,start='2023-10-01',end='2023-10-02')['status'],'failed')

    def test_prior_window_excludes_current_and_future(self):
        counts=self.counts(); counts['2023-10-01']=80
        before=coverage_report(counts,start='2023-10-01',end='2023-10-01')
        counts['2023-10-02']=1
        self.assertEqual(before,coverage_report(counts,start='2023-10-01',end='2023-10-01'))
        self.assertEqual(before['days'][0]['previous_7_day_median'],100)

    def test_runner_stops_before_score_and_keeps_failure_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory); source=output/'source.csv'
            rows=[row((dt.date(2023,9,18)+dt.timedelta(days=i)).isoformat(),f'disk-{j}')
                  for i in range(16) for j in range(10 if i < 13 else 7)]
            _write_csv(source,rows)
            with patch('tools.run_small_replay._score_and_select',side_effect=AssertionError('must not score')):
                with self.assertRaisesRegex(StageRunnerError,'coverage stopped: consecutive_low'):
                    run_stage(output/'attempt',root=ROOT,profile='synthetic',source_csv=source,
                              spec={'score_start':'2023-10-01','score_end':'2023-10-03',
                                    'outcome_cutoff':'2023-10-03','coverage_check':True})
            chain=output/'attempt/cli_chain'
            self.assertEqual(json.loads((chain/'coverage.json').read_text())['stopped_on'],'2023-10-03')
            self.assertFalse((chain/'selection.sqlite').exists())
            self.assertFalse((chain/'chain_results.json').exists())


if __name__=='__main__': unittest.main()
