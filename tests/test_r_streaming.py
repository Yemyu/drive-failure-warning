"""Regression evidence for bounded daily scoring and reduced feature storage."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation.zip_chain import run_zip_fixture
from pipeline.r_validation.stage_runner import StageRunnerError
from tools import run_small_replay as replay
from pipeline.replay_selection import select_alerts

ROOT = Path(__file__).resolve().parents[1]


class StreamingTests(unittest.TestCase):
    def test_each_day_is_selected_before_next_day_is_scored(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as temp:
            output = Path(temp) / 'fixture'
            calls = []
            original = replay._score_day
            def score(view, day, *args, **kwargs):
                if calls:
                    self.assertEqual(calls[-2:], ['current_lr', 'smart_nonzero'])
                calls.append(day)
                yield from original(view, day, *args, **kwargs)
            def choose(rows, method, *args, **kwargs):
                calls.append(method)
                return select_alerts(rows, method, *args, **kwargs)
            with patch.object(replay, '_score_day', side_effect=score), patch(
                    'pipeline.replay_selection.select_alerts', side_effect=choose):
                run_zip_fixture(output)
            self.assertEqual(len(calls), 14 * 3)

    def test_compact_features_preserve_scores_selections_and_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as temp:
            root = Path(temp)
            output = root / 'compact'
            run_zip_fixture(output)
            panel = output / 'prepared/panel.sqlite'
            original = output / 'evaluation/selection.sqlite'
            full = root / 'full.sqlite'
            spec = {'score_start': '2023-10-01', 'score_end': '2023-10-14',
                    'event_start': '2023-10-08', 'event_end': '2023-10-15',
                    'outcome_cutoff': '2023-10-21', 'horizon_days': 7,
                    'allow_smart_decreases': True}
            access = []
            with closing(sqlite3.connect(panel)) as db:
                db.row_factory = sqlite3.Row
                replay._score_and_select(replay._asof_reader(db, access), [], replay._load_model(), full, access, spec=spec)
                evaluated = replay._evaluate(full, db, replay.sha256(full), {}, access, spec=spec)
            with closing(sqlite3.connect(original)) as compact, closing(sqlite3.connect(full)) as expanded:
                for table in ('scores', 'daily', 'selections', 'access_log'):
                    self.assertEqual(sorted(compact.execute(f'SELECT * FROM {table}').fetchall()),
                                     sorted(expanded.execute(f'SELECT * FROM {table}').fetchall()))
                for day, serial, payload in compact.execute('SELECT * FROM features'):
                    saved = json.loads(expanded.execute('SELECT payload_json FROM features WHERE decision_date=? AND serial_number=?', (day, serial)).fetchone()[0])
                    reduced = json.loads(payload)
                    self.assertLess(len(reduced), len(saved))
                    self.assertEqual(reduced, {key: saved[key] for key in reduced})
            saved = json.loads((output / 'evaluation/evaluation.json').read_text())
            self.assertEqual(evaluated['methods'], saved['methods'])

    def test_invalid_scale_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as temp:
            for count in (True, 0, 2001, 10.5):
                output = Path(temp) / str(count)
                with self.assertRaisesRegex(StageRunnerError, 'device count'):
                    run_zip_fixture(output, device_count=count)
                self.assertFalse(output.exists())
