"""Daily consumers cannot advance their fixed information date."""
from contextlib import closing
import unittest
from unittest.mock import patch

from tools import run_small_replay as replay


class DecisionDayTests(unittest.TestCase):
    def source(self):
        rows, _ = replay._parse_source(replay.SOURCE_CSV)
        future = dict(rows[-1], serial_number='future-only', date='2023-02-04')
        return replay._source_connection(rows + [future], [])

    def test_future_request_rejected_before_sql(self):
        with closing(self.source()) as connection:
            view = replay._asof_reader(connection, []).for_day('2023-01-15')
            queries = []
            connection.set_trace_callback(queries.append)
            with self.assertRaisesRegex(replay.ReplayError, 'bound decision day'):
                view.rows_until('demo-A', '2023-01-16')
            self.assertEqual(queries, [])
            for name in ('execute', 'for_day', '_connection', 'connection'):
                self.assertFalse(hasattr(view, name))

    def test_roster_excludes_future_device(self):
        with closing(self.source()) as connection:
            factory = replay._asof_reader(connection, [])
            early = factory.for_day('2023-01-15')
            late = factory.for_day('2023-02-04')
            self.assertNotIn('future-only', early.serials())
            self.assertIn('future-only', late.serials())
            self.assertNotIn('future-only', early.serials())
            self.assertLessEqual(max(r['date'] for r in early.rows_until('demo-A','2023-01-15')), '2023-01-15')

    def test_actual_daily_scorer_cannot_advance_date(self):
        with closing(self.source()) as connection:
            view = replay._asof_reader(connection, []).for_day('2023-01-15')
            with self.assertRaisesRegex(replay.ReplayError, 'bound decision day'):
                list(replay._score_day(view,'2023-01-16',replay._load_model(),allow_smart_decreases=False))

    def test_coordinator_uses_bound_views_without_full_roster(self):
        from pathlib import Path
        import tempfile
        root = Path(__file__).resolve().parents[1]
        with closing(self.source()) as connection, tempfile.TemporaryDirectory(dir=root/'.tmp') as directory:
            access = []
            factory = replay._asof_reader(connection, access)
            received = []
            original = replay._score_day
            def check(view, day, *args, **kwargs):
                self.assertIsInstance(view, replay.DecisionDayReader)
                received.append(day)
                return original(view, day, *args, **kwargs)
            with patch.object(factory, 'serials', side_effect=AssertionError('full-period roster read')), patch.object(replay, '_score_day', side_effect=check):
                replay._score_and_select(factory, [], replay._load_model(), Path(directory)/'selection.sqlite', access)
            self.assertEqual(len(received), 14)


if __name__ == '__main__':
    unittest.main()
