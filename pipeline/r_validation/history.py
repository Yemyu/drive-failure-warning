"""Bind historical identity and failure records into the R scoring panel."""
from contextlib import closing
import datetime as dt
import hashlib
from pathlib import Path
import sqlite3

from tools.run_small_replay import SOURCE_COLUMNS, MODEL


class HistoryError(RuntimeError):
    pass


def coverage_report(counts, *, start, end):
    """Use the previous seven calendar days, including earlier low days."""
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if first > last:
        raise HistoryError('invalid coverage calendar')
    report = {'status': 'pass', 'start': start, 'end': end, 'days': [], 'reason': None}
    consecutive = cumulative = 0
    day = first
    while day <= last:
        keys = [(day-dt.timedelta(days=i)).isoformat() for i in range(7,0,-1)]
        current = day.isoformat()
        required = keys + [current]
        if any(key not in counts or type(counts[key]) is not int or counts[key] < 0 for key in required):
            report.update(status='failed', reason='missing_or_invalid_daily_count', stopped_on=current)
            return report
        baseline = sorted(counts[key] for key in keys)[3]
        if baseline == 0:
            report.update(status='failed', reason='zero_baseline', stopped_on=current)
            return report
        low = counts[current] * 5 < baseline * 4
        consecutive = consecutive + 1 if low else 0
        cumulative += int(low)
        report['days'].append({'date': current, 'observed': counts[current],
                               'previous_7_day_median': baseline, 'low': low,
                               'consecutive_low': consecutive, 'cumulative_low': cumulative})
        if consecutive >= 3 or cumulative >= 10:
            report.update(status='failed', reason='consecutive_low' if consecutive >= 3 else 'cumulative_low', stopped_on=current)
            return report
        day += dt.timedelta(days=1)
    return report


def check_panel_coverage(panel, *, start, end):
    with closing(sqlite3.connect(Path(panel).resolve().as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
        counts = dict(db.execute('SELECT date,COUNT(*) FROM daily WHERE model=? GROUP BY date', (MODEL,)))
    return coverage_report(counts, start=start, end=end)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def merge_history(panel, bindings, *, score_start):
    """Keep all past failure rows and the preceding 13 daily observations.

    Registries are consumed before filtering the model. Input hashes are
    checked both before and after reading. A conflicting or overlapping
    history is rejected; records are never silently deduplicated.
    """
    start = dt.date.fromisoformat(score_start)
    history_start = (start - dt.timedelta(days=13)).isoformat()
    coverage_start = (start - dt.timedelta(days=7)).isoformat()
    if not bindings:
        raise HistoryError('historical panel bindings are required')
    receipts = []
    with closing(sqlite3.connect(panel)) as target:
        target.execute('CREATE TABLE IF NOT EXISTS serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT NOT NULL)')

        def register(serial, model):
            old = target.execute('SELECT model FROM serial_model_registry WHERE serial_number=?', (serial,)).fetchone()
            if old is not None and old[0] != model:
                raise HistoryError(f'cross-source serial/model conflict: {serial}')
            target.execute('INSERT OR IGNORE INTO serial_model_registry VALUES (?,?)', (serial, model))

        for serial, model in target.execute('SELECT DISTINCT serial_number,model FROM daily').fetchall():
            register(serial, model)
        if target.execute('SELECT 1 FROM daily WHERE date<? LIMIT 1', (score_start,)).fetchone():
            raise HistoryError('quarter source overlaps historical dates')
        for role, binding in sorted(bindings.items()):
            path = Path(binding['path']).resolve()
            expected = binding['sha256']
            if file_sha(path) != expected:
                raise HistoryError(f'historical SHA mismatch: {role}')
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)) as source:
                bounds = source.execute('SELECT MIN(date),MAX(date) FROM daily').fetchone()
                if bounds[1] is not None and bounds[1] >= score_start:
                    raise HistoryError('historical panel includes scoring-period or future records')
                registry_count = 0
                for serial, model in source.execute('SELECT serial_number,model FROM serial_model_registry'):
                    register(serial, model)
                    registry_count += 1
                for serial, model in source.execute('SELECT DISTINCT serial_number,model FROM daily'):
                    declared = source.execute('SELECT model FROM serial_model_registry WHERE serial_number=?', (serial,)).fetchone()
                    if declared is None or declared[0] != model:
                        raise HistoryError(f'historical daily/registry mismatch: {serial}')
                query = f"SELECT {','.join(SOURCE_COLUMNS)} FROM daily WHERE model=? AND (date>=? OR failure=1) ORDER BY date,serial_number"
                added = 0
                for row in source.execute(query, (MODEL, history_start)):
                    target.execute(f"INSERT INTO daily ({','.join(SOURCE_COLUMNS)}) VALUES ({','.join('?' for _ in SOURCE_COLUMNS)})", tuple(row))
                    added += 1
            if file_sha(path) != expected:
                raise HistoryError(f'historical input changed while reading: {role}')
            receipts.append({'role': role, 'path': str(path), 'sha256': expected,
                             'registry_rows': registry_count, 'rows_added': added,
                             'source_date_min': bounds[0], 'source_date_max': bounds[1]})
        coverage = dict(target.execute('SELECT date,COUNT(*) FROM daily WHERE model=? AND date>=? AND date<? GROUP BY date',
                                       (MODEL, coverage_start, score_start)))
        expected_dates = [(start-dt.timedelta(days=i)).isoformat() for i in range(7,0,-1)]
        if any(coverage.get(day, 0) <= 0 for day in expected_dates):
            raise HistoryError('missing historical coverage seed')
        target.execute('CREATE INDEX IF NOT EXISTS daily_serial_date ON daily(serial_number,date)')
        target.commit()
    return {'inputs': receipts, 'history_start': history_start, 'score_start': score_start,
            'coverage_seed': {day: coverage[day] for day in expected_dates}}
