"""Diagnostic decreases across all source days, independent of alert eligibility."""
from contextlib import ExitStack, closing
import heapq
import json
from pathlib import Path
import sqlite3

from pipeline.r_validation.history import HistoryError, file_sha
from tools.run_small_replay import MODEL

FIELDS = ('smart_5_raw', 'smart_9_raw', 'smart_187_raw')


def _sources(candidate, history, start, end, stack):
    paths = [Path(candidate).resolve()] + [Path(b['path']).resolve() for b in history.values()]
    if len(set(paths)) != len(paths):
        raise HistoryError('duplicate decrease ledger sources')
    databases = []
    hashes = {}
    for index, path in enumerate(paths):
        hashes[str(path)] = file_sha(path)
        db = stack.enter_context(closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)))
        bounds = db.execute('SELECT MIN(date),MAX(date) FROM daily').fetchone()
        if index == 0:
            if bounds[0] != start or bounds[1] != end:
                raise HistoryError('decrease ledger source calendar differs')
        elif bounds[1] is not None and bounds[1] >= start:
            raise HistoryError('decrease ledger historical future row')
        databases.append(db)
    for binding in history.values():
        if hashes[str(Path(binding['path']).resolve())] != binding['sha256']:
            raise HistoryError('decrease ledger history SHA mismatch')
    return databases, hashes


def _unchanged(hashes):
    if any(file_sha(path) != sha for path, sha in hashes.items()):
        raise HistoryError('decrease ledger source changed')


def build_decrease_ledger(candidate, history, output, *, start, end):
    """Compare each non-NULL value with its latest prior non-NULL observation.

    Past quarters seed state, including observations before the 13-day bridge.
    Missing values and absent days do not reset state. No failure, history-count,
    score-date, signal or cooldown filter is applied to the main model.
    """
    output = Path(output)
    with output.open('xb'):
        pass
    with ExitStack() as stack:
        databases, hashes = _sources(candidate, history, start, end, stack)
        target = stack.enter_context(closing(sqlite3.connect(output)))
        target.execute('CREATE TABLE decreases(date TEXT,serial_number TEXT,field TEXT,previous_date TEXT,previous_value INTEGER,current_value INTEGER,PRIMARY KEY(date,serial_number,field))')
        target.execute('CREATE TABLE days(date TEXT PRIMARY KEY,observations INTEGER,nonnull_5 INTEGER,nonnull_9 INTEGER,nonnull_187 INTEGER,decreases INTEGER)')
        target.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        cursors = [db.execute('SELECT date,serial_number,'+','.join(FIELDS)+' FROM daily WHERE model=? ORDER BY date,serial_number', (MODEL,)) for db in databases]
        state = {}; previous_key = None; counts = {}; total = 0
        for day, serial, *values in heapq.merge(*cursors, key=lambda row: row[:2]):
            key = (day, serial)
            if key == previous_key:
                raise HistoryError('overlapping decrease ledger source keys')
            previous_key = key
            current = start <= day <= end
            if current:
                daily = counts.setdefault(day, [0, 0, 0, 0, 0]); daily[0] += 1
            for index, (field, value) in enumerate(zip(FIELDS, values)):
                if value is None:
                    continue
                old = state.get((serial, field))
                if current:
                    daily[index+1] += 1
                    if old is not None and value < old[1]:
                        target.execute('INSERT INTO decreases VALUES (?,?,?,?,?,?)', (day, serial, field, old[0], old[1], value))
                        daily[4] += 1; total += 1
                state[(serial, field)] = (day, value)
        target.executemany('INSERT INTO days VALUES (?,?,?,?,?,?)', ((day, *values) for day, values in sorted(counts.items())))
        metadata = {'schema':'r-decrease-ledger-v1', 'model':MODEL, 'start':start, 'end':end,
                    'comparison':'latest_prior_nonnull', 'eligibility_filter':False, 'input_sha256':hashes}
        target.executemany('INSERT INTO metadata VALUES (?,?)', ((key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()))
        target.commit()
        _unchanged(hashes)
    return {'status':'recorded', 'decreases':total, 'days':len(counts), 'sha256':file_sha(output), **metadata}


def audit_decrease_ledger(candidate, history, ledger, *, start, end):
    """Recompute by field and serial, separate from the builder's daily state."""
    with ExitStack() as stack:
        databases, hashes = _sources(candidate, history, start, end, stack)
        ledger = Path(ledger).resolve(); before = file_sha(ledger)
        saved = stack.enter_context(closing(sqlite3.connect(ledger.as_uri()+'?mode=ro&immutable=1', uri=True)))
        expected_metadata = {'schema':'r-decrease-ledger-v1', 'model':MODEL, 'start':start, 'end':end,
                             'comparison':'latest_prior_nonnull', 'eligibility_filter':False, 'input_sha256':hashes}
        actual_metadata = {k:json.loads(v) for k,v in saved.execute('SELECT key,value FROM metadata')}
        if actual_metadata != expected_metadata:
            raise HistoryError('decrease ledger metadata differs')
        total = 0; daily_decreases = {}
        for field in FIELDS:
            cursors = [db.execute(f'SELECT serial_number,date,{field} FROM daily WHERE model=? AND {field} IS NOT NULL ORDER BY serial_number,date', (MODEL,)) for db in databases]
            actual = iter(saved.execute('SELECT serial_number,date,previous_date,previous_value,current_value FROM decreases WHERE field=? ORDER BY serial_number,date', (field,)))
            previous = None
            for serial, day, value in heapq.merge(*cursors, key=lambda row: row[:2]):
                if previous is not None and previous[:2] == (serial, day):
                    raise HistoryError('overlapping decrease audit keys')
                if previous is not None and previous[0] == serial and start <= day <= end and value < previous[2]:
                    expected = (serial, day, previous[1], previous[2], value)
                    if next(actual, None) != expected:
                        raise HistoryError('decrease ledger row differs')
                    total += 1; daily_decreases[day] = daily_decreases.get(day, 0)+1
                previous = (serial, day, value)
            if next(actual, None) is not None:
                raise HistoryError('unexpected decrease ledger row')
        if saved.execute('SELECT COUNT(*) FROM decreases').fetchone()[0] != total:
            raise HistoryError('unexpected decrease ledger field')
        actual_days = list(saved.execute('SELECT * FROM days ORDER BY date'))
        expected_days = [(*row, daily_decreases.get(row[0], 0)) for row in databases[0].execute(
            'SELECT date,COUNT(*),COUNT(smart_5_raw),COUNT(smart_9_raw),COUNT(smart_187_raw) FROM daily WHERE model=? GROUP BY date ORDER BY date', (MODEL,))]
        if actual_days != expected_days:
            raise HistoryError('decrease ledger daily coverage differs')
        _unchanged(hashes)
        if file_sha(ledger) != before:
            raise HistoryError('decrease ledger changed during audit')
    return {'status':'pass', 'decreases':total, 'days':len(expected_days), 'ledger_sha256':before,
            'comparison':'independent_serial_field_scan', 'input_sha256':hashes}
