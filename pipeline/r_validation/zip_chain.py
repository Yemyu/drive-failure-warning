"""Self-generated ZIP fixture through the existing scoring/evaluation functions.

This entry accepts no external source and cannot open a real-quarter ZIP.
"""
from contextlib import closing
import csv
import datetime as dt
import io
import json
from pathlib import Path
import sqlite3
import zipfile

from pipeline.r_validation.cli_fixture import _write_panel
from pipeline.r_validation.history import file_sha
from pipeline.r_validation.stage_runner import StageRunnerError
from tools import run_small_replay as replay

from pipeline.reproducible.runtime_context import project_root
ROOT=project_root()


def build_zip_fixture_source(output, *, calendar='short', device_count=10):
    if type(device_count) is not int or not 2 <= device_count <= 2000:
        raise StageRunnerError('synthetic device count must be between 2 and 2000')
    if calendar not in ('short','quarter'): raise StageRunnerError('unknown synthetic calendar')
    raw_end,score_end,event_end,days=(('2023-10-21','2023-10-14','2023-10-15',21) if calendar=='short'
                                  else ('2023-12-31','2023-12-24','2023-12-25',92))
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT): raise StageRunnerError('fixture output must be in project')
    output.mkdir(parents=True,exist_ok=False)
    try:
        def row(day,serial,failure=0):
            return {'date':day,'serial_number':serial,'model':replay.MODEL,
                    'capacity_bytes':4000787030016,'failure':failure,
                    **{f'smart_{field}_raw':100 if field==9 else 1 for field in replay.SMART_FIELDS}}
        history=output/'synthetic_history.sqlite'
        rows=[row((dt.date(2023,9,18)+dt.timedelta(days=i)).isoformat(),f'zip-fixture-{j}')
              for i in range(13) for j in range(device_count)]
        rows.append(row('2023-02-01','zip-fixture-1',1))
        _write_panel(history,rows)
        with closing(sqlite3.connect(history)) as db:
            db.execute('CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY,model TEXT)')
            db.executemany('INSERT INTO serial_model_registry VALUES (?,?)',[(f'zip-fixture-{j}',replay.MODEL) for j in range(device_count)])
            db.commit()
        bindings={'history':{'path':str(history),'sha256':file_sha(history)}}
        archive=output/'synthetic.zip'
        columns=list(replay.SOURCE_COLUMNS)+[f'extra_{i}' for i in range(193-len(replay.SOURCE_COLUMNS))]
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('fixture/.DS_Store',b'synthetic Finder metadata, not CSV')
            z.writestr('__MACOSX/fixture/._.DS_Store',b'synthetic resource fork')
            for offset in range(days):
                day=(dt.date(2023,10,1)+dt.timedelta(days=offset)).isoformat()
                text=io.StringIO(newline=''); writer=csv.DictWriter(text,fieldnames=columns)
                writer.writeheader()
                for j in range(device_count): writer.writerow(row(day,f'zip-fixture-{j}',int(j==0 and offset==14)))
                z.writestr(f'fixture/{day}.csv',text.getvalue())
        source_args=dict(expected_sha256=file_sha(archive),start='2023-10-01',end=raw_end,prefix='fixture',historical_inputs=bindings)
        from pipeline.r_validation.zip_execution import run_bound_zip_chain
        model_source=ROOT/'examples/small_replay/current_lr.json'
        spec={'score_start':'2023-10-01','score_end':score_end,'event_start':'2023-10-08',
              'event_end':event_end,'outcome_cutoff':raw_end,'horizon_days':7,'allow_smart_decreases':True}
        return dict(archive=archive,source_args=source_args,model_source=model_source,
                    expected_model_sha256=file_sha(model_source),spec=spec)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps({'status':'failed','reason':str(exc)},indent=2)+'\n')
        raise


def run_zip_fixture(output, *, calendar='short', device_count=10):
    args = build_zip_fixture_source(output, calendar=calendar, device_count=device_count)
    try:
        from pipeline.r_validation.zip_execution import run_bound_zip_chain
        return run_bound_zip_chain(output, **args)
    except BaseException as exc:
        (Path(output)/'failure.json').write_text(json.dumps({'status':'failed','reason':str(exc)},indent=2)+'\n')
        raise


def run_bound_fixture(output):
    """A complete external-input CLI exercise that generates its own source."""
    from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT): raise StageRunnerError('fixture output must be in project')
    output.mkdir(parents=True,exist_ok=False)
    args=build_zip_fixture_source(output/'source',calendar='quarter')
    source=args['source_args']
    contract={'schema':'r-bound-zip-v1','profile':'synthetic_bound_zip',
        'source':{'archive':{'path':str(args['archive']),'sha256':source['expected_sha256']},
                  'receipt':None,'prefix':source['prefix'],'start':source['start'],'end':source['end']},
        'history':source['historical_inputs'],
        'model':{'path':str(args['model_source']),'sha256':args['expected_model_sha256']},
        'spec':args['spec'],'authorization':None}
    return supervise_zip_fixture(output/'supervised',calendar='quarter',bound_contract=contract)
