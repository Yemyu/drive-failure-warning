"""Project an independently audited source candidate and inherit bound history."""
from contextlib import closing
import datetime as dt
import json
from pathlib import Path
import sqlite3

from pipeline.r_validation.history import merge_history, file_sha, HistoryError
from pipeline.r_validation.zip_audit import audit_zip_candidate
from tools.run_small_replay import SOURCE_COLUMNS, MODEL

from pipeline.reproducible.runtime_context import project_root
ROOT=project_root()


def verify_merged_panel(merged, candidate_panel, historical_inputs, *, start):
    """Independent row/registry comparison without using the merge predicate."""
    fields=','.join(SOURCE_COLUMNS)
    first=(dt.date.fromisoformat(start)-dt.timedelta(days=13)).isoformat()
    sources=[Path(candidate_panel).resolve()]+[Path(b['path']).resolve() for b in historical_inputs.values()]
    before={str(p):file_sha(p) for p in sources+[Path(merged).resolve()]}
    identities={}; keys=set()
    with closing(sqlite3.connect(Path(merged).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)) as target:
        for index,path in enumerate(sources):
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as source:
                for serial,model in source.execute('SELECT serial_number,model FROM serial_model_registry'):
                    if serial in identities and identities[serial]!=model: raise HistoryError('merged identity conflict')
                    identities[serial]=model
                for values in source.execute(f'SELECT {fields} FROM daily'):
                    day,serial,model,_capacity,failure,*_=values
                    if index:
                        if day>=start: raise HistoryError('historical future row in merge audit')
                        if model!=MODEL or (day<first and failure!=1): continue
                    key=(day,serial)
                    if key in keys: raise HistoryError('overlap in independent merge inputs')
                    keys.add(key)
                    actual=target.execute(f'SELECT {fields} FROM daily WHERE date=? AND serial_number=?',key).fetchone()
                    if actual!=values: raise HistoryError(f'merged row differs: {key}')
        if target.execute('SELECT COUNT(*) FROM daily').fetchone()[0]!=len(keys):
            raise HistoryError('merged row count differs')
        if dict(target.execute('SELECT serial_number,model FROM serial_model_registry'))!=identities:
            raise HistoryError('merged full-source registry differs')
    for role,binding in historical_inputs.items():
        if before[str(Path(binding['path']).resolve())]!=binding['sha256']: raise HistoryError(f'historical binding changed: {role}')
    if any(file_sha(path)!=sha for path,sha in before.items()): raise HistoryError('merge audit input changed')
    return {'status':'pass','rows':len(keys),'full_source_identities':len(identities),'input_sha256':before}


def prepare_zip_scoring_panel(archive, candidate, output, *, expected_sha256, start, end, prefix, historical_inputs):
    archive,candidate,output=Path(archive).resolve(),Path(candidate).resolve(),Path(output).resolve()
    if not all(p.is_relative_to(ROOT) for p in (archive,candidate,output)):
        raise HistoryError('ZIP preparation paths must be inside project')
    if output.exists(): raise HistoryError('refusing existing preparation directory')
    # Recompute now; a saved pass report cannot authorise a modified candidate.
    audit=audit_zip_candidate(archive,candidate,expected_sha256=expected_sha256,start=start,end=end,prefix=prefix,historical_inputs=historical_inputs)
    output.mkdir(parents=True)
    try:
        panel=output/'panel.sqlite'; original=candidate/'candidate.sqlite'
        columns=','.join(SOURCE_COLUMNS)
        with closing(sqlite3.connect(original.as_uri()+'?mode=ro&immutable=1',uri=True)) as source, closing(sqlite3.connect(panel)) as target:
            definitions=','.join(f"{name} {'TEXT' if name in ('date','serial_number','model') else 'INTEGER'}" for name in SOURCE_COLUMNS)
            target.execute(f'CREATE TABLE daily({definitions},PRIMARY KEY(date,serial_number))')
            target.executemany(f"INSERT INTO daily VALUES ({','.join('?' for _ in SOURCE_COLUMNS)})",source.execute(f'SELECT {columns} FROM daily'))
            target.execute('CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY,model TEXT NOT NULL)')
            target.executemany('INSERT INTO serial_model_registry VALUES (?,?)',source.execute('SELECT serial_number,model FROM serial_model_registry'))
            target.commit()
        history=merge_history(panel,historical_inputs,score_start=start)
        merged_audit=verify_merged_panel(panel,original,historical_inputs,start=start)
        # Pin every source to the exact bytes verified before projection.
        if any(file_sha(path)!=sha for path,sha in audit['input_sha256'].items()):
            raise HistoryError('verified candidate or source changed during preparation')
        result={'status':'prepared_candidate','panel_sha256':file_sha(panel),'source_audit':audit,
                'history_merge':history,'merge_audit':merged_audit,
                'coverage':json.loads((candidate/'coverage.json').read_text()),
                'publication':'not_authorized','scoring_cli_connected':False}
        (output/'prepared_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps({'status':'failed','reason':str(exc)},indent=2)+'\n')
        raise
