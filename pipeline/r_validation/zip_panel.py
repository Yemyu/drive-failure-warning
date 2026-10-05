"""Build a candidate quarter panel; no final publication or scoring permission."""
from contextlib import closing
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

from pipeline.panel import PanelWriter
from pipeline.r_validation.history import file_sha, coverage_report, HistoryError
from pipeline.r_validation.zip_source import RArchiveSource
from tools.run_small_replay import MODEL

from pipeline.reproducible.runtime_context import project_root
ROOT = project_root()


def build_zip_candidate(archive, output, *, expected_sha256, start, end, prefix, historical_inputs):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if not output.is_relative_to(ROOT) or not archive.is_relative_to(ROOT):
        raise HistoryError('ZIP candidate paths must be inside the project')
    if not historical_inputs:
        raise HistoryError('bound historical panels required')
    output.mkdir(parents=True, exist_ok=False)
    writer = None
    try:
        seed_start = (dt.date.fromisoformat(start)-dt.timedelta(days=7)).isoformat()
        seed = {}
        identities = {}
        bindings = []
        for role, binding in sorted(historical_inputs.items()):
            path = Path(binding['path']).resolve()
            if not path.is_relative_to(ROOT) or file_sha(path) != binding['sha256']:
                raise HistoryError('historical path or SHA mismatch')
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)) as db:
                latest = db.execute('SELECT MAX(date) FROM daily').fetchone()[0]
                if latest is not None and latest >= start:
                    raise HistoryError('history overlaps current source calendar')
                for serial, model in db.execute('SELECT serial_number,model FROM serial_model_registry'):
                    if serial in identities and identities[serial] != model:
                        raise HistoryError('historical registry conflict')
                    identities[serial] = model
                for day, count in db.execute('SELECT date,COUNT(*) FROM daily WHERE model=? AND date>=? GROUP BY date', (MODEL,seed_start)):
                    if day in seed:
                        raise HistoryError('overlapping coverage seed')
                    seed[day] = count
            bindings.append({'role':role,'path':str(path),'sha256':binding['sha256']})
        expected_seed = [(dt.date.fromisoformat(start)-dt.timedelta(days=i)).isoformat() for i in range(7,0,-1)]
        if any(seed.get(day,0) <= 0 for day in expected_seed):
            raise HistoryError('missing historical coverage seed')
        writer = PanelWriter(output/'candidate.sqlite')
        writer.set_metadata({'build_status':'candidate_building','cross_quarter_identity':'pending'})
        source = RArchiveSource(archive,expected_sha256=expected_sha256,start=start,end=end,prefix=prefix)
        counts = dict(seed)
        with source:
            for name in source.names:
                with source.rows(name) as rows:
                    columns = list(source.columns)
                    schema_sha = hashlib.sha256(json.dumps(columns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                    writer.schema_declarations = {prefix:{'count':len(columns),'sha256':schema_sha}}

                    def checked_rows():
                        for index, row in enumerate(rows,2):
                            previous = identities.get(row['serial_number'])
                            if previous is not None and previous != row['model']:
                                raise HistoryError('cross-quarter full-source serial/model conflict')
                            yield index, [row[column] for column in columns]

                    day = Path(name).stem
                    writer.append_member({'name':name}, ROOT, MODEL,
                                         source_data=(day,'pending',columns,checked_rows(),{}),strict_smart=True)
                fact = source.facts[-1]
                with writer.connection:
                    writer.connection.execute('UPDATE daily SET source_sha256=? WHERE date=?',(fact['sha256'],day))
                    writer.connection.execute('UPDATE member_counts SET source_sha256=? WHERE date=?',(fact['sha256'],day))
                counts[day] = fact['selected_rows']
                coverage = coverage_report(counts,start=start,end=day)
                (output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
                if coverage['status'] != 'pass':
                    raise HistoryError(f"coverage stopped: {coverage['reason']} on {day}")
        for binding in bindings:
            if file_sha(binding['path']) != binding['sha256']:
                raise HistoryError('historical source changed during ZIP build')
        writer.set_metadata({'build_status':'candidate_ready','cross_quarter_identity':'verified_against_bound_registries'})
        writer.close(); writer = None
        panel = output/'candidate.sqlite'
        if any(Path(str(panel)+suffix).exists() for suffix in ('-wal','-shm')):
            raise HistoryError('candidate database sidecar remains')
        receipt = {'status':'candidate_ready','scope':'quarter_source_panel_only',
                   'panel_sha256':file_sha(panel),'source':source.receipt(),'historical_inputs':bindings,
                   'coverage':coverage,'historical_daily_rows_merged':False,
                   'independent_audit':'pending','publication':'not_authorized'}
        (output/'candidate_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt
    except BaseException as exc:
        if writer is not None:
            writer.connection.rollback()
            writer.close()
        (output/'failure.json').write_text(json.dumps({'status':'failed','type':type(exc).__name__,'reason':str(exc)},indent=2)+'\n')
        raise
