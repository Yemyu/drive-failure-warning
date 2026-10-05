"""Read-only ZIP/candidate comparison, independent of the source and panel writers."""
from contextlib import closing
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import zipfile


class ZipAuditError(ValueError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''): h.update(block)
    return h.hexdigest()


def integer(value):
    if value.strip()=='': return None
    if re.fullmatch(r'[+-]?\d+',value.strip()) is None:
        raise ZipAuditError('non-integer source value')
    return int(value)


def audit_zip_candidate(archive, candidate, *, expected_sha256, start, end, prefix, historical_inputs):
    """Recompute source rows, member facts, identity and daily coverage.

    The expected archive and history bindings must come from the caller's lock,
    not be discovered from the candidate receipt. No files are modified.
    """
    archive,candidate=Path(archive).resolve(),Path(candidate).resolve()
    receipt_path=candidate/'candidate_receipt.json'; panel=candidate/'candidate.sqlite'
    coverage_path=candidate/'coverage.json'
    paths=[archive,receipt_path,panel,coverage_path]+[Path(b['path']).resolve() for b in historical_inputs.values()]
    before={str(path):digest(path) for path in paths}
    if before[str(archive)]!=expected_sha256: raise ZipAuditError('archive SHA mismatch')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='candidate_ready' or receipt.get('panel_sha256')!=before[str(panel)]:
        raise ZipAuditError('candidate status or hash mismatch')
    lo,hi=dt.date.fromisoformat(start),dt.date.fromisoformat(end)
    if lo>hi or not historical_inputs: raise ZipAuditError('invalid calendar or missing history')
    expected_names=[f'{prefix}/{(lo+dt.timedelta(days=i)).isoformat()}.csv' for i in range((hi-lo).days+1)]
    history_identity={}; counts={}; bound=[]
    for role,binding in sorted(historical_inputs.items()):
        path=Path(binding['path']).resolve()
        if before[str(path)]!=binding['sha256']: raise ZipAuditError('historical SHA mismatch')
        bound.append({'role':role,'path':str(path),'sha256':binding['sha256']})
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
            registry=dict(db.execute('SELECT serial_number,model FROM serial_model_registry'))
            for serial,model in registry.items():
                if serial in history_identity and history_identity[serial]!=model: raise ZipAuditError('history identity conflict')
                history_identity[serial]=model
            for serial,model in db.execute('SELECT DISTINCT serial_number,model FROM daily'):
                if registry.get(serial)!=model: raise ZipAuditError('historical daily/registry mismatch')
            for day,count in db.execute("SELECT date,COUNT(*) FROM daily WHERE model='ST4000DM000' GROUP BY date"):
                if day>=start: raise ZipAuditError('history overlaps source')
                if (lo-dt.timedelta(days=7)).isoformat()<=day:
                    if day in counts: raise ZipAuditError('overlapping historical seed')
                    counts[day]=count
    if receipt.get('historical_inputs')!=bound: raise ZipAuditError('history receipt mismatch')
    if any(counts.get((lo-dt.timedelta(days=i)).isoformat(),0)<=0 for i in range(1,8)):
        raise ZipAuditError('missing coverage seed')
    raw_fields=[f'smart_{i}_raw' for i in (5,9,187,188,197,198)]
    basic=['date','serial_number','model','capacity_bytes','failure']+raw_fields
    compare=basic+['capacity_clean_bytes','capacity_missing']+[f'smart_{i}_missing' for i in (5,9,187,188,197,198)]+['source_member','source_sha256','source_row']
    identity={}; facts=[]; headers=None; selected_total=0; coverage_days=[]; cumulative=consecutive=0
    with closing(sqlite3.connect(panel.as_uri()+'?mode=ro&immutable=1',uri=True)) as db, zipfile.ZipFile(archive) as z:
        infos=z.infolist()
        data=[i for i in infos if not i.is_dir() and not i.filename.startswith('__MACOSX/')
              and i.filename != prefix+'/.DS_Store']
        ignored=[{'name':i.filename,'expanded_bytes':i.file_size,'compressed_bytes':i.compress_size,
                  'crc32':i.CRC,'reason':'directory' if i.is_dir() else 'macos_metadata'}
                 for i in infos if i not in data]
        if receipt.get('source',{}).get('ignored_members') != ignored:
            raise ZipAuditError('ignored ZIP metadata inventory differs')
        if len({i.filename for i in infos})!=len(infos) or {i.filename for i in data}!=set(expected_names):
            raise ZipAuditError('ZIP calendar mismatch')
        if sum(i.file_size for i in infos)>12*1024**3: raise ZipAuditError('expanded total exceeds budget')
        if any(i.file_size>256*1024**2 for i in infos): raise ZipAuditError('member expansion budget exceeded')
        for name in expected_names:
            info=z.getinfo(name); day=Path(name).stem
            if info.file_size>256*1024**2 or info.compress_type!=8 or info.flag_bits&1:
                raise ZipAuditError('unsupported ZIP member')
            h=hashlib.sha256(); size=0
            with z.open(info) as stream:
                for block in iter(lambda:stream.read(65536),b''):
                    size+=len(block)
                    if size>info.file_size: raise ZipAuditError('expanded size mismatch')
                    h.update(block)
            if size!=info.file_size: raise ZipAuditError('expanded size mismatch')
            member_sha=h.hexdigest(); seen=set(); selected=failures=source_rows=0
            with z.open(info) as stream, io.TextIOWrapper(stream,encoding='utf-8-sig',newline='') as text:
                reader=csv.reader(text,strict=True); columns=next(reader,None)
                if columns is None or len(columns)!=len(set(columns)) or not set(basic)<=set(columns):
                    raise ZipAuditError('invalid source schema')
                if headers is not None and headers!=columns: raise ZipAuditError('source schema changed')
                headers=columns
                schema_sha=hashlib.sha256(json.dumps(columns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                for line,values in enumerate(reader,2):
                    if len(values)!=len(columns): raise ZipAuditError('row width mismatch')
                    row=dict(zip(columns,values)); serial=row['serial_number']; model=row['model']
                    if row['date']!=day or not serial or not model or serial in seen: raise ZipAuditError('source key mismatch')
                    seen.add(serial); source_rows+=1
                    if serial in history_identity and history_identity[serial]!=model: raise ZipAuditError('cross-quarter identity mismatch')
                    if serial in identity and identity[serial][0]!=model: raise ZipAuditError('current identity conflict')
                    if serial not in identity: identity[serial]=[model,day,day,name,'full_source_member']
                    identity[serial][2]=day
                    failure=integer(row['failure']); capacity=integer(row['capacity_bytes'])
                    if failure not in (0,1) or (capacity is not None and capacity < -1): raise ZipAuditError('invalid source failure/capacity')
                    if model!='ST4000DM000': continue
                    smart=[integer(row[f]) for f in raw_fields]
                    if any(v is not None and v<0 for v in smart): raise ZipAuditError('negative SMART')
                    expected=(day,serial,model,capacity,failure,*smart,
                              None if capacity in (None,-1) else capacity,int(capacity in (None,-1)),
                              *(int(v is None) for v in smart),name,member_sha,line)
                    actual=db.execute(f"SELECT {','.join(compare)} FROM daily WHERE date=? AND serial_number=?",(day,serial)).fetchone()
                    if actual!=expected: raise ZipAuditError(f'panel row mismatch: {day}/{serial}')
                    selected+=1; failures+=failure
            member=db.execute('SELECT source_member,source_sha256,source_rows,selected_rows,failure_rows,schema_columns,schema_sha256,schema_json FROM member_counts WHERE date=?',(day,)).fetchone()
            expected_member=(name,member_sha,source_rows,selected,failures,len(columns),schema_sha,json.dumps(columns,ensure_ascii=False,separators=(',',':')))
            if member!=expected_member: raise ZipAuditError('member count/schema mismatch')
            facts.append({'name':name,'date':day,'source_rows':source_rows,'selected_rows':selected,
                          'expanded_bytes':size,'compressed_bytes':info.compress_size,'crc32':info.CRC,'sha256':member_sha})
            selected_total+=selected; counts[day]=selected
            date=dt.date.fromisoformat(day)
            median=sorted(counts[(date-dt.timedelta(days=i)).isoformat()] for i in range(1,8))[3]
            if median<=0: raise ZipAuditError('zero coverage baseline')
            low=selected*5<median*4; consecutive=consecutive+1 if low else 0; cumulative+=int(low)
            coverage_days.append({'date':day,'observed':selected,'previous_7_day_median':median,'low':low,
                                  'consecutive_low':consecutive,'cumulative_low':cumulative})
            if consecutive>=3 or cumulative>=10: raise ZipAuditError('candidate crossed coverage stop')
        if db.execute('SELECT COUNT(*) FROM daily').fetchone()[0]!=selected_total: raise ZipAuditError('extra panel rows')
        if db.execute('SELECT COUNT(*) FROM member_counts').fetchone()[0]!=len(facts): raise ZipAuditError('extra member rows')
        actual_identity={r[0]:list(r[1:]) for r in db.execute('SELECT serial_number,model,first_date,last_date,first_source_member,source_scope FROM serial_model_registry')}
        if actual_identity!=identity: raise ZipAuditError('full-source registry mismatch')
    expected_source={'status':'verified_source','scope':'bytes_csv_structure_and_full_source_identity',
                     'archive_sha256':expected_sha256,'members':facts,'schema_columns':headers,
                     'schema_sha256':schema_sha,'full_source_serial_count':len(identity)}
    coverage={'status':'pass','start':start,'end':end,'days':coverage_days,'reason':None}
    expected_source['ignored_members']=ignored
    if receipt.get('source')!=expected_source: raise ZipAuditError('source receipt mismatch')
    if receipt.get('coverage')!=coverage or json.loads(coverage_path.read_text())!=coverage:
        raise ZipAuditError('coverage receipt mismatch')
    if any(digest(path)!=before[str(path)] for path in paths): raise ZipAuditError('input changed during audit')
    return {'status':'pass','scope':'source_panel_identity_and_coverage',
            'members':len(facts),'selected_rows':selected_total,'full_source_identities':len(identity),
            'input_sha256':before,'historical_daily_rows_merged':False,'publication':'not_authorized'}
