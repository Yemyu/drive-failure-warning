"""Fixed synthetic audit consumer executing inside the bound source snapshot."""
import os
from pathlib import Path
import sys
import time

from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
from pipeline.reproducible.runtime_context import validate_context
from pipeline.reproducible.source_snapshot import verify_source_snapshot
from pipeline.reproducible.supervisor import _load_small
from pipeline.r_validation.snapshot_worker import module_records
from pipeline.r_validation.post_worker_audit import audit_worker_output


def run_verifier(request_path, expected_sha256):
    if not sys.flags.isolated or not sys.dont_write_bytecode: raise ValueError('verifier requires -I -B')
    path=bound_path(request_path,'verification request',must_exist=True)
    if sha256_file(path)!=expected_sha256: raise ValueError('verification request SHA mismatch')
    request=_load_small(path,'verification request')
    fields={'schema','project_root','code_root','expectation','attempt','inventory','model_sha256','output','ready','go'}
    if request.get('schema') in ('r-synthetic-verifier-v2','r-bound-verifier-v1'): fields.add('calendar')
    if request.get('schema')=='r-bound-verifier-v1': fields.add('source_contract')
    if set(request)!=fields:
        raise ValueError('verification request fields mismatch')
    if request['schema'] not in ('r-synthetic-verifier-v1','r-synthetic-verifier-v2','r-bound-verifier-v1'): raise ValueError('unknown verification schema')
    calendar=request.get('calendar','short')
    if calendar not in ('short','quarter'): raise ValueError('unknown synthetic calendar')
    code,project=validate_context(expected_code_root=request['code_root'],expected_project_root=request['project_root'])
    if not Path(__file__).resolve().is_relative_to(code): raise ValueError('verifier outside bound snapshot')
    manifest=verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    module_records(manifest,code)
    ready=bound_path(request['ready'],'verifier ready'); go=bound_path(request['go'],'verifier go')
    target=bound_path(request['output'],'verification output'); attempt=bound_path(request['attempt'],'verification input')
    if any(p.parent!=path.parent for p in (ready,go,target)) or len({ready,go,target,path})!=4:
        raise ValueError('verification control paths must be distinct siblings')
    if target.exists() or target.is_relative_to(attempt): raise ValueError('verification output unavailable')
    identity={'pid':os.getpid(),'pgid':os.getpgrp(),'request_sha256':expected_sha256,
              'snapshot_manifest_sha256':manifest['manifest_sha256']}
    write_json_exclusive(ready,identity)
    deadline=time.monotonic()+30
    while not go.exists():
        if time.monotonic()>=deadline: raise RuntimeError('verifier permission timeout')
        time.sleep(.01)
    if _load_small(go,'verifier permission')!=identity: raise ValueError('verifier permission mismatch')
    if sha256_file(path)!=expected_sha256: raise ValueError('verification request changed before audit')
    verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    result=audit_worker_output(attempt,request['inventory'],expected_model_sha256=request['model_sha256'],calendar=calendar,bound_contract=request.get('source_contract'))
    manifest=verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    result.update(request_sha256=expected_sha256,snapshot_manifest_sha256=manifest['manifest_sha256'],
                  modules=module_records(manifest,code))
    if sha256_file(path)!=expected_sha256: raise ValueError('verification request changed during audit')
    write_json_exclusive(target,result)
    return result
