"""Fixed synthetic R worker; produces a candidate, never a final publication."""
import json
from pathlib import Path
import sys
import os
import time

from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
from pipeline.reproducible.environment import verify_project_modules, ModuleIdentityError
from pipeline.reproducible.runtime_context import validate_context
from pipeline.reproducible.source_snapshot import verify_source_snapshot
from pipeline.reproducible.supervisor import _load_small


def module_records(manifest, code):
    pipeline = verify_project_modules(manifest['files'])
    allowed={item['path']:item['sha256'] for item in manifest['files']}
    tools=[]
    for name,module in sorted(sys.modules.items()):
        if not (name.startswith('tools.') or name=='__main__'): continue
        origin=getattr(module,'__file__',None)
        if not origin or not str(origin).endswith('.py'): continue
        path=Path(origin).resolve()
        if not path.is_relative_to(code): raise ModuleIdentityError(f'tool loaded outside snapshot: {name}')
        relative=path.relative_to(code).as_posix()
        if allowed.get(relative)!=sha256_file(path): raise ModuleIdentityError(f'unbound tool module: {name}')
        tools.append({'module':name,'path':relative,'sha256':allowed[relative]})
    return {'pipeline':pipeline,'tools':tools}


def run_worker(request_path, expected_request_sha256):
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('R snapshot worker requires -I -B')
    request_path=bound_path(request_path,'R worker request',must_exist=True)
    if sha256_file(request_path)!=expected_request_sha256:
        raise ValueError('R worker request SHA mismatch')
    request=_load_small(request_path,'R worker request')
    fields={'schema','mode','project_root','code_root','output','expectation'}
    version=request.get('schema')
    supervised=version in ('r-snapshot-worker-v2','r-snapshot-worker-v3','r-snapshot-worker-v4')
    required=fields|({'handshake'} if supervised else set())|({'calendar'} if version in ('r-snapshot-worker-v3','r-snapshot-worker-v4') else set())
    if version=='r-snapshot-worker-v4': required.add('source_contract')
    if set(request)!=required:
        raise ValueError('R worker request fields mismatch')
    allowed_mode = ('synthetic_bound_zip', 'released_zip', 'synthetic_q1_bound_zip', 'released_q1_zip') if version=='r-snapshot-worker-v4' else ('self_generated_zip_fixture',)
    if version not in ('r-snapshot-worker-v1','r-snapshot-worker-v2','r-snapshot-worker-v3','r-snapshot-worker-v4') or request['mode'] not in allowed_mode:
        raise ValueError('R worker only accepts self-generated ZIP fixture mode')
    calendar=request.get('calendar','short')
    if calendar not in ('short','quarter'): raise ValueError('unknown synthetic calendar')
    code,project=validate_context(expected_code_root=request['code_root'],expected_project_root=request['project_root'])
    if not Path(__file__).resolve().is_relative_to(code): raise ValueError('R worker is outside requested snapshot')
    if not isinstance(request['expectation'],dict) or not request['expectation']:
        raise ValueError('external snapshot expectation required')
    manifest=verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    output=bound_path(request['output'],'R worker output')
    if output.exists() or output.is_relative_to(code): raise ValueError('R worker output unavailable')
    from pipeline.r_validation.zip_chain import run_zip_fixture
    module_records(manifest,code)
    if supervised:
        handshake=request['handshake']
        if set(handshake)!={'ready','go'}: raise ValueError('invalid handshake fields')
        ready=bound_path(handshake['ready'],'worker ready')
        go=bound_path(handshake['go'],'worker go')
        if ready.parent!=request_path.parent or go.parent!=request_path.parent or ready==go:
            raise ValueError('handshake paths must be distinct request siblings')
        identity={'pid':os.getpid(),'pgid':os.getpgrp(),'request_sha256':expected_request_sha256,
                  'snapshot_manifest_sha256':manifest['manifest_sha256']}
        write_json_exclusive(ready,identity)
        deadline=time.monotonic()+30
        while not go.exists():
            if time.monotonic()>=deadline: raise RuntimeError('worker permission timeout')
            time.sleep(.01)
        if _load_small(go,'worker permission')!=identity: raise ValueError('worker permission mismatch')
        if sha256_file(request_path)!=expected_request_sha256: raise ValueError('worker request changed before permission')
        verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    if version=='r-snapshot-worker-v4':
        from pipeline.r_validation.bound_zip import run_bound_contract
        if request['source_contract'].get('profile') != request['mode']: raise ValueError('worker source profile mismatch')
        result=run_bound_contract(output, request['source_contract'])
    else:
        result=run_zip_fixture(output,calendar=calendar)
    if result.get('status')!='pass': raise ValueError('ZIP worker chain did not pass')
    manifest=verify_source_snapshot(code,project_root=project,expectation=request['expectation'])
    modules=module_records(manifest,code)
    if sha256_file(request_path)!=expected_request_sha256: raise ValueError('worker request changed')
    candidate={'schema':'r-worker-candidate-v1','status':'pending_external_verification',
               'request_sha256':expected_request_sha256,'snapshot_manifest_sha256':manifest['manifest_sha256'],
               'chain_result_sha256':sha256_file(output/'zip_chain_result.json'),
               'modules':modules,'calendar':calendar,'publication':'not_authorized'}
    write_json_exclusive(output/'worker_candidate.json',candidate)
    return candidate
