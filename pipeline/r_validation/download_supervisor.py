"""Resource-supervised single download with a snapshot-bound worker."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceLimits
from pipeline.reproducible.runtime_context import project_root, code_root, validate_context
from pipeline.reproducible.source_snapshot import create_source_snapshot, verify_source_snapshot
from pipeline.reproducible.supervisor import _bounded_cleanup, _load_small
from pipeline.r_validation.acquisition import authorize, authorize_q1, fetch_once, validate_download_records
from pipeline.r_validation.budget_scope import cumulative_roots
from pipeline.r_validation.snapshot_worker import module_records
from pipeline.r_validation.synthetic_publication import _publish


def _fixture_source(source):
    url = urlsplit(source['url'])
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
            or source['bz_file_id'] != 'r0-local-fixture'
            or type(source['content_length']) is not int or not 0 < source['content_length'] <= 1024*1024):
        raise ValueError('download fixture must be a bounded loopback source')


def run_download_worker(request_path, request_sha):
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('download worker requires -I -B')
    request_path=bound_path(request_path,'download request',must_exist=True)
    if sha256_file(request_path)!=request_sha: raise ValueError('download request changed')
    request=_load_small(request_path,'download request')
    if set(request)!={'schema','profile','project_root','code_root','expectation','output','source','lock','seconds','max_bytes'} or request['schema']!='r-download-request-v1':
        raise ValueError('invalid download request')
    code,root=validate_context(expected_code_root=request['code_root'],expected_project_root=request['project_root'])
    manifest=verify_source_snapshot(code,project_root=root,expectation=request['expectation'])
    output=bound_path(request['output'],'download output')
    if not output.is_dir(): raise ValueError('download output directory missing')
    if request_path.parent!=output or not Path(__file__).resolve().is_relative_to(code):
        raise ValueError('download worker context differs')
    if request['profile'] in {'released_download', 'released_q1_download'}:
        authorize_fn = authorize_q1 if request['profile'] == 'released_q1_download' else authorize
        config,lock,release=authorize_fn(request['lock'])
        if request['profile'] == 'released_q1_download':
            timeout = config['budget']['download_timeout_seconds']
            max_network_bytes = config['budget']['download_max_body_bytes']
        else:
            budget=config['budget']['download']
            timeout = budget['timeout_seconds']
            max_network_bytes = budget['max_network_body_bytes']
        if (request['source']!=config['source_zip'] or request['seconds']!=timeout
            or request['max_bytes']!=max_network_bytes
            or output != root/config['paths']['raw']): raise ValueError('download contract differs from protocol')
    elif request['profile']=='local_http_fixture':
        _fixture_source(request['source'])
        if request['lock'] is not None or not 0 < request['seconds'] <= 30 or request['max_bytes']>1024*1024:
            raise ValueError('invalid download fixture controls')
    else: raise ValueError('invalid download profile')
    module_records(manifest,code)
    identity={'pid':os.getpid(),'pgid':os.getpgrp(),'request_sha256':request_sha,
              'snapshot_manifest_sha256':manifest['manifest_sha256']}
    write_json_exclusive(output/'ready.json',identity)
    deadline=time.monotonic()+30
    while not (output/'go.json').exists():
        if time.monotonic()>=deadline: raise RuntimeError('download worker permission timeout')
        time.sleep(.01)
    if _load_small(output/'go.json','download permission')!=identity: raise ValueError('download permission differs')
    if sha256_file(request_path)!=request_sha: raise ValueError('download request changed before body')
    verify_source_snapshot(code,project_root=root,expectation=request['expectation'])
    result=fetch_once(request['source'],output,max_bytes=request['max_bytes'],seconds=request['seconds'])
    manifest=verify_source_snapshot(code,project_root=root,expectation=request['expectation'])
    write_json_exclusive(output/'worker_identity.json',{'request_sha256':request_sha,
        'snapshot_manifest_sha256':manifest['manifest_sha256'],'modules':module_records(manifest,code)})
    return result


def supervise_download(*, lock_path=None, fixture_source=None, output=None, limits=None,
                       profile='released_download'):
    root=project_root()
    if code_root()!=root: raise ValueError('download supervisor requires checkout')
    real=fixture_source is None
    if real:
        if profile not in {'released_download', 'released_q1_download'}:
            raise ValueError('unknown released download profile')
        authorize_fn = authorize_q1 if profile == 'released_q1_download' else authorize
        config,lock,release=authorize_fn(lock_path)
        if output is not None or limits is not None: raise ValueError('real download budgets and path are fixed')
        source=config['source_zip']; budget=config['budget']
        if profile == 'released_q1_download':
            timeout = budget['download_timeout_seconds']
            max_bytes = budget['download_max_body_bytes']
        else:
            download=budget['download']
            timeout = download['timeout_seconds']
            max_bytes = download['max_network_body_bytes']
        output=root/config['paths']['raw']
        limits=ResourceLimits(max_rss_bytes=budget['process_rss_bytes'],max_output_bytes=budget['retention_bytes'],
            initial_free_bytes=budget['min_free_at_start_bytes'],min_free_bytes=budget['min_free_running_bytes'],
            max_elapsed_seconds=timeout,poll_seconds=budget['sampling_seconds'])
    else:
        _fixture_source(fixture_source)
        if lock_path is not None: raise ValueError('fixture cannot carry real authorization')
        source=fixture_source; limits=limits or ResourceLimits(max_elapsed_seconds=30)
        if limits.max_elapsed_seconds>30: raise ValueError('fixture deadline exceeds 30 seconds')
        max_bytes=1024*1024
    output=bound_path(output,'download output'); output.mkdir(parents=True,exist_ok=False)
    roots=cumulative_roots(output)
    guard=ResourceGuard(root,roots[0],limits,extra_outputs=roots[1:],reject_output_symlinks=True)
    guard.register_pid(os.getpid())
    cancel=threading.Event(); handlers={}; process=None; cleanup={'state':'not_started'}
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig]=signal.getsignal(sig); signal.signal(sig,lambda *_:cancel.set())
        guard.start()
        if guard.check_or_raise()['free_bytes']<limits.initial_free_bytes: raise RuntimeError('initial free space below budget')
        snapshot=output/'snapshot'; created=create_source_snapshot(snapshot,project_root=root)
        request={'schema':'r-download-request-v1','profile':profile if real else 'local_http_fixture',
            'project_root':str(root),'code_root':str(snapshot),'expectation':created['expectation'],
            'output':str(output),'source':source,'lock':str(Path(lock_path).resolve()) if real else None,
            'seconds':limits.max_elapsed_seconds,'max_bytes':max_bytes}
        path=output/'request.json'; write_json_exclusive(path,request); request_sha=sha256_file(path)
        env={k:v for k,v in os.environ.items() if not k.startswith('REPRO_') and k!='PYTHONPATH'}
        env.update(REPRO_PROJECT_ROOT=str(root),REPRO_CODE_ROOT=str(snapshot),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        guard.check_or_raise()
        if cancel.is_set(): raise RuntimeError('download cancelled before launch')
        with (output/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-I','-B',str(snapshot/'tools/run_r_validation.py'),
                'download-worker','--request',str(path),'--request-sha256',request_sha],cwd=root,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            guard.register_group(process.pid);guard.expect_member(process.pid);permitted=False
            while process.poll() is None:
                guard.check_or_raise()
                if cancel.is_set(): raise RuntimeError('download cancelled')
                if not permitted and (output/'ready.json').exists():
                    identity={'pid':process.pid,'pgid':process.pid,'request_sha256':request_sha,
                              'snapshot_manifest_sha256':created['manifest_sha256']}
                    if _load_small(output/'ready.json','download ready')!=identity: raise RuntimeError('download identity differs')
                    guard.check_or_raise()
                    if guard.expected_members_observed.get(process.pid):
                        if cancel.is_set(): raise RuntimeError('download cancelled before permission')
                        write_json_exclusive(output/'go.json',identity);permitted=True
                time.sleep(.05)
        guard.begin_cleanup();cleanup=_bounded_cleanup(process.pid,process,known_members=guard.observed_group_pids)
        if cleanup['state']!='complete': raise RuntimeError('download group cleanup unconfirmed')
        guard.release_group()
        if process.returncode!=0 or not permitted: raise RuntimeError(f'download worker failed: {process.returncode}')
        candidate=_load_small(output/'download_candidate.json','download candidate')
        worker_identity=_load_small(output/'worker_identity.json','download worker identity')
        if (worker_identity.get('request_sha256')!=request_sha
            or worker_identity.get('snapshot_manifest_sha256')!=created['manifest_sha256']):
            raise RuntimeError('download worker identity binding differs')
        validate_download_records(output,source,candidate.get('archive_sha256'))
        partial=output/'source.partial'
        if (candidate.get('status')!='download_candidate' or candidate.get('bytes')!=source['content_length']
            or partial.stat().st_size!=source['content_length'] or sha256_file(partial)!=candidate.get('archive_sha256')):
            raise RuntimeError('download candidate bytes differ')
        archive_name = 'data_Q1_2024.zip' if profile == 'released_q1_download' else 'data_Q4_2023.zip'
        archive=output/(archive_name if real else 'fixture.zip')
        os.link(partial,archive);partial.unlink()
        write_json_exclusive(output/'supervision.json',{'cleanup':cleanup,'resource_peak':guard.peak_snapshot,
                                                       'reader':guard.reader_facts(),'scope':guard.sampling_scope()})
        artifacts={str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()}
        def validate():
            validate_download_records(output,source,candidate['archive_sha256'])
            verify_source_snapshot(snapshot,project_root=root,expectation=created['expectation'])
            if sha256_file(path)!=request_sha or sha256_file(archive)!=candidate['archive_sha256']:
                raise RuntimeError('download changed before publication')
            if real:
                authorize_fn = authorize_q1 if profile == 'released_q1_download' else authorize
                current_config,current_lock,current_release=authorize_fn(lock_path)
                if (current_config!=config or current_lock!=lock or current_release!=release):
                    raise RuntimeError('download authorization changed')
        return _publish(output,artifacts=artifacts,guard=guard,cancel=cancel,validate=validate,
            name='download_complete' if real else 'synthetic_download_complete',metadata={
                'schema':'r-download-completion-v1','status':'download_complete' if real else 'synthetic_download_complete',
                'verification':'supervised_download_bytes_and_identity','source_object_id':source['bz_file_id'],
                'url':source['url'],'bytes':candidate['bytes'],'archive_sha256':candidate['archive_sha256'],
                'archive':str(archive),'cleanup':cleanup,'real_data_release':real,
                'release_digest':release['release_digest'] if real else None,
                'r0_lock_digest':lock['lock_digest'] if real else None})
    except BaseException as exc:
        if process is not None and cleanup.get('state')!='complete':
            guard.begin_cleanup();cleanup=_bounded_cleanup(process.pid,process,known_members=guard.observed_group_pids)
            if cleanup.get('state')=='complete':guard.release_group()
        write_json_exclusive(output/'failure.json',{'status':'failed','reason':str(exc),'cleanup':cleanup,
            'resource_peak':guard.peak_snapshot,'reader':guard.reader_facts(),'automatic_retry':False})
        raise
    finally:
        guard.stop()
        for sig,handler in handlers.items():signal.signal(sig,handler)
