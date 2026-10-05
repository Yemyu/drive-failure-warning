"""Supervise ZIP computation and independent verification from bound sources."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceLimits
from pipeline.reproducible.runtime_context import project_root, code_root
from pipeline.reproducible.source_snapshot import create_source_snapshot, verify_source_snapshot
from pipeline.reproducible.supervisor import _bounded_cleanup, _load_small
from pipeline.r_validation.post_worker_audit import inventory
from pipeline.r_validation.synthetic_publication import publish_synthetic
from pipeline.r_validation.budget_scope import cumulative_roots


def supervise_zip_fixture(output, *, limits=None, worker_timeout_seconds=30, calendar='short', bound_contract=None):
    if calendar not in ('short','quarter'): raise ValueError('unknown synthetic calendar')
    if worker_timeout_seconds<=0: raise ValueError('worker timeout must be positive')
    if bound_contract is not None and calendar != 'quarter': raise ValueError('bound ZIP requires full-quarter calendar')
    root=project_root(); output=bound_path(output,'R supervised output')
    if code_root()!=root: raise RuntimeError('outer fixture supervisor requires project checkout')
    output.mkdir(parents=True,exist_ok=False)
    limits=limits or ResourceLimits(max_elapsed_seconds=30)
    scope_roots=cumulative_roots(output)
    guard=ResourceGuard(root,scope_roots[0],limits,extra_outputs=scope_roots[1:],reject_output_symlinks=True)
    guard.register_pid(os.getpid())
    process=None; cleanup={'state':'not_started'}; old_handlers={}; cancel=threading.Event()
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,lambda *_:cancel.set())
        guard.start()
        initial=guard.check_or_raise()
        if initial['free_bytes']<limits.initial_free_bytes: raise RuntimeError('initial free space below budget')
        if bound_contract is not None:
            from pipeline.r_validation.bound_zip import validate_bound_zip
            validate_bound_zip(bound_contract)
        snapshot=output/'snapshot'
        created=create_source_snapshot(snapshot,project_root=code_root())
        # Normal calling checkout uses the data root; cross-host/snapshot
        # nesting is not an authorised mode of this narrow fixture wrapper.
        guard.check_or_raise()
        if cancel.is_set(): raise RuntimeError('cancelled before worker launch')
        request={'schema':'r-snapshot-worker-v3','mode':'self_generated_zip_fixture','calendar':calendar,
                 'project_root':str(root),'code_root':str(snapshot),'output':str(output/'work'),
                 'expectation':created['expectation'],
                 'handshake':{'ready':str(output/'ready.json'),'go':str(output/'go.json')}}
        if bound_contract is not None:
            request.update(schema='r-snapshot-worker-v4', mode=bound_contract['profile'], source_contract=bound_contract)
        path=output/'request.json'; write_json_exclusive(path,request); request_sha=sha256_file(path)
        env={k:v for k,v in os.environ.items() if not k.startswith('REPRO_') and k!='PYTHONPATH'}
        env.update(REPRO_PROJECT_ROOT=str(root),REPRO_CODE_ROOT=str(snapshot),
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with (output/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-I','-B',str(snapshot/'tools/run_r_validation.py'),
                                      'snapshot-worker','--request',str(path),'--request-sha256',request_sha],
                                     cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            guard.register_group(process.pid); guard.expect_member(process.pid)
            deadline=time.monotonic()+worker_timeout_seconds
            permitted=False
            while process.poll() is None:
                if cancel.is_set(): raise RuntimeError('cancelled while worker running')
                if time.monotonic()>=deadline: raise RuntimeError('R worker timeout')
                guard.check_or_raise()
                if not permitted and (output/'ready.json').exists():
                    ready=_load_small(output/'ready.json','worker ready')
                    expected={'pid':process.pid,'pgid':process.pid,'request_sha256':request_sha,
                              'snapshot_manifest_sha256':created['manifest_sha256']}
                    if ready!=expected: raise RuntimeError('worker ready identity mismatch')
                    # Ready may have appeared after the loop's resource sample.
                    # Recheck before making the permission decision.
                    guard.check_or_raise()
                    if guard.expected_members_observed.get(process.pid):
                        if cancel.is_set(): raise RuntimeError('cancelled before worker permission')
                        write_json_exclusive(output/'go.json',expected)
                        permitted=True
                time.sleep(0.05)
        guard.begin_cleanup()
        cleanup=_bounded_cleanup(process.pid,process,known_members=guard.observed_group_pids)
        if cleanup['state']!='complete': raise RuntimeError('work group cleanup unconfirmed')
        guard.release_group()
        if process.returncode!=0: raise RuntimeError(f'R worker exited {process.returncode}')
        if not permitted: raise RuntimeError('worker exited without permission')
        if not guard.membership_proven_sample: raise RuntimeError('worker membership never observed')
        guard.check_or_raise()
        if cancel.is_set(): raise RuntimeError('cancelled after worker exit')
        verify_source_snapshot(snapshot,project_root=root,expectation=created['expectation'])
        candidate=_load_small(output/'work/worker_candidate.json','R worker candidate')
        if (candidate.get('calendar')!=calendar or candidate.get('status')!='pending_external_verification' or candidate.get('request_sha256')!=request_sha or
            candidate.get('snapshot_manifest_sha256')!=created['manifest_sha256'] or
            candidate.get('chain_result_sha256')!=sha256_file(output/'work/zip_chain_result.json')):
            raise RuntimeError('worker candidate binding mismatch')
        if sha256_file(path)!=request_sha: raise RuntimeError('worker request changed')
        worker_record={'cleanup':cleanup,'reader':guard.reader_facts(),'returncode':process.returncode}
        process=None; cleanup={'state':'not_started'}
        guard.prepare_next_group()
        guard.check_or_raise()
        inputs=inventory(output/'work')
        verification_request={'schema':'r-synthetic-verifier-v2','calendar':calendar,'project_root':str(root),
            'code_root':str(snapshot),'expectation':created['expectation'],'attempt':str(output/'work'),
            'inventory':inputs,'model_sha256':sha256_file(root/'examples/small_replay/current_lr.json'),
            'output':str(output/'verification.json'),'ready':str(output/'verification_ready.json'),
            'go':str(output/'verification_go.json')}
        if bound_contract is not None:
            verification_request.update(schema='r-bound-verifier-v1', source_contract=bound_contract,
                                        model_sha256=bound_contract['model']['sha256'])
        verification_path=output/'verification_request.json'
        write_json_exclusive(verification_path,verification_request)
        verification_sha=sha256_file(verification_path)
        if cancel.is_set(): raise RuntimeError('cancelled before verifier launch')
        with (output/'verifier.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-I','-B',str(snapshot/'tools/run_r_validation.py'),
                'snapshot-verifier','--request',str(verification_path),'--request-sha256',verification_sha],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            guard.register_group(process.pid); guard.expect_member(process.pid)
            deadline=time.monotonic()+worker_timeout_seconds
            permitted=False
            while process.poll() is None:
                if cancel.is_set(): raise RuntimeError('cancelled while verifier running')
                if time.monotonic()>=deadline: raise RuntimeError('R verifier timeout')
                guard.check_or_raise()
                if not permitted and (output/'verification_ready.json').exists():
                    expected={'pid':process.pid,'pgid':process.pid,'request_sha256':verification_sha,
                              'snapshot_manifest_sha256':created['manifest_sha256']}
                    if _load_small(output/'verification_ready.json','verifier ready')!=expected:
                        raise RuntimeError('verifier ready identity mismatch')
                    guard.check_or_raise()
                    if guard.expected_members_observed.get(process.pid):
                        if cancel.is_set(): raise RuntimeError('cancelled before verifier permission')
                        write_json_exclusive(output/'verification_go.json',expected)
                        permitted=True
                time.sleep(.05)
        guard.begin_cleanup()
        cleanup=_bounded_cleanup(process.pid,process,known_members=guard.observed_group_pids)
        if cleanup['state']!='complete': raise RuntimeError('verifier cleanup unconfirmed')
        guard.release_group()
        if process.returncode!=0: raise RuntimeError(f'R verifier exited {process.returncode}')
        if not permitted or not guard.membership_proven_sample: raise RuntimeError('verifier permission or observation missing')
        verification=_load_small(output/'verification.json','verification result')
        if (verification.get('calendar')!=calendar or verification.get('status')!='pass' or verification.get('request_sha256')!=verification_sha
            or verification.get('snapshot_manifest_sha256')!=created['manifest_sha256']
            or verification.get('input_sha256')!=inputs or inventory(output/'work')!=inputs
            or sha256_file(verification_path)!=verification_sha):
            raise RuntimeError('verification result binding mismatch')
        verify_source_snapshot(snapshot,project_root=root,expectation=created['expectation'])
        guard.check_or_raise()
        if cancel.is_set(): raise RuntimeError('cancelled before supervision record')
        result={'status':'supervised_candidate','calendar':calendar,'worker_returncode':worker_record['returncode'],
                'verifier_returncode':process.returncode,'cleanup':cleanup,
                'resource_peak':guard.peak_snapshot,'reader':guard.reader_facts(),'scope':guard.sampling_scope(),
                'output_budget_roots':[str(p) for p in scope_roots],
                'output_budget_scope':'all_retained_research_outputs_and_local_records',
                'worker_candidate_sha256':sha256_file(output/'work/worker_candidate.json'),
                'handshake':{'ready_sha256':sha256_file(output/'ready.json'),'go_sha256':sha256_file(output/'go.json')},
                'worker_stage':worker_record,'verification_sha256':sha256_file(output/'verification.json'),
                'independent_post_worker_verification':'pass','publication':'not_authorized'}
        write_json_exclusive(output/'supervision.json',result)
        # The completion marker binds the candidate, audits, controls and code.
        # This small synthetic path rehashes them in the supervised parent.
        if inventory(output/'work')!=inputs: raise RuntimeError('worker files changed before publication')
        artifacts={str(p.relative_to(output)):p for p in output.rglob('*') if p.is_file()}
        def validate_publication():
            if (inventory(output/'work')!=inputs or sha256_file(path)!=request_sha
                or sha256_file(verification_path)!=verification_sha
                or sha256_file(output/'verification.json')!=result['verification_sha256']):
                raise RuntimeError('verified inputs changed before publication')
            verify_source_snapshot(snapshot,project_root=root,expectation=created['expectation'])
            if bound_contract is not None: validate_bound_zip(bound_contract)
        if bound_contract is None:
            completion=publish_synthetic(output,artifacts=artifacts,guard=guard,cancel=cancel,validate=validate_publication,calendar=calendar)
            marker='synthetic_complete.json'
        else:
            from pipeline.r_validation.synthetic_publication import publish_bound
            evaluation=_load_small(output/'work/evaluation/evaluation.json','bound evaluation')
            completion=publish_bound(output,artifacts=artifacts,guard=guard,cancel=cancel,validate=validate_publication,
                                     contract=bound_contract,evaluation=evaluation)
            marker='r_complete.json' if bound_contract['profile'] in {'released_zip', 'released_q1_zip'} else 'bound_synthetic_complete.json'
        result={**result,'completion_status':completion['status'],
                'completion_manifest':str(output/marker)}
        if bound_contract is None:
            result['synthetic_completion']=completion['status']
        else:
            result.update(status=completion['status'], publication='complete' if bound_contract['profile'] in {'released_zip', 'released_q1_zip'} else 'synthetic_only')
        return result
    except BaseException as exc:
        if process is not None and cleanup.get('state')!='complete':
            guard.begin_cleanup()
            cleanup=_bounded_cleanup(process.pid,process,known_members=guard.observed_group_pids)
            if cleanup.get('state')=='complete': guard.release_group()
        write_json_exclusive(output/'failure.json',{'status':'failed','reason':str(exc),'cleanup':cleanup,
                                                   'resource_peak':guard.peak_snapshot,'reader':guard.reader_facts()})
        raise
    finally:
        guard.stop()
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
