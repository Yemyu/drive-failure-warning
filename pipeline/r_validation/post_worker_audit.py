"""Reopen bound worker outputs and recompute all three audit layers."""
from pathlib import Path
from contextlib import closing
import sqlite3

from pipeline.reproducible.artifacts import bound_path, sha256_file
from pipeline.reproducible.supervisor import _load_small
from pipeline.r_validation.zip_audit import audit_zip_candidate
from pipeline.r_validation.zip_history import verify_merged_panel
from pipeline.r_validation.independent_audit import audit_selection
from pipeline.r_validation.decrease_ledger import audit_decrease_ledger
from tools import run_small_replay as replay


def inventory(directory):
    directory=bound_path(directory,'synthetic worker directory')
    if not directory.is_dir(): raise ValueError('missing synthetic worker directory')
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink(): raise ValueError('symlink in worker output')
        if path.is_file(): result[path.relative_to(directory).as_posix()]=sha256_file(path)
    if not result: raise ValueError('empty worker inventory')
    return result


def audit_worker_output(directory, expected_inventory, *, expected_model_sha256, calendar='short', bound_contract=None):
    """Verify a fixed calendar and externally supplied source/model bindings."""
    if calendar not in ('short','quarter'): raise ValueError('unknown synthetic calendar')
    score_end,event_end,cutoff=(('2023-10-14','2023-10-15','2023-10-21') if calendar=='short'
                              else ('2023-12-24','2023-12-25','2023-12-31'))
    bound = None
    if bound_contract is not None:
        from pipeline.r_validation.bound_zip import validate_bound_zip
        bound = validate_bound_zip(bound_contract)
        score_end = bound['spec']['score_end']
        event_end = bound['spec']['event_end']
        cutoff = bound['source_args']['end']
        score_start = bound['spec']['score_start']
        event_start = bound['spec']['event_start']
    else:
        score_start, event_start = '2023-10-01', '2023-10-08'
    directory=bound_path(directory,'synthetic worker directory')
    if not directory.is_dir(): raise ValueError('missing synthetic worker directory')
    if inventory(directory)!=expected_inventory: raise ValueError('worker inventory changed before audit')
    chain=_load_small(directory/'zip_chain_result.json','ZIP chain result')
    profile = 'self_generated_zip_fixture' if bound is None else bound_contract['profile']
    if chain.get('profile') != profile: raise ValueError('worker source profile mismatch')
    if bound is None:
        history=directory/'synthetic_history.sqlite'
        archive=directory/'synthetic.zip'
        bindings={'history':{'path':str(history),'sha256':expected_inventory['synthetic_history.sqlite']}}
        source_args=dict(expected_sha256=expected_inventory['synthetic.zip'],start='2023-10-01',
                         end=cutoff,prefix='fixture',historical_inputs=bindings)
    else:
        archive=bound['archive']; source_args=bound['source_args']; bindings=source_args['historical_inputs']
        if (bound['spec']['score_end'] != score_end or source_args['end'] != cutoff
                or bound['expected_model_sha256'] != expected_model_sha256):
            raise ValueError('bound source and verifier contract differ')
    panel=directory/'prepared/panel.sqlite'
    selection=directory/'evaluation/selection.sqlite'
    model_path=directory/'evaluation/current_lr.json'
    seal=_load_small(directory/'evaluation/selection_seal.json','selection seal')
    if seal!={'schema':'r0-fixture-seal-v2','selection_sha256':sha256_file(selection),'selection_closed':True}:
        raise ValueError('selection seal mismatch')
    with closing(sqlite3.connect(selection.as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        metadata=dict(db.execute('SELECT key,value FROM metadata'))
    expected={'phase':'complete','start':score_start,'end':score_end,'outcome_cutoff':cutoff,
              'event_start':event_start,'event_end':event_end,'horizon_days':'7','history_days':'14',
              'min_history':'12','budget_denominator':'1000','cooldown_days':'7','allow_smart_decreases':'true'}
    if any(metadata.get(k)!=v for k,v in expected.items()): raise ValueError('synthetic evaluation contract mismatch')
    if sha256_file(model_path)!=expected_model_sha256: raise ValueError('frozen model binding mismatch')
    for key,path in [('panel_sha256',panel),('selection_sha256',selection),
                     ('decrease_ledger_sha256',directory/'smart_decreases.sqlite'),
                     ('model_sha256',model_path),('evaluation_sha256',directory/'evaluation/evaluation.json')]:
        if chain.get(key)!=sha256_file(path): raise ValueError('chain binding mismatch: '+key)
    source=audit_zip_candidate(archive,directory/'candidate',**source_args)
    decreases=audit_decrease_ledger(directory/'candidate/candidate.sqlite',bindings,
                                   directory/'smart_decreases.sqlite',start=source_args['start'],end=source_args['end'])
    if chain.get('decrease_audit') != decreases: raise ValueError('decrease audit record differs')
    merged=verify_merged_panel(panel,directory/'candidate/candidate.sqlite',bindings,start=score_start)
    evaluation=_load_small(directory/'evaluation/evaluation.json','worker evaluation')
    scores=audit_selection(selection,panel,evaluation,None,model=replay._load_model(model_path))
    if any(item.get('status')!='pass' for item in (source,merged,scores,decreases)):
        raise ValueError('post-worker audit did not pass')
    if inventory(directory)!=expected_inventory: raise ValueError('worker inventory changed during audit')
    if bound_contract is not None: validate_bound_zip(bound_contract)
    return {'status':'pass','profile':profile,'calendar':calendar,'source_audit':source,
            'decrease_audit':decreases,
            'merge_audit':merged,'score_evaluation_audit':scores,'input_sha256':expected_inventory,
            'publication':'not_authorized'}
