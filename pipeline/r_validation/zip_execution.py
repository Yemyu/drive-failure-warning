"""ZIP computation shared by synthetic and release-checked consumers."""
from contextlib import closing
import hashlib
import datetime as dt
import json
from pathlib import Path
import sqlite3
from pipeline.reproducible.artifacts import bound_path
from pipeline.r_validation.cli_fixture import _seal
from pipeline.r_validation.history import file_sha
from pipeline.r_validation.zip_panel import build_zip_candidate
from pipeline.r_validation.zip_history import prepare_zip_scoring_panel, verify_merged_panel
from pipeline.r_validation.stage_runner import _protocol_metrics, StageRunnerError
from pipeline.r_validation.independent_audit import audit_selection
from pipeline.r_validation.decrease_ledger import build_decrease_ledger, audit_decrease_ledger
from tools import run_small_replay as replay


def run_bound_zip_chain(output, archive, *, source_args, model_source, expected_model_sha256, spec, profile='self_generated_zip_fixture'):
    """Caller supplies bound source/history/model; no release is issued here."""
    if profile not in ('self_generated_zip_fixture','synthetic_bound_zip','released_zip',
                       'synthetic_q1_bound_zip','released_q1_zip'):
        raise StageRunnerError('unknown ZIP execution profile')
    output=bound_path(output,'ZIP chain output')
    archive=bound_path(archive,'ZIP archive',must_exist=True)
    model_source=bound_path(model_source,'ZIP frozen model',must_exist=True)
    model_bytes=model_source.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest()!=expected_model_sha256:
        raise StageRunnerError('bound model SHA mismatch before ZIP computation')
    start=dt.date.fromisoformat(spec['score_start']); end=dt.date.fromisoformat(spec['score_end'])
    horizon=spec['horizon_days']
    if (horizon!=7 or start>end or source_args['start']!=spec['score_start']
        or source_args['end']!=spec['outcome_cutoff']
        or spec['outcome_cutoff']!=(end+dt.timedelta(days=7)).isoformat()
        or spec['event_start']!=(start+dt.timedelta(days=7)).isoformat()
        or spec['event_end']!=(end+dt.timedelta(days=1)).isoformat()
        or spec.get('allow_smart_decreases') is not True):
        raise StageRunnerError('ZIP calculation date/policy contract mismatch')
    spec = {**spec, 'feature_storage': 'current'}
    candidate=output/'candidate'
    build_zip_candidate(archive,candidate,**source_args)
    ledger=output/'smart_decreases.sqlite'
    ledger_args=dict(start=source_args['start'], end=source_args['end'])
    build_decrease_ledger(candidate/'candidate.sqlite', source_args['historical_inputs'], ledger, **ledger_args)
    decreases=audit_decrease_ledger(candidate/'candidate.sqlite', source_args['historical_inputs'], ledger, **ledger_args)
    prepared=output/'prepared'
    preparation=prepare_zip_scoring_panel(archive,candidate,prepared,**source_args)
    panel=prepared/'panel.sqlite'
    expected_panel_sha=preparation['panel_sha256']
    if file_sha(panel)!=expected_panel_sha: raise StageRunnerError('prepared panel changed before scoring')
    chain=output/'evaluation'; chain.mkdir()
    model_path=chain/'current_lr.json'
    model_path.write_bytes(model_bytes)
    model_sha=file_sha(model_path); model=replay._load_model(model_path)
    selection=chain/'selection.sqlite'; access=[]
    with closing(sqlite3.connect(panel.as_uri()+'?mode=ro&immutable=1',uri=True)) as source:
        source.row_factory=sqlite3.Row
        replay._score_and_select(replay._asof_reader(source,access),[],model,selection,access,spec=spec)
    selection_sha=file_sha(selection); _seal(selection,chain/'selection_seal.json')
    with closing(sqlite3.connect(panel.as_uri()+'?mode=ro&immutable=1',uri=True)) as source:
        source.row_factory=sqlite3.Row
        evaluation=replay._evaluate(selection,source,selection_sha,{},access,spec=spec)
    # Per-query facts are already sealed in selection.sqlite and summarized
    # in evaluation. Do not retain the full in-memory log through both audits.
    access.clear()
    evaluation['protocol_metrics']=_protocol_metrics(selection,panel,evaluation)
    (chain/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
    audit=audit_selection(selection,panel,evaluation,chain/'audit.json',model=model)
    merged=verify_merged_panel(panel,candidate/'candidate.sqlite',source_args['historical_inputs'],start=source_args['start'])
    if any(item.get('status')!='pass' for item in (preparation['source_audit'],merged,audit)):
        raise StageRunnerError('one or more ZIP chain audits did not pass')
    for path,sha in preparation['source_audit']['input_sha256'].items():
        if file_sha(path)!=sha: raise StageRunnerError('source binding changed during scoring')
    if file_sha(panel)!=expected_panel_sha or file_sha(selection)!=selection_sha or file_sha(model_path)!=model_sha:
        raise StageRunnerError('sealed panel, selection or model changed')
    if not evaluation['evaluation_opened_after_selection_close']:
        raise StageRunnerError('evaluation was not after selection close')
    result={'status':'pass','profile':profile,'source_audit':preparation['source_audit'],
            'decrease_audit':decreases,'decrease_ledger_sha256':file_sha(ledger),
            'merge_audit':merged,'score_evaluation_audit':audit,'evaluation_after_seal':True,
            'panel_sha256':expected_panel_sha,'selection_sha256':selection_sha,'model_sha256':model_sha,
            'evaluation_sha256':file_sha(chain/'evaluation.json'),
            'external_supervision':'pending','publication':'not_authorized','fit':0}
    (output/'zip_chain_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
