"""Bind an external ZIP to its source receipt, frozen model and approved run."""
from pathlib import Path
from pipeline.reproducible.artifacts import bound_path, sha256_file
from pipeline.reproducible.supervisor import _load_small
from pipeline.reproducible.runtime_context import project_root
from pipeline.r_validation.release import require_release, ReleaseError

SCHEMA = 'r-bound-zip-v1'
PROFILES = {'released_zip', 'synthetic_bound_zip', 'released_q1_zip', 'synthetic_q1_bound_zip'}


def _file(binding, field):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ReleaseError(f'invalid {field} binding')
    path = bound_path(binding['path'], field, must_exist=True)
    if sha256_file(path) != binding['sha256']:
        raise ReleaseError(f'{field} bytes changed')
    return path


def validate_bound_zip(contract):
    """Authorise before touching archive bytes; all consumers repeat this check."""
    fields = {'schema', 'profile', 'source', 'model', 'history', 'spec', 'authorization'}
    if not isinstance(contract, dict) or set(contract) != fields or contract['schema'] != SCHEMA:
        raise ReleaseError('invalid bound ZIP contract')
    profile = contract['profile']
    if profile not in PROFILES:
        raise ReleaseError('unknown bound ZIP profile')
    if profile in {'released_q1_zip', 'synthetic_q1_bound_zip'}:
        from pipeline.r_validation.q1_binding import validate_q1_bound_zip
        return validate_q1_bound_zip(contract)
    root = project_root()
    source = contract['source']
    if not isinstance(source, dict) or set(source) != {'archive', 'receipt', 'prefix', 'start', 'end'}:
        raise ReleaseError('invalid bound source fields')
    if profile == 'released_zip':
        from tools.run_r_validation import _verify_lock_binding, _load_config
        auth = contract['authorization']
        if not isinstance(auth, dict) or set(auth) != {'lock', 'release'}:
            raise ReleaseError('missing real authorization binding')
        # These files contain only approval facts. Never inspect source content
        # in order to decide whether its initial access is authorised.
        lock_path = _file(auth['lock'], 'R lock')
        release_path = bound_path(auth['release']['path'], 'R release', must_exist=True)
        if release_path != root / 'evidence/r_validation_v1/release.json':
            raise ReleaseError('release must use the fixed project entry')
        lock = _verify_lock_binding(lock_path)
        config = _load_config()
        release = require_release(release_path, r0_lock_digest=lock['lock_digest'],
                                  source_object_id=config['source_zip']['bz_file_id'], stage='panel')
        if set(release['allowed_stages']) != {'panel', 'score', 'evaluate', 'audit'}:
            raise ReleaseError('release must cover the entire ZIP chain')
        _file(auth['release'], 'R release')
        from pipeline.r_validation.continuation import validate_continuation
        continuation=validate_continuation(release,lock,config['source_zip'])
        expected_spec = {key: config['dates'][key] for key in
                         ('score_start', 'score_end', 'event_start', 'event_end', 'outcome_cutoff')}
        expected_spec.update(horizon_days=config['labels']['horizon_days'], allow_smart_decreases=True)
        if contract['spec'] != expected_spec:
            raise ReleaseError('bound ZIP dates differ from the approved protocol')
        expected_history = {role: {'path': str(bound_path(item['path'], role, must_exist=True)),
                                   'sha256': item['sha256']}
                            for role, item in lock['historical_inputs']['panels'].items()}
        expected_model = {'path': str(bound_path(lock['approved_parameters']['path'], 'model', must_exist=True)),
                          'sha256': lock['approved_parameters']['sha256']}
        if contract['history'] != expected_history or contract['model'] != expected_model:
            raise ReleaseError('model or history differs from the approved lock')
        if (source['prefix'] != 'data_Q4_2023' or
                [source['start'], source['end']] != config['dates']['quarter_raw']):
            raise ReleaseError('real ZIP source calendar or prefix differs')
        receipt_path = _file(source['receipt'], 'download receipt')
        if continuation is not None and (receipt_path!=continuation['receipt_path']
                or source['receipt']['sha256']!=continuation['receipt_sha256']):
            raise ReleaseError('source receipt differs from reviewed continuation')
        receipt = _load_small(receipt_path, 'download receipt')
        archive = bound_path(source['archive']['path'], 'source ZIP', must_exist=True)
        expected_archive = root / config['paths']['raw'] / 'data_Q4_2023.zip'
        if archive != expected_archive:
            raise ReleaseError('source ZIP must use the protocol attempt path')
        expected_receipt = {'status': 'download_complete', 'source_object_id': config['source_zip']['bz_file_id'],
                            'url': config['source_zip']['url'], 'bytes': config['source_zip']['content_length'],
                            'archive_sha256': source['archive']['sha256'], 'archive': str(archive),
                            'release_digest': continuation['release_digest'] if continuation else release['release_digest'],
                            'r0_lock_digest': continuation['r0_lock_digest'] if continuation else lock['lock_digest']}
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ReleaseError('download receipt is not bound to the approved source and release')
        from pipeline.r_validation.acquisition import validate_acquisition_receipt
        validate_acquisition_receipt(receipt, receipt_path, config['source_zip'])
        if archive.stat().st_size != config['source_zip']['content_length']:
            raise ReleaseError('ZIP file length differs from the official object')
    else:
        if contract['authorization'] is not None or source['receipt'] is not None:
            raise ReleaseError('synthetic source cannot carry real authorization')
        if source['prefix'] != 'fixture':
            raise ReleaseError('synthetic ZIP must use fixture prefix')
    archive = _file(source['archive'], 'source ZIP')
    model = _file(contract['model'], 'frozen model')
    if not isinstance(contract['history'], dict) or not contract['history']:
        raise ReleaseError('bound ZIP has no history')
    history = {role: {'path': str(_file(binding, role)), 'sha256': binding['sha256']}
               for role, binding in contract['history'].items()}
    return {'archive': archive, 'model_source': model,
            'expected_model_sha256': contract['model']['sha256'], 'spec': contract['spec'],
            'source_args': {'expected_sha256': source['archive']['sha256'], 'start': source['start'],
                            'end': source['end'], 'prefix': source['prefix'], 'historical_inputs': history}}


def run_bound_contract(output, contract):
    from pipeline.r_validation.zip_execution import run_bound_zip_chain
    args = validate_bound_zip(contract)
    Path(output).mkdir(parents=True, exist_ok=False)
    return run_bound_zip_chain(output, **args, profile=contract['profile'])
