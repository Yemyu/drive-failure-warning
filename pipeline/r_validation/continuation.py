"""A reviewed continuation reuses the exact previously approved download."""
import json
from pathlib import Path

from pipeline.reproducible.artifacts import bound_path, sha256_file
from pipeline.reproducible.runtime_context import project_root
from pipeline.reproducible.supervisor import _load_small
from pipeline.r_validation.release import require_release, ReleaseError
from pipeline.r_validation.acquisition import validate_acquisition_receipt


def validate_continuation(release, current_lock, source):
    value=release.get('continuation')
    if value is None:return None
    expected={'previous_release','previous_lock','download_receipt','failure_record','decision','output'}
    if not isinstance(value,dict) or set(value)!=expected:
        raise ReleaseError('invalid reviewed continuation fields')
    paths={}
    for name in expected-{'output'}:
        item=value[name]
        if not isinstance(item,dict) or set(item)!={'path','sha256'}:
            raise ReleaseError('invalid continuation binding: '+name)
        paths[name]=bound_path(item['path'],name,must_exist=True)
        if sha256_file(paths[name])!=item['sha256']:
            raise ReleaseError('continuation evidence changed: '+name)
    root=project_root();output=bound_path(value['output'],'continuation output')
    if output != root/'evidence/r_validation_v1/attempt_001/continuation_001':
        raise ReleaseError('unapproved continuation output')
    if paths['download_receipt'] != root/'data/raw/r_validation_v1/attempt_001/download_complete.json':
        raise ReleaseError('continuation must reuse the original completed download')
    if paths['failure_record'] != root/'evidence/r_validation_v1/attempt_001/failure.json':
        raise ReleaseError('continuation must preserve the original failed computation')
    failure=_load_small(paths['failure_record'],'original computation failure')
    if failure.get('status')!='failed' or failure.get('cleanup',{}).get('state')!='complete':
        raise ReleaseError('original computation has not stopped and cleaned up')
    from tools.run_r_validation import _canonical_digest
    previous=_load_small(paths['previous_lock'],'previous lock')
    if previous.get('lock_digest')!=_canonical_digest({k:v for k,v in previous.items() if k!='lock_digest'}):
        raise ReleaseError('previous lock digest differs')
    old_release=require_release(paths['previous_release'],r0_lock_digest=previous['lock_digest'],
                                source_object_id=source['bz_file_id'],stage='panel')
    if old_release.get('continuation') is not None:
        raise ReleaseError('chained continuations are not authorized')
    if (previous.get('protocol_source_sha256')!=current_lock.get('protocol_source_sha256')
        or previous.get('config_sha256')!=current_lock.get('config_sha256')
        or previous.get('approved_parameters',{}).get('sha256')!=current_lock.get('approved_parameters',{}).get('sha256')
        or previous.get('historical_inputs')!=current_lock.get('historical_inputs')):
        raise ReleaseError('continuation changed protocol, configuration, model or history')
    receipt=_load_small(paths['download_receipt'],'previous download receipt')
    validate_acquisition_receipt(receipt,paths['download_receipt'],source)
    if (receipt.get('r0_lock_digest')!=previous['lock_digest']
        or receipt.get('release_digest')!=old_release['release_digest']
        or receipt.get('source_object_id')!=source['bz_file_id'] or receipt.get('url')!=source['url']
        or receipt.get('bytes')!=source['content_length']):
        raise ReleaseError('download did not use the previous approved source')
    return {'output':output,'receipt_path':paths['download_receipt'],
            'receipt_sha256':value['download_receipt']['sha256'],
            'release_digest':old_release['release_digest'],'r0_lock_digest':previous['lock_digest']}
