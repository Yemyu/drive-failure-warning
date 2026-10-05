"""Single-response streaming acquisition; real access requires the fixed release."""
import hashlib
import os
from pathlib import Path
import ssl
import time
import urllib.request

from pipeline.reproducible.artifacts import bound_path, sha256_file, write_json_exclusive
from pipeline.reproducible.runtime_context import project_root
from pipeline.r_validation.release import require_release, ReleaseError


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReleaseError('download redirect refused')


def _verified_opener():
    """Build an HTTPS opener without disabling certificate verification.

    The bundled Python on macOS can lack a usable default CA path even though
    the host supplies ``/etc/ssl/cert.pem``.  Prefer the interpreter's normal
    verification paths and use that host CA bundle only as an explicit,
    still-verified fallback.  A missing CA remains a hard error.
    """
    candidates = []
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile:
        candidates.append(defaults.cafile)
    candidates.append('/etc/ssl/cert.pem')
    context = None
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.R_OK):
            try:
                context = ssl.create_default_context(cafile=candidate)
                break
            except OSError:
                continue
    if context is None:
        context = ssl.create_default_context()
    return urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))


def authorize(lock_path):
    from tools.run_r_validation import _verify_lock_binding, _load_config
    root = project_root()
    lock_path = bound_path(lock_path, 'download lock', must_exist=True)
    lock = _verify_lock_binding(lock_path)
    config = _load_config()
    release_path = root/'evidence/r_validation_v1/release.json'
    release = require_release(release_path, r0_lock_digest=lock['lock_digest'],
                              source_object_id=config['source_zip']['bz_file_id'], stage='panel')
    if set(release['allowed_stages']) != {'panel','score','evaluate','audit'}:
        raise ReleaseError('download requires release for the complete chain')
    if release.get('continuation') is not None:
        raise ReleaseError('reviewed continuation reuses its completed download; no new request authorized')
    return config, lock, release


def authorize_q1(lock_path):
    """Authorize the Q1 download against the Q1 lock and release entry."""
    from pipeline.r_validation.q1_binding import RELEASE_PATH, _config, _verify_lock

    lock_path = bound_path(lock_path, "Q1 download lock", must_exist=True)
    lock = _verify_lock(lock_path)
    config = _config()
    release = require_release(
        RELEASE_PATH,
        r0_lock_digest=lock["lock_digest"],
        source_object_id=config["source_zip"]["bz_file_id"],
        stage="panel",
    )
    if set(release.get("allowed_stages", ())) != set(config["release_gate"]["allowed_stages"]):
        raise ReleaseError("Q1 download requires a release for the complete chain")
    return config, lock, release


def response_facts(response, source):
    """Inspect status, framing and identity before the first body read."""
    headers = {key.lower(): value for key,value in response.headers.items()}
    for key in ('content-length','x-bz-file-id','x-bz-content-sha1','accept-ranges'):
        if len(response.headers.get_all(key, [])) != 1:
            raise ReleaseError('missing or duplicate download header: '+key)
    expected = {'content-length':str(source['content_length']), 'x-bz-file-id':source['bz_file_id'],
                'x-bz-content-sha1':source['content_sha1'], 'accept-ranges':source['accept_ranges']}
    if (response.status != 200 or response.geturl() != source['url']
            or any(headers.get(key) != value for key,value in expected.items())
            or headers.get('content-encoding','identity') != 'identity'
            or 'transfer-encoding' in headers or 'content-range' in headers):
        raise ReleaseError('download response identity, status or framing mismatch')
    return {'status':response.status,'url':response.geturl(),'headers':headers,
            'validated_before_body':True,'redirects_followed':0}


def fetch_once(source, output, *, max_bytes, seconds, opener=None, clock=time.monotonic):
    """Write a candidate only. The external supervisor owns final publication."""
    output = Path(output)
    if type(source['content_length']) is not int or not 0 < source['content_length'] <= max_bytes:
        raise ReleaseError('source length exceeds network budget')
    # Exclusive marker is committed before any network access, even on failure.
    write_json_exclusive(output/'network_attempt.json', {'attempts':1,'source':source,
        'max_network_body_bytes':max_bytes,'timeout_seconds':seconds,'automatic_retry':False})
    start = clock(); received = 0; digest = hashlib.sha256()
    def deadline():
        if clock()-start >= seconds: raise TimeoutError('download deadline exceeded')
    try:
        deadline()
        open_response = opener or _verified_opener().open
        request = urllib.request.Request(source['url'], headers={'Accept-Encoding':'identity'})
        with open_response(request, timeout=min(seconds,30)) as response:
            facts = response_facts(response, source)
            write_json_exclusive(output/'response.json', facts)
            deadline()
            with (output/'source.partial').open('xb') as stream:
                while True:
                    deadline()
                    block = response.read(min(1024*1024, max_bytes-received+1))
                    received += len(block)
                    if received > max_bytes or received > source['content_length']:
                        raise ReleaseError('download body exceeds declared length or budget')
                    if not block: break
                    stream.write(block); digest.update(block)
                    deadline()
                stream.flush(); os.fsync(stream.fileno())
        if received != source['content_length']:
            raise ReleaseError('download body is truncated')
        deadline()
        result = {'status':'download_candidate','bytes':received,'archive_sha256':digest.hexdigest(),
                  'elapsed_seconds':clock()-start,'response_sha256':sha256_file(output/'response.json'),
                  'network_attempt_sha256':sha256_file(output/'network_attempt.json')}
        write_json_exclusive(output/'download_candidate.json',result)
        return result
    except BaseException as exc:
        write_json_exclusive(output/'network_failure.json', {'status':'failed','reason':str(exc),
                             'body_bytes_received':received,'automatic_retry':False})
        raise


def validate_acquisition_receipt(receipt, receipt_path, source):
    """Consumers require the supervised marker and its actual response evidence."""
    from pipeline.reproducible.artifacts import verify_manifest
    from pipeline.reproducible.supervisor import _load_small
    verified = verify_manifest(receipt_path, expected_status='download_complete')
    verified.pop('verified_artifacts', None)
    if verified != receipt or receipt.get('schema') != 'r-download-completion-v1':
        raise ReleaseError('download completion manifest differs')
    validate_download_records(Path(receipt_path).parent, source, receipt['archive_sha256'])
    if receipt.get('cleanup',{}).get('state') != 'complete':
        raise ReleaseError('download cleanup unconfirmed')


def validate_download_records(directory, source, archive_sha256):
    from pipeline.reproducible.supervisor import _load_small
    directory = Path(directory)
    response = _load_small(directory/'response.json','download response')
    attempt = _load_small(directory/'network_attempt.json','download attempt')
    candidate = _load_small(directory/'download_candidate.json','download candidate')
    expected = {'content-length':str(source['content_length']),'x-bz-file-id':source['bz_file_id'],
                'x-bz-content-sha1':source['content_sha1'],'accept-ranges':source['accept_ranges']}
    if (response.get('status') != 200 or response.get('url') != source['url']
        or response.get('validated_before_body') is not True or response.get('redirects_followed') != 0
        or any(response.get('headers',{}).get(k) != v for k,v in expected.items())
        or attempt.get('attempts') != 1 or attempt.get('automatic_retry') is not False
        or attempt.get('source') != source or candidate.get('status') != 'download_candidate'
        or candidate.get('bytes') != source['content_length']
        or candidate.get('archive_sha256') != archive_sha256
        or candidate.get('response_sha256') != sha256_file(directory/'response.json')
        or candidate.get('network_attempt_sha256') != sha256_file(directory/'network_attempt.json')
        or response.get('headers',{}).get('content-encoding','identity') != 'identity'
        or 'transfer-encoding' in response.get('headers',{}) or 'content-range' in response.get('headers',{})):
        raise ReleaseError('download response or acquisition proof differs')
