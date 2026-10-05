"""Reviewed reuse of a prior download; all approvals here are test fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation.continuation import validate_continuation
from pipeline.r_validation.release import build_release, ReleaseError, _digest
from pipeline.reproducible.artifacts import sha256_file, write_manifest
from tools.run_r_validation import _canonical_digest

ROOT=Path(__file__).resolve().parents[1]


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        raw=self.root/'data/raw/r_validation_v1/attempt_001';raw.mkdir(parents=True)
        evidence=self.root/'evidence/r_validation_v1/attempt_001';evidence.mkdir(parents=True)
        self.source={'url':'https://example.invalid/unit.zip','content_length':4,'bz_file_id':'unit-object',
                     'content_sha1':'none','accept_ranges':'bytes'}
        self.old={'schema':'r0-lock-v1','protocol_source_sha256':'a'*64,'config_sha256':'b'*64,
                  'approved_parameters':{'sha256':'c'*64},'historical_inputs':{'test':'unchanged'}}
        self.old['lock_digest']=_canonical_digest(self.old);self.current=copy.deepcopy(self.old)
        old_release=build_release(r0_lock_digest=self.old['lock_digest'],source_object_id='unit-object',allowed_stages=['panel','score','evaluate','audit'])
        archive=raw/'data_Q4_2023.zip';archive.write_bytes(b'test')
        response={'status':200,'url':self.source['url'],'validated_before_body':True,'redirects_followed':0,
                  'headers':{'content-length':'4','x-bz-file-id':'unit-object','x-bz-content-sha1':'none','accept-ranges':'bytes'}}
        self.write(raw/'response.json',response)
        self.write(raw/'network_attempt.json',{'attempts':1,'automatic_retry':False,'source':self.source})
        self.write(raw/'download_candidate.json',{'status':'download_candidate','bytes':4,'archive_sha256':sha256_file(archive),
            'response_sha256':sha256_file(raw/'response.json'),'network_attempt_sha256':sha256_file(raw/'network_attempt.json')})
        receipt=raw/'download_complete.json'
        write_manifest(receipt,{'schema':'r-download-completion-v1','status':'download_complete',
            'source_object_id':'unit-object','url':self.source['url'],'bytes':4,'archive_sha256':sha256_file(archive),
            'archive':str(archive),'cleanup':{'state':'complete'},'r0_lock_digest':self.old['lock_digest'],
            'release_digest':old_release['release_digest']},artifacts={p.name:p for p in raw.iterdir()})
        self.paths={'previous_lock':self.root/'old_lock.json','previous_release':self.root/'old_release.json',
                    'download_receipt':receipt,'failure_record':evidence/'failure.json','decision':self.root/'decision.md'}
        self.write(self.paths['previous_lock'],self.old);self.write(self.paths['previous_release'],old_release)
        self.write(self.paths['failure_record'],{'status':'failed','cleanup':{'state':'complete'}})
        self.paths['decision'].write_text('Explicit unit-test approval only')
        self.value={name:self.bind(path) for name,path in self.paths.items()}
        self.value['output']=str(evidence/'continuation_001')
        self.release={'continuation':self.value}

    def write(self,path,value):path.write_text(json.dumps(value))
    def bind(self,path):return {'path':str(path),'sha256':sha256_file(path)}
    def validate(self):
        with patch('pipeline.r_validation.continuation.project_root',return_value=self.root):
            return validate_continuation(self.release,self.current,self.source)

    def test_approved_reuse_keeps_old_acquisition_digests(self):
        result=self.validate()
        self.assertEqual(result['r0_lock_digest'],self.old['lock_digest'])
        self.assertEqual(result['receipt_path'],self.paths['download_receipt'])

    def test_unreviewed_missing_fields_are_refused(self):
        del self.value['decision']
        with self.assertRaisesRegex(ReleaseError,'fields'):self.validate()

    def test_changed_decision_is_refused(self):
        self.paths['decision'].write_text('changed')
        with self.assertRaisesRegex(ReleaseError,'evidence changed'):self.validate()

    def test_changed_methods_model_or_history_are_refused(self):
        for field in ('protocol_source_sha256','config_sha256','approved_parameters','historical_inputs'):
            with self.subTest(field=field):
                self.current=copy.deepcopy(self.old)
                self.current[field]={'sha256':'d'*64} if field=='approved_parameters' else 'changed'
                with self.assertRaisesRegex(ReleaseError,'changed protocol'):self.validate()

    def test_unknown_cleanup_does_not_authorize_continuation(self):
        self.write(self.paths['failure_record'],{'status':'failed','cleanup':{'state':'unknown'}})
        self.value['failure_record']=self.bind(self.paths['failure_record'])
        with self.assertRaisesRegex(ReleaseError,'cleaned up'):self.validate()

    def test_arbitrary_next_attempt_is_refused(self):
        self.value['output']=str(self.root/'evidence/r_validation_v1/attempt_002')
        with self.assertRaisesRegex(ReleaseError,'output'):self.validate()

    def test_other_source_approval_cannot_be_reused(self):
        release=json.loads(self.paths['previous_release'].read_text());release.pop('release_digest');release['source_object_id']='other'
        release['release_digest']=_digest(release);self.write(self.paths['previous_release'],release)
        self.value['previous_release']=self.bind(self.paths['previous_release'])
        with self.assertRaisesRegex(ReleaseError,'source object'):self.validate()

    def test_original_download_bytes_remain_bound(self):
        (self.paths['download_receipt'].parent/'data_Q4_2023.zip').write_bytes(b'edit')
        with self.assertRaisesRegex(Exception,'artifact changed'):self.validate()
