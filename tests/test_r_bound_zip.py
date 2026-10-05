"""External-bound ZIP consumers and supervised synthetic end-to-end chain."""
from contextlib import closing
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.reproducible.artifacts import sha256_file, verify_manifest
from pipeline.r_validation.zip_chain import run_zip_fixture
from pipeline.r_validation.bound_zip import validate_bound_zip
from pipeline.r_validation.release import ReleaseError
from pipeline.r_validation.outer_supervisor import supervise_zip_fixture

ROOT=Path(__file__).resolve().parents[1]


def contract_for(directory):
    def binding(path): return {'path':str(path), 'sha256':sha256_file(path)}
    return {'schema':'r-bound-zip-v1','profile':'synthetic_bound_zip',
            'source':{'archive':binding(directory/'synthetic.zip'),'receipt':None,
                      'prefix':'fixture','start':'2023-10-01','end':'2023-12-31'},
            'model':binding(ROOT/'examples/small_replay/current_lr.json'),
            'history':{'history':binding(directory/'synthetic_history.sqlite')},
            'spec':{'score_start':'2023-10-01','score_end':'2023-12-24',
                    'event_start':'2023-10-08','event_end':'2023-12-25',
                    'outcome_cutoff':'2023-12-31','horizon_days':7,'allow_smart_decreases':True},
            'authorization':None}


class BoundZipTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        run_zip_fixture(self.root/'source', calendar='quarter')
        self.contract=contract_for(self.root/'source')

    def test_bound_input_runs_under_snapshot_verification_and_publishes_only_synthetic(self):
        output=self.root/'supervised'
        result=supervise_zip_fixture(output,calendar='quarter',bound_contract=self.contract)
        self.assertEqual(result['completion_status'],'synthetic_complete')
        manifest=verify_manifest(output/'bound_synthetic_complete.json',expected_status='synthetic_complete')
        self.assertFalse(manifest['real_data_release'])
        self.assertFalse(manifest['gain_evidence']['support_gain'])
        self.assertFalse((output/'r_complete.json').exists())
        self.assertEqual(json.loads((output/'verification.json').read_text())['profile'],'synthetic_bound_zip')

    def test_real_profile_without_authorization_never_reads_archive(self):
        changed=copy.deepcopy(self.contract);changed['profile']='released_zip'
        with patch('pipeline.r_validation.bound_zip._file',side_effect=AssertionError('content touched')):
            with self.assertRaisesRegex(ReleaseError,'authorization'):
                validate_bound_zip(changed)

    def test_tampered_archive_fails_before_worker(self):
        archive=self.root/'source/synthetic.zip'
        archive.write_bytes(archive.read_bytes()+b'changed')
        output=self.root/'refused'
        with self.assertRaisesRegex(ReleaseError,'source ZIP bytes changed'):
            supervise_zip_fixture(output,calendar='quarter',bound_contract=self.contract)
        self.assertFalse((output/'request.json').exists())
        self.assertFalse((output/'bound_synthetic_complete.json').exists())
        self.assertEqual(json.loads((output/'failure.json').read_text())['status'],'failed')

    def test_changed_source_between_worker_and_verifier_blocks_completion(self):
        from pipeline.r_validation.post_worker_audit import inventory
        output=self.root/'changed_between_stages'
        changed=False
        def observe(directory):
            nonlocal changed
            result=inventory(directory)
            if not changed:
                changed=True
                archive=self.root/'source/synthetic.zip'
                archive.write_bytes(archive.read_bytes()+b'changed after scoring')
            return result
        with patch('pipeline.r_validation.outer_supervisor.inventory',side_effect=observe):
            with self.assertRaisesRegex(RuntimeError,'verifier exited'):
                supervise_zip_fixture(output,calendar='quarter',bound_contract=self.contract)
        self.assertTrue((output/'work/worker_candidate.json').exists())
        self.assertFalse((output/'bound_synthetic_complete.json').exists())
        self.assertIn('source ZIP bytes changed',(output/'verifier.log').read_text())

    def real_contract(self):
        from pipeline.r_validation.release import build_release
        import shutil
        project=self.root/'isolated_real_gate'
        raw=project/'data/raw/attempt'; raw.mkdir(parents=True)
        archive=raw/'data_Q4_2023.zip'; shutil.copyfile(self.root/'source/synthetic.zip',archive)
        release_path=project/'evidence/r_validation_v1/release.json';release_path.parent.mkdir(parents=True)
        release=build_release(r0_lock_digest='a'*64,source_object_id='test-object',
                              allowed_stages=['panel','score','evaluate','audit'])
        release_path.write_text(json.dumps(release))
        lock_path=project/'lock.json';lock_path.write_text('{}')
        def binding(p):return {'path':str(p),'sha256':sha256_file(p)}
        c=copy.deepcopy(self.contract);c['profile']='released_zip'
        c['source'].update(archive=binding(archive),prefix='data_Q4_2023')
        c['authorization']={'lock':binding(lock_path),'release':binding(release_path)}
        receipt=raw/'receipt.json'
        receipt_data={'schema':'r-download-completion-v1','status':'download_complete','source_object_id':'test-object',
            'url':'https://example.invalid/fixture.zip','bytes':archive.stat().st_size,
            'archive_sha256':sha256_file(archive),'archive':str(archive),
            'release_digest':release['release_digest'],'r0_lock_digest':'a'*64,'cleanup':{'state':'complete'}}
        lock={'lock_digest':'a'*64,'approved_parameters':c['model'],
              'historical_inputs':{'panels':c['history']}}
        config={'dates':{**c['spec'],'quarter_raw':['2023-10-01','2023-12-31']},
                'labels':{'horizon_days':7},'paths':{'raw':'data/raw/attempt'},
                'source_zip':{'bz_file_id':'test-object','url':'https://example.invalid/fixture.zip',
                              'content_length':archive.stat().st_size,'content_sha1':'none','accept_ranges':'bytes'}}
        (raw/'response.json').write_text(json.dumps({'status':200,'url':config['source_zip']['url'],
            'validated_before_body':True,'redirects_followed':0,'headers':{
                'content-length':str(archive.stat().st_size),'x-bz-file-id':'test-object',
                'x-bz-content-sha1':'none','accept-ranges':'bytes'}}))
        (raw/'network_attempt.json').write_text(json.dumps({'attempts':1,'automatic_retry':False,'source':config['source_zip']}))
        (raw/'download_candidate.json').write_text(json.dumps({'status':'download_candidate','bytes':archive.stat().st_size,
            'archive_sha256':sha256_file(archive),'response_sha256':sha256_file(raw/'response.json'),
            'network_attempt_sha256':sha256_file(raw/'network_attempt.json')}))
        from pipeline.reproducible.artifacts import write_manifest
        write_manifest(receipt,receipt_data,artifacts={p.name:p for p in raw.iterdir() if p.is_file()})
        c['source']['receipt']=binding(receipt)
        return project,c,lock,config

    def test_release_receipt_lock_and_model_must_all_agree(self):
        project,c,lock,config=self.real_contract()
        with patch('pipeline.r_validation.bound_zip.project_root',return_value=project), \
             patch('tools.run_r_validation._verify_lock_binding',return_value=lock), \
             patch('tools.run_r_validation._load_config',return_value=config):
            self.assertEqual(validate_bound_zip(c)['archive'],Path(c['source']['archive']['path']))
            receipt=Path(c['source']['receipt']['path'])
            data=json.loads(receipt.read_text());data['source_object_id']='different-object'
            receipt.write_text(json.dumps(data));c['source']['receipt']['sha256']=sha256_file(receipt)
            with self.assertRaisesRegex(ReleaseError,'download receipt is not bound'):
                validate_bound_zip(c)

    def test_unapproved_date_change_is_refused_before_archive(self):
        project,c,lock,config=self.real_contract()
        c['spec']['score_end']='2023-12-23'
        with patch('pipeline.r_validation.bound_zip.project_root',return_value=project), \
             patch('tools.run_r_validation._verify_lock_binding',return_value=lock), \
             patch('tools.run_r_validation._load_config',return_value=config):
            with self.assertRaisesRegex(ReleaseError,'dates differ'):
                validate_bound_zip(c)
