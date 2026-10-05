"""Regression checks for Q1 entry, acquisition evidence and retained outputs."""
import json
import datetime as dt
import tempfile
import unittest
from pathlib import Path
import hashlib
import zipfile
from unittest.mock import patch

from tools import run_q1_validation as cli
from pipeline.reproducible.source_snapshot import _files
from pipeline.reproducible.artifacts import ArtifactError
from pipeline.r_validation.budget_scope import cumulative_roots
from pipeline.r_validation.q1_binding import _validate_receipt
from pipeline.r_validation.q1_binding import historical_source_proof
from pipeline.r_validation.q1_directory import expected_members, verify_directory_proof, _response_range, fetch_directory
from pipeline.r_validation.release import ReleaseError
from pipeline.r_validation.engine import opportunity_events
from pipeline.r_validation.zip_source import RArchiveSource, SourceError

ROOT = Path(__file__).resolve().parents[1]


class Q1ReviewBoundaries(unittest.TestCase):
    def test_cli_accepts_output_option_without_executing(self):
        with patch.object(cli, 'cmd_run', return_value=0) as execute:
            self.assertEqual(cli.main(['run', '--stage', 'panel', '--lock', 'unused',
                                      '--output', 'unused']), 0)
            self.assertEqual(execute.call_args.args[0].output, 'unused')

    def test_release_parent_can_exist_without_blocking_attempt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as name:
            parent = Path(name)
            release = parent / 'release.json'
            release.write_text('{}')
            receipt = parent / 'receipt.json'
            receipt.write_text(json.dumps({'archive_sha256': 'unused'}))
            archive = parent / 'archive.zip'
            archive.write_bytes(b'synthetic')
            model = parent / 'model.json'
            model.write_text('{}')
            config = cli._config()
            config['paths']['evidence'] = str(parent)
            lock = {'lock_digest': 'fixture', 'approved_parameters': {
                'path': str(model), 'sha256': 'fixture'},
                'historical_inputs': {'panels': {}}}
            # Stop at contract validation: no real authorization or worker is created.
            with patch.object(cli, '_config', return_value=config), \
                 patch.object(cli, '_verify_lock', return_value=lock), \
                 patch.object(cli, 'require_release', return_value={}), \
                 patch.object(cli, 'RELEASE', release), \
                 patch.object(cli, 'validate_q1_bound_zip', side_effect=ReleaseError('contract reached')):
                self.assertEqual(cli.main(['run', '--stage', 'panel', '--lock', str(release),
                                           '--source-zip', str(archive), '--source-receipt', str(receipt),
                                           '--output', str(parent / 'attempt_001')]), 3)
            self.assertFalse((parent / 'attempt_001').exists())

    def test_q1_cli_is_in_snapshot_closure(self):
        self.assertIn(ROOT / 'tools/run_q1_validation.py', _files(ROOT))

    def test_q1_download_and_prior_attempts_are_counted(self):
        roots = cumulative_roots(ROOT / 'evidence/q1_2024_validation_v1/attempt_001')
        for relative in ('data/raw/q1_2024_validation_v1/source.partial',
                         'evidence/q1_2024_validation_v1/failed_attempt/failure.json'):
            self.assertTrue(any((ROOT / relative).is_relative_to(p) for p in roots))

    def test_unsigned_receipt_is_rejected_before_identity_checks(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as name:
            receipt = Path(name) / 'download_complete.json'
            receipt.write_text('{}')
            with self.assertRaises(ArtifactError):
                _validate_receipt(receipt, {}, cli._config(), Path(name)/'archive.zip', 'unused')

    def test_q1_download_profile_refuses_before_network_without_release(self):
        with patch('pipeline.r_validation.download_supervisor.authorize_q1',
                   side_effect=ReleaseError('q1 release missing')), \
             patch('urllib.request.OpenerDirector.open',
                   side_effect=AssertionError('network accessed')):
            from pipeline.r_validation.download_supervisor import supervise_download
            with self.assertRaisesRegex(ReleaseError, 'q1 release missing'):
                supervise_download(lock_path='missing', profile='released_q1_download')

    def test_historical_sources_are_bound_to_role_specific_completion_records(self):
        proof = historical_source_proof()
        self.assertEqual(set(proof['entries']), {'q1q2_panel', 'q3_panel', 'q4_candidate_panel'})
        self.assertEqual(proof['status'], 'pass')
        self.assertEqual(proof['entries']['q4_candidate_panel']['sqlite_sidecars']['wal_bytes'], 0)
        self.assertTrue(proof['entries']['q1q2_panel']['source']['sha256'])

    def test_directory_proof_rejects_an_unapproved_member(self):
        config = cli._config()
        csv_members, metadata = expected_members(
            config['source_zip']['prefix'], *config['dates']['quarter_raw'])
        proof = {
            'schema': 'q1-directory-proof-v1', 'status': 'pass',
            'source': {key: config['source_zip'][key] for key in
                       ('url', 'content_length', 'bz_file_id', 'content_sha1', 'accept_ranges', 'prefix')},
            'quarter': {'start': '2024-01-01', 'end': '2024-03-31', 'csv_member_count': 91},
            'csv_members': sorted(csv_members),
            'metadata_exception_members': sorted(metadata),
            'all_member_names': sorted(csv_members | metadata | {'data_Q1_2024/2024-04-01.csv'}),
            'central_directory': {'entry_count': len(csv_members | metadata) + 1},
        }
        with self.assertRaisesRegex(ReleaseError, 'member map differs'):
            verify_directory_proof(proof, config['source_zip'], start='2024-01-01', end='2024-03-31')

    def test_directory_preflight_stops_when_server_ignores_range(self):
        class Response:
            status = 200
            headers = {'Content-Length': '16'}
        with self.assertRaisesRegex(ReleaseError, '206 Content-Range'):
            _response_range(Response(), expected_start=0, expected_end=15, expected_total=16)

    def test_directory_preflight_rejects_wrong_object_identity(self):
        class Response:
            status = 206
            headers = {'Content-Range': 'bytes 0-15/16', 'Content-Length': '16',
                       'x-bz-file-id': 'wrong', 'x-bz-content-sha1': 'none',
                       'Accept-Ranges': 'bytes'}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def geturl(self): return 'https://example.test/q1.zip'
            def read(self, *args): raise AssertionError('body read before identity')
        class Opener:
            def open(self, *args, **kwargs): return Response()
        source = {'url': 'https://example.test/q1.zip', 'content_length': 16,
                  'bz_file_id': 'approved', 'content_sha1': 'none', 'accept_ranges': 'bytes',
                  'prefix': 'data_Q1_2024'}
        with self.assertRaisesRegex(ReleaseError, 'identity differs'):
            fetch_directory(source, opener=Opener())

    def test_q1_event_denominator_uses_only_the_locked_main_event_window(self):
        eligibility = {'main': [dt.date(2024, 3, 24)],
                       'outside': [dt.date(2024, 3, 24)],
                       'cutoff': [dt.date(2024, 3, 24)],
                       'future': [dt.date(2024, 3, 24)]}
        events = opportunity_events(
            {'main': dt.date(2024, 3, 25), 'outside': dt.date(2024, 3, 26),
             'cutoff': dt.date(2024, 3, 31), 'future': dt.date(2024, 4, 1)},
            event_start=dt.date(2024, 1, 8), event_end=dt.date(2024, 3, 25),
            eligibility=eligibility)
        self.assertEqual(set(events), {'main'})

    def test_q1_archive_reader_rejects_april_member_before_csv_reads(self):
        from pipeline.r_validation.q1_fixture import build_q1_inputs
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as name:
            base = Path(name)
            inputs = build_q1_inputs(ROOT, base / 'inputs')
            extra = base / 'extra.zip'
            with zipfile.ZipFile(inputs['archive']) as source, zipfile.ZipFile(extra, 'w') as target:
                for info in source.infolist():
                    target.writestr(info, source.read(info.filename))
                target.writestr('q1_fixture/2024-04-01.csv', source.read('q1_fixture/2024-03-31.csv'))
            digest = hashlib.sha256(extra.read_bytes()).hexdigest()
            with self.assertRaisesRegex(SourceError, 'inventory mismatch'):
                with RArchiveSource(extra, expected_sha256=digest, start='2024-01-01',
                                    end='2024-03-31', prefix='q1_fixture'):
                    pass


if __name__ == '__main__':
    unittest.main()
