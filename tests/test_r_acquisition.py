"""No external requests: fake responses and bounded loopback HTTP only."""
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import signal
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from pipeline.r_validation.acquisition import fetch_once, validate_download_records, validate_acquisition_receipt
from pipeline.r_validation.download_supervisor import supervise_download
from pipeline.r_validation.release import ReleaseError
from pipeline.reproducible.artifacts import verify_manifest, ArtifactError
from pipeline.reproducible.resource_guard import ResourceLimits

ROOT=Path(__file__).resolve().parents[1]


def source(url='http://127.0.0.1:1/fixture.zip'):
    return {'url':url,'content_length':4,'bz_file_id':'r0-local-fixture',
            'content_sha1':'none','accept_ranges':'bytes'}


class Response(io.BytesIO):
    def __init__(self, data=b'test', **headers):
        super().__init__(data);self.status=200;self.reads=0;self.headers=Message()
        for k,v in {'Content-Length':'4','x-bz-file-id':'r0-local-fixture',
                    'x-bz-content-sha1':'none','Accept-Ranges':'bytes',**headers}.items():self.headers[k]=v
    def geturl(self):return source()['url']
    def read(self,*args):self.reads+=1;return super().read(*args)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fetch(self,response,**kwargs):
        return fetch_once(source(),self.root,max_bytes=8,seconds=30,opener=lambda *a,**k:response,**kwargs)

    def test_success_binds_actual_response_and_body(self):
        result=self.fetch(Response());self.assertEqual(result['bytes'],4)
        validate_download_records(self.root,source(),result['archive_sha256'])
        self.assertFalse((self.root/'download_complete.json').exists())

    def test_wrong_identity_rejected_before_any_body_read(self):
        response=Response(**{'x-bz-file-id':'wrong'})
        with self.assertRaisesRegex(ReleaseError,'identity'):self.fetch(response)
        self.assertEqual(response.reads,0)
        self.assertFalse((self.root/'source.partial').exists())

    def test_duplicate_length_rejected_before_body(self):
        response=Response();response.headers['Content-Length']='4'
        with self.assertRaisesRegex(ReleaseError,'duplicate'):self.fetch(response)
        self.assertEqual(response.reads,0)

    def test_truncated_body_preserves_partial_and_failure(self):
        with self.assertRaisesRegex(ReleaseError,'truncated'):self.fetch(Response(b'te'))
        self.assertEqual((self.root/'source.partial').read_bytes(),b'te')
        self.assertFalse((self.root/'download_candidate.json').exists())

    def test_larger_body_rejected(self):
        with self.assertRaisesRegex(ReleaseError,'exceeds'):self.fetch(Response(b'test-more'))
        self.assertEqual(json.loads((self.root/'network_failure.json').read_text())['body_bytes_received'],9)

    def test_failed_attempt_cannot_retry(self):
        with self.assertRaises(ReleaseError):self.fetch(Response(b'x'))
        response=Response()
        with self.assertRaises(ArtifactError):self.fetch(response)
        self.assertEqual(response.reads,0)

    def test_deadline_after_body_blocks_candidate(self):
        ticks=iter([0,0,0,0,31])
        with self.assertRaises(TimeoutError):self.fetch(Response(),clock=lambda:next(ticks))
        self.assertFalse((self.root/'download_candidate.json').exists())

    def test_real_access_without_release_never_starts_network_or_output(self):
        with patch('pipeline.r_validation.download_supervisor.authorize',side_effect=ReleaseError('no release')), \
             patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('network accessed')):
            with self.assertRaisesRegex(ReleaseError,'no release'):supervise_download(lock_path='missing')

    def server(self,mode):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                requests.append(self.path)
                self.send_response(302 if mode=='redirect' else 200)
                self.send_header('Content-Length','4')
                self.send_header('x-bz-file-id','wrong' if mode=='wrong' else 'r0-local-fixture')
                self.send_header('x-bz-content-sha1','none');self.send_header('Accept-Ranges','bytes')
                if mode=='redirect':self.send_header('Location','/second')
                self.end_headers()
                if mode=='stall':time.sleep(4)
                try:self.wfile.write(b'te' if mode=='short' else b'test')
                except (BrokenPipeError,ConnectionResetError):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        return source(f'http://127.0.0.1:{server.server_port}/fixture.zip'),requests

    def test_real_subprocess_and_snapshot_download_complete(self):
        fixture,requests=self.server('complete');output=self.root/'download'
        result=supervise_download(fixture_source=fixture,output=output)
        self.assertEqual(result['status'],'synthetic_download_complete');self.assertEqual(requests,['/fixture.zip'])
        verified=verify_manifest(output/'synthetic_download_complete.json',expected_status='synthetic_download_complete')
        self.assertEqual(verified['verification'],'supervised_download_bytes_and_identity')
        self.assertEqual(verified['cleanup']['state'],'complete')
        with self.assertRaises(Exception):
            validate_acquisition_receipt(verified,output/'synthetic_download_complete.json',fixture)
        with self.assertRaises(FileExistsError):supervise_download(fixture_source=fixture,output=output)
        self.assertEqual(len(requests),1)

    def test_network_failures_never_publish_and_do_not_retry(self):
        for mode in ('wrong','short','redirect'):
            with self.subTest(mode=mode):
                fixture,requests=self.server(mode);output=self.root/mode
                with self.assertRaisesRegex(RuntimeError,'worker failed'):
                    supervise_download(fixture_source=fixture,output=output)
                self.assertEqual(len(requests),1)
                self.assertFalse((output/'synthetic_download_complete.json').exists())
                self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')

    def test_stalled_body_is_stopped_by_parent_budget(self):
        fixture,requests=self.server('stall');output=self.root/'stalled'
        with self.assertRaisesRegex(RuntimeError,'elapsed'):
            supervise_download(fixture_source=fixture,output=output,limits=ResourceLimits(max_elapsed_seconds=2))
        self.assertEqual(len(requests),1)
        failure=json.loads((output/'failure.json').read_text())
        self.assertEqual(failure['cleanup']['state'],'complete')
        self.assertFalse((output/'synthetic_download_complete.json').exists())

    def test_cancel_during_body_cleans_group_without_completion(self):
        fixture,requests=self.server('stall');output=self.root/'cancelled'
        def cancel_when_started():
            end=time.monotonic()+3
            while not requests and time.monotonic()<end:time.sleep(.01)
            if requests:os.kill(os.getpid(),signal.SIGTERM)
        thread=threading.Thread(target=cancel_when_started);thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                supervise_download(fixture_source=fixture,output=output)
        finally:thread.join()
        self.assertEqual(len(requests),1)
        self.assertEqual(json.loads((output/'failure.json').read_text())['cleanup']['state'],'complete')
        self.assertFalse((output/'synthetic_download_complete.json').exists())

    def test_initial_resource_failure_makes_no_request(self):
        fixture,requests=self.server('complete');output=self.root/'resource'
        with self.assertRaises(RuntimeError):
            supervise_download(fixture_source=fixture,output=output,limits=ResourceLimits(max_rss_bytes=1,max_elapsed_seconds=30))
        self.assertEqual(requests,[])
        self.assertFalse((output/'synthetic_download_complete.json').exists())
