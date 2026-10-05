import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline.reproducible.source_snapshot import create_source_snapshot
from pipeline.r_validation.history import file_sha

ROOT=Path(__file__).resolve().parents[1]


class RWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.tmp'); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.code=self.base/'code'
        snapshot=create_source_snapshot(self.code,project_root=ROOT)
        self.request={'schema':'r-snapshot-worker-v1','mode':'self_generated_zip_fixture','project_root':str(ROOT),
                      'code_root':str(self.code),'output':str(self.base/'output'),'expectation':snapshot['expectation']}

    def run_request(self,wrong_digest=False):
        path=self.base/'request.json'; path.write_text(json.dumps(self.request))
        env={**os.environ,'REPRO_PROJECT_ROOT':str(ROOT),'REPRO_CODE_ROOT':str(self.code),'PYTHONPATH':'/wrong'}
        return subprocess.run([sys.executable,'-I','-B',str(self.code/'tools/run_r_validation.py'),
                               'snapshot-worker','--request',str(path),'--request-sha256','0'*64 if wrong_digest else file_sha(path)],
                              cwd='/',env=env,capture_output=True,text=True,timeout=30)

    def test_actual_worker_records_only_snapshot_modules(self):
        result=self.run_request(); self.assertEqual(result.returncode,0,result.stderr)
        candidate=json.loads((self.base/'output/worker_candidate.json').read_text())
        self.assertEqual(candidate['status'],'pending_external_verification')
        self.assertEqual(candidate['modules']['pipeline']['violations'],[])
        self.assertTrue(any(m['path']=='tools/run_small_replay.py' for m in candidate['modules']['tools']))

    def test_real_mode_refused_before_output(self):
        self.request['mode']='production'
        result=self.run_request(); self.assertNotEqual(result.returncode,0)
        self.assertIn('only accepts self-generated',result.stderr)
        self.assertFalse((self.base/'output').exists())

    def test_request_binding_refused_before_output(self):
        result=self.run_request(wrong_digest=True); self.assertNotEqual(result.returncode,0)
        self.assertIn('request SHA mismatch',result.stderr)
        self.assertFalse((self.base/'output').exists())

    def test_tampered_snapshot_refused_before_output(self):
        tool=self.code/'tools/run_small_replay.py'; tool.write_text(tool.read_text()+'\n# altered\n')
        result=self.run_request(); self.assertNotEqual(result.returncode,0)
        self.assertIn('source snapshot file changed',result.stderr)
        self.assertFalse((self.base/'output').exists())


if __name__=='__main__': unittest.main()
