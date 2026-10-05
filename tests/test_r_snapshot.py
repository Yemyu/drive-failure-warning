import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline.reproducible.source_snapshot import create_source_snapshot, verify_source_snapshot, SourceSnapshotError

ROOT=Path(__file__).resolve().parents[1]


class RSnapshotTests(unittest.TestCase):
    def test_zip_cli_runs_from_verified_snapshot_with_separate_data_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            base=Path(directory); snapshot=base/'code'
            created=create_source_snapshot(snapshot,project_root=ROOT)
            verify_source_snapshot(snapshot,project_root=ROOT,expectation=created['expectation'])
            env={**os.environ,'REPRO_PROJECT_ROOT':str(ROOT),'REPRO_CODE_ROOT':str(snapshot),
                 'PYTHONPATH':'/nonexistent/misleading'}
            completed=subprocess.run([sys.executable,'-I','-B',str(snapshot/'tools/run_r_validation.py'),
                                      'zip-fixture','--output',str(base/'attempt')],cwd='/',env=env,
                                     capture_output=True,text=True,timeout=30)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertEqual(json.loads((base/'attempt/zip_chain_result.json').read_text())['status'],'pass')
            verify_source_snapshot(snapshot,project_root=ROOT,expectation=created['expectation'])

    def test_changed_r_tool_is_rejected_by_snapshot_verifier(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            snapshot=Path(directory)/'code'
            created=create_source_snapshot(snapshot,project_root=ROOT)
            target=snapshot/'tools/run_r_validation.py'
            target.write_text(target.read_text()+'\n# changed\n')
            with self.assertRaisesRegex(SourceSnapshotError,'source snapshot file changed'):
                verify_source_snapshot(snapshot,project_root=ROOT,expectation=created['expectation'])


if __name__=='__main__': unittest.main()
