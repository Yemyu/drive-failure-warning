"""R lock includes shared execution code and rejects stale acceptance code."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pipeline.reproducible.source_snapshot import create_source_snapshot, _files
from tools import run_r_validation as cli

ROOT = Path(__file__).resolve().parents[1]


class LockClosureTests(unittest.TestCase):
    def test_shared_execution_dependencies_are_bound(self):
        bound = set(cli.implementation_files())
        self.assertTrue({p.relative_to(ROOT).as_posix() for p in _files(ROOT)} <= bound)
        self.assertIn('pipeline/reproducible/resource_guard.py', bound)
        self.assertIn('pipeline/reproducible/supervisor.py', bound)

    def test_old_passing_snapshot_cannot_bind_changed_current_code(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as temp:
            attempt = Path(temp)
            create_source_snapshot(attempt / 'snapshot', project_root=ROOT)
            (attempt / 'synthetic_complete.json').write_text('{}')
            verified = {'calendar': 'quarter', 'real_data_release': False, 'manifest_hash': 'test'}
            # Isolate the current-source comparison after manifest validation;
            # real artifact/manifest validation is covered by publication tests.
            with patch('pipeline.reproducible.artifacts.verify_manifest', return_value=verified):
                self.assertEqual(cli._acceptance_binding(attempt)['status'], 'pass')
                code = attempt / 'snapshot/pipeline/reproducible/resource_guard.py'
                code.write_text(code.read_text() + '\n# stale snapshot\n')
                with self.assertRaisesRegex(SystemExit, 'different source bytes'):
                    cli._acceptance_binding(attempt)
