"""Audit CLI must recompute artifacts even if saved reports still say pass."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing

from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.synthetic_suite import run_synthetic_suite
from pipeline.r_validation.history import file_sha

ROOT = Path(__file__).resolve().parents[1]


class AuditCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        run_cli_fixture_chain(self.path, root=ROOT)
        run_synthetic_suite(self.path, root=ROOT)

    def cli(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'tools/run_r_validation.py'),
                                 'audit', '--attempt', str(self.path)], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        return result.returncode, json.loads(result.stdout)

    def test_clean_audit_recomputes_without_writing(self):
        before = {str(p): file_sha(p) for p in self.path.rglob('*') if p.is_file()}
        code, result = self.cli()
        self.assertEqual(code, 0, result)
        self.assertEqual(result['mode'], 'read_only_recomputed')
        after = {str(p): file_sha(p) for p in self.path.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_corrupt_score_with_updated_hash_is_still_rejected(self):
        chain = self.path/'cli_chain'
        with closing(sqlite3.connect(chain/'selection.sqlite')) as db:
            db.execute("UPDATE scores SET score=score+100 WHERE model='smart_nonzero'")
            db.commit()
        # Defeat a superficial file-hash-only checker while retaining old pass flags.
        record = json.loads((chain/'chain_results.json').read_text())
        record['selection_sha256'] = file_sha(chain/'selection.sqlite')
        (chain/'chain_results.json').write_text(json.dumps(record))
        code, result = self.cli()
        self.assertEqual(code, 1, result)
        self.assertIn('SMART score mismatch', result['reason'])

    def test_missing_actual_panel_is_not_masked_by_saved_pass(self):
        (self.path/'cli_chain/panel.sqlite').unlink()
        code, result = self.cli()
        self.assertEqual(code, 1, result)


if __name__ == '__main__':
    unittest.main()
