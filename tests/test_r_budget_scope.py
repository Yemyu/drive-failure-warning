from pathlib import Path
import tempfile
import unittest

from pipeline.r_validation.budget_scope import cumulative_roots
from pipeline.reproducible.resource_guard import _owned_bytes,ResourceViolation
from pipeline.reproducible.resource_guard import ResourceLimits
from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
import json

ROOT=Path(__file__).resolve().parents[1]


class RBudgetScopeTests(unittest.TestCase):
    def test_existing_retained_bytes_stop_empty_new_attempt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with self.assertRaisesRegex(ResourceViolation,'output'):
                supervise_zip_fixture(output,limits=ResourceLimits(max_output_bytes=1))
            failure=json.loads((output/'failure.json').read_text())
            self.assertGreater(failure['resource_peak']['owned_bytes'],1)
            self.assertEqual(failure['cleanup']['state'],'not_started')
            self.assertFalse((output/'request.json').exists())

    def test_current_handoff_run_is_not_counted_twice(self):
        roots=cumulative_roots(ROOT/'handoff/runs/new-R-test')
        self.assertIn(ROOT/'handoff/runs',roots)
        self.assertNotIn(ROOT/'handoff/runs/new-R-test',roots)
        self.assertFalse(any(a!=b and a.is_relative_to(b) for a in roots for b in roots))

    def test_all_real_attempt_directories_are_in_scope(self):
        roots=cumulative_roots(ROOT/'evidence/r_validation_v1/fixtures/new')
        for directory in ('data/raw/r_validation_v1','data/derived/r_validation_v1','evidence/r_validation_v1'):
            self.assertIn(ROOT/directory,roots)

    def test_failed_attempt_wal_temporary_and_new_files_count(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            root=Path(directory); (root/'failed').mkdir(); (root/'new').mkdir()
            for name,size in [('failed/partial.sqlite',11),('failed/partial.sqlite-wal',13),
                              ('new/staged.tmp',17),('failed/error.json',19)]:
                (root/name).write_bytes(b'x'*size)
            self.assertEqual(_owned_bytes(root,reject_symlinks=True),60)
            (root/'new/later').write_bytes(b'123')
            self.assertEqual(_owned_bytes(root,reject_symlinks=True),63)

    def test_symlink_is_not_silently_excluded(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            root=Path(directory); (root/'value').write_bytes(b'123')
            (root/'alias').symlink_to(root/'value')
            with self.assertRaisesRegex(ResourceViolation,'symlink'): _owned_bytes(root,reject_symlinks=True)

    def test_individual_source_file_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            path=Path(directory)/'code.py'; path.write_bytes(b'12345')
            self.assertEqual(_owned_bytes(path,reject_symlinks=True),5)


if __name__=='__main__': unittest.main()
