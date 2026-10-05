import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation.outer_supervisor import supervise_zip_fixture
from pipeline.r_validation import synthetic_publication as publication
from pipeline.reproducible.artifacts import verify_manifest
from pipeline.reproducible.artifacts import ArtifactError

ROOT=Path(__file__).resolve().parents[1]


class RSyntheticPublicationTests(unittest.TestCase):
    def test_short_manifest_write_has_no_completion(self):
        original=publication.write_manifest
        def short(*args,**kwargs):
            return original(*args,**kwargs,_write_text=lambda stream,text:stream.write(text[:-1]))
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            with patch.object(publication,'write_manifest',side_effect=short):
                with self.assertRaises(ArtifactError): supervise_zip_fixture(output)
            self.assertFalse((output/'synthetic_complete.json').exists())
            self.assertTrue((output/'failure.json').exists())

    def test_changed_verified_file_during_staging_refused(self):
        original=publication.write_manifest
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'
            def stage(*args,**kwargs):
                result=original(*args,**kwargs)
                (output/'work/extra.txt').write_text('changed after verification')
                return result
            with patch.object(publication,'write_manifest',side_effect=stage):
                with self.assertRaisesRegex(RuntimeError,'verified inputs changed'):
                    supervise_zip_fixture(output)
            self.assertFalse((output/'synthetic_complete.json').exists())

    def test_complete_binds_files_without_real_release(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'; result=supervise_zip_fixture(output)
            manifest=verify_manifest(output/'synthetic_complete.json',expected_status='synthetic_complete')
            self.assertEqual(result['synthetic_completion'],'synthetic_complete')
            self.assertFalse(manifest['real_data_release'])
            self.assertIn('verification.json',manifest['artifacts'])
            self.assertFalse((output/'synthetic_complete.staged.json').exists())
            self.assertFalse((output/'failure.json').exists())

    def run_link_case(self,mode):
        original=publication.os.link
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
            output=Path(directory)/'run'; final=output/'synthetic_complete.json'
            def link(source,target,*args,**kwargs):
                if Path(target)==final:
                    if mode=='competition': final.write_text('existing owner')
                    if mode=='failure': raise OSError('injected publication failure')
                    result=original(source,target,*args,**kwargs)
                    if mode=='after_signal': os.kill(os.getpid(),signal.SIGTERM)
                    return result
                return original(source,target,*args,**kwargs)
            with patch.object(publication.os,'link',side_effect=link):
                if mode=='after_signal':
                    result=supervise_zip_fixture(output)
                    self.assertEqual(result['synthetic_completion'],'synthetic_complete')
                    verify_manifest(final,expected_status='synthetic_complete')
                    self.assertFalse((output/'failure.json').exists())
                else:
                    with self.assertRaises(OSError): supervise_zip_fixture(output)
                    self.assertTrue((output/'failure.json').exists())
                    self.assertTrue((output/'synthetic_complete.staged.json').exists())
                    if mode=='competition': self.assertEqual(final.read_text(),'existing owner')
                    else: self.assertFalse(final.exists())

    def test_competing_target_is_not_overwritten(self): self.run_link_case('competition')
    def test_link_failure_has_no_completion(self): self.run_link_case('failure')
    def test_signal_after_commit_does_not_create_failure(self): self.run_link_case('after_signal')

    def test_signal_before_commit_restores_nonempty_mask(self):
        original=publication.write_manifest
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1})
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
                output=Path(directory)/'run'
                def stage(*args,**kwargs):
                    result=original(*args,**kwargs)
                    os.kill(os.getpid(),signal.SIGTERM)
                    return result
                with patch.object(publication,'write_manifest',side_effect=stage):
                    with self.assertRaisesRegex(RuntimeError,'cancelled before synthetic publication'):
                        supervise_zip_fixture(output)
                self.assertFalse((output/'synthetic_complete.json').exists())
                self.assertIn(signal.SIGUSR1,signal.pthread_sigmask(signal.SIG_BLOCK,set()))
        finally: signal.pthread_sigmask(signal.SIG_SETMASK,mask)


if __name__=='__main__': unittest.main()
