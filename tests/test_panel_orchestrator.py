import json
import os
import signal
import time
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline.reproducible.panel_orchestrator import rebuild_panels, PanelOrchestratorError
from pipeline.reproducible.artifacts import verify_manifest
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceLimits, ResourceViolation
from tests.test_source_streams import ROOT, fixture, q1_manifest_fixture


def verification_with_descendant(kwargs):
    os.setsid()
    evidence = Path(kwargs['evidence_dir']); evidence.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    (evidence / 'descendant.json').write_text(json.dumps({'pid': child.pid, 'group': os.getpgrp()}))
    (evidence / 'candidate_manifest.json').write_text('{"status":"complete"}')
    (evidence / 'verification.log').write_bytes(b'x' * 2 * 1024 * 1024)
    time.sleep(30)


def cancel_supervisor(kwargs):
    os.setsid()
    evidence = Path(kwargs['evidence_dir']); evidence.mkdir()
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(30)


class PanelSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q1 = q1_manifest_fixture(self.root)
        columns = json.loads((ROOT / 'evidence/q3/validation_v1/source_receipt_v1.json').read_text())['schema']
        fixture(self.root, [('2023-07-01', [{'serial_number': 'disk-a'}])], columns)

    def run_cli(self, *options):
        return subprocess.run([
            sys.executable, '-B', str(ROOT / 'tools/run_research.py'), 'rebuild-panels',
            '--q1q2-manifest', str(self.q1), '--q3-receipt', str(self.root / 'receipt.json'),
            '--output-root', str(self.root / 'output'), '--evidence', str(self.root / 'evidence'),
            '--max-panel-seconds', '30', '--poll-seconds', '0.01', *options,
        ], cwd=self.root, capture_output=True, text=True, timeout=45)

    def test_complete_pair_is_verified_before_publication(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = verify_manifest(self.root / 'evidence/real_panel_rebuild_manifest.json', expected_status='complete')
        self.assertEqual(manifest['panels']['q1q2']['daily_rows'], 44)
        self.assertEqual(manifest['panels']['q3']['daily_rows'], 1)
        self.assertEqual(manifest['cross_quarter_identity']['status'], 'pass')

    def test_preflight_error_is_recorded_without_completion(self):
        self.q1.write_text('{bad json')
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertTrue((self.root / 'evidence/failure.json').exists())
        self.assertTrue((self.root / 'evidence/supervisor_failure.json').exists())
        self.assertFalse((self.root / 'evidence/real_panel_rebuild_manifest.json').exists())

    def test_timeout_stops_before_publication(self):
        result = self.run_cli('--max-panel-seconds', '0.01')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'evidence/real_panel_rebuild_manifest.json').exists())
        self.assertIn('elapsed', (self.root / 'evidence/supervisor_failure.json').read_text())

    def test_separate_evidence_directory_counts_toward_limit(self):
        output = self.root / 'output'; output.mkdir()
        evidence = self.root / 'evidence'; evidence.mkdir()
        (evidence / 'log').write_bytes(b'x' * 1024)
        guard = ResourceGuard(ROOT, output, ResourceLimits(max_output_bytes=512), extra_outputs=(evidence,))
        with self.assertRaisesRegex(ResourceViolation, 'output'):
            guard.check_or_raise()
        self.assertEqual(guard.last_snapshot['owned_bytes'], 1024)

    def test_zero_output_budget_stops_worker(self):
        result = self.run_cli('--max-output-mib', '0')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'evidence/real_panel_rebuild_manifest.json').exists())
        self.assertIn('output', (self.root / 'evidence/supervisor_failure.json').read_text())

    def test_overflow_during_verification_kills_descendant_and_withholds_completion(self):
        with patch('pipeline.reproducible.panel_orchestrator._panel_job', verification_with_descendant):
            with self.assertRaises(PanelOrchestratorError):
                rebuild_panels(q1q2_manifest=self.q1, q3_receipt=self.root/'receipt.json',
                    output_root=self.root/'output', evidence_dir=self.root/'evidence',
                    limits=ResourceLimits(max_output_bytes=1024*1024, max_elapsed_seconds=10, poll_seconds=0.01))
        self.assertFalse((self.root/'evidence/real_panel_rebuild_manifest.json').exists())
        pid = json.loads((self.root/'evidence/descendant.json').read_text())['pid']
        # A killed child can briefly remain a zombie until the OS reaps it.
        for _ in range(50):
            state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            time.sleep(0.02)
        self.assertTrue(not state or state.startswith('Z'), state)

    def test_sigterm_cancels_without_completion(self):
        previous = signal.getsignal(signal.SIGTERM)
        with patch('pipeline.reproducible.panel_orchestrator._panel_job', cancel_supervisor):
            with self.assertRaisesRegex(PanelOrchestratorError, 'cancellation'):
                rebuild_panels(q1q2_manifest=self.q1, q3_receipt=self.root/'receipt.json',
                    output_root=self.root/'output', evidence_dir=self.root/'evidence',
                    limits=ResourceLimits(max_elapsed_seconds=10, poll_seconds=0.01))
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertFalse((self.root/'evidence/real_panel_rebuild_manifest.json').exists())
