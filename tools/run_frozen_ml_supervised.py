"""Run one frozen ML diagnostic in an externally supervised staging directory.

The child owns no publication decision.  This parent samples the child
process group through the project's ps/libproc reader, enforces elapsed/RSS/
free-space limits, audits the staged result, and promotes it atomically only
after the audit succeeds.  A failed or killed child leaves no formal attempt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "evidence/ml_evaluation_v1"
PYTHON = sys.executable
MAX_SECONDS = 1800.0
MAX_RSS_BYTES = 2 * 1024 ** 3
MIN_FREE_BYTES = 2 * 1024 ** 3
MAX_OUTPUT_BYTES = 512 * 1024 ** 2

sys.path.insert(0, str(ROOT))
from pipeline.reproducible import resource_guard  # noqa: E402
from pipeline.reproducible.supervisor import _bounded_cleanup


def check_budget(started, stage_root, pgid=None):
    elapsed = time.monotonic() - started
    if elapsed > MAX_SECONDS:
        raise RuntimeError('cumulative elapsed limit')
    free = shutil.disk_usage(stage_root if stage_root.exists() else stage_root.parent).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError('free space below 2GiB')
    total = resource_guard._owned_bytes(stage_root, reject_symlinks=True) if stage_root.exists() else 0
    if total > MAX_OUTPUT_BYTES:
        raise RuntimeError('output limit')
    table, reader, failure, problems = resource_guard._read_process_table()
    if failure or not table:
        raise RuntimeError(f'resource sampling unavailable: {failure}')
    if pgid is not None:
        unresolved = [int(pid) for pid in problems.get('identity_unknown', {})
                      if resource_guard.classify_unknown_pid(int(pid), pgid) not in ('exited','other_group')]
        if unresolved:
            raise RuntimeError(f'unknown group membership: {unresolved}')
    rss, members, _ = resource_guard._group_rss_bytes_from(table, pgid, (os.getpid(),))
    if rss is None or rss > MAX_RSS_BYTES:
        raise RuntimeError(f'RSS unavailable or over limit: {rss}')
    return dict(elapsed_seconds=elapsed, free_bytes=free, rss_bytes=rss,
                owned_bytes=total, reader=reader, members=members)


def run_phase(command, started, stage_root, phase, records):
    """Monitor every child phase under the same parent deadline."""
    check_budget(started, stage_root)
    process = None
    members = set()
    record = dict(phase=phase, command=command, samples=[], cleanup=None)
    records.append(record)
    try:
        with (stage_root / f'{phase}.stdout').open('w') as stdout, (stage_root / f'{phase}.stderr').open('w') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       text=True, start_new_session=True,
                                       env={k:v for k,v in os.environ.items() if k != 'PYTHONPATH'})
            record['pid'] = process.pid
            while process.poll() is None:
                try:
                    sample = check_budget(started, stage_root, process.pid)
                except resource_guard.ResourceViolation:
                    if process.poll() is None:
                        raise
                    break  # exited between poll and group enumeration
                record['samples'].append(sample)
                members.update(sample['members'])
                time.sleep(.2)
            record['returncode'] = process.wait()
            if process.returncode:
                raise RuntimeError(f'{phase} failed with code {process.returncode}')
        check_budget(started, stage_root)
    finally:
        if process is not None:
            record['cleanup'] = _bounded_cleanup(process.pid, process, known_members=members)
            record['returncode'] = process.poll()
            if record['cleanup']['state'] != 'complete':
                raise RuntimeError(f'{phase}: cleanup not confirmed: {record["cleanup"]}')


def publish(stage, final):
    """Reserve the directory exclusively; install success marker last."""
    final.mkdir()  # refuses a competing file or directory, including empty ones
    try:
        for source in stage.iterdir():
            if source.name != 'summary.json':
                os.link(source, final / source.name)
        os.link(stage / 'summary.json', final / 'summary.json')
    except BaseException:
        # No success marker exists before the commit. Preserve partial files
        # and the sibling failure record for diagnosis, rather than overwrite.
        raise


def supervise(attempt_id):
    import analyze_frozen_ml as analysis
    analysis.safe_attempt_id(attempt_id)
    final = OUT_ROOT / attempt_id
    if final.exists():
        raise FileExistsError(final)
    started = time.monotonic()
    stage_root = OUT_ROOT / f'.staging_{attempt_id}_{os.getpid()}'
    check_budget(started, stage_root)
    stage_root.mkdir()
    stage = stage_root / attempt_id
    records = []
    old_handlers = {}
    def cancelled(signum, frame):
        raise RuntimeError(f'cancelled by signal {signum}')
    try:
        resource_guard.assert_fixture_controls_allowed('frozen_ml_real')
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, cancelled)
        worker = [PYTHON, '-B', str(ROOT / 'tools/analyze_frozen_ml.py'), '_worker',
                  '--attempt-id', attempt_id, '--staged-root', str(stage_root)]
        run_phase(worker, started, stage_root, 'compute', records)
        verifier = [PYTHON, '-B', str(ROOT / 'tools/analyze_frozen_ml.py'), 'audit', '--path', str(stage)]
        run_phase(verifier, started, stage_root, 'audit', records)
        check_budget(started, stage_root)
        receipt = dict(status='verified_before_publication', attempt_id=attempt_id, phases=records)
        (stage_root / 'supervision.json').write_text(json.dumps(receipt, indent=2)+'\n')
        check_budget(started, stage_root)
        # Block cancellation only around the commit decision, then restore
        # the caller's original mask. A signal after commit cannot roll it back.
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        committed = False
        try:
            publish(stage, final)
            committed = True
        finally:
            if committed:
                for sig in old_handlers:
                    signal.signal(sig, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        return dict(status='pass', attempt_id=attempt_id, receipt=str(stage_root/'supervision.json'))
    except BaseException as exc:
        # Preserve candidate bytes, but leave no completion marker in a failed
        # staging attempt. The final marker is installed only after audit.
        candidate = stage / 'summary.json'
        if candidate.exists() and not (final / 'summary.json').exists():
            candidate.rename(stage / 'candidate_summary.failed.json')
        failure = dict(status='failed', attempt_id=attempt_id, reason=str(exc), phases=records)
        (stage_root/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(supervise(args.attempt_id), ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
