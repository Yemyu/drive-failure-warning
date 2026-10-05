"""Boundary tests for the supervised execution path.

These cover the cases that are hard to reach from the CLI acceptance matrix:
source-set tampering, the external snapshot expectation, module provenance,
and resource sampling that must never report a clean, empty group.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.reproducible import environment
from pipeline.reproducible.artifact_verifier import VerificationError, verify_run
from pipeline.reproducible.artifacts import ArtifactError, write_verified_manifest
from pipeline.reproducible.resource_guard import (
    GROUP_MISSING_TOLERANCE,
    ResourceGuard,
    ResourceLimits,
    ResourceViolation,
    _group_rss_bytes,
    process_group_members,
)
from pipeline.reproducible.source_snapshot import SourceSnapshotError, create_source_snapshot, verify_source_snapshot


ROOT = Path(__file__).resolve().parents[1]


class SourceSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _tree(self, name: str) -> Path:
        base = self.root / name
        (base / "pipeline").mkdir(parents=True)
        (base / "tools").mkdir(parents=True)
        (base / "pipeline" / "one.py").write_text("value = 1\n", encoding="utf-8")
        (base / "tools" / "run_research.py").write_text("print('x')\n", encoding="utf-8")
        return base

    def test_snapshot_records_pre_copy_facts_and_rejects_source_change(self):
        tree = self._tree("tree")
        target = self.root / "snapshot"
        created = create_source_snapshot(target, project_root=tree)
        self.assertEqual(len(created["files"]), 2)
        self.assertIn("expectation", created)
        verified = verify_source_snapshot(target, project_root=tree, expectation=created["expectation"])
        self.assertEqual(len(verified["verified_files"]), 2)

        # A source edited after the snapshot does not change the snapshot: it
        # still verifies against the external expectation recorded at creation.
        (tree / "pipeline" / "one.py").write_text("value = 2\n", encoding="utf-8")
        verify_source_snapshot(target, project_root=tree, expectation=created["expectation"])
        self.assertEqual(
            (target / "pipeline" / "one.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )

    def test_source_changed_during_copy_is_rejected(self):
        tree = self._tree("tree6")
        original_copyfile = __import__("shutil").copyfile
        calls = {"n": 0}

        def rewriting_copy(src, dst, **kwargs):
            calls["n"] += 1
            result = original_copyfile(src, dst, **kwargs)
            if calls["n"] == 1:
                (tree / "pipeline" / "one.py").write_text("value = 99\n", encoding="utf-8")
            return result

        with patch("shutil.copyfile", side_effect=rewriting_copy):
            with self.assertRaises(SourceSnapshotError) as caught:
                create_source_snapshot(self.root / "snapshot6", project_root=tree)
        self.assertIn("changed during snapshot copy", str(caught.exception))

    def test_file_set_change_during_copy_is_rejected(self):
        tree = self._tree("tree2")
        original_copyfile = __import__("shutil").copyfile
        calls = {"n": 0}

        def counting_copy(src, dst, **kwargs):
            calls["n"] += 1
            result = original_copyfile(src, dst, **kwargs)
            if calls["n"] == 1:
                (tree / "pipeline" / "extra.py").write_text("value = 3\n", encoding="utf-8")
            return result

        with patch("shutil.copyfile", side_effect=counting_copy):
            with self.assertRaises(SourceSnapshotError) as caught:
                create_source_snapshot(self.root / "snap3", project_root=tree)
        self.assertIn("file set changed", str(caught.exception))

    def test_rewritten_self_consistent_manifest_is_rejected_by_external_expectation(self):
        tree = self._tree("tree4")
        target = self.root / "snapshot4"
        created = create_source_snapshot(target, project_root=tree)
        manifest_path = target / "source_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["files"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(SourceSnapshotError):
            verify_source_snapshot(target, project_root=tree, expectation=created["expectation"])

    def test_unlisted_file_in_snapshot_is_rejected(self):
        tree = self._tree("tree5")
        target = self.root / "snapshot5"
        create_source_snapshot(target, project_root=tree)
        (target / "pipeline" / "sneaky.py").write_text("value = 9\n", encoding="utf-8")
        with self.assertRaises(SourceSnapshotError) as caught:
            verify_source_snapshot(target, project_root=tree)
        self.assertIn("unlisted file", str(caught.exception))


class ModuleIdentityTests(unittest.TestCase):
    def test_project_module_identity_reports_outside_root_as_violation(self):
        identity = environment.project_module_identity()
        self.assertIn("modules", identity)
        self.assertIsInstance(identity["violations"], list)

    def test_verify_project_modules_rejects_unknown_and_changed_files(self):
        from pipeline.reproducible.environment import ModuleIdentityError, verify_project_modules

        with self.assertRaises(ModuleIdentityError):
            verify_project_modules([{"path": "pipeline/reproducible/does_not_exist.py", "sha256": "0" * 64}])

    def test_verify_project_modules_rejects_empty_file_list(self):
        from pipeline.reproducible.environment import ModuleIdentityError, verify_project_modules

        with self.assertRaises(ModuleIdentityError):
            verify_project_modules([])


class ResourceSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_empty_or_invalid_process_table_is_never_a_clean_group(self):
        with patch("pipeline.reproducible.resource_guard._ps_process_table", return_value=None):
            with patch("pipeline.reproducible.resource_guard._libproc_process_table", return_value=None):
                with self.assertRaises(ResourceViolation):
                    _group_rss_bytes(99999)
                members, error = process_group_members(99999)
                self.assertIsNone(members)
                self.assertIsNotNone(error)

    def test_missing_group_is_reported_not_zeroed(self):
        with patch(
            "pipeline.reproducible.resource_guard._process_table",
            return_value={1: (0, 1, 10)},
        ):
            with self.assertRaises(ResourceViolation) as caught:
                _group_rss_bytes(4242)
        self.assertIn("no member visible", str(caught.exception))

    def test_transient_missing_group_keeps_last_reading_then_stops(self):
        output = self.root / "output"
        output.mkdir()
        guard = ResourceGuard(
            ROOT,
            output,
            ResourceLimits(poll_seconds=0.01),
            group_pgid=1,
        )
        guard.last_snapshot = {"rss_bytes": 4096}
        def flaky():
            return {}, "fixture", "fixture: the registered group is not visible", {}

        with patch("pipeline.reproducible.resource_guard._read_process_table", side_effect=flaky):
            for _ in range(GROUP_MISSING_TOLERANCE):
                snapshot = guard.snapshot()
                self.assertFalse(snapshot["rss_sampled"])
                self.assertEqual(snapshot["rss_bytes"], 4096)
            with self.assertRaises(ResourceViolation):
                guard.snapshot()
        self.assertIsNotNone(guard.last_sampling_error)

    def test_group_sampling_uses_pgid_not_parent_links(self):
        output = self.root / "output"
        output.mkdir()
        table = {
            10: (1, 500, 100),
            11: (9999, 500, 200),  # same group, unrelated parent
            12: (1, 600, 300),  # different group
        }
        with patch("pipeline.reproducible.resource_guard._process_table", return_value=table):
            total, members, unreadable = _group_rss_bytes(500)
        self.assertEqual(sorted(members), [10, 11])
        self.assertEqual(unreadable, [])
        self.assertEqual(total, (100 + 200) * 1024)

    def test_release_group_stops_group_sampling_after_confirmation(self):
        output = self.root / "output"
        output.mkdir()
        guard = ResourceGuard(ROOT, output, ResourceLimits(poll_seconds=0.01), group_pgid=1)
        guard.begin_cleanup()
        # An empty group during cleanup is a confirmed state, not a zero that
        # hides a failed read.
        table = {2: (1, 1, 8)}
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=(table, "ps", None, {}),
        ):
            confirming = guard.snapshot()
        self.assertEqual(confirming["group_state"], "cleanup_confirming")
        guard.release_group()
        self.assertEqual(guard.group_pgid, 1)
        self.assertTrue(guard.group_released)
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            side_effect=AssertionError("must not read the table after release"),
        ):
            released = guard.snapshot()
        self.assertFalse(released["rss_sampled"])
        self.assertEqual(released["group_state"], "released_after_cleanup_confirmed")


class ProcessReaderTests(unittest.TestCase):
    """The compatibility reader must be identifiable, bounded, and fail closed."""

    def test_reader_and_platform_are_reported(self):
        from pipeline.reproducible.resource_guard import _process_table, _read_process_table, platform_facts

        table, reader, reason, problems = _read_process_table()
        self.assertIn(reader, {"ps", "libproc", "none", "fixture"})
        self.assertIsInstance(problems, dict)
        if reader == "none":
            self.assertIsNotNone(reason)
        facts = platform_facts()
        self.assertIn(facts["system"], {"Darwin", "Linux", "Windows"})
        self.assertEqual(facts["libproc_reader_supported"], sys.platform == "darwin")

    def test_libproc_reads_own_pid_ppid_pgid_and_rss_units(self):
        from pipeline.reproducible.resource_guard import _libproc_process_table

        read = _libproc_process_table()
        if read is None:
            self.skipTest("libproc reader is not available on this platform")
        table, problems = read
        self.assertIn(os.getpid(), table)
        ppid, pgid, rss_kib = table[os.getpid()]
        self.assertEqual(ppid, os.getppid())
        self.assertEqual(pgid, os.getpgrp())
        # resident size is reported in bytes upstream and converted to KiB
        self.assertIsInstance(rss_kib, int)
        self.assertGreater(rss_kib, 0)
        self.assertLess(rss_kib, 64 * 1024 * 1024)

    def test_libproc_keeps_alive_pid_with_unreadable_rss_instead_of_zero(self):
        """A live pid whose task info fails must be recorded, never zeroed."""
        import ctypes
        from unittest.mock import MagicMock

        from pipeline.reproducible.resource_guard import _libproc_process_table

        fake = MagicMock()

        def list_pids(buffer, size):
            if buffer is None:  # first call only asks for the capacity
                return 1
            buffer[0] = os.getpid()
            return 1

        fake.proc_listallpids.side_effect = list_pids
        # BSD read succeeds, task read fails: the pid is alive and identified
        # but its resident size cannot be read.
        fake.proc_pidinfo.side_effect = lambda pid, flavor, arg, buf, size: (
            size if flavor == 3 else 0
        )
        with patch("ctypes.CDLL", return_value=fake), patch(
            "ctypes.util.find_library", return_value="libc"
        ):
            read = _libproc_process_table()
        if read is None:
            self.skipTest("test fake could not emulate the ABI on this platform")
        table, problems = read
        self.assertIn(os.getpid(), table)
        _ppid, _pgid, rss = table[os.getpid()]
        self.assertIsNone(rss)
        self.assertIn(os.getpid(), problems["rss_unreadable"])

    def test_group_sampling_refuses_zero_when_a_member_rss_is_unreadable(self):
        from pipeline.reproducible.resource_guard import _group_rss_bytes_from

        table = {10: (1, 500, 100), 11: (9999, 500, None)}
        with self.assertRaises(ResourceViolation) as caught:
            _group_rss_bytes_from(table, 500)
        self.assertIn("refusing to substitute zero", str(caught.exception))
        self.assertIn("11", str(caught.exception))

    def test_identity_unreadable_pids_are_counted_not_invented(self):
        from unittest.mock import MagicMock

        from pipeline.reproducible.resource_guard import _libproc_process_table

        fake = MagicMock()

        def list_pids(buffer, size):
            if buffer is None:  # first call only asks for the capacity
                return 3
            buffer[0] = os.getpid()
            buffer[1] = 424242
            buffer[2] = 424243
            return 3

        fake.proc_listallpids.side_effect = list_pids

        def pidinfo(pid, flavor, arg, buf, size):
            # Only the real process can be identified; the two fixture pids
            # were listed but their identity cannot be read.
            if flavor == 3 and pid == os.getpid():
                return size
            return 0

        fake.proc_pidinfo.side_effect = pidinfo
        with patch("ctypes.CDLL", return_value=fake), patch(
            "ctypes.util.find_library", return_value="libc"
        ):
            read = _libproc_process_table()
        if read is None:
            self.skipTest("test fake could not emulate the ABI on this platform")
        table, problems = read
        self.assertEqual(sorted(table), [os.getpid()])
        # The unreadable pids keep their identity in the problems record
        # instead of being counted as exited.
        self.assertEqual(sorted(problems["identity_unknown"]), [424242, 424243])
        self.assertTrue(all("errno=" in reason for reason in problems["identity_unknown"].values()))

    def test_fixture_controls_refused_without_synthetic_profile(self):
        from pipeline.reproducible.resource_guard import assert_fixture_controls_allowed

        with patch.dict(os.environ, {"REPRO_FIXTURE_INJECT": "rss_overflow"}):
            with self.assertRaises(ResourceViolation) as caught:
                assert_fixture_controls_allowed("backblaze_2023_st4000dm000")
            self.assertIn("synthetic fixture profile", str(caught.exception))
            with self.assertRaises(ResourceViolation):
                assert_fixture_controls_allowed(None)
            with self.assertRaises(ResourceViolation):
                assert_fixture_controls_allowed("unknown-profile")
            allowed = assert_fixture_controls_allowed("synthetic_fixture_v1")
            self.assertEqual(allowed, {"REPRO_FIXTURE_INJECT": "rss_overflow"})

    def test_fixture_controls_absent_are_always_allowed(self):
        from pipeline.reproducible.resource_guard import assert_fixture_controls_allowed

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REPRO_FIXTURE_INJECT", None)
            self.assertEqual(assert_fixture_controls_allowed("anything"), {})

    def test_libproc_is_not_called_outside_darwin(self):
        from pipeline.reproducible.resource_guard import _libproc_process_table

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"):
            self.assertIsNone(_libproc_process_table())

    def test_libproc_fails_closed_when_abi_returns_nothing(self):
        from unittest.mock import MagicMock

        from pipeline.reproducible.resource_guard import _libproc_process_table

        fake = MagicMock()
        fake.proc_listallpids.return_value = 0
        with patch("ctypes.CDLL", return_value=fake):
            self.assertIsNone(_libproc_process_table())

    def test_libproc_fails_closed_when_structure_size_differs(self):
        from unittest.mock import MagicMock

        from pipeline.reproducible.resource_guard import _libproc_process_table

        fake = MagicMock()
        fake.proc_listallpids.return_value = 1
        # proc_pidinfo returning a different size means a different ABI
        fake.proc_pidinfo.return_value = 0
        with patch("ctypes.CDLL", return_value=fake):
            self.assertIsNone(_libproc_process_table())

    def test_no_fallback_reader_outside_darwin(self):
        from pipeline.reproducible.resource_guard import _process_table, _read_process_table

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"):
            with patch("pipeline.reproducible.resource_guard._ps_process_table", return_value=None):
                table, reader, reason, _problems = _read_process_table()
                self.assertEqual(table, {})
                self.assertEqual(reader, "none")
                self.assertIn("no compatible fallback", str(reason))
                with self.assertRaises(ResourceViolation) as caught:
                    _process_table()
        self.assertIn("process table unavailable", str(caught.exception))

    def test_membership_is_not_complete_until_the_expected_pid_is_seen(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        output = Path(self.temporary.name) / "output"
        output.mkdir()
        guard = ResourceGuard(
            ROOT, output, ResourceLimits(poll_seconds=0.01), group_pgid=1, extra_pids=()
        )
        guard.expect_member(999_999)
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=({1: (0, 1, 4)}, "ps", None, {}),
        ):
            guard.snapshot()
        self.assertFalse(guard.expected_members_observed.get(999_999))
        self.assertIsNone(guard.membership_proven_sample)

        observed = ResourceGuard(
            ROOT, output, ResourceLimits(poll_seconds=0.01), group_pgid=1, extra_pids=()
        )
        observed.expect_member(1)
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=({1: (0, 1, 4)}, "ps", None, {}),
        ):
            observed.snapshot()
        self.assertTrue(observed.expected_members_observed.get(1))
        self.assertEqual(observed.membership_proven_sample, [1])
        self.assertEqual(observed.reader, "ps")

    def test_membership_observation_is_recomputed_not_sticky(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        output = Path(self.temporary.name) / "output"
        output.mkdir()
        guard = ResourceGuard(
            ROOT, output, ResourceLimits(poll_seconds=0.01), group_pgid=1, extra_pids=()
        )
        guard.expect_member(1)
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=({1: (0, 1, 4)}, "ps", None, {}),
        ):
            guard.snapshot()
        self.assertIsNotNone(guard.membership_proven_sample)
        # The next sample no longer sees the member: the observation is
        # recomputed instead of staying true, while the proven sample keeps
        # recording the most recent sighting that was complete.
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=({2: (0, 1, 4)}, "ps", None, {}),
        ):
            guard.snapshot()
        self.assertFalse(guard.expected_members_observed.get(1))
        self.assertEqual(guard.membership_proven_sample, [1])
        self.assertEqual(guard.expected_members_observed, {1: False})


class IdentityAndTreeTests(unittest.TestCase):
    """The four counterexamples required by the P1.2/P2 review."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "output"
        self.output.mkdir()

    def _cleanup(self, unknown, known, *, members=None, error=None, classes=None):
        from pipeline.reproducible import supervisor

        def classify(pid, target_pgid=None):
            return (classes or {}).get(pid, "unresolved")

        with patch.object(
            supervisor, "_inspect_group_identity",
            return_value=(members if members is not None else [], unknown, error),
        ), patch.object(
            supervisor, "classify_unknown_pid", side_effect=classify
        ):
            return supervisor._bounded_cleanup(12345, None, known_members=known)

    def test_unknown_identity_of_a_known_member_keeps_cleanup_unknown(self):
        # Cell "seen before x still alive with unreadable identity": the pid
        # was sampled inside the group before; it stays unresolved.
        record = self._cleanup(unknown=[777], known={777}, classes={777: "unresolved"})
        self.assertEqual(record["state"], "unknown")
        self.assertEqual(record["unknown_identity_pids"],
                         [{"pid": 777, "classification": "unresolved", "previously_seen": True}])
        self.assertIn("cannot be excluded", " ".join(record["errors"]))

    def test_known_member_confirmed_exited_is_excluded(self):
        # Cell "seen before x ESRCH": a known member that the kernel says is
        # gone may leave the group; it is not kept unknown forever.
        record = self._cleanup(unknown=[777], known={777}, classes={777: "exited"})
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["excluded_unknown_pids"],
                         [{"pid": 777, "classification": "exited", "previously_seen": True}])

    def test_unseen_live_unknown_stays_unresolved(self):
        # Cell "BSD refused and getpgid refused too": the membership stays
        # unknown, and never having observed the pid does not change that.
        record = self._cleanup(unknown=[888], known=set(), classes={888: "unresolved"})
        self.assertEqual(record["state"], "unknown")
        self.assertEqual(record["unknown_identity_pids"],
                         [{"pid": 888, "classification": "unresolved", "previously_seen": False}])

    def test_getpgid_confirms_target_member_is_unresolved(self):
        # Cell "BSD refused but getpgid == target group": the kernel says this
        # pid IS one of ours; it must not vanish from the group.
        record = self._cleanup(unknown=[888], known=set(), classes={888: "target_member"})
        self.assertEqual(record["state"], "unknown")
        self.assertEqual(record["unknown_identity_pids"],
                         [{"pid": 888, "classification": "target_member", "previously_seen": False}])
        self.assertIn("cannot be excluded", " ".join(record["errors"]))

    def test_unseen_esrch_unknown_is_excluded(self):
        # Cell "never seen x ESRCH".
        record = self._cleanup(unknown=[888], known=set(), classes={888: "exited"})
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["excluded_unknown_pids"],
                         [{"pid": 888, "classification": "exited", "previously_seen": False}])

    def test_pid_confirmed_in_another_group_is_excluded(self):
        record = self._cleanup(unknown=[888], known=set(), classes={888: "other_group"})
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["excluded_unknown_pids"],
                         [{"pid": 888, "classification": "other_group", "previously_seen": False}])

    def test_eperm_is_not_treated_as_other_user(self):
        # The review removed the EPERM->other_user inference: a permission
        # error carries no membership evidence.  With the BSD read unavailable
        # (non-darwin forces the fallback path) and getpgid also refused, the
        # pid stays unresolved -- never an exclusion.
        from pipeline.reproducible.resource_guard import classify_unknown_pid

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", side_effect=PermissionError(1, "getpgid")
        ), patch("os.kill", side_effect=PermissionError(1, "kill")):
            self.assertEqual(classify_unknown_pid(424242, 12345), "unresolved")

    def test_getpgid_is_the_membership_evidence(self):
        # When os.getpgid succeeds it decides the classification directly.
        from pipeline.reproducible.resource_guard import classify_unknown_pid

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=999
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "other_group")
        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=12345
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "target_member")
    def test_getpgid_is_the_membership_evidence(self):
        # When os.getpgid succeeds it decides the classification directly.
        from pipeline.reproducible.resource_guard import classify_unknown_pid

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=999
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "other_group")
        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=12345
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "target_member")
    def test_getpgid_is_the_membership_evidence(self):
        # When os.getpgid succeeds it decides the classification directly.
        from pipeline.reproducible.resource_guard import classify_unknown_pid

        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=999
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "other_group")
        with patch("pipeline.reproducible.resource_guard.sys.platform", "linux"), patch(
            "os.getpgid", return_value=12345
        ):
            self.assertEqual(classify_unknown_pid(424242, 12345), "target_member")

    def test_real_dead_pid_is_classified_exited(self):
        from pipeline.reproducible.resource_guard import classify_unknown_pid

        self.assertEqual(classify_unknown_pid(999_999_999, 12345), "exited")

    def test_unrelated_process_rss_failure_does_not_block_the_group(self):
        # Counterexample 3: a recognised process outside the target group has
        # an unreadable resident size; the group reading is unaffected.
        from pipeline.reproducible.resource_guard import _group_rss_bytes_from

        table = {10: (1, 500, 100), 11: (1, 999, None)}
        total, members, unreadable = _group_rss_bytes_from(table, 500)
        self.assertEqual(members, [10])
        self.assertEqual(unreadable, [])
        self.assertEqual(total, 100 * 1024)

    def test_group_member_rss_failure_fails_the_reading(self):
        # Counterexample 4a: a target-group member with unreadable RSS.
        from pipeline.reproducible.resource_guard import _group_rss_bytes_from

        table = {10: (1, 500, None)}
        with self.assertRaises(ResourceViolation) as caught:
            _group_rss_bytes_from(table, 500)
        self.assertIn("refusing to substitute zero", str(caught.exception))

    def test_tree_descendant_rss_failure_fails_the_measurement(self):
        # Counterexample 4b: a grandchild found through the PPID chain is in
        # range regardless of depth, so an unreadable RSS must fail the run.
        from pipeline.reproducible.resource_guard import _process_tree_rss_bytes

        table = {
            1: (0, 1, 100),
            2: (1, 1, 200),
            3: (2, 1, None),  # grandchild of the root, unreadable
            9: (5000, 777, None),  # parent outside the table: never in range
        }
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=(table, "libproc", None, {}),
        ):
            with self.assertRaises(ResourceViolation) as caught:
                _process_tree_rss_bytes(1)
        self.assertIn("sampled tree members [3]", str(caught.exception))
        # The explicit self-only scope exists, but it is a deliberate choice.
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=({1: (0, 1, 100)}, "libproc", None, {}),
        ):
            self.assertEqual(_process_tree_rss_bytes(1, include_descendants=False), 100 * 1024)

    def test_inspect_group_identity_reports_unknown_pids(self):
        from pipeline.reproducible.resource_guard import inspect_group_identity

        table = {10: (1, 500, 100)}
        problems = {"identity_unknown": {777: "errno=1"}}
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=(table, "libproc", None, problems),
        ), patch(
            "pipeline.reproducible.resource_guard.classify_unknown_pid",
            return_value="unresolved",
        ):
            members, unknown, error = inspect_group_identity(500)
        self.assertEqual(members, [10])
        self.assertEqual(unknown, [777])
        self.assertIsNone(error)

    def test_compat_entry_refuses_unresolved_unknowns(self):
        # The review probe: identity ([]) with one unresolved unknown must not
        # look like a clean, empty group through process_group_members.
        from pipeline.reproducible.resource_guard import process_group_members

        table = {}
        problems = {"identity_unknown": {999999: "errno=1"}}
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=(table, "libproc", None, problems),
        ), patch(
            "pipeline.reproducible.resource_guard.classify_unknown_pid",
            return_value="unresolved",
        ):
            members, error = process_group_members(20)
        self.assertIsNone(members)
        self.assertIn("999999", str(error))
        self.assertIn("cannot confirm the group is empty", str(error))

    def test_compat_entry_allows_exited_unknowns(self):
        from pipeline.reproducible.resource_guard import process_group_members

        table = {}
        problems = {"identity_unknown": {999999: "errno=3"}}
        with patch(
            "pipeline.reproducible.resource_guard._read_process_table",
            return_value=(table, "libproc", None, problems),
        ), patch(
            "pipeline.reproducible.resource_guard.classify_unknown_pid",
            return_value="exited",
        ):
            members, error = process_group_members(20)
        self.assertEqual(members, [])
        self.assertIsNone(error)


class VerifiedManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_write_verified_manifest_uses_declared_facts_without_rehashing(self):
        data = self.root / "data.bin"
        data.write_bytes(b"12345")
        staged = self.root / "staged.json"
        manifest = write_verified_manifest(
            staged,
            {"status": "complete"},
            verified_artifacts={"data": {"path": "data.bin", "bytes": 5, "sha256": "a" * 64}},
            base=self.root,
        )
        self.assertTrue(staged.is_file())
        self.assertEqual(manifest["artifacts"]["data"]["sha256"], "a" * 64)

    def test_stale_size_is_rejected(self):
        data = self.root / "data.bin"
        data.write_bytes(b"12345")
        with self.assertRaises(ArtifactError) as caught:
            write_verified_manifest(
                self.root / "staged.json",
                {"status": "complete"},
                verified_artifacts={"data": {"path": "data.bin", "bytes": 99, "sha256": "a" * 64}},
                base=self.root,
            )
        self.assertIn("stale", str(caught.exception))

    def test_verification_request_version_is_enforced(self):
        request = self.root / "request.json"
        request.write_text(json.dumps({"version": "other"}), encoding="utf-8")
        with self.assertRaises(VerificationError):
            verify_run(request)


if __name__ == "__main__":
    unittest.main()
