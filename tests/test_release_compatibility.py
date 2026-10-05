"""Approval-record migration and unchanged implementation-lock boundaries.

All approval records here are synthetic.  Historical production records are
not regenerated, copied into the tests, or used to authorise real data access.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import release as approval


ROOT = Path(__file__).resolve().parents[1]


def rehash(record):
    record["release_digest"] = approval._digest(
        {key: value for key, value in record.items() if key != "release_digest"}
    )
    return record


class ReleaseCompatibilityTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "approval.json"
        self.lock = "b" * 64
        self.source = "synthetic-source-object"

    def record(self):
        return approval.build_release(
            r0_lock_digest=self.lock,
            source_object_id=self.source,
            allowed_stages=["panel", "score"],
            note="Synthetic approval only",
        )

    def legacy_record(self):
        record = self.record()
        record.update(schema=approval.LEGACY_RELEASE_SCHEMA,
                      signed_by="historical-review")
        return rehash(record)

    def require(self, record, **overrides):
        self.path.write_text(json.dumps(record), encoding="utf-8")
        parameters = {"r0_lock_digest": self.lock,
                      "source_object_id": self.source, "stage": "panel"}
        parameters.update(overrides)
        return approval.require_release(self.path, **parameters)

    def test_new_record_uses_version_2_and_neutral_role(self):
        record = self.record()
        self.assertEqual(record["schema"], "r-validation-release-v2")
        self.assertEqual(record["signed_by"], "protocol-review")
        self.assertEqual(self.require(record), record)

    def test_rehashed_unrecognised_version_2_role_is_refused(self):
        record = self.record()
        record["signed_by"] = "unapproved-role"
        with self.assertRaisesRegex(approval.ReleaseError, "approval role"):
            self.require(rehash(record))

    def test_unknown_schema_is_refused(self):
        for schema in ("r-validation-release-unknown", [], {}, None, 2):
            with self.subTest(schema=schema):
                record = self.record()
                record["schema"] = schema
                with self.assertRaisesRegex(approval.ReleaseError, "schema"):
                    self.require(rehash(record))

    def test_unregistered_self_consistent_legacy_record_is_refused(self):
        record = self.legacy_record()
        self.assertNotIn(record["release_digest"], approval.LEGACY_RELEASE_DIGESTS)
        with self.assertRaisesRegex(approval.ReleaseError, "registered historical"):
            self.require(record)

    def test_registered_legacy_record_returns_original_fields(self):
        record = self.legacy_record()
        with patch.object(approval, "LEGACY_RELEASE_DIGESTS",
                          frozenset({record["release_digest"]})):
            self.assertEqual(self.require(record), record)
            self.assertEqual(json.loads(self.path.read_text()), record)

    def test_legacy_tampering_still_fails_after_recomputing_self_digest(self):
        original = self.legacy_record()
        changes = {
            "signed_by": "another-review",
            "r0_lock_digest": "c" * 64,
            "source_object_id": "another-source",
            "allowed_stages": ["panel", "score", "evaluate"],
            "note": "Changed historical note",
            "continuation": {"new": "unapproved continuation"},
        }
        with patch.object(approval, "LEGACY_RELEASE_DIGESTS",
                          frozenset({original["release_digest"]})):
            for field, value in changes.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = value
                    with self.assertRaisesRegex(approval.ReleaseError,
                                                "registered historical"):
                        self.require(rehash(changed))

    def test_claiming_registered_digest_does_not_bypass_body_check(self):
        original = self.legacy_record()
        changed = copy.deepcopy(original)
        changed["signed_by"] = "another-review"
        with patch.object(approval, "LEGACY_RELEASE_DIGESTS",
                          frozenset({original["release_digest"]})):
            with self.assertRaisesRegex(approval.ReleaseError, "content changed"):
                self.require(changed)

    def test_both_versions_keep_lock_source_and_stage_boundaries(self):
        records = [self.record(), self.legacy_record()]
        with patch.object(approval, "LEGACY_RELEASE_DIGESTS",
                          frozenset({records[1]["release_digest"]})):
            for record in records:
                for parameters in ({"r0_lock_digest": "c" * 64},
                                   {"source_object_id": "other-source"},
                                   {"source_object_id": ""},
                                   {"stage": "evaluate"}):
                    with self.subTest(schema=record["schema"], overrides=parameters):
                        with self.assertRaises(approval.ReleaseError):
                            self.require(record, **parameters)

    def test_registered_legacy_digest_is_insufficient_for_version_2(self):
        record = self.record()
        record["signed_by"] = "historical-review"
        rehash(record)
        with patch.object(approval, "LEGACY_RELEASE_DIGESTS",
                          frozenset({record["release_digest"]})):
            with self.assertRaisesRegex(approval.ReleaseError, "approval role"):
                self.require(record)


class ImplementationLockBoundaryTests(unittest.TestCase):
    """A self-consistent old lock cannot waive the current implementation SHA."""

    def setUp(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "evidence/protocols/r_validation_v1.md"
        self.protocol.parent.mkdir(parents=True)
        self.protocol.write_text("Synthetic protocol metadata", encoding="utf-8")
        self.config = self.root / "config.json"
        self.config_value = {"version": "r_validation_v1"}
        self.config.write_text(json.dumps(self.config_value), encoding="utf-8")
        self.source_path = self.root / "pipeline/r_validation/release.py"
        self.source_path.parent.mkdir(parents=True)
        self.source_path.write_text("# current synthetic implementation\n", encoding="utf-8")
        self.lock_path = self.root / "lock.json"

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def lock(self, schema):
        return {
            "schema": schema,
            "protocol_source_sha256": self.sha(self.protocol),
            "config_sha256": self.sha(self.config),
            "config_digest": approval._digest(self.config_value),
            # This is deliberately the old implementation's hash.  Neither an
            # intact approval record nor an intact lock self-digest replaces
            # comparison against the currently loaded source files.
            "implementation_sha256": {"pipeline/r_validation/release.py": "f" * 64},
        }

    def save(self, record, digest=approval._digest):
        record["lock_digest"] = digest(record)
        self.lock_path.write_text(json.dumps(record), encoding="utf-8")

    def test_r_old_source_lock_refused_before_historical_inputs(self):
        from tools import run_r_validation as cli

        self.save(self.lock("r0-lock-v1"))
        with patch.object(cli, "ROOT", self.root), \
             patch.object(cli, "CODE_ROOT", self.root), \
             patch.object(cli, "CONFIG", self.config), \
             patch.object(cli, "implementation_files",
                          return_value=("pipeline/r_validation/release.py",)), \
             patch.object(cli, "_project_file",
                          side_effect=AssertionError("historical input opened")) as history:
            with self.assertRaisesRegex(approval.ReleaseError,
                                        "implementation changed after lock creation"):
                cli._verify_lock_binding(self.lock_path)
            history.assert_not_called()

    def test_q1_old_source_lock_refused_before_historical_inputs(self):
        from pipeline.r_validation import q1_binding

        record = self.lock("q1-lock-v1")
        record["config_digest"] = q1_binding._canonical(self.config_value)
        self.save(record, digest=q1_binding._canonical)
        with patch.object(q1_binding, "ROOT", self.root), \
             patch.object(q1_binding, "PROTOCOL_PATH", self.protocol), \
             patch.object(q1_binding, "CONFIG_PATH", self.config), \
             patch.object(q1_binding, "_config", return_value=self.config_value), \
             patch("tools.run_r_validation.implementation_files",
                   return_value=("pipeline/r_validation/release.py",)), \
             patch.object(q1_binding, "_file",
                          side_effect=AssertionError("historical input opened")) as history:
            with self.assertRaisesRegex(approval.ReleaseError,
                                        "implementation dependency closure changed"):
                q1_binding._verify_lock(self.lock_path)
            history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
