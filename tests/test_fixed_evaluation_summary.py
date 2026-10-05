"""Negative tests for the fixed-evaluation summary tool."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from investigation.summarize_fixed_evaluation import (
    FIELD_MAPPING,
    SummaryError,
    _bind_manifest,
    _compare,
    _selftest,
)

ROOT = Path(__file__).resolve().parents[1]


def _stored(**overrides) -> dict[str, object]:
    base = {
        "event_opportunity_total": 126,
        "event_hits": 65,
        "earliest_lead_median": 3.0,
        "earliest_lead_q25": 2.0,
        "earliest_lead_q75": 5.0,
        "early_event_hits_ge2_days": 50,
        "early_event_hits_ge3_days": 37,
        "early_recall_at_opportunity_ge2_days": 50 / 126,
        "early_recall_at_opportunity_ge3_days": 37 / 126,
    }
    base.update(overrides)
    return base


def _interface(**overrides) -> dict[str, object]:
    base = {
        "lead_days_median": 3.0,
        "lead_days_q25": 2.0,
        "lead_days_q75": 5.0,
        "events_lead_ge_2": 50,
        "events_lead_ge_3": 37,
        "lead_days_captured_events": 65,
        "opportunity_recall_lead_ge_2": 50 / 126,
        "opportunity_recall_lead_ge_3": 37 / 126,
    }
    base.update(overrides)
    return base


def _sql(**overrides) -> dict[str, object]:
    base = {
        "opportunity_total": 126,
        "event_hits": 65,
        "lead_median": 3,
        "lead_q25": 2,
        "lead_q75": 5,
        "lead_ge2": 50,
        "lead_ge3": 37,
        "outside_alerts": 1,
    }
    base.update(overrides)
    return base


class CompareTests(unittest.TestCase):
    def test_selftest_example_matches_the_plan(self):
        result = _selftest()
        self.assertEqual(result["status"], "pass")

    def test_every_stored_field_disagrees_when_it_is_changed(self):
        for field in FIELD_MAPPING:
            stored = _stored(**{field: 999})
            checks = _compare(stored, _interface(), _sql())
            self.assertFalse(checks[field]["agree"], f"{field} was not detected")

    def test_integer_and_real_storage_types_agree(self):
        checks = _compare(_stored(), _interface(), _sql())
        for field, item in checks.items():
            self.assertTrue(item["agree"], f"{field}: {item['values']}")

    def test_none_only_agrees_with_none(self):
        checks = _compare(
            _stored(earliest_lead_median=None),
            _interface(lead_days_median=None),
            _sql(lead_median=None),
        )
        self.assertTrue(checks["earliest_lead_median"]["agree"])
        self.assertTrue(checks["earliest_lead_median"]["null_means_no_hit"])
        # A stored number next to a missing recomputation is a disagreement.
        checks = _compare(
            _stored(earliest_lead_median=3.0),
            _interface(lead_days_median=None),
            _sql(lead_median=None),
        )
        self.assertFalse(checks["earliest_lead_median"]["agree"])
        # And the reverse.
        checks = _compare(
            _stored(earliest_lead_median=None),
            _interface(lead_days_median=3.0),
            _sql(lead_median=3),
        )
        self.assertFalse(checks["earliest_lead_median"]["agree"])

    def test_missing_manifest_is_refused(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        with self.assertRaises(SummaryError):
            _bind_manifest(
                Path(self.temporary.name) / "missing.json",
                Path(self.temporary.name) / "db.sqlite",
                "0" * 64,
            )

    def test_manifest_identity_gaps_are_refused(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        database = Path(self.temporary.name) / "db.sqlite"
        database.write_bytes(b"db")
        manifest = Path(self.temporary.name) / "manifest.json"
        base = {"database": str(database), "database_sha256": "a" * 64,
                "status": "complete", "attempt_id": "attempt_001"}

        def write(payload):
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            return manifest

        with self.assertRaises(SummaryError):
            _bind_manifest(write({**base, "database_sha256": None}), database, "a" * 64)
        with self.assertRaises(SummaryError):
            _bind_manifest(write({**base, "database": None}), database, "a" * 64)
        with self.assertRaises(SummaryError):
            _bind_manifest(write({**base, "status": "running"}), database, "a" * 64)
        with self.assertRaises(SummaryError):
            _bind_manifest(write({**base, "attempt_id": None}), database, "a" * 64)
        with self.assertRaises(SummaryError):
            _bind_manifest(write(base), database, "b" * 64)
        with self.assertRaises(SummaryError):
            _bind_manifest(write({**base, "database": "other.sqlite"}), database, "a" * 64)
        binding = _bind_manifest(write(base), database, "a" * 64)
        self.assertEqual(binding["status"], "complete")


class PublishTests(unittest.TestCase):
    """A failed or late write must never leave a consumable success."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "out"
        self.output.mkdir()
        self.body = '{"status": "pass"}'
        self.markdown = "# ok"
        self.runtime = "{}"

    def _publish(self, **kwargs):
        import investigation.summarize_fixed_evaluation as s

        return s._publish_outputs(
            self.output,
            body=self.body,
            markdown=self.markdown,
            runtime_text=self.runtime,
            declared_parts={"summary.json": len(self.body.encode()),
                            "summary.md": len(self.markdown.encode()),
                            "runtime_check.json": len(self.runtime.encode())},
            declared_total=len(self.body.encode()) + len(self.markdown.encode()) + len(self.runtime.encode()),
            deadline=time.monotonic() + 60,
            **kwargs,
        )

    def _assert_no_formal_result(self):
        for name in ("summary.json", "summary.md", "runtime_check.json"):
            self.assertFalse((self.output / name).is_file(), f"{name} survived a failed publish")

    def test_short_write_leaves_no_formal_result(self):
        import investigation.summarize_fixed_evaluation as s

        real_write = Path.write_text

        def half_write(self, data, *a, **k):
            return real_write(self, data[: len(data) // 2], *a, **k)

        with patch.object(Path, "write_text", half_write):
            with self.assertRaises(SummaryError) as caught:
                self._publish()
        self.assertIn("short write", str(caught.exception))
        self._assert_no_formal_result()

    def test_post_write_deadline_leaves_no_formal_result(self):
        import investigation.summarize_fixed_evaluation as s

        real_time = s.time.monotonic
        writes = {"n": 0}

        def drifting():
            # the pre-write check is inside the budget; the post-write check is
            # past the cooperative deadline
            writes["n"] += 1
            return real_time() + (0 if writes["n"] <= 1 else 120)

        with patch.object(s.time, "monotonic", drifting):
            with self.assertRaises(SummaryError) as caught:
                self._publish()
        self.assertIn("budget", str(caught.exception))
        self._assert_no_formal_result()

    def _promote_failure_case(self, fail_at: int):
        """fail_at: which promotion call (1=md, 2=runtime_check, 3=summary.json)."""
        import investigation.summarize_fixed_evaluation as s

        real_replace = s.os.replace
        calls = {"n": 0}

        def failing_replace(src, dst, *a, **k):
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise OSError(28, "No space left on device")
            return real_replace(src, dst, *a, **k)

        with patch.object(s.os, "replace", side_effect=failing_replace):
            with self.assertRaises(SummaryError) as caught:
                self._publish()
        self.assertIn("no consumable success summary", str(caught.exception))
        self.assertFalse((self.output / "summary.json").is_file(),
                         "a success summary survived a failed promotion")
        return calls["n"]

    def test_first_promotion_failure_leaves_no_success_summary(self):
        n = self._promote_failure_case(fail_at=1)
        self.assertEqual(n, 1)
        # auxiliary evidence may survive, the completion marker must not
        self.assertFalse((self.output / "summary.md").is_file())
        self.assertFalse((self.output / "runtime_check.json").is_file())

    def test_second_promotion_failure_keeps_auxiliary_but_no_summary(self):
        self._promote_failure_case(fail_at=2)
        self.assertTrue((self.output / "summary.md").is_file(),
                        "auxiliary evidence may survive a failed promotion")
        self.assertFalse((self.output / "runtime_check.json").is_file())
        self.assertFalse((self.output / "summary.json").is_file())

    def test_last_promotion_failure_keeps_auxiliary_but_no_summary(self):
        self._promote_failure_case(fail_at=3)
        self.assertTrue((self.output / "summary.md").is_file())
        self.assertTrue((self.output / "runtime_check.json").is_file())
        self.assertFalse((self.output / "summary.json").is_file())

    def test_successful_publish_promotes_all_three_files(self):
        self._publish()
        for name in ("summary.json", "summary.md", "runtime_check.json"):
            self.assertTrue((self.output / name).is_file(), name)
            self.assertFalse((self.output / (name + ".staged")).exists())


class RssMeasurementTests(unittest.TestCase):
    """An unusable RSS measurement must never allow a silent pass."""

    def test_unsupported_platform_is_not_measured(self):
        import investigation.summarize_fixed_evaluation as s

        with patch.object(s, "_RSS_UNIT_SCALE", {}):
            rss = s._rss_measurement()
        self.assertEqual(rss["measurement"], "unsupported_platform")

    def test_startup_over_budget_is_unreliable(self):
        import investigation.summarize_fixed_evaluation as s

        with patch.object(s, "_STARTUP_RU_MAXRSS", 10_000_000_000):
            rss = s._rss_measurement()
        self.assertEqual(rss["measurement"], "unreliable")
        self.assertGreater(rss["startup_rss_bytes"], s.RSS_LIMIT_BYTES)

    def test_measured_reading_is_within_budget_on_this_machine(self):
        import investigation.summarize_fixed_evaluation as s

        rss = s._rss_measurement()
        if rss["measurement"] != "measured":
            self.skipTest(f"rss measurement unusable here: {rss['measurement']}")
        self.assertTrue(rss["within_budget"])


if __name__ == "__main__":
    unittest.main()
