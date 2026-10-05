"""Directed tests for the fixed validation protocol and execution boundaries."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.r_validation import (
    BindingError,
    ProtocolError,
    ReleaseError,
    alert_precision,
    apply_cooldown,
    approve_parameters,
    capacity_for,
    eligible_dates,
    event_window,
    label_for,
    lead_statistics,
    linear_quantile,
    opportunity_events,
    paired_bootstrap_difference,
    build_release,
    require_release,
)
from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.independent_audit import IndependentAuditError, audit_selection
from pipeline.r_validation.protocol_fixture import run_protocol_fixture
from pipeline.r_validation.stage_runner import StageRunnerError, run_stage, validate_profile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "data/derived/simple_baseline_v2/model_results.sqlite"
FULL_PARAMS = ROOT / "examples/small_replay/current_lr.json"


def _approved_copy(directory: Path, mutate=None) -> Path:
    params = json.loads(FULL_PARAMS.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(params)
    path = directory / "params_mutated.json"
    path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class ParameterBindingTests(unittest.TestCase):
    """Model binding: metadata-only or any field change must be refused."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_source_and_full_parameters_are_approved_unchanged(self):
        approved = approve_parameters(SOURCE_DB, FULL_PARAMS,
                                      expected_source_sha256="f" * 64) if False else \
            approve_parameters(SOURCE_DB, FULL_PARAMS)
        self.assertEqual(approved["approval"]["checked_features"], 16)

    def test_wrong_source_sha_is_refused(self):
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, FULL_PARAMS,
                               expected_source_sha256="a" * 64)

    def test_dropping_a_coefficient_is_refused(self):
        path = _approved_copy(self.directory, lambda p: p["features"][0].pop("coefficient"))
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_changed_coefficient_is_refused(self):
        def bump(params):
            params["features"][0]["coefficient"] += 0.5
        path = _approved_copy(self.directory, bump)
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_changed_imputation_is_refused(self):
        def bump(params):
            params["features"][3]["imputation_mean"] += 1.0
        path = _approved_copy(self.directory, bump)
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_changed_scale_is_refused(self):
        def bump(params):
            params["features"][3]["standardization_scale"] += 1.0
        path = _approved_copy(self.directory, bump)
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_reordered_features_are_refused(self):
        def swap(params):
            params["features"][0], params["features"][1] = (
                params["features"][1], params["features"][0]
            )
        path = _approved_copy(self.directory, swap)
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_metadata_only_payload_is_refused(self):
        payload = {
            "model": "current_lr",
            "model_run_status": "reused_v1",
            "feature_count": 16,
            "intercept": -8.312916413119973,
            "features": [{"position": i, "feature_name": f"f{i}"} for i in range(16)],
        }
        path = self.directory / "payload_only.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BindingError):
            approve_parameters(SOURCE_DB, path)

    def test_source_database_change_is_refused(self):
        other = self.directory / "other.sqlite"
        connection = __import__("sqlite3").connect(other)
        connection.execute("CREATE TABLE model_runs (model TEXT)")
        connection.commit()
        connection.close()
        with self.assertRaises(BindingError):
            approve_parameters(other, FULL_PARAMS,
                               expected_source_sha256="f" * 64)


class ProtocolRuleTests(unittest.TestCase):
    """Date, eligibility, cooldown, capacity, label and opportunity rules."""

    def test_event_window_follows_the_protocol(self):
        start, end = event_window(dt.date(2023, 10, 1), dt.date(2023, 12, 24))
        self.assertEqual((start, end), (dt.date(2023, 10, 8), dt.date(2023, 12, 25)))

    def test_bridge_history_makes_the_first_day_eligible(self):
        observed = {dt.date(2023, 9, 18) + dt.timedelta(days=i) for i in range(30)}
        days = eligible_dates(dt.date(2023, 10, 1), dt.date(2023, 10, 5), observed, "S")
        self.assertEqual(days[0], dt.date(2023, 10, 1))

    def test_missing_current_day_is_not_eligible(self):
        observed = {dt.date(2023, 9, 18) + dt.timedelta(days=i) for i in range(30)}
        observed.remove(dt.date(2023, 10, 1))
        self.assertEqual(
            eligible_dates(dt.date(2023, 10, 1), dt.date(2023, 10, 1), observed, "S"), []
        )

    def test_eleven_observations_are_not_enough(self):
        # 09-21..09-30 is 10 days; plus 10-01 the 14-day window sees 11 —
        # one short of the 12-observation minimum.
        observed = {dt.date(2023, 9, 21) + dt.timedelta(days=i) for i in range(10)}
        observed |= {dt.date(2023, 10, 1)}
        self.assertEqual(
            eligible_dates(dt.date(2023, 10, 1), dt.date(2023, 10, 3), observed, "S"), []
        )

    def test_first_failure_removes_later_scoring_days(self):
        observed = {dt.date(2023, 9, 18) + dt.timedelta(days=i) for i in range(30)}
        days = eligible_dates(dt.date(2023, 10, 1), dt.date(2023, 10, 5), observed,
                              "S", first_failure=dt.date(2023, 10, 3))
        self.assertEqual(days, [dt.date(2023, 10, 1), dt.date(2023, 10, 2)])

    def test_capacity_ceil_and_zero(self):
        cap = capacity_for({dt.date(2023, 10, 1): 1, dt.date(2023, 10, 2): 1000,
                            dt.date(2023, 10, 3): 0, dt.date(2023, 10, 4): 999})
        self.assertEqual(cap[dt.date(2023, 10, 1)], 1)
        self.assertEqual(cap[dt.date(2023, 10, 2)], 1)
        self.assertEqual(cap[dt.date(2023, 10, 3)], 0)
        self.assertEqual(cap[dt.date(2023, 10, 4)], 1)

    def test_cooldown_eight_day_gap_is_allowed(self):
        kept, rejected = apply_cooldown([
            (dt.date(2023, 10, 1), "A"),
            (dt.date(2023, 10, 8), "A"),
            (dt.date(2023, 10, 9), "A"),
        ])
        self.assertEqual([d for d, _ in kept], [dt.date(2023, 10, 1), dt.date(2023, 10, 9)])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["days_since"], 7)

    def test_label_same_day_failure_is_refused(self):
        decision = dt.date(2023, 10, 10)
        observed = {decision + dt.timedelta(days=i) for i in range(1, 8)}
        with self.assertRaises(ProtocolError):
            label_for(decision, decision, observed, outcome_cutoff=dt.date(2023, 12, 31))

    def test_label_day_eight_failure_is_negative_for_this_decision(self):
        decision = dt.date(2023, 10, 10)
        observed = {decision + dt.timedelta(days=i) for i in range(1, 8)}
        self.assertEqual(
            label_for(dt.date(2023, 10, 18), decision, observed,
                      outcome_cutoff=dt.date(2023, 12, 31)),
            "negative_observed",
        )

    def test_label_failure_after_outcome_cutoff_is_unknown(self):
        decision = dt.date(2023, 12, 20)
        observed = {decision + dt.timedelta(days=i) for i in range(1, 8)}
        self.assertEqual(
            label_for(dt.date(2023, 12, 27), decision, observed,
                      outcome_cutoff=dt.date(2023, 12, 26)),
            "unknown",
        )

    def test_lead_quantiles_linear(self):
        stats = lead_statistics([1, 3, 7])
        self.assertEqual((stats["median"], stats["q25"], stats["q75"]), (3.0, 2.0, 5.0))

    def test_precision_bounds_keep_unknown_in_denominator(self):
        bounds = alert_precision(alerts=10, known_hits=2, unknown_alerts=3)
        self.assertAlmostEqual(bounds["lower"], 0.2)
        self.assertAlmostEqual(bounds["upper"], 0.5)
        self.assertAlmostEqual(bounds["known_outcome"], 2 / 7)
        self.assertIsNone(alert_precision(alerts=0, known_hits=0, unknown_alerts=0)["lower"])

    def test_linear_quantile_empty_is_none(self):
        self.assertIsNone(linear_quantile([], 0.5))


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(self.temporary.cleanup)
        self.release = Path(self.temporary.name) / "release.json"
        self.lock_digest = "b" * 64
        self.object_id = "4_zefbb636e885eab02543a061a"

    def test_missing_release_is_refused(self):
        with self.assertRaises(ReleaseError):
            require_release(self.release, r0_lock_digest=self.lock_digest,
                            source_object_id=self.object_id, stage="panel")

    def test_valid_release_allows_the_stage(self):
        release = build_release(r0_lock_digest=self.lock_digest,
                                source_object_id=self.object_id,
                                allowed_stages=["panel", "score"])
        self.release.write_text(json.dumps(release), encoding="utf-8")
        accepted = require_release(self.release, r0_lock_digest=self.lock_digest,
                                   source_object_id=self.object_id, stage="panel")
        self.assertEqual(accepted["allowed_stages"], ["panel", "score"])

    def test_changed_release_content_is_refused(self):
        release = build_release(r0_lock_digest=self.lock_digest,
                                source_object_id=self.object_id,
                                allowed_stages=["panel"])
        release["allowed_stages"] = ["panel", "score", "evaluate"]  # tamper
        self.release.write_text(json.dumps(release), encoding="utf-8")
        with self.assertRaises(ReleaseError):
            require_release(self.release, r0_lock_digest=self.lock_digest,
                            source_object_id=self.object_id, stage="panel")

    def test_wrong_stage_is_refused(self):
        release = build_release(r0_lock_digest=self.lock_digest,
                                source_object_id=self.object_id,
                                allowed_stages=["panel"])
        self.release.write_text(json.dumps(release), encoding="utf-8")
        with self.assertRaises(ReleaseError):
            require_release(self.release, r0_lock_digest=self.lock_digest,
                            source_object_id=self.object_id, stage="evaluate")

    def test_wrong_lock_digest_is_refused(self):
        release = build_release(r0_lock_digest=self.lock_digest,
                                source_object_id=self.object_id,
                                allowed_stages=["panel"])
        self.release.write_text(json.dumps(release), encoding="utf-8")
        with self.assertRaises(ReleaseError):
            require_release(self.release, r0_lock_digest="c" * 64,
                            source_object_id=self.object_id, stage="panel")

    def test_unrecognised_approval_role_is_refused_at_build(self):
        with self.assertRaises(ReleaseError):
            build_release(r0_lock_digest=self.lock_digest, source_object_id=self.object_id,
                          allowed_stages=["panel"], signed_by="unapproved-role")

    def test_unknown_stage_is_refused_at_build(self):
        with self.assertRaises(ReleaseError):
            build_release(r0_lock_digest=self.lock_digest, source_object_id=self.object_id,
                          allowed_stages=["q4-download"])


class BootstrapDirectionTests(unittest.TestCase):
    def test_direction_is_current_minus_smart(self):
        result = paired_bootstrap_difference(
            ["A", "B", "C", "D"],
            {"A": 1, "B": 1}, {},  # current captures 2/4
        )
        self.assertEqual(result["direction"], "current_minus_smart")
        self.assertAlmostEqual(result["point_estimate_pp"], 50.0)

    def test_swapping_methods_negates_the_estimate(self):
        events = ["A", "B", "C", "D"]
        forward = paired_bootstrap_difference(events, {"A": 1}, {})
        backward = paired_bootstrap_difference(events, {}, {"A": 1})
        self.assertAlmostEqual(
            forward["point_estimate_pp"], -backward["point_estimate_pp"]
        )

    def test_minimum_valid_replicates_is_enforced(self):
        with self.assertRaises(ProtocolError):
            paired_bootstrap_difference(["A"], {"A": 1}, {}, replicates=10, min_valid=50)

    def test_small_event_set_suppresses_confirmatory_interval(self):
        result = paired_bootstrap_difference(["A", "B"], {"A": 1}, {}, minimum_events=100)
        self.assertTrue(result["insufficient_events"])
        self.assertIsNone(result["interval"])

    def test_full_roster_recomputes_opportunity_denominator(self):
        roster = {
            "A": {"opportunity": 1, "current_hit": 1, "smart_hit": 0},
            "B": {"opportunity": 0, "current_hit": 0, "smart_hit": 0},
            "C": {"opportunity": 1, "current_hit": 0, "smart_hit": 0},
        }
        result = paired_bootstrap_difference(
            ["A", "C"], {"A": 1}, {}, device_flags=roster, minimum_events=0
        )
        self.assertEqual(result["roster_mode"], "all_eligible_devices")
        self.assertEqual(result["roster_devices"], 3)
        self.assertEqual(result["opportunity_events"], 2)
        self.assertEqual(len(result["differences"]), result["valid_replicates"])


class CliFixtureChainTests(unittest.TestCase):
    def test_file_backed_chain_seals_before_outcome_and_survives_future_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            result = run_cli_fixture_chain(Path(temporary), root=ROOT)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["stages"], ["source", "panel", "score", "seal", "evaluate", "audit"])
            self.assertTrue(result["future_perturbation_preserved_selection"])
            self.assertTrue(result["outcome_access_after_seal"])
            self.assertFalse(result["source_access"]["score_receives_full_source_rows"])


class FinalR0BoundaryTests(unittest.TestCase):
    def test_production_profile_requires_release_and_source_adapter(self):
        with self.assertRaises(StageRunnerError):
            validate_profile("production", release_verified=False, source_csv=None)
        with self.assertRaises(StageRunnerError):
            validate_profile("production", release_verified=True, source_csv=None)

    def test_production_runner_requires_verified_source_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            with self.assertRaises(StageRunnerError):
                run_stage(
                    Path(temporary) / "attempt", root=ROOT, profile="production",
                    source_csv=ROOT / "examples/small_replay/daily.csv",
                    model_json=ROOT / "examples/small_replay/current_lr.json",
                    release_verified=True,
                    spec={"score_start": "2023-01-15", "score_end": "2023-01-28",
                          "event_start": "2023-01-22", "event_end": "2023-01-29",
                          "outcome_cutoff": "2023-02-04", "horizon_days": 7},
                )

    def test_resource_guard_rejects_post_write_output_overflow(self):
        from pipeline.reproducible.resource_guard import ResourceLimits
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            with self.assertRaises(StageRunnerError):
                run_stage(
                    Path(temporary) / "attempt", root=ROOT, profile="synthetic",
                    resource_limits=ResourceLimits(
                        max_output_bytes=0, initial_free_bytes=0,
                        min_free_bytes=0, max_elapsed_seconds=120, poll_seconds=0.05,
                    ),
                )

    def test_independent_audit_rejects_score_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            output = Path(temporary)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            corrupt = output / "corrupt.sqlite"
            corrupt.write_bytes((chain / "selection.sqlite").read_bytes())
            connection = __import__("sqlite3").connect(corrupt)
            connection.execute("UPDATE scores SET score=score+1000")
            connection.commit()
            connection.close()
            evaluation = json.loads((chain / "evaluation.json").read_text(encoding="utf-8"))
            with self.assertRaises(IndependentAuditError):
                audit_selection(
                    corrupt, chain / "panel.sqlite", evaluation,
                    output / "audit.json",
                    model=json.loads((chain / "current_lr.json").read_text(encoding="utf-8")),
                )

    def test_independent_audit_rejects_smart_score_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            output = Path(temporary)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            corrupt = output / "smart_corrupt.sqlite"
            corrupt.write_bytes((chain / "selection.sqlite").read_bytes())
            connection = __import__("sqlite3").connect(corrupt)
            connection.execute(
                "UPDATE scores SET score=score+1000 WHERE rowid=("
                "SELECT rowid FROM scores WHERE model='smart_nonzero' LIMIT 1)"
            )
            connection.commit()
            connection.close()
            evaluation = json.loads((chain / "evaluation.json").read_text(encoding="utf-8"))
            with self.assertRaises(IndependentAuditError):
                audit_selection(
                    corrupt, chain / "panel.sqlite", evaluation,
                    output / "audit_smart.json",
                    model=json.loads((chain / "current_lr.json").read_text(encoding="utf-8")),
                )

    def test_independent_audit_rejects_zero_day_or_denominator_tamper(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            output = Path(temporary)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            corrupt = output / "daily_corrupt.sqlite"
            corrupt.write_bytes((chain / "selection.sqlite").read_bytes())
            connection = __import__("sqlite3").connect(corrupt)
            connection.execute(
                "UPDATE daily SET budget_k=999 WHERE rowid=("
                "SELECT rowid FROM daily WHERE model='current_lr' LIMIT 1)"
            )
            connection.commit()
            connection.close()
            evaluation = json.loads((chain / "evaluation.json").read_text(encoding="utf-8"))
            with self.assertRaises(IndependentAuditError):
                audit_selection(
                    corrupt, chain / "panel.sqlite", evaluation,
                    output / "audit_daily.json",
                    model=json.loads((chain / "current_lr.json").read_text(encoding="utf-8")),
                )

    def test_independent_audit_rejects_missing_eligible_feature_key(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            output = Path(temporary)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            corrupt = output / "missing_feature.sqlite"
            corrupt.write_bytes((chain / "selection.sqlite").read_bytes())
            connection = __import__("sqlite3").connect(corrupt)
            key = connection.execute("SELECT decision_date,serial_number FROM features LIMIT 1").fetchone()
            connection.execute("DELETE FROM features WHERE decision_date=? AND serial_number=?", key)
            connection.commit()
            connection.close()
            evaluation = json.loads((chain / "evaluation.json").read_text(encoding="utf-8"))
            with self.assertRaises(IndependentAuditError):
                audit_selection(
                    corrupt, chain / "panel.sqlite", evaluation,
                    output / "audit_missing.json",
                    model=json.loads((chain / "current_lr.json").read_text(encoding="utf-8")),
                )

    def test_asof_reader_does_not_expose_unrestricted_sql(self):
        from tools import run_small_replay as replay
        rows, _ = replay._parse_source(replay.SOURCE_CSV)
        access: list[dict[str, object]] = []
        connection = replay._source_connection(rows, access)
        try:
            reader = replay._asof_reader(connection, access)
            self.assertFalse(hasattr(reader, "execute"))
            with self.assertRaises(AttributeError):
                reader.execute("SELECT failure FROM daily")  # type: ignore[attr-defined]
            visible = reader.rows_until("demo-A", "2023-01-15")
            self.assertTrue(visible)
            self.assertLessEqual(max(row["date"] for row in visible), "2023-01-15")
        finally:
            connection.close()

    def test_production_receipt_alone_cannot_replace_historical_inputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            temporary_path = Path(temporary)
            source = ROOT / "examples/small_replay/daily.csv"
            import hashlib
            receipt = temporary_path / "source_receipt.json"
            receipt.write_text(json.dumps({
                "schema": "r-validation-source-receipt-v1",
                "status": "verified",
                "source_object_id": "local-test-object",
                "transport": "local_verified_copy",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with self.assertRaisesRegex(StageRunnerError, 'bound historical panels'):
                run_stage(
                    temporary_path / "attempt", root=ROOT, profile="production",
                    source_csv=source,
                    model_json=ROOT / "examples/small_replay/current_lr.json",
                    source_receipt=receipt,
                    release_verified=True,
                    spec={"score_start": "2023-01-15", "score_end": "2023-01-28",
                          "event_start": "2023-01-22", "event_end": "2023-01-29",
                          "outcome_cutoff": "2023-02-04",
                          "serials": tuple(f"demo-{letter}" for letter in "ABCDEFGHIJKL")},
                )

    def test_protocol_shaped_fixture_uses_non_demo_roster_and_october_dates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            result = run_protocol_fixture(Path(temporary), root=ROOT)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["source_bytes_differ"])
            self.assertTrue(result["future_perturbation_preserved_selection"])
            self.assertEqual(result["score_start"], "2023-10-01")
            self.assertEqual(result["event_start"], "2023-10-08")
            self.assertTrue(all(name.startswith("synthetic-") for name in result["source_serials"]))
            self.assertTrue(result["legal_smart_decrease_preserved"])


if __name__ == "__main__":
    unittest.main()
