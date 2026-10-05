import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pipeline.reproducible import supervisor
from pipeline.reproducible.artifacts import ArtifactError, artifact_facts, verify_manifest, write_json_exclusive, write_manifest
from pipeline.reproducible.current import ReproducibleStopped, build_current_features, verify_cross_quarter_identity
from pipeline.reproducible.input_binding import create_binding_file
from pipeline.reproducible.process_contract import ProcessAccessError, authorize_inputs
from pipeline.reproducible.resource_guard import ResourceGuard, ResourceLimits, ResourceViolation
from pipeline.reproducible.cancellation import CancellationRequested, CancellationToken
from pipeline.reproducible.worker import WorkerError, _fit_with_timeout
from tests.test_reproducible_pipeline import make_panel


ROOT = Path(__file__).resolve().parents[1]


def _identity_panel(path: Path, rows: dict[str, str], *, status: str = "panel_complete", scope: str = "full_source_rows") -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT NOT NULL, "
        "first_date TEXT NOT NULL, last_date TEXT NOT NULL, first_source_member TEXT NOT NULL, source_scope TEXT NOT NULL);"
    )
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", [("build_status", status), ("serial_model_registry_scope", scope)])
    connection.executemany(
        "INSERT INTO serial_model_registry VALUES (?, ?, '2023-01-01', '2023-01-02', 'fixture/2023-01-01.csv', 'full_source_member')",
        rows.items(),
    )
    connection.commit()
    connection.close()


class ArtifactBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_manifest_binds_outputs_and_detects_tamper(self):
        data = self.root / "result.json"
        data.write_text("stable\n", encoding="utf-8")
        manifest_path = self.root / "run" / "complete_manifest.json"
        manifest = write_manifest(manifest_path, {"status": "complete", "scope": "fixture"}, artifacts={"result": data})
        self.assertEqual(manifest["manifest_hash"], json.loads(manifest_path.read_text())["manifest_hash"])
        verified = verify_manifest(manifest_path, expected_status="complete")
        self.assertEqual(verified["verified_artifacts"]["result"]["bytes"], len("stable\n"))
        data.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "artifact changed"):
            verify_manifest(manifest_path)

    def test_manifest_never_overwrites_or_leaves_short_write(self):
        data = self.root / "result.json"
        data.write_text("x", encoding="utf-8")
        manifest_path = self.root / "manifest.json"
        write_manifest(manifest_path, {"status": "complete"}, artifacts={"result": data})
        with self.assertRaisesRegex(ArtifactError, "overwrite"):
            write_manifest(manifest_path, {"status": "complete"}, artifacts={"result": data})
        short = self.root / "short.json"
        with self.assertRaisesRegex(ArtifactError, "short"):
            write_manifest(short, {"status": "complete"}, _write_text=lambda _handle, text: len(text) - 1)
        self.assertFalse(short.exists())
        self.assertFalse((self.root / ".short.json.partial").exists())

    def test_manifest_competing_writer_is_rejected_after_preflight(self):
        target = self.root / "collision.json"

        def competing_writer(handle, text):
            target.write_text('{"owner":"competing"}\n', encoding="utf-8")
            return handle.write(text)

        with self.assertRaisesRegex(ArtifactError, "overwrite"):
            write_json_exclusive(target, {"owner": "publisher"}, _write_text=competing_writer)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["owner"], "competing")
        self.assertFalse((self.root / ".collision.json.partial").exists())


class ProcessBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        for name in ("panel.sqlite", "history.sqlite", "config.json", "q3.sqlite", "outcomes.sqlite", "selection.json", "scores.json", "model.json"):
            (self.root / name).write_text(name, encoding="utf-8")

    def test_process_group_empty_or_failed_inspection_is_not_clean(self):
        empty = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(supervisor.subprocess, "run", return_value=empty):
            with self.assertRaisesRegex(supervisor.SupervisorError, "empty process table"):
                supervisor._process_group_members(12345)

        with patch.object(supervisor.os, "killpg") as killpg, patch.object(
            supervisor, "_inspect_group_identity", return_value=(None, [], "injected ps failure")
        ):
            with self.assertRaisesRegex(supervisor.SupervisorError, "cleanup"):
                supervisor._terminate_group(12345)
            self.assertEqual(
                [call.args[1] for call in killpg.call_args_list],
                [signal.SIGTERM, signal.SIGKILL],
            )

    def test_role_contract_rejects_future_or_outcome_inputs(self):
        with self.assertRaisesRegex(ProcessAccessError, "q3_panel"):
            authorize_inputs("train", {
                "train_panel": self.root / "panel.sqlite",
                "config": self.root / "config.json",
                "q3_panel": self.root / "q3.sqlite",
            })
        with self.assertRaisesRegex(ProcessAccessError, "eval_labels"):
            authorize_inputs("score", {
                "eval_panel": self.root / "panel.sqlite",
                "history_panel": self.root / "history.sqlite",
                "model": self.root / "model.json",
                "config": self.root / "config.json",
                "eval_labels": self.root / "outcomes.sqlite",
            })
        with self.assertRaisesRegex(ProcessAccessError, "missing required"):
            authorize_inputs("build_q3", {"q3_receipt": self.root / "config.json", "q3_archive": self.root / "config.json"})

    def test_real_cli_child_accepts_only_declared_role_inputs(self):
        bindings = self.root / "train_bindings.json"
        bindings.write_text(json.dumps({
            "train_panel": str(self.root / "panel.sqlite"),
            "config": str(self.root / "config.json"),
        }), encoding="utf-8")
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "check-access", "--role", "train", "--bindings", str(bindings)]
        accepted = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(json.loads(accepted.stdout)["role"], "train")
        bindings.write_text(json.dumps({
            "train_panel": str(self.root / "panel.sqlite"),
            "config": str(self.root / "config.json"),
            "q3_panel": str(self.root / "q3.sqlite"),
        }), encoding="utf-8")
        rejected = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("q3_panel", rejected.stderr)


class IdentityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_cross_quarter_identity_passes_and_reports_hashes(self):
        prior = self.root / "prior.sqlite"
        q3 = self.root / "q3.sqlite"
        _identity_panel(prior, {"disk-a": "ST4000DM000", "disk-b": "OTHER"})
        _identity_panel(q3, {"disk-a": "ST4000DM000", "disk-c": "ST4000DM000"})
        result = verify_cross_quarter_identity(prior, q3)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["overlap_serials"], 1)
        self.assertEqual(result["conflict_count"], 0)
        self.assertEqual(len(result["prior_registry_sha256"]), 64)

    def test_cross_quarter_identity_conflict_stops_before_publication(self):
        prior = self.root / "prior.sqlite"
        q3 = self.root / "q3.sqlite"
        _identity_panel(prior, {"disk-a": "OTHER"})
        _identity_panel(q3, {"disk-a": "ST4000DM000"})
        with self.assertRaisesRegex(ReproducibleStopped, "identity conflict"):
            verify_cross_quarter_identity(prior, q3)


class TrainWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_train_worker_runs_in_a_child_and_persists_sample_contract(self):
        panel = self.root / "train.sqlite"
        config = self.root / "config.json"
        spec = self.root / "train_spec.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1", "model": "current_lr", "horizon_days": 7}), encoding="utf-8")
        binding = self.root / "train.binding.json"
        create_binding_file(binding, "train", config, {"train_panel": panel, "config": config})
        spec.write_text(json.dumps({
            "run_id": "worker_train_h7_v1",
            "bindings": {"train_panel": str(panel), "config": str(config)},
            "binding_file": str(binding),
            "train_start": "2023-01-15",
            "train_end": "2023-01-28",
            "train_cutoff": "2023-02-04",
        }), encoding="utf-8")
        output = self.root / "train_output"
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "train", "--spec", str(spec), "--output", str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = verify_manifest(output / "train_manifest.json", expected_status="complete")
        self.assertEqual(manifest["role"], "train")
        self.assertEqual(manifest["input_contract"]["role"], "train")
        self.assertNotIn("q3_panel", manifest["input_contract"]["inputs"])
        fit_check = manifest["training"]["fit_check"]
        self.assertEqual(fit_check["version"], "training-check-v2")
        self.assertEqual(fit_check["panel_reference"]["version"], "panel-reference-v1")
        self.assertEqual(fit_check["panel_reference"]["status"], "pass")
        self.assertIn("package_manifest_fact", fit_check)
        samples = json.loads((output / "train_samples.json").read_text(encoding="utf-8"))
        self.assertEqual(len(samples["keys"]), len(samples["labels"]))
        self.assertEqual(len(samples["labels"]), len(samples["weights"]))
        self.assertEqual(len(samples["labels"]), manifest["training"]["model"]["training_rows"])
        self.assertTrue((output / "model.json").is_file())
        self.assertTrue((output / "environment.json").is_file())
        self.assertTrue((output / "qualification_calendar.json").is_file())
        self.assertEqual(len(manifest["training"]["features"]["qualification_calendar"]["days"]), 14)

    def test_train_worker_rejects_q3_binding_before_creating_output(self):
        panel = self.root / "train.sqlite"
        config = self.root / "config.json"
        q3 = self.root / "q3.sqlite"
        spec = self.root / "bad_spec.json"
        make_panel(panel)
        config.write_text("{}", encoding="utf-8")
        q3.write_text("placeholder", encoding="utf-8")
        spec.write_text(json.dumps({
            "bindings": {"train_panel": str(panel), "config": str(config), "q3_panel": str(q3)},
            "train_start": "2023-01-15", "train_end": "2023-01-28", "train_cutoff": "2023-02-04",
        }), encoding="utf-8")
        output = self.root / "rejected"
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "train", "--spec", str(spec), "--output", str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("q3_panel", result.stderr)
        self.assertFalse(output.exists())

    def test_score_worker_persists_full_scores_and_closes_both_selections(self):
        panel = self.root / "eval.sqlite"
        config = self.root / "config.json"
        model = self.root / "model.json"
        spec = self.root / "score_spec.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1", "budget_denominator": 1000}), encoding="utf-8")
        model.write_text(json.dumps({
            "model": "current_lr", "fit_mode": "fitted", "columns": [
                "smart_5_current_log1p", "smart_5_current_missing", "smart_5_current_nonzero",
                "smart_9_current_log1p", "smart_9_current_missing", "smart_187_current_log1p",
                "smart_187_current_missing", "smart_187_current_nonzero", "smart_188_current_nonzero",
                "smart_188_current_missing", "smart_197_current_log1p", "smart_197_current_missing",
                "smart_197_current_nonzero", "smart_198_current_log1p", "smart_198_current_missing",
                "smart_198_current_nonzero",
            ], "coef": [0.0] * 16, "intercept": 0.0,
            "preprocessing": {"imputation_mean": [0.0] * 16, "standardization_mean": [0.0] * 16, "standardization_scale": [1.0] * 16},
        }), encoding="utf-8")
        binding = self.root / "score.binding.json"
        create_binding_file(binding, "score", config, {"eval_panel": panel, "history_panel": panel, "model": model, "config": config})
        spec.write_text(json.dumps({
            "bindings": {"eval_panel": str(panel), "history_panel": str(panel), "model": str(model), "config": str(config)},
            "binding_file": str(binding),
            "eval_start": "2023-02-05", "eval_end": "2023-02-09", "eval_cutoff": "2023-02-27",
        }), encoding="utf-8")
        output = self.root / "score_output"
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "score", "--spec", str(spec), "--output", str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = verify_manifest(output / "score_manifest.json", expected_status="complete")
        self.assertEqual(manifest["input_contract"]["role"], "score")
        self.assertGreater(manifest["score_rows"], 0)
        selection = verify_manifest(output / "selection_manifest.json", expected_status="closed")
        self.assertEqual(selection["score_rows"], manifest["score_rows"])
        with sqlite3.connect(output / "scores.sqlite") as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0], manifest["score_rows"])
        self.assertTrue((output / "current_lr_selection.json").is_file())
        self.assertTrue((output / "smart_nonzero_selection.json").is_file())
        self.assertTrue((output / "environment.json").is_file())
        self.assertTrue((output / "qualification_calendar.json").is_file())
        self.assertEqual(len(manifest["features"]["qualification_calendar"]["days"]), 5)

    def test_score_worker_rejects_outcome_binding_before_creating_output(self):
        panel = self.root / "eval.sqlite"
        config = self.root / "config.json"
        model = self.root / "model.json"
        outcomes = self.root / "outcomes.sqlite"
        spec = self.root / "bad_score_spec.json"
        make_panel(panel)
        for path, payload in ((config, "{}"), (model, "{}"), (outcomes, "future")):
            path.write_text(payload, encoding="utf-8")
        spec.write_text(json.dumps({
            "bindings": {"eval_panel": str(panel), "history_panel": str(panel), "model": str(model), "config": str(config), "eval_labels": str(outcomes)},
            "eval_start": "2023-02-05", "eval_end": "2023-02-09", "eval_cutoff": "2023-02-27",
        }), encoding="utf-8")
        output = self.root / "rejected_score"
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "score", "--spec", str(spec), "--output", str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("eval_labels", result.stderr)
        self.assertFalse(output.exists())

    def test_evaluate_worker_verifies_closed_selection_before_reading_outcomes(self):
        panel = self.root / "eval.sqlite"
        config = self.root / "config.json"
        model = self.root / "model.json"
        score_spec = self.root / "score_spec.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1", "budget_denominator": 1000}), encoding="utf-8")
        model.write_text(json.dumps({
            "model": "current_lr", "fit_mode": "fitted", "columns": [
                "smart_5_current_log1p", "smart_5_current_missing", "smart_5_current_nonzero",
                "smart_9_current_log1p", "smart_9_current_missing", "smart_187_current_log1p",
                "smart_187_current_missing", "smart_187_current_nonzero", "smart_188_current_nonzero",
                "smart_188_current_missing", "smart_197_current_log1p", "smart_197_current_missing",
                "smart_197_current_nonzero", "smart_198_current_log1p", "smart_198_current_missing",
                "smart_198_current_nonzero",
            ], "coef": [0.0] * 16, "intercept": 0.0,
            "preprocessing": {"imputation_mean": [0.0] * 16, "standardization_mean": [0.0] * 16, "standardization_scale": [1.0] * 16},
        }), encoding="utf-8")
        score_binding = self.root / "score.binding.json"
        create_binding_file(score_binding, "score", config, {"eval_panel": panel, "history_panel": panel, "model": model, "config": config})
        score_spec.write_text(json.dumps({
            "bindings": {"eval_panel": str(panel), "history_panel": str(panel), "model": str(model), "config": str(config)},
            "binding_file": str(self.root / "score.binding.json"),
            "eval_start": "2023-02-05", "eval_end": "2023-02-09", "eval_cutoff": "2023-02-27",
        }), encoding="utf-8")
        score_output = self.root / "score_output"
        score_command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "score", "--spec", str(score_spec), "--output", str(score_output)]
        scored = subprocess.run(score_command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(scored.returncode, 0, scored.stderr)
        eval_spec = self.root / "eval_spec.json"
        eval_binding = self.root / "evaluate.binding.json"
        create_binding_file(
            eval_binding,
            "evaluate",
            config,
            {
                "eval_panel": panel,
                "history_panel": panel,
                "selection_manifest": score_output / "selection_manifest.json",
                "score_manifest": score_output / "score_manifest.json",
                "config": config,
            },
        )
        eval_spec.write_text(json.dumps({
            "run_id": "worker_eval_h7_v1",
            "bindings": {
                "eval_panel": str(panel), "history_panel": str(panel),
                "selection_manifest": str(score_output / "selection_manifest.json"),
                "score_manifest": str(score_output / "score_manifest.json"), "config": str(config),
            },
            "binding_file": str(eval_binding),
            "eval_start": "2023-02-05", "eval_end": "2023-02-09", "eval_cutoff": "2023-02-27",
            "event_start": "2023-02-05", "event_end": "2023-02-13",
        }), encoding="utf-8")
        output = self.root / "evaluation_output"
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "worker", "--role", "evaluate", "--spec", str(eval_spec), "--output", str(output)]
        evaluated = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        manifest = verify_manifest(output / "evaluation_manifest.json", expected_status="complete")
        self.assertEqual(manifest["input_contract"]["role"], "evaluate")
        self.assertEqual(manifest["selection_manifest_hash"], json.loads((score_output / "selection_manifest.json").read_text())["manifest_hash"])
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["scope"], "evaluation_metrics_v2")
        self.assertEqual(metrics["event_window"], {"start": "2023-02-05", "end": "2023-02-13"})
        for method in ("current_lr", "smart_nonzero"):
            self.assertIn("lead_days_median", metrics["methods"][method])
            self.assertIn("opportunity_recall_lead_ge_2", metrics["methods"][method])
            self.assertIn("outside_main_event_alerts", metrics["methods"][method])
        self.assertTrue((output / "work/eval_labels.sqlite").is_file())
        self.assertTrue((output / "current_lr_alerts.json").is_file())
        self.assertTrue((output / "smart_nonzero_alerts.json").is_file())

    def test_parent_isolated_run_verifies_each_child_before_next_stage(self):
        panel = self.root / "panel.sqlite"
        config = self.root / "config.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1", "pipeline": "repro-current-v1"}), encoding="utf-8")
        output = self.root / "isolated"
        command = [
            sys.executable, str(ROOT / "tools/run_research.py"), "isolated-run",
            "--train-panel", str(panel), "--eval-panel", str(panel), "--config", str(config),
            "--output", str(output),
            "--train-start", "2023-01-15", "--train-end", "2023-01-28", "--train-cutoff", "2023-02-04",
            "--eval-start", "2023-02-05", "--eval-end", "2023-02-09", "--eval-cutoff", "2023-02-27",
            "--event-start", "2023-02-05", "--event-end", "2023-02-13",
        ]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = verify_manifest(output / "isolated_run_manifest.json", expected_status="complete")
        self.assertEqual(set(manifest["stages"]), {"train", "score", "evaluate"})
        self.assertLessEqual(manifest["resource_snapshot_before_publish"]["owned_bytes"], manifest["resource_limits"]["max_output_bytes"])
        self.assertTrue((output / "logs/train.log").is_file())
        self.assertTrue((output / "logs/score.log").is_file())
        self.assertTrue((output / "logs/evaluate.log").is_file())
        self.assertTrue((output / "stages/evaluate/evaluation_manifest.json").is_file())
        self.assertTrue((output / "environment.json").is_file())
        self.assertEqual(manifest["environment"]["sha256"], artifact_facts(output / "environment.json")["sha256"])


class ResourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_output_limit_is_observed_and_reported(self):
        output = self.root / "run"
        output.mkdir()
        (output / "one.bin").write_bytes(b"x")
        guard = ResourceGuard(ROOT, output, ResourceLimits(max_output_bytes=0, min_free_bytes=0, poll_seconds=0.01))
        with self.assertRaisesRegex(ResourceViolation, "output"):
            guard.check_or_raise()
        self.assertIn("owned_bytes", guard.last_snapshot)
        guard.stop()

    def test_isolated_run_stops_and_publishes_failure_when_output_budget_is_zero(self):
        panel = self.root / "panel.sqlite"
        config = self.root / "config.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1"}), encoding="utf-8")
        output = self.root / "limited"
        command = [
            sys.executable, str(ROOT / "tools/run_research.py"), "isolated-run",
            "--train-panel", str(panel), "--eval-panel", str(panel), "--config", str(config),
            "--output", str(output),
            "--train-start", "2023-01-15", "--train-end", "2023-01-28", "--train-cutoff", "2023-02-04",
            "--eval-start", "2023-02-05", "--eval-end", "2023-02-09", "--eval-cutoff", "2023-02-27",
            "--event-start", "2023-02-05", "--event-end", "2023-02-13",
            "--max-output-mib", "0", "--poll-seconds", "0.01",
        ]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("resource", result.stderr.lower())
        self.assertTrue((output / "failure.json").is_file())
        self.assertFalse((output / "isolated_run_manifest.json").exists())

    def test_fit_timeout_stops_before_function_returns(self):
        token = CancellationToken()
        token.install()
        try:
            with self.assertRaisesRegex(WorkerError, "fit timed out"):
                _fit_with_timeout(lambda: time.sleep(0.2), 0.02, token)
        finally:
            token.restore()


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_sql_stage_cancellation_keeps_partial_and_no_final_database(self):
        panel = self.root / "panel.sqlite"
        output = self.root / "features.sqlite"
        make_panel(panel)
        with self.assertRaisesRegex(CancellationRequested, "eligibility query cancelled"):
            build_current_features(
                panel,
                output,
                start="2023-01-15",
                end="2023-02-09",
                dataset_end="2023-02-27",
                progress_callback=lambda: 1,
            )
        self.assertFalse(output.exists())
        self.assertTrue(Path(str(output) + ".partial").exists())


class QualificationCalendarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_calendar_keeps_zero_eligible_days_and_is_persisted(self):
        panel = self.root / "panel.sqlite"
        history = self.root / "history.sqlite"
        output = self.root / "features.sqlite"
        make_panel(panel)
        make_panel(history)
        result = build_current_features(
            panel,
            output,
            start="2023-01-01",
            end="2023-01-03",
            dataset_end="2023-01-03",
            history_panels=[history],
        )
        days = result["qualification_calendar"]["days"]
        self.assertEqual([row["date"] for row in days], ["2023-01-01", "2023-01-02", "2023-01-03"])
        self.assertEqual([row["eligible_rows"] for row in days], [0, 0, 0])
        with sqlite3.connect(output) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM qualification_calendar").fetchone()[0], 3)
            source_hashes = json.loads(connection.execute("SELECT value FROM metadata WHERE key='source_panel_sha256'").fetchone()[0])
        self.assertEqual(len(source_hashes), 2)


if __name__ == "__main__":
    unittest.main()
