import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import run_small_replay as replay


def read_rows(path: Path, table: str, cutoff: str) -> list[tuple]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        return connection.execute(f"SELECT * FROM {table} WHERE decision_date<=? ORDER BY 1,2,3", (cutoff,)).fetchall()
    finally:
        connection.close()


class SmallReplayTests(unittest.TestCase):
    def _run(self, output: Path, cwd: Path | None = None) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(replay.ROOT / "tools/run_small_replay.py"),
                "--output",
                str(output),
            ],
            cwd=cwd or replay.ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_real_fixture_runs_and_preserves_selection_boundary(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            output = Path(directory) / "run"
            self._run(output)
            evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluation["status"], "pass")
            self.assertEqual(evaluation["methods"]["current_lr"]["event_hits"], 1)
            self.assertEqual(evaluation["methods"]["smart_nonzero"]["event_hits"], 2)
            self.assertEqual(evaluation["methods"]["current_lr"]["event_total"], 2)
            self.assertEqual(evaluation["methods"]["current_lr"]["event_opportunity_total"], 2)
            self.assertEqual(evaluation["selection_sha256_before_evaluation"], evaluation["selection_sha256_after_evaluation"])
            self.assertTrue(evaluation["evaluation_opened_after_selection_close"])
            self.assertTrue(evaluation["access_checks"]["score_queries_have_date_upper_bound"])
            self.assertTrue(evaluation["access_checks"]["evaluation_after_selection_close"])
            self.assertEqual(evaluation["access_checks"]["selection_close_marker_count"], 1)
            self.assertEqual(evaluation["access_checks"]["evaluation_selection_close_marker_count"], 1)
            run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["input_sha256_before"], run_manifest["input_sha256_after"])
            self.assertTrue(run_manifest["limits"]["rss_is_observation"])
            self.assertEqual(run_manifest["limits"]["timeout_seconds"], 120.0)
            with sqlite3.connect(output / "selection.sqlite") as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertNotIn("labels", tables)
                self.assertEqual(connection.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()[0], "complete")
            connection.close()

    def test_same_logic_from_other_cwd(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            base = Path(directory)
            first = base / "first"
            second = base / "second"
            self._run(first, cwd=replay.ROOT)
            self._run(second, cwd=Path("/tmp"))
            first_evaluation = json.loads((first / "evaluation.json").read_text(encoding="utf-8"))
            second_evaluation = json.loads((second / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(first_evaluation["methods"], second_evaluation["methods"])
            self.assertEqual(first_evaluation["selection_sha256_before_evaluation"], second_evaluation["selection_sha256_before_evaluation"])

    def test_future_perturbation_cannot_change_asof_features(self):
        rows, _ = replay._parse_source(replay.SOURCE_CSV)
        changed = [dict(row) for row in rows if not (row["serial_number"] == "demo-D" and row["date"] == "2023-01-23")]
        for row in changed:
            if row["serial_number"] == "demo-A" and row["date"] == "2023-01-24":
                row["failure"] = 0
            if row["serial_number"] == "demo-A" and row["date"] >= "2023-01-24":
                row["smart_5_raw"] = 999
        model = replay._load_model()
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            base = Path(directory)
            paths = (base / "original.sqlite", base / "changed.sqlite")
            for path, source_rows in zip(paths, (rows, changed)):
                access = []
                source = replay._source_connection(source_rows, access)
                try:
                    replay._score_and_select(source, source_rows, model, path, access)
                finally:
                    source.close()
            for table in ("features", "scores", "daily", "selections"):
                self.assertEqual(read_rows(paths[0], table, "2023-01-18"), read_rows(paths[1], table, "2023-01-18"), table)

    def test_no_opportunity_event_keeps_event_denominator(self):
        rows, _ = replay._parse_source(replay.SOURCE_CSV)
        model = replay._load_model()
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            database = Path(directory) / "selection.sqlite"
            access = []
            source = replay._source_connection(rows, access)
            try:
                replay._score_and_select(source, rows, model, database, access)
            finally:
                source.close()
            with sqlite3.connect(database) as connection:
                connection.execute("DELETE FROM features WHERE serial_number='demo-A'")
            connection.close()
            access = []
            source = replay._source_connection(rows, access)
            try:
                evaluation = replay._evaluate(database, source, replay.sha256(database), replay._json_read(replay.EXPECTED_JSON), access)
            finally:
                source.close()
            current = evaluation["methods"]["current_lr"]
            self.assertEqual(current["event_total"], 2)
            self.assertEqual(current["event_opportunity_total"], 1)
            self.assertEqual(current["event_hits"], 1)
            self.assertEqual(current["event_recall_at_opportunity"], 1.0)
            self.assertEqual(current["event_summary"][0]["opportunity"], 0)
            self.assertEqual(current["event_summary"][0]["hit"], 0)

    def test_zero_event_denominator_is_null(self):
        rows, _ = replay._parse_source(replay.SOURCE_CSV)
        model = replay._load_model()
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            database = Path(directory) / "selection.sqlite"
            access = []
            source = replay._source_connection(rows, access)
            try:
                replay._score_and_select(source, rows, model, database, access)
            finally:
                source.close()
            access = []
            source = replay._source_connection(rows, access)
            try:
                with patch.object(replay, "EVENT_START", replay.dt.date(2030, 1, 1)), patch.object(replay, "EVENT_END", replay.dt.date(2030, 1, 2)):
                    evaluation = replay._evaluate(database, source, replay.sha256(database), replay._json_read(replay.EXPECTED_JSON), access)
            finally:
                source.close()
            current = evaluation["methods"]["current_lr"]
            self.assertEqual(current["event_total"], 0)
            self.assertEqual(current["event_opportunity_total"], 0)
            self.assertEqual(current["event_hits"], 0)
            self.assertIsNone(current["event_recall_at_opportunity"])
            self.assertEqual(current["event_summary"], [])

    def test_invalid_fixture_and_model_inputs_use_real_readers(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            root = Path(directory)
            source = root / "daily.csv"
            model = root / "current_lr.json"
            expected = root / "expected.json"
            manifest = root / "manifest.json"
            shutil.copy2(replay.SOURCE_CSV, source)
            shutil.copy2(replay.MODEL_JSON, model)
            shutil.copy2(replay.EXPECTED_JSON, expected)
            code_hashes = replay._json_read(replay.FIXTURE_MANIFEST)["code_sha256"]

            def write_manifest():
                value = {
                    "schema_version": 1,
                    "scope": "small_replay_fixture_v1",
                    "sha256": {
                        "daily": replay.sha256(source),
                        "model": replay.sha256(model),
                        "expected": replay.sha256(expected),
                    },
                    "code_sha256": code_hashes,
                }
                manifest.write_text(json.dumps(value), encoding="utf-8")

            with patch.object(replay, "SOURCE_CSV", source), patch.object(replay, "MODEL_JSON", model), patch.object(replay, "EXPECTED_JSON", expected), patch.object(replay, "FIXTURE_MANIFEST", manifest):
                write_manifest()
                source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaisesRegex(replay.ReplayError, "fixture SHA mismatch for daily"):
                    replay._load_fixture_manifest()
                shutil.copy2(replay.ROOT / "examples/small_replay/daily.csv", source)
                write_manifest()
                altered = replay._json_read(model)
                altered["features"] = altered["features"][:-1]
                model.write_text(json.dumps(altered), encoding="utf-8")
                write_manifest()
                replay._load_fixture_manifest()
                with self.assertRaisesRegex(replay.ReplayError, "must contain 16 features"):
                    replay._load_model()
                shutil.copy2(replay.ROOT / "examples/small_replay/current_lr.json", model)
                write_manifest()
                altered = replay._json_read(model)
                altered["features"][0]["standardization_scale"] = 0
                model.write_text(json.dumps(altered), encoding="utf-8")
                write_manifest()
                replay._load_fixture_manifest()
                with self.assertRaisesRegex(replay.ReplayError, "scale is not positive"):
                    replay._load_model()
                shutil.copy2(replay.ROOT / "examples/small_replay/current_lr.json", model)
                write_manifest()
                altered = replay._json_read(model)
                altered["features"][0]["coefficient"] = float("nan")
                model.write_text(json.dumps(altered), encoding="utf-8")
                write_manifest()
                replay._load_fixture_manifest()
                with self.assertRaisesRegex(replay.ReplayError, "non-finite coefficient"):
                    replay._load_model()
                bad_source = root / "bad.csv"
                bad_source.write_text(source.read_text(encoding="utf-8").replace(",4000787030016,0,0,10000,", ",4000787030016,0,-1,10000,", 1), encoding="utf-8")
                with self.assertRaisesRegex(replay.ReplayError, "negative SMART5"):
                    replay._parse_source(bad_source)

    def test_output_boundary_and_failure_after_selection_close(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            base = Path(directory)
            existing = base / "existing"
            existing.mkdir()
            completed = subprocess.run(
                [sys.executable, "-B", str(replay.ROOT / "tools/run_small_replay.py"), "--output", str(existing)],
                cwd=replay.ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            outside = Path("/tmp") / "small-replay-outside"
            completed = subprocess.run(
                [sys.executable, "-B", str(replay.ROOT / "tools/run_small_replay.py"), "--output", str(outside)],
                cwd=replay.ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)

            failed = base / "failed"
            with patch.object(replay, "_evaluate", side_effect=replay.ReplayError("injected evaluation failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        replay.main(["--output", str(failed)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertTrue((failed / "selection.sqlite").is_file())
            self.assertTrue((failed / "error.json").is_file())
            self.assertFalse((failed / "run_manifest.json").exists())
            self.assertEqual((failed / "error.json").read_text(encoding="utf-8").find("injected evaluation failure") >= 0, True)

            oversized = base / "oversized"
            oversized.mkdir()
            (oversized / "existing.bin").write_bytes(b"12345678")
            with patch.object(replay, "MAX_OUTPUT_BYTES", 10):
                with self.assertRaisesRegex(replay.ReplayError, "output exceeds 10 bytes"):
                    replay._write_new(oversized / "new.bin", b"123")

            timed = base / "timed"
            with patch.object(replay, "run", side_effect=lambda *_args, **_kwargs: time.sleep(0.02)):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as stopped:
                        replay.main(["--output", str(timed), "--timeout-seconds", "0.001"])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("exceeded", (timed / "error.json").read_text(encoding="utf-8"))

    def test_completion_gates_reject_missing_selection_close_marker(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            output = Path(directory) / "missing-marker"
            output.mkdir()
            original = replay._score_and_select

            def without_close_marker(*args):
                result = original(*args)
                args[-1][:] = [item for item in args[-1] if item.get("phase") != "selection_closed"]
                return result

            with patch.object(replay, "_score_and_select", side_effect=without_close_marker):
                with self.assertRaisesRegex(replay.ReplayError, "completion access gates failed"):
                    replay.run(output, timeout_seconds=120.0)
            self.assertTrue((output / "selection.sqlite").is_file())
            self.assertFalse((output / "evaluation.json").exists())
            self.assertFalse((output / "summary.md").exists())
            self.assertFalse((output / "run_manifest.json").exists())

    def test_completion_gates_reject_bad_order_or_date_bound(self):
        original_evaluate = replay._evaluate
        mutations = {
            "order": lambda result: result.update(evaluation_opened_after_selection_close=False),
            "date_bound": lambda result: result["access_checks"].update(score_queries_have_date_upper_bound=False),
        }
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    output = Path(directory) / name
                    output.mkdir()

                    def bad_evaluate(*args, _mutate=mutate, **kwargs):
                        result = original_evaluate(*args, **kwargs)
                        _mutate(result)
                        return result

                    with patch.object(replay, "_evaluate", side_effect=bad_evaluate):
                        with self.assertRaisesRegex(replay.ReplayError, "completion access gates failed"):
                            replay.run(output, timeout_seconds=120.0)
                    self.assertTrue((output / "selection.sqlite").is_file())
                    self.assertFalse((output / "evaluation.json").exists())
                    self.assertFalse((output / "run_manifest.json").exists())

    def test_atomic_output_failure_leaves_no_final_or_partial_file(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            root = Path(directory)
            staged_failure = root / "staged.json"

            def interrupted_stage(fd, value):
                with replay.os.fdopen(fd, "wb") as stream:
                    stream.write(value[:1])
                    raise OSError("injected short staging write")

            with patch.object(replay, "_stage_output", side_effect=interrupted_stage):
                with self.assertRaisesRegex(OSError, "short staging write"):
                    replay._write_new(staged_failure, b"complete payload")
            self.assertFalse(staged_failure.exists())
            self.assertEqual(list(root.glob(f".{staged_failure.name}.partial-*")), [])

            publish_race = root / "race.json"
            with patch.object(replay.os, "link", side_effect=FileExistsError):
                with self.assertRaisesRegex(replay.ReplayError, "already exists"):
                    replay._write_new(publish_race, b"complete payload")
            self.assertFalse(publish_race.exists())
            self.assertEqual(list(root.glob(f".{publish_race.name}.partial-*")), [])

    def test_input_binding_is_checked_after_evaluation(self):
        with tempfile.TemporaryDirectory(dir=replay.ROOT) as directory:
            root = Path(directory)
            source = root / "daily.csv"
            model = root / "current_lr.json"
            expected = root / "expected.json"
            manifest = root / "manifest.json"
            shutil.copy2(replay.SOURCE_CSV, source)
            shutil.copy2(replay.MODEL_JSON, model)
            shutil.copy2(replay.EXPECTED_JSON, expected)
            code_hashes = replay._json_read(replay.FIXTURE_MANIFEST)["code_sha256"]
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "scope": "small_replay_fixture_v1",
                "sha256": {"daily": replay.sha256(source), "model": replay.sha256(model), "expected": replay.sha256(expected)},
                "code_sha256": code_hashes,
            }), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            original_evaluate = replay._evaluate

            def mutate_expected(*args, **kwargs):
                result = original_evaluate(*args, **kwargs)
                changed = replay._json_read(expected)
                changed["scope"] = "mutated-after-evaluation"
                expected.write_text(json.dumps(changed), encoding="utf-8")
                return result

            with patch.object(replay, "SOURCE_CSV", source), patch.object(replay, "MODEL_JSON", model), patch.object(replay, "EXPECTED_JSON", expected), patch.object(replay, "FIXTURE_MANIFEST", manifest), patch.object(replay, "_evaluate", side_effect=mutate_expected):
                with self.assertRaisesRegex(replay.ReplayError, "bound input changed during replay"):
                    replay.run(output, timeout_seconds=120.0)
            self.assertTrue((output / "selection.sqlite").is_file())
            self.assertFalse((output / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
