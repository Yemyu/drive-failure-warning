import datetime as dt
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
import subprocess
import sys
import csv
import hashlib
import io
import struct
import zlib
from unittest.mock import patch
from pipeline.reproducible import current

from pipeline.reproducible.current import (
    CURRENT_COLUMNS,
    ReproducibleStopped,
    build_current_features,
    build_labels,
    run_current_research,
)


ROOT = Path(__file__).resolve().parents[1]


def make_panel(path: Path, *, future_smart_boost: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE daily(
            date TEXT NOT NULL, serial_number TEXT NOT NULL, model TEXT NOT NULL,
            capacity_bytes INTEGER, failure INTEGER NOT NULL,
            smart_5_raw INTEGER, smart_9_raw INTEGER, smart_187_raw INTEGER,
            smart_188_raw INTEGER, smart_197_raw INTEGER, smart_198_raw INTEGER,
            PRIMARY KEY(date, serial_number)
        )"""
    )
    # Worker tests use a deliberately small fixture.  Mark it explicitly so
    # the stage contract never treats a metadata-less SQLite file as a real
    # Backblaze panel by accident.
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("build_status", "fixture"),
            ("fixture_profile", "synthetic_fixture_v1"),
            ("fixture_id", "test_panel_v1"),
            ("serial_model_registry_scope", "fixture_rows"),
        ],
    )
    rows = []
    start = dt.date(2023, 1, 1)
    for offset in range(59):
        day = start + dt.timedelta(days=offset)
        date = day.isoformat()
        for serial, failure_day, base in (
            ("disk-a", dt.date(2023, 2, 4), 1),
            ("disk-b", None, 0),
            ("disk-c", dt.date(2023, 2, 12), 0),
        ):
            smart5 = base
            if serial == "disk-c" and future_smart_boost and day >= dt.date(2023, 2, 10):
                smart5 = 999
            rows.append((date, serial, "ST4000DM000", 4_000_787_030_016, int(day == failure_day), smart5, 100 + offset, base, 0, base, 0))
    connection.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()
    connection.close()


def make_manifest_member(root: Path, date: str = "2023-04-12") -> tuple[Path, dict]:
    columns = ["date", "serial_number", "model", "capacity_bytes", "failure", "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerow([date, "disk-a", "ST4000DM000", "-1", "0", "", "10", "0", "0", "0", "0"])
    raw = stream.getvalue().encode()
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    name = f"fixture/{date}.csv"
    name_bytes = name.encode()
    payload = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 8, 8, 0, 0, 0, 0, 0, len(name_bytes), 0) + name_bytes + compressed
    local = root / f"{date}.member"
    local.write_bytes(payload)
    schema_hash = hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    entry = {"name": name, "date": date, "compressed": len(compressed), "size": len(raw), "crc": zlib.crc32(raw) & 0xffffffff, "range_bytes": len(payload), "local": str(local.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(), "expected_counts": {"source_rows": 1, "selected_rows": 1, "failure_rows": 0, "schema_columns": len(columns), "schema_sha256": schema_hash}}
    return local, entry


class ReproduciblePipelineTests(unittest.TestCase):
    def test_panel_prototype_fits_and_closes_selection_before_outcomes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            panel = root / "panel.sqlite"
            make_panel(panel)
            original_labels = current.build_labels
            def checked_labels(*args, **kwargs):
                if kwargs["run_id"] == "repro_eval_h7_v1":
                    self.assertTrue((root / "run/model.json").is_file())
                    manifest = json.loads((root / "run/selection_manifest.json").read_text())
                    self.assertEqual(manifest["status"], "closed")
                    for name, digest in manifest["outputs"].items():
                        self.assertEqual(current._sha256_file(root / "run" / name), digest)
                return original_labels(*args, **kwargs)
            with patch.object(current, "build_labels", side_effect=checked_labels):
                result = run_current_research(
                train_panel=panel,
                eval_panel=panel,
                output_dir=root / "run",
                train_start="2023-01-15",
                train_end="2023-01-28",
                train_cutoff="2023-02-04",
                eval_start="2023-02-05",
                eval_end="2023-02-20",
                eval_cutoff="2023-02-27",
                event_start="2023-02-05",
                event_end="2023-02-13",
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["fit_mode"], "fitted")
            self.assertEqual(result["training"]["model"]["columns"], list(CURRENT_COLUMNS))
            self.assertGreater(result["training"]["model"]["training_rows"], 0)
            self.assertEqual(set(result["evaluation"]["summaries"]), {"current_lr", "smart_nonzero"})
            for summary in result["evaluation"]["summaries"].values():
                self.assertGreater(summary["metrics"]["eligible_rows"], 0)
                self.assertEqual(summary["metrics"]["budget_denominator"], 1000)
            manifest = json.loads((root / "run" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((root / "run" / "result.json").is_file())

    def test_future_deletion_does_not_change_earlier_scores_or_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            original, changed = root / "original.sqlite", root / "changed.sqlite"
            make_panel(original)
            make_panel(changed, future_smart_boost=True)
            connection = sqlite3.connect(changed)
            connection.execute("DELETE FROM daily WHERE date >= '2023-02-10'")
            connection.commit()
            connection.close()
            prep = {"imputation_mean": [0.] * 16, "standardization_mean": [0.] * 16, "standardization_scale": [1.] * 16}
            payload = {"coef": [0.1] * 16, "intercept": 0., "preprocessing": prep}
            selections = []
            score_sets = []
            for index, panel in enumerate((original, changed)):
                features = root / f"features{index}.sqlite"
                build_current_features(panel, features, start="2023-02-05", end="2023-02-09", dataset_end="2023-02-27")
                # The scorer must have no path to a label database.
                real_open = current._open_readonly
                def feature_only(path):
                    self.assertEqual(path, features)
                    return real_open(path)
                with patch.object(current, "_open_readonly", side_effect=feature_only):
                    scores = list(current._rule_and_model_scores(features, payload))
                self.assertTrue(scores)
                self.assertTrue(all("label" not in row and "first_failure_date" not in row for row in scores))
                self.assertTrue(all(row["serial_number"] != "disk-a" for row in scores))
                score_sets.append(scores)
                selections.append([current._select_scores(scores, method) for method in ("current_lr", "smart_nonzero")])
            self.assertEqual(score_sets[0], score_sets[1])
            self.assertEqual(selections[0], selections[1])

    def test_cross_panel_history_and_same_day_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            train, evaluation = root / "train.sqlite", root / "eval.sqlite"
            make_panel(train)
            make_panel(evaluation)
            for path, sql in ((train, "DELETE FROM daily WHERE date >= '2023-02-05'"), (evaluation, "DELETE FROM daily WHERE date < '2023-02-05'")):
                connection = sqlite3.connect(path)
                connection.execute(sql)
                connection.commit()
                connection.close()
            rows = list(current._eligible_current_rows([train, evaluation], dt.date(2023, 2, 5), dt.date(2023, 2, 12)))
            keys = {(r["date"], r["serial_number"]) for r in rows}
            self.assertIn(("2023-02-05", "disk-c"), keys)
            self.assertNotIn(("2023-02-12", "disk-c"), keys)
            self.assertFalse(any(serial == "disk-a" for _, serial in keys))
            early = list(current._eligible_current_rows([train], dt.date(2023, 1, 1), dt.date(2023, 1, 12)))
            self.assertEqual({r["date"] for r in early}, {"2023-01-12"})

    def test_raw_whitelist_and_integer_validation(self):
        row = dict(date="2023-02-05", serial_number="disk-b", model="ST4000DM000", **{f"smart_{field}_raw": 0 for field in current.RAW_FIELDS})
        for bad in (dict(row, label=1), dict(row, smart_5_raw=0.5), dict(row, smart_5_raw=float("inf"))):
            with self.assertRaises(ReproducibleStopped):
                current._current_feature(bad)
        missing = current._current_feature(dict(row, smart_5_raw=None))
        self.assertEqual(missing["smart_5_current_missing"], 1)
        self.assertIsNone(missing["smart_5_current_nonzero"])

    def test_event_opportunities_require_a_decision_inside_score_period(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            panel = Path(directory) / "panel.sqlite"
            make_panel(panel)
            events = current._event_map([panel], dataset_end=dt.date(2023, 2, 28), start=dt.date(2023, 2, 12), end=dt.date(2023, 2, 12), score_start=dt.date(2023, 2, 12), score_end=dt.date(2023, 2, 20))
            self.assertEqual(events["disk-c"]["opportunity"], 0)
            self.assertEqual(events["disk-c"]["opportunity_dates"], [])

    def test_source_manifest_requires_bound_member_facts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            local, entry = make_manifest_member(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"members": [entry]}), encoding="utf-8")
            verified = current.verify_source_manifest(manifest, verify_members=True)
            self.assertEqual(verified["member_count"], 1)
            self.assertEqual(verified["members"][0]["rows_scanned"], 1)
            tampered = dict(entry, range_bytes=entry["range_bytes"] + 1)
            bad = root / "bad.json"
            bad.write_text(json.dumps({"members": [tampered]}), encoding="utf-8")
            with self.assertRaises(ReproducibleStopped):
                current.verify_source_manifest(bad)

    def test_panel_output_and_evidence_cannot_alias(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ReproducibleStopped, "different paths"):
                current.build_panel_from_manifest(root / "missing.json", root / "same", root / "same")

    def test_source_manifest_rejects_schema_hash_and_local_name_mismatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            _local, entry = make_manifest_member(root)
            bad_schema = json.loads(json.dumps(entry))
            bad_schema["expected_counts"]["schema_sha256"] = "0" * 64
            manifest = root / "bad_schema.json"
            manifest.write_text(json.dumps({"members": [bad_schema]}), encoding="utf-8")
            with self.assertRaisesRegex(ReproducibleStopped, "schema hash"):
                current.verify_source_manifest(manifest, verify_members=True)
            bad_name = json.loads(json.dumps(entry))
            bad_name["name"] = "fixture/2023-04-13.csv"
            name_manifest = root / "bad_name.json"
            name_manifest.write_text(json.dumps({"members": [bad_name]}), encoding="utf-8")
            with self.assertRaisesRegex(ReproducibleStopped, "name/date"):
                current.verify_source_manifest(name_manifest)
            bad_local = json.loads(json.dumps(entry))
            bad_local["name"] = "fixture2/2023-04-12.csv"
            bad_local["local"] = entry["local"]
            # The manifest name is valid, but the bounded ZIP local header names
            # a different member; decoding must reject the binding.
            bad_local_manifest = root / "bad_local.json"
            bad_local_manifest.write_text(json.dumps({"members": [bad_local]}), encoding="utf-8")
            with self.assertRaisesRegex(ReproducibleStopped, "filename mismatch"):
                current.verify_source_manifest(bad_local_manifest, verify_members=True)

    def test_cli_rejects_overlap_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            panel = root / "panel.sqlite"
            make_panel(panel)
            output = root / "run"
            args = [sys.executable, str(ROOT / "tools/run_research.py"), "run", "--train-panel", str(panel), "--eval-panel", str(panel), "--output", str(output), "--train-start", "2023-01-15", "--train-end", "2023-02-02", "--train-cutoff", "2023-02-09", "--eval-start", "2023-02-05", "--eval-end", "2023-02-20", "--eval-cutoff", "2023-02-27", "--event-start", "2023-02-05", "--event-end", "2023-02-13"]
            result = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("training outcomes must end", result.stderr)
            self.assertFalse(output.exists())
            args[args.index("--train-end") + 1] = "2023-01-28"
            args[args.index("--train-cutoff") + 1] = "2023-02-04"
            success = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=30)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertTrue((output / "selection_manifest.json").is_file())

    def test_asof_feature_rows_ignore_future_value_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            original, changed = root / "original.sqlite", root / "changed.sqlite"
            make_panel(original)
            make_panel(changed, future_smart_boost=True)
            first = root / "first.sqlite"
            second = root / "second.sqlite"
            build_current_features(original, first, start="2023-02-05", end="2023-02-20", dataset_end="2023-02-27")
            build_current_features(changed, second, start="2023-02-05", end="2023-02-20", dataset_end="2023-02-27")
            with sqlite3.connect(first) as one, sqlite3.connect(second) as two:
                a = one.execute("SELECT decision_date,serial_number," + ",".join(CURRENT_COLUMNS) + " FROM feature_rows WHERE decision_date <= '2023-02-09' ORDER BY decision_date,serial_number").fetchall()
                b = two.execute("SELECT decision_date,serial_number," + ",".join(CURRENT_COLUMNS) + " FROM feature_rows WHERE decision_date <= '2023-02-09' ORDER BY decision_date,serial_number").fetchall()
            self.assertEqual(a, b)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            panel = root / "panel.sqlite"
            make_panel(panel)
            output = root / "features.sqlite"
            build_current_features(panel, output, start="2023-01-15", end="2023-02-02", dataset_end="2023-02-09")
            with self.assertRaises(ReproducibleStopped):
                build_current_features(panel, output, start="2023-01-15", end="2023-02-02", dataset_end="2023-02-09")


if __name__ == "__main__":
    unittest.main()
