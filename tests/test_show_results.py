import contextlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import re
from urllib.parse import unquote

from tools.show_results import (
    Binding,
    CAPACITY_LABELS,
    DEFAULT_BINDINGS,
    EXPECTED_ALERTS,
    EXPECTED_DENOMINATORS,
    EXPECTED_EVENT_HITS,
    EXPECTED_MODELS,
    PROJECT_ROOT,
    ReaderError,
    ResultReader,
    _open_readonly,
    _output_path,
    _require_model_metrics,
    _validate_bindings,
    main,
    render_markdown,
    validate_capacity_snapshot,
)


def _empty_capacity_connection(metric_rows):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE metrics (model TEXT, denominator INTEGER, capacity_label TEXT);
        CREATE TABLE selections (model TEXT, denominator INTEGER);
        CREATE TABLE daily (model TEXT, denominator INTEGER);
        CREATE TABLE alert_outcomes (model TEXT, denominator INTEGER);
        CREATE TABLE event_summary (model TEXT, denominator INTEGER);
        """
    )
    connection.executemany("INSERT INTO metrics VALUES (?, ?, ?)", metric_rows)
    return connection


def _valid_capacity_connection(wrong_event_path=None):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE metrics (
            model TEXT, denominator INTEGER, capacity_label TEXT,
            eligible_device_days INTEGER, alerts INTEGER, unique_devices INTEGER,
            unused_slots INTEGER, known_hit_alerts INTEGER, known_no_hit_alerts INTEGER,
            unknown_alerts INTEGER, unknown_ratio REAL, precision_lower_bound REAL,
            precision_upper_bound REAL, known_outcome_precision REAL, event_total INTEGER,
            opportunity_total INTEGER, event_hits INTEGER, event_recall REAL,
            early_event_hits_ge2 INTEGER, early_event_hits_ge3 INTEGER,
            earliest_lead_count INTEGER, earliest_lead_median REAL,
            earliest_lead_q25 REAL, earliest_lead_q75 REAL, repeated_alert_devices INTEGER,
            max_alerts_per_device INTEGER, minimum_alert_gap_days INTEGER
        );
        CREATE TABLE selections (model TEXT, denominator INTEGER);
        CREATE TABLE daily (model TEXT, denominator INTEGER);
        CREATE TABLE alert_outcomes (model TEXT, denominator INTEGER);
        CREATE TABLE event_summary (model TEXT, denominator INTEGER, event_key TEXT, opportunity INTEGER, hit INTEGER);
        """
    )
    for model in EXPECTED_MODELS:
        for denominator in EXPECTED_DENOMINATORS:
            alerts = EXPECTED_ALERTS[denominator]
            hits = EXPECTED_EVENT_HITS[model][denominator]
            event_total = 125 if wrong_event_path == (model, denominator) else 126
            connection.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model, denominator, CAPACITY_LABELS[denominator], 1, alerts, 1, 0,
                    0, alerts, 0, 0.0, 0.0, 1.0, None, event_total, 126, hits,
                    hits / 126.0, 0, 0, 0, None, None, None, 0, 1, 1,
                ),
            )
            connection.executemany("INSERT INTO selections VALUES (?, ?)", [(model, denominator)] * alerts)
            connection.executemany("INSERT INTO daily VALUES (?, ?)", [(model, denominator)] * 85)
            connection.executemany("INSERT INTO alert_outcomes VALUES (?, ?)", [(model, denominator)] * alerts)
            connection.executemany(
                "INSERT INTO event_summary VALUES (?, ?, ?, ?, ?)",
                [(model, denominator, f"{model}:{denominator}:{index}", 1, int(index < hits)) for index in range(126)],
            )
    connection.commit()
    return connection


class ResultReaderTests(unittest.TestCase):
    def test_export_links_follow_destination(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            target = Path(directory) / "overview.md"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output", str(target)]), 0)
            for link in re.findall(r"\]\(([^)]+)\)", target.read_text()):
                self.assertTrue((target.parent / unquote(link)).is_file(), link)

    def test_export_refuses_file_created_after_precheck(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            target = Path(directory) / "overview.md"
            def race(root, raw):
                result = _output_path(root, raw)
                result.write_bytes(b"concurrent output")
                return result
            with patch("tools.show_results._output_path", side_effect=race):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        main(["--output", str(target)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(target.read_bytes(), b"concurrent output")

    @classmethod
    def setUpClass(cls):
        cls.snapshot = ResultReader().read()

    def test_real_snapshot_has_frozen_values_and_input_hashes(self):
        self.assertEqual(self.snapshot.capacity_metrics[("current_lr", 1000)]["event_hits"], 65)
        self.assertEqual(self.snapshot.capacity_metrics[("history_lr", 1000)]["event_hits"], 72)
        self.assertEqual(self.snapshot.capacity_metrics[("history_hgb_v1", 500)]["event_hits"], 75)
        self.assertEqual(
            self.snapshot.input_sha256,
            {key: binding.sha256 for key, binding in DEFAULT_BINDINGS.items()},
        )

    def test_render_contains_scope_units_and_links(self):
        document = render_markdown(self.snapshot)
        for fragment in (
            "2023-07-01 至 09-23",
            "126 个有预警机会的事件",
            "1509 / 65/126",
            "history_hgb_v1",
            "Q3_RESEARCH_RESULT_BRIEF.md",
            "容量结果审查",
            "不是新的独立泛化测试",
        ):
            self.assertIn(fragment, document)

    def test_default_reader_does_not_write_overview(self):
        target = PROJECT_ROOT / "reports" / "RESULTS_OVERVIEW.md"
        before = target.stat().st_mtime_ns if target.exists() else None
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([]), 0)
        after = target.stat().st_mtime_ns if target.exists() else None
        self.assertEqual(before, after)
        self.assertIn("# 硬盘故障提前预警：固定结果总览", output.getvalue())

    def test_reader_accepts_an_injected_bound_manifest_path(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            source = PROJECT_ROOT / DEFAULT_BINDINGS["q3_manifest"].relative_path
            injected = Path(directory) / "q3_manifest.json"
            injected.symlink_to(source)
            bindings = dict(DEFAULT_BINDINGS)
            bindings["q3_manifest"] = Binding(
                injected.relative_to(PROJECT_ROOT).as_posix(),
                DEFAULT_BINDINGS["q3_manifest"].sha256,
                "json",
            )
            snapshot = ResultReader(PROJECT_ROOT, bindings).read()
            self.assertEqual(snapshot.capacity_metrics[("history_lr", 1000)]["event_hits"], 72)

    def test_missing_and_tampered_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ReaderError, "missing bound input"):
                _validate_bindings(root, {"x": Binding("missing.json", "0" * 64, "json")})
            path = root / "input.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ReaderError, "SHA-256 mismatch"):
                _validate_bindings(root, {"x": Binding("input.json", "0" * 64, "json")})

    def test_binding_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(ReaderError, "escapes project root"):
                _validate_bindings(root, {"x": Binding("../outside.json", "0" * 64, "json")})

    def test_missing_model_and_wrong_denominator_are_rejected(self):
        expected = [(model, denominator, "0.1%") for model in ("current_lr", "history_lr") for denominator in (2000, 1000, 500)]
        connection = _empty_capacity_connection(expected)
        try:
            with self.assertRaisesRegex(ReaderError, "paths differ"):
                validate_capacity_snapshot(connection)
        finally:
            connection.close()

    def test_wrong_event_denominator_is_rejected(self):
        connection = _valid_capacity_connection(("current_lr", 1000))
        try:
            with self.assertRaisesRegex(ReaderError, "event denominator differs"):
                validate_capacity_snapshot(connection)
        finally:
            connection.close()

        rows = [(model, denominator, "0.1%") for model in ("current_lr", "history_lr", "history_hgb_v1") for denominator in (2000, 1000)]
        rows.append(("history_hgb_v1", 999, "wrong"))
        connection = _empty_capacity_connection(rows)
        try:
            with self.assertRaisesRegex(ReaderError, "paths differ"):
                validate_capacity_snapshot(connection)
        finally:
            connection.close()

    def test_q3_missing_model_is_rejected(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE model_metrics (model TEXT);
            CREATE TABLE model_event_summary (model TEXT);
            INSERT INTO model_metrics VALUES ('current_lr');
            """
        )
        connection.executemany("INSERT INTO model_event_summary VALUES ('current_lr')", [() for _ in range(126)])
        try:
            with self.assertRaisesRegex(ReaderError, "missing model_metrics row: history_lr"):
                _require_model_metrics(connection, ("current_lr", "history_lr"), "fixture")
        finally:
            connection.close()

    def test_output_must_be_new_and_inside_project(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            root = Path(directory).resolve()
            (root / "reports").mkdir()
            existing = root / "reports" / "existing.md"
            existing.write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(ReaderError, "already exists"):
                _output_path(root, "reports/existing.md")
            with self.assertRaisesRegex(ReaderError, "inside the project"):
                _output_path(root, str(root.parent / "outside.md"))

    def test_sqlite_consumer_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / ".tmp") as directory:
            path = Path(directory) / "small.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE t (value INTEGER)")
            connection.commit()
            connection.close()
            readonly = _open_readonly(path)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    readonly.execute("INSERT INTO t VALUES (1)")
            finally:
                readonly.close()


if __name__ == "__main__":
    unittest.main()
