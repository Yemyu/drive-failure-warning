import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

from pipeline.panel import PanelWriter
from pipeline.reproducible import current
from pipeline.reproducible.source_streams import CHUNK, ArchiveSource, DeflateReader, SourceError, range_rows

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure", *[f"smart_{f}_raw" for f in (5, 9, 187, 188, 197, 198)]]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def csv_bytes(date, rows, columns=COLUMNS):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for supplied in rows:
        row = dict(date=date, serial_number="a", model="ST4000DM000", capacity_bytes="4000787030016", failure="0")
        row.update(supplied)
        writer.writerow([row.get(column, "") for column in columns])
    return buffer.getvalue().encode()


def fixture(root, dates_rows, columns=COLUMNS):
    archive_path = root / "source.zip"
    members, ranges = [], []
    schema = sha(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode())
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for date, rows in dates_rows:
            raw = csv_bytes(date, rows, columns)
            name = f"data_Q3_2023/{date}.csv"
            archive.writestr(name, raw)
            info = archive.getinfo(name)
            member = dict(date=date, name=name, compressed_bytes=info.compress_size, expanded_bytes=len(raw), crc=info.CRC,
                          sha256=sha(raw), source_rows=len(rows), selected_rows=sum(r.get("model", "ST4000DM000") == "ST4000DM000" for r in rows),
                          failure_rows=sum(r.get("failure", "0") == "1" and r.get("model", "ST4000DM000") == "ST4000DM000" for r in rows))
            members.append(member)
            compressor = zlib.compressobj(wbits=-15)
            compressed = compressor.compress(raw) + compressor.flush()
            name_bytes = name.encode()
            payload = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 8, 8, 0, 0, 0, 0, 0, len(name_bytes), 0) + name_bytes + compressed
            local = root / f"{date}.member"
            local.write_bytes(payload)
            ranges.append(dict(date=date, name=name, local=str(local.relative_to(ROOT)), range_bytes=len(payload), sha256=sha(payload),
                               compressed=len(compressed), size=len(raw), crc=info.CRC,
                               expected_counts={**{key: member[key] for key in ("source_rows", "selected_rows", "failure_rows")},
                                                "schema_columns": len(columns), "schema_sha256": schema}))
    receipt = dict(status="verified", archive_path=str(archive_path.relative_to(ROOT)), archive_bytes=archive_path.stat().st_size,
                   archive_sha256=sha(archive_path.read_bytes()), member_count=len(members), dates=[d for d, _ in dates_rows],
                   members=members, schema=columns, schema_columns=len(columns), schema_sha256=schema, q4_access=False, remote_setup=False)
    save_receipt(root, receipt)
    return receipt, ranges


def save_receipt(root, receipt):
    receipt["receipt_hash"] = current._canonical({k: v for k, v in receipt.items() if k != "receipt_hash"})
    (root / "receipt.json").write_text(json.dumps(receipt))


def q1_manifest_fixture(root, *, future_smart5=None, drop_dates=()):
    """Create a legal Q1 range manifest for an as-of perturbation check."""
    root.mkdir(parents=True, exist_ok=True)
    columns = json.loads((ROOT / "evidence/observed_schemas.json").read_text())["data_Q1_2023/2023-01-01.csv"]
    schema = sha(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode())
    dropped = set(drop_dates)
    members = []
    for day in range(1, 23):
        date = f"2023-01-{day:02d}"
        if date in dropped:
            continue
        rows = []
        for serial, default_smart5 in (("disk-a", 0), ("disk-b", 1)):
            row = {column: "" for column in columns}
            row.update(
                date=date,
                serial_number=serial,
                model="ST4000DM000",
                capacity_bytes="4000787030016",
                failure="0",
                smart_5_raw=str(future_smart5 if future_smart5 is not None and date == "2023-01-20" else default_smart5),
                smart_9_raw=str(100 + day),
                smart_187_raw="1" if serial == "disk-b" else "0",
                smart_188_raw="0",
                smart_197_raw="0",
                smart_198_raw="0",
            )
            rows.append([row[column] for column in columns])
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
        raw = stream.getvalue().encode()
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(raw) + compressor.flush()
        name = f"data_Q1_2023/{date}.csv"
        name_bytes = name.encode()
        payload = struct.pack(
            "<4s5H3L2H", b"PK\x03\x04", 20, 8, 8, 0, 0, 0, 0, 0, len(name_bytes), 0
        ) + name_bytes + compressed
        local = root / f"{date}.member"
        local.write_bytes(payload)
        members.append(
            {
                "name": name,
                "date": date,
                "compressed": len(compressed),
                "size": len(raw),
                "crc": zlib.crc32(raw) & 0xFFFFFFFF,
                "range_bytes": len(payload),
                "local": str(local.relative_to(ROOT)),
                "sha256": sha(payload),
                "expected_counts": {
                    "source_rows": 2,
                    "selected_rows": 2,
                    "failure_rows": 0,
                    "schema_columns": len(columns),
                    "schema_sha256": schema,
                },
            }
        )
    manifest = {"status": "verified", "member_count": len(members), "members": members}
    path = root / "source_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


def identity_panel(path, model="ST4000DM000"):
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT NOT NULL, "
        "first_date TEXT NOT NULL, last_date TEXT NOT NULL, first_source_member TEXT NOT NULL, source_scope TEXT NOT NULL);"
    )
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("build_status", "panel_complete"), ("serial_model_registry_scope", "full_source_rows")],
    )
    connection.execute(
        "INSERT INTO serial_model_registry VALUES (?, ?, '2023-01-01', '2023-01-01', 'fixture/2023-01-01.csv', 'full_source_member')",
        ("a", model),
    )
    connection.commit()
    connection.close()


class SourceStreamTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def writer(self, receipt):
        writer = PanelWriter(self.root / "panel.sqlite", schema_declarations={"data_Q3_2023": {"count": receipt["schema_columns"], "sha256": receipt["schema_sha256"]}})
        self.addCleanup(writer.close)
        return writer

    def test_ranges_and_archive_share_rows_without_whole_member_reads(self):
        receipt, ranges = fixture(self.root, [("2023-07-01", [{}, {"serial_number": "b", "model": "OTHER", "capacity_bytes": "-1"}])])
        with patch("zlib.decompress", side_effect=AssertionError("whole decompression forbidden")), patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("whole ZIP read forbidden")):
            with range_rows(ranges[0], ROOT) as data:
                range_values = list(data[3])
            with ArchiveSource(receipt, ROOT) as archive:
                with archive.rows(receipt["members"][0]) as (_, data):
                    archive_values = list(data[3])
        self.assertEqual(range_values, archive_values)
        self.assertEqual(len(range_values), 2)

    def test_high_compression_ratio_has_bounded_read_chunks(self):
        raw = b"x" * (CHUNK * 200)
        compressor = zlib.compressobj(wbits=-15)
        encoded = compressor.compress(raw) + compressor.flush()
        decoder = DeflateReader(io.BytesIO(encoded), len(encoded))
        total = 0
        while block := decoder.read(CHUNK):
            self.assertLessEqual(len(block), CHUNK)
            total += len(block)
        self.assertEqual(total, len(raw))

    def test_wrong_counts_leave_all_day_tables_empty(self):
        receipt, ranges = fixture(self.root, [("2023-07-01", [{}])])
        entry = ranges[0]
        entry["expected_counts"]["source_rows"] = 2
        writer = self.writer(receipt)
        with self.assertRaisesRegex(ValueError, "source_rows mismatch"):
            with range_rows(entry, ROOT) as data:
                writer.append_member(entry, ROOT, current.MODEL, source_data=data, expected_counts=entry["expected_counts"], strict_smart=True)
        for table in ("daily", "member_counts", "serial_model_registry"):
            self.assertEqual(writer.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_crc_and_expanded_hash_fail_before_commit(self):
        receipt, ranges = fixture(self.root, [("2023-07-01", [{}])])
        entry = ranges[0]
        entry["crc"] ^= 1
        writer = self.writer(receipt)
        with self.assertRaisesRegex(SourceError, "CRC mismatch"):
            with range_rows(entry, ROOT) as data:
                writer.append_member(entry, ROOT, current.MODEL, source_data=data, expected_counts=entry["expected_counts"])
        self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0], 0)
        receipt["members"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(SourceError, "SHA256 mismatch"):
            with ArchiveSource(receipt, ROOT) as archive:
                with archive.rows(receipt["members"][0]) as (_, data):
                    list(data[3])

    def test_bad_final_deflate_marker_is_rejected(self):
        compressor = zlib.compressobj(wbits=-15)
        raw = compressor.compress(b"hello world") + compressor.flush()
        with self.assertRaisesRegex(SourceError, "end marker"):
            decoder = DeflateReader(io.BytesIO(raw[:-1]), len(raw) - 1)
            while decoder.read(CHUNK):
                pass

    def test_cli_full_header_success_and_cross_date_identity_failure(self):
        # Official ordered columns, but entirely fictional devices and outcomes.
        columns = json.loads((ROOT / "evidence/q3/validation_v1/source_receipt_v1.json").read_text())["schema"]
        fixture(self.root, [("2023-07-01", [{"smart_5_raw": "0"}]), ("2023-07-02", [{"smart_5_raw": "1"}])], columns)
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "build-zip-panel", str(self.root / "receipt.json"), "--output", str(self.root / "success")]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.root / "success/panel_manifest.json").read_text())
        self.assertEqual(manifest["summary"]["panel_rows"], 2)
        conn = sqlite3.connect(self.root / "success/panel.sqlite")
        try:
            self.assertEqual(conn.execute("SELECT smart_5_raw,smart_187_missing FROM daily ORDER BY date").fetchall(), [(0, 1), (1, 1)])
        finally:
            conn.close()
        bad = self.root / "bad"
        bad.mkdir()
        fixture(bad, [("2023-07-01", [{}]), ("2023-07-02", [{"model": "OTHER"}])], columns)
        command[-3] = str(bad / "receipt.json")
        command[-1] = str(bad / "run")
        failed = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((bad / "run/panel_manifest.json").exists())
        self.assertTrue((bad / "run/failure.json").exists())
        conn = sqlite3.connect(bad / "run/panel.sqlite.partial")
        try:
            self.assertEqual(conn.execute("SELECT date FROM member_counts").fetchall(), [("2023-07-01",)])
        finally:
            conn.close()

    def test_cli_q3_panel_requires_optional_cross_quarter_identity_when_supplied(self):
        columns = json.loads((ROOT / "evidence/q3/validation_v1/source_receipt_v1.json").read_text())["schema"]
        q3 = self.root / "q3"
        q3.mkdir()
        fixture(q3, [("2023-07-01", [{}])], columns)
        prior = self.root / "prior.sqlite"
        identity_panel(prior)
        command = [sys.executable, str(ROOT / "tools/run_research.py"), "build-zip-panel", str(q3 / "receipt.json"), "--output", str(self.root / "verified"), "--prior-panel", str(prior)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.root / "verified/panel_manifest.json").read_text())
        self.assertEqual(manifest["cross_quarter_identity"]["status"], "pass")
        self.assertEqual(manifest["artifacts"]["panel"]["path"], str((self.root / "verified/panel.sqlite").relative_to(ROOT)))
        final = self.root / "verified/panel.sqlite"
        connection = current._open_readonly(final)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            self.assertEqual(metadata["build_status"], "panel_complete")
            self.assertEqual(metadata["cross_quarter_identity"], "verified")
        finally:
            connection.close()
        self.assertEqual(manifest["cross_quarter_identity"]["q3_panel_sha256"], sha(final.read_bytes()))
        self.assertEqual(manifest["cross_quarter_identity"]["q3_panel"], str(final.relative_to(ROOT)))
        for name in ("panel.sqlite", "panel.sqlite.partial"):
            for suffix in ("-wal", "-shm"):
                self.assertFalse((final.parent / (name + suffix)).exists())
        bad_prior = self.root / "bad_prior.sqlite"
        identity_panel(bad_prior, model="OTHER")
        bad_command = [sys.executable, str(ROOT / "tools/run_research.py"), "build-zip-panel", str(q3 / "receipt.json"), "--output", str(self.root / "conflict"), "--prior-panel", str(bad_prior)]
        failed = subprocess.run(bad_command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("identity conflict", failed.stderr)
        self.assertFalse((self.root / "conflict/panel_manifest.json").exists())
        self.assertTrue((self.root / "conflict/failure.json").exists())

    def test_q3_checkpoint_failure_never_publishes_complete_manifest(self):
        columns = json.loads((ROOT / "evidence/q3/validation_v1/source_receipt_v1.json").read_text())["schema"]
        fixture(self.root, [("2023-07-01", [{}])], columns)
        real_connect = sqlite3.connect

        class BusyCheckpoint(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                    return super().execute("SELECT 1, 1, 0")
                return super().execute(sql, *args, **kwargs)

        def connect(*args, **kwargs):
            kwargs["factory"] = BusyCheckpoint
            return real_connect(*args, **kwargs)

        output = self.root / "busy"
        with patch.object(current.sqlite3, "connect", side_effect=connect):
            with self.assertRaisesRegex(current.ReproducibleStopped, "checkpoint incomplete"):
                current.build_archive_panel(self.root / "receipt.json", output)
        self.assertFalse((output / "panel.sqlite").exists())
        self.assertFalse((output / "panel_manifest.json").exists())
        self.assertTrue((output / "failure.json").exists())

    def test_legal_future_manifest_perturbations_preserve_earlier_features_scores_and_selection(self):
        """Rebound source inputs must leave pre-perturbation as-of outputs unchanged."""
        original_manifest = q1_manifest_fixture(self.root / "original", future_smart5=None)
        changed_root = self.root / "changed"
        changed_root.mkdir()
        changed_manifest = q1_manifest_fixture(changed_root, future_smart5=999)
        deleted_root = self.root / "deleted"
        deleted_root.mkdir()
        deleted_manifest = q1_manifest_fixture(deleted_root, drop_dates=("2023-01-20",))

        built = []
        for label, manifest in (("original", original_manifest), ("changed", changed_manifest), ("deleted", deleted_manifest)):
            verified = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_research.py"), "verify-source", str(manifest), "--scan-members"],
                cwd=self.root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            output = self.root / f"{label}_panel.sqlite"
            evidence = self.root / f"{label}_panel_evidence.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/run_research.py"), "build-panel", str(manifest),
                 "--output", str(output), "--evidence", str(evidence)],
                cwd=self.root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(evidence.read_text(encoding="utf-8"))["status"], "complete")
            built.append((label, manifest, output))

        manifest_hashes = [current._sha256_file(manifest) for _label, manifest, _panel in built]
        panel_hashes = [current._sha256_file(panel) for _label, _manifest, panel in built]
        self.assertEqual(len(set(manifest_hashes)), 3)
        self.assertEqual(len(set(panel_hashes)), 3)

        payload = {
            "coef": [0.1] * len(current.CURRENT_COLUMNS),
            "intercept": 0.0,
            "preprocessing": {
                "imputation_mean": [0.0] * len(current.CURRENT_COLUMNS),
                "standardization_mean": [0.0] * len(current.CURRENT_COLUMNS),
                "standardization_scale": [1.0] * len(current.CURRENT_COLUMNS),
            },
        }
        earlier = []
        for label, _manifest, panel in built:
            features = self.root / f"{label}_features.sqlite"
            current.build_current_features(
                panel, features, start="2023-01-15", end="2023-01-18", dataset_end="2023-01-22"
            )
            with sqlite3.connect(features) as connection:
                feature_rows = connection.execute(
                    "SELECT decision_date,serial_number," + ",".join(current.CURRENT_COLUMNS) +
                    ",smart_nonzero_signal_count,tie_break_sha256 FROM feature_rows ORDER BY decision_date,serial_number"
                ).fetchall()
                calendar = connection.execute(
                    "SELECT * FROM qualification_calendar ORDER BY decision_date"
                ).fetchall()
            scores = list(current._rule_and_model_scores(features, payload))
            selections = [current._select_scores(scores, method) for method in ("current_lr", "smart_nonzero")]
            earlier.append((feature_rows, calendar, scores, selections))
        self.assertEqual(earlier[0], earlier[1])
        self.assertEqual(earlier[0], earlier[2])


if __name__ == "__main__":
    unittest.main()
