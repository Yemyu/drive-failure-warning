import csv
import hashlib
import io
import json
import pathlib
import struct
import tempfile
import unittest
import zlib
from unittest.mock import patch

from pipeline.panel import PanelWriter


def _member(root: pathlib.Path, rows: list[list[str]]) -> dict:
    columns = [
        "date", "serial_number", "model", "capacity_bytes", "failure",
        "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
        "smart_197_raw", "smart_198_raw", "vault_id", "pod_id", "is_legacy_format",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    name = "data_Q2_2023/2023-04-12.csv"
    filename = name.encode()
    payload = struct.pack(
        "<4s5H3L2H", b"PK\x03\x04", 20, 8, 8, 0, 0, 0, 0, 0, len(filename), 0
    ) + filename + compressed
    local = root / "member"
    local.write_bytes(payload)
    return {
        "name": name,
        "date": "2023-04-12",
        "compressed": len(compressed),
        "size": len(raw),
        "crc": zlib.crc32(raw) & 0xFFFFFFFF,
        "local": "member",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture_declaration() -> dict:
    columns = [
        "date", "serial_number", "model", "capacity_bytes", "failure",
        "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
        "smart_197_raw", "smart_198_raw", "vault_id", "pod_id", "is_legacy_format",
    ]
    return {"data_Q2_2023": {
        "count": len(columns),
        "sha256": hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
    }}


class PanelTest(unittest.TestCase):
    def test_schema_and_group_audit_are_persisted(self):
        project_tmp = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            root = pathlib.Path(directory)
            rows = [
                ["2023-04-12", "a", "ST4000DM000", "4000787030016", "0", "0", "10", "0", "0", "0", "0", "1007", "00", "0"],
                ["2023-04-12", "b", "OTHER", "1", "0", "", "", "", "", "", "", "1007", "00", "1"],
            ]
            entry = _member(root, rows)
            writer = PanelWriter(root / "panel.sqlite", schema_declarations=_fixture_declaration())
            try:
                result = writer.append_member(entry, root, "ST4000DM000")
                member = writer.connection.execute("SELECT * FROM member_counts").fetchone()
                audit = json.loads(member["audit_json"])
                self.assertEqual(result["selected_rows"], 1)
                self.assertEqual(member["schema_columns"], 14)
                self.assertTrue(member["schema_sha256"])
                self.assertEqual(tuple(writer.connection.execute(
                    "SELECT capacity_clean_bytes, capacity_missing FROM daily WHERE serial_number='a'"
                ).fetchone()), (4000787030016, 0))
                self.assertEqual(audit["selected"]["vault_id"], {"1007": 1})
                self.assertEqual(audit["all"]["vault_id"], {"1007": 2})
            finally:
                writer.close()

    def test_undeclared_schema_is_rejected_before_write(self):
        project_tmp = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            root = pathlib.Path(directory)
            entry = _member(root, [["2023-04-12", "a", "ST4000DM000", "-1", "0", "", "", "", "", "", "", "1007", "00", "0"]])
            writer = PanelWriter(root / "panel.sqlite")
            try:
                with self.assertRaisesRegex(ValueError, "does not match declared"):
                    writer.append_member(entry, root, "ST4000DM000")
                self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0], 0)
                self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM member_counts").fetchone()[0], 0)
            finally:
                writer.close()

    def test_unexpected_negative_capacity_is_rejected(self):
        project_tmp = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            root = pathlib.Path(directory)
            entry = _member(root, [["2023-04-12", "a", "ST4000DM000", "-2", "0", "", "", "", "", "", "", "1007", "00", "0"]])
            writer = PanelWriter(root / "panel.sqlite", schema_declarations=_fixture_declaration())
            try:
                with self.assertRaisesRegex(ValueError, "unexpected negative capacity"):
                    writer.append_member(entry, root, "ST4000DM000")
                self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0], 0)
            finally:
                writer.close()

    def test_duplicate_serial_across_models_is_rejected(self):
        project_tmp = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            root = pathlib.Path(directory)
            rows = [
                ["2023-04-12", "same", "ST4000DM000", "1", "0", "", "", "", "", "", "", "1007", "00", "0"],
                ["2023-04-12", "same", "OTHER", "1", "0", "", "", "", "", "", "", "1007", "00", "0"],
            ]
            entry = _member(root, rows)
            writer = PanelWriter(root / "panel.sqlite", schema_declarations=_fixture_declaration())
            try:
                with self.assertRaises(ValueError):
                    writer.append_member(entry, root, "ST4000DM000")
            finally:
                writer.close()

    def test_cross_date_serial_model_change_is_rejected_before_write(self):
        project_tmp = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"
        project_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=project_tmp) as directory:
            root = pathlib.Path(directory)
            columns = [
                "date", "serial_number", "model", "capacity_bytes", "failure",
                "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw",
                "smart_197_raw", "smart_198_raw",
            ]
            writer = PanelWriter(root / "panel.sqlite", schema_declarations={"fixture": {
                "count": len(columns),
                "sha256": hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            }})
            def append(date, model):
                row = [date, "same", model, "-1", "0", "", "", "", "", "", ""]
                with patch(
                    "pipeline.panel.iter_member_rows",
                    return_value=(date, "fixture", columns, iter([(2, row)]), {}),
                ):
                    return writer.append_member({"name": f"fixture/{date}.csv"}, root, "ST4000DM000")
            try:
                append("2023-01-01", "ST4000DM000")
                with self.assertRaisesRegex(ValueError, "identity conflict"):
                    append("2023-01-02", "OTHER")
                self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM member_counts").fetchone()[0], 1)
                self.assertEqual(writer.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0], 1)
                self.assertEqual(
                    writer.connection.execute("SELECT capacity_clean_bytes FROM daily").fetchone()[0],
                    None,
                )
            finally:
                writer.close()


if __name__ == "__main__":
    unittest.main()
