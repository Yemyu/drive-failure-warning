import csv
import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from pipeline.acquire_q3_validation import (
    AcquisitionError,
    REQUIRED_FIELDS,
    SMART_FIELDS,
    validate_archive,
)
from pipeline.build_q3_validation import BuildStopped, _member_rows


CAPACITY = "4000787030016"


def csv_bytes(date, rows, columns=None):
    columns = columns or list(REQUIRED_FIELDS)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = {field: "" for field in columns}
        values.update(row)
        values["date"] = date
        writer.writerow(values)
    return output.getvalue().encode("utf-8")


def row(serial, model="ST4000DM000", failure="0"):
    return {
        "serial_number": serial,
        "model": model,
        "capacity_bytes": CAPACITY if model == "ST4000DM000" else "1000",
        "failure": failure,
        **{field: "0" for field in SMART_FIELDS},
    }


def write_archive(root, members, compression=zipfile.ZIP_DEFLATED):
    archive = root / "fixture.zip"
    with zipfile.ZipFile(archive, "w", compression=compression) as output:
        for name, payload in members.items():
            output.writestr(name, payload)
    return archive


class Q3ArchiveValidationTest(unittest.TestCase):
    def test_valid_archive_records_full_and_selected_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = csv_bytes(
                "2023-07-01",
                [row("disk-a"), row("disk-b", model="OTHER")],
            )
            archive = write_archive(root, {"2023-07-01.csv": payload})
            result = validate_archive(
                archive,
                expected_date_values=["2023-07-01"],
                expected_archive_bytes=archive.stat().st_size,
            )
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["member_count"], 1)
            self.assertEqual(result["totals"]["source_rows"], 2)
            self.assertEqual(result["totals"]["selected_rows"], 1)
            self.assertEqual(result["totals"]["failure_rows"], 0)
            self.assertNotEqual(result["members"][0]["crc"], 0)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = [field for field in REQUIRED_FIELDS if field != "smart_198_raw"]
            archive = write_archive(
                root,
                {"2023-07-01.csv": csv_bytes("2023-07-01", [row("disk-a")], columns)},
            )
            with self.assertRaisesRegex(AcquisitionError, "missing required columns"):
                validate_archive(
                    archive,
                    expected_date_values=["2023-07-01"],
                    expected_archive_bytes=archive.stat().st_size,
                )

    def test_duplicate_serial_within_day_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = write_archive(
                root,
                {"2023-07-01.csv": csv_bytes("2023-07-01", [row("disk-a"), row("disk-a")])},
            )
            with self.assertRaisesRegex(AcquisitionError, "duplicate serial"):
                validate_archive(
                    archive,
                    expected_date_values=["2023-07-01"],
                    expected_archive_bytes=archive.stat().st_size,
                )

    def test_schema_change_across_days_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = list(REQUIRED_FIELDS)
            changed[0], changed[1] = changed[1], changed[0]
            archive = write_archive(
                root,
                {
                    "2023-07-01.csv": csv_bytes("2023-07-01", [row("disk-a")]),
                    "2023-07-02.csv": csv_bytes("2023-07-02", [row("disk-a")], changed),
                },
            )
            with self.assertRaisesRegex(AcquisitionError, "schema changed"):
                validate_archive(
                    archive,
                    expected_date_values=["2023-07-01", "2023-07-02"],
                    expected_archive_bytes=archive.stat().st_size,
                )

    def test_crc_or_decompression_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = write_archive(
                root,
                {"2023-07-01.csv": csv_bytes("2023-07-01", [row("disk-a")])},
            )
            blob = bytearray(archive.read_bytes())
            central_offset = blob.find(b"PK\x01\x02")
            self.assertGreaterEqual(central_offset, 0)
            crc_offset = central_offset + 16
            crc = struct.unpack_from("<I", blob, crc_offset)[0]
            struct.pack_into("<I", blob, crc_offset, crc ^ 0x01)
            archive.write_bytes(blob)
            with self.assertRaisesRegex(AcquisitionError, "CRC or decompression"):
                validate_archive(
                    archive,
                    expected_date_values=["2023-07-01"],
                    expected_archive_bytes=archive.stat().st_size,
                )

    def test_cross_day_model_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = write_archive(
                root,
                {
                    "2023-07-01.csv": csv_bytes("2023-07-01", [row("disk-a")]),
                    "2023-07-02.csv": csv_bytes("2023-07-02", [row("disk-a", model="OTHER")]),
                },
            )
            with self.assertRaisesRegex(AcquisitionError, "cross-day model conflict"):
                validate_archive(
                    archive,
                    expected_date_values=["2023-07-01", "2023-07-02"],
                    expected_archive_bytes=archive.stat().st_size,
                )


class Q3MemberRegistryTest(unittest.TestCase):
    def test_member_reader_returns_all_source_identities_for_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = csv_bytes(
                "2023-07-01",
                [row("disk-a"), row("disk-b", model="OTHER")],
            )
            archive = write_archive(root, {"2023-07-01.csv": payload})
            verified = validate_archive(
                archive,
                expected_date_values=["2023-07-01"],
                expected_archive_bytes=archive.stat().st_size,
            )
            expected = verified["members"][0]
            with zipfile.ZipFile(archive, "r") as source:
                info = source.getinfo("2023-07-01.csv")
                facts, selected, columns, source_models = _member_rows(
                    source,
                    info,
                    expected,
                    None,
                    {},
                    {},
                )
            self.assertEqual(facts["source_rows"], 2)
            self.assertEqual(len(selected), 1)
            self.assertEqual(source_models, {"disk-a": "ST4000DM000", "disk-b": "OTHER"})

    def test_member_reader_rejects_prior_model_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = csv_bytes("2023-07-01", [row("disk-a")])
            archive = write_archive(root, {"2023-07-01.csv": payload})
            verified = validate_archive(
                archive,
                expected_date_values=["2023-07-01"],
                expected_archive_bytes=archive.stat().st_size,
            )
            with zipfile.ZipFile(archive, "r") as source:
                with self.assertRaisesRegex(BuildStopped, "Q1/Q2 to Q3 model conflict"):
                    _member_rows(
                        source,
                        source.getinfo("2023-07-01.csv"),
                        verified["members"][0],
                        None,
                        {"disk-a": "OTHER"},
                        {},
                    )


if __name__ == "__main__":
    unittest.main()
