import pathlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline import fetch_q2_and_build


PROJECT_TMP = pathlib.Path(__file__).resolve().parents[1] / ".tmp" / "tests"


class FetchQ2Test(unittest.TestCase):
    def test_member_verification_failure_stays_downloaded_and_is_not_verified(self):
        entry = {"date": "2023-04-01", "name": "data_Q2_2023/2023-04-01.csv"}
        downloaded = {**entry, "local": ".tmp/tests/failed.member", "sha256": "fixture"}
        with patch.object(fetch_q2_and_build, "download_member", return_value=downloaded), \
             patch.object(fetch_q2_and_build, "_verify_member_payload",
                          side_effect=ValueError("bad CRC")):
            with self.assertRaises(fetch_q2_and_build.MemberVerificationError) as caught:
                fetch_q2_and_build._download_and_verify_member(
                    entry, pathlib.Path(".tmp/tests"), 100,
                )
        self.assertEqual(caught.exception.entry["state"], "downloaded")
        self.assertEqual(caught.exception.entry["date"], entry["date"])

    def test_range_response_is_verified_and_identity_recorded(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = pathlib.Path(directory)
            target = root / "range.member"
            payload = b"0123456789"
            start, end, total = 10, 19, 100

            def fake_run(command, **kwargs):
                header_path = pathlib.Path(command[command.index("-D") + 1])
                output_path = pathlib.Path(command[command.index("-o") + 1])
                output_path.write_bytes(payload)
                header_path.write_text(
                    "HTTP/2 206\r\n"
                    "Content-Range: bytes 10-19/100\r\n"
                    "Content-Length: 10\r\n"
                    "ETag: \"fixture\"\r\n\r\n",
                    encoding="iso-8859-1",
                )
                return None

            with patch("pipeline.fetch_q2_and_build.subprocess.run", side_effect=fake_run):
                response = fetch_q2_and_build._download_range_to_file(
                    start, end, total, target, timeout=2,
                    expected_identity={"etag": '"fixture"'},
                )
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(response["http_status"], "206")
            self.assertIn("fixture", response["source_identity"])
            self.assertFalse(target.with_name(target.name + ".partial").exists())

    def test_failed_range_keeps_partial_evidence(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = pathlib.Path(directory)
            target = root / "range.member"

            def fake_run(command, **kwargs):
                header_path = pathlib.Path(command[command.index("-D") + 1])
                output_path = pathlib.Path(command[command.index("-o") + 1])
                output_path.write_bytes(b"bad")
                header_path.write_text(
                    "HTTP/2 200\r\nContent-Length: 3\r\n\r\n",
                    encoding="iso-8859-1",
                )
                return None

            with patch("pipeline.fetch_q2_and_build.subprocess.run", side_effect=fake_run):
                with self.assertRaises(RuntimeError):
                    fetch_q2_and_build._download_range_to_file(0, 9, 100, target, timeout=2)
            self.assertTrue(target.with_name(target.name + ".partial").exists())
            self.assertTrue(target.with_name(target.name + ".headers").exists())

    def test_scheduler_records_real_states_and_stops_after_three_low_days(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = pathlib.Path(directory)
            manifest = root / "progress.json"
            summary = root / "summary.json"
            entries = [
                {
                    "name": f"data_Q2_2023/2023-04-{day:02d}.csv",
                    "date": f"2023-04-{day:02d}",
                    "compressed": 1,
                    "size": 1,
                    "crc": 0,
                    "offset": day,
                    "flags": 0,
                    "method": 8,
                    "central_filename_length": 0,
                    "central_extra_length": 0,
                }
                for day in range(1, 7)
            ]

            class FakeConnection:
                def execute(self, *args, **kwargs):
                    if args and "FROM member_counts WHERE date" in args[0]:
                        requested_date = args[1][0]
                        class Result:
                            def fetchone(self):
                                return {
                                    "source_member": f"data_Q2_2023/{requested_date}.csv",
                                    "source_sha256": "fixture-hash",
                                    "source_rows": 1,
                                    "selected_rows": 1,
                                    "failure_rows": 0,
                                    "schema_columns": 11,
                                    "schema_sha256": "schema",
                                }
                        return Result()
                    return []

            class FakeWriter:
                def __init__(self, *args, **kwargs):
                    self.seen_dates = set()
                    self.connection = FakeConnection()
                    self.count = 0

                def set_metadata(self, values):
                    self.metadata = values

                def append_member(self, entry, root_path, model, **kwargs):
                    self.count += 1
                    return {
                        "source_rows": 1,
                        "selected_rows": 1,
                        "failure_rows": 0,
                        "schema_columns": 11,
                        "schema_sha256": "schema",
                    }

                def summary(self):
                    return {"panel_rows": self.count}

                def close(self):
                    pass

            def fake_download(entry, temp_dir, total_bytes, **kwargs):
                path = temp_dir / f"{entry['date']}.member"
                path.write_bytes(b"fixture")
                return {
                    **entry,
                    "local": str(path.relative_to(fetch_q2_and_build.ROOT)),
                    "sha256": "fixture-hash",
                    "downloaded_bytes": 7,
                    "range_start": entry["offset"],
                    "range_end": entry["offset"] + 1,
                    "http_status": "206",
                    "content_range": "bytes 0-1/100",
                    "content_length": "2",
                    "source_identity": json.dumps({"etag": '"fixture"'}),
                }

            low_flags = iter([False, True, True, True])
            with patch.object(fetch_q2_and_build, "head_archive", return_value=(859_566_204, {"etag": '"fixture"', "content-length": "859566204"})), \
                 patch.object(fetch_q2_and_build, "central_directory", return_value=entries), \
                 patch.object(fetch_q2_and_build, "PanelWriter", FakeWriter), \
                 patch.object(fetch_q2_and_build, "download_member", side_effect=fake_download), \
                 patch.object(fetch_q2_and_build, "_verify_member_payload", return_value=None), \
                 patch.object(fetch_q2_and_build, "_coverage_low_for_date", side_effect=lambda *args: next(low_flags)):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "fetch_q2_and_build",
                        "--database", str(root / "panel.sqlite"),
                        "--manifest-output", str(manifest),
                        "--summary-output", str(summary),
                        "--workers", "1",
                    ],
                ):
                    self.assertEqual(fetch_q2_and_build.main(), 0)
            state = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(state["stop_reason"], "three_consecutive_low_coverage_days_through:2023-04-04")
            self.assertEqual(state["completed_dates"], [
                "2023-04-01", "2023-04-02", "2023-04-03", "2023-04-04"
            ])
            self.assertEqual(state["pending_dates"], ["2023-04-05", "2023-04-06"])
            self.assertTrue(all(state["states"][date] == "ingested" for date in state["completed_dates"]))

    def test_explicit_retry_moves_failed_bytes_and_records_history(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = pathlib.Path(directory)
            manifest = root / "progress.json"
            summary = root / "summary.json"
            date = "2023-04-01"
            entry = {
                "name": f"data_Q2_2023/{date}.csv",
                "date": date,
                "compressed": 1,
                "size": 1,
                "crc": 0,
                "offset": 1,
                "flags": 0,
                "method": 8,
                "central_filename_length": 0,
                "central_extra_length": 0,
            }
            temp_dir = root / ".tmp" / "q2_members"
            temp_dir.mkdir(parents=True)
            retained = temp_dir / f"{date}.member"
            retained.write_bytes(b"failed-forensics")
            failed_hash = __import__("hashlib").sha256(retained.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "source_url": fetch_q2_and_build.URL,
                        "archive_bytes": 859_566_204,
                        "source_identity": {"etag": '"fixture"'},
                        "failed": {date: "bad CRC"},
                        "failed_members": [
                            {**entry, "local": str(retained.relative_to(root)),
                             "sha256": failed_hash, "state": "failed",
                             "failure_reason": "bad CRC"}
                        ],
                        "states": {date: "failed"},
                    }
                )
            )
            metadata_writes = []

            class Result:
                def __init__(self, rows):
                    self.rows = rows
                def __iter__(self):
                    return iter(self.rows)
                def fetchone(self):
                    return self.rows[0] if self.rows else None

            class FakeConnection:
                def execute(self, sql, params=()):
                    if "FROM member_counts WHERE date" in sql:
                        requested = params[0]
                        return Result([{
                            "source_member": f"data_Q2_2023/{requested}.csv",
                            "source_sha256": "fresh-hash",
                            "source_rows": 1,
                            "selected_rows": 1,
                            "failure_rows": 0,
                            "schema_columns": 11,
                            "schema_sha256": "schema",
                        }])
                    if "SELECT * FROM member_counts" in sql:
                        return Result([])
                    return Result([])

            class FakeWriter:
                def __init__(self, *args, **kwargs):
                    self.seen_dates = set()
                    self.connection = FakeConnection()
                    self.count = 0
                def set_metadata(self, values):
                    metadata_writes.append(dict(values))
                def append_member(self, entry, root_path, model, **kwargs):
                    self.count += 1
                    return {
                        "source_sha256": "fresh-hash",
                        "source_rows": 1,
                        "selected_rows": 1,
                        "failure_rows": 0,
                        "schema_columns": 11,
                        "schema_sha256": "schema",
                    }
                def summary(self):
                    return {"panel_rows": self.count}
                def close(self):
                    pass

            def fake_verified(download_entry, temp_root, total_bytes, **kwargs):
                path = temp_root / f"{date}.member"
                path.write_bytes(b"fresh-member")
                return {
                    **download_entry,
                    "local": str(path.relative_to(fetch_q2_and_build.ROOT)),
                    "sha256": "fresh-hash",
                    "source_identity": json.dumps({"etag": '"fixture"'}),
                    "state": "verified",
                }

            with patch.object(fetch_q2_and_build, "ROOT", root), \
                 patch.object(fetch_q2_and_build, "head_archive", return_value=(859_566_204, {"etag": '"fixture"'})), \
                 patch.object(fetch_q2_and_build, "central_directory", return_value=[entry]), \
                 patch.object(fetch_q2_and_build, "PanelWriter", FakeWriter), \
                 patch.object(fetch_q2_and_build, "_download_and_verify_member", side_effect=fake_verified), \
                 patch.object(fetch_q2_and_build, "_coverage_low_for_date", return_value=False), \
                 patch.object(sys, "argv", [
                     "fetch_q2_and_build", "--database", str(root / "panel.sqlite"),
                     "--manifest-output", str(manifest), "--summary-output", str(summary),
                     "--workers", "1", "--retry-failed",
                 ]):
                self.assertEqual(fetch_q2_and_build.main(), 0)
            state = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["failed_members"], [])
            self.assertEqual(state["states"][date], "ingested")
            self.assertEqual(len(state["retry_history"]), 1)
            forensic = root / state["retry_history"][0]["forensic_local"]
            self.assertTrue(forensic.is_file())
            self.assertFalse(retained.is_file())
            self.assertEqual(state["source_verification_status"], "verified")
            self.assertTrue(any("q2_source_identity" in item for item in metadata_writes))


if __name__ == "__main__":
    unittest.main()
