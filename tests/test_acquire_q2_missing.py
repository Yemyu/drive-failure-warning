import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import acquire_q2_missing as acquire


PROJECT_TMP = Path(__file__).resolve().parents[1] / ".tmp" / "tests"
MANIFEST = Path(__file__).resolve().parents[1] / "evidence/q2/input_review/stable_input_manifest.json"


class AcquireQ2MissingTest(unittest.TestCase):
    def test_plan_is_fixed_and_network_free(self):
        manifest = acquire.load_manifest(MANIFEST)
        with patch.object(acquire.subprocess, "run", side_effect=AssertionError("plan used network")):
            plan = acquire.build_plan(manifest, MANIFEST)
        self.assertEqual(plan["member_count"], 87)
        self.assertEqual(plan["legacy_compressed_bytes"], 835_127_808)
        self.assertEqual(plan["planned_range_lower_bound_bytes"], 840_834_399)
        self.assertEqual(plan["budgets"]["workers"], 1)
        self.assertEqual(plan["entries"][0]["date"], "2023-04-01")
        self.assertEqual(plan["entries"][-1]["date"], "2023-06-30")

    def test_plan_hash_is_self_consistent(self):
        plan = acquire.build_plan(acquire.load_manifest(MANIFEST), MANIFEST)
        self.assertEqual(plan["plan_sha256"], acquire.canonical_hash(plan, ("plan_sha256",)))

    def test_range_response_requires_exact_206_and_identity(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            path = Path(directory) / "member.bin"
            path.write_bytes(b"0123456789")
            headers = {
                "content-range": "bytes 10-19/100",
                "content-length": "10",
                "x-bz-file-id": "fixture",
            }
            result = acquire.validate_range_response(
                206, headers, path, 10, 19, 100, {"x-bz-file-id": "fixture"}
            )
            self.assertEqual(result["content_length"], 10)
            with self.assertRaises(acquire.AcquisitionError):
                acquire.validate_range_response(
                    200, headers, path, 10, 19, 100, {"x-bz-file-id": "fixture"}
                )
            with self.assertRaises(acquire.AcquisitionError):
                acquire.validate_range_response(
                    206, headers | {"x-bz-file-id": "other"}, path, 10, 19, 100,
                    {"x-bz-file-id": "fixture"},
                )

    def test_failed_range_retains_partial_headers_and_has_no_implicit_retry(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = Path(directory)
            member = root / "member.bin"
            headers_path = root / "response.headers"
            stderr_path = root / "response.stderr"
            command_seen = []

            def fake_run(command, **kwargs):
                command_seen.extend(command)
                partial = Path(command[command.index("-o") + 1])
                header = Path(command[command.index("-D") + 1])
                partial.write_bytes(b"partial")
                header.write_text("HTTP/2 200\r\nContent-Length: 7\r\n\r\n", encoding="iso-8859-1")
                return type("Result", (), {"returncode": 0})()

            with patch.object(acquire.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(acquire.AcquisitionError):
                    acquire._curl_range(
                        "https://fixture.invalid/archive.zip", 10, 19, 100,
                        member, headers_path, stderr_path, {"x-bz-file-id": "fixture"},
                    )
            self.assertTrue(member.with_name("member.bin.partial").is_file())
            self.assertTrue(headers_path.is_file())
            self.assertTrue(stderr_path.is_file())
            self.assertNotIn("--retry", command_seen)

    def test_valid_member_streams_rows_and_schema(self):
        manifest = acquire.load_manifest(MANIFEST)
        item = next(item for item in manifest["members"] if item["date"] == "2023-04-12")
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            result = acquire.verify_member_file(
                Path(__file__).resolve().parents[1] / item["local"],
                item,
                Path(directory) / "decoded.csv",
            )
        self.assertEqual(result["schema_columns"], 186)
        self.assertEqual(result["schema_sha256"], "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2")
        self.assertEqual(result["source_rows"], 240737)
        self.assertEqual(result["selected_rows"], 18036)
        self.assertEqual(result["failure_rows"], 4)

    def test_member_crc_is_rejected(self):
        manifest = acquire.load_manifest(MANIFEST)
        item = next(item for item in manifest["members"] if item["date"] == "2023-04-12")
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = Path(directory)
            copy = root / "member.bin"
            copy.write_bytes((Path(__file__).resolve().parents[1] / item["local"]).read_bytes())
            corrupt = dict(item, crc=(int(item["crc"]) + 1) % (2**32))
            with self.assertRaisesRegex(acquire.AcquisitionError, "CRC mismatch"):
                acquire.verify_member_file(copy, corrupt, root / "decoded.csv")

    def test_coverage_uses_only_previous_seven_days_and_stops_at_three_lows(self):
        counts = {}
        for day in range(25, 32):
            date = f"2023-03-{day:02d}"
            counts[date] = {"source_rows": 100, "selected_rows": 100, "origin": "fixture"}
        for day in range(1, 11):
            date = f"2023-04-{day:02d}"
            low = day in (8, 9, 10)
            counts[date] = {"source_rows": 70 if low else 100, "selected_rows": 70 if low else 100, "origin": "fixture"}
        last, records, streak, total, stop = acquire.advance_coverage(
            "2023-04-07", "2023-04-10", counts
        )
        self.assertEqual(last, "2023-04-10")
        self.assertEqual([record["date"] for record in records], ["2023-04-08", "2023-04-09", "2023-04-10"])
        self.assertEqual(streak, 3)
        self.assertEqual(total, 3)
        self.assertEqual(stop, "three_consecutive_low_coverage_days_through:2023-04-10")
        with self.assertRaises(acquire.AcquisitionStopped):
            acquire.coverage_record("2023-04-11", counts)

    def test_orphan_artifacts_are_not_auto_verified(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = Path(directory)
            raw_root = root / "raw"
            evidence_root = root / "evidence"
            orphan = raw_root / "attempts/2023-04-01/attempt-001"
            orphan.mkdir(parents=True)
            (orphan / "member.bin").write_bytes(b"orphan")
            with patch.object(acquire, "RAW_ROOT", raw_root), patch.object(acquire, "EVIDENCE_ROOT", evidence_root):
                found = acquire._orphan_artifacts("2023-04-01")
            self.assertEqual(found, [str((orphan / "member.bin").relative_to(acquire.ROOT))])

    def test_execute_is_panel_free_and_resume_does_not_redownload_verified_member(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as directory:
            root = Path(directory)
            date = "2023-04-01"
            legacy_sha = "a" * 64
            plan = {
                "plan_sha256": "",
                "manifest_stable_hash": "fixture-manifest",
                "source_url": "https://fixture.invalid/archive.zip",
                "terms_url": "https://fixture.invalid/terms",
                "archive_bytes": 100,
                "archive_identity": {"x-bz-file-id": "fixture"},
                "entries": [{
                    "date": date,
                    "name": f"data_Q2_2023/{date}.csv",
                    "offset": 0,
                    "compressed": 1,
                    "size": 1,
                    "crc": 0,
                    "legacy_source_sha256": legacy_sha,
                }],
                "member_count": 1,
            }
            plan["plan_sha256"] = acquire.canonical_hash(plan)
            central = {
                "date": date,
                "name": plan["entries"][0]["name"],
                "offset": 0,
                "compressed": 1,
                "size": 1,
                "crc": 0,
                "flags": 0,
                "method": 8,
                "central_filename_length": len(plan["entries"][0]["name"].encode()),
                "central_extra_length": 0,
            }
            baseline = {
                f"2023-03-{day:02d}": {"source_rows": 100, "selected_rows": 100, "origin": "fixture"}
                for day in range(25, 32)
            }
            calls = []

            def fake_range(url, start, end, total, member_path, headers_path, stderr_path, identity):
                calls.append(date)
                member_path.parent.mkdir(parents=True, exist_ok=True)
                headers_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                member_path.write_bytes(b"member")
                headers_path.write_text("fixture", encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                return {
                    "http_status": 206,
                    "content_range": "bytes 0-1/100",
                    "content_length": 2,
                    "observed_range_sha256": acquire.sha256_file(member_path),
                    "observed_source_identity": identity,
                    "member_path": str(member_path.relative_to(root)),
                    "headers_path": str(headers_path.relative_to(root)),
                    "stderr_path": str(stderr_path.relative_to(root)),
                }

            def fake_verify(path, entry, decoded_path=None):
                return {
                    "source_rows": 100,
                    "selected_rows": 100,
                    "failure_rows": 0,
                    "unique_serials": 100,
                    "schema_columns": 186,
                    "schema_sha256": "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2",
                    "expanded_bytes": 1,
                    "crc32": 0,
                    "observed_compressed_payload_sha256": "b" * 64,
                    "observed_range_sha256": acquire.sha256_file(path),
                    "observed_range_bytes": path.stat().st_size,
                }

            with patch.object(acquire, "ROOT", root), \
                 patch.object(acquire, "RAW_ROOT", root / "raw"), \
                 patch.object(acquire, "EVIDENCE_ROOT", root / "evidence"), \
                 patch.object(acquire, "EXPECTED_MISSING_MEMBERS", 1), \
                 patch.object(acquire, "EXPECTED_Q2_DATES", [date]), \
                 patch.object(acquire, "load_audit_counts", return_value=baseline), \
                 patch.object(acquire, "_source_control", return_value=([central], {})), \
                 patch.object(acquire, "_curl_range", side_effect=fake_range), \
                 patch.object(acquire, "verify_member_file", side_effect=fake_verify), \
                 patch.object(acquire, "_rss_bytes", return_value=0):
                first = acquire.execute(plan, root / "evidence/index.json")
                second = acquire.execute(plan, root / "evidence/index.json")
            self.assertEqual(first["status"], "complete")
            self.assertEqual(second["status"], "complete")
            self.assertEqual(calls, [date])
            state = second["member_states"][date]
            self.assertEqual(state["status"], "verified")
            self.assertEqual(len(state["receipts"]), 1)


if __name__ == "__main__":
    unittest.main()
