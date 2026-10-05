"""Acquire the 87 missing Q2 members as retained, independently audited bytes.

This module intentionally does not import the panel writer or the legacy Q2
downloader.  It keeps source acquisition, verification and the per-attempt
receipt in separate paths so a crash cannot turn an unverified file into a
verified source claim.  The default command is a network-free plan; real
network access requires the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evidence/q2/input_review/stable_input_manifest.json"
PLAN_PATH = ROOT / "evidence/q2/reacquired_v1/plan_v1.json"
RAW_ROOT = ROOT / "data/raw/q2_reacquired_v1"
EVIDENCE_ROOT = ROOT / "evidence/q2/reacquired_v1"
AUDIT_DATABASE = ROOT / "data/derived/panel_retained_audit_v1.sqlite"

ARCHIVE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2023.zip"
TERMS_URL = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
EXPECTED_ARCHIVE_BYTES = 859_566_204
EXPECTED_Q2_DATES = [
    (dt.date(2023, 4, 1) + dt.timedelta(days=index)).isoformat()
    for index in range(91)
]
MODEL = "ST4000DM000"
EXPECTED_MISSING_MEMBERS = 87
EXPECTED_RETAINED_MEMBERS = 4
MAX_RETAINED_BYTES = 1 * 1024 * 1024 * 1024
MAX_NETWORK_BYTES = int(1.1 * 1024 * 1024 * 1024)
MAX_ACTIVE_TEMP_BYTES = 256 * 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
RANGE_PAD_BYTES = 65_536
HTTP_TIMEOUT_SECONDS = 180
TERMS_MAX_BYTES = 16 * 1024 * 1024
SMART_FIELDS = tuple(f"smart_{number}_raw" for number in (5, 9, 187, 188, 197, 198))
REQUIRED_FIELDS = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    *SMART_FIELDS,
)


class AcquisitionError(RuntimeError):
    """A source, receipt or verification contract was violated."""


class AcquisitionStopped(RuntimeError):
    """A registered stop gate prevented another request."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def code_sha256() -> str:
    return sha256_file(Path(__file__))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def canonical_hash(value: dict, excluded: tuple[str, ...] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _hex_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AcquisitionError(f"{field} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AcquisitionError(f"{field} is not hexadecimal") from exc
    return value


def _q2_members(manifest: dict) -> list[dict]:
    members = manifest.get("members")
    if not isinstance(members, list):
        raise AcquisitionError("stable manifest has no members list")
    q2 = [item for item in members if item.get("quarter") == "2023Q2"]
    if len(q2) != len(EXPECTED_Q2_DATES):
        raise AcquisitionError(f"stable manifest has {len(q2)} Q2 members, expected 91")
    q2.sort(key=lambda item: item.get("date", ""))
    if [item.get("date") for item in q2] != EXPECTED_Q2_DATES:
        raise AcquisitionError("stable manifest Q2 dates are not exactly 2023-04-01 through 2023-06-30")
    identities = {json.dumps(item.get("archive_identity"), sort_keys=True) for item in q2}
    if len(identities) != 1:
        raise AcquisitionError("Q2 members do not share one archive identity")
    identity = q2[0].get("archive_identity") or {}
    if not identity.get("x-bz-file-id"):
        raise AcquisitionError("stable Q2 archive identity lacks x-bz-file-id")
    for item in q2:
        if int(item.get("archive_bytes", -1)) != EXPECTED_ARCHIVE_BYTES:
            raise AcquisitionError(f"archive size changed in stable manifest for {item.get('date')}")
        _hex_sha(item.get("source_sha256"), f"legacy source SHA for {item.get('date')}")
        for field in ("offset", "compressed", "size", "crc"):
            if int(item.get(field, -1)) < 0:
                raise AcquisitionError(f"invalid {field} for {item.get('date')}")
    return q2


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _q2_members(manifest)
    if not manifest.get("stable_hash"):
        raise AcquisitionError("stable manifest has no stable_hash")
    return manifest


def _attempt_relpaths(date: str, attempt: int) -> dict[str, str]:
    prefix = f"attempts/{date}/attempt-{attempt:03d}"
    return {
        "raw_dir": f"data/raw/q2_reacquired_v1/{prefix}",
        "member": f"data/raw/q2_reacquired_v1/{prefix}/member.bin",
        "partial": f"data/raw/q2_reacquired_v1/{prefix}/member.bin.partial",
        "evidence_dir": f"evidence/q2/reacquired_v1/{prefix}",
        "headers": f"evidence/q2/reacquired_v1/{prefix}/response.headers",
        "stderr": f"evidence/q2/reacquired_v1/{prefix}/response.stderr",
        "receipt": f"evidence/q2/reacquired_v1/{prefix}/receipt.json",
        "decoded": f"data/raw/q2_reacquired_v1/{prefix}/decoded.csv",
    }


def build_plan(manifest: dict, manifest_path: Path = MANIFEST_PATH) -> dict:
    q2 = _q2_members(manifest)
    missing = [item for item in q2 if not item.get("local")]
    retained = [item for item in q2 if item.get("local")]
    if len(missing) != EXPECTED_MISSING_MEMBERS or len(retained) != EXPECTED_RETAINED_MEMBERS:
        raise AcquisitionError(
            f"stable manifest scope is {len(missing)} missing/{len(retained)} retained, expected 87/4"
        )
    if any(item.get("content_evidence") != "legacy_manifest_only" for item in missing):
        raise AcquisitionError("a missing member is not explicitly legacy_manifest_only")
    if any(item.get("content_evidence") != "verified_range_crc" for item in retained):
        raise AcquisitionError("a retained member lacks verified_range_crc evidence")
    identity = q2[0]["archive_identity"]
    entries = []
    for item in missing:
        date = item["date"]
        name = item["name"]
        lower_bound = 30 + len(name.encode("utf-8")) + RANGE_PAD_BYTES + int(item["compressed"])
        paths = _attempt_relpaths(date, 1)
        entries.append(
            {
                "date": date,
                "name": name,
                "offset": int(item["offset"]),
                "compressed": int(item["compressed"]),
                "size": int(item["size"]),
                "crc": int(item["crc"]),
                "legacy_source_sha256": item["source_sha256"],
                "planned_range_lower_bound_bytes": lower_bound,
                "first_attempt_member": paths["member"],
                "first_attempt_receipt": paths["receipt"],
            }
        )
    plan = {
        "plan_version": "q2-reacquired-v1",
        "manifest": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_stable_hash": manifest["stable_hash"],
        "source_url": ARCHIVE_URL,
        "terms_url": TERMS_URL,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_identity": identity,
        "selected_model": MODEL,
        "scope": "87 Q2 members with legacy_manifest_only evidence; four retained members are never redownloaded",
        "member_count": len(entries),
        "legacy_compressed_bytes": sum(item["compressed"] for item in entries),
        "planned_range_lower_bound_bytes": sum(item["planned_range_lower_bound_bytes"] for item in entries),
        "budgets": {
            "retained_raw_bytes": MAX_RETAINED_BYTES,
            "network_response_bytes": MAX_NETWORK_BYTES,
            "active_temp_bytes": MAX_ACTIVE_TEMP_BYTES,
            "rss_bytes": MAX_RSS_BYTES,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "workers": 1,
        },
        "output_roots": {
            "raw": str(RAW_ROOT),
            "evidence": str(EVIDENCE_ROOT),
        },
        "entries": entries,
        "boundary": "Plan is network-free; legacy_range SHA is retained for provenance and is not expected to equal a new padded range SHA.",
        "code_sha256": code_sha256(),
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def save_plan(plan: dict, path: Path = PLAN_PATH) -> None:
    atomic_json(path, plan)


def load_plan(path: Path = PLAN_PATH, manifest_path: Path = MANIFEST_PATH) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = canonical_hash(plan, ("plan_sha256",))
    if plan.get("plan_sha256") != expected:
        raise AcquisitionError("plan SHA256 does not match its contents")
    manifest = load_manifest(manifest_path)
    if plan.get("manifest_stable_hash") != manifest.get("stable_hash"):
        raise AcquisitionError("plan belongs to a different stable manifest")
    if plan.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise AcquisitionError("stable manifest file changed after plan creation")
    if len(plan.get("entries", [])) != EXPECTED_MISSING_MEMBERS:
        raise AcquisitionError("plan does not contain exactly 87 missing members")
    return plan


def _parse_header_blocks(path: Path) -> tuple[int, dict[str, str]]:
    raw = path.read_text(encoding="iso-8859-1")
    blocks = [block for block in raw.replace("\r\n", "\n").split("\n\n") if block.strip()]
    if not blocks:
        raise AcquisitionError(f"no HTTP response headers at {path}")
    lines = blocks[-1].splitlines()
    if not lines:
        raise AcquisitionError(f"empty HTTP response header block at {path}")
    status_line = lines[0].strip().split()
    if len(status_line) < 2 or not status_line[1].isdigit():
        raise AcquisitionError(f"invalid HTTP status line at {path}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.lower().strip()] = value.strip()
    return int(status_line[1]), headers


def source_identity(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: headers[key]
        for key in ("x-bz-file-id", "etag", "last-modified", "content-md5")
        if headers.get(key)
    }


def validate_identity(observed: dict[str, str], expected: dict[str, str]) -> None:
    if not expected.get("x-bz-file-id"):
        raise AcquisitionError("expected source identity has no x-bz-file-id")
    if observed.get("x-bz-file-id") != expected["x-bz-file-id"]:
        raise AcquisitionError(
            f"source identity mismatch: {observed.get('x-bz-file-id')!r} != {expected['x-bz-file-id']!r}"
        )


def validate_range_response(
    status: int,
    headers: dict[str, str],
    path: Path,
    start: int,
    end: int,
    total_bytes: int,
    expected_identity: dict[str, str],
) -> dict:
    expected_span = end - start + 1
    expected_range = f"bytes {start}-{end}/{total_bytes}"
    if status != 206:
        raise AcquisitionError(f"expected HTTP 206, received {status}")
    if headers.get("content-range") != expected_range:
        raise AcquisitionError(
            f"Content-Range mismatch: {headers.get('content-range')!r} != {expected_range!r}"
        )
    try:
        content_length = int(headers.get("content-length", "-1"))
    except ValueError as exc:
        raise AcquisitionError("Content-Length is not an integer") from exc
    if content_length != expected_span or path.stat().st_size != expected_span:
        raise AcquisitionError(
            f"range length mismatch: header={content_length}, file={path.stat().st_size}, expected={expected_span}"
        )
    observed_identity = source_identity(headers)
    validate_identity(observed_identity, expected_identity)
    return {
        "http_status": status,
        "content_range": headers["content-range"],
        "content_length": content_length,
        "observed_range_sha256": sha256_file(path),
        "observed_source_identity": observed_identity,
    }


def _curl_range(
    url: str,
    start: int,
    end: int,
    total_bytes: int,
    member_path: Path,
    headers_path: Path,
    stderr_path: Path,
    expected_identity: dict[str, str],
) -> dict:
    member_path.parent.mkdir(parents=True, exist_ok=True)
    headers_path.parent.mkdir(parents=True, exist_ok=True)
    partial = member_path.with_name(member_path.name + ".partial")
    partial.unlink(missing_ok=True)
    headers_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    expected_span = end - start + 1
    if shutil.disk_usage(member_path.parent).free < expected_span + MIN_FREE_BYTES:
        raise AcquisitionStopped(f"free space below minimum before range {start}-{end}")
    command = [
        "curl",
        "-fLsS",
        "--max-time",
        str(HTTP_TIMEOUT_SECONDS),
        "--max-filesize",
        str(expected_span),
        "-D",
        str(headers_path),
        "-r",
        f"{start}-{end}",
        "-o",
        str(partial),
        url,
    ]
    with stderr_path.open("wb") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    if completed.returncode != 0:
        raise AcquisitionError(f"curl failed with exit code {completed.returncode}")
    status, headers = _parse_header_blocks(headers_path)
    result = validate_range_response(
        status, headers, partial, start, end, total_bytes, expected_identity
    )
    os.replace(partial, member_path)
    result["member_path"] = str(member_path.relative_to(ROOT))
    result["headers_path"] = str(headers_path.relative_to(ROOT))
    result["stderr_path"] = str(stderr_path.relative_to(ROOT))
    return result


def _curl_document(
    url: str,
    document_path: Path,
    headers_path: Path,
    stderr_path: Path,
    max_bytes: int,
) -> dict:
    document_path.parent.mkdir(parents=True, exist_ok=True)
    headers_path.parent.mkdir(parents=True, exist_ok=True)
    partial = document_path.with_name(document_path.name + ".partial")
    partial.unlink(missing_ok=True)
    headers_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    command = [
        "curl", "-fLsS", "--max-time", "60", "--max-filesize", str(max_bytes),
        "-D", str(headers_path), "-o", str(partial), url,
    ]
    with stderr_path.open("wb") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    if completed.returncode != 0:
        raise AcquisitionError(f"curl document failed with exit code {completed.returncode}")
    status, headers = _parse_header_blocks(headers_path)
    if status != 200:
        raise AcquisitionError(f"terms page expected HTTP 200, received {status}")
    if partial.stat().st_size > max_bytes:
        raise AcquisitionError("terms page exceeded registered response budget")
    os.replace(partial, document_path)
    return {
        "http_status": status,
        "content_length": document_path.stat().st_size,
        "sha256": sha256_file(document_path),
        "headers_path": str(headers_path.relative_to(ROOT)),
        "stderr_path": str(stderr_path.relative_to(ROOT)),
        "observed_headers": headers,
    }


def _eocd(blob: bytes) -> tuple[int, int, int]:
    marker = blob.rfind(b"PK\x05\x06")
    if marker < 0:
        raise AcquisitionError("ZIP end-of-central-directory record not found")
    if marker + 22 > len(blob):
        raise AcquisitionError("truncated end-of-central-directory record")
    _, disk, central_disk, entries_disk, entries_total, size, offset, comment = struct.unpack_from(
        "<4s4H2LH", blob, marker
    )
    if disk != 0 or central_disk != 0 or entries_disk != entries_total:
        raise AcquisitionError("multi-disk or inconsistent ZIP central directory")
    if comment != len(blob) - marker - 22 and marker + 22 + comment > len(blob):
        raise AcquisitionError("invalid ZIP comment length")
    return int(entries_total), int(size), int(offset)


def parse_central_directory(blob: bytes, blob_start: int, total_bytes: int) -> list[dict]:
    entries_total, central_size, central_offset = _eocd(blob)
    if not (blob_start <= central_offset and central_offset + central_size <= blob_start + len(blob)):
        raise AcquisitionError("central directory is outside the retained tail response")
    begin = central_offset - blob_start
    central = blob[begin : begin + central_size]
    entries: list[dict] = []
    cursor = 0
    for _ in range(entries_total):
        if cursor + 46 > len(central) or central[cursor : cursor + 4] != b"PK\x01\x02":
            raise AcquisitionError(f"invalid central directory entry at offset {cursor}")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        (
            _signature,
            _made_by,
            _needed,
            flags,
            method,
            _time,
            _date,
            crc,
            compressed,
            uncompressed,
            filename_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = fields
        start = cursor + 46
        finish = start + filename_length + extra_length + comment_length
        if finish > len(central):
            raise AcquisitionError("central directory entry extends past response")
        try:
            name = central[start : start + filename_length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AcquisitionError("central directory filename is not UTF-8") from exc
        cursor = finish
        if name.startswith("data_Q2_2023/") and name.endswith(".csv"):
            entries.append(
                {
                    "name": name,
                    "date": name.rsplit("/", 1)[-1][:-4],
                    "compressed": int(compressed),
                    "size": int(uncompressed),
                    "crc": int(crc),
                    "offset": int(local_offset),
                    "flags": int(flags),
                    "method": int(method),
                    "central_filename_length": int(filename_length),
                    "central_extra_length": int(extra_length),
                }
            )
    if len(entries) != len(EXPECTED_Q2_DATES) or [item["date"] for item in sorted(entries, key=lambda item: item["date"])] != EXPECTED_Q2_DATES:
        raise AcquisitionError("central directory does not contain exactly the 91 Q2 daily members")
    return sorted(entries, key=lambda item: item["date"])


def compare_directory_to_plan(entries: list[dict], plan_entries: list[dict]) -> None:
    expected = {item["date"]: item for item in plan_entries}
    for item in entries:
        old = expected.get(item["date"])
        if old is None:
            raise AcquisitionError(f"central directory has an unexpected date {item['date']}")
        for field in ("name", "offset", "compressed", "size", "crc"):
            if item[field] != old[field]:
                raise AcquisitionError(
                    f"central directory {field} changed for {item['date']}: {item[field]!r} != {old[field]!r}"
                )
        if item["flags"] & 1 or item["method"] != 8:
            raise AcquisitionError(f"unsupported/encrypted member {item['name']}")


def _parse_int(value: str, field: str, date: str, row_number: int) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise AcquisitionError(f"non-integer {field} at {date} row {row_number}") from exc


def _declared_q2_schema(columns: list[str]) -> str:
    schema = sha256_bytes(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode())
    if len(columns) != 186 or schema != "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2":
        raise AcquisitionError(f"Q2 schema mismatch: count={len(columns)} sha256={schema}")
    missing = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing:
        raise AcquisitionError(f"Q2 schema missing required fields: {missing}")
    return schema


def verify_member_file(path: Path, entry: dict, decoded_path: Path | None = None) -> dict:
    """Verify one range and stream its CSV rows through bounded disk storage."""
    if not path.is_file():
        raise AcquisitionError(f"member file is missing: {path}")
    if path.stat().st_size > MAX_RETAINED_BYTES:
        raise AcquisitionError("one retained range exceeds raw-byte budget")
    with path.open("rb") as handle:
        local_header = handle.read(30)
        if len(local_header) != 30:
            raise AcquisitionError("truncated local ZIP header")
        fields = struct.unpack("<4s5H3L2H", local_header)
        signature, _version, flags, method, _time, _date, _crc, _compressed, _size, filename_length, extra_length = fields
        if signature != b"PK\x03\x04":
            raise AcquisitionError("invalid local ZIP signature")
        if flags & 1:
            raise AcquisitionError("encrypted ZIP member")
        if method != 8:
            raise AcquisitionError(f"unsupported local ZIP compression method {method}")
        filename = handle.read(filename_length).decode("utf-8")
        handle.read(extra_length)
        if filename != entry["name"]:
            raise AcquisitionError(f"local filename mismatch: {filename!r} != {entry['name']!r}")
        data_start = 30 + filename_length + extra_length
        compressed_size = int(entry["compressed"])
        data_end = data_start + compressed_size
        if data_end > path.stat().st_size:
            raise AcquisitionError("range ends before compressed member payload")
        if decoded_path is None:
            decoded_path = path.with_name(path.name + ".decoded")
        decoded_path.unlink(missing_ok=True)
        digest = zlib.crc32(b"")
        expanded_size = 0
        decoder = zlib.decompressobj(-15)
        with decoded_path.open("wb") as decoded:
            remaining = compressed_size
            while remaining:
                compressed = handle.read(min(64 * 1024, remaining))
                if not compressed:
                    raise AcquisitionError("truncated compressed member")
                remaining -= len(compressed)
                pending = compressed
                while pending:
                    raw = decoder.decompress(pending, 64 * 1024)
                    pending = decoder.unconsumed_tail
                    if raw:
                        decoded.write(raw)
                        expanded_size += len(raw)
                        digest = zlib.crc32(raw, digest)
                        if expanded_size > MAX_ACTIVE_TEMP_BYTES:
                            raise AcquisitionError("decoded temporary file exceeds active-temp budget")
            if not decoder.eof or decoder.unused_data:
                raise AcquisitionError("deflate stream did not end exactly at the central-directory payload")
        if expanded_size != int(entry["size"]):
            raise AcquisitionError(f"expanded size mismatch: {expanded_size} != {entry['size']}")
        observed_crc = digest & 0xFFFFFFFF
        if observed_crc != int(entry["crc"]):
            raise AcquisitionError(f"CRC mismatch: {observed_crc} != {entry['crc']}")

    source_rows = 0
    selected_rows = 0
    failure_rows = 0
    serials: set[str] = set()
    with decoded_path.open("r", encoding="utf-8-sig", newline="") as decoded:
        reader = csv.reader(decoded)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise AcquisitionError("CSV member has no header") from exc
        schema_sha = _declared_q2_schema(columns)
        index = {column: position for position, column in enumerate(columns)}
        for row_number, values in enumerate(reader, start=2):
            source_rows += 1
            if len(values) != len(columns):
                raise AcquisitionError(f"row length mismatch at {entry['date']} row {row_number}")
            if values[index["date"]] != entry["date"]:
                raise AcquisitionError(f"date mismatch at {entry['date']} row {row_number}")
            serial = values[index["serial_number"]]
            model = values[index["model"]]
            if not serial or not model:
                raise AcquisitionError(f"empty serial/model at {entry['date']} row {row_number}")
            if serial in serials:
                raise AcquisitionError(f"duplicate serial_number at {entry['date']}: {serial}")
            serials.add(serial)
            failure = _parse_int(values[index["failure"]], "failure", entry["date"], row_number)
            if failure not in (0, 1):
                raise AcquisitionError(f"invalid failure at {entry['date']} row {row_number}")
            capacity = _parse_int(values[index["capacity_bytes"]], "capacity_bytes", entry["date"], row_number)
            if capacity is not None and capacity < -1:
                raise AcquisitionError(f"invalid negative capacity at {entry['date']} row {row_number}")
            if model != MODEL:
                continue
            selected_rows += 1
            failure_rows += int(failure)
            for field in SMART_FIELDS:
                _parse_int(values[index[field]], field, entry["date"], row_number)
    decoded_path.unlink(missing_ok=True)
    return {
        "source_rows": source_rows,
        "selected_rows": selected_rows,
        "failure_rows": failure_rows,
        "unique_serials": len(serials),
        "schema_columns": len(columns),
        "schema_sha256": schema_sha,
        "expanded_bytes": expanded_size,
        "crc32": observed_crc,
        "observed_compressed_payload_sha256": _compressed_payload_sha256(path, data_start, compressed_size),
        "observed_range_sha256": sha256_file(path),
        "observed_range_bytes": path.stat().st_size,
    }


def _compressed_payload_sha256(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise AcquisitionError("compressed payload ended while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def load_audit_counts(path: Path = AUDIT_DATABASE, expected_manifest_hash: str | None = None) -> dict[str, dict]:
    if not path.exists():
        raise AcquisitionError(f"retained audit database is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        if metadata.get("audit_scope") != "partial_retained_94_days":
            raise AcquisitionError("audit database is not the retained-94-day audit scope")
        if expected_manifest_hash and metadata.get("audit_manifest_stable_hash") != expected_manifest_hash:
            raise AcquisitionError("audit database belongs to a different stable manifest")
        rows = connection.execute(
            "SELECT date, source_member, source_sha256, source_rows, selected_rows, failure_rows FROM member_counts"
        ).fetchall()
        counts = {row["date"]: dict(row) | {"origin": "retained_audit"} for row in rows}
    finally:
        connection.close()
    q2_retained = [date for date in EXPECTED_Q2_DATES if date in counts]
    if len(q2_retained) != EXPECTED_RETAINED_MEMBERS:
        raise AcquisitionError(f"audit database has {len(q2_retained)} retained Q2 dates, expected 4")
    return counts


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def coverage_record(date_text: str, counts: dict[str, dict]) -> dict:
    date = dt.date.fromisoformat(date_text)
    current = counts.get(date_text)
    if current is None:
        raise AcquisitionStopped(f"coverage count missing for {date_text}")
    previous = []
    for offset in range(1, 8):
        previous_date = (date - dt.timedelta(days=offset)).isoformat()
        row = counts.get(previous_date)
        if row is None:
            raise AcquisitionStopped(f"coverage predecessor missing for {date_text}: {previous_date}")
        previous.append(row)
    source_baseline = _median([int(row["source_rows"]) for row in previous])
    selected_baseline = _median([int(row["selected_rows"]) for row in previous])
    source_ratio = float(current["source_rows"]) / source_baseline if source_baseline else 0.0
    selected_ratio = float(current["selected_rows"]) / selected_baseline if selected_baseline else 0.0
    low = source_ratio < 0.80 or selected_ratio < 0.80
    return {
        "date": date_text,
        "source_rows": int(current["source_rows"]),
        "selected_rows": int(current["selected_rows"]),
        "source_baseline_median": source_baseline,
        "selected_baseline_median": selected_baseline,
        "source_ratio": source_ratio,
        "selected_ratio": selected_ratio,
        "low": low,
        "origin": current.get("origin"),
    }


def advance_coverage(
    last_date: str,
    target_date: str,
    counts: dict[str, dict],
    low_streak: int = 0,
    low_total: int = 0,
) -> tuple[str, list[dict], int, int, str | None]:
    current = dt.date.fromisoformat(last_date)
    target = dt.date.fromisoformat(target_date)
    records: list[dict] = []
    stop_reason = None
    while current < target:
        current += dt.timedelta(days=1)
        record = coverage_record(current.isoformat(), counts)
        records.append(record)
        if record["low"]:
            low_streak += 1
            low_total += 1
        else:
            low_streak = 0
        if low_streak >= 3:
            stop_reason = f"three_consecutive_low_coverage_days_through:{record['date']}"
            break
        if low_total > 9:
            stop_reason = f"more_than_9_low_coverage_days_through:{record['date']}"
            break
    return current.isoformat(), records, low_streak, low_total, stop_reason


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _attempt_number(date: str) -> int:
    parent = RAW_ROOT / "attempts" / date
    if not parent.exists():
        return 1
    numbers = []
    for path in parent.glob("attempt-*"):
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            raise AcquisitionError(f"invalid attempt directory: {path}")
    return max(numbers, default=0) + 1


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _receipt_paths(date: str, attempt: int) -> dict[str, Path]:
    rel = _attempt_relpaths(date, attempt)
    return {key: ROOT / value for key, value in rel.items()}


def _new_receipt(entry: dict, central: dict, attempt: int, plan: dict) -> dict:
    paths = _receipt_paths(entry["date"], attempt)
    start = int(central["offset"])
    end = min(
        plan["archive_bytes"] - 1,
        start + 30 + int(central["central_filename_length"]) + int(central["central_extra_length"])
        + RANGE_PAD_BYTES + int(central["compressed"]) - 1,
    )
    return {
        "receipt_version": "q2-reacquired-receipt-v1",
        "status": "planned",
        "date": entry["date"],
        "name": entry["name"],
        "attempt": attempt,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "code_sha256": code_sha256(),
        "legacy_source_sha256": entry["legacy_source_sha256"],
        "request": {
            "url": plan["source_url"],
            "range_start": start,
            "range_end": end,
            "expected_span": end - start + 1,
            "archive_bytes": plan["archive_bytes"],
            "source_identity": plan["archive_identity"],
        },
        "expected": {
            "offset": entry["offset"],
            "compressed": entry["compressed"],
            "size": entry["size"],
            "crc": entry["crc"],
            "schema_columns": 186,
            "schema_sha256": "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2",
        },
        "observed": {},
        "paths": {key: _rel(value) for key, value in paths.items()},
    }


def _initial_index(plan: dict) -> dict:
    return {
        "index_version": "q2-reacquired-index-v1",
        "status": "planned",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "updated_at_utc": None,
        "code_sha256": code_sha256(),
        "plan_sha256": plan["plan_sha256"],
        "manifest_stable_hash": plan["manifest_stable_hash"],
        "source_url": plan["source_url"],
        "archive_bytes": plan["archive_bytes"],
        "archive_identity": plan["archive_identity"],
        "terms": None,
        "source_directory": None,
        "member_states": {},
        "coverage": [],
        "coverage_last_date": "2023-03-31",
        "low_streak": 0,
        "low_total": 0,
        "network_response_bytes": 0,
        "retained_range_bytes": 0,
        "stop_reason": None,
        "boundary": "Original members and receipts are retained; no panel, labels or model writes are permitted.",
    }


def _load_or_create_index(plan: dict, path: Path) -> dict:
    if not path.exists():
        return _initial_index(plan)
    index = json.loads(path.read_text(encoding="utf-8"))
    for key in ("plan_sha256", "manifest_stable_hash", "archive_identity"):
        if index.get(key) != plan.get(key):
            raise AcquisitionError(f"existing index {key} differs from plan")
    return index


def _write_index(index: dict, path: Path) -> None:
    index["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    atomic_json(path, index)


def _discover_receipts() -> list[Path]:
    if not EVIDENCE_ROOT.exists():
        return []
    return sorted(EVIDENCE_ROOT.glob("attempts/*/attempt-*/receipt.json"))


def _merge_discovered_receipts(index: dict, allowed_dates: set[str]) -> None:
    for receipt_path in _discover_receipts():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        date = receipt.get("date")
        if not date:
            raise AcquisitionError(f"receipt has no date: {receipt_path}")
        if date not in allowed_dates:
            raise AcquisitionError(f"receipt date is outside the planned missing scope: {date}")
        state = index["member_states"].setdefault(date, {"status": "pending", "receipts": []})
        rel = _rel(receipt_path)
        if rel not in state["receipts"]:
            state["receipts"].append(rel)
        if receipt.get("status") == "verified":
            state["status"] = "verified"
            state["verified_receipt"] = rel
        elif state.get("status") != "verified":
            state["status"] = receipt.get("status", "unverified")


def _reconcile_verified_receipt(receipt_path: Path, entry: dict | None = None) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "verified":
        raise AcquisitionError(f"receipt is not verified: {receipt_path}")
    if entry is not None:
        if receipt.get("date") != entry.get("date") or receipt.get("name") != entry.get("name"):
            raise AcquisitionError(f"verified receipt identity mismatch: {receipt_path}")
        if receipt.get("legacy_source_sha256") != entry.get("legacy_source_sha256"):
            raise AcquisitionError(f"verified receipt belongs to a different legacy source: {receipt_path}")
        expected = receipt.get("expected", {})
        for field in ("compressed", "size", "crc", "schema_columns", "schema_sha256"):
            expected_value = entry.get(field) if field in entry else (
                186 if field == "schema_columns" else
                "a52fdf4ffdb8230a5bf4143018731c59342cb33b311a8a4a77661af8c9963bb2" if field == "schema_sha256" else None
            )
            if expected_value is not None and expected.get(field) != expected_value:
                raise AcquisitionError(f"verified receipt expected {field} mismatch: {receipt_path}")
    member = ROOT / receipt["paths"]["member"]
    expected = receipt["observed"].get("observed_range_sha256")
    if not expected or not member.is_file() or sha256_file(member) != expected:
        raise AcquisitionError(f"verified member bytes no longer match receipt: {receipt_path}")
    if entry is not None:
        verification = verify_member_file(member, entry, ROOT / receipt["paths"]["decoded"])
        for field in ("source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256", "observed_range_sha256"):
            if verification.get(field) != receipt.get("observed", {}).get(field):
                raise AcquisitionError(f"verified receipt observation mismatch for {field}: {receipt_path}")
    return receipt


def _orphan_artifacts(date: str) -> list[str]:
    raw_parent = RAW_ROOT / "attempts" / date
    evidence_parent = EVIDENCE_ROOT / "attempts" / date
    receipts = {path.parent.name for path in evidence_parent.glob("attempt-*/receipt.json")} if evidence_parent.exists() else set()
    orphans = []
    if raw_parent.exists():
        for attempt in raw_parent.glob("attempt-*"):
            if attempt.name not in receipts and any(path.is_file() for path in attempt.iterdir()):
                orphans.extend(_rel(path) for path in attempt.iterdir() if path.is_file())
    return sorted(orphans)


def _source_control(plan: dict, index: dict) -> tuple[list[dict], dict]:
    source_dir = EVIDENCE_ROOT / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    terms_path = source_dir / "terms.html"
    terms_headers = source_dir / "terms.headers"
    terms_stderr = source_dir / "terms.stderr"
    if terms_path.exists() and terms_headers.exists():
        status, headers = _parse_header_blocks(terms_headers)
        if status != 200:
            raise AcquisitionError("retained terms response is not HTTP 200")
        terms = {
            "url": plan["terms_url"],
            "http_status": status,
            "content_length": terms_path.stat().st_size,
            "sha256": sha256_file(terms_path),
            "document_path": _rel(terms_path),
            "headers_path": _rel(terms_headers),
            "stderr_path": _rel(terms_stderr),
        }
    else:
        terms_result = _curl_document(plan["terms_url"], terms_path, terms_headers, terms_stderr, TERMS_MAX_BYTES)
        terms = {"url": plan["terms_url"], "document_path": _rel(terms_path), **terms_result}
    index["terms"] = terms

    head_headers = source_dir / "head.headers"
    head_stderr = source_dir / "head.stderr"
    head_partial = source_dir / "head.body"
    command = ["curl", "-fLsS", "--max-time", "60", "-D", str(head_headers), "-o", str(head_partial), "-I", plan["source_url"]]
    with head_stderr.open("wb") as stderr_handle:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle, check=False)
    if completed.returncode != 0:
        raise AcquisitionError(f"Q2 HEAD failed with exit code {completed.returncode}")
    status, headers = _parse_header_blocks(head_headers)
    if status != 200:
        raise AcquisitionError(f"Q2 HEAD expected HTTP 200, received {status}")
    try:
        archive_bytes = int(headers.get("content-length", "-1"))
    except ValueError as exc:
        raise AcquisitionError("Q2 HEAD Content-Length is not an integer") from exc
    if archive_bytes != plan["archive_bytes"]:
        raise AcquisitionError(f"Q2 archive size changed: {archive_bytes} != {plan['archive_bytes']}")
    identity = source_identity(headers)
    validate_identity(identity, plan["archive_identity"])
    head_partial.unlink(missing_ok=True)
    index["source_head"] = {
        "url": plan["source_url"],
        "http_status": status,
        "archive_bytes": archive_bytes,
        "identity": identity,
        "headers_path": _rel(head_headers),
        "stderr_path": _rel(head_stderr),
        "headers_sha256": sha256_file(head_headers),
    }

    tail_size = min(plan["archive_bytes"], 4 * 1024 * 1024)
    tail_start = plan["archive_bytes"] - tail_size
    tail_path = source_dir / "central_tail.bin"
    tail_headers = source_dir / "central_tail.headers"
    tail_stderr = source_dir / "central_tail.stderr"
    if not tail_path.exists() or not tail_headers.exists():
        tail_result = _curl_range(
            plan["source_url"], tail_start, plan["archive_bytes"] - 1, plan["archive_bytes"],
            tail_path, tail_headers, tail_stderr, plan["archive_identity"],
        )
    else:
        tail_status, tail_response_headers = _parse_header_blocks(tail_headers)
        tail_result = validate_range_response(
            tail_status, tail_response_headers, tail_path, tail_start, plan["archive_bytes"] - 1,
            plan["archive_bytes"], plan["archive_identity"],
        )
        tail_result.update({"member_path": _rel(tail_path), "headers_path": _rel(tail_headers), "stderr_path": _rel(tail_stderr)})
    try:
        entries = parse_central_directory(tail_path.read_bytes(), tail_start, plan["archive_bytes"])
    except AcquisitionError as first_error:
        # If the central directory is outside a 4 MiB tail, retain the failed
        # tail and stop. The Q2 archive is expected to fit; expanding blindly
        # would violate the registered network budget.
        raise AcquisitionError(f"central directory parse failed from retained tail: {first_error}") from first_error
    # The central directory must describe the complete 91-day archive. The
    # acquisition plan intentionally contains only the 87 dates without local
    # bytes, so comparing against it alone would misclassify the four retained
    # dates as unexpected members.
    manifest = load_manifest(Path(plan["manifest"]))
    full_entries = [
        {
            "date": item["date"],
            "name": item["name"],
            "offset": int(item["offset"]),
            "compressed": int(item["compressed"]),
            "size": int(item["size"]),
            "crc": int(item["crc"]),
        }
        for item in _q2_members(manifest)
    ]
    compare_directory_to_plan(entries, full_entries)
    directory = {
        "entries": entries,
        "tail": tail_result,
        "tail_sha256": sha256_file(tail_path),
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    index["source_directory"] = directory
    return entries, directory


def _baseline_counts_with_receipts(plan: dict) -> dict[str, dict]:
    counts = load_audit_counts(expected_manifest_hash=plan["manifest_stable_hash"])
    for entry in plan["entries"]:
        state = _find_verified_receipt(entry["date"], entry)
        if state is not None:
            counts[entry["date"]] = {
                "date": entry["date"],
                "source_rows": state["observed"]["source_rows"],
                "selected_rows": state["observed"]["selected_rows"],
                "failure_rows": state["observed"]["failure_rows"],
                "origin": "reacquired_receipt",
            }
    return counts


def _find_verified_receipt(date: str, entry: dict | None = None) -> dict | None:
    for path in sorted(EVIDENCE_ROOT.glob(f"attempts/{date}/attempt-*/receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") == "verified":
            _reconcile_verified_receipt(path, entry)
            return receipt
    return None


def _recompute_byte_totals(index: dict) -> None:
    network = 0
    retained = 0
    source = index.get("source_head") or {}
    head_path = source.get("headers_path")
    if head_path and (ROOT / head_path).exists():
        network += (ROOT / head_path).stat().st_size
    terms = index.get("terms") or {}
    terms_path = terms.get("headers_path")
    if terms_path and (ROOT / terms_path).exists():
        network += (ROOT / terms_path).stat().st_size
    terms_document = terms.get("document_path")
    if terms_document and (ROOT / terms_document).exists():
        network += (ROOT / terms_document).stat().st_size
    directory = index.get("source_directory") or {}
    tail = directory.get("tail") or {}
    for key in ("member_path", "headers_path", "stderr_path"):
        path = tail.get(key)
        if path and (ROOT / path).exists():
            network += (ROOT / path).stat().st_size
    for state in index.get("member_states", {}).values():
        for receipt_rel in state.get("receipts", []):
            receipt_path = ROOT / receipt_rel
            if not receipt_path.exists():
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for key in ("member", "partial", "headers", "stderr"):
                path_rel = receipt.get("paths", {}).get(key)
                if path_rel and (ROOT / path_rel).exists():
                    size = (ROOT / path_rel).stat().st_size
                    network += size
                    if key in {"member", "partial"}:
                        retained += size
    index["network_response_bytes"] = network
    index["retained_range_bytes"] = retained


def execute(plan: dict, index_path: Path = EVIDENCE_ROOT / "index_v1.json", *, retry_failed: bool = False) -> dict:
    if plan.get("member_count") != EXPECTED_MISSING_MEMBERS:
        raise AcquisitionError("execute requires a plan for exactly 87 members")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    index = _load_or_create_index(plan, index_path)
    planned_dates = {entry["date"] for entry in plan["entries"]}
    _merge_discovered_receipts(index, planned_dates)
    _recompute_byte_totals(index)
    if index["network_response_bytes"] > MAX_NETWORK_BYTES or index["retained_range_bytes"] > MAX_RETAINED_BYTES:
        raise AcquisitionStopped("existing evidence already exceeds a registered byte budget")
    if _rss_bytes() > MAX_RSS_BYTES:
        raise AcquisitionStopped("RSS exceeds registered stop line before execution")
    _write_index(index, index_path)

    try:
        entries, _directory = _source_control(plan, index)
    except BaseException as exc:
        index["status"] = "stopped"
        index["stop_reason"] = f"source_control_failed:{type(exc).__name__}"
        index["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _recompute_byte_totals(index)
        _write_index(index, index_path)
        return index
    _write_index(index, index_path)
    _recompute_byte_totals(index)
    if index["network_response_bytes"] > MAX_NETWORK_BYTES:
        index["status"] = "stopped"
        index["stop_reason"] = "network_budget_reached_after_source_control"
        _write_index(index, index_path)
        return index
    counts = _baseline_counts_with_receipts(plan)
    coverage_last = index.get("coverage_last_date", "2023-03-31")
    low_streak = int(index.get("low_streak", 0))
    low_total = int(index.get("low_total", 0))
    existing_coverage = {record["date"] for record in index.get("coverage", [])}
    index["status"] = "running"

    for entry in plan["entries"]:
        date = entry["date"]
        state = index["member_states"].setdefault(date, {"status": "pending", "receipts": []})
        if state.get("status") == "verified":
            receipt_rel = state.get("verified_receipt")
            if not receipt_rel:
                raise AcquisitionError(f"verified state has no receipt for {date}")
            _reconcile_verified_receipt(ROOT / receipt_rel, entry)
            receipt = json.loads((ROOT / receipt_rel).read_text(encoding="utf-8"))
            counts[date] = {
                "date": date,
                "source_rows": receipt["observed"]["source_rows"],
                "selected_rows": receipt["observed"]["selected_rows"],
                "failure_rows": receipt["observed"]["failure_rows"],
                "origin": "reacquired_receipt",
            }
            if date not in existing_coverage:
                coverage_last, records, low_streak, low_total, stop = advance_coverage(
                    coverage_last, date, counts, low_streak, low_total
                )
                index["coverage"].extend(records)
                existing_coverage.update(record["date"] for record in records)
                index.update({"coverage_last_date": coverage_last, "low_streak": low_streak, "low_total": low_total})
                if stop:
                    index["status"] = "stopped"
                    index["stop_reason"] = stop
                    _write_index(index, index_path)
                    break
            continue
        if state.get("status") in {"planned", "running"}:
            raise AcquisitionStopped(f"unresolved in-flight receipt for {date}")
        if state.get("status") == "failed" and not retry_failed:
            index["status"] = "stopped"
            index["stop_reason"] = f"failed_evidence_requires_explicit_retry:{date}"
            _write_index(index, index_path)
            break
        orphans = _orphan_artifacts(date)
        if orphans:
            state["status"] = "unverified_orphan"
            state["orphans"] = orphans
            index["status"] = "stopped"
            index["stop_reason"] = f"unverified_orphan_requires_review:{date}"
            _write_index(index, index_path)
            break

        # The gate is evaluated before the next request, including retained
        # dates between two missing dates. No future count is used here.
        if coverage_last < (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat():
            coverage_last, records, low_streak, low_total, stop = advance_coverage(
                coverage_last,
                (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat(),
                counts,
                low_streak,
                low_total,
            )
            index["coverage"].extend(records)
            existing_coverage.update(record["date"] for record in records)
            index.update({"coverage_last_date": coverage_last, "low_streak": low_streak, "low_total": low_total})
            if stop:
                index["status"] = "stopped"
                index["stop_reason"] = stop
                _write_index(index, index_path)
                break
        if _rss_bytes() > MAX_RSS_BYTES:
            raise AcquisitionStopped("RSS exceeds registered stop line before member request")
        central = entries[EXPECTED_Q2_DATES.index(date)]
        attempt = _attempt_number(date)
        receipt = _new_receipt(entry, central, attempt, plan)
        receipt_path = ROOT / receipt["paths"]["receipt"]
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, receipt)
        state["status"] = "planned"
        state.setdefault("receipts", []).append(_rel(receipt_path))
        _write_index(index, index_path)
        expected_span = receipt["request"]["expected_span"]
        _recompute_byte_totals(index)
        if index["network_response_bytes"] + expected_span > MAX_NETWORK_BYTES:
            index["status"] = "stopped"
            index["stop_reason"] = f"network_budget_reached_before:{date}"
            _write_index(index, index_path)
            break
        if index["retained_range_bytes"] + expected_span > MAX_RETAINED_BYTES:
            index["status"] = "stopped"
            index["stop_reason"] = f"retained_budget_reached_before:{date}"
            _write_index(index, index_path)
            break
        paths = _receipt_paths(date, attempt)
        try:
            range_observed = _curl_range(
                plan["source_url"], receipt["request"]["range_start"], receipt["request"]["range_end"],
                plan["archive_bytes"], paths["member"], paths["headers"], paths["stderr"], plan["archive_identity"],
            )
            observed = verify_member_file(paths["member"], {**entry, **central}, paths["decoded"])
            observed.update(range_observed)
            receipt["observed"] = observed
            receipt["status"] = "verified"
            receipt["verified_at_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
            atomic_json(receipt_path, receipt)
            state["status"] = "verified"
            state["verified_receipt"] = _rel(receipt_path)
            counts[date] = {
                "date": date,
                "source_rows": observed["source_rows"],
                "selected_rows": observed["selected_rows"],
                "failure_rows": observed["failure_rows"],
                "origin": "reacquired_receipt",
            }
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["failed_at_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            member_path = paths["member"]
            partial_path = paths["partial"]
            receipt["observed"] = {
                "observed_partial_bytes": partial_path.stat().st_size if partial_path.exists() else 0,
                "observed_partial_sha256": sha256_file(partial_path) if partial_path.exists() else None,
                "member_bytes": member_path.stat().st_size if member_path.exists() else 0,
                "member_sha256": sha256_file(member_path) if member_path.exists() else None,
            }
            atomic_json(receipt_path, receipt)
            state["status"] = "failed"
            index["status"] = "stopped"
            index["stop_reason"] = f"member_failed_requires_review:{date}"
            _recompute_byte_totals(index)
            _write_index(index, index_path)
            break
        _recompute_byte_totals(index)
        state["status"] = "verified"
        if date not in existing_coverage:
            coverage_last, records, low_streak, low_total, stop = advance_coverage(
                coverage_last, date, counts, low_streak, low_total
            )
            index["coverage"].extend(records)
            existing_coverage.update(record["date"] for record in records)
            index.update({"coverage_last_date": coverage_last, "low_streak": low_streak, "low_total": low_total})
            if stop:
                index["status"] = "stopped"
                index["stop_reason"] = stop
                _write_index(index, index_path)
                break
        _write_index(index, index_path)
    else:
        index["status"] = "complete"
        index["stop_reason"] = None
        _write_index(index, index_path)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="write a network-free fixed 87-member plan")
    mode.add_argument("--execute", action="store_true", help="perform the explicitly requested network acquisition")
    parser.add_argument("--retry-failed", action="store_true", help="create new attempts for prior failed dates")
    parser.add_argument("--index", type=Path, default=EVIDENCE_ROOT / "index_v1.json", help="separate index path for an explicit new run")
    args = parser.parse_args(argv)
    if args.retry_failed and not args.execute:
        parser.error("--retry-failed requires --execute")
    manifest = load_manifest()
    if not args.execute:
        plan = build_plan(manifest)
        save_plan(plan)
        print(json.dumps({
            "status": "planned",
            "plan": str(PLAN_PATH),
            "member_count": plan["member_count"],
            "legacy_compressed_bytes": plan["legacy_compressed_bytes"],
            "planned_range_lower_bound_bytes": plan["planned_range_lower_bound_bytes"],
            "network_access": False,
        }, ensure_ascii=False))
        return 0
    plan = load_plan()
    index = execute(plan, index_path=args.index, retry_failed=args.retry_failed)
    print(json.dumps({
        "status": index["status"],
        "member_count": sum(1 for value in index["member_states"].values() if value.get("status") == "verified"),
        "network_response_bytes": index.get("network_response_bytes"),
        "retained_range_bytes": index.get("retained_range_bytes"),
        "stop_reason": index.get("stop_reason"),
        "index": str(args.index),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
