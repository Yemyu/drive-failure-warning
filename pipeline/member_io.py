"""Read the bounded ZIP-member downloads recorded in a source manifest."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import re
import struct
import zlib
from typing import Iterator


ROOT = pathlib.Path(__file__).resolve().parents[1]
SMART_IDS = (5, 9, 187, 188, 197, 198)
SMART_FIELDS = tuple(f"smart_{value}_raw" for value in SMART_IDS)
REQUIRED_FIELDS = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    *SMART_FIELDS,
)
DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\.csv$")


class MemberError(RuntimeError):
    """A source member failed an integrity or schema check."""


def load_manifest(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def manifest_entries(manifest: dict) -> list[dict]:
    entries = []
    for entry in manifest["members"]:
        name = entry["name"]
        match = DATE_RE.search(name)
        if not match:
            continue
        entry = dict(entry)
        entry["date"] = match.group("date")
        entries.append(entry)
    entries.sort(key=lambda item: item["date"])
    if not entries:
        raise MemberError("manifest has no daily CSV members")
    dates = [entry["date"] for entry in entries]
    if len(set(dates)) != len(dates):
        raise MemberError("manifest has duplicate daily dates")
    return entries


def _member_payload(entry: dict, root: pathlib.Path = ROOT) -> tuple[bytes, str]:
    local = entry.get("local")
    if not local:
        raise MemberError(f"manifest entry has no local file: {entry['name']}")
    root = root.resolve()
    path = (root / local).resolve()
    if not path.is_relative_to(root):
        raise MemberError(f"manifest local path escapes root: {entry['local']!r}")
    if path.is_symlink():
        raise MemberError(f"manifest local file must not be a symlink: {path}")
    if not path.is_file():
        raise MemberError(f"missing local member: {path}")
    payload = path.read_bytes()
    expected = entry.get("sha256")
    actual = hashlib.sha256(payload).hexdigest()
    if expected and actual != expected:
        raise MemberError(f"SHA256 mismatch for {entry['name']}: {actual} != {expected}")
    return payload, actual


def read_member(entry: dict, root: pathlib.Path = ROOT) -> tuple[str, str, list[str], list[list[str]], dict]:
    """Return date, member hash, header, rows, and integrity diagnostics.

    The local files are bounded HTTP ranges beginning at a ZIP local-file header.
    The central-directory compressed and uncompressed sizes in the manifest are
    authoritative; bytes after the compressed payload are ignored.
    """

    payload, actual_sha256 = _member_payload(entry, root)
    if len(payload) < 30 or payload[:4] != b"PK\x03\x04":
        raise MemberError(f"invalid local ZIP header for {entry['name']}")
    header = struct.unpack_from("<4s5H3L2H", payload, 0)
    _, _, flags, method, _, _, _, _, _, filename_length, extra_length = header
    if method != 8:
        raise MemberError(f"unsupported compression method {method}: {entry['name']}")
    try:
        member_name = payload[30 : 30 + filename_length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemberError(f"invalid local ZIP filename: {entry['name']}") from exc
    if member_name != entry["name"]:
        raise MemberError(f"local ZIP filename mismatch: {member_name!r} != {entry['name']!r}")
    if entry.get("range_bytes") is not None and len(payload) != int(entry["range_bytes"]):
        raise MemberError(f"range byte count mismatch for {entry['name']}")
    data_start = 30 + filename_length + extra_length
    compressed_size = int(entry["compressed"])
    uncompressed_size = int(entry["size"])
    data_end = data_start + compressed_size
    if data_end > len(payload):
        raise MemberError(f"truncated member {entry['name']}: {len(payload)} < {data_end}")
    raw = zlib.decompress(payload[data_start:data_end], -15)
    if len(raw) != uncompressed_size:
        raise MemberError(f"size mismatch for {entry['name']}: {len(raw)} != {uncompressed_size}")
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    if crc != int(entry["crc"]):
        raise MemberError(f"CRC mismatch for {entry['name']}: {crc} != {entry['crc']}")
    stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="")
    reader = csv.reader(stream)
    try:
        columns = next(reader)
    except StopIteration as exc:
        raise MemberError(f"empty CSV member: {entry['name']}") from exc
    missing = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing:
        raise MemberError(f"missing required columns in {entry['name']}: {missing}")
    rows = list(reader)
    return entry["date"], actual_sha256, columns, rows, {
        "payload_bytes": len(payload),
        "data_start": data_start,
        "compressed_bytes": compressed_size,
        "uncompressed_bytes": uncompressed_size,
        "crc32": crc,
    }


def iter_member_rows(entry: dict, root: pathlib.Path = ROOT) -> tuple[str, str, list[str], Iterator[tuple[int, list[str]]], dict]:
    """Stream validated CSV rows from a bounded member.

    The compressed member is decompressed once, but rows are yielded rather than
    materialized so the panel writer can keep its own memory bounded.
    """

    payload, actual_sha256 = _member_payload(entry, root)
    if len(payload) < 30 or payload[:4] != b"PK\x03\x04":
        raise MemberError(f"invalid local ZIP header for {entry['name']}")
    header = struct.unpack_from("<4s5H3L2H", payload, 0)
    _, _, flags, method, _, _, _, _, _, filename_length, extra_length = header
    if method != 8:
        raise MemberError(f"unsupported local ZIP flags/method for {entry['name']}")
    try:
        member_name = payload[30 : 30 + filename_length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemberError(f"invalid local ZIP filename: {entry['name']}") from exc
    if member_name != entry["name"]:
        raise MemberError(f"local ZIP filename mismatch: {member_name!r} != {entry['name']!r}")
    if entry.get("range_bytes") is not None and len(payload) != int(entry["range_bytes"]):
        raise MemberError(f"range byte count mismatch for {entry['name']}")
    data_start = 30 + filename_length + extra_length
    compressed_size = int(entry["compressed"])
    uncompressed_size = int(entry["size"])
    data_end = data_start + compressed_size
    if data_end > len(payload):
        raise MemberError(f"truncated member {entry['name']}: {len(payload)} < {data_end}")
    compressed = payload[data_start:data_end]
    raw = zlib.decompress(compressed, -15)
    if len(raw) != uncompressed_size:
        raise MemberError(f"size mismatch for {entry['name']}: {len(raw)} != {uncompressed_size}")
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    if crc != int(entry["crc"]):
        raise MemberError(f"CRC mismatch for {entry['name']}: {crc} != {entry['crc']}")
    stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="")
    reader = csv.reader(stream)
    try:
        columns = next(reader)
    except StopIteration as exc:
        raise MemberError(f"empty CSV member: {entry['name']}") from exc
    missing = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing:
        raise MemberError(f"missing required columns in {entry['name']}: {missing}")

    def rows() -> Iterator[tuple[int, list[str]]]:
        for source_row, values in enumerate(reader, start=2):
            yield source_row, values

    return entry["date"], actual_sha256, columns, rows(), {
        "payload_bytes": len(payload),
        "data_start": data_start,
        "compressed_bytes": compressed_size,
        "uncompressed_bytes": uncompressed_size,
        "crc32": crc,
    }
