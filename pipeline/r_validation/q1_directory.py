"""Bounded ZIP central-directory preflight for the official Q1 object.

The preflight reads at most the final 16 MiB of the remote object.  It never
opens a CSV member and it refuses a server that ignores the requested range.
The resulting member map is a release input, rather than an informal note
about what the archive happened to contain on one machine.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import struct
from pathlib import Path
import urllib.request

from pipeline.reproducible.artifacts import bound_path, write_json_exclusive
from pipeline.r_validation.acquisition import NoRedirect, _verified_opener
from pipeline.r_validation.release import ReleaseError

MAX_RANGE_BYTES = 16 * 1024 * 1024
_EOCD = b"PK\x05\x06"
_CENTRAL = b"PK\x01\x02"
_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _dates(start: str, end: str) -> list[str]:
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if last < first:
        raise ReleaseError("directory date range is reversed")
    return [(first + dt.timedelta(days=i)).isoformat()
            for i in range((last - first).days + 1)]


def expected_members(prefix: str, start: str, end: str) -> tuple[set[str], set[str]]:
    """Return the exact CSV members and known macOS metadata exceptions."""
    days = _dates(start, end)
    csv_members = {f"{prefix}/{day}.csv" for day in days}
    metadata = {
        f"{prefix}/",
        "__MACOSX/._" + prefix,
        f"{prefix}/.DS_Store",
        f"__MACOSX/{prefix}/._.DS_Store",
    }
    metadata.update(f"__MACOSX/{prefix}/._{day}.csv" for day in days)
    return csv_members, metadata


def _zip64_extra(extra: bytes, *, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    """Resolve ZIP64 values when a central-directory field uses 0xffffffff."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        payload = extra[pos + 4:pos + 4 + size]
        pos += 4 + size
        if tag != 0x0001:
            continue
        cursor = 0
        if usize == 0xffffffff:
            if cursor + 8 > len(payload):
                raise ReleaseError("truncated ZIP64 uncompressed size")
            usize = struct.unpack_from("<Q", payload, cursor)[0]; cursor += 8
        if csize == 0xffffffff:
            if cursor + 8 > len(payload):
                raise ReleaseError("truncated ZIP64 compressed size")
            csize = struct.unpack_from("<Q", payload, cursor)[0]; cursor += 8
        if offset == 0xffffffff:
            if cursor + 8 > len(payload):
                raise ReleaseError("truncated ZIP64 local offset")
            offset = struct.unpack_from("<Q", payload, cursor)[0]
        return usize, csize, offset
    if 0xffffffff in (usize, csize, offset):
        raise ReleaseError("ZIP64 central entry has no usable ZIP64 extra")
    return usize, csize, offset


def parse_central_directory(tail: bytes, *, range_start: int, object_length: int) -> dict:
    """Parse EOCD and every central record from a bounded tail response."""
    if not isinstance(tail, bytes) or len(tail) > MAX_RANGE_BYTES:
        raise ReleaseError("directory range body exceeds the 16 MiB metadata budget")
    eocd = tail.rfind(_EOCD)
    if eocd < 0 or eocd + 22 > len(tail):
        raise ReleaseError("ZIP end-of-central-directory record is outside the range")
    _, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_len = \
        struct.unpack_from("<4s4H2LH", tail, eocd)
    if disk != 0 or cd_disk != 0 or disk_entries != total_entries:
        raise ReleaseError("multi-disk ZIP central directory is not supported")
    if eocd + 22 + comment_len > len(tail):
        raise ReleaseError("truncated ZIP end-of-central-directory comment")
    local_offset = cd_offset - range_start
    if local_offset < 0 or local_offset + cd_size > len(tail):
        raise ReleaseError("central directory is not fully contained in the requested range")
    entries = []
    cursor = local_offset
    for _ in range(total_entries):
        if cursor + 46 > len(tail) or tail[cursor:cursor + 4] != _CENTRAL:
            raise ReleaseError("invalid ZIP central-directory record")
        values = struct.unpack_from("<4s6H3L5H2L", tail, cursor)
        (_, made, needed, flags, method, mod_time, mod_date, crc32,
         compressed, uncompressed, name_len, extra_len, comment_len,
         disk_start, internal_attr, external_attr, local_header) = values
        start = cursor + 46
        end = start + name_len + extra_len + comment_len
        if end > len(tail):
            raise ReleaseError("truncated ZIP central-directory name or extra")
        raw_name = tail[start:start + name_len]
        try:
            name = raw_name.decode("utf-8") if flags & 0x800 else raw_name.decode("cp437")
        except UnicodeDecodeError as exc:
            raise ReleaseError("ZIP member name is not decodable") from exc
        extra = tail[start + name_len:start + name_len + extra_len]
        uncompressed, compressed, local_header = _zip64_extra(
            extra, usize=uncompressed, csize=compressed, offset=local_header)
        entries.append({
            "name": name,
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "local_header_offset": local_header,
            "crc32": crc32,
            "compression": method,
            "flags": flags,
            "directory": name.endswith("/"),
        })
        cursor = end
    if cursor != local_offset + cd_size:
        raise ReleaseError("ZIP central-directory size does not match its records")
    return {
        "eocd_offset_in_range": eocd,
        "central_directory_offset": cd_offset,
        "central_directory_bytes": cd_size,
        "entry_count": total_entries,
        "entries": entries,
    }


def _response_range(response, *, expected_start: int, expected_end: int, expected_total: int) -> dict:
    headers = {key.lower(): value for key, value in response.headers.items()}
    content_range = headers.get("content-range")
    match = _RANGE.fullmatch(content_range or "")
    if response.status != 206 or not match:
        raise ReleaseError("directory preflight requires a 206 Content-Range response")
    start, end, total = (int(value) for value in match.groups())
    if (start, end, total) != (expected_start, expected_end, expected_total):
        raise ReleaseError("directory Content-Range does not match the approved object")
    if headers.get("content-length") != str(expected_end - expected_start + 1):
        raise ReleaseError("directory response length differs from the requested range")
    if headers.get("content-encoding", "identity") != "identity" or "transfer-encoding" in headers:
        raise ReleaseError("directory response framing is not identity and bounded")
    return {"status": response.status, "headers": headers}


def fetch_directory(source: dict, *, opener=None, timeout: int = 30) -> tuple[dict, bytes]:
    """Fetch and parse only the approved central-directory range."""
    length = source.get("content_length")
    if type(length) is not int or length <= 0:
        raise ReleaseError("source length is not an integer")
    span = min(MAX_RANGE_BYTES, length)
    start, end = length - span, length - 1
    request = urllib.request.Request(
        source["url"], headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    opener_obj = opener or _verified_opener()
    try:
        with opener_obj.open(request, timeout=min(timeout, 30)) as response:
            response_facts = _response_range(response, expected_start=start,
                                             expected_end=end, expected_total=length)
            headers = response_facts["headers"]
            if (response.geturl() != source["url"]
                    or headers.get("x-bz-file-id") != source["bz_file_id"]
                    or headers.get("x-bz-content-sha1") != source["content_sha1"]
                    or headers.get("accept-ranges") != source["accept_ranges"]):
                raise ReleaseError("directory response identity differs from the approved object")
            tail = response.read(span + 1)
    except ReleaseError:
        raise
    except Exception as exc:
        raise ReleaseError(f"directory range request failed: {exc}") from exc
    if len(tail) != span:
        raise ReleaseError("directory range body is truncated")
    parsed = parse_central_directory(tail, range_start=start, object_length=length)
    return {"range": {"start": start, "end": end, "bytes": span},
            "response": response_facts, "tail_sha256": hashlib.sha256(tail).hexdigest(),
            "central_directory": parsed}, tail


def build_directory_proof(source: dict, *, start: str, end: str, opener=None) -> dict:
    """Return a stable, JSON-safe directory proof and reject unknown members."""
    facts, _ = fetch_directory(source, opener=opener)
    entries = facts["central_directory"]["entries"]
    actual = {entry["name"] for entry in entries}
    csv_members, metadata = expected_members(source["prefix"], start, end)
    expected = csv_members | metadata
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    csv_names = sorted(name for name in actual if name in csv_members)
    if missing or unexpected or len(csv_names) != len(csv_members):
        raise ReleaseError(
            f"Q1 directory mapping differs: missing={missing[:5]} unexpected={unexpected[:5]} "
            f"csv_count={len(csv_names)}")
    facts.update({
        "schema": "q1-directory-proof-v1",
        "status": "pass",
        "source": {key: source[key] for key in
                   ("url", "content_length", "bz_file_id", "content_sha1", "accept_ranges", "prefix")},
        "quarter": {"start": start, "end": end, "csv_member_count": len(csv_names)},
        "csv_members": csv_names,
        "metadata_exception_members": sorted(metadata),
        "all_member_names": sorted(actual),
    })
    return facts


def verify_directory_proof(proof: dict, source: dict, *, start: str, end: str) -> None:
    """Recheck a saved proof without making a network request."""
    if not isinstance(proof, dict) or proof.get("schema") != "q1-directory-proof-v1" or proof.get("status") != "pass":
        raise ReleaseError("Q1 directory proof is not passing")
    if proof.get("source") != {key: source[key] for key in
                               ("url", "content_length", "bz_file_id", "content_sha1", "accept_ranges", "prefix")}:
        raise ReleaseError("Q1 directory proof is bound to a different object")
    if proof.get("quarter") != {"start": start, "end": end,
                                 "csv_member_count": len(_dates(start, end))}:
        raise ReleaseError("Q1 directory proof calendar differs")
    csv_members, metadata = expected_members(source["prefix"], start, end)
    actual = set(proof.get("all_member_names", ()))
    if (set(proof.get("csv_members", ())) != csv_members
            or set(proof.get("metadata_exception_members", ())) != metadata
            or actual != csv_members | metadata
            or proof.get("central_directory", {}).get("entry_count") != len(actual)):
        raise ReleaseError("Q1 directory proof member map differs")


def write_directory_proof(path: Path | str, source: dict, *, start: str, end: str) -> dict:
    path = bound_path(path, "Q1 directory proof")
    proof = build_directory_proof(source, start=start, end=end)
    write_json_exclusive(path, proof)
    return proof


__all__ = ["MAX_RANGE_BYTES", "build_directory_proof", "expected_members",
           "fetch_directory", "parse_central_directory", "verify_directory_proof",
           "write_directory_proof"]
