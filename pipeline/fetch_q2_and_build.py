"""Fetch Q2 2023 ZIP members by HTTP range and append only the selected model.

The complete Q2 ZIP is never retained or expanded. At most a small number of
compressed member ranges exist in ``.tmp/q2_members`` while the selected rows
are appended to the project panel; each temporary member is removed after its
CRC, size, schema, and selected-row checks pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import resource
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import time

from .member_io import iter_member_rows
from .panel import PanelWriter


ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q2_2023.zip"
DATE_RE = re.compile(r"data_Q2_2023/(?P<date>\d{4}-\d{2}-\d{2})\.csv$")
HARD_MAX_TEMP_BYTES = 256 * 1024 * 1024
HARD_MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024


class MemberVerificationError(RuntimeError):
    """A downloaded member failed ZIP/CSV verification; retain its evidence."""

    def __init__(self, entry: dict, cause: BaseException):
        self.entry = dict(entry)
        self.cause = cause
        super().__init__(str(cause))


class LedgerReconciliationError(RuntimeError):
    """The database and persisted member ledger disagree."""

    def __init__(self, item: dict, reason: str):
        self.item = dict(item)
        self.reason = reason
        super().__init__(reason)


def curl_bytes(*args: str, timeout: int = 60) -> bytes:
    command = ["curl", "-fLsS", "--retry", "2", "--max-time", str(timeout), *args]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def _last_response_headers(path: pathlib.Path) -> tuple[int, dict[str, str]]:
    raw = path.read_text(encoding="iso-8859-1")
    blocks = [block for block in re.split(r"\r?\n\r?\n", raw) if block.strip()]
    if not blocks:
        raise RuntimeError("curl returned no HTTP response headers")
    block = blocks[-1]
    lines = block.splitlines()
    match = re.match(r"HTTP/\S+\s+(\d{3})", lines[0])
    if not match:
        raise RuntimeError(f"unrecognized HTTP status line: {lines[0]!r}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.lower().strip()] = value.strip()
    return int(match.group(1)), headers


def _source_identity(headers: dict[str, str]) -> dict[str, str]:
    """Return stable server identifiers, excluding per-request range headers."""
    keys = ("x-bz-file-id", "etag", "last-modified", "content-md5")
    return {key: headers[key] for key in keys if headers.get(key)}


def _has_strong_source_identity(identity: dict[str, str]) -> bool:
    """A date or URL is not enough to bind ranges to one immutable archive."""
    return bool(identity.get("x-bz-file-id") or identity.get("etag") or identity.get("content-md5"))


def _read_existing_q2_state(path: pathlib.Path) -> tuple[dict[str, str] | None, set[str]]:
    """Read source identity and existing Q2 dates without opening the writer.

    This read-only preflight prevents a changed archive from triggering schema
    migrations or metadata commits before it is rejected.
    """
    if not path.exists():
        return None, set()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        identity = None
        if "metadata" in tables:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'q2_source_identity'"
            ).fetchone()
            if row is not None and row["value"]:
                try:
                    decoded = json.loads(row["value"])
                except json.JSONDecodeError as exc:
                    raise RuntimeError("existing q2_source_identity is not valid JSON") from exc
                if not isinstance(decoded, dict):
                    raise RuntimeError("existing q2_source_identity is not an object")
                identity = {str(key): str(value) for key, value in decoded.items()}
        q2_dates: set[str] = set()
        if "member_counts" in tables:
            q2_dates = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT date FROM member_counts
                    WHERE date BETWEEN '2023-04-01' AND '2023-06-30'
                    """
                )
            }
        return identity, q2_dates
    finally:
        connection.close()


def _database_member_match(connection, item: dict) -> tuple[bool, str | None]:
    """Compare one ledger item with the durable member_counts fact."""
    date = item.get("date")
    row = connection.execute(
        """
        SELECT source_member, source_sha256, source_rows, selected_rows,
               failure_rows, schema_columns, schema_sha256
        FROM member_counts WHERE date = ?
        """,
        (date,),
    ).fetchone()
    if row is None:
        return False, f"database member_counts row missing for {date}"
    expected_member = item.get("name") or item.get("source_member")
    expected_sha = item.get("sha256") or item.get("source_sha256")
    if not expected_member or not expected_sha:
        return False, f"ledger member or sha256 missing for {date}"
    if row["source_member"] != expected_member:
        return False, f"source member mismatch for {date}"
    if row["source_sha256"] != expected_sha:
        return False, f"source sha256 mismatch for {date}"
    for field in ("source_rows", "selected_rows", "failure_rows", "schema_columns"):
        if item.get(field) is not None and int(item[field]) != int(row[field]):
            return False, f"{field} mismatch for {date}"
    expected_schema = item.get("schema_sha256")
    if expected_schema and row["schema_sha256"] and expected_schema != row["schema_sha256"]:
        return False, f"schema sha256 mismatch for {date}"
    return True, None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if os.uname().sysname == "Darwin" else value * 1024


def _temp_bytes(directory: pathlib.Path) -> int:
    return sum(path.stat().st_size for path in directory.glob("*") if path.is_file())


def _download_range_to_file(
    start: int,
    end: int,
    total_bytes: int,
    path: pathlib.Path,
    *,
    timeout: int,
    expected_identity: dict[str, str] | None = None,
) -> dict[str, str]:
    if not 0 <= start <= end < total_bytes:
        raise ValueError(f"invalid range {start}-{end}/{total_bytes}")
    partial = path.with_name(path.name + ".partial")
    header_path = path.with_name(path.name + ".headers")
    partial.unlink(missing_ok=True)
    header_path.unlink(missing_ok=True)
    expected_span = end - start + 1
    if shutil.disk_usage(path.parent).free < expected_span:
        raise RuntimeError(f"insufficient free space for range {start}-{end}")
    command = [
        "curl", "-fLsS", "--retry", "2", "--max-time", str(timeout),
        "--max-filesize", str(expected_span),
        "-D", str(header_path), "-r", f"{start}-{end}", "-o", str(partial), URL,
    ]
    success = False
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        status, headers = _last_response_headers(header_path)
        content_range = headers.get("content-range", "")
        expected_range = f"bytes {start}-{end}/{total_bytes}"
        if status != 206 or content_range != expected_range:
            raise RuntimeError(
                f"range response mismatch for {start}-{end}: status={status}, "
                f"content-range={content_range!r}, expected={expected_range!r}"
            )
        if int(headers.get("content-length", "-1")) != expected_span:
            raise RuntimeError(
                f"range length mismatch for {start}-{end}: "
                f"{headers.get('content-length')!r} != {expected_span}"
            )
        if partial.stat().st_size != expected_span:
            raise RuntimeError(
                f"downloaded range length mismatch for {start}-{end}: "
                f"{partial.stat().st_size} != {expected_span}"
            )
        identity = _source_identity(headers)
        if expected_identity:
            missing = sorted(set(expected_identity) - set(identity))
            changed = {
                key: (expected_identity[key], identity.get(key))
                for key in expected_identity
                if key in identity and identity[key] != expected_identity[key]
            }
            if missing or changed:
                raise RuntimeError(
                    f"source identity mismatch for {start}-{end}: missing={missing}, changed={changed}"
                )
        os.replace(partial, path)
        success = True
        return {
            "http_status": str(status),
            "content_range": content_range,
            "content_length": headers["content-length"],
            "source_identity": json.dumps(identity, sort_keys=True),
        }
    finally:
        if success:
            partial.unlink(missing_ok=True)
            header_path.unlink(missing_ok=True)


def head_archive() -> tuple[int, dict[str, str]]:
    result = subprocess.run(
        ["curl", "-fLsSI", "--max-time", "30", URL],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    blocks = [block for block in re.split(r"\r?\n\r?\n", result.stdout) if block.strip()]
    if not blocks:
        raise RuntimeError("Q2 HEAD response did not include an HTTP header block")
    headers: dict[str, str] = {}
    for line in blocks[-1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.lower().strip()] = value.strip()
    if "content-length" not in headers:
        raise RuntimeError("Q2 HEAD response did not include Content-Length")
    return int(headers["content-length"]), headers


def _verified_range_bytes(
    start: int,
    end: int,
    total_bytes: int,
    *,
    expected_identity: dict[str, str] | None = None,
    timeout: int = 60,
) -> tuple[bytes, dict[str, str]]:
    if not 0 <= start <= end < total_bytes:
        raise ValueError(f"invalid range {start}-{end}/{total_bytes}")
    tmp_root = ROOT / ".tmp" / "q2_ranges"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=tmp_root, prefix="range-", suffix=".bin", delete=False) as data_handle:
        data_path = pathlib.Path(data_handle.name)
    header_path = data_path.with_suffix(".headers")
    expected_span = end - start + 1
    if shutil.disk_usage(tmp_root).free < expected_span:
        data_path.unlink(missing_ok=True)
        raise RuntimeError(f"insufficient free space for range {start}-{end}")
    command = [
        "curl", "-fLsS", "--retry", "2", "--max-time", str(timeout),
        "--max-filesize", str(expected_span), "-D", str(header_path),
        "-r", f"{start}-{end}", "-o", str(data_path), URL,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        status, headers = _last_response_headers(header_path)
        expected_range = f"bytes {start}-{end}/{total_bytes}"
        if status != 206 or headers.get("content-range") != expected_range:
            raise RuntimeError(
                f"range response mismatch for {start}-{end}: "
                f"status={status}, content-range={headers.get('content-range')!r}"
            )
        if int(headers.get("content-length", "-1")) != expected_span or data_path.stat().st_size != expected_span:
            raise RuntimeError(f"range length mismatch for {start}-{end}")
        identity = _source_identity(headers)
        if expected_identity:
            if any(identity.get(key) != value for key, value in expected_identity.items()):
                raise RuntimeError(f"source identity changed for range {start}-{end}")
        return data_path.read_bytes(), {
            "http_status": str(status),
            "content_range": headers["content-range"],
            "content_length": headers["content-length"],
            "source_identity": json.dumps(identity, sort_keys=True),
        }
    finally:
        data_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)


def central_directory(
    total_bytes: int,
    *,
    expected_identity: dict[str, str] | None = None,
) -> list[dict]:
    tail_size = min(total_bytes, 262_144)
    tail_start = total_bytes - tail_size
    tail, _ = _verified_range_bytes(
        tail_start,
        total_bytes - 1,
        total_bytes,
        expected_identity=expected_identity,
        timeout=60,
    )
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("Q2 ZIP end-of-central-directory record was not found")
    _, _, _, entries_disk, entries_total, central_size, central_offset, comment_len = struct.unpack_from(
        "<4s4H2LH", tail, eocd_at
    )
    if entries_disk != entries_total:
        raise RuntimeError("multi-disk ZIP is unsupported")
    if central_offset < tail_start:
        tail_size = min(total_bytes, 4 * 1024 * 1024)
        tail_start = total_bytes - tail_size
        tail, _ = _verified_range_bytes(
            tail_start,
            total_bytes - 1,
            total_bytes,
            expected_identity=expected_identity,
            timeout=90,
        )
        eocd_at = tail.rfind(b"PK\x05\x06")
        if eocd_at < 0:
            raise RuntimeError("Q2 ZIP end-of-central-directory record was not found in enlarged tail")
        _, _, _, entries_disk, entries_total, central_size, central_offset, comment_len = struct.unpack_from(
            "<4s4H2LH", tail, eocd_at
        )
    central_end = central_offset + central_size
    if not (tail_start <= central_offset and central_end <= total_bytes):
        raise RuntimeError("Q2 central directory is outside the fetched tail")
    offset_in_tail = central_offset - tail_start
    blob = tail[offset_in_tail : offset_in_tail + central_size]
    entries = []
    cursor = 0
    for _ in range(entries_total):
        if blob[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid Q2 central-directory signature at {cursor}")
        fields = struct.unpack_from("<4s6H3L5H2L", blob, cursor)
        (
            _, _, _, flags, method, _, _, crc, compressed, uncompressed,
            filename_length, extra_length, comment_length, _, _, _, local_offset,
        ) = fields
        begin = cursor + 46
        name = blob[begin : begin + filename_length].decode("utf-8")
        cursor = begin + filename_length + extra_length + comment_length
        match = DATE_RE.fullmatch(name)
        if not match:
            continue
        entries.append(
            {
                "name": name,
                "date": match.group("date"),
                "compressed": compressed,
                "size": uncompressed,
                "crc": crc,
                "offset": local_offset,
                "flags": flags,
                "method": method,
                "central_filename_length": filename_length,
                "central_extra_length": extra_length,
            }
        )
    entries.sort(key=lambda item: item["date"])
    expected_dates = [
        (dt.date(2023, 4, 1) + dt.timedelta(days=index)).isoformat()
        for index in range(91)
    ]
    if [item["date"] for item in entries] != expected_dates:
        raise RuntimeError("Q2 daily member dates are not exactly 2023-04-01 through 2023-06-30")
    return entries


def download_member(
    entry: dict,
    temp_dir: pathlib.Path,
    total_bytes: int,
    *,
    expected_identity: dict[str, str] | None = None,
) -> dict:
    path = temp_dir / f"{entry['date']}.member"
    start = int(entry["offset"])
    # The local extra field can differ from the central extra field. A 64KiB
    # pad is bounded overhead and lets member_io read the local header exactly.
    end = min(
        total_bytes - 1,
        start
        + 30
        + int(entry["central_filename_length"])
        + int(entry["central_extra_length"])
        + 65_536
        + int(entry["compressed"])
        - 1,
    )
    expected_span = end - start + 1
    response = _download_range_to_file(
        start,
        end,
        total_bytes,
        path,
        timeout=180,
        expected_identity=expected_identity,
    )
    payload = path.read_bytes()
    updated = dict(entry)
    updated.update(
        {
            "local": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "downloaded_bytes": len(payload),
            "range_start": start,
            "range_end": end,
            **response,
        }
    )
    if len(payload) != expected_span:
        raise RuntimeError(f"range response unexpectedly sized for {entry['name']}: {len(payload)}")
    return updated


def _download_and_verify_member(
    entry: dict,
    temp_dir: pathlib.Path,
    total_bytes: int,
    *,
    expected_identity: dict[str, str] | None = None,
) -> dict:
    """Download, then perform the full ZIP/CSV integrity pass before ``verified``."""
    downloaded = download_member(
        entry, temp_dir, total_bytes, expected_identity=expected_identity
    )
    downloaded["state"] = "downloaded"
    # iter_member_rows validates the local header, decompressed size, CRC and
    # required header before yielding rows.  The generator is intentionally not
    # consumed here; PanelWriter performs the streaming row pass after this
    # durable verification boundary.
    try:
        _verify_member_payload(downloaded)
    except BaseException as exc:
        raise MemberVerificationError(downloaded, exc) from exc
    downloaded["state"] = "verified"
    return downloaded


def _verify_member_payload(entry: dict) -> None:
    """Run the expensive member decoder before publishing ``verified`` state."""
    _date, _sha, _columns, rows, _diagnostics = iter_member_rows(entry, ROOT)
    del rows


def _member_range_span(entry: dict, total_bytes: int) -> int:
    start = int(entry["offset"])
    end = min(
        total_bytes - 1,
        start
        + 30
        + int(entry["central_filename_length"])
        + int(entry["central_extra_length"])
        + 65_536
        + int(entry["compressed"])
        - 1,
    )
    return end - start + 1


def _coverage_low_for_date(connection, date_text: str) -> bool | None:
    """Return the registered trailing-seven-day low-coverage flag."""
    current = connection.execute(
        "SELECT source_rows, selected_rows FROM member_counts WHERE date = ?",
        (date_text,),
    ).fetchone()
    if current is None:
        return None
    date = dt.date.fromisoformat(date_text)
    previous = [
        connection.execute(
            "SELECT source_rows, selected_rows FROM member_counts WHERE date = ?",
            ((date - dt.timedelta(days=offset)).isoformat(),),
        ).fetchone()
        for offset in range(1, 8)
    ]
    if any(row is None for row in previous):
        return None

    def median(values: list[int]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    source_base = median([row[0] for row in previous])
    selected_base = median([row[1] for row in previous])
    if source_base <= 0 or selected_base <= 0:
        return True
    return current[0] / source_base < 0.80 or current[1] / selected_base < 0.80


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--summary-output", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="ST4000DM000")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--min-selected-rows",
        type=int,
        default=None,
        help="optional hard stop; leave unset to retain low-coverage days for audit",
    )
    parser.add_argument(
        "--max-temp-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="maximum in-flight and retained temporary member bytes",
    )
    parser.add_argument(
        "--max-rss-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="stop before process resident memory exceeds this bound",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="explicitly discard retained failed-member evidence and retry those dates",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_temp_bytes <= 0 or args.max_rss_bytes <= 0:
        raise ValueError("resource limits must be positive")
    if args.max_temp_bytes > HARD_MAX_TEMP_BYTES or args.max_rss_bytes > HARD_MAX_RSS_BYTES:
        raise ValueError("resource limits cannot exceed the registered hard bounds")
    total_bytes, headers = head_archive()
    if total_bytes != 859_566_204:
        raise RuntimeError(f"Q2 archive size changed from pre-registered HEAD value: {total_bytes}")
    archive_identity = _source_identity(headers)
    if not _has_strong_source_identity(archive_identity):
        raise RuntimeError("Q2 HEAD response has no stable source identity")
    entries = central_directory(total_bytes, expected_identity=archive_identity)
    temp_dir = ROOT / ".tmp" / "q2_members"
    temp_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)

    existing_identity, existing_q2_dates = _read_existing_q2_state(args.database)
    if existing_identity is not None and existing_identity != archive_identity:
        raise RuntimeError("existing database has a different Q2 source identity")
    q2_integrity_status = (
        "legacy_not_verified"
        if existing_q2_dates and existing_identity is None
        else ("identity_verified_only" if existing_q2_dates else "verified")
    )

    prior: dict = {}
    if args.manifest_output.exists():
        prior = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        if prior.get("source_url") not in (None, URL) or prior.get("archive_bytes") not in (None, total_bytes):
            raise RuntimeError("existing progress manifest belongs to a different Q2 source")
        prior_identity = prior.get("source_identity")
        if prior_identity and prior_identity != archive_identity:
            raise RuntimeError("existing progress manifest has a different source identity")
    writer = PanelWriter(args.database, reset=False)
    def compact_member(item: dict) -> dict:
        keep = {
            "name", "date", "compressed", "size", "crc", "offset", "flags", "method",
            "central_filename_length", "central_extra_length", "sha256", "downloaded_bytes",
            "local",
            "range_start", "range_end", "http_status", "content_range", "content_length",
            "source_identity", "source_sha256", "state",
            "source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256",
            "resumed_from_panel", "failure_reason",
        }
        return {key: value for key, value in item.items() if key in keep}

    prior_members = {
        item["date"]: compact_member(item)
        for item in prior.get("members", [])
        if item.get("date") and item["date"] in writer.seen_dates
    }
    completed: dict[str, dict] = dict(prior_members)
    submitted_dates: set[str] = set(prior.get("submitted_dates", []))
    failed: dict[str, str] = dict(prior.get("failed", {}))
    failed_members: dict[str, dict] = {
        item["date"]: dict(item)
        for item in prior.get("failed_members", [])
        if item.get("date")
    }
    states: dict[str, str] = dict(prior.get("states", {}))
    verified: dict[str, dict] = {
        item["date"]: dict(item)
        for item in prior.get("verified_members", [])
        if item.get("date") and item.get("date") not in completed
    }
    ledger_quality = prior.get(
        "ledger_quality",
        "legacy_unverified_submission_history" if prior else "tracked",
    )
    retry_history: list[dict] = list(prior.get("retry_history", []))
    pending_entries = [entry for entry in entries if entry["date"] not in writer.seen_dates]
    entry_by_date = {entry["date"]: entry for entry in entries}
    # A prior panel may have been produced before its progress manifest was
    # flushed. Reconstruct completed records from the panel's source counts and
    # keep the central-directory fields from this run.
    existing_counts = {
        row["date"]: dict(row)
        for row in writer.connection.execute("SELECT * FROM member_counts")
    }
    for date in sorted(writer.seen_dates.intersection(entry_by_date)):
        if date in completed:
            continue
        entry = dict(entry_by_date[date])
        count = existing_counts[date]
        entry.update(
            {
                "local": None,
                "sha256": count["source_sha256"],
                "source_rows": count["source_rows"],
                "selected_rows": count["selected_rows"],
                "failure_rows": count["failure_rows"],
                "schema_columns": count["schema_columns"],
                "resumed_from_panel": True,
            }
        )
        completed[date] = entry
        states.setdefault(date, "legacy_ingested")

    reconciliation_failures: dict[str, str] = {}

    def mark_reconciliation_failure(date: str, item: dict, reason: str) -> None:
        failure_item = dict(item, state="failed", failure_reason=reason)
        failed[date] = reason
        failed_members[date] = compact_member(failure_item)
        states[date] = "failed"
        reconciliation_failures[date] = reason

    # A database row and its ledger record must agree before any retained
    # member is removed. Legacy rows reconstructed from the panel deliberately
    # keep their legacy_ingested state and are not upgraded by this pass.
    pending_cleanup: list[tuple[str, dict]] = []
    for date, item in completed.items():
        if item.get("state") == "legacy_ingested":
            continue
        if item.get("sha256") or item.get("source_sha256"):
            matched, reason = _database_member_match(writer.connection, item)
            if not matched:
                mark_reconciliation_failure(date, item, reason or f"ledger/database mismatch for {date}")
                continue
        if item.get("state") == "ingested_pending_cleanup":
            local = item.get("local")
            if not local:
                mark_reconciliation_failure(date, item, f"pending cleanup path missing for {date}")
            elif not (ROOT / local).is_file():
                mark_reconciliation_failure(date, item, f"pending cleanup file missing for {date}")
            else:
                pending_cleanup.append((date, item))

    # Reuse a verified range only when its retained local bytes still match the
    # recorded hash. A mismatch is failed evidence and blocks automatic retry;
    # the manifest never silently turns it into a fresh download.
    for date, item in list(verified.items()):
        local = item.get("local")
        path = ROOT / local if local else None
        if date not in entry_by_date or path is None or not path.is_file():
            verified.pop(date, None)
            mark_reconciliation_failure(
                date, item, f"verified member file missing or no longer in source for {date}"
            )
            continue
        expected_hash = item.get("sha256")
        if not expected_hash:
            verified.pop(date, None)
            mark_reconciliation_failure(date, item, f"verified member sha256 missing for {date}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            verified.pop(date, None)
            mark_reconciliation_failure(date, item, f"retained verified range hash mismatch for {date}")
            continue

    def write_progress(status: str, *, error: str | None = None) -> None:
        payload = {
            "source_url": URL,
            "retrieved_at_utc": prior.get("retrieved_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            "archive_bytes": total_bytes,
            "source_identity": archive_identity,
            "source_identity_status": q2_integrity_status,
            "source_verification_status": source_verification_status,
            "server_headers": headers,
            "status": status,
            "submitted_dates": sorted(submitted_dates),
            "completed_dates": sorted(completed),
            "pending_dates": sorted(set(entry_by_date) - set(completed)),
            "failed": failed,
            "failed_members": [compact_member(failed_members[date]) for date in sorted(failed_members)],
            "states": states,
            "verified_members": [compact_member(verified[date]) for date in sorted(verified)],
            "members": [compact_member(completed[date]) for date in sorted(completed)],
            "ledger_quality": ledger_quality,
            "retry_history": retry_history,
            "stop_reason": stop_reason,
            "retention": "ingested ranges removed; verified or interrupted ranges retained until reconciled",
        }
        if error:
            payload["error"] = error
        temp_manifest = args.manifest_output.with_name(args.manifest_output.name + ".partial")
        temp_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_manifest, args.manifest_output)

    stop_reason: str | None = None
    stop_submitting = bool(reconciliation_failures)
    if reconciliation_failures:
        first_failure = sorted(reconciliation_failures)[0]
        stop_reason = f"ledger_database_mismatch:{first_failure}"
    source_metadata_updates: dict[str, str] = {}
    source_verification_status = (
        "not_verified" if q2_integrity_status != "verified" else "pending"
    )
    if not existing_q2_dates and not reconciliation_failures:
        source_metadata_updates = {
            "q2_source_url": URL,
            "q2_archive_bytes": str(total_bytes),
            "q2_selected_model": args.model,
            "q2_source_identity": json.dumps(archive_identity, sort_keys=True),
            "q2_source_identity_status": "verified",
        }
    if source_metadata_updates:
        writer.set_metadata(source_metadata_updates)
    write_progress("running")
    if not reconciliation_failures:
        for date, item in pending_cleanup:
            # The pending state is durable before removing the file. A crash
            # between these writes leaves a recoverable reconciliation record.
            write_progress("running")
            (ROOT / item["local"]).unlink(missing_ok=True)
            item["local"] = None
            item["state"] = "ingested"
            states[date] = "ingested"
            write_progress("running")
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures: dict[concurrent.futures.Future, dict] = {}
    ready: dict[str, dict] = {}
    ready.update(verified)
    reserved: dict[str, int] = {}
    next_index = 0
    next_process = 0
    processed_count = len(completed)
    status = "complete"
    error_message: str | None = None
    low_streak = 0
    last_processed_date: dt.date | None = None
    # Rebuild the trailing coverage state on resume. A process restart must not
    # turn the third consecutive low day into a fresh first day.
    for date_text in sorted(completed):
        current_date = dt.date.fromisoformat(date_text)
        low = _coverage_low_for_date(writer.connection, date_text)
        if low is True and last_processed_date == current_date - dt.timedelta(days=1):
            low_streak += 1
        elif low is True:
            low_streak = 1
        else:
            low_streak = 0
        last_processed_date = current_date
    if low_streak >= 3:
        stop_submitting = True
        stop_reason = f"three_consecutive_low_coverage_days_through:{last_processed_date.isoformat()}"
    if failed_members and not args.retry_failed:
        first_failed = sorted(failed_members)[0]
        stop_submitting = True
        stop_reason = f"failed_evidence_requires_review:{first_failed}"
    elif args.retry_failed:
        for date_text, item in list(failed_members.items()):
            local = item.get("local")
            if local:
                failed_path = ROOT / local
                if failed_path.is_file():
                    # Keep the failed bytes under a unique forensic path; the
                    # explicit retry flag is the only path that may replace a
                    # normal member filename.
                    forensic_path = failed_path.with_name(failed_path.name + ".failed")
                    suffix = 2
                    while forensic_path.exists():
                        forensic_path = failed_path.with_name(
                            failed_path.name + f".failed.{suffix}"
                        )
                        suffix += 1
                    failed_path.rename(forensic_path)
                    retry_history.append(
                        {
                            "date": date_text,
                            "original_local": local,
                            "forensic_local": str(forensic_path.relative_to(ROOT)),
                            "sha256": item.get("sha256") or item.get("source_sha256"),
                            "reason": item.get("failure_reason") or failed.get(date_text),
                            "retried_at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                    )
                else:
                    retry_history.append(
                        {
                            "date": date_text,
                            "original_local": local,
                            "forensic_local": None,
                            "sha256": item.get("sha256") or item.get("source_sha256"),
                            "reason": item.get("failure_reason") or failed.get(date_text),
                            "retried_at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                        }
                    )
            else:
                retry_history.append(
                    {
                        "date": date_text,
                        "original_local": None,
                        "forensic_local": None,
                        "sha256": item.get("sha256") or item.get("source_sha256"),
                        "reason": item.get("failure_reason") or failed.get(date_text),
                        "retried_at_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                )
            if date_text in writer.seen_dates:
                stop_submitting = True
                stop_reason = f"retry_requires_reconcile_existing_date:{date_text}"
            else:
                reconciliation_failures.pop(date_text, None)
            failed_members.pop(date_text, None)
            failed.pop(date_text, None)
        if not reconciliation_failures:
            stop_submitting = False
            stop_reason = None

    def submit_available() -> None:
        nonlocal next_index, stop_submitting, stop_reason
        while (
            not stop_submitting
            and next_index < len(pending_entries)
            and len(futures) < args.workers
        ):
            if pending_entries[next_index]["date"] in ready:
                next_index += 1
                continue
            date_text = pending_entries[next_index]["date"]
            if date_text in failed_members:
                stop_submitting = True
                stop_reason = f"failed_evidence_requires_review:{date_text}"
                break
            if _rss_bytes() > args.max_rss_bytes:
                stop_submitting = True
                stop_reason = f"rss_limit_exceeded:{_rss_bytes()}>{args.max_rss_bytes}"
                break
            entry = pending_entries[next_index]
            span = _member_range_span(entry, total_bytes)
            projected = _temp_bytes(temp_dir) + sum(reserved.values()) + span
            if projected > args.max_temp_bytes:
                stop_submitting = True
                stop_reason = f"temp_limit_reached:{projected}>{args.max_temp_bytes}"
                break
            future = pool.submit(
                _download_and_verify_member,
                entry,
                temp_dir,
                total_bytes,
                expected_identity=archive_identity,
            )
            futures[future] = entry
            reserved[entry["date"]] = span
            submitted_dates.add(entry["date"])
            states[entry["date"]] = "submitted"
            next_index += 1

    try:
        submit_available()
        write_progress("running")
        while futures or ready:
            if futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    entry = futures.pop(future)
                    reserved.pop(entry["date"], None)
                    try:
                        downloaded_entry = future.result()
                    except MemberVerificationError as exc:
                        reason = f"{type(exc.cause).__name__}: {exc.cause}"
                        failed[entry["date"]] = reason
                        failed_members[entry["date"]] = compact_member(
                            dict(exc.entry, state="failed", failure_reason=reason)
                        )
                        states[entry["date"]] = "failed"
                        write_progress("running")
                        raise exc.cause
                    except BaseException as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        failed[entry["date"]] = reason
                        failed_members[entry["date"]] = compact_member(
                            dict(entry, state="failed", failure_reason=reason)
                        )
                        states[entry["date"]] = "failed"
                        raise
                    verified[entry["date"]] = downloaded_entry
                    ready[entry["date"]] = downloaded_entry
                    states[entry["date"]] = "verified"
                write_progress("running")

            # Download workers may finish out of order, but ingestion and the
            # rolling coverage stop gate follow the official calendar order.
            while next_process < len(pending_entries):
                entry = pending_entries[next_process]
                date_text = entry["date"]
                if date_text not in ready:
                    break
                downloaded_entry = ready[date_text]
                try:
                    result = writer.append_member(
                        downloaded_entry,
                        ROOT,
                        args.model,
                        min_selected_rows=args.min_selected_rows,
                    )
                    matched, reason = _database_member_match(writer.connection, {
                        **downloaded_entry,
                        **result,
                        "name": downloaded_entry.get("name"),
                        "sha256": downloaded_entry.get("sha256"),
                    })
                    if not matched:
                        raise LedgerReconciliationError(
                            downloaded_entry,
                            reason or f"ledger/database mismatch for {date_text}",
                        )
                except BaseException as exc:
                    reason = (
                        exc.reason
                        if isinstance(exc, LedgerReconciliationError)
                        else f"{type(exc).__name__}: {exc}"
                    )
                    failed[date_text] = reason
                    failed_members[date_text] = compact_member(
                        dict(downloaded_entry, state="failed", failure_reason=reason)
                    )
                    states[date_text] = "failed"
                    ready.pop(date_text, None)
                    verified.pop(date_text, None)
                    write_progress("running")
                    raise
                ready.pop(date_text, None)
                verified.pop(date_text, None)
                path = ROOT / downloaded_entry["local"]
                completed[date_text] = dict(
                    downloaded_entry,
                    state="ingested_pending_cleanup",
                    source_sha256=result.get(
                        "source_sha256", downloaded_entry.get("sha256")
                    ),
                    source_rows=result["source_rows"],
                    selected_rows=result["selected_rows"],
                    failure_rows=result["failure_rows"],
                    schema_columns=result["schema_columns"],
                    schema_sha256=result["schema_sha256"],
                )
                states[date_text] = "ingested_pending_cleanup"
                failed.pop(date_text, None)
                # Persist the DB-backed member and its retained bytes before
                # cleanup. If interrupted here, the next run sees this state.
                write_progress("running")
                path.unlink(missing_ok=True)
                completed[date_text]["local"] = None
                completed[date_text]["state"] = "ingested"
                states[date_text] = "ingested"
                processed_count += 1
                current_date = dt.date.fromisoformat(date_text)
                low = _coverage_low_for_date(writer.connection, date_text)
                if low is True:
                    low_streak = low_streak + 1 if last_processed_date == current_date - dt.timedelta(days=1) else 1
                    if low_streak >= 3:
                        stop_submitting = True
                        stop_reason = f"three_consecutive_low_coverage_days_through:{date_text}"
                elif low is False:
                    low_streak = 0
                last_processed_date = current_date
                next_process += 1
                print(json.dumps(result, ensure_ascii=False), flush=True)
                submit_available()
                write_progress("running")
                if processed_count % 10 == 0 or processed_count == len(entries):
                    print(
                        f"progress={processed_count}/{len(entries)} panel_rows={writer.summary()['panel_rows']}",
                        flush=True,
                    )
            if not futures and not ready and next_process < len(pending_entries):
                # The only normal reason is a resource/coverage stop. Leave the
                # remaining dates pending for a deliberate later resume.
                break
            if (
                not futures
                and ready
                and next_process < len(pending_entries)
                and pending_entries[next_process]["date"] not in ready
            ):
                stop_submitting = True
                stop_reason = f"missing_prior_ready:{pending_entries[next_process]['date']}"
                break
        if stop_reason:
            status = "stopped"
        if (
            status == "complete"
            and len(completed) == len(entries)
            and q2_integrity_status == "verified"
            and all(item.get("state") == "ingested" for item in completed.values())
        ):
            writer.set_metadata({"q2_source_verification_status": "verified"})
            source_verification_status = "verified"
        summary = writer.summary()
    except BaseException as exc:
        status = "aborted"
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        # Even on an abort, resolve every submitted future so the manifest
        # reflects files that really reached verified state instead of leaving
        # them as an unqualified submission claim.
        for future, entry in list(futures.items()):
            date_text = entry["date"]
            if future.cancelled():
                states[date_text] = "cancelled"
                continue
            if future.done():
                try:
                    downloaded_entry = future.result()
                except MemberVerificationError as exc:
                    reason = f"{type(exc.cause).__name__}: {exc.cause}"
                    failed[date_text] = reason
                    failed_members[date_text] = compact_member(
                        dict(exc.entry, state="failed", failure_reason=reason)
                    )
                    states[date_text] = "failed"
                except BaseException as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failed.setdefault(date_text, reason)
                    failed_members.setdefault(
                        date_text,
                        compact_member(dict(entry, state="failed", failure_reason=reason)),
                    )
                    states[date_text] = "failed"
                else:
                    verified.setdefault(date_text, downloaded_entry)
                    states[date_text] = "verified"
        write_progress(status, error=error_message)
        writer.close()
        for path in temp_dir.glob("*.member"):
            date_text = path.stem
            if states.get(date_text) == "ingested":
                path.unlink(missing_ok=True)
    q2_summary = {
        "quarter": "2023Q2",
        "model": args.model,
        "archive_bytes": total_bytes,
        "members": len(completed),
        "status": status,
        "source_identity_status": q2_integrity_status,
        "source_verification_status": source_verification_status,
        "stop_reason": stop_reason,
        "completed_dates": sorted(completed),
        "pending_dates": sorted(set(entry_by_date) - set(completed)),
        "panel_summary": summary,
        "source_url": URL,
        "manifest": str(args.manifest_output),
        "retained_raw_q2": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(q2_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(q2_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
