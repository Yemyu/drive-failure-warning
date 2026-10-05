"""Acquire and validate the single Q3 2023 Drive Stats archive.

The command is deliberately separate from panel construction.  By default it
only writes a fixed plan.  ``--execute`` performs one full-archive download,
keeps a failed partial file, and validates every CSV member without using
``extractall``.  No Q4 data or model code is imported here.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/q3_validation_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence/q3/validation_v1"
DEFAULT_RAW = ROOT / "data/raw/q3_validation_v1"
EXPECTED_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q3_2023.zip"
EXPECTED_FILE_ID = "4_zefbb636e885eab02543a061a_f219e4a0a6ccbd88d_d20231113_m174247_c001_v0001178_t0019_u01699897367159"
EXPECTED_ARCHIVE_BYTES = 983_387_372
EXPECTED_DATES = [
    (dt.date(2023, 7, 1) + dt.timedelta(days=index)).isoformat()
    for index in range(92)
]
REQUIRED_FIELDS = (
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "smart_5_raw",
    "smart_9_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
)
SMART_FIELDS = tuple(f"smart_{value}_raw" for value in (5, 9, 187, 188, 197, 198))
DATE_MEMBER = re.compile(r"^(?:[^/]+/)?(?P<date>2023-\d{2}-\d{2})\.csv$")
HEX64 = set("0123456789abcdef")


class AcquisitionError(RuntimeError):
    """A source or archive gate stopped acquisition."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def project_path(value: str | Path, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise AcquisitionError(f"{field} escapes project root: {path}")
    return path


def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"cannot read config: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise AcquisitionError("Q3 config must be an object")
    if config.get("status") != "design_locked_not_executed":
        raise AcquisitionError("Q3 config is not in the locked pre-execution state")
    if config.get("q4_access") is not False or config.get("remote_setup") is not False:
        raise AcquisitionError("Q4 and remote access must remain disabled")
    if config.get("fit_allowed") is not False or config.get("scoring_allowed") is not False:
        raise AcquisitionError("Q3 acquisition config cannot authorize fitting or scoring")
    if config.get("source", {}).get("url") != EXPECTED_URL:
        raise AcquisitionError("Q3 source URL differs from the reviewed source")
    return config


def expected_dates(config: dict) -> list[str]:
    dates = list(config.get("source", {}).get("dates", []))
    if dates != EXPECTED_DATES:
        raise AcquisitionError("Q3 source date list is not the locked 92-day range")
    return dates


def make_plan(config: dict, *, evidence_root: Path = DEFAULT_EVIDENCE, raw_root: Path = DEFAULT_RAW) -> dict:
    expected_dates(config)
    evidence_root = project_path(evidence_root, "evidence root")
    raw_root = project_path(raw_root, "raw root")
    plan = {
        "plan_version": "q3-validation-acquisition-v1",
        "status": "planned",
        "created_at_utc": utc_now(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_hash": canonical_hash(config),
        "source": {
            "url": EXPECTED_URL,
            "file_id": EXPECTED_FILE_ID,
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "dates": EXPECTED_DATES,
            "archive_sha256": None,
        },
        "archive_path": str((raw_root / "attempt_001" / "data_Q3_2023.zip").relative_to(ROOT)),
        "attempt_root": str((raw_root / "attempt_001").relative_to(ROOT)),
        "evidence_root": str(evidence_root.relative_to(ROOT)),
        "limits": config["resources"],
        "q4_access": False,
        "remote_setup": False,
        "fit_allowed": False,
        "scoring_allowed": False,
        "training_approval": False,
    }
    plan["plan_hash"] = canonical_hash(plan)
    return plan


def _parse_headers(text: str) -> dict[str, str]:
    blocks = [block for block in re.split(r"\r?\n\r?\n", text) if block.strip()]
    if not blocks:
        raise AcquisitionError("HEAD returned no headers")
    block = blocks[-1]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("HTTP/"):
        raise AcquisitionError("HEAD response status is missing")
    try:
        status = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise AcquisitionError(f"invalid HEAD status: {lines[0]!r}") from exc
    headers: dict[str, str] = {"__status": str(status)}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def head_source(url: str = EXPECTED_URL) -> dict[str, str]:
    result = subprocess.run(
        ["curl", "--head", "--location", "--fail", "--max-time", "25", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AcquisitionError(f"HEAD failed with exit {result.returncode}: {result.stderr[-500:]}")
    headers = _parse_headers(result.stdout)
    if headers.get("__status") != "200":
        raise AcquisitionError(f"expected HTTP 200, got {headers.get('__status')}")
    if headers.get("content-length") != str(EXPECTED_ARCHIVE_BYTES):
        raise AcquisitionError(f"archive Content-Length differs: {headers.get('content-length')}")
    if headers.get("x-bz-file-id") != EXPECTED_FILE_ID:
        raise AcquisitionError("Q3 x-bz-file-id differs from the reviewed source")
    if "bytes" not in headers.get("accept-ranges", "").lower():
        raise AcquisitionError("Q3 source does not advertise byte ranges")
    return headers


def _integer(value: str, field: str, member: str, row_number: int) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise AcquisitionError(f"non-integer {field} in {member} row {row_number}") from exc


def _member_date(name: str) -> str | None:
    match = DATE_MEMBER.match(name)
    return match.group("date") if match else None


def validate_archive(
    archive: Path,
    *,
    expected_date_values: list[str] | None = None,
    expected_archive_bytes: int | None = EXPECTED_ARCHIVE_BYTES,
    max_member_bytes: int = 256 * 1024 * 1024,
    max_expanded_bytes: int = 12 * 1024 * 1024 * 1024,
) -> dict:
    """Validate ZIP members and produce a compact source manifest.

    ``expected_date_values`` is injectable only for unit fixtures.  Production
    calls use all 92 locked dates and the exact archive byte count.
    """
    archive = archive.resolve()
    if not archive.is_file():
        raise AcquisitionError(f"archive is missing: {archive}")
    if expected_archive_bytes is not None and archive.stat().st_size != expected_archive_bytes:
        raise AcquisitionError(f"archive bytes differ: {archive.stat().st_size} != {expected_archive_bytes}")
    dates = list(expected_date_values or EXPECTED_DATES)
    try:
        zf = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise AcquisitionError(f"cannot open Q3 ZIP: {exc}") from exc
    members: list[dict] = []
    seen_dates: set[str] = set()
    schema: list[str] | None = None
    expanded_total = 0
    total_source = total_selected = total_failure = 0
    all_serial_models: dict[str, str] = {}
    extras: list[str] = []
    try:
        for info in zf.infolist():
            date = _member_date(info.filename)
            if date is None:
                extras.append(info.filename)
                continue
            if date not in dates:
                raise AcquisitionError(f"unexpected Q3 member date: {info.filename}")
            if date in seen_dates:
                raise AcquisitionError(f"duplicate Q3 member date: {date}")
            if info.file_size > max_member_bytes:
                raise AcquisitionError(f"expanded member exceeds limit: {info.filename}")
            expanded_total += int(info.file_size)
            if expanded_total > max_expanded_bytes:
                raise AcquisitionError("expanded Q3 archive exceeds configured limit")
            try:
                raw = zf.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise AcquisitionError(f"CRC or decompression failure: {info.filename}: {exc}") from exc
            if len(raw) != info.file_size:
                raise AcquisitionError(f"expanded size mismatch: {info.filename}")
            digest = hashlib.sha256(raw).hexdigest()
            stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="")
            reader = csv.reader(stream)
            try:
                columns = next(reader)
            except StopIteration as exc:
                raise AcquisitionError(f"empty CSV member: {info.filename}") from exc
            if len(columns) != len(set(columns)):
                raise AcquisitionError(f"duplicate columns: {info.filename}")
            missing = [field for field in REQUIRED_FIELDS if field not in columns]
            if missing:
                raise AcquisitionError(f"missing required columns in {info.filename}: {missing}")
            if schema is None:
                schema = columns
            elif columns != schema:
                raise AcquisitionError(f"schema changed within Q3: {info.filename}")
            index = {column: columns.index(column) for column in columns}
            source_rows = selected_rows = failure_rows = 0
            serials: set[str] = set()
            selected_serials: set[str] = set()
            selected_digest = hashlib.sha256()
            for row_number, values in enumerate(reader, start=2):
                source_rows += 1
                if len(values) != len(columns):
                    raise AcquisitionError(f"row length mismatch in {info.filename} row {row_number}")
                if values[index["date"]] != date:
                    raise AcquisitionError(f"member/date mismatch in {info.filename} row {row_number}")
                serial = values[index["serial_number"]]
                model = values[index["model"]]
                if not serial or not model:
                    raise AcquisitionError(f"empty identity in {info.filename} row {row_number}")
                if serial in serials:
                    raise AcquisitionError(f"duplicate serial within day {date}: {serial}")
                serials.add(serial)
                previous_model = all_serial_models.get(serial)
                if previous_model is not None and previous_model != model:
                    raise AcquisitionError(f"cross-day model conflict for {serial}: {previous_model} vs {model}")
                all_serial_models[serial] = model
                failure = _integer(values[index["failure"]], "failure", info.filename, row_number)
                if failure not in (0, 1):
                    raise AcquisitionError(f"invalid failure in {info.filename} row {row_number}")
                capacity = _integer(values[index["capacity_bytes"]], "capacity_bytes", info.filename, row_number)
                if capacity is not None and capacity < -1:
                    raise AcquisitionError(f"unexpected negative capacity in {info.filename} row {row_number}")
                smart_values = []
                for field in SMART_FIELDS:
                    value = _integer(values[index[field]], field, info.filename, row_number)
                    if value is not None and value < 0:
                        raise AcquisitionError(f"negative SMART value in {info.filename} row {row_number}")
                    smart_values.append(value)
                if model == "ST4000DM000":
                    selected_rows += 1
                    selected_serials.add(serial)
                    failure_rows += int(failure or 0)
                    selected_digest.update(
                        json.dumps([serial, model, capacity, failure, *smart_values], ensure_ascii=False, separators=(",", ":")).encode()
                    )
                    selected_digest.update(b"\n")
            members.append(
                {
                    "date": date,
                    "name": info.filename,
                    "compressed_bytes": int(info.compress_size),
                    "expanded_bytes": int(info.file_size),
                    "crc": int(info.CRC),
                    "sha256": digest,
                    "source_rows": source_rows,
                    "selected_rows": selected_rows,
                    "failure_rows": failure_rows,
                    "selected_serials": len(selected_serials),
                    "selected_digest_sha256": selected_digest.hexdigest(),
                }
            )
            seen_dates.add(date)
            total_source += source_rows
            total_selected += selected_rows
            total_failure += failure_rows
    finally:
        zf.close()
    if sorted(seen_dates) != sorted(dates):
        missing = sorted(set(dates) - seen_dates)
        raise AcquisitionError(f"Q3 archive missing dates: {missing[:5]}")
    if schema is None:
        raise AcquisitionError("Q3 archive contains no CSV members")
    schema_sha = hashlib.sha256(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    members.sort(key=lambda item: item["date"])
    return {
        "status": "verified",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "member_count": len(members),
        "dates": [item["date"] for item in members],
        "schema_columns": len(schema),
        "schema_sha256": schema_sha,
        "schema": schema,
        "members": members,
        "extras": extras,
        "expanded_bytes": expanded_total,
        "totals": {
            "source_rows": total_source,
            "selected_rows": total_selected,
            "failure_rows": total_failure,
            "unique_q3_serial_models": len(all_serial_models),
        },
    }


def _download(url: str, destination: Path, stderr_path: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--max-time",
                str(timeout_seconds),
                "--output",
                str(destination),
                url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            check=False,
        )
    if result.returncode != 0:
        raise AcquisitionError(f"archive download failed with exit {result.returncode}")


def execute(
    config: dict,
    *,
    plan: dict,
    archive_override: Path | None = None,
) -> dict:
    evidence_root = project_path(plan["evidence_root"], "evidence root")
    attempt_root = project_path(plan["attempt_root"], "attempt root")
    archive = project_path(archive_override, "archive") if archive_override else attempt_root / "data_Q3_2023.zip"
    partial_archive = archive.with_name(archive.name + ".partial")
    evidence_root.mkdir(parents=True, exist_ok=True)
    attempt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = attempt_root / "verified_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "verified" and receipt.get("archive_sha256") == sha256_file(archive):
            return receipt
        raise AcquisitionError("existing Q3 attempt has an invalid receipt; preserve it and use a new attempt")
    headers_path = attempt_root / "head.txt"
    if archive_override is None:
        headers = head_source()
        headers_path.write_text("\n".join(f"{key}: {value}" for key, value in headers.items()) + "\n", encoding="utf-8")
        _download(EXPECTED_URL, partial_archive, attempt_root / "download.stderr", int(config["resources"]["download_attempt_seconds"]))
        archive = partial_archive
    else:
        headers = {"__status": "local_fixture", "content-length": str(archive.stat().st_size)}
    verification = validate_archive(archive)
    if verification["archive_bytes"] != EXPECTED_ARCHIVE_BYTES and archive_override is None:
        raise AcquisitionError("downloaded archive does not match the reviewed byte count")
    if archive_override is None:
        final_archive = attempt_root / "data_Q3_2023.zip"
        os.replace(archive, final_archive)
        archive = final_archive
    receipt = {
        "receipt_version": "q3-validation-receipt-v1",
        "status": "verified",
        "verified_at_utc": utc_now(),
        "url": EXPECTED_URL,
        "file_id": EXPECTED_FILE_ID,
        "head": headers,
        "archive_path": str(archive.relative_to(ROOT)),
        **verification,
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(Path(__file__)),
        "training_approval": False,
        "q4_access": False,
        "remote_setup": False,
    }
    receipt["receipt_hash"] = canonical_hash(receipt)
    atomic_json(receipt_path, receipt)
    atomic_json(evidence_root / "source_receipt_v1.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--archive", type=Path, help="local archive fixture; never used without --execute")
    parser.add_argument("--plan", type=Path, default=DEFAULT_EVIDENCE / "plan_v1.json")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    config = load_config()
    plan_path = project_path(args.plan, "plan")
    if not args.execute:
        plan = make_plan(config, evidence_root=args.evidence_root, raw_root=args.raw_root)
        atomic_json(plan_path, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.archive is not None and not args.archive.is_file():
        raise AcquisitionError(f"fixture archive is missing: {args.archive}")
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("plan_hash") != canonical_hash({key: value for key, value in plan.items() if key != "plan_hash"}):
            raise AcquisitionError("plan hash mismatch")
    else:
        plan = make_plan(config, evidence_root=args.evidence_root, raw_root=args.raw_root)
        atomic_json(plan_path, plan)
    receipt = execute(config, plan=plan, archive_override=args.archive)
    print(json.dumps({"status": receipt["status"], "archive_sha256": receipt["archive_sha256"], "member_count": receipt["member_count"], "selected_rows": receipt["totals"]["selected_rows"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as exc:
        raise SystemExit(f"Q3_ACQUISITION_FAILED: {exc}")
