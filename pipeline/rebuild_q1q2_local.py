"""Build and audit the verified Q1/Q2 panel from local member bytes.

The source input manifest has already been verified.  This entry point performs the
remaining mechanical build in one process, one date at a time.  It never
opens a network connection and it does not use any of the older candidate
panels as an ingestion source.  A partially written database is resumable,
but a different manifest, code fingerprint, or model is rejected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.acquire_q2_missing import advance_coverage
from pipeline.member_io import REQUIRED_FIELDS
from pipeline.panel import DECLARED_SCHEMAS, PanelWriter, SCHEMA_VERSION


MODEL = "ST4000DM000"
MANIFEST_DEFAULT = ROOT / "evidence/q2/input_review/q1q2_build_manifest_v1.json"
DATABASE_DEFAULT = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
EVIDENCE_DEFAULT = ROOT / "evidence/q2/rebuild_v1"
LEGACY_DATABASE_DEFAULT = ROOT / "data/derived/panel_q2_candidate.sqlite"
EXPECTED_DATES = [
    (dt.date(2023, 1, 1) + dt.timedelta(days=index)).isoformat()
    for index in range(181)
]
MAX_MEMBER_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
MAX_OWNED_BYTES = 4 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
SOFT_TIMEOUT_SECONDS = 60 * 60
HEX64 = set("0123456789abcdef")
DIGEST_FIELDS = (
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


class BuildStopped(RuntimeError):
    """A resource or integrity gate stopped the local build."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict, excluded: tuple[str, ...] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _rss_max_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _owned_bytes(database: Path, evidence_root: Path) -> int:
    total = 0
    candidates = [
        database,
        database.with_name(database.name + "-wal"),
        database.with_name(database.name + "-shm"),
    ]
    if evidence_root.exists():
        candidates.extend(path for path in evidence_root.rglob("*") if path.is_file())
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        total += resolved.stat().st_size
    return total


def _resource_snapshot(database: Path, evidence_root: Path) -> dict:
    return {
        "rss_max_bytes": _rss_max_bytes(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "owned_bytes": _owned_bytes(database, evidence_root),
    }


def _relative_project_path(path: Path, field: str) -> Path:
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise BuildStopped(f"{field} escapes project root: {path}")
    return resolved


def _validate_hex(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value.lower()) - HEX64:
        raise BuildStopped(f"{field} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _load_manifest(path: Path) -> tuple[dict, list[dict], str, str]:
    path = _relative_project_path(path, "manifest")
    if not path.is_file():
        raise BuildStopped(f"manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared_hash = _validate_hex(manifest.get("manifest_hash"), "manifest_hash")
    actual_hash = _canonical_hash(manifest, ("manifest_hash",))
    if declared_hash != actual_hash:
        raise BuildStopped(f"manifest hash mismatch: {actual_hash} != {declared_hash}")
    if manifest.get("verification_status") != "local_source_bytes_verified":
        raise BuildStopped("manifest is not marked local_source_bytes_verified")
    if manifest.get("training_approval") is not False:
        raise BuildStopped("source input manifest must keep training_approval=false")

    input_hashes = manifest.get("input_sha256") or {}
    for relative, expected in input_hashes.items():
        input_path = _relative_project_path(Path(relative), "manifest input")
        if not input_path.is_file():
            raise BuildStopped(f"manifest input is missing: {relative}")
        actual = _sha256_file(input_path)
        if actual != expected:
            raise BuildStopped(f"manifest input SHA mismatch for {relative}: {actual} != {expected}")

    for relative, expected in (manifest.get("code_sha256") or {}).items():
        code_path = _relative_project_path(Path(relative), "manifest code")
        if not code_path.is_file():
            raise BuildStopped(f"manifest code file is missing: {relative}")
        actual = _sha256_file(code_path)
        if actual != expected:
            raise BuildStopped(f"manifest code SHA mismatch for {relative}: {actual} != {expected}")

    members = [dict(item) for item in manifest.get("members", [])]
    if len(members) != 181 or manifest.get("member_count") != 181:
        raise BuildStopped(f"manifest must contain exactly 181 members, found {len(members)}")
    members.sort(key=lambda item: item.get("date", ""))
    if [item.get("date") for item in members] != EXPECTED_DATES:
        raise BuildStopped("manifest dates are not the complete 2023-01-01..2023-06-30 calendar")

    for entry in members:
        date = entry["date"]
        name = entry.get("name") or ""
        if not name.endswith(f"{date}.csv"):
            raise BuildStopped(f"member/date mismatch in manifest: {name} vs {date}")
        source_sha = _validate_hex(entry.get("source_sha256"), f"source_sha256:{date}")
        reader_sha = _validate_hex(entry.get("sha256"), f"sha256:{date}")
        if source_sha != reader_sha:
            raise BuildStopped(f"reader SHA and source SHA differ for {date}")
        local = _relative_project_path(Path(entry.get("local", "")), f"local member:{date}")
        if not local.is_file():
            raise BuildStopped(f"local member is missing for {date}: {local}")
        if "range_bytes" in entry and int(entry["range_bytes"]) != local.stat().st_size:
            raise BuildStopped(f"local member size differs from range_bytes for {date}")
        if int(entry.get("size", 0)) <= 0 or int(entry["size"]) > MAX_MEMBER_EXPANDED_BYTES:
            raise BuildStopped(f"expanded member size exceeds 256 MiB gate for {date}")
        counts = entry.get("expected_counts") or {}
        for field in ("source_rows", "selected_rows", "failure_rows", "schema_columns"):
            if int(counts.get(field, -1)) < 0:
                raise BuildStopped(f"invalid expected {field} for {date}")
        family = name.split("/", 1)[0]
        declared = DECLARED_SCHEMAS.get(family)
        if declared is None or int(counts["schema_columns"]) != declared["count"] or counts["schema_sha256"] != declared["sha256"]:
            raise BuildStopped(f"declared schema facts do not match {family} for {date}")

    totals = {
        field: sum(int(item["expected_counts"][field]) for item in members)
        for field in ("source_rows", "selected_rows", "failure_rows")
    }
    if totals != manifest.get("expected_totals"):
        raise BuildStopped(f"manifest expected totals are inconsistent: {totals}")
    expected_totals = {"source_rows": 43_290_148, "selected_rows": 3_241_790, "failure_rows": 337}
    if totals != expected_totals:
        raise BuildStopped(f"manifest totals differ from locked contract: {totals}")
    return manifest, members, _sha256_file(path), declared_hash


def _build_code_fingerprints() -> dict[str, str]:
    paths = {
        "rebuild_script": Path(__file__),
        "panel": Path(__file__).with_name("panel.py"),
        "member_io": Path(__file__).with_name("member_io.py"),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _config_hash(manifest_hash: str, manifest_file_sha256: str, code: dict[str, str], database: Path) -> str:
    config = {
        "database": str(database.relative_to(ROOT)),
        "manifest_hash": manifest_hash,
        "manifest_file_sha256": manifest_file_sha256,
        "code": code,
        "model": MODEL,
        "schema_version": SCHEMA_VERSION,
        "expected_dates": EXPECTED_DATES,
    }
    return _canonical_hash(config)


def _expected_by_date(members: list[dict]) -> dict[str, dict]:
    return {item["date"]: item for item in members}


def _validate_committed(writer: PanelWriter, members: list[dict]) -> list[str]:
    expected = _expected_by_date(members)
    rows = writer.connection.execute(
        "SELECT date, source_member, source_sha256, source_rows, selected_rows, "
        "failure_rows, schema_columns, schema_sha256 FROM member_counts ORDER BY date"
    ).fetchall()
    dates = [row["date"] for row in rows]
    if any(date not in expected for date in dates) or len(set(dates)) != len(dates):
        raise BuildStopped("existing database has a member_counts date outside the manifest or a duplicate")
    if dates != EXPECTED_DATES[: len(dates)]:
        raise BuildStopped("existing committed dates are not an ordered prefix of the manifest")
    for row in rows:
        item = expected[row["date"]]
        counts = item["expected_counts"]
        facts = {
            "source_member": item["name"],
            "source_sha256": item["source_sha256"],
            "source_rows": int(counts["source_rows"]),
            "selected_rows": int(counts["selected_rows"]),
            "failure_rows": int(counts["failure_rows"]),
            "schema_columns": int(counts["schema_columns"]),
            "schema_sha256": counts["schema_sha256"],
        }
        for key, value in facts.items():
            if row[key] != value:
                raise BuildStopped(f"committed source fact changed for {row['date']}: {key}")
        daily = writer.connection.execute(
            "SELECT COUNT(*) AS selected_rows, COALESCE(SUM(failure), 0) AS failure_rows, "
            "COUNT(DISTINCT source_member) AS members, COUNT(DISTINCT source_sha256) AS hashes "
            "FROM daily WHERE date = ?",
            (row["date"],),
        ).fetchone()
        if int(daily["selected_rows"]) != int(counts["selected_rows"]):
            raise BuildStopped(f"daily selected rows do not match member_counts for {row['date']}")
        if int(daily["failure_rows"]) != int(counts["failure_rows"]):
            raise BuildStopped(f"daily failure rows do not match member_counts for {row['date']}")
        if int(counts["selected_rows"]) and (daily["members"] != 1 or daily["hashes"] != 1):
            raise BuildStopped(f"daily source identity is not unique for {row['date']}")
    conflicts = writer.connection.execute(
        "SELECT COUNT(*) FROM (SELECT serial_number FROM daily GROUP BY serial_number "
        "HAVING COUNT(DISTINCT model) > 1)"
    ).fetchone()[0]
    if conflicts:
        raise BuildStopped(f"selected daily serial/model conflicts: {conflicts}")
    return dates


def _digest_selected(connection: sqlite3.Connection, date: str) -> tuple[str, int]:
    rows = connection.execute(
        "SELECT serial_number, model, capacity_bytes, failure, smart_5_raw, smart_9_raw, "
        "smart_187_raw, smart_188_raw, smart_197_raw, smart_198_raw "
        "FROM daily WHERE date = ? ORDER BY serial_number",
        (date,),
    ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row[index] for index in range(len(DIGEST_FIELDS))], ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(rows)


def _compare_legacy(database: Path, members: list[dict], output: Path) -> dict:
    if not LEGACY_DATABASE_DEFAULT.is_file():
        raise BuildStopped(f"legacy candidate panel is missing: {LEGACY_DATABASE_DEFAULT}")
    current = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    legacy = sqlite3.connect(f"file:{LEGACY_DATABASE_DEFAULT}?mode=ro", uri=True)
    current.row_factory = sqlite3.Row
    legacy.row_factory = sqlite3.Row
    records: list[dict] = []
    content_mismatches: list[str] = []
    range_sha_mismatches: list[str] = []
    try:
        legacy_dates = [row[0] for row in legacy.execute("SELECT date FROM member_counts ORDER BY date")]
        if legacy_dates != EXPECTED_DATES:
            raise BuildStopped("legacy candidate does not cover the locked 181-day calendar")
        for item in members:
            date = item["date"]
            old = legacy.execute(
                "SELECT source_member, source_sha256, source_rows, selected_rows, failure_rows, "
                "schema_columns, schema_sha256 FROM member_counts WHERE date = ?", (date,)
            ).fetchone()
            if old is None:
                raise BuildStopped(f"legacy candidate lacks member_counts for {date}")
            new = current.execute(
                "SELECT source_member, source_sha256, source_rows, selected_rows, failure_rows, "
                "schema_columns, schema_sha256 FROM member_counts WHERE date = ?", (date,)
            ).fetchone()
            current_digest, current_rows = _digest_selected(current, date)
            legacy_digest, legacy_rows = _digest_selected(legacy, date)
            # The legacy Q1 candidate predates persisted schema hashes.  The
            # retained audit established that an empty legacy value means the
            # declared family schema, so record that normalization explicitly
            # instead of treating metadata absence as source-content drift.
            legacy_schema = old["schema_sha256"] or new["schema_sha256"]
            counts_match = all(
                old[field] == new[field]
                for field in ("source_member", "source_rows", "selected_rows", "failure_rows", "schema_columns")
            ) and legacy_schema == new["schema_sha256"]
            content_match = current_rows == legacy_rows and current_digest == legacy_digest
            sha_match = old["source_sha256"] == new["source_sha256"]
            if not sha_match:
                range_sha_mismatches.append(date)
            if not counts_match or not content_match:
                content_mismatches.append(date)
            records.append({
                "date": date,
                "legacy_source_sha256": old["source_sha256"],
                "current_source_sha256": new["source_sha256"],
                "legacy_range_sha_match": sha_match,
                "legacy_counts_match": counts_match,
                "legacy_schema_sha256": old["schema_sha256"],
                "legacy_schema_normalized": old["schema_sha256"] in ("", new["schema_sha256"]),
                "legacy_selected_rows": legacy_rows,
                "current_selected_rows": current_rows,
                "legacy_selected_digest": legacy_digest,
                "current_selected_digest": current_digest,
                "selected_content_match": content_match,
            })
    finally:
        current.close()
        legacy.close()
    result = {
        "status": "pass" if not content_mismatches else "fail",
        "legacy_database": str(LEGACY_DATABASE_DEFAULT),
        "members": records,
        "range_sha_mismatches": range_sha_mismatches,
        "content_mismatches": content_mismatches,
        "comparison_method": "sorted selected daily digest over serial/model/capacity/failure/SMART fields",
    }
    _atomic_json(output, result)
    if content_mismatches:
        raise BuildStopped(f"legacy selected content/count comparison failed for {content_mismatches[:5]}")
    return result


def _panel_qa(writer: PanelWriter, members: list[dict], output: Path) -> dict:
    expected = _expected_by_date(members)
    actual_dates = [row[0] for row in writer.connection.execute("SELECT date FROM member_counts ORDER BY date")]
    failures: list[str] = []
    if actual_dates != EXPECTED_DATES:
        failures.append("member_counts does not cover exactly 181 dates")
    totals = {
        field: writer.connection.execute(f"SELECT SUM({field}) FROM member_counts").fetchone()[0]
        for field in ("source_rows", "selected_rows", "failure_rows")
    }
    expected_totals = {field: sum(int(item["expected_counts"][field]) for item in members) for field in totals}
    if totals != expected_totals:
        failures.append(f"totals mismatch: {totals} != {expected_totals}")

    member_facts = {
        row["date"]: dict(row)
        for row in writer.connection.execute("SELECT * FROM member_counts ORDER BY date")
    }
    daily_by_date = {
        row["date"]: dict(row)
        for row in writer.connection.execute(
            "SELECT date, COUNT(*) AS selected_rows, COALESCE(SUM(failure), 0) AS failure_rows "
            "FROM daily GROUP BY date ORDER BY date"
        )
    }
    mismatched_dates = []
    for date, item in expected.items():
        fact = member_facts.get(date)
        if fact is None:
            mismatched_dates.append(date)
            continue
        counts = item["expected_counts"]
        if any(fact[field] != (item["name"] if field == "source_member" else item["source_sha256"] if field == "source_sha256" else int(counts[field]) if field in ("source_rows", "selected_rows", "failure_rows", "schema_columns") else counts[field]) for field in ("source_member", "source_sha256", "source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256")):
            mismatched_dates.append(date)
        daily = daily_by_date.get(date, {"selected_rows": 0, "failure_rows": 0})
        if int(daily["selected_rows"]) != int(counts["selected_rows"]) or int(daily["failure_rows"]) != int(counts["failure_rows"]):
            mismatched_dates.append(date)
    if mismatched_dates:
        failures.append(f"daily/member_counts mismatch dates: {sorted(set(mismatched_dates))[:10]}")

    integrity = writer.connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        failures.append(f"sqlite integrity_check={integrity}")
    registry_scope = writer.connection.execute(
        "SELECT value FROM metadata WHERE key='serial_model_registry_scope'"
    ).fetchone()
    registry_scope_value = registry_scope[0] if registry_scope else None
    if registry_scope_value != "full_source_rows":
        failures.append(f"serial registry scope={registry_scope_value!r}")
    registry_rows = writer.connection.execute("SELECT COUNT(*) FROM serial_model_registry").fetchone()[0]
    serials = writer.connection.execute("SELECT COUNT(DISTINCT serial_number) FROM daily").fetchone()[0]
    identity_conflicts = writer.connection.execute(
        "SELECT COUNT(*) FROM (SELECT serial_number FROM daily GROUP BY serial_number HAVING COUNT(DISTINCT model) > 1)"
    ).fetchone()[0]
    if identity_conflicts:
        failures.append(f"serial/model conflicts={identity_conflicts}")

    positive_capacity = [
        dict(row)
        for row in writer.connection.execute(
            "SELECT capacity_bytes, COUNT(*) AS rows FROM daily WHERE capacity_bytes > 0 "
            "GROUP BY capacity_bytes ORDER BY capacity_bytes"
        )
    ]
    unexpected_capacity = [row for row in positive_capacity if row["capacity_bytes"] != 4_000_787_030_016]
    if unexpected_capacity:
        failures.append(f"unexpected positive capacity values={unexpected_capacity}")
    capacity_inconsistency = writer.connection.execute(
        "SELECT COUNT(*) FROM daily WHERE "
        "((capacity_bytes IS NULL OR capacity_bytes = -1) AND (capacity_clean_bytes IS NOT NULL OR capacity_missing != 1)) OR "
        "(capacity_bytes > 0 AND (capacity_clean_bytes != capacity_bytes OR capacity_missing != 0))"
    ).fetchone()[0]
    if capacity_inconsistency:
        failures.append(f"capacity derivation inconsistencies={capacity_inconsistency}")

    negative_smart = {
        field: writer.connection.execute(f"SELECT COUNT(*) FROM daily WHERE {field} < 0").fetchone()[0]
        for field in SMART_FIELDS
    }
    if any(negative_smart.values()):
        failures.append(f"negative SMART values={negative_smart}")
    decreases: dict[str, int] = {}
    for field in ("smart_5_raw", "smart_9_raw", "smart_187_raw"):
        decreases[field] = writer.connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {field}, LAG({field}) OVER "
            f"(PARTITION BY serial_number ORDER BY date) AS previous FROM daily WHERE {field} IS NOT NULL) "
            f"WHERE previous IS NOT NULL AND {field} < previous"
        ).fetchone()[0]
    if any(decreases.values()):
        failures.append(f"SMART 5/9/187 decreases={decreases}")

    counts_for_coverage = {
        row["date"]: {"source_rows": row["source_rows"], "selected_rows": row["selected_rows"], "origin": "verified_panel"}
        for row in writer.connection.execute("SELECT date, source_rows, selected_rows FROM member_counts")
    }
    last, coverage, low_streak, low_total, stop_reason = advance_coverage(
        "2023-03-31", "2023-06-30", counts_for_coverage
    )
    low_dates = [record["date"] for record in coverage if record["low"]]
    if last != "2023-06-30" or stop_reason is not None or low_dates != ["2023-04-13", "2023-04-16"]:
        failures.append(f"Q2 coverage replay failed: last={last}, low_dates={low_dates}, stop={stop_reason}")

    summary = {
        "status": "pass" if not failures else "fail",
        "database": str(writer.path),
        "schema_version": SCHEMA_VERSION,
        "member_count": len(actual_dates),
        "daily_rows": writer.connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0],
        "unique_selected_serials": serials,
        "registry_rows": registry_rows,
        "identity_conflicts": identity_conflicts,
        "totals": totals,
        "expected_totals": expected_totals,
        "integrity_check": integrity,
        "registry_scope": registry_scope_value,
        "positive_capacity_values": positive_capacity,
        "unexpected_positive_capacity_values": unexpected_capacity,
        "capacity_derivation_inconsistencies": capacity_inconsistency,
        "negative_smart_values": negative_smart,
        "adjacent_decreases_5_9_187": decreases,
        "coverage": {
            "last_date": last,
            "low_dates": low_dates,
            "low_streak": low_streak,
            "low_total": low_total,
            "stop_reason": stop_reason,
            "records": coverage,
        },
        "failures": failures,
    }
    _atomic_json(output, summary)
    if failures:
        raise BuildStopped("panel QA failed: " + "; ".join(failures[:5]))
    return summary


def _progress_base(manifest_path: Path, database: Path, manifest_hash: str, config_hash: str, code: dict[str, str]) -> dict:
    return {
        "status": "running",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_hash": manifest_hash,
        "database": str(database.relative_to(ROOT)),
        "config_hash": config_hash,
        "code": code,
        "selected_model": MODEL,
        "expected_members": len(EXPECTED_DATES),
        "completed_dates": [],
        "members": [],
        "errors": [],
        "started_at_utc": _utc_now(),
    }


def _metadata_values(manifest_path: Path, manifest_hash: str, manifest_file_sha256: str, config_hash: str, code: dict[str, str], status: str) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_model": MODEL,
        "build_manifest": str(manifest_path.relative_to(ROOT)),
        "build_manifest_hash": manifest_hash,
        "build_manifest_file_sha256": manifest_file_sha256,
        "build_config_hash": config_hash,
        "build_rebuild_script_sha256": code["rebuild_script"],
        "build_panel_code_sha256": code["panel"],
        "build_member_io_code_sha256": code["member_io"],
        "build_status": status,
        "training_approval": "false",
        "labels_status": "not_started",
        "serial_model_registry_scope": "full_source_rows",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--database", type=Path, default=DATABASE_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = _relative_project_path(args.manifest, "manifest")
    database = _relative_project_path(args.database, "database")
    evidence_root = _relative_project_path(args.evidence_root, "evidence root")
    progress_path = evidence_root / "build_progress_v1.json"
    panel_qa_path = evidence_root / "panel_qa_v1.json"
    legacy_comparison_path = evidence_root / "legacy_comparison_v1.json"
    manifest, members, manifest_file_sha256, manifest_hash = _load_manifest(manifest_path)
    code = _build_code_fingerprints()
    config_hash = _config_hash(manifest_hash, manifest_file_sha256, code, database)

    if database.exists() and not args.resume:
        raise BuildStopped(f"database already exists; use --resume explicitly: {database}")
    if progress_path.exists() and not args.resume:
        raise BuildStopped(f"progress evidence already exists; use --resume explicitly: {progress_path}")
    database.parent.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    writer = PanelWriter(database, reset=False, schema_declarations=DECLARED_SCHEMAS)
    progress = _progress_base(manifest_path, database, manifest_hash, config_hash, code)
    try:
        metadata = {row["key"]: row["value"] for row in writer.connection.execute("SELECT key, value FROM metadata")}
        if args.resume:
            for key, value in {
                "build_manifest_hash": manifest_hash,
                "build_manifest_file_sha256": manifest_file_sha256,
                "build_config_hash": config_hash,
                "selected_model": MODEL,
                "build_rebuild_script_sha256": code["rebuild_script"],
                "build_panel_code_sha256": code["panel"],
                "build_member_io_code_sha256": code["member_io"],
            }.items():
                if metadata.get(key) != value:
                    raise BuildStopped(f"resume metadata mismatch for {key}")
            if progress_path.exists():
                prior = json.loads(progress_path.read_text(encoding="utf-8"))
                if prior.get("config_hash") != config_hash or prior.get("manifest_hash") != manifest_hash:
                    raise BuildStopped("existing progress evidence belongs to a different build")
                progress.update(prior)
        else:
            writer.set_metadata(_metadata_values(manifest_path, manifest_hash, manifest_file_sha256, config_hash, code, "running"))
            _atomic_json(progress_path, progress)

        committed = _validate_committed(writer, members)
        if len(committed) == len(EXPECTED_DATES) and metadata.get("build_status") == "panel_complete":
            _panel_qa(writer, members, panel_qa_path)
            _compare_legacy(database, members, legacy_comparison_path)
            print(json.dumps({"status": "already_complete", "database": str(database)}, ensure_ascii=False, indent=2))
            return 0

        if committed and metadata.get("build_status") == "panel_complete":
            raise BuildStopped("database says panel_complete but committed dates are incomplete")
        completed_dates = set(committed)
        records_by_date = {record["date"]: record for record in progress.get("members", [])}
        started = time.monotonic()
        for index, entry in enumerate(members, start=1):
            date = entry["date"]
            if date in completed_dates:
                continue
            if completed_dates and date != EXPECTED_DATES[len(completed_dates)]:
                raise BuildStopped(f"resume would skip an uncommitted date before {date}")
            if time.monotonic() - started >= SOFT_TIMEOUT_SECONDS:
                raise BuildStopped("soft timeout reached before next member")
            before = _resource_snapshot(database, evidence_root)
            if before["free_bytes"] < MIN_FREE_BYTES:
                raise BuildStopped(f"free space below 2 GiB before {date}: {before['free_bytes']}")
            if before["rss_max_bytes"] >= MAX_RSS_BYTES:
                raise BuildStopped(f"RSS stop line reached before {date}: {before['rss_max_bytes']}")
            if before["owned_bytes"] >= MAX_OWNED_BYTES:
                raise BuildStopped(f"owned output budget reached before {date}: {before['owned_bytes']}")
            local = _relative_project_path(Path(entry["local"]), f"local member:{date}")
            actual_sha = _sha256_file(local)
            if actual_sha != entry["sha256"]:
                raise BuildStopped(f"local member SHA mismatch for {date}: {actual_sha} != {entry['sha256']}")
            member_start = time.monotonic()
            member_peak = before["rss_max_bytes"]
            samples = 0

            def progress_callback(sample: dict) -> None:
                nonlocal member_peak, samples
                samples += 1
                snapshot = _resource_snapshot(database, evidence_root)
                member_peak = max(member_peak, snapshot["rss_max_bytes"])
                if snapshot["rss_max_bytes"] >= MAX_RSS_BYTES:
                    raise BuildStopped(f"RSS stop line reached while reading {date}: {snapshot['rss_max_bytes']}")
                if snapshot["owned_bytes"] >= MAX_OWNED_BYTES:
                    raise BuildStopped(f"owned output budget reached while reading {date}: {snapshot['owned_bytes']}")
                if time.monotonic() - started >= SOFT_TIMEOUT_SECONDS:
                    raise BuildStopped(f"soft timeout reached while reading {date}")

            result = writer.append_member(
                entry,
                ROOT,
                MODEL,
                progress_callback=progress_callback,
            )
            after = _resource_snapshot(database, evidence_root)
            member_peak = max(member_peak, after["rss_max_bytes"])
            expected = entry["expected_counts"]
            for field in ("source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256"):
                if result[field] != (int(expected[field]) if field != "schema_sha256" else expected[field]):
                    raise BuildStopped(f"member result mismatch for {date}: {field}")
            if result["source_sha256"] != entry["source_sha256"]:
                raise BuildStopped(f"member source SHA mismatch after read for {date}")
            completed_dates.add(date)
            records_by_date[date] = {
                "date": date,
                "source_member": result["source_member"],
                "source_sha256": result["source_sha256"],
                "source_rows": result["source_rows"],
                "selected_rows": result["selected_rows"],
                "failure_rows": result["failure_rows"],
                "schema_columns": result["schema_columns"],
                "schema_sha256": result["schema_sha256"],
                "resources_before": before,
                "resources_after": after,
                "member_peak_rss_max_bytes": member_peak,
                "row_samples": samples,
                "duration_seconds": round(time.monotonic() - member_start, 3),
            }
            progress["completed_dates"] = sorted(completed_dates)
            progress["members"] = [records_by_date[key] for key in sorted(records_by_date)]
            progress["last_completed_at_utc"] = _utc_now()
            _atomic_json(progress_path, progress)
            print(json.dumps({"progress": f"{len(completed_dates)}/{len(members)}", **records_by_date[date]}, ensure_ascii=False), flush=True)

        _validate_committed(writer, members)
        qa = _panel_qa(writer, members, panel_qa_path)
        comparison = _compare_legacy(database, members, legacy_comparison_path)
        writer.set_metadata({
            **_metadata_values(manifest_path, manifest_hash, manifest_file_sha256, config_hash, code, "panel_complete"),
            "build_panel_qa_status": qa["status"],
            "build_legacy_comparison_status": comparison["status"],
            "build_completed_at_utc": _utc_now(),
            "build_registry_rows": str(qa["registry_rows"]),
            "build_daily_rows": str(qa["daily_rows"]),
        })
        progress.update({
            "status": "panel_complete",
            "completed_dates": EXPECTED_DATES,
            "completed_at_utc": _utc_now(),
            "panel_qa": qa,
            "legacy_comparison": {
                "status": comparison["status"],
                "range_sha_mismatches": comparison["range_sha_mismatches"],
                "content_mismatches": comparison["content_mismatches"],
            },
        })
        _atomic_json(progress_path, progress)
        print(json.dumps({"status": "panel_complete", "database": str(database), "qa": qa}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        progress["status"] = "failed"
        progress.setdefault("errors", []).append({"type": type(exc).__name__, "message": str(exc), "at_utc": _utc_now()})
        try:
            progress["completed_dates"] = sorted(_validate_committed(writer, members))
        except Exception:
            pass
        _atomic_json(progress_path, progress)
        try:
            writer.set_metadata({"build_status": "failed", "build_error": str(exc), "training_approval": "false"})
        except Exception:
            pass
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
