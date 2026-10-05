"""Build the Q3 validation panel and H=7 labels from a verified ZIP.

The input archive is already verified by :mod:`acquire_q3_validation`.  This
module reads it again without extracting the nine-gigabyte expanded dataset,
keeps Q1/Q2 in a separate immutable database for history and prior-failure
state, and writes only a new Q3 validation database.  It never fits a model or
reads Q4.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import time
import zipfile

from .acquire_q3_validation import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_DATES,
    EXPECTED_FILE_ID,
    EXPECTED_URL,
    REQUIRED_FIELDS,
    SMART_FIELDS,
    AcquisitionError,
    canonical_hash,
    load_config,
    project_path,
    sha256_file,
)
from .labeling import classify_device_rows
from .panel import SCHEMA_VERSION, create_schema


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/q3_validation_v1.json"
PLAN_PATH = ROOT / "evidence/q3/validation_v1/plan_v1.json"
RECEIPT_PATH = ROOT / "evidence/q3/validation_v1/source_receipt_v1.json"
DEFAULT_DATABASE = ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite"
DEFAULT_EVIDENCE = ROOT / "evidence/q3/validation_v1"
PRIOR_DATABASE = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
PRIOR_DATABASE_SHA256 = "294062e4ebf5bb8c95ef61cc5d57a114f65c6954b803f0f327f9aa3c9652dfa2"
MODEL = "ST4000DM000"
NORMAL_CAPACITY = 4_000_787_030_016
HISTORY_DAYS = 14
MIN_HISTORY = 12
HORIZON = 7
SCORE_START = dt.date(2023, 7, 1)
SCORE_END = dt.date(2023, 9, 23)
LABEL_CUTOFF = dt.date(2023, 9, 30)
EVENT_START = dt.date(2023, 7, 8)
EVENT_END = dt.date(2023, 9, 24)
GROUP_FIELDS = ("vault_id", "pod_id", "datacenter", "cluster_id", "is_legacy_format")
MAX_RSS = 2 * 1024 * 1024 * 1024
MAX_PROJECT_BYTES = 4 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
HEX64 = set("0123456789abcdef")


class BuildStopped(RuntimeError):
    """A source, leakage, identity, coverage, or resource gate stopped work."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if __import__("sys").platform == "darwin" else value * 1024


def _owned_bytes(database: Path, evidence: Path) -> int:
    paths = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
    if evidence.exists():
        paths.extend(item for item in evidence.rglob("*") if item.is_file())
    total = 0
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            total += resolved.stat().st_size
        except OSError:
            continue
    return total


def resource_snapshot(database: Path, evidence: Path) -> dict[str, int]:
    return {
        "rss_bytes": _rss_bytes(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "project_owned_bytes": _owned_bytes(database, evidence),
    }


def check_resources(database: Path, evidence: Path, label: str) -> dict[str, int]:
    snapshot = resource_snapshot(database, evidence)
    if snapshot["rss_bytes"] > MAX_RSS:
        raise BuildStopped(f"RSS limit exceeded at {label}: {snapshot['rss_bytes']}")
    if snapshot["project_owned_bytes"] > MAX_PROJECT_BYTES:
        raise BuildStopped(f"project increment limit exceeded at {label}: {snapshot['project_owned_bytes']}")
    if snapshot["free_bytes"] < MIN_FREE_BYTES:
        raise BuildStopped(f"free-space floor exceeded at {label}: {snapshot['free_bytes']}")
    return snapshot


def _integer(value: str, field: str, member: str, row_number: int) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise BuildStopped(f"non-integer {field} in {member} row {row_number}") from exc


def _schema_hash(columns: list[str]) -> str:
    return hashlib.sha256(json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _receipt(path: Path = RECEIPT_PATH) -> dict:
    if not path.is_file():
        raise BuildStopped(f"verified Q3 receipt is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildStopped(f"invalid Q3 receipt: {exc}") from exc
    if value.get("status") != "verified":
        raise BuildStopped("Q3 receipt is not verified")
    if value.get("url") != EXPECTED_URL or value.get("file_id") != EXPECTED_FILE_ID:
        raise BuildStopped("Q3 receipt source identity differs from the locked source")
    if value.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES:
        raise BuildStopped("Q3 receipt archive size differs from locked source")
    receipt_hash = value.get("receipt_hash")
    if receipt_hash != canonical_hash({key: item for key, item in value.items() if key != "receipt_hash"}):
        raise BuildStopped("Q3 receipt hash mismatch")
    archive = project_path(value.get("archive_path", ""), "Q3 archive")
    if not archive.is_file():
        raise BuildStopped(f"Q3 archive is missing: {archive}")
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES or sha256_file(archive) != value.get("archive_sha256"):
        raise BuildStopped("Q3 archive does not match verified receipt")
    value["_archive"] = archive
    return value


def _load_plan(path: Path = PLAN_PATH) -> dict:
    if not path.is_file():
        raise BuildStopped(f"Q3 plan is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "planned":
        raise BuildStopped("Q3 plan is not in planned state")
    if value.get("plan_hash") != canonical_hash({key: item for key, item in value.items() if key != "plan_hash"}):
        raise BuildStopped("Q3 plan hash mismatch")
    if value.get("source", {}).get("dates") != EXPECTED_DATES:
        raise BuildStopped("Q3 plan date range differs from locked range")
    if value.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise BuildStopped("Q3 plan config hash differs from current config")
    if value.get("q4_access") is not False or value.get("remote_setup") is not False:
        raise BuildStopped("Q4 and remote flags must remain false")
    return value


def _prior_registry(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute("SELECT serial_number, model FROM serial_model_registry").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _prior_counts(connection: sqlite3.Connection) -> list[int]:
    values = [
        int(row[0])
        for row in connection.execute(
            "SELECT COUNT(*) FROM daily WHERE model=? AND date BETWEEN '2023-06-24' AND '2023-06-30' GROUP BY date ORDER BY date",
            (MODEL,),
        )
    ]
    if len(values) != 7:
        raise BuildStopped(f"prior panel lacks the 7-day Q3 coverage seed: {len(values)} days")
    return values


def coverage_gate(receipt: dict, prior: sqlite3.Connection) -> dict:
    seed = _prior_counts(prior)
    ordered = sorted(seed)
    baseline = (ordered[3] if len(ordered) % 2 else (ordered[3] + ordered[4]) / 2)
    threshold = baseline * 0.8
    by_date = {str(item["date"]): int(item["selected_rows"]) for item in receipt["members"]}
    low_dates: list[str] = []
    consecutive = 0
    stop_reason: str | None = None
    records = []
    for date in EXPECTED_DATES:
        count = by_date[date]
        low = count < threshold
        consecutive = consecutive + 1 if low else 0
        if low:
            low_dates.append(date)
        records.append({"date": date, "selected_rows": count, "baseline_median": baseline, "threshold": threshold, "low": low, "consecutive_low": consecutive, "cumulative_low": len(low_dates)})
        if consecutive >= 3:
            stop_reason = f"three_consecutive_low_coverage_days_through:{date}"
            break
        if len(low_dates) >= 10:
            stop_reason = f"more_than_9_low_coverage_days_through:{date}"
            break
    return {"status": "pass" if stop_reason is None else "stopped", "seed_counts": seed, "baseline_median": baseline, "threshold": threshold, "low_dates": low_dates, "stop_reason": stop_reason, "records": records}


def _member_rows(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected: dict,
    schema: list[str] | None,
    prior_registry: dict[str, str],
    q3_registry: dict[str, str],
) -> tuple[dict, list[tuple], list[str], dict[str, str]]:
    date = str(expected["date"])
    try:
        raw = zf.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BuildStopped(f"Q3 member CRC/decompression failed for {info.filename}: {exc}") from exc
    if len(raw) != int(info.file_size):
        raise BuildStopped(f"Q3 member expanded size differs for {date}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"] or len(raw) != int(expected["expanded_bytes"]):
        raise BuildStopped(f"Q3 member receipt mismatch for {date}")
    reader = csv.reader(io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline=""))
    try:
        columns = next(reader)
    except StopIteration as exc:
        raise BuildStopped(f"empty Q3 CSV member: {date}") from exc
    if columns != (schema or columns):
        raise BuildStopped(f"Q3 schema changed within quarter at {date}")
    if len(columns) != len(set(columns)):
        raise BuildStopped(f"duplicate Q3 column names at {date}")
    missing = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing:
        raise BuildStopped(f"Q3 required columns missing at {date}: {missing}")
    index = {name: columns.index(name) for name in columns}
    selected: list[tuple] = []
    source_rows = selected_rows = failure_rows = 0
    source_serials: set[str] = set()
    source_models: dict[str, str] = {}
    all_models = collections.Counter()
    groups = {field: collections.Counter() for field in GROUP_FIELDS if field in index}
    selected_groups = {field: collections.Counter() for field in GROUP_FIELDS if field in index}
    capacity_missing = 0
    smart_negative = collections.Counter()
    selected_digest = hashlib.sha256()
    for row_number, values in enumerate(reader, start=2):
        source_rows += 1
        if len(values) != len(columns):
            raise BuildStopped(f"Q3 row length mismatch at {date} row {row_number}")
        if values[index["date"]] != date:
            raise BuildStopped(f"Q3 member/date mismatch at {date} row {row_number}")
        serial = values[index["serial_number"]]
        model = values[index["model"]]
        if not serial or not model or serial in source_serials:
            raise BuildStopped(f"Q3 duplicate/empty identity at {date} row {row_number}")
        source_serials.add(serial)
        all_models[model] += 1
        prior_model = prior_registry.get(serial)
        if prior_model is not None and prior_model != model:
            raise BuildStopped(f"Q1/Q2 to Q3 model conflict for {serial}: {prior_model} vs {model}")
        q3_model = q3_registry.get(serial)
        if q3_model is not None and q3_model != model:
            raise BuildStopped(f"Q3 cross-date model conflict for {serial}: {q3_model} vs {model}")
        source_models[serial] = model
        failure = _integer(values[index["failure"]], "failure", info.filename, row_number)
        if failure not in (0, 1):
            raise BuildStopped(f"invalid Q3 failure at {date} row {row_number}")
        capacity = _integer(values[index["capacity_bytes"]], "capacity_bytes", info.filename, row_number)
        for field in GROUP_FIELDS:
            if field in index:
                groups[field][values[index[field]]] += 1
        if model != MODEL:
            continue
        if capacity is not None and capacity not in (-1, NORMAL_CAPACITY):
            raise BuildStopped(f"unexpected Q3 capacity at {date} row {row_number}: {capacity}")
        smart_values: list[int | None] = []
        missing_values: list[int] = []
        for field in SMART_FIELDS:
            value = _integer(values[index[field]], field, info.filename, row_number)
            if value is not None and value < 0:
                smart_negative[field] += 1
                raise BuildStopped(f"negative Q3 SMART value at {date} row {row_number}")
            smart_values.append(value)
            missing_values.append(int(value is None))
        selected_rows += 1
        failure_rows += int(failure or 0)
        capacity_missing += int(capacity is None or capacity == -1)
        for field in GROUP_FIELDS:
            if field in index:
                selected_groups[field][values[index[field]]] += 1
        selected.append((date, serial, MODEL, capacity, None if capacity is None or capacity == -1 else capacity, int(capacity is None or capacity == -1), failure, *smart_values, *missing_values, info.filename, digest, row_number))
        selected_digest.update(json.dumps([serial, MODEL, capacity, failure, *smart_values], ensure_ascii=False, separators=(",", ":")).encode())
        selected_digest.update(b"\n")
    facts = {
        "date": date,
        "name": info.filename,
        "source_sha256": digest,
        "source_rows": source_rows,
        "selected_rows": selected_rows,
        "failure_rows": failure_rows,
        "schema_columns": len(columns),
        "schema_sha256": _schema_hash(columns),
        "schema_json": columns,
        "compressed_bytes": int(info.compress_size),
        "expanded_bytes": int(info.file_size),
        "crc": int(info.CRC),
        "capacity_missing_or_sentinel": capacity_missing,
        "smart_negative_values": dict(smart_negative),
        "group_fields": list(groups),
        "all_model_counts": dict(all_models),
        "groups": {field: {"all_unique": len(counter), "selected_unique": len(selected_groups[field]), "all_rows": sum(counter.values()), "selected_rows": sum(selected_groups[field].values())} for field, counter in groups.items()},
        "selected_digest_sha256": selected_digest.hexdigest(),
    }
    return facts, selected, columns, source_models


def _metadata_set(connection: sqlite3.Connection, values: dict[str, str]) -> None:
    connection.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", values.items())


def _existing_dates(connection: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in connection.execute("SELECT date FROM member_counts ORDER BY date")]


def _panel_qa(connection: sqlite3.Connection, receipt: dict, coverage: dict, output: Path) -> dict:
    failures: list[str] = []
    dates = _existing_dates(connection)
    if dates != EXPECTED_DATES:
        failures.append(f"panel dates differ: {len(dates)}")
    expected_by_date = {str(item["date"]): item for item in receipt["members"]}
    member_rows = {str(row["date"]): dict(row) for row in connection.execute("SELECT * FROM member_counts")}
    for date, expected in expected_by_date.items():
        row = member_rows.get(date)
        if row is None:
            failures.append(f"missing member_counts {date}")
            continue
        schema_columns = expected.get("schema_columns", receipt.get("schema_columns"))
        schema_sha256 = expected.get("schema_sha256", receipt.get("schema_sha256"))
        for field, expected_value in (("source_sha256", expected["sha256"]), ("source_rows", expected["source_rows"]), ("selected_rows", expected["selected_rows"]), ("failure_rows", expected["failure_rows"]), ("schema_columns", schema_columns), ("schema_sha256", schema_sha256)):
            if row[field] != expected_value:
                failures.append(f"member fact mismatch {date}/{field}")
        daily = connection.execute("SELECT COUNT(*) AS n, COALESCE(SUM(failure),0) AS f FROM daily WHERE date=?", (date,)).fetchone()
        if int(daily["n"]) != int(expected["selected_rows"]) or int(daily["f"]) != int(expected["failure_rows"]):
            failures.append(f"daily count mismatch {date}")
    duplicate_daily = connection.execute("SELECT COUNT(*) FROM (SELECT date,serial_number,COUNT(*) n FROM daily GROUP BY date,serial_number HAVING n>1)").fetchone()[0]
    if duplicate_daily:
        failures.append(f"duplicate daily keys={duplicate_daily}")
    conflicts = connection.execute("SELECT COUNT(*) FROM (SELECT serial_number,COUNT(DISTINCT model) n FROM serial_model_registry GROUP BY serial_number HAVING n>1)").fetchone()[0]
    if conflicts:
        failures.append(f"registry conflicts={conflicts}")
    unexpected_capacity = connection.execute("SELECT COUNT(*) FROM daily WHERE capacity_bytes IS NOT NULL AND capacity_bytes NOT IN (-1,?)", (NORMAL_CAPACITY,)).fetchone()[0]
    if unexpected_capacity:
        failures.append(f"unexpected positive capacity={unexpected_capacity}")
    negative_smart = {field: int(connection.execute(f"SELECT COUNT(*) FROM daily WHERE {field}<0").fetchone()[0]) for field in SMART_FIELDS}
    if any(negative_smart.values()):
        failures.append(f"negative SMART={negative_smart}")
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        failures.append(f"integrity={integrity}")
    result = {
        "status": "pass" if not failures and coverage["status"] == "pass" else "fail",
        "database_rows": int(connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0]),
        "unique_serials": int(connection.execute("SELECT COUNT(DISTINCT serial_number) FROM daily").fetchone()[0]),
        "failure_rows": int(connection.execute("SELECT COALESCE(SUM(failure),0) FROM daily").fetchone()[0]),
        "member_count": len(dates),
        "schema": {"columns": int(connection.execute("SELECT schema_columns FROM member_counts LIMIT 1").fetchone()[0]) if dates else 0, "sha256": str(connection.execute("SELECT schema_sha256 FROM member_counts LIMIT 1").fetchone()[0]) if dates else ""},
        "coverage": coverage,
        "negative_smart": negative_smart,
        "integrity_check": integrity,
        "failures": failures,
    }
    atomic_json(output, result)
    if result["status"] != "pass":
        raise BuildStopped("Q3 panel QA failed: " + "; ".join(failures[:5]))
    return result


def build_panel(config: dict, receipt: dict, plan: dict, database: Path, evidence: Path, *, resume: bool = False) -> dict:
    if database.exists() and not resume:
        raise BuildStopped(f"Q3 panel exists; use --resume: {database}")
    evidence.mkdir(parents=True, exist_ok=True)
    prior = sqlite3.connect(f"file:{PRIOR_DATABASE}?mode=ro&immutable=1", uri=True)
    prior.row_factory = sqlite3.Row
    if sha256_file(PRIOR_DATABASE) != PRIOR_DATABASE_SHA256:
        prior.close()
        raise BuildStopped("prior verified panel SHA differs")
    coverage = coverage_gate(receipt, prior)
    atomic_json(evidence / "coverage_v1.json", coverage)
    if coverage["status"] != "pass":
        prior.close()
        raise BuildStopped(coverage["stop_reason"] or "coverage gate stopped")
    prior_registry = _prior_registry(prior)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    committed = _existing_dates(connection)
    expected_dates = EXPECTED_DATES
    if committed != expected_dates[: len(committed)]:
        connection.close(); prior.close()
        raise BuildStopped("existing Q3 dates are not an ordered prefix")
    code_hash = sha256_file(Path(__file__))
    config_hash = canonical_hash({"config": config, "plan_hash": plan["plan_hash"], "receipt_hash": receipt["receipt_hash"], "prior_panel_sha256": PRIOR_DATABASE_SHA256, "code_sha256": code_hash})
    _metadata_set(connection, {"schema_version": SCHEMA_VERSION, "selected_model": MODEL, "q3_source_url": EXPECTED_URL, "q3_source_file_id": EXPECTED_FILE_ID, "q3_archive_sha256": receipt["archive_sha256"], "q3_receipt_hash": receipt["receipt_hash"], "q3_plan_hash": plan["plan_hash"], "q3_build_config_hash": config_hash, "q3_build_code_sha256": code_hash, "prior_panel_sha256": PRIOR_DATABASE_SHA256, "training_approval": "false", "q4_access": "false", "remote_setup": "false", "panel_status": "running", "serial_model_registry_scope": "q3_source_rows"})
    connection.commit()
    q3_registry: dict[str, str] = {str(row[0]): str(row[1]) for row in connection.execute("SELECT serial_number,model FROM serial_model_registry")}
    schema: list[str] | None = None
    registered_schema = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_registry:data_Q3_2023'"
    ).fetchone()
    if registered_schema is not None:
        try:
            registered_value = json.loads(str(registered_schema[0]))
            registered_columns = registered_value.get("columns")
        except (TypeError, json.JSONDecodeError) as exc:
            connection.close(); prior.close()
            raise BuildStopped("existing Q3 schema registry is invalid") from exc
        if not isinstance(registered_columns, list) or not registered_columns:
            connection.close(); prior.close()
            raise BuildStopped("existing Q3 schema registry has no columns")
        schema = [str(column) for column in registered_columns]
    elif committed:
        existing_schema = connection.execute(
            "SELECT schema_json FROM member_counts ORDER BY date LIMIT 1"
        ).fetchone()
        if existing_schema is not None:
            try:
                existing_columns = json.loads(str(existing_schema[0]))
            except (TypeError, json.JSONDecodeError) as exc:
                connection.close(); prior.close()
                raise BuildStopped("existing Q3 member schema is invalid") from exc
            if not isinstance(existing_columns, list) or not existing_columns:
                connection.close(); prior.close()
                raise BuildStopped("existing Q3 member schema has no columns")
            schema = [str(column) for column in existing_columns]
    member_by_date = {str(item["date"]): item for item in receipt["members"]}
    progress = {"status": "running", "database": str(database.relative_to(ROOT)), "config_hash": config_hash, "completed_dates": committed, "members": [], "started_at_utc": utc_now()}
    try:
        with zipfile.ZipFile(receipt["_archive"], "r") as zf:
            infos = {str(Path(info.filename).name).removesuffix(".csv"): info for info in zf.infolist() if info.filename.endswith(".csv") and Path(info.filename).name[:10] in EXPECTED_DATES}
            for date in expected_dates[len(committed):]:
                info = infos.get(date)
                if info is None:
                    raise BuildStopped(f"Q3 ZIP member missing: {date}")
                facts, selected, columns, source_models = _member_rows(
                    zf,
                    info,
                    member_by_date[date],
                    schema,
                    prior_registry,
                    q3_registry,
                )
                if schema is None:
                    schema = columns
                elif columns != schema:
                    raise BuildStopped(f"Q3 schema changed at {date}")
                for serial, model in source_models.items():
                    q3_registry[serial] = model
                schema_key = "schema_registry:data_Q3_2023"
                with connection:
                    connection.executemany("INSERT INTO daily(date,serial_number,model,capacity_bytes,capacity_clean_bytes,capacity_missing,failure,smart_5_raw,smart_9_raw,smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw,smart_5_missing,smart_9_missing,smart_187_missing,smart_188_missing,smart_197_missing,smart_198_missing,source_member,source_sha256,source_row) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", selected)
                    connection.execute("INSERT INTO member_counts(date,source_member,source_sha256,source_rows,selected_rows,failure_rows,schema_columns,schema_sha256,schema_json,audit_json) VALUES (?,?,?,?,?,?,?,?,?,?)", (date, facts["name"], facts["source_sha256"], facts["source_rows"], facts["selected_rows"], facts["failure_rows"], facts["schema_columns"], facts["schema_sha256"], json.dumps(facts["schema_json"], ensure_ascii=False, separators=(",", ":")), json.dumps({key: value for key, value in facts.items() if key not in {"schema_json"}}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))))
                    # Persist the complete source identity for this member.  The
                    # in-memory registry still carries earlier dates for the
                    # cross-date conflict gate, while only current observations
                    # need a SQLite upsert here.
                    for serial, model in source_models.items():
                        connection.execute("INSERT INTO serial_model_registry(serial_number,model,first_date,last_date,first_source_member,source_scope) VALUES (?,?,?,?,?,?) ON CONFLICT(serial_number) DO UPDATE SET last_date=CASE WHEN excluded.last_date>last_date THEN excluded.last_date ELSE last_date END", (serial, model, date, date, facts["name"], "q3_source_rows"))
                    connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (schema_key, json.dumps({"sha256": facts["schema_sha256"], "columns": facts["schema_json"]}, ensure_ascii=False, separators=(",", ":"))))
                progress["completed_dates"].append(date)
                progress["members"].append(facts)
                progress["resource"] = check_resources(database, evidence, f"after_{date}")
                atomic_json(evidence / "panel_progress_v1.json", progress)
        _metadata_set(connection, {"panel_status": "complete", "q3_schema_sha256": _schema_hash(schema or []), "q3_schema_columns": str(len(schema or [])), "q3_panel_completed_at_utc": utc_now()})
        connection.commit()
        qa = _panel_qa(connection, receipt, coverage, evidence / "panel_qa_v1.json")
        progress["status"] = "complete"
        progress["qa_status"] = qa["status"]
        atomic_json(evidence / "panel_progress_v1.json", progress)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        manifest = {"status": "complete", "database": str(database.relative_to(ROOT)), "database_sha256": sha256_file(database), "config_sha256": sha256_file(CONFIG_PATH), "config_hash": config_hash, "plan_hash": plan["plan_hash"], "receipt_hash": receipt["receipt_hash"], "code_sha256": code_hash, "prior_panel_sha256": PRIOR_DATABASE_SHA256, "completed_dates": progress["completed_dates"], "training_approval": False, "q4_access": False, "remote_setup": False, "qa": str((evidence / "panel_qa_v1.json").relative_to(ROOT)), "published_at_utc": utc_now()}
        manifest["manifest_hash"] = canonical_hash(manifest)
        atomic_json(evidence / "panel_complete_manifest_v1.json", manifest)
        return {"status": "complete", "database": str(database), "qa": qa, "manifest": manifest}
    except BaseException as exc:
        progress["status"] = "failed"
        progress["failed_at_utc"] = utc_now()
        progress["error"] = str(exc)
        atomic_json(evidence / "panel_progress_v1.json", progress)
        try:
            _metadata_set(connection, {"panel_status": "failed", "panel_failure": str(exc)})
            connection.commit()
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        prior.close()


def _q3_rows_by_serial(prior: sqlite3.Connection, q3: sqlite3.Connection, serial: str) -> list[dict]:
    rows = []
    for row in prior.execute("SELECT date,serial_number,model,capacity_bytes,failure FROM daily WHERE serial_number=? AND date<=? ORDER BY date", (serial, "2023-06-30")):
        rows.append(dict(row))
    for row in q3.execute("SELECT date,serial_number,model,capacity_bytes,failure FROM daily WHERE serial_number=? AND date<=? ORDER BY date", (serial, "2023-09-30")):
        rows.append(dict(row))
    rows.sort(key=lambda row: row["date"])
    return rows


def _independent_label_check(rows: list[dict], flow: list[dict]) -> None:
    by_date = {str(row["date"]): row for row in rows}
    if len(by_date) != len(rows):
        raise BuildStopped("Q3 label source has duplicate serial/date")
    failure_dates = sorted(str(row["date"]) for row in rows if int(row["failure"] or 0) == 1)
    first_failure = failure_dates[0] if failure_dates else None
    for item in flow:
        decision = dt.date.fromisoformat(str(item["decision_date"]))
        if not SCORE_START <= decision <= SCORE_END:
            raise BuildStopped(f"Q3 label row outside score dates: {item['decision_date']}")
        if first_failure is not None and str(item["decision_date"]) > first_failure:
            expected_status, expected_label = "post_failure", None
            expected_eligible = 0
        elif int(by_date[str(item["decision_date"])]["failure"] or 0) == 1:
            expected_status, expected_label = "same_day_failure", None
            expected_eligible = 0
        else:
            history = sum((decision - dt.timedelta(days=offset)).isoformat() in by_date for offset in range(HISTORY_DAYS))
            future_dates = [(decision + dt.timedelta(days=offset)).isoformat() for offset in range(1, HORIZON + 1)]
            future_seen = [date in by_date for date in future_dates]
            if history < MIN_HISTORY:
                expected_status, expected_label = "history_insufficient", None
                expected_eligible = 0
            elif first_failure is not None and decision < dt.date.fromisoformat(first_failure) <= decision + dt.timedelta(days=HORIZON):
                expected_status = "positive_observed" if all(future_seen[: max(0, (dt.date.fromisoformat(first_failure) - decision).days - 1)]) else "positive_with_gap"
                expected_label, expected_eligible = 1, 1
            elif all(future_seen):
                expected_status, expected_label, expected_eligible = "negative_observed", 0, 1
            elif decision + dt.timedelta(days=HORIZON) > LABEL_CUTOFF:
                expected_status, expected_label, expected_eligible = "end_censored", None, 1
            else:
                expected_status, expected_label, expected_eligible = "gap_or_exit_censored", None, 1
        if (item["status"], item["label"], int(item["eligible"])) != (expected_status, expected_label, expected_eligible):
            raise BuildStopped(f"Q3 independent label mismatch at {item['serial_number']}/{item['decision_date']}")


def build_labels(config: dict, plan: dict, receipt: dict, database: Path, evidence: Path, *, replace: bool = False) -> dict:
    if not database.is_file():
        raise BuildStopped(f"Q3 panel database is missing: {database}")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    prior = sqlite3.connect(f"file:{PRIOR_DATABASE}?mode=ro&immutable=1", uri=True)
    prior.row_factory = sqlite3.Row
    try:
        metadata = {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key,value FROM metadata")}
        if metadata.get("panel_status") != "complete":
            raise BuildStopped("Q3 panel is not complete")
        if metadata.get("q3_receipt_hash") != receipt["receipt_hash"] or metadata.get("prior_panel_sha256") != PRIOR_DATABASE_SHA256:
            raise BuildStopped("Q3 panel provenance does not match receipt/prior panel")
        code_digest = hashlib.sha256()
        for path in (Path(__file__), Path(__file__).with_name("labeling.py"), Path(__file__).with_name("panel.py")):
            code_digest.update(path.name.encode()); code_digest.update(path.read_bytes())
        code_hash = code_digest.hexdigest()
        run_id = "validation_q3_h7_v1"
        split = "validation"
        config_hash = canonical_hash({"run_id": run_id, "split": split, "start": str(SCORE_START), "end": str(SCORE_END), "dataset_end": str(LABEL_CUTOFF), "horizon": HORIZON, "history_days": HISTORY_DAYS, "min_history": MIN_HISTORY, "receipt_hash": receipt["receipt_hash"], "prior_panel_sha256": PRIOR_DATABASE_SHA256, "code_hash": code_hash})
        existing = connection.execute("SELECT config_hash,status FROM label_runs WHERE run_id=? AND split=? AND horizon_days=?", (run_id, split, HORIZON)).fetchone()
        if existing and existing["config_hash"] != config_hash and not replace:
            raise BuildStopped("Q3 label run has different configuration; use a new run or --replace")
        serials = [str(row[0]) for row in connection.execute("SELECT DISTINCT serial_number FROM daily WHERE model=? ORDER BY serial_number", (MODEL,))]
        flows: list[dict] = []
        counters = collections.Counter()
        for serial in serials:
            source_rows = _q3_rows_by_serial(prior, connection, serial)
            classified = classify_device_rows(source_rows, start=str(SCORE_START), end=str(SCORE_END), dataset_end=str(LABEL_CUTOFF), horizon_days=HORIZON, history_days=HISTORY_DAYS, min_history=MIN_HISTORY)
            selected = [row for row in classified if SCORE_START <= dt.date.fromisoformat(str(row["decision_date"])) <= SCORE_END]
            _independent_label_check(source_rows, selected)
            for row in selected:
                row["run_id"] = run_id; row["split"] = split; row["config_hash"] = config_hash; row["horizon_days"] = HORIZON
                flows.append(row); counters[str(row["status"])] += 1
        source_manifest_rows = [list(row) for row in connection.execute("SELECT date,source_member,source_sha256,source_rows,selected_rows,failure_rows,schema_columns,schema_sha256 FROM member_counts ORDER BY date")]
        manifest_hash = hashlib.sha256(json.dumps(source_manifest_rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        with connection:
            connection.execute("DELETE FROM label_flow WHERE run_id=? AND split=? AND horizon_days=?", (run_id, split, HORIZON))
            connection.execute("DELETE FROM label_runs WHERE run_id=? AND split=? AND horizon_days=?", (run_id, split, HORIZON))
            connection.executemany("INSERT INTO label_flow(run_id,split,config_hash,horizon_days,decision_date,serial_number,model,capacity_bytes,first_failure_date,label,status,eligible,history_observations,future_observations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(row["run_id"],row["split"],row["config_hash"],row["horizon_days"],row["decision_date"],row["serial_number"],row["model"],row["capacity_bytes"],row["first_failure_date"],row["label"],row["status"],row["eligible"],row["history_observations"],row["future_observations"]) for row in flows])
            summary = {
                "flow_rows": len(flows),
                "status_counts": dict(counters),
                "eligible_rows": sum(int(row["eligible"]) for row in flows),
                "positive_rows": sum(int(row["label"] == 1) for row in flows),
                "negative_rows": sum(int(row["label"] == 0) for row in flows),
                "unknown_rows": sum(int(row["eligible"]) and int(row["label"] is None) for row in flows),
                "source_manifest_hash": manifest_hash,
                "prior_panel_sha256": PRIOR_DATABASE_SHA256,
                "receipt_hash": receipt["receipt_hash"],
            }
            connection.execute("INSERT INTO label_runs(run_id,split,horizon_days,start_date,end_date,dataset_end,config_hash,code_hash,manifest_hash,status,summary_json,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (run_id,split,HORIZON,str(SCORE_START),str(SCORE_END),str(LABEL_CUTOFF),config_hash,code_hash,manifest_hash,"complete",json.dumps(summary,ensure_ascii=False,sort_keys=True,separators=(",", ":")),utc_now()))
            connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", (f"label_summary:{run_id}:{split}:{HORIZON}", json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))))
        try:
            qa = independent_label_qa(connection, prior, flows, evidence / "label_qa_v1.json")
        except BuildStopped:
            with connection:
                connection.execute(
                    "UPDATE label_runs SET status='failed' WHERE run_id=? AND split=? AND horizon_days=?",
                    (run_id, split, HORIZON),
                )
            raise
        # Flush the label append before hashing the SQLite file.  The panel
        # manifest was created before label rows existed in this same database;
        # refresh its bound file hash explicitly so every published component
        # points at the final on-disk state.
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        database_sha256 = sha256_file(database)
        result = {
            "status": "complete",
            "run_id": run_id,
            "summary": summary,
            "qa": qa,
            "database": str(database.relative_to(ROOT)),
            "database_sha256": database_sha256,
        }
        labels_path = evidence / "labels_v1.json"
        atomic_json(labels_path, result)
        panel_manifest_path = evidence / "panel_complete_manifest_v1.json"
        if panel_manifest_path.is_file():
            panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
            panel_manifest["database_sha256"] = database_sha256
            panel_manifest["database_state_scope"] = "final_panel_plus_label_rows"
            panel_manifest["revalidated_after_label_run"] = run_id
            panel_manifest["revalidated_at_utc"] = utc_now()
            panel_manifest.pop("manifest_hash", None)
            panel_manifest["manifest_hash"] = canonical_hash(panel_manifest)
            atomic_json(panel_manifest_path, panel_manifest)
        final_manifest = {
            "manifest_version": "q3-validation-complete-v1",
            "status": "complete",
            "database": str(database.relative_to(ROOT)),
            "database_sha256": database_sha256,
            "database_state_scope": "final_panel_plus_label_rows",
            "source_receipt": str((evidence / "source_receipt_v1.json").relative_to(ROOT)),
            "source_receipt_file_sha256": sha256_file(evidence / "source_receipt_v1.json"),
            "plan": str((evidence / "plan_v1.json").relative_to(ROOT)),
            "plan_file_sha256": sha256_file(evidence / "plan_v1.json"),
            "coverage": str((evidence / "coverage_v1.json").relative_to(ROOT)),
            "coverage_file_sha256": sha256_file(evidence / "coverage_v1.json"),
            "panel_qa": str((evidence / "panel_qa_v1.json").relative_to(ROOT)),
            "panel_qa_file_sha256": sha256_file(evidence / "panel_qa_v1.json"),
            "panel_manifest": str(panel_manifest_path.relative_to(ROOT)),
            "panel_manifest_file_sha256": sha256_file(panel_manifest_path),
            "label_qa": str((evidence / "label_qa_v1.json").relative_to(ROOT)),
            "label_qa_file_sha256": sha256_file(evidence / "label_qa_v1.json"),
            "labels": str(labels_path.relative_to(ROOT)),
            "labels_file_sha256": sha256_file(labels_path),
            "config_sha256": sha256_file(CONFIG_PATH),
            "acquisition_code_sha256": sha256_file(Path(__file__).with_name("acquire_q3_validation.py")),
            "build_code_sha256": sha256_file(Path(__file__)),
            "label_code_sha256": code_hash,
            "run_id": run_id,
            "split": split,
            "horizon_days": HORIZON,
            "training_approval": False,
            "scoring_allowed": False,
            "q4_access": False,
            "remote_setup": False,
            "published_at_utc": utc_now(),
        }
        final_manifest["manifest_hash"] = canonical_hash(final_manifest)
        atomic_json(evidence / "q3_complete_manifest_v1.json", final_manifest)
        return result
    finally:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        prior.close()


def independent_label_qa(connection: sqlite3.Connection, prior: sqlite3.Connection, flows: list[dict], output: Path) -> dict:
    failures: list[str] = []
    rows = [dict(row) for row in connection.execute("SELECT * FROM label_flow WHERE run_id='validation_q3_h7_v1' AND split='validation' AND horizon_days=7 ORDER BY decision_date,serial_number")]
    if len(rows) != len(flows):
        failures.append(f"flow row count {len(rows)} != {len(flows)}")
    q3_dates = {str(row[0]) for row in connection.execute("SELECT DISTINCT date FROM daily")}
    if q3_dates != set(EXPECTED_DATES):
        failures.append("Q3 daily date coverage differs from 92-day source")
    status_counts = collections.Counter(str(row["status"]) for row in rows)
    allowed_statuses = {
        "positive_observed",
        "positive_with_gap",
        "negative_observed",
        "end_censored",
        "gap_or_exit_censored",
        "history_insufficient",
        "same_day_failure",
        "post_failure",
    }
    consistency_failures = 0
    for row in rows:
        status = str(row["status"])
        eligible = int(row["eligible"])
        label = row["label"]
        if status not in allowed_statuses or eligible not in (0, 1):
            consistency_failures += 1
            continue
        if status in {"positive_observed", "positive_with_gap"} and (eligible != 1 or label != 1):
            consistency_failures += 1
        elif status == "negative_observed" and (eligible != 1 or label != 0):
            consistency_failures += 1
        elif status in {"end_censored", "gap_or_exit_censored"} and (eligible != 1 or label is not None):
            consistency_failures += 1
        elif status in {"history_insufficient", "same_day_failure", "post_failure"} and (eligible != 0 or label is not None):
            consistency_failures += 1
    if consistency_failures:
        failures.append(f"status/label consistency failures={consistency_failures}")
    event_serials: dict[str, str] = {}
    for row in prior.execute("SELECT serial_number,MIN(date) FROM daily WHERE model=? AND failure=1 GROUP BY serial_number", (MODEL,)):
        event_serials[str(row[0])] = str(row[1])
    for row in connection.execute("SELECT serial_number,MIN(date) FROM daily WHERE model=? AND failure=1 GROUP BY serial_number", (MODEL,)):
        serial = str(row[0]); event_serials.setdefault(serial, str(row[1]))
    event_total = event_opportunity = 0
    for serial, first in event_serials.items():
        first_date = dt.date.fromisoformat(first)
        if not EVENT_START <= first_date <= EVENT_END:
            continue
        event_total += 1
        if any(row["serial_number"] == serial and int(row["eligible"]) for row in rows):
            event_opportunity += 1
    eligible_rows = sum(int(row["eligible"]) for row in rows)
    positive_rows = sum(int(row["label"] == 1) for row in rows)
    negative_rows = sum(int(row["label"] == 0) for row in rows)
    unknown_rows = sum(int(row["eligible"]) and int(row["label"] is None) for row in rows)
    unknown_ratio = unknown_rows / eligible_rows if eligible_rows else 1.0
    if eligible_rows == 0:
        failures.append("Q3 eligible row count is zero")
    elif unknown_ratio > 0.20:
        failures.append(f"Q3 unknown ratio exceeds 20%: {unknown_ratio:.4f}")
    if status_counts["positive_observed"] + status_counts["positive_with_gap"] != positive_rows:
        failures.append("positive status/label mismatch")
    result = {
        "status": "pass" if not failures else "fail",
        "flow_rows": len(rows),
        "status_counts": dict(status_counts),
        "eligible_rows": eligible_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "unknown_rows": unknown_rows,
        "unknown_ratio": unknown_ratio,
        "unknown_ratio_threshold": 0.20,
        "event_total": event_total,
        "event_opportunity": event_opportunity,
        "failures": failures,
        "qa_method": "independent calendar/key/status checks over prior Q1/Q2 plus Q3 daily; does not call classify_device_rows",
    }
    atomic_json(output, result)
    if failures:
        raise BuildStopped("Q3 label QA failed: " + "; ".join(failures))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    config = load_config()
    plan = _load_plan(project_path(args.plan, "plan"))
    receipt = _receipt(project_path(args.receipt, "receipt"))
    database = project_path(args.database, "database")
    evidence = project_path(args.evidence_root, "evidence root")
    panel_result = None
    if not args.labels_only:
        panel_result = build_panel(config, receipt, plan, database, evidence, resume=args.resume)
    if not args.panel_only:
        label_result = build_labels(config, plan, receipt, database, evidence, replace=args.replace)
    else:
        label_result = None
    print(json.dumps({"status": "complete", "panel": panel_result and panel_result["status"], "labels": label_result and label_result["status"], "database": str(database)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BuildStopped, AcquisitionError) as exc:
        raise SystemExit(f"Q3_BUILD_FAILED: {exc}")
