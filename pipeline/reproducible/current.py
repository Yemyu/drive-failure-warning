"""Panel-input prototype for current-model training and alert evaluation.

Raw-source integration, process isolation and full-run acceptance are pending;
see reports/REPRODUCIBLE_PIPELINE_REVIEW.md before a real-data run.
"""

from __future__ import annotations

import csv
from collections import Counter, deque
import datetime as dt
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ..labeling import classify_device_rows
from ..member_io import MemberError, iter_member_rows, load_manifest, manifest_entries
from ..panel import PanelWriter
from ..replay_selection import ceil_budget
from .artifacts import write_manifest
from .cancellation import CancellationRequested, attach_sqlite_progress
from .contract import ContractError, require_panel_contract
from .source_streams import ArchiveSource, SourceError, range_rows
from .runtime_context import code_root, project_root


ROOT = project_root()
CODE_ROOT = code_root()
MODEL = "ST4000DM000"
HORIZON = 7
HISTORY_DAYS = 14
MIN_HISTORY = 12
BUDGET_DENOMINATOR = 1000
COOLDOWN_DAYS = 7
MAX_RSS_BYTES = 3 * 1024 * 1024 * 1024
MAX_OWNED_BYTES = 12 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
STAGE_SECONDS = 3 * 60 * 60
NONZERO_FIELDS = (5, 187, 188, 197, 198)
CURRENT_COLUMNS = (
    "smart_5_current_log1p",
    "smart_5_current_missing",
    "smart_5_current_nonzero",
    "smart_9_current_log1p",
    "smart_9_current_missing",
    "smart_187_current_log1p",
    "smart_187_current_missing",
    "smart_187_current_nonzero",
    "smart_188_current_nonzero",
    "smart_188_current_missing",
    "smart_197_current_log1p",
    "smart_197_current_missing",
    "smart_197_current_nonzero",
    "smart_198_current_log1p",
    "smart_198_current_missing",
    "smart_198_current_nonzero",
)
RAW_FIELDS = (5, 9, 187, 188, 197, 198)
KNOWN_STATUSES = {"positive_observed", "positive_with_gap", "negative_observed"}
HEX64 = set("0123456789abcdef")


class ReproducibleStopped(RuntimeError):
    """A declared provenance, leakage, numerical, or resource stop."""


def _path(value: Path | str, field: str, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ReproducibleStopped(f"{field} escapes project root: {path}")
    if must_exist and not resolved.is_file():
        raise ReproducibleStopped(f"missing {field}: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _snapshot(output: Path) -> dict[str, int]:
    owned = 0
    if output.exists():
        for item in output.rglob("*"):
            if item.is_file():
                try:
                    owned += item.stat().st_size
                except OSError:
                    pass
    return {
        "rss_bytes": _rss_bytes(),
        "owned_bytes": owned,
        "free_bytes": shutil.disk_usage(ROOT).free,
    }


def _guard(started: float, output: Path, label: str) -> dict[str, int | str]:
    snapshot: dict[str, int | str] = {"label": label, "elapsed_seconds": round(time.monotonic() - started, 3), **_snapshot(output)}
    if int(snapshot["rss_bytes"]) > MAX_RSS_BYTES:
        raise ReproducibleStopped(f"RSS limit during {label}: {snapshot}")
    if int(snapshot["owned_bytes"]) > MAX_OWNED_BYTES:
        raise ReproducibleStopped(f"output-size limit during {label}: {snapshot}")
    if int(snapshot["free_bytes"]) < MIN_FREE_BYTES:
        raise ReproducibleStopped(f"free-space floor during {label}: {snapshot}")
    if float(snapshot["elapsed_seconds"]) > STAGE_SECONDS:
        raise ReproducibleStopped(f"stage timeout during {label}: {snapshot}")
    return snapshot


def _open_readonly(path: Path, *, progress_callback: Callable[[], int] | None = None) -> sqlite3.Connection:
    path = _path(path, "SQLite input", must_exist=True)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ReproducibleStopped(f"non-empty SQLite sidecar: {sidecar}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    if progress_callback is not None:
        attach_sqlite_progress(connection, progress_callback)
    return connection


def _registry_snapshot(connection: sqlite3.Connection, label: str) -> tuple[dict[str, str], dict[str, object]]:
    """Read the full-source identity registry in deterministic order."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "serial_model_registry" not in tables:
        raise ReproducibleStopped(f"{label} has no serial_model_registry")
    models: dict[str, str] = {}
    digest = hashlib.sha256()
    previous = None
    for row in connection.execute(
        "SELECT serial_number, model FROM serial_model_registry ORDER BY serial_number"
    ):
        serial, model = str(row[0]), str(row[1])
        if not serial or not model:
            raise ReproducibleStopped(f"{label} identity registry contains an empty value")
        if previous is not None and serial <= previous:
            raise ReproducibleStopped(f"{label} identity registry is not strictly ordered")
        previous = serial
        models[serial] = model
        digest.update(serial.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model.encode("utf-8"))
        digest.update(b"\n")
    if not models:
        raise ReproducibleStopped(f"{label} identity registry is empty")
    return models, {"serials": len(models), "sha256": digest.hexdigest()}


def verify_cross_quarter_identity(
    prior_panel: Path | str,
    q3_panel: Path | str,
    *,
    allow_q3_building: bool = False,
) -> dict:
    """Join full-source registries and stop on any model conflict.

    The Q3 builder calls this while its database is still a private ``.partial``
    file.  The public default accepts completed panels only.
    """
    prior = _path(prior_panel, "prior quarter panel", must_exist=True)
    q3 = _path(q3_panel, "Q3 panel", must_exist=True)
    if prior == q3:
        raise ReproducibleStopped("prior and Q3 panels must be different files")
    connections: list[sqlite3.Connection] = []
    try:
        for path, label in ((prior, "prior quarter panel"), (q3, "Q3 panel")):
            connection = _open_readonly(path)
            connections.append(connection)
            metadata = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            allowed_statuses = {"panel_complete", "complete"}
            if label == "Q3 panel" and allow_q3_building:
                allowed_statuses.add("running")
            if metadata.get("build_status") not in allowed_statuses:
                raise ReproducibleStopped(f"{label} is not a completed panel")
            if metadata.get("serial_model_registry_scope") != "full_source_rows":
                raise ReproducibleStopped(f"{label} identity registry is not full-source")
        prior_models, prior_facts = _registry_snapshot(connections[0], "prior quarter panel")
        q3_models, q3_facts = _registry_snapshot(connections[1], "Q3 panel")
        overlap = 0
        for serial in sorted(set(prior_models).intersection(q3_models)):
            overlap += 1
            if prior_models[serial] != q3_models[serial]:
                raise ReproducibleStopped(
                    f"cross-quarter serial/model identity conflict for {serial}: "
                    f"prior={prior_models[serial]!r}, q3={q3_models[serial]!r}"
                )
        return {
            "status": "pass",
            "prior_panel": str(prior.relative_to(ROOT)),
            "q3_panel": str(q3.relative_to(ROOT)),
            "prior_panel_sha256": _sha256_file(prior),
            "q3_panel_sha256": _sha256_file(q3),
            "prior_registry_serials": prior_facts["serials"],
            "q3_registry_serials": q3_facts["serials"],
            "overlap_serials": overlap,
            "conflict_count": 0,
            "prior_registry_sha256": prior_facts["sha256"],
            "q3_registry_sha256": q3_facts["sha256"],
        }
    finally:
        for connection in connections:
            connection.close()


def verify_source_manifest(manifest_path: Path | str, *, verify_members: bool = False) -> dict:
    """Validate the retained member inventory without changing any file."""
    manifest_path = _path(manifest_path, "source manifest", must_exist=True)
    try:
        manifest = load_manifest(manifest_path)
        if not isinstance(manifest, Mapping):
            raise ReproducibleStopped("source manifest must be an object")
        raw_members = manifest.get("members")
        if not isinstance(raw_members, list):
            raise ReproducibleStopped("source manifest members must be a list")
        for raw_entry in raw_members:
            raw_name = raw_entry.get("name") if isinstance(raw_entry, Mapping) else None
            raw_date = raw_entry.get("date") if isinstance(raw_entry, Mapping) else None
            if raw_date is not None and raw_name and not str(raw_name).endswith(f"/{raw_date}.csv"):
                raise ReproducibleStopped(f"member name/date mismatch: {raw_name}")
        entries = manifest_entries(manifest)
        declared_count = manifest.get("member_count")
        if declared_count is not None and int(declared_count) != len(entries):
            raise ReproducibleStopped(f"source manifest member count mismatch: {len(entries)} != {declared_count}")
    except (KeyError, TypeError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise ReproducibleStopped(f"invalid source manifest: {manifest_path}") from exc
    checked = []
    for entry in entries:
        _validate_source_entry(entry)
        local = _path(entry.get("local", ""), f"member {entry.get('name')}", must_exist=True)
        actual = {"bytes": local.stat().st_size, "sha256": _sha256_file(local)}
        if actual["bytes"] != int(entry["range_bytes"]):
            raise ReproducibleStopped(f"member byte count mismatch: {entry['name']}")
        if actual["sha256"] != entry["sha256"]:
            raise ReproducibleStopped(f"member SHA256 mismatch: {entry['name']}")
        row_count = None
        if verify_members:
            try:
                with range_rows(entry, ROOT) as (_date, _digest, _columns, rows, _diagnostics):
                    row_count = sum(1 for _ in rows)
            except (MemberError, SourceError) as exc:
                raise ReproducibleStopped(f"member validation failed for {entry['name']}: {exc}") from exc
            expected_counts = entry.get("expected_counts") or {}
            expected_rows = expected_counts.get("source_rows")
            if expected_rows is not None and int(expected_rows) != row_count:
                raise ReproducibleStopped(f"source row count mismatch for {entry['name']}: {row_count} != {expected_rows}")
            expected_columns = expected_counts.get("schema_columns")
            if expected_columns is not None and int(expected_columns) != len(_columns):
                raise ReproducibleStopped(f"schema column count mismatch for {entry['name']}: {len(_columns)} != {expected_columns}")
            expected_schema = expected_counts.get("schema_sha256")
            schema_hash = hashlib.sha256(json.dumps(_columns, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if expected_schema is not None and schema_hash != expected_schema:
                raise ReproducibleStopped(f"schema hash mismatch for {entry['name']}")
        checked.append({"name": entry["name"], "local": str(local.relative_to(ROOT)), "file": actual, "rows_scanned": row_count})
    return {
        "status": "pass",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256_file(manifest_path),
        "member_count": len(checked),
        "members": checked,
    }


def _validate_source_entry(entry: Mapping[str, object]) -> None:
    """Require the immutable facts needed to bind a retained range member."""
    required = ("name", "local", "sha256", "compressed", "size", "crc", "range_bytes", "expected_counts")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ReproducibleStopped(f"source member entry missing fields: {missing}")
    sha = str(entry["sha256"])
    if len(sha) != 64 or sha != sha.lower() or set(sha) - HEX64:
        raise ReproducibleStopped(f"invalid source member SHA256: {entry['name']}")
    if not str(entry["name"]).endswith(f"/{entry['date']}.csv"):
        raise ReproducibleStopped(f"member name/date mismatch: {entry['name']}")
    for key in ("compressed", "size", "crc", "range_bytes"):
        try:
            value = int(entry[key])
        except (TypeError, ValueError) as exc:
            raise ReproducibleStopped(f"invalid source member {key}: {entry['name']}") from exc
        if value < 0:
            raise ReproducibleStopped(f"negative source member {key}: {entry['name']}")
    counts = entry["expected_counts"]
    if not isinstance(counts, Mapping):
        raise ReproducibleStopped(f"invalid expected_counts: {entry['name']}")
    for key in ("source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256"):
        if key not in counts:
            raise ReproducibleStopped(f"expected_counts missing {key}: {entry['name']}")
    for key in ("source_rows", "selected_rows", "failure_rows", "schema_columns"):
        try:
            value = int(counts[key])
        except (TypeError, ValueError) as exc:
            raise ReproducibleStopped(f"invalid expected count {key}: {entry['name']}") from exc
        if value < 0:
            raise ReproducibleStopped(f"negative expected count {key}: {entry['name']}")
    if int(counts["selected_rows"]) > int(counts["source_rows"]):
        raise ReproducibleStopped(f"selected rows exceed source rows: {entry['name']}")
    schema = str(counts["schema_sha256"])
    if len(schema) != 64 or schema != schema.lower() or set(schema) - HEX64:
        raise ReproducibleStopped(f"invalid schema SHA256: {entry['name']}")


def build_panel_from_manifest(
    manifest_path: Path | str,
    output_db: Path | str,
    evidence_path: Path | str,
    *,
    model: str = MODEL,
) -> dict:
    """Build a new panel from retained Q1/Q2 members using one save point/day."""
    output_db = _path(output_db, "panel output")
    evidence_path = _path(evidence_path, "panel evidence")
    if output_db == evidence_path:
        raise ReproducibleStopped("panel output and evidence must be different paths")
    manifest_path = _path(manifest_path, "source manifest", must_exist=True)
    if output_db.exists():
        raise ReproducibleStopped(f"refusing to overwrite panel output: {output_db}")
    if evidence_path.exists():
        raise ReproducibleStopped(f"refusing to overwrite panel evidence: {evidence_path}")
    source = verify_source_manifest(manifest_path)
    manifest = load_manifest(manifest_path)
    entries = manifest_entries(manifest)
    partial = output_db.with_name(output_db.name + ".partial")
    if partial.exists():
        raise ReproducibleStopped(f"failed panel partial already exists: {partial}")
    writer = PanelWriter(partial, reset=False)
    progress = {"status": "running", "manifest_sha256": source["manifest_sha256"], "members": []}
    try:
        writer.set_metadata(
            {
                "reproducible_source_manifest_sha256": source["manifest_sha256"],
                "selected_model": model,
                "training_approval": "false",
                "q4_access": "false",
                "remote_setup": "false",
                "build_status": "running",
            }
        )
        for entry in entries:
            with range_rows(entry, ROOT) as source_data:
                facts = writer.append_member(entry, ROOT, model, source_data=source_data,
                                             expected_counts=entry["expected_counts"], strict_smart=True)
            progress["members"].append(facts)
            _atomic_json(evidence_path, progress)
        writer.set_metadata({"build_status": "panel_complete", "reproducible_panel_manifest_sha256": source["manifest_sha256"]})
        summary = writer.summary()
        writer.close()
        os.replace(partial, output_db)
        result = {"status": "complete", "database": str(output_db.relative_to(ROOT)), "source": source, "summary": summary}
        _atomic_json(evidence_path, result)
        return result
    except BaseException as exc:
        progress.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _atomic_json(evidence_path, progress)
        try:
            writer.close()
        except Exception:
            pass
        raise


def build_archive_panel(
    receipt_path: Path | str,
    output_dir: Path | str,
    *,
    prior_panel: Path | str | None = None,
) -> dict:
    """Build Q3 daily records and optionally verify the prior-quarter identity."""
    receipt_path = _path(receipt_path, "archive receipt", must_exist=True)
    output_dir = _path(output_dir, "archive panel output")
    prior_path = _path(prior_panel, "prior quarter panel", must_exist=True) if prior_panel is not None else None
    if output_dir.exists():
        raise ReproducibleStopped("archive panel output already exists")
    if prior_path is not None and prior_path == output_dir:
        raise ReproducibleStopped("prior quarter panel cannot be the output directory")
    receipt_sha = _sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "verified" or receipt.get("receipt_hash") != _canonical({k: v for k, v in receipt.items() if k != "receipt_hash"}):
        raise ReproducibleStopped("archive receipt status/hash mismatch")
    if receipt.get("q4_access") is not False or receipt.get("remote_setup") is not False:
        raise ReproducibleStopped("archive receipt scope differs")
    if receipt.get("schema_columns") != 193 or receipt.get("schema_sha256") != "e9e64775931750527fe5623674b4462580c660729115aed9ade641c4724a7d4d":
        raise ReproducibleStopped("archive schema is not the declared Q3 schema")
    dates = [member["date"] for member in receipt["members"]]
    if dates != sorted(set(dates)) or dates != receipt["dates"] or not dates:
        raise ReproducibleStopped("archive dates are duplicated, unordered or inconsistent")
    for member in receipt["members"]:
        date = dt.date.fromisoformat(member["date"])
        if not dt.date(2023, 7, 1) <= date <= dt.date(2023, 9, 30) or member["name"] != f"data_Q3_2023/{date}.csv":
            raise ReproducibleStopped("archive date/member is outside Q3")
    output_dir.mkdir(parents=True)
    partial = output_dir / "panel.sqlite.partial"
    writer = None
    try:
        declarations = {"data_Q3_2023": {"count": receipt["schema_columns"], "sha256": receipt["schema_sha256"]}}
        writer = PanelWriter(partial, schema_declarations=declarations)
        writer.set_metadata({"build_status": "running", "source_receipt_sha256": receipt_sha,
                             "selected_model": MODEL, "cross_quarter_identity": "pending"})
        facts = []
        with ArchiveSource(receipt, ROOT) as source:
            for member in receipt["members"]:
                with source.rows(member) as (entry, data):
                    facts.append(writer.append_member(entry, ROOT, MODEL, source_data=data,
                                                      expected_counts=entry["expected_counts"], strict_smart=True))
        if _sha256_file(receipt_path) != receipt_sha:
            raise ReproducibleStopped("archive receipt changed during build")
        identity = {"status": "pending", "reason": "prior quarter panel was not supplied"}
        summary = writer.summary()
        writer.close()
        writer = None
        if prior_path is not None:
            identity = verify_cross_quarter_identity(prior_path, partial, allow_q3_building=True)
        # A connection context commits but does not close the connection.
        # Finish WAL writes before moving the standalone database file.
        connection = sqlite3.connect(partial)
        try:
            if prior_path is not None:
                connection.executemany(
                    "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                    [
                        ("cross_quarter_identity", "verified"),
                        ("cross_quarter_identity_prior_panel_sha256", identity["prior_panel_sha256"]),
                        ("cross_quarter_identity_overlap_serials", str(identity["overlap_serials"])),
                    ],
                )
            connection.execute("UPDATE metadata SET value='panel_complete' WHERE key='build_status'")
            connection.commit()
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if tuple(checkpoint) != (0, 0, 0):
                raise ReproducibleStopped(f"Q3 publication checkpoint incomplete: {checkpoint}")
        finally:
            connection.close()
        # Refuse to publish a main file that still depends on a sidecar.
        with_read = _open_readonly(partial)
        with_read.close()
        final = output_dir / "panel.sqlite"
        os.rename(partial, final)
        check = _open_readonly(final)
        try:
            metadata = dict(check.execute("SELECT key,value FROM metadata"))
            expected_identity = "verified" if prior_path is not None else "pending"
            if (metadata.get("build_status") != "panel_complete"
                    or metadata.get("serial_model_registry_scope") != "full_source_rows"
                    or metadata.get("cross_quarter_identity") != expected_identity):
                raise ReproducibleStopped("final Q3 metadata does not satisfy publication contract")
        finally:
            check.close()
        if prior_path is not None:
            identity = verify_cross_quarter_identity(prior_path, final)
        summary["database"] = str(final.relative_to(ROOT))
        result = {"status": "complete", "scope": "q3_raw_panel", "specification_acceptance": "incomplete",
                  "receipt_sha256": receipt_sha, "archive_sha256": receipt["archive_sha256"],
                  "cross_quarter_identity": identity, "panel_sha256": _sha256_file(final), "members": facts, "summary": summary,
                  "code_sha256": {str(p.relative_to(ROOT)): _sha256_file(p) for p in
                      (Path(__file__), CODE_ROOT / "pipeline/reproducible/source_streams.py", CODE_ROOT / "pipeline/panel.py")}}
        published = write_manifest(
            output_dir / "panel_manifest.json",
            result,
            artifacts={"panel": final, "source_receipt": receipt_path},
        )
        return published
    except BaseException as exc:
        if writer is not None:
            writer.close()
        _atomic_json(output_dir / "failure.json", {"status": "failed", "error": str(exc), "scope": "q3_raw_panel"})
        raise


def _panel_row_stream(
    paths: Sequence[Path | str],
    *,
    dataset_end: dt.date,
    progress_callback: Callable[[], int] | None = None,
):
    """Yield panel rows in serial/date order without materializing a quarter."""
    streams = []
    connections = []
    seen_paths: set[Path] = set()
    try:
        for raw_path in paths:
            path = _path(raw_path, "panel input", must_exist=True)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            connection = _open_readonly(path, progress_callback=progress_callback)
            connections.append(connection)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(daily)")}
            required = {"date", "serial_number", "model", "capacity_bytes", "failure", *(f"smart_{field}_raw" for field in RAW_FIELDS)}
            if not required.issubset(columns):
                raise ReproducibleStopped(f"panel missing required daily columns: {path}")
            query = (
                "SELECT date,serial_number,model,capacity_bytes,failure," +
                ",".join(f"smart_{field}_raw" for field in RAW_FIELDS) +
                " FROM daily WHERE date<=? ORDER BY serial_number,date"
            )

            def rows_for_connection(conn: sqlite3.Connection):
                try:
                    for row in conn.execute(query, (dataset_end.isoformat(),)):
                        if progress_callback is not None and progress_callback():
                            raise CancellationRequested("panel query cancelled")
                        yield (str(row["serial_number"]), str(row["date"]), dict(row))
                finally:
                    conn.close()

            streams.append(rows_for_connection(connection))
        yield from (item[2] for item in heapq.merge(*streams, key=lambda item: (item[0], item[1])))
    finally:
        for stream in streams:
            stream.close()
        for connection in connections:
            connection.close()


def _serial_groups(
    paths: Sequence[Path | str],
    *,
    dataset_end: dt.date,
    progress_callback: Callable[[], int] | None = None,
):
    stream = _panel_row_stream(paths, dataset_end=dataset_end, progress_callback=progress_callback)
    for serial, group in itertools.groupby(stream, key=lambda item: str(item["serial_number"])):
        by_date: dict[str, dict] = {}
        for item in group:
            date = str(item["date"])
            prior = by_date.get(date)
            if prior is not None and prior != item:
                raise ReproducibleStopped(f"conflicting duplicate panel key: {date}/{serial}")
            by_date[date] = item
        yield serial, [by_date[key] for key in sorted(by_date)]


def _panel_rows(paths: Sequence[Path | str], *, dataset_end: dt.date) -> dict[str, list[dict]]:
    """Compatibility helper used only by small diagnostics."""
    return {serial: rows for serial, rows in _serial_groups(paths, dataset_end=dataset_end)}


def build_labels(
    panel_paths: Sequence[Path | str],
    output_db: Path | str,
    *,
    start: str,
    end: str,
    dataset_end: str,
    run_id: str,
    progress_callback: Callable[[], int] | None = None,
) -> dict:
    """Build an isolated label database from one or more panel inputs."""
    output_db = _path(output_db, "label output")
    if output_db.exists():
        raise ReproducibleStopped(f"refusing to overwrite label output: {output_db}")
    lo, hi, cutoff = dt.date.fromisoformat(start), dt.date.fromisoformat(end), dt.date.fromisoformat(dataset_end)
    if hi > cutoff or lo > hi:
        raise ReproducibleStopped("invalid label date range")
    partial = output_db.with_name(output_db.name + ".partial")
    if partial.exists():
        raise ReproducibleStopped(f"failed label partial already exists: {partial}")
    code_hash = _sha256_file(Path(__file__))
    source_hashes = [_sha256_file(_path(path, "panel input", must_exist=True)) for path in panel_paths]
    config = {"run_id": run_id, "start": start, "end": end, "dataset_end": dataset_end, "horizon": HORIZON, "history_days": HISTORY_DAYS, "min_history": MIN_HISTORY, "source_panel_sha256": source_hashes}
    connection = sqlite3.connect(partial)
    connection.row_factory = sqlite3.Row
    if progress_callback is not None:
        attach_sqlite_progress(connection, progress_callback)
    try:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE label_flow(run_id TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, model TEXT NOT NULL, capacity_bytes INTEGER, first_failure_date TEXT, label INTEGER, status TEXT NOT NULL, eligible INTEGER NOT NULL, history_observations INTEGER NOT NULL, future_observations INTEGER NOT NULL, PRIMARY KEY(run_id,decision_date,serial_number));"
            "CREATE INDEX label_flow_date ON label_flow(decision_date);"
        )
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [("status", "running"), ("config_sha256", _canonical(config)), ("code_sha256", code_hash), ("source_panel_sha256", json.dumps(source_hashes)), ("training_approval", "false")])
        insert = "INSERT INTO label_flow VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        count = 0
        status_counts: dict[str, int] = {}
        for serial, rows in _serial_groups(panel_paths, dataset_end=cutoff, progress_callback=progress_callback):
            for item in classify_device_rows(rows, start=start, end=end, dataset_end=dataset_end, horizon_days=HORIZON, history_days=HISTORY_DAYS, min_history=MIN_HISTORY):
                connection.execute(insert, (run_id, item["decision_date"], serial, item["model"], item["capacity_bytes"], item["first_failure_date"], item["label"], item["status"], item["eligible"], item["history_observations"], item["future_observations"]))
                count += 1
                status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?,?)", [("status", "complete"), ("flow_rows", str(count)), ("status_counts", json.dumps(status_counts, sort_keys=True)), ("completed_at_utc", dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat())])
        connection.commit()
        connection.close()
        os.replace(partial, output_db)
        return {"status": "complete", "database": str(output_db.relative_to(ROOT)), "flow_rows": count, "status_counts": status_counts, "config": config}
    except BaseException:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        connection.close()
        raise


def _current_feature(row: Mapping[str, object]) -> dict[str, object]:
    expected = {"date", "serial_number", "model", *(f"smart_{field}_raw" for field in RAW_FIELDS)}
    if set(row.keys()) != expected:
        raise ReproducibleStopped("current feature input must contain only current raw fields and identity")
    values: dict[str, object] = {}
    for field in RAW_FIELDS:
        raw = row[f"smart_{field}_raw"]
        if raw is not None:
            if type(raw) is not int or raw < 0:
                raise ReproducibleStopped(f"invalid integer SMART {field} at {row['date']}/{row['serial_number']}")
        missing = int(raw is None)
        prefix = f"smart_{field}"
        values[f"{prefix}_current_missing"] = missing
        if field == 188:
            values[f"{prefix}_current_nonzero"] = None if missing else int(raw > 0)
        else:
            values[f"{prefix}_current_log1p"] = None if missing else math.log1p(raw)
            if field != 9:
                values[f"{prefix}_current_nonzero"] = None if missing else int(raw > 0)
    values["decision_date"] = row["date"]
    values["serial_number"] = row["serial_number"]
    values["model"] = row["model"]
    values["smart_nonzero_signal_count"] = sum(
        int(row[f"smart_{field}_raw"] is not None and int(row[f"smart_{field}_raw"]) > 0) for field in NONZERO_FIELDS
    )
    values["tie_break_sha256"] = hashlib.sha256(f"drive-v1|20260912|{row['date']}|{row['serial_number']}".encode()).hexdigest()
    return values


def _eligible_current_rows(
    panel_paths,
    start: dt.date,
    end: dt.date,
    *,
    progress_callback: Callable[[], int] | None = None,
    calendar_rows: list[dict[str, object]] | None = None,
):
    """Read one calendar day at a time; eligibility uses observed history only."""
    paths = list(dict.fromkeys(_path(p, "feature panel", must_exist=True) for p in panel_paths))
    connections = []
    try:
        for path in paths:
            connections.append(_open_readonly(path, progress_callback=progress_callback))
        first = start - dt.timedelta(days=HISTORY_DAYS - 1)
        failed = set()
        for connection in connections:
            failed.update(row[0] for row in connection.execute(
                "SELECT DISTINCT serial_number FROM daily WHERE failure=1 AND date<?", (first.isoformat(),)))
        window = deque()
        counts = Counter()
        selected = ["date", "serial_number", "model", "failure"] + [f"smart_{field}_raw" for field in RAW_FIELDS]
        day = first
        while day <= end:
            if progress_callback is not None and progress_callback():
                raise CancellationRequested("eligibility query cancelled")
            current = {}
            for connection in connections:
                for row in connection.execute("SELECT " + ",".join(selected) + " FROM daily WHERE date=? ORDER BY serial_number", (day.isoformat(),)):
                    if progress_callback is not None and progress_callback():
                        raise CancellationRequested("eligibility query cancelled")
                    item = dict(row)
                    serial = item["serial_number"]
                    if serial in current and current[serial] != item:
                        raise ReproducibleStopped(f"conflicting device day: {day}/{serial}")
                    if item["failure"] not in (0, 1):
                        raise ReproducibleStopped("invalid failure flag")
                    current[serial] = item
            observed = {s for s, row in current.items() if row["model"] == MODEL}
            counts.update(observed)
            window.append(observed)
            if len(window) > HISTORY_DAYS:
                counts.subtract(window.popleft())
                counts += Counter()  # Discard expired zero counts.
            failed.update(s for s, row in current.items() if row["failure"] == 1)
            history_ready = sum(1 for serial in observed if counts[serial] >= MIN_HISTORY)
            eligible_serials = [serial for serial in sorted(observed) if serial not in failed and counts[serial] >= MIN_HISTORY]
            if day >= start:
                if calendar_rows is not None:
                    calendar_rows.append({
                        "date": day.isoformat(),
                        "observed_rows": len(current),
                        "model_rows": len(observed),
                        "failed_model_rows": sum(serial in failed for serial in observed),
                        "history_ready_rows": history_ready,
                        "eligible_rows": len(eligible_serials),
                    })
                for serial in eligible_serials:
                    yield {key: value for key, value in current[serial].items() if key != "failure"}
            day += dt.timedelta(days=1)
    finally:
        for connection in connections:
            connection.close()


def build_current_features(
    panel_db: Path | str,
    output_db: Path | str,
    *,
    start: str,
    end: str,
    dataset_end: str,
    history_panels: Sequence[Path | str] = (),
    progress_callback: Callable[[], int] | None = None,
) -> dict:
    """Create only as-of current features; labels are never read here."""
    panel_db = _path(panel_db, "feature panel", must_exist=True)
    source_panels = list(dict.fromkeys(_path(path, "feature panel", must_exist=True) for path in [*history_panels, panel_db]))
    output_db = _path(output_db, "feature output")
    if output_db.exists():
        raise ReproducibleStopped(f"refusing to overwrite feature output: {output_db}")
    lo, hi, cutoff = dt.date.fromisoformat(start), dt.date.fromisoformat(end), dt.date.fromisoformat(dataset_end)
    if lo > hi or hi > cutoff:
        raise ReproducibleStopped("invalid feature date range")
    partial = output_db.with_name(output_db.name + ".partial")
    if partial.exists():
        raise ReproducibleStopped(f"failed feature partial already exists: {partial}")
    source_hashes = [_sha256_file(path) for path in source_panels]
    calendar_rows: list[dict[str, object]] = []
    out = sqlite3.connect(partial)
    try:
        columns = ["decision_date TEXT NOT NULL", "serial_number TEXT NOT NULL", "model TEXT NOT NULL"]
        for name in CURRENT_COLUMNS:
            columns.append(f'"{name}" REAL')
        columns.extend(["smart_nonzero_signal_count INTEGER NOT NULL", "tie_break_sha256 TEXT NOT NULL", "PRIMARY KEY(decision_date,serial_number)"])
        out.execute("CREATE TABLE feature_rows(" + ",".join(columns) + ")")
        out.execute(
            "CREATE TABLE qualification_calendar(" 
            "decision_date TEXT PRIMARY KEY, observed_rows INTEGER NOT NULL, model_rows INTEGER NOT NULL, "
            "failed_model_rows INTEGER NOT NULL, history_ready_rows INTEGER NOT NULL, eligible_rows INTEGER NOT NULL)"
        )
        out.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        if progress_callback is not None:
            attach_sqlite_progress(out, progress_callback)
        out.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("status", "running"),
            ("source_panel_sha256", json.dumps(source_hashes)),
            ("feature_columns", json.dumps(CURRENT_COLUMNS)),
        ])
        insert_columns = ["decision_date", "serial_number", "model", *CURRENT_COLUMNS, "smart_nonzero_signal_count", "tie_break_sha256"]
        insert_sql = "INSERT INTO feature_rows(" + ",".join(insert_columns) + ") VALUES (" + ",".join("?" for _ in insert_columns) + ")"
        count = 0
        for row in _eligible_current_rows(
            source_panels, lo, hi,
            progress_callback=progress_callback,
            calendar_rows=calendar_rows,
        ):
            if progress_callback is not None and progress_callback():
                raise CancellationRequested("feature write cancelled")
            values = _current_feature(row)
            out.execute(insert_sql, tuple(values.get(name) for name in insert_columns))
            count += 1
        expected_days = (hi - lo).days + 1
        if len(calendar_rows) != expected_days:
            raise ReproducibleStopped(f"qualification calendar is incomplete: {len(calendar_rows)} != {expected_days}")
        out.executemany(
            "INSERT INTO qualification_calendar VALUES (?,?,?,?,?,?)",
            [(
                row["date"], row["observed_rows"], row["model_rows"], row["failed_model_rows"],
                row["history_ready_rows"], row["eligible_rows"],
            ) for row in calendar_rows],
        )
        out.executemany("INSERT OR REPLACE INTO metadata VALUES (?,?)", [
            ("status", "complete"),
            ("feature_rows", str(count)),
            ("calendar_days", str(len(calendar_rows))),
            ("calendar_history_start", (lo - dt.timedelta(days=HISTORY_DAYS - 1)).isoformat()),
            ("calendar_decision_start", lo.isoformat()),
            ("calendar_decision_end", hi.isoformat()),
            ("feature_code_sha256", _sha256_file(Path(__file__))),
        ])
        out.commit()
        out.close()
        os.replace(partial, output_db)
        return {
            "status": "complete",
            "database": str(output_db.relative_to(ROOT)),
            "feature_rows": count,
            "columns": list(CURRENT_COLUMNS),
            "source_panel_sha256": source_hashes,
            "qualification_calendar": {
                "history_start": (lo - dt.timedelta(days=HISTORY_DAYS - 1)).isoformat(),
                "decision_start": lo.isoformat(),
                "decision_end": hi.isoformat(),
                "days": calendar_rows,
            },
        }
    except BaseException:
        try:
            out.rollback()
        except sqlite3.Error:
            pass
        out.close()
        raise


def _row_value(row: Mapping[str, object], name: str) -> float:
    value = row[name]
    if value is None:
        return float("nan")
    number = float(value)
    if not math.isfinite(number):
        raise ReproducibleStopped(f"non-finite feature value: {name}")
    return number


def _weighted_preprocess(raw: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(raw)
    total = float(weights.sum())
    if raw.ndim != 2 or len(raw) != len(weights) or total <= 0:
        raise ReproducibleStopped("invalid training matrix or weights")
    means = np.zeros(raw.shape[1], dtype=np.float64)
    observed = valid.sum(axis=0).astype(int)
    for j in range(raw.shape[1]):
        if observed[j]:
            means[j] = np.sum(weights[valid[:, j]] * raw[valid[:, j], j]) / np.sum(weights[valid[:, j]])
    filled = np.where(valid, raw, means)
    center = np.sum(filled * weights[:, None], axis=0) / total
    variance = np.sum(((filled - center) ** 2) * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    transformed = (filled - center) / scale
    if not np.isfinite(transformed).all():
        raise ReproducibleStopped("preprocessing produced non-finite values")
    return transformed, {"imputation_mean": means.tolist(), "standardization_mean": center.tolist(), "standardization_scale": scale.tolist(), "observed_count": observed.tolist(), "weighted_total": total}


def _apply_preprocess(raw: np.ndarray, params: Mapping[str, Sequence[float]]) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    means = np.asarray(params["imputation_mean"], dtype=np.float64)
    center = np.asarray(params["standardization_mean"], dtype=np.float64)
    scale = np.asarray(params["standardization_scale"], dtype=np.float64)
    result = (np.where(np.isfinite(values), values, means) - center) / scale
    if not np.isfinite(result).all():
        raise ReproducibleStopped("scoring preprocessing produced non-finite values")
    return result


def _sampling_hash(date_text: str, serial: str) -> str:
    return hashlib.sha256(f"fit-v1|20260913|{date_text}|{serial}".encode()).hexdigest()


def _prepare_training_arrays(
    feature_db: Path,
    label_db: Path,
    run_id: str,
    *,
    progress_callback: Callable[[], int] | None = None,
) -> dict[str, object]:
    """Prepare the sampled training matrix without fitting a model.

    Keeping preparation separate from fitting lets the worker publish and
    independently re-read an immutable training package before any estimator
    sees the matrix.
    """
    features = (
        _open_readonly(feature_db)
        if progress_callback is None
        else _open_readonly(feature_db, progress_callback=progress_callback)
    )
    labels = _open_readonly(label_db, progress_callback=progress_callback)
    try:
        feature_columns = set(row[1] for row in features.execute("PRAGMA table_info(feature_rows)"))
        if not set(CURRENT_COLUMNS).issubset(feature_columns):
            raise ReproducibleStopped("current feature whitelist is not present")
        matrix: list[list[float]] = []
        y: list[int] = []
        weights: list[float] = []
        sample_keys: list[tuple[str, str]] = []
        sample_stats = []
        grouped = itertools.groupby(features.execute("SELECT * FROM feature_rows ORDER BY decision_date,serial_number"), key=lambda row: str(row["decision_date"]))
        any_rows = False
        for date_text, date_iter in grouped:
            if progress_callback is not None and progress_callback():
                raise CancellationRequested("training query cancelled")
            any_rows = True
            date_rows = list(date_iter)
            label_by_serial = {str(row["serial_number"]): row for row in labels.execute("SELECT * FROM label_flow WHERE run_id=? AND decision_date=? AND eligible=1 ORDER BY serial_number", (run_id, date_text))}
            if {str(row["serial_number"]) for row in date_rows} != set(label_by_serial):
                raise ReproducibleStopped(f"feature/label eligibility differs on {date_text}")
            negatives = [row for row in date_rows if label_by_serial.get(str(row["serial_number"])) is not None and label_by_serial[str(row["serial_number"])] ["status"] == "negative_observed"]
            positives = [row for row in date_rows if label_by_serial.get(str(row["serial_number"])) is not None and str(label_by_serial[str(row["serial_number"])]["status"]).startswith("positive")]
            ranked = sorted(((_sampling_hash(date_text, str(row["serial_number"])), str(row["serial_number"]), row) for row in negatives), key=lambda item: (item[0], item[1]))
            sample_count = min(len(ranked), max(1, len(ranked) // 20)) if ranked else 0
            selected_negatives = {serial for _digest, serial, _row in ranked[:sample_count]}
            inverse = len(ranked) / sample_count if sample_count else 0.0
            for row in date_rows:
                if progress_callback is not None and progress_callback():
                    raise CancellationRequested("training query cancelled")
                label = label_by_serial.get(str(row["serial_number"]))
                if label is None or str(label["status"]) not in KNOWN_STATUSES:
                    continue
                target = int(label["label"])
                if target == 0 and str(row["serial_number"]) not in selected_negatives:
                    continue
                matrix.append([_row_value(row, name) for name in CURRENT_COLUMNS])
                y.append(target)
                weights.append(1.0 if target == 1 else inverse)
                sample_keys.append((date_text, str(row["serial_number"])))
            sample_stats.append({"date": date_text, "eligible": len(date_rows), "positive": len(positives), "negative": len(negatives), "sampled_negative": sample_count, "inverse_weight": inverse})
        if not any_rows:
            raise ReproducibleStopped("no feature rows for fitting")
        raw = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(y, dtype=np.int8)
        raw_weights = np.asarray(weights, dtype=np.float64)
        if len(set(target.tolist())) != 2:
            raise ReproducibleStopped("training sample must contain both classes")
        raw_weights /= raw_weights.mean()
        transformed, prep = _weighted_preprocess(raw, raw_weights)
        return {
            "columns": list(CURRENT_COLUMNS),
            "raw": raw,
            "labels": target,
            "weights": raw_weights,
            "keys": sample_keys,
            "transformed": transformed,
            "preprocessing": prep,
            "sampling": sample_stats,
            "run_id": run_id,
        }
    finally:
        features.close()
        labels.close()


def _fit_prepared(
    prepared: Mapping[str, object],
    *,
    progress_callback: Callable[[], int] | None = None,
) -> tuple[dict, dict]:
    """Fit current_lr on a matrix that has already passed preparation checks."""
    if progress_callback is not None and progress_callback():
        raise CancellationRequested("training fit cancelled")
    columns = list(prepared.get("columns", CURRENT_COLUMNS))
    if columns != list(CURRENT_COLUMNS):
        raise ReproducibleStopped("prepared training columns differ from current whitelist")
    transformed = np.asarray(prepared["transformed"], dtype=np.float64)
    target = np.asarray(prepared["labels"], dtype=np.int8)
    raw_weights = np.asarray(prepared["weights"], dtype=np.float64)
    if transformed.ndim != 2 or transformed.shape[1] != len(CURRENT_COLUMNS):
        raise ReproducibleStopped("prepared transformed matrix has the wrong shape")
    if len(target) != len(transformed) or len(raw_weights) != len(transformed):
        raise ReproducibleStopped("prepared arrays have inconsistent row counts")
    if len(set(target.tolist())) != 2 or not np.isfinite(transformed).all() or not np.isfinite(raw_weights).all() or (raw_weights <= 0).any():
        raise ReproducibleStopped("prepared training arrays are invalid")
    model = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", tol=1e-6, max_iter=1000,
        fit_intercept=True, class_weight=None, random_state=20260913,
    )
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(transformed, target, sample_weight=raw_weights)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise ReproducibleStopped("; ".join(convergence))
    prep = dict(prepared["preprocessing"])
    sample_stats = list(prepared.get("sampling", []))
    payload = {
        "model": "current_lr", "fit_mode": "fitted", "columns": list(CURRENT_COLUMNS),
        "coef": model.coef_[0].astype(float).tolist(), "intercept": float(model.intercept_[0]),
        "n_iter": int(model.n_iter_[0]), "preprocessing": prep,
        "params": {
            "penalty": "l2", "solver": "lbfgs", "C": 1.0, "tol": 1e-6,
            "max_iter": 1000, "fit_intercept": True, "class_weight": None,
            "random_state": 20260913,
        },
        "training_rows": len(target), "positive_rows": int(target.sum()),
        "negative_rows": int((target == 0).sum()), "sampling": sample_stats,
    }
    arrays = {
        "raw": np.asarray(prepared["raw"], dtype=np.float64),
        "transformed": transformed,
        "labels": target,
        "weights": raw_weights,
        "keys": list(prepared["keys"]),
    }
    return payload, arrays


def _train_model(
    feature_db: Path,
    label_db: Path,
    run_id: str,
    *,
    progress_callback: Callable[[], int] | None = None,
) -> tuple[dict, dict]:
    """Legacy in-process helper retained for the synthetic prototype."""
    prepared = _prepare_training_arrays(
        feature_db, label_db, run_id, progress_callback=progress_callback,
    )
    return _fit_prepared(prepared, progress_callback=progress_callback)


def _rule_and_model_scores(
    feature_db: Path,
    payload: Mapping[str, object],
    *,
    progress_callback: Callable[[], int] | None = None,
):
    features = (
        _open_readonly(feature_db)
        if progress_callback is None
        else _open_readonly(feature_db, progress_callback=progress_callback)
    )
    try:
        prep = payload["preprocessing"]
        coef = np.asarray(payload["coef"], dtype=float)
        grouped = itertools.groupby(features.execute("SELECT * FROM feature_rows ORDER BY decision_date,serial_number"), key=lambda row: str(row["decision_date"]))
        for date_text, date_iter in grouped:
            if progress_callback is not None and progress_callback():
                raise CancellationRequested("score query cancelled")
            for row in date_iter:
                if progress_callback is not None and progress_callback():
                    raise CancellationRequested("score query cancelled")
                serial = str(row["serial_number"])
                raw = np.asarray([[_row_value(row, name) for name in CURRENT_COLUMNS]], dtype=float)
                score = float(_apply_preprocess(raw, prep)[0] @ coef + float(payload["intercept"]))
                yield {"decision_date": date_text, "serial_number": serial, "score": score, "rule_score": int(row["smart_nonzero_signal_count"]), "tie_break_sha256": str(row["tie_break_sha256"])}
    finally:
        features.close()


def _select(rows: Sequence[Mapping[str, object]], method: str, last_alert: dict[str, dt.date], day: dt.date) -> tuple[list[Mapping[str, object]], int]:
    if method == "current_lr":
        ranked = sorted(rows, key=lambda row: (-float(row["score"]), str(row["tie_break_sha256"]), str(row["serial_number"])))
    elif method == "smart_nonzero":
        ranked = sorted((row for row in rows if int(row["rule_score"]) > 0), key=lambda row: (-int(row["rule_score"]), str(row["tie_break_sha256"]), str(row["serial_number"])))
    else:
        raise ReproducibleStopped(f"unknown method: {method}")
    excluded = 0
    available = []
    for row in ranked:
        previous = last_alert.get(str(row["serial_number"]))
        if previous is not None and (day - previous).days <= COOLDOWN_DAYS:
            excluded += 1
        else:
            available.append(row)
    selected = available[:ceil_budget(len(rows), BUDGET_DENOMINATOR)]
    for row in selected:
        last_alert[str(row["serial_number"])] = day
    return selected, excluded


def _event_map(
    panel_paths: Sequence[Path | str],
    *,
    dataset_end: dt.date,
    start: dt.date,
    end: dt.date,
    score_start: dt.date,
    score_end: dt.date,
    progress_callback: Callable[[], int] | None = None,
) -> dict[str, dict]:
    events = {}
    for serial, rows in _serial_groups(panel_paths, dataset_end=dataset_end, progress_callback=progress_callback):
        if any(row["model"] != MODEL for row in rows):
            continue
        failures = [dt.date.fromisoformat(str(row["date"])) for row in rows if int(row["failure"]) == 1]
        if not failures or not start <= failures[0] <= end:
            continue
        failure = failures[0]
        observed = {dt.date.fromisoformat(str(row["date"])) for row in rows}
        opportunities = []
        for offset in range(1, HORIZON + 1):
            decision = failure - dt.timedelta(days=offset)
            if score_start <= decision <= score_end and decision in observed and sum(decision - dt.timedelta(days=i) in observed for i in range(HISTORY_DAYS)) >= MIN_HISTORY:
                opportunities.append(decision)
        events[serial] = {"event_key": serial, "first_failure_date": failure.isoformat(), "opportunity": int(bool(opportunities)), "hit": 0, "opportunity_dates": [d.isoformat() for d in sorted(opportunities)]}
    return events


def _select_scores(scores: Iterable[Mapping[str, object]], method: str) -> dict:
    last_alert: dict[str, dt.date] = {}
    alerts = []
    daily = []
    grouped = itertools.groupby(scores, key=lambda row: str(row["decision_date"]))
    for date_text, date_iter in grouped:
        date_rows = list(date_iter)
        day = dt.date.fromisoformat(date_text)
        selected, excluded = _select(date_rows, method, last_alert, day)
        daily.append({"date": date_text, "eligible": len(date_rows), "budget": ceil_budget(len(date_rows), BUDGET_DENOMINATOR), "cooldown_excluded": excluded, "alerts": len(selected)})
        alerts.extend(dict(row, method=method) for row in selected)
    return {"method": method, "alerts": alerts, "daily": daily}


def _evaluate_selection(
    selection: dict,
    label_db: Path,
    run_id: str,
    event_map: dict[str, dict],
    *,
    progress_callback: Callable[[], int] | None = None,
) -> tuple[list[dict], dict]:
    method, daily = selection["method"], selection["daily"]
    alerts = []
    labels = _open_readonly(label_db, progress_callback=progress_callback)
    try:
        for selected_row in selection["alerts"]:
            if progress_callback is not None and progress_callback():
                raise CancellationRequested("evaluation query cancelled")
            label = labels.execute(
                "SELECT label,status,first_failure_date FROM label_flow WHERE run_id=? AND decision_date=? AND serial_number=?",
                (run_id, selected_row["decision_date"], selected_row["serial_number"])).fetchone()
            if label is None:
                raise ReproducibleStopped("selected device day has no outcome record")
            alerts.append(dict(selected_row, **dict(label)))
    finally:
        labels.close()
    for row in alerts:
        day = dt.date.fromisoformat(row["decision_date"])
        date_text = row["decision_date"]
        serial = str(row["serial_number"])
        failure = row.get("first_failure_date")
        if serial in event_map and failure:
            lead = (dt.date.fromisoformat(str(failure)) - day).days
            if 1 <= lead <= HORIZON and event_map[serial]["opportunity"]:
                current = event_map[serial]
                if not current["hit"] or lead > int(current.get("earliest_lead_days") or 0):
                    current["hit"] = 1
                    current["earliest_alert_date"] = date_text
                    current["earliest_lead_days"] = lead
    known_hit = sum(row.get("label") == 1 for row in alerts)
    known_no_hit = sum(row.get("label") == 0 for row in alerts)
    unknown = sum(row.get("label") is None for row in alerts)
    opportunities = [item for item in event_map.values() if item["opportunity"]]
    hits = [item for item in opportunities if item["hit"]]
    metrics = {"method": method, "eligible_rows": sum(int(item["eligible"]) for item in daily), "alerts": len(alerts), "known_hit_alerts": known_hit, "known_no_hit_alerts": known_no_hit, "unknown_alerts": unknown, "event_count": len(event_map), "opportunity_events": len(opportunities), "event_hits": len(hits), "event_recall": len(hits) / len(opportunities) if opportunities else None, "cooldown_days": COOLDOWN_DAYS, "budget_denominator": BUDGET_DENOMINATOR}
    return alerts, {"daily": daily, "metrics": metrics, "events": event_map}


def run_current_research(
    *,
    train_panel: Path | str,
    eval_panel: Path | str,
    output_dir: Path | str,
    train_start: str,
    train_end: str,
    train_cutoff: str,
    eval_start: str,
    eval_end: str,
    eval_cutoff: str,
    event_start: str,
    event_end: str,
) -> dict:
    """Run current_lr and smart_nonzero from panel inputs to evaluation."""
    try:
        train_contract = require_panel_contract(train_panel, role="training")
        eval_contract = require_panel_contract(eval_panel, role="evaluation")
    except ContractError as exc:
        raise ReproducibleStopped(str(exc)) from exc
    if train_contract.get("mode") == "real_panel" or eval_contract.get("mode") == "real_panel":
        raise ReproducibleStopped("legacy run is disabled for real panels; use isolated-run")
    try:
        ts, te, tc, es, ee, ec, evs, eve = map(dt.date.fromisoformat, (train_start, train_end, train_cutoff, eval_start, eval_end, eval_cutoff, event_start, event_end))
    except ValueError as exc:
        raise ReproducibleStopped("invalid ISO calendar date") from exc
    if not (ts <= te and te + dt.timedelta(days=HORIZON) <= tc < es <= ee and ee + dt.timedelta(days=HORIZON) <= ec and es <= evs <= eve <= ec):
        raise ReproducibleStopped("training outcomes must end before evaluation starts; both decision ranges need seven days of follow-up")
    train_panel = _path(train_panel, "training panel", must_exist=True)
    eval_panel = _path(eval_panel, "evaluation panel", must_exist=True)
    output_dir = _path(output_dir, "research output")
    if output_dir.exists():
        raise ReproducibleStopped(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    work = output_dir / "work"
    work.mkdir()
    try:
        resource_events: list[dict[str, int | str]] = []
        train_labels = work / "train_labels.sqlite"
        train_features = work / "train_features.sqlite"
        eval_labels = work / "eval_labels.sqlite"
        eval_features = work / "eval_features.sqlite"
        train_label_result = build_labels([train_panel], train_labels, start=train_start, end=train_end, dataset_end=train_cutoff, run_id="repro_train_h7_v1")
        resource_events.append(dict(_guard(started, output_dir, "after_train_labels")))
        train_feature_result = build_current_features(train_panel, train_features, start=train_start, end=train_end, dataset_end=train_cutoff)
        resource_events.append(dict(_guard(started, output_dir, "after_train_features")))
        payload, _training_arrays = _train_model(train_features, train_labels, "repro_train_h7_v1")
        resource_events.append(dict(_guard(started, output_dir, "after_fit_current_lr")))
        _atomic_json(output_dir / "model.json", payload)
        eval_feature_result = build_current_features(eval_panel, eval_features, start=eval_start, end=eval_end, dataset_end=eval_cutoff, history_panels=[train_panel])
        resource_events.append(dict(_guard(started, output_dir, "after_eval_features")))
        selection_files = {}
        for method in ("current_lr", "smart_nonzero"):
            name = f"{method}_selection.json"
            _atomic_json(output_dir / name, _select_scores(_rule_and_model_scores(eval_features, payload), method))
            selection_files[name] = _sha256_file(output_dir / name)
        _atomic_json(output_dir / "selection_manifest.json", {"status": "closed", "outputs": selection_files, "model_sha256": _sha256_file(output_dir / "model.json")})
        # Outcome access starts only after both methods' selections are closed.
        for name, digest in selection_files.items():
            if _sha256_file(output_dir / name) != digest:
                raise ReproducibleStopped("selection changed before evaluation")
        eval_label_result = build_labels([train_panel, eval_panel], eval_labels, start=eval_start, end=eval_end, dataset_end=eval_cutoff, run_id="repro_eval_h7_v1")
        resource_events.append(dict(_guard(started, output_dir, "after_eval_labels")))
        events = _event_map([train_panel, eval_panel], dataset_end=ec, start=evs, end=eve, score_start=es, score_end=ee)
        summaries = {}
        output_files = ["result.json", "model.json", "selection_manifest.json", *selection_files]
        for method in ("current_lr", "smart_nonzero"):
            selection = json.loads((output_dir / f"{method}_selection.json").read_text())
            alerts, summary = _evaluate_selection(selection, eval_labels, "repro_eval_h7_v1", {key: dict(value) for key, value in events.items()})
            alert_name = f"{method}_alerts.json"
            _atomic_json(output_dir / alert_name, {"status": "complete", "method": method, "alerts": alerts})
            summary["alerts_file"] = alert_name
            summaries[method] = summary
            output_files.append(alert_name)
            resource_events.append(dict(_guard(started, output_dir, f"after_{method}_evaluation")))
        result = {"status": "complete", "version": "repro-current-v1", "fit_mode": payload["fit_mode"], "training": {"labels": train_label_result, "features": train_feature_result, "model": payload}, "evaluation": {"labels": eval_label_result, "features": eval_feature_result, "summaries": summaries}, "provenance": {"train_panel_sha256": _sha256_file(train_panel), "eval_panel_sha256": _sha256_file(eval_panel), "code_sha256": _sha256_file(Path(__file__)), "elapsed_seconds": round(time.monotonic() - started, 3), "resource_events": resource_events, "resource": _snapshot(output_dir)}}
        result["scope"] = "panel_prototype"
        result["specification_acceptance"] = "incomplete"
        _atomic_json(output_dir / "result.json", result)
        _atomic_json(output_dir / "run_manifest.json", {"status": "complete", "scope": result["scope"], "specification_acceptance": "incomplete", "result_sha256": _sha256_file(output_dir / "result.json"), "provenance": result["provenance"], "outputs": output_files + ["work/train_labels.sqlite", "work/train_features.sqlite", "work/eval_labels.sqlite", "work/eval_features.sqlite"]})
        return result
    except BaseException as exc:
        _atomic_json(output_dir / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.monotonic() - started, 3), "resource": _snapshot(output_dir)})
        raise
