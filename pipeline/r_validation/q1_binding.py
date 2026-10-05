"""Release and source binding for the frozen 2024 Q1 validation protocol.

This module deliberately keeps the Q1 contract separate from the historical
Q4 contract.  The computation engine is shared, but a Q4 lock or source
prefix can never authorise a Q1 read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from pipeline.reproducible.artifacts import bound_path, sha256_file
from pipeline.reproducible.supervisor import _load_small
from pipeline.reproducible.runtime_context import project_root
from pipeline.r_validation.release import ReleaseError, require_release

ROOT = project_root()
CONFIG_PATH = ROOT / "configs/q1_2024_validation_v1.json"
PROTOCOL_PATH = ROOT / "evidence/protocols/q1_2024_validation_v1.md"
RELEASE_PATH = ROOT / "evidence/q1_2024_validation_v1/release.json"
Q1_PROFILES = {"synthetic_q1_bound_zip", "released_q1_zip"}

# These are completion records produced by the earlier, separately accepted
# stages.  The Q1 preflight must read the recorded source binding from each
# record and compare it with the current bytes; it may not create a new
# "completion" from the current bytes alone.
HISTORICAL_COMPLETION_SPECS = {
    "q1q2_panel": {
        "manifest": "evidence/q3/scoring_amended_v2/attempt_001/q3_scoring_amended_v2_complete_manifest_v1.json",
        "kind": "input_binding",
        "field": "data/derived/panel_q1q2_verified_v1.sqlite",
    },
    "q3_panel": {
        "manifest": "evidence/q3/validation_v1/q3_complete_manifest_v1.json",
        "kind": "database",
    },
    "q4_candidate_panel": {
        "manifest": "evidence/r_validation_v1/attempt_001/continuation_001/r_complete.json",
        "kind": "artifact",
        "field": "work/candidate/candidate.sqlite",
    },
}


def _canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _config() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != "q1_2024_validation_v1":
        raise ReleaseError("Q1 configuration version mismatch")
    return payload


def _file(binding: object, label: str) -> Path:
    if not isinstance(binding, dict) or not {"path", "sha256"}.issubset(binding):
        raise ReleaseError(f"invalid {label} binding")
    path = bound_path(binding["path"], label, must_exist=True)
    if sha256_file(path) != binding["sha256"]:
        raise ReleaseError(f"{label} bytes changed")
    return path


def _spec(config: dict) -> dict:
    dates = config["dates"]
    return {
        key: dates[key]
        for key in ("score_start", "score_end", "event_start", "event_end", "outcome_cutoff")
    } | {"horizon_days": config["labels"]["horizon_days"], "allow_smart_decreases": True}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def historical_source_proof() -> dict:
    """Build and verify the three role-specific completion bindings.

    The returned payload is intentionally explicit about the manifest field
    that supplied each source path and digest.  Q4's WAL is required to be
    empty; its normal 32 KiB ``-shm`` sidecar is recorded but is not treated as
    database content.
    """
    entries = {}
    for role, spec in HISTORICAL_COMPLETION_SPECS.items():
        manifest_path = bound_path(spec["manifest"], f"Q1 {role} completion manifest", must_exist=True)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ReleaseError(f"Q1 {role} completion manifest is unreadable") from exc
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise ReleaseError(f"Q1 {role} completion manifest is not complete")
        if spec["kind"] == "input_binding":
            bindings = manifest.get("input_bindings")
            source_relative = spec["field"]
            expected_sha = bindings.get(source_relative) if isinstance(bindings, dict) else None
            field = f"input_bindings[{source_relative}]"
        elif spec["kind"] == "database":
            source_relative = manifest.get("database")
            expected_sha = manifest.get("database_sha256")
            field = "database/database_sha256"
        else:
            artifacts = manifest.get("artifacts")
            artifact = artifacts.get(spec["field"]) if isinstance(artifacts, dict) else None
            source_relative = artifact.get("path") if isinstance(artifact, dict) else None
            expected_sha = artifact.get("sha256") if isinstance(artifact, dict) else None
            field = f"artifacts[{spec['field']}]"
        if not isinstance(source_relative, str) or not isinstance(expected_sha, str):
            raise ReleaseError(f"Q1 {role} completion manifest has no usable {field}")
        source_path = bound_path(source_relative, f"Q1 {role} historical source", must_exist=True)
        current_sha = sha256_file(source_path)
        if current_sha != expected_sha:
            raise ReleaseError(f"Q1 {role} current source differs from its completion manifest")
        entry = {
            "role": role,
            "manifest": {"path": _relative(manifest_path), "sha256": sha256_file(manifest_path)},
            "manifest_status": manifest["status"],
            "source_field": field,
            "source": {"path": _relative(source_path), "sha256": current_sha},
        }
        if role == "q4_candidate_panel":
            wal = source_path.with_name(source_path.name + "-wal")
            shm = source_path.with_name(source_path.name + "-shm")
            wal_bytes = wal.stat().st_size if wal.exists() else 0
            shm_bytes = shm.stat().st_size if shm.exists() else 0
            if wal_bytes != 0:
                raise ReleaseError("Q1 Q4 candidate has a non-empty WAL sidecar")
            try:
                uri = source_path.as_uri() + "?mode=ro"
                db = sqlite3.connect(uri, uri=True)
                try:
                    journal_mode = db.execute("PRAGMA journal_mode").fetchone()[0]
                finally:
                    db.close()
            except (OSError, sqlite3.Error) as exc:
                raise ReleaseError("Q1 Q4 candidate WAL state cannot be inspected") from exc
            entry["sqlite_sidecars"] = {"wal_bytes": wal_bytes, "shm_bytes": shm_bytes,
                                         "journal_mode": journal_mode}
        entries[role] = entry
    return {"schema": "q1-historical-source-proof-v1", "status": "pass", "entries": entries}


def _verify_historical_source_proof(proof: dict) -> None:
    if not isinstance(proof, dict) or proof.get("schema") != "q1-historical-source-proof-v1" or proof.get("status") != "pass":
        raise ReleaseError("Q1 historical source proof is not passing")
    if set(proof.get("entries", {})) != set(HISTORICAL_COMPLETION_SPECS):
        raise ReleaseError("Q1 historical source proof roles differ")
    # Re-read the fixed completion records and compare every recorded path and
    # digest.  This prevents a proof file from becoming a second self-reported
    # source of truth.
    current = historical_source_proof()
    if current != proof:
        raise ReleaseError("Q1 historical source proof changed")


def _history_from_lock(lock: dict) -> dict:
    historical = lock.get("historical_inputs")
    if not isinstance(historical, dict) or not isinstance(historical.get("panels"), dict):
        raise ReleaseError("Q1 lock has no historical panel binding")
    return {
        role: {"path": str(_file(item, f"Q1 historical panel {role}")), "sha256": item["sha256"]}
        for role, item in historical["panels"].items()
    }


def _validate_receipt(receipt_path: Path, receipt: dict, config: dict, archive: Path, archive_sha: str) -> None:
    """Check the small, supervised acquisition facts before archive access."""
    from pipeline.r_validation.acquisition import validate_acquisition_receipt
    validate_acquisition_receipt(receipt, receipt_path, config["source_zip"])
    if receipt.get("schema") != "r-download-completion-v1" or receipt.get("status") != "download_complete":
        raise ReleaseError("Q1 download receipt is not a completed supervised receipt")
    expected = {"source_object_id": config["source_zip"]["bz_file_id"],
                "url": config["source_zip"]["url"],
                "bytes": config["source_zip"]["content_length"],
                "archive_sha256": archive_sha, "archive": str(archive)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ReleaseError("Q1 download receipt identity differs from the approved object")
    if receipt.get("cleanup", {}).get("state") != "complete":
        raise ReleaseError("Q1 download cleanup is not confirmed")
    parent = receipt_path.parent
    for name in ("response.json", "network_attempt.json", "download_candidate.json"):
        if not (parent / name).is_file():
            raise ReleaseError(f"Q1 acquisition evidence is missing: {name}")
    response = _load_small(parent / "response.json", "Q1 download response")
    attempt = _load_small(parent / "network_attempt.json", "Q1 network attempt")
    candidate = _load_small(parent / "download_candidate.json", "Q1 download candidate")
    headers = response.get("headers", {})
    if (response.get("status") != 200 or response.get("url") != config["source_zip"]["url"]
            or headers.get("content-length") != str(config["source_zip"]["content_length"])
            or headers.get("x-bz-file-id") != config["source_zip"]["bz_file_id"]
            or headers.get("x-bz-content-sha1") != config["source_zip"]["content_sha1"]
            or attempt.get("attempts") != 1 or attempt.get("automatic_retry") is not False
            or candidate.get("status") != "download_candidate"
            or candidate.get("archive_sha256") != archive_sha):
        raise ReleaseError("Q1 acquisition evidence differs from the receipt")


def _verify_lock(lock_path: Path) -> dict:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"Q1 lock is unreadable: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema") != "q1-lock-v1":
        raise ReleaseError("Q1 lock schema is invalid")
    claimed = lock.get("lock_digest")
    body = {key: value for key, value in lock.items() if key != "lock_digest"}
    if not isinstance(claimed, str) or _canonical(body) != claimed:
        raise ReleaseError("Q1 lock self-digest does not match")
    if lock.get("protocol_source_sha256") != sha256_file(PROTOCOL_PATH):
        raise ReleaseError("Q1 protocol changed after lock creation")
    if lock.get("config_sha256") != sha256_file(CONFIG_PATH) or lock.get("config_digest") != _canonical(_config()):
        raise ReleaseError("Q1 configuration changed after lock creation")
    implementation = lock.get("implementation_sha256")
    if not isinstance(implementation, dict):
        raise ReleaseError("Q1 lock has no implementation binding")
    from tools.run_r_validation import implementation_files
    expected_implementation = {}
    for relative in implementation_files():
        path = ROOT / relative
        if not path.is_file():
            raise ReleaseError(f"Q1 implementation file is missing: {relative}")
        expected_implementation[relative] = sha256_file(path)
    if implementation != expected_implementation:
        raise ReleaseError("Q1 implementation dependency closure changed")
    approved = lock.get("approved_parameters")
    if not isinstance(approved, dict):
        raise ReleaseError("Q1 lock has no approved parameters")
    _file(approved, "Q1 approved parameters")
    historical = lock.get("historical_inputs")
    if not isinstance(historical, dict) or historical.get("roles") != ["q1q2_panel", "q3_panel", "q4_candidate_panel"]:
        raise ReleaseError("Q1 historical roles are not the approved three-source set")
    source_proof_binding = historical.get("source_proof")
    source_proof_path = _file(source_proof_binding, "Q1 historical source proof")
    try:
        source_proof = json.loads(source_proof_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError("Q1 historical source proof is unreadable") from exc
    _verify_historical_source_proof(source_proof)
    completion_manifests = historical.get("completion_manifests")
    if (not isinstance(completion_manifests, dict)
            or set(completion_manifests) != set(HISTORICAL_COMPLETION_SPECS)):
        raise ReleaseError("Q1 historical completion manifest set differs")
    for role, item in completion_manifests.items():
        _file(item, f"Q1 {role} completion manifest")
        if item != source_proof["entries"][role]["manifest"]:
            raise ReleaseError(f"Q1 {role} completion manifest is not the recorded source")
    panels = historical.get("panels", {})
    if set(panels) != set(HISTORICAL_COMPLETION_SPECS):
        raise ReleaseError("Q1 historical panel role set differs")
    for role, item in panels.items():
        _file(item, f"Q1 historical panel {role}")
        if item.get("path") != source_proof["entries"][role]["source"]["path"] \
                or item.get("sha256") != source_proof["entries"][role]["source"]["sha256"]:
            raise ReleaseError(f"Q1 {role} panel is not the completion-record source")
    directory_binding = lock.get("directory_preflight")
    directory_path = _file(directory_binding, "Q1 directory preflight")
    try:
        directory_proof = json.loads(directory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError("Q1 directory preflight is unreadable") from exc
    from pipeline.r_validation.q1_directory import verify_directory_proof
    config = _config()
    verify_directory_proof(directory_proof, config["source_zip"],
                           start=config["dates"]["quarter_raw"][0],
                           end=config["dates"]["quarter_raw"][1])
    evidence = lock.get("acceptance_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "pass" or not isinstance(evidence.get("files"), dict):
        raise ReleaseError("Q1 lock has no passing synthetic acceptance binding")
    for relative, expected in evidence["files"].items():
        path = bound_path(relative, "Q1 acceptance evidence", must_exist=True)
        if sha256_file(path) != expected:
            raise ReleaseError(f"Q1 acceptance evidence changed: {relative}")
    return lock


def validate_q1_bound_zip(contract: dict) -> dict:
    """Validate a Q1 synthetic or released contract before opening ZIP bytes."""
    fields = {"schema", "profile", "source", "model", "history", "spec", "authorization"}
    if not isinstance(contract, dict) or set(contract) != fields or contract.get("schema") != "r-bound-zip-v1":
        raise ReleaseError("invalid Q1 bound ZIP contract")
    profile = contract["profile"]
    if profile not in Q1_PROFILES:
        raise ReleaseError("unknown Q1 bound ZIP profile")
    config = _config()
    source = contract["source"]
    if not isinstance(source, dict) or set(source) != {"archive", "receipt", "prefix", "start", "end"}:
        raise ReleaseError("invalid Q1 source fields")
    expected_spec = _spec(config)
    if contract["spec"] != expected_spec:
        raise ReleaseError("Q1 contract dates or label policy differ from protocol")
    expected_prefix = "q1_fixture" if profile == "synthetic_q1_bound_zip" else config["source_zip"]["prefix"]
    if source["prefix"] != expected_prefix or [source["start"], source["end"]] != config["dates"]["quarter_raw"]:
        raise ReleaseError("Q1 ZIP prefix or raw calendar differs from protocol")
    if profile == "synthetic_q1_bound_zip":
        if contract["authorization"] is not None or source["receipt"] is not None:
            raise ReleaseError("synthetic Q1 source cannot carry real authorization")
    else:
        auth = contract["authorization"]
        if not isinstance(auth, dict) or set(auth) != {"lock", "release"}:
            raise ReleaseError("Q1 release authorization is missing")
        lock_path = _file(auth["lock"], "Q1 lock")
        release_path = _file(auth["release"], "Q1 release")
        if release_path != RELEASE_PATH:
            raise ReleaseError("Q1 release must use the fixed project entry")
        lock = _verify_lock(lock_path)
        release = require_release(RELEASE_PATH, r0_lock_digest=lock["lock_digest"],
                                  source_object_id=config["source_zip"]["bz_file_id"], stage="panel")
        if set(release.get("allowed_stages", ())) != set(config["release_gate"]["allowed_stages"]):
            raise ReleaseError("Q1 release does not cover the complete chain")
        expected_history = _history_from_lock(lock)
        expected_model = {"path": str(_file(lock["approved_parameters"], "Q1 model")),
                          "sha256": lock["approved_parameters"]["sha256"]}
        if contract["history"] != expected_history or contract["model"] != expected_model:
            raise ReleaseError("Q1 model or history differs from the lock")
        archive = bound_path(source["archive"]["path"], "Q1 source ZIP", must_exist=True)
        expected_archive = ROOT / config["paths"]["raw"] / "data_Q1_2024.zip"
        if archive != expected_archive:
            raise ReleaseError("Q1 source ZIP is outside the fixed raw path")
        receipt_path = _file(source["receipt"], "Q1 download receipt")
        receipt = _load_small(receipt_path, "Q1 download receipt")
        expected = {
            "status": "download_complete", "source_object_id": config["source_zip"]["bz_file_id"],
            "url": config["source_zip"]["url"], "bytes": config["source_zip"]["content_length"],
            "archive_sha256": source["archive"]["sha256"], "archive": str(archive),
            "r0_lock_digest": lock["lock_digest"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ReleaseError("Q1 download receipt is not bound to the lock and object")
        _validate_receipt(receipt_path, receipt, config, archive, source["archive"]["sha256"])
        if archive.stat().st_size != config["source_zip"]["content_length"]:
            raise ReleaseError("Q1 ZIP length differs from the official object")
    archive = _file(source["archive"], "Q1 source ZIP")
    model = _file(contract["model"], "Q1 frozen model")
    if not isinstance(contract["history"], dict) or set(contract["history"]) != set(config["history"]["roles"]):
        raise ReleaseError("Q1 contract must bind exactly three historical panels")
    history = {role: {"path": str(_file(item, f"Q1 history {role}")), "sha256": item["sha256"]}
               for role, item in contract["history"].items()}
    return {"archive": archive, "model_source": model,
            "expected_model_sha256": contract["model"]["sha256"], "spec": contract["spec"],
            "source_args": {"expected_sha256": contract["source"]["archive"]["sha256"],
                            "start": source["start"], "end": source["end"],
                            "prefix": source["prefix"], "historical_inputs": history}}


__all__ = ["HISTORICAL_COMPLETION_SPECS", "Q1_PROFILES", "historical_source_proof",
           "validate_q1_bound_zip"]
