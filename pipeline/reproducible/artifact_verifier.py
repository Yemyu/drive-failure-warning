"""Independent artifact verification for one supervised work group.

The module runs as its own subprocess inside the supervised process group,
after every stage has exited and before the outer supervisor decides whether
to publish.  It re-hashes the source snapshot, every stage artifact, the
approval/binding records and the declared inputs, and writes one small
``verification.json``.  The supervisor then builds the final manifest from
these already-checked facts, so the commit point itself never rescans large
files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
import os

from .artifacts import ArtifactError, _canonical, bound_path, sha256_file, write_json_exclusive
from .resource_guard import assert_fixture_controls_allowed
from .source_snapshot import SourceSnapshotError, verify_source_snapshot


VERIFICATION_VERSION = "verification-request-v1"
RESULT_SCHEMA = "artifact-verification-v1"


class VerificationError(RuntimeError):
    """The verification request is invalid or a required artifact is missing."""


def _reject_constant(value: str) -> object:
    raise VerificationError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_no_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise VerificationError(f"{label} must be an object")
    return payload


def _self_hash_valid(manifest: Mapping[str, object]) -> bool:
    declared = manifest.get("manifest_hash")
    if type(declared) is not str or len(declared) != 64:
        return False
    return declared == _canonical({key: value for key, value in manifest.items() if key != "manifest_hash"})


def _collect(output: Path, snapshot_dir: Path, skip_names: set[str]) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Hash every produced file except the separately verified snapshot tree."""
    facts: dict[str, dict[str, object]] = {}
    problems: list[str] = []
    snapshot_resolved = snapshot_dir.resolve()
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            problems.append(f"symlink in output tree: {path.relative_to(output).as_posix()}")
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in skip_names or path.name.startswith("."):
            continue
        if path.resolve().is_relative_to(snapshot_resolved):
            continue
        facts[relative] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    source_manifest = snapshot_dir / "source_manifest.json"
    relative_source = source_manifest.resolve().relative_to(output.resolve()).as_posix()
    if source_manifest.is_file():
        facts[relative_source] = {
            "path": relative_source,
            "bytes": source_manifest.stat().st_size,
            "sha256": sha256_file(source_manifest),
        }
    else:
        problems.append("source manifest is missing from the snapshot")
    return facts, problems


def verify_run(request_path: Path | str) -> dict[str, object]:
    """Verify one supervised output tree and write ``verification.json``."""
    request_file = Path(request_path)
    request = _load_json(request_file, "verification request")
    if request.get("version") != VERIFICATION_VERSION:
        raise VerificationError("verification request version is invalid")
    assert_fixture_controls_allowed(request.get("config_profile"))
    output_dir = Path(str(request.get("output_dir")))
    snapshot_dir = Path(str(request.get("snapshot_dir")))
    project_root = Path(str(request.get("project_root")))
    declared_source_sha = request.get("source_manifest_sha256")
    if not output_dir.is_dir() or not snapshot_dir.is_dir():
        raise VerificationError("verification request directories are missing")
    if type(declared_source_sha) is not str or len(declared_source_sha) != 64:
        raise VerificationError("verification request source digest is invalid")

    reasons: list[str] = []
    checks: dict[str, object] = {}

    try:
        snapshot = verify_source_snapshot(
            snapshot_dir,
            project_root=project_root,
            expectation=request.get("expectation") if isinstance(request.get("expectation"), Mapping) else None,
        )
    except SourceSnapshotError as exc:
        reasons.append(f"source snapshot: {exc}")
        snapshot = None
    checks["source_snapshot"] = {
        "status": "pass" if snapshot is not None else "fail",
        "manifest_sha256": snapshot["manifest_sha256"] if snapshot else None,
        "matches_request": bool(snapshot and snapshot["manifest_sha256"] == declared_source_sha),
        "file_count": len(snapshot["verified_files"]) if snapshot else 0,
    }
    if snapshot is not None and snapshot["manifest_sha256"] != declared_source_sha:
        reasons.append("source snapshot digest differs from the bound request")

    output_name = str(request.get("output") or "verification.json")
    candidate_name = str(request.get("candidate") or "candidate_manifest.json")
    skip = {output_name, f".{output_name}.partial"}
    facts, problems = _collect(output_dir, snapshot_dir, skip)
    reasons.extend(problems)

    candidate_path = output_dir / candidate_name
    candidate: Mapping[str, object] | None = None
    if not candidate_path.is_file():
        reasons.append(f"candidate manifest is missing: {candidate_name}")
    else:
        candidate = _load_json(candidate_path, "candidate manifest")
        if not _self_hash_valid(candidate):
            reasons.append("candidate manifest self-hash is invalid")
        if candidate.get("status") != "complete":
            reasons.append("candidate manifest status is not complete")
        if candidate.get("completion_state") != "candidate_pending_external_checks":
            reasons.append("candidate is not pending external checks")
        declared_artifacts = candidate.get("artifacts")
        if not isinstance(declared_artifacts, Mapping) or not declared_artifacts:
            reasons.append("candidate manifest declares no artifacts")
        else:
            mismatched: list[str] = []
            for name, raw in sorted(declared_artifacts.items()):
                if not isinstance(raw, Mapping):
                    mismatched.append(str(name))
                    continue
                relative = str(raw.get("path", ""))
                known = facts.get(relative)
                if known is None:
                    continue
                if int(known["bytes"]) != int(raw.get("bytes", -1)) or str(known["sha256"]) != str(raw.get("sha256", "")):
                    mismatched.append(str(name))
            if mismatched:
                reasons.append(f"candidate artifact facts differ from recomputed hashes: {mismatched}")
    checks["candidate_manifest"] = {
        "status": "pass" if not [item for item in reasons if item.startswith("candidate")] else "fail",
        "present": candidate_path.is_file(),
        "self_hash_valid": bool(candidate is not None and _self_hash_valid(candidate)),
    }

    stage_manifests: dict[str, dict[str, object]] = {}
    stages_dir = output_dir / "stages"
    if stages_dir.is_dir():
        for path in sorted(stages_dir.rglob("*_manifest.json")):
            relative = path.relative_to(output_dir).as_posix()
            try:
                payload = _load_json(path, "stage manifest")
            except (VerificationError, OSError, json.JSONDecodeError) as exc:
                stage_manifests[relative] = {"status": "fail", "error": str(exc)}
                reasons.append(f"stage manifest unreadable: {relative}")
                continue
            valid = _self_hash_valid(payload)
            stage_manifests[relative] = {"status": "pass" if valid else "fail", "self_hash_valid": valid}
            if not valid:
                reasons.append(f"stage manifest self-hash invalid: {relative}")
    checks["stage_manifests"] = stage_manifests

    bindings: dict[str, dict[str, object]] = {}
    specs_dir = output_dir / "specs"
    if specs_dir.is_dir():
        for path in sorted(specs_dir.glob("*.binding.json")):
            relative = path.relative_to(output_dir).as_posix()
            try:
                payload = _load_json(path, "binding record")
                bindings[relative] = {"status": "pass", "keys": sorted(payload)}
            except (VerificationError, OSError, json.JSONDecodeError) as exc:
                bindings[relative] = {"status": "fail", "error": str(exc)}
                reasons.append(f"approval binding record unreadable: {relative}")
    checks["approval_binding_records"] = bindings

    module_records: dict[str, object] = {}
    if stages_dir.is_dir():
        for path in sorted(stages_dir.rglob("*_manifest.json")):
            relative = path.relative_to(output_dir).as_posix()
            try:
                payload = _load_json(path, "stage manifest")
            except (VerificationError, OSError, json.JSONDecodeError):
                continue
            record = payload.get("source_modules")
            if record is None:
                module_records[relative] = "absent"
                continue
            if not isinstance(record, Mapping):
                module_records[relative] = "invalid"
                reasons.append(f"module identity record is invalid: {relative}")
                continue
            violations = record.get("violations")
            modules = record.get("modules")
            if not isinstance(modules, list) or not modules:
                module_records[relative] = "empty"
                reasons.append(f"module identity record is empty: {relative}")
            elif isinstance(violations, list) and violations:
                module_records[relative] = "violations"
                reasons.append(f"module identity violations: {relative}: {violations}")
            else:
                module_records[relative] = {"modules": len(modules), "status": "pass"}
    checks["module_identity_records"] = module_records

    status = "pass" if not reasons else "fail"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "scope": "supervised_work_group_artifact_verification",
        "output_dir": str(output_dir),
        "snapshot_dir": str(snapshot_dir),
        "source_manifest_sha256": declared_source_sha,
        "checked_file_count": len(facts),
        "artifacts": facts,
        "checks": checks,
        "reasons": reasons,
    }
    write_json_exclusive(output_dir / output_name, result)
    return result


__all__ = ["RESULT_SCHEMA", "VERIFICATION_VERSION", "VerificationError", "verify_run"]
