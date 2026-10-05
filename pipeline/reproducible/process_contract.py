"""Semantic input contracts for the reproducible pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .artifacts import ArtifactError, ROOT, bound_path, sha256_file


class ProcessAccessError(RuntimeError):
    """A child was given an input outside its declared role contract."""


ROLE_INPUTS: dict[str, frozenset[str]] = {
    "build_q1q2": frozenset({"q1q2_manifest", "q1q2_members"}),
    "build_q3": frozenset({"q3_receipt", "q3_archive", "q1q2_identity"}),
    "train": frozenset({"train_panel", "config"}),
    "score": frozenset({"eval_panel", "history_panel", "model", "config", "train_manifest"}),
    "evaluate": frozenset({"eval_panel", "history_panel", "selection_manifest", "score_manifest", "config"}),
    "compare": frozenset({"new_outputs", "frozen_outputs"}),
}


ROLE_REQUIRED: dict[str, frozenset[str]] = {
    "build_q1q2": frozenset({"q1q2_manifest", "q1q2_members"}),
    "build_q3": frozenset({"q3_receipt", "q3_archive", "q1q2_identity"}),
    "train": frozenset({"train_panel", "config"}),
    "score": frozenset({"eval_panel", "history_panel", "model", "config"}),
    "evaluate": frozenset({"eval_panel", "history_panel", "selection_manifest", "score_manifest", "config"}),
    "compare": frozenset({"new_outputs", "frozen_outputs"}),
}


def _input_facts(value: Path | str, name: str) -> dict[str, object]:
    path = bound_path(value, f"{name} input")
    if not path.exists():
        raise ProcessAccessError(f"missing {name} input: {path}")
    if path.is_dir():
        digest = hashlib.sha256()
        count = 0
        total = 0
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise ProcessAccessError(f"{name} input contains symlink: {child}")
            if not child.is_file():
                continue
            count += 1
            size = child.stat().st_size
            total += size
            relative = child.relative_to(path).as_posix()
            child_sha = sha256_file(child)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(child_sha.encode("ascii"))
            digest.update(b"\n")
        return {"path": str(path.relative_to(ROOT)),
                "kind": "directory", "files": count, "bytes": total,
                "tree_sha256": digest.hexdigest()}
    return {"path": str(path.relative_to(ROOT)),
            "kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def authorize_inputs(role: str, bindings: Mapping[str, Path | str]) -> dict[str, object]:
    """Validate and fingerprint the inputs allowed for one role."""
    if role not in ROLE_INPUTS:
        raise ProcessAccessError(f"unknown process role: {role}")
    if not isinstance(bindings, Mapping):
        raise ProcessAccessError("process input bindings must be an object")
    names = set(bindings)
    unknown = sorted(names - ROLE_INPUTS[role])
    if unknown:
        raise ProcessAccessError(f"{role} is not permitted to receive: {', '.join(unknown)}")
    missing = sorted(ROLE_REQUIRED[role] - names)
    if missing:
        raise ProcessAccessError(f"{role} missing required inputs: {', '.join(missing)}")
    try:
        facts = {name: _input_facts(bindings[name], name) for name in sorted(names)}
    except ArtifactError as exc:
        raise ProcessAccessError(str(exc)) from exc
    return {"contract_version": "process-access-v1", "role": role, "inputs": facts}


def authorize_bindings_file(role: str, path: Path | str) -> dict[str, object]:
    """Load a JSON object of semantic input names and validate it."""
    try:
        binding_path = bound_path(path, "bindings file", must_exist=True)
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProcessAccessError(f"invalid bindings file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ProcessAccessError("bindings file must contain an object")
    if any(type(key) is not str or type(value) is not str for key, value in payload.items()):
        raise ProcessAccessError("bindings file keys and paths must be strings")
    try:
        return authorize_inputs(role, dict(payload))
    except ArtifactError as exc:
        raise ProcessAccessError(str(exc)) from exc


__all__ = ["ProcessAccessError", "ROLE_INPUTS", "ROLE_REQUIRED", "authorize_inputs", "authorize_bindings_file"]
