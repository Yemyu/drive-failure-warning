"""Small, stable runtime snapshot used by reproducible stage manifests."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import sqlite3
import sys
import sys
from .artifacts import bound_path, sha256_file, write_json_exclusive
from .runtime_context import code_root, project_root


ROOT = project_root()
CODE_ROOT = code_root()
PACKAGE_NAMES = ("numpy", "scipy", "scikit-learn")
SOURCE_MANIFEST_ENV = "REPRO_SOURCE_MANIFEST"


class ModuleIdentityError(RuntimeError):
    """A loaded project module is outside the bound source snapshot."""


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot() -> dict[str, object]:
    """Return runtime facts without including process IDs or timestamps."""
    executable = Path(sys.executable).resolve()
    try:
        executable_text = str(executable.relative_to(ROOT))
    except ValueError:
        executable_text = str(executable)
    source_modules: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin or not str(origin).endswith(".py"):
            continue
        try:
            path = Path(origin).resolve()
            relative = path.relative_to(CODE_ROOT)
        except (OSError, ValueError):
            continue
        source_modules[name] = relative.as_posix()
    return {
        "schema": "runtime-environment-v1",
        "execution_roots": {
            "project_root": str(ROOT),
            "code_root": str(CODE_ROOT),
            "source_modules": source_modules,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": executable_text,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "sqlite": {
            "library_version": sqlite3.sqlite_version,
        },
        "cpu_count": os.cpu_count(),
    }


def write_snapshot(path: Path | str) -> dict[str, object]:
    """Write a new environment JSON file and refuse an existing name."""
    target = bound_path(path, "environment snapshot")
    payload = snapshot()
    write_json_exclusive(target, payload)
    return payload


def _loaded_project_modules() -> list[tuple[str, Path]]:
    """Return every imported ``pipeline`` module and its resolved source file."""
    found: list[tuple[str, Path]] = []
    for name, module in sorted(sys.modules.items()):
        if name != "pipeline" and not name.startswith("pipeline."):
            continue
        origin = getattr(module, "__file__", None)
        if not origin or not str(origin).endswith(".py"):
            continue
        found.append((name, Path(origin).resolve()))
    return found


def project_module_identity() -> dict[str, object]:
    """Record the source path and digest of every loaded ``pipeline`` module.

    Paths are reported relative to the bound code root.  A module that resolves
    outside that root is reported as a violation instead of being skipped.
    """
    modules: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    for name, path in _loaded_project_modules():
        try:
            relative = path.relative_to(CODE_ROOT).as_posix()
        except ValueError:
            violations.append({"module": name, "reason": "loaded outside the bound code root", "path": str(path)})
            continue
        modules.append({"module": name, "path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "project-module-identity-v1",
        "code_root": str(CODE_ROOT),
        "modules": modules,
        "violations": violations,
    }


def verify_project_modules(snapshot_files: object) -> dict[str, object]:
    """Check loaded ``pipeline`` modules against the bound snapshot manifest.

    ``snapshot_files`` is the verified ``files`` list of a source manifest.  A
    module outside the snapshot, or with a different digest, raises instead of
    being tolerated.
    """
    allowed: dict[str, str] = {}
    if isinstance(snapshot_files, list):
        for item in snapshot_files:
            if not isinstance(item, dict):
                raise ModuleIdentityError("source snapshot file facts are invalid")
            relative, digest = item.get("path"), item.get("sha256")
            if type(relative) is str and type(digest) is str:
                allowed[relative] = digest
    if not allowed:
        raise ModuleIdentityError("source snapshot file list is empty")
    identity = project_module_identity()
    violations = list(identity["violations"])
    for module in identity["modules"]:
        relative = str(module["path"])
        if relative not in allowed:
            violations.append({"module": module["module"], "reason": "not part of the source snapshot", "path": relative})
        elif allowed[relative] != module["sha256"]:
            violations.append({"module": module["module"], "reason": "snapshot module changed", "path": relative})
    if violations:
        raise ModuleIdentityError(f"project modules do not match the bound snapshot: {violations}")
    return {**identity, "snapshot_file_count": len(allowed), "violations": []}


def module_identity_record() -> dict[str, object] | None:
    """Verify loaded modules when the supervisor bound a source manifest.

    Returns ``None`` for unsupervised processes, which have no snapshot to bind.
    """
    import json
    from pathlib import Path

    declared = os.environ.get(SOURCE_MANIFEST_ENV)
    if not declared:
        return None
    manifest_path = Path(declared)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != "source-snapshot-v1":
        raise ModuleIdentityError("bound source manifest is invalid")
    record = verify_project_modules(payload.get("files"))
    return {
        "source_manifest": str(manifest_path),
        "code_root": str(CODE_ROOT),
        "module_count": len(record["modules"]),
        "modules": record["modules"],
        "violations": [],
    }


__all__ = [
    "ModuleIdentityError",
    "PACKAGE_NAMES",
    "ROOT",
    "SOURCE_MANIFEST_ENV",
    "module_identity_record",
    "project_module_identity",
    "snapshot",
    "verify_project_modules",
    "write_snapshot",
]
