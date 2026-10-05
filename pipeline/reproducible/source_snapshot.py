"""Create and verify the Python source snapshot used by supervised runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from collections.abc import Mapping

from .artifacts import ArtifactError, bound_path, sha256_file, write_json_exclusive


SNAPSHOT_VERSION = "source-snapshot-v1"
ENTRYPOINT = "tools/run_research.py"


class SourceSnapshotError(RuntimeError):
    """The source snapshot is incomplete, changed, or unsafe to execute."""


def _reject_constant(value: str) -> object:
    raise SourceSnapshotError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SourceSnapshotError(f"duplicate source snapshot key: {key}")
        result[key] = value
    return result


def _digest(path: Path) -> str:
    try:
        return sha256_file(path)
    except ArtifactError as exc:
        raise SourceSnapshotError(str(exc)) from exc


def _files(project_root: Path) -> list[Path]:
    pipeline = project_root / "pipeline"
    entrypoint = project_root / ENTRYPOINT
    if not pipeline.is_dir() or not entrypoint.is_file() or entrypoint.is_symlink():
        raise SourceSnapshotError("project source tree is missing pipeline or run_research.py")
    result: list[Path] = []
    for source in sorted(pipeline.rglob("*.py")):
        if source.is_symlink() or not source.is_file():
            raise SourceSnapshotError(f"source tree contains a symlink or non-file: {source}")
        result.append(source)
    result.append(entrypoint)
    for relative in ('tools/run_r_validation.py', 'tools/run_small_replay.py', 'tools/run_q1_validation.py'):
        source = project_root / relative
        if source.is_symlink():
            raise SourceSnapshotError(f"source tree contains a symlink: {source}")
        if source.exists():
            if not source.is_file():
                raise SourceSnapshotError(f"source tool is not a file: {source}")
            result.append(source)
    return result


def _facts_before_copy(root: Path, sources: list[Path]) -> dict[str, tuple[int, str]]:
    """Hash every source file before anything is copied."""
    facts: dict[str, tuple[int, str]] = {}
    for source in sources:
        relative = source.relative_to(root).as_posix()
        facts[relative] = (source.stat().st_size, _digest(source))
    return facts


def _expectation(files: list[dict[str, object]]) -> dict[str, object]:
    """Digest the file list so a rewritten manifest cannot declare itself valid."""
    canonical = json.dumps(sorted((str(item["path"]), int(item["bytes"]), str(item["sha256"])) for item in files), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "source-snapshot-expectation-v1",
        "file_count": len(files),
        "files_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def create_source_snapshot(destination: Path | str, *, project_root: Path | str) -> dict[str, object]:
    """Copy the executable source tree into a fresh project-local directory.

    Every source file is hashed before the copy, the file set is re-listed
    after the copy, and both the source and the copy are re-hashed.  A change
    in the set or in any file stops the run instead of recapturing silently.
    """
    root = Path(project_root).resolve()
    try:
        target = bound_path(destination, "source snapshot")
    except ArtifactError as exc:
        raise SourceSnapshotError(str(exc)) from exc
    if not root.is_dir() or target.exists():
        raise SourceSnapshotError(f"source snapshot destination is unavailable: {target}")
    before = _files(root)
    before_names = sorted(source.relative_to(root).as_posix() for source in before)
    expected_facts = _facts_before_copy(root, before)
    target.mkdir(parents=True)
    for source in before:
        relative = source.relative_to(root)
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
    after = _files(root)
    after_names = sorted(source.relative_to(root).as_posix() for source in after)
    if before_names != after_names:
        added = sorted(set(after_names) - set(before_names))
        removed = sorted(set(before_names) - set(after_names))
        raise SourceSnapshotError(
            f"source tree file set changed during snapshot copy: added={added} removed={removed}"
        )
    files: list[dict[str, object]] = []
    for source in after:
        relative = source.relative_to(root).as_posix()
        expected_size, expected_sha = expected_facts[relative]
        source_size, source_sha = source.stat().st_size, _digest(source)
        if (source_size, source_sha) != (expected_size, expected_sha):
            raise SourceSnapshotError(f"source file changed during snapshot copy: {relative}")
        copied = target / relative
        copy_size, copy_sha = copied.stat().st_size, _digest(copied)
        if (copy_size, copy_sha) != (expected_size, expected_sha):
            raise SourceSnapshotError(f"source snapshot copy differs from its source: {relative}")
        files.append({"path": relative, "bytes": copy_size, "sha256": copy_sha})
    manifest = {
        "version": SNAPSHOT_VERSION,
        "project_root": str(root),
        "entrypoint": ENTRYPOINT,
        "files": files,
    }
    write_json_exclusive(target / "source_manifest.json", manifest)
    manifest_sha = _digest(target / "source_manifest.json")
    expectation = _expectation(files)
    return {
        **manifest,
        "path": str(target),
        "manifest_sha256": manifest_sha,
        "expectation": {**expectation, "manifest_sha256": manifest_sha},
    }


def verify_source_snapshot(
    snapshot: Path | str,
    *,
    project_root: Path | str | None = None,
    expectation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Re-hash every copied source file and return the verified manifest."""
    try:
        target = bound_path(snapshot, "source snapshot", must_exist=False)
        manifest_path = target / "source_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise SourceSnapshotError("source snapshot manifest is missing or a symlink")
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError(f"invalid source snapshot: {snapshot}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"version", "project_root", "entrypoint", "files"}:
        raise SourceSnapshotError("source snapshot manifest keys are invalid")
    if payload.get("version") != SNAPSHOT_VERSION or payload.get("entrypoint") != ENTRYPOINT:
        raise SourceSnapshotError("source snapshot version or entrypoint is invalid")
    declared_root = payload.get("project_root")
    if type(declared_root) is not str or not declared_root:
        raise SourceSnapshotError("source snapshot project root is invalid")
    if project_root is not None and Path(declared_root).resolve() != Path(project_root).resolve():
        raise SourceSnapshotError("source snapshot project root differs from request")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise SourceSnapshotError("source snapshot file list is empty")
    seen: set[str] = set()
    checked: list[dict[str, object]] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise SourceSnapshotError("source snapshot file facts are invalid")
        relative = item.get("path")
        if type(relative) is not str or not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SourceSnapshotError("source snapshot file path is invalid")
        size, digest = item.get("bytes"), item.get("sha256")
        if type(size) is not int or size < 0 or type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(char not in "0123456789abcdef" for char in digest):
            raise SourceSnapshotError("source snapshot file facts are invalid")
        seen.add(relative)
        path = target / relative
        if not path.is_file() or path.is_symlink():
            raise SourceSnapshotError(f"source snapshot file is missing: {relative}")
        actual = {"path": relative, "bytes": path.stat().st_size, "sha256": _digest(path)}
        if actual["bytes"] != size or actual["sha256"] != digest:
            raise SourceSnapshotError(f"source snapshot file changed: {relative}")
        checked.append(actual)
    if ENTRYPOINT not in seen:
        raise SourceSnapshotError("source snapshot entrypoint is missing")
    allowed = seen | {"source_manifest.json"}
    for item in target.rglob("*"):
        if item.is_symlink():
            raise SourceSnapshotError(f"source snapshot contains a symlink: {item.relative_to(target)}")
        if item.is_file() and item.relative_to(target).as_posix() not in allowed:
            raise SourceSnapshotError(f"source snapshot contains an unlisted file: {item.relative_to(target)}")
    manifest_sha = _digest(target / "source_manifest.json")
    if expectation is not None:
        local = _expectation(checked)
        for key in ("schema", "file_count", "files_sha256"):
            if key not in expectation or expectation[key] != local[key]:
                raise SourceSnapshotError(f"source snapshot expectation mismatch: {key}")
        declared_manifest = expectation.get("manifest_sha256")
        if type(declared_manifest) is not str or declared_manifest != manifest_sha:
            raise SourceSnapshotError("source snapshot expectation manifest digest mismatch")
    return {
        **dict(payload),
        "path": str(target),
        "manifest_sha256": manifest_sha,
        "verified_files": checked,
    }


__all__ = ["ENTRYPOINT", "SNAPSHOT_VERSION", "SourceSnapshotError", "create_source_snapshot", "verify_source_snapshot"]
