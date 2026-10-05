"""Artifact paths, fingerprints, and atomic manifest publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from .runtime_context import code_root, project_root


ROOT = project_root()
CODE_ROOT = code_root()
CHUNK = 1024 * 1024


class ArtifactError(RuntimeError):
    """A path, write, or artifact binding violated the publication contract."""


_HEX64 = frozenset("0123456789abcdef")


def _reject_constant(value: str) -> object:
    raise ArtifactError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {label}: {path}") from exc


def bound_path(value: Path | str, field: str, *, must_exist: bool = False) -> Path:
    """Resolve a project-local path and reject symlink aliases and escapes."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    cursor = path
    while cursor != ROOT and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ArtifactError(f"{field} must not use a symlink: {cursor}")
        cursor = cursor.parent
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(ROOT):
        raise ArtifactError(f"{field} escapes project root: {path}")
    if must_exist and not resolved.is_file():
        raise ArtifactError(f"missing {field}: {resolved}")
    return resolved


def sha256_file(path: Path | str) -> str:
    """Hash a file in bounded chunks without loading it into memory."""
    path = bound_path(path, "artifact", must_exist=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def artifact_facts(path: Path | str) -> dict[str, object]:
    """Return the stable facts used to bind one output file."""
    bound = bound_path(path, "artifact", must_exist=True)
    return {
        "path": str(bound.relative_to(ROOT)),
        "bytes": bound.stat().st_size,
        "sha256": sha256_file(bound),
    }


def _write_json_exclusive(
    path: Path,
    text: str,
    *,
    write_text: Callable[[object, str], int] | None = None,
) -> None:
    """Write text to a fresh temporary file and publish it atomically.

    ``write_text`` is private test plumbing used to exercise the short-write
    stop.  Production callers use the normal file object's ``write`` method.
    """
    path = bound_path(path, "manifest output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    if path.exists():
        raise ArtifactError(f"refusing to overwrite artifact manifest: {path}")
    if temporary.exists():
        raise ArtifactError(f"partial artifact manifest already exists: {temporary}")
    writer = write_text or (lambda handle, value: handle.write(value))
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            written = writer(handle, text)
            if written != len(text):
                raise ArtifactError(f"short artifact manifest write: {written} != {len(text)}")
            handle.flush()
            os.fsync(handle.fileno())
        # A rename replaces an existing target on POSIX.  The completion
        # contract requires a competing writer to lose with an explicit
        # error, so publish by linking the already-fsynced temporary inode.
        # `link` is atomic within this directory and fails with EEXIST when a
        # target appeared after the preflight check.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ArtifactError(f"refusing to overwrite artifact manifest: {path}") from exc
        except OSError as exc:
            raise ArtifactError(f"exclusive artifact publish failed: {path}: {exc}") from exc
        temporary.unlink()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file has already been atomically published.  Some filesystems
            # do not allow fsync on directory descriptors; content and rename
            # semantics remain checked by the manifest itself.
            pass
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_manifest(
    path: Path | str,
    body: Mapping[str, object],
    *,
    artifacts: Mapping[str, Path | str] | None = None,
    _write_text: Callable[[object, str], int] | None = None,
) -> dict[str, object]:
    """Publish a self-hashed manifest that binds the supplied output files."""
    # Freeze the body before hashing: a shallow copy would still share the
    # nested dicts, and anything mutating them between the hash calculation and
    # the write would make the published self-hash unverifiable.  A JSON
    # round-trip also normalises integer dict keys to the string keys a reload
    # will produce.
    manifest: dict[str, object] = json.loads(json.dumps(dict(body), ensure_ascii=False, allow_nan=False))
    manifest.setdefault("manifest_version", "artifact-manifest-v1")
    if artifacts is not None:
        manifest["artifacts"] = {
            name: artifact_facts(artifact) for name, artifact in sorted(artifacts.items())
        }
    manifest["manifest_hash"] = _canonical(manifest)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_json_exclusive(bound_path(path, "manifest output"), text, write_text=_write_text)
    return manifest


def write_verified_manifest(
    path: Path | str,
    body: Mapping[str, object],
    *,
    verified_artifacts: Mapping[str, Mapping[str, object]],
    base: Path | str | None = None,
    _write_text: Callable[[object, str], int] | None = None,
) -> dict[str, object]:
    """Publish a manifest from facts an independent checker already verified.

    This is the narrow interface used at the commit point: it re-checks the
    declared shape and the file size, but deliberately does not re-hash large
    artifacts, because that work belongs to the verification subprocess.
    ``base`` resolves relative artifact paths when the checker recorded them
    against a run directory instead of the project root.
    """
    root = Path(base).resolve() if base is not None else ROOT
    checked: dict[str, dict[str, object]] = {}
    for name, raw in sorted(verified_artifacts.items()):
        if not isinstance(raw, Mapping):
            raise ArtifactError(f"invalid verified artifact facts: {name}")
        declared_path = raw.get("path")
        size, digest = raw.get("bytes"), raw.get("sha256")
        if type(declared_path) is not str or not declared_path:
            raise ArtifactError(f"invalid verified artifact path: {name}")
        if type(size) is not int or size < 0:
            raise ArtifactError(f"invalid verified artifact size: {name}")
        if type(digest) is not str or len(digest) != 64 or any(char not in _HEX64 for char in digest):
            raise ArtifactError(f"invalid verified artifact digest: {name}")
        candidate = Path(declared_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(ROOT):
            raise ArtifactError(f"verified artifact escapes project root: {name}")
        if not resolved.is_file():
            raise ArtifactError(f"missing verified artifact {name}: {resolved}")
        if resolved.stat().st_size != size:
            raise ArtifactError(f"verified artifact size is stale: {name}")
        checked[str(name)] = {"path": str(resolved.relative_to(ROOT)), "bytes": size, "sha256": digest}
    # Freeze the body before hashing: a shallow copy still shares the nested
    # dicts, and mutating them between the hash calculation and the write makes
    # the published self-hash unverifiable.  A JSON round-trip also normalises
    # integer dict keys to the string keys a reload produces.
    manifest: dict[str, object] = json.loads(
        json.dumps(dict(body), ensure_ascii=False, allow_nan=False)
    )
    manifest.setdefault("manifest_version", "artifact-manifest-v1")
    manifest["artifacts"] = checked
    manifest["manifest_hash"] = _canonical(manifest)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_json_exclusive(bound_path(path, "manifest output"), text, write_text=_write_text)
    return manifest


def write_json_exclusive(
    path: Path | str,
    payload: object,
    *,
    _write_text: Callable[[object, str], int] | None = None,
) -> None:
    """Publish one JSON file without replacing an existing name."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_json_exclusive(bound_path(path, "JSON output"), text, write_text=_write_text)


def verify_manifest(path: Path | str, *, expected_status: str | None = None) -> dict[str, object]:
    """Verify a published manifest and every file it claims to contain."""
    manifest_path = bound_path(path, "artifact manifest", must_exist=True)
    manifest = _load_json(manifest_path, "artifact manifest")
    if not isinstance(manifest, dict):
        raise ArtifactError("artifact manifest must be an object")
    declared = manifest.get("manifest_hash")
    if (
        type(declared) is not str
        or len(declared) != 64
        or any(char not in _HEX64 for char in declared)
    ):
        raise ArtifactError("artifact manifest hash must be a lowercase SHA256 string")
    actual = _canonical({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if declared != actual:
        raise ArtifactError(f"artifact manifest self-hash mismatch: {declared} != {actual}")
    if expected_status is not None and manifest.get("status") != expected_status:
        raise ArtifactError(f"artifact manifest status mismatch: {manifest.get('status')!r}")
    raw_artifacts = manifest.get("artifacts", {})
    if not isinstance(raw_artifacts, dict):
        raise ArtifactError("artifact manifest artifacts must be an object")
    checked: dict[str, dict[str, object]] = {}
    for name, raw in raw_artifacts.items():
        if not isinstance(raw, Mapping):
            raise ArtifactError(f"invalid artifact facts: {name}")
        artifact_path = bound_path(str(raw.get("path", "")), f"artifact {name}", must_exist=True)
        expected_bytes = raw.get("bytes")
        expected_sha = raw.get("sha256")
        if (
            type(expected_bytes) is not int
            or expected_bytes < 0
            or type(expected_sha) is not str
            or len(expected_sha) != 64
            or any(char not in _HEX64 for char in expected_sha)
        ):
            raise ArtifactError(f"invalid artifact facts: {name}")
        actual_bytes = artifact_path.stat().st_size
        actual_sha = sha256_file(artifact_path)
        if (actual_bytes, actual_sha) != (expected_bytes, expected_sha):
            raise ArtifactError(f"artifact changed: {name}")
        checked[str(name)] = {"path": str(artifact_path.relative_to(ROOT)), "bytes": actual_bytes, "sha256": actual_sha}
    manifest["verified_artifacts"] = checked
    return manifest
