"""Bounded roots for code execution and project artifacts.

The normal checkout uses one root for both values. A supervised run executes
from a read-only source snapshot, so the code root and the project root are
intentionally separate.
"""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_CODE_ROOT = Path(__file__).resolve().parents[2]


def code_root() -> Path:
    value = os.environ.get("REPRO_CODE_ROOT")
    return Path(value).expanduser().resolve() if value else _DEFAULT_CODE_ROOT


def project_root() -> Path:
    value = os.environ.get("REPRO_PROJECT_ROOT")
    return Path(value).expanduser().resolve() if value else _DEFAULT_CODE_ROOT


def validate_context(
    *,
    expected_code_root: Path | str | None = None,
    expected_project_root: Path | str | None = None,
) -> tuple[Path, Path]:
    code = code_root()
    project = project_root()
    if not code.is_dir() or not project.is_dir():
        raise RuntimeError("runtime code/project root is not a directory")
    if expected_code_root is not None and code != Path(expected_code_root).expanduser().resolve():
        raise RuntimeError("runtime code root differs from supervisor request")
    if expected_project_root is not None and project != Path(expected_project_root).expanduser().resolve():
        raise RuntimeError("runtime project root differs from supervisor request")
    return code, project


__all__ = ["code_root", "project_root", "validate_context"]
