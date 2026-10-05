"""Bind real data access to a reviewed lock, source object and allowed stages.

New approval records use a neutral protocol-review role.  Version 1 is read
only for the registered, unchanged historical records below; it cannot be
used to create another approval.  A content digest checks record integrity,
not the identity of an approver or a cryptographic signature.  Callers must
also verify the implementation and input bindings in the approved lock.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

RELEASE_SCHEMA = "r-validation-release-v2"
APPROVAL_ROLE = "protocol-review"
LEGACY_RELEASE_SCHEMA = "r-validation-release-v1"
# Digests cover each complete historical body, including its original signer
# label, lock, source, stages, note and continuation fields.  This is a closed
# migration registry, not a list of permitted signer labels.  Do not extend it
# as a way to authorise a new run; new approvals must use version 2.
LEGACY_RELEASE_DIGESTS = frozenset({
    "37073fa28782d0423c13cb1f90fc176f53c5ee8ce4127685350f9a9d2c96ab61",  # initial Q4
    "8435a9a05c463d62d2a168733fcf91d2851e878d8f8cefaea3de2de593332ba3",  # Q4 continuation
    "92ac109bd95fa7428914aadb475f21bc5393b7a67b4e3a71d81ee9d2f01e81fc",  # Q1
})
ALLOWED_REAL_STAGES = frozenset({"panel", "score", "evaluate", "audit"})


class ReleaseError(RuntimeError):
    """The release gate refused real Q4 access."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_release(
    *,
    r0_lock_digest: str,
    source_object_id: str,
    allowed_stages: list[str],
    signed_by: str = APPROVAL_ROLE,
    note: str = "",
) -> dict[str, Any]:
    """Compose a version 2 approval record for the protocol-review role.

    The ``signed_by`` field name remains for consumers of the record format;
    its value identifies a workflow role, not an authenticated person.
    """
    if not allowed_stages:
        raise ReleaseError("a release must allow at least one stage")
    if signed_by != APPROVAL_ROLE:
        raise ReleaseError("release approval role must be protocol-review")
    if not set(allowed_stages).issubset(ALLOWED_REAL_STAGES):
        raise ReleaseError("release contains an unknown real stage")
    body = {
        "schema": RELEASE_SCHEMA,
        "r0_lock_digest": r0_lock_digest,
        "source_object_id": source_object_id,
        "allowed_stages": sorted(allowed_stages),
        "signed_by": signed_by,
        "note": note,
    }
    body["release_digest"] = _digest(body)
    return body


def require_release(
    release_path: Path | str,
    *,
    r0_lock_digest: str,
    source_object_id: str,
    stage: str,
) -> dict[str, Any]:
    """Gate a real stage on a valid, matching release file.

    Refuses when the file is missing, malformed, signed for a different lock
    digest or source object, or when the requested stage is not allowed.
    Historical version 1 records must match the closed migration registry.
    """
    release_path = Path(release_path)
    if not release_path.is_file():
        raise ReleaseError(
            f"no release file at {release_path}: real data access is not authorised yet"
        )
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"release file is unreadable: {exc}") from exc
    if (not isinstance(release, dict)
            or release.get("schema") not in (RELEASE_SCHEMA, LEGACY_RELEASE_SCHEMA)):
        raise ReleaseError("release file schema is invalid")
    body = {k: v for k, v in release.items() if k != "release_digest"}
    body_digest = _digest(body)
    if body_digest != release.get("release_digest"):
        raise ReleaseError("release file content changed after approval")
    if release["schema"] == LEGACY_RELEASE_SCHEMA:
        if body_digest not in LEGACY_RELEASE_DIGESTS:
            raise ReleaseError("version 1 release is not a registered historical approval")
    elif release.get("signed_by") != APPROVAL_ROLE:
        raise ReleaseError("release approval role must be protocol-review")
    if release.get("r0_lock_digest") != r0_lock_digest:
        raise ReleaseError(
            f"release was signed for lock {release.get('r0_lock_digest')}, "
            f"current lock is {r0_lock_digest}"
        )
    if release.get("source_object_id") != source_object_id:
        raise ReleaseError(
            f"release was signed for source object {release.get('source_object_id')!r}, "
            f"current is {source_object_id!r}"
        )
    allowed = release.get("allowed_stages")
    if (not isinstance(allowed, list)
            or not set(allowed).issubset(ALLOWED_REAL_STAGES)
            or stage not in allowed):
        raise ReleaseError(f"stage {stage!r} is not allowed by this release")
    return release
