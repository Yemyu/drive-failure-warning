"""Exact structural and typed numeric comparisons for run evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any


ABS_TOL = 1e-12
REL_TOL = 1e-10


class ComparisonError(ValueError):
    """A required comparison field is missing or has an invalid type."""


def scalar_equal(left: object, right: object, *, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    """Compare a scalar without converting integers to floating point.

    Counts, keys, booleans and nulls are exact.  The tolerance is reserved for
    two finite JSON/SQLite floating-point values.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) is int or type(right) is int:
        return type(left) is int and type(right) is int and left == right
    if type(left) is float or type(right) is float:
        if type(left) is not float or type(right) is not float:
            return False
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        return math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
    return left == right


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{label} must be an object")
    return value


def require_keys(
    value: object,
    required: Iterable[str],
    label: str,
    *,
    allowed: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    mapping = require_mapping(value, label)
    required_set = set(required)
    missing = sorted(required_set - set(mapping))
    if missing:
        raise ComparisonError(f"{label} missing required fields: {', '.join(missing)}")
    if allowed is not None:
        unknown = sorted(set(mapping) - set(allowed))
        if unknown:
            raise ComparisonError(f"{label} has unknown fields: {', '.join(unknown)}")
    return mapping


def require_sequence(value: object, label: str, *, length: int | None = None, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list):
        raise ComparisonError(f"{label} must be an array")
    if nonempty and not value:
        raise ComparisonError(f"{label} must not be empty")
    if length is not None and len(value) != length:
        raise ComparisonError(f"{label} length {len(value)} != {length}")
    return value


def require_finite_number(value: object, label: str) -> int | float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ComparisonError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ComparisonError(f"{label} must be finite")
    return value


def require_positive_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ComparisonError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ComparisonError(f"{label} must be finite and positive")
    return number


def numeric_summary(
    pairs: Iterable[tuple[object, object]],
    label: str,
    *,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
    require_nonempty: bool = True,
) -> dict[str, object]:
    """Summarise float comparisons while preserving exact integer semantics."""
    materialized = list(pairs)
    if require_nonempty and not materialized:
        raise ComparisonError(f"{label} has no comparable values")
    absolute: list[int | float] = []
    relative: list[float] = []
    outside = 0
    for index, (left, right) in enumerate(materialized):
        require_finite_number(left, f"{label}[{index}].left")
        require_finite_number(right, f"{label}[{index}].right")
        if type(left) is int and type(right) is int:
            integer_delta = abs(left - right)
            absolute.append(integer_delta)
            relative.append(integer_delta / max(abs(right), 1))
        else:
            left_float, right_float = float(left), float(right)
            absolute.append(abs(left_float - right_float))
            relative.append(abs(left_float - right_float) / max(abs(right_float), 1e-300))
        outside += int(not scalar_equal(left, right, abs_tol=abs_tol, rel_tol=rel_tol))
    return {
        "count": len(materialized),
        "max_abs": max(absolute, default=0.0),
        "max_rel": max(relative, default=0.0),
        "outside_tolerance": outside,
        "tolerances": {"abs": abs_tol, "rel": rel_tol},
    }


__all__ = [
    "ABS_TOL",
    "REL_TOL",
    "ComparisonError",
    "numeric_summary",
    "require_finite_number",
    "require_keys",
    "require_mapping",
    "require_positive_float",
    "require_sequence",
    "scalar_equal",
]
