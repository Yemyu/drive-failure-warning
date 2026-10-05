"""Independent training-matrix package and pre-fit verification.

This module intentionally does not call the production feature, label, or
sampling functions.  It re-reads the two private stage databases and rebuilds
the expected sample using the locked protocol before a worker is allowed to
fit an estimator.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .artifacts import ArtifactError, artifact_facts, bound_path, write_json_exclusive
from .contract import LOCKED_FEATURES
from .runtime_context import project_root


PACKAGE_VERSION = "training-package-v1"
APPROVAL_VERSION = "training-check-v1"
REFERENCE_APPROVAL_VERSION = "training-check-v2"
ABS_TOL = 1e-12
REL_TOL = 1e-10
KNOWN_STATUSES = {"positive_observed", "positive_with_gap", "negative_observed"}
ROOT = project_root()
_BASE_APPROVAL_KEYS = {
    "status", "version", "run_id", "rows", "positive_rows", "negative_rows",
    "columns", "checks", "package", "package_manifest_sha256",
}
_OPTIONAL_APPROVAL_KEYS = {"inputs", "package_manifest_fact", "panel_reference"}
_APPROVAL_CHECK_KEYS = {
    "package_internal_facts", "keys", "labels", "weights",
    "raw_values_and_missingness", "preprocessing", "transformed_matrix",
    "sampling",
}


class TrainingCheckError(RuntimeError):
    """The prepared training input was malformed, stale, or inconsistent."""


def _raise(message: str) -> None:
    raise TrainingCheckError(message)


def _fact(path: Path | str, label: str) -> dict[str, object]:
    try:
        return dict(artifact_facts(path))
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc


def _strict_fact(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        _raise(f"{label} artifact facts are invalid")
    path, size, digest = value.get("path"), value.get("bytes"), value.get("sha256")
    if type(path) is not str or not path or type(size) is not int or size < 0:
        _raise(f"{label} artifact facts are invalid")
    if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(c not in "0123456789abcdef" for c in digest):
        _raise(f"{label} artifact facts are invalid")
    return {"path": path, "bytes": size, "sha256": digest}


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except (OSError, ValueError) as exc:
        raise TrainingCheckError(f"cannot read {label}: {path}") from exc


def _memory_fact(path: Path, data: bytes, label: str) -> dict[str, object]:
    try:
        bound = bound_path(path, label, must_exist=False)
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    return {
        "path": str(bound.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _json_reject_constant(value: str) -> object:
    _raise(f"JSON constant is not allowed: {value}")


def _json_pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _raise(f"duplicate JSON key in training package: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_pairs_no_duplicates,
            parse_constant=_json_reject_constant,
        )
    except TrainingCheckError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingCheckError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        _raise(f"{label} must be a JSON object")
    return payload


def _load_array_bytes(data: bytes, name: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise TrainingCheckError(f"cannot load training package {name}") from exc
    if not isinstance(value, np.ndarray) or value.dtype == object:
        _raise(f"training package {name} is invalid")
    return value


def _publish_array(path: Path, array: np.ndarray) -> None:
    if path.exists():
        _raise(f"training package output already exists: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        _raise(f"training package partial already exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _array(value: object, name: str, *, dtype: np.dtype | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TrainingCheckError(f"training {name} array is invalid") from exc
    if result.dtype == object:
        _raise(f"training {name} array must not use object dtype")
    return result


def write_training_package(package_dir: Path | str, prepared: Mapping[str, object]) -> dict[str, object]:
    """Write a fresh, atomically published package for a prepared matrix."""
    try:
        directory = bound_path(package_dir, "training package")
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    if directory.exists():
        _raise(f"training package directory already exists: {directory}")
    columns = prepared.get("columns")
    if columns != list(LOCKED_FEATURES):
        _raise("training package columns differ from the locked whitelist")
    required = ("raw", "labels", "weights", "transformed", "keys", "preprocessing", "sampling", "run_id")
    missing = [key for key in required if key not in prepared]
    if missing:
        _raise(f"prepared training input is missing: {', '.join(missing)}")
    raw = _array(prepared["raw"], "raw", dtype=np.float64)
    labels = _array(prepared["labels"], "labels", dtype=np.int8)
    weights = _array(prepared["weights"], "weights", dtype=np.float64)
    transformed = _array(prepared["transformed"], "transformed", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(LOCKED_FEATURES):
        _raise("training raw matrix has the wrong shape")
    rows = raw.shape[0]
    if labels.shape != (rows,) or weights.shape != (rows,) or transformed.shape != raw.shape:
        _raise("training package array shapes are inconsistent")
    if len(labels) == 0 or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isfinite(transformed).all():
        _raise("training package arrays contain invalid values")
    if set(int(value) for value in labels.tolist()) != {0, 1}:
        _raise("training package labels must contain both classes")
    keys = prepared["keys"]
    if not isinstance(keys, Sequence) or len(keys) != rows:
        _raise("training package keys do not match row count")
    clean_keys: list[list[str]] = []
    for index, key in enumerate(keys):
        if not isinstance(key, Sequence) or len(key) != 2 or any(type(value) is not str or not value for value in key):
            _raise(f"training package key {index} is invalid")
        clean_keys.append([str(key[0]), str(key[1])])
    preprocessing = prepared["preprocessing"]
    if not isinstance(preprocessing, Mapping):
        _raise("training preprocessing is not an object")
    prep = dict(preprocessing)
    for field in ("imputation_mean", "standardization_mean", "standardization_scale"):
        values = prep.get(field)
        if not isinstance(values, Sequence) or len(values) != len(LOCKED_FEATURES):
            _raise(f"training preprocessing.{field} has the wrong dimension")
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
            _raise(f"training preprocessing.{field} contains a non-finite value")
    if any(float(value) <= 0 for value in prep["standardization_scale"]):
        _raise("training preprocessing scales must be positive")
    sampling = prepared["sampling"]
    if not isinstance(sampling, list):
        _raise("training sampling statistics must be a list")
    directory.mkdir(parents=True)
    paths = {
        "raw": directory / "raw.npy",
        "raw_missing_mask": directory / "raw_missing_mask.npy",
        "labels": directory / "labels.npy",
        "weights": directory / "weights.npy",
        "transformed": directory / "transformed.npy",
        "keys": directory / "keys.json",
        "preprocessing": directory / "preprocessing.json",
    }
    _publish_array(paths["raw"], raw)
    _publish_array(paths["raw_missing_mask"], (~np.isfinite(raw)).astype(np.bool_))
    _publish_array(paths["labels"], labels)
    _publish_array(paths["weights"], weights)
    _publish_array(paths["transformed"], transformed)
    write_json_exclusive(paths["keys"], {"keys": clean_keys})
    write_json_exclusive(paths["preprocessing"], {"preprocessing": prep})
    files = {name: _fact(path, f"training package {name}") for name, path in sorted(paths.items())}
    package = {
        "version": PACKAGE_VERSION,
        "run_id": str(prepared["run_id"]),
        "columns": list(LOCKED_FEATURES),
        "rows": rows,
        "positive_rows": int((labels == 1).sum()),
        "negative_rows": int((labels == 0).sum()),
        "files": files,
        "sampling": sampling,
    }
    write_json_exclusive(directory / "package.json", package)
    return package


def load_training_package(
    package_dir: Path | str,
    *,
    expected_package_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Load and decode each package artifact from the bytes that were hashed."""
    try:
        directory = bound_path(package_dir, "training package", must_exist=False)
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    package_path = directory / "package.json"
    package_bytes = _read_bytes(package_path, "training package manifest")
    package_digest = hashlib.sha256(package_bytes).hexdigest()
    if expected_package_manifest_sha256 is not None and package_digest != expected_package_manifest_sha256:
        _raise("training package manifest changed after approval")
    package = _load_json_bytes(package_bytes, "training package manifest")
    expected_keys = {"version", "run_id", "columns", "rows", "positive_rows", "negative_rows", "files", "sampling"}
    if set(package) != expected_keys or package.get("version") != PACKAGE_VERSION:
        _raise("training package manifest keys or version are invalid")
    if package.get("columns") != list(LOCKED_FEATURES):
        _raise("training package feature order is invalid")
    rows = package.get("rows")
    if type(rows) is not int or rows <= 0:
        _raise("training package row count is invalid")
    if any(type(package.get(field)) is not int or package[field] < 0 for field in ("positive_rows", "negative_rows")):
        _raise("training package class counts are invalid")
    if package["positive_rows"] + package["negative_rows"] != rows:
        _raise("training package class counts do not sum to rows")
    files = package.get("files")
    expected_files = {"raw", "raw_missing_mask", "labels", "weights", "transformed", "keys", "preprocessing"}
    if not isinstance(files, Mapping) or set(files) != expected_files:
        _raise("training package file list is invalid")
    contents: dict[str, bytes] = {}
    for name, raw_fact in files.items():
        fact = _strict_fact(raw_fact, f"training package {name}")
        try:
            target = bound_path(fact["path"], f"training package {name}", must_exist=True)
        except ArtifactError as exc:
            raise TrainingCheckError(str(exc)) from exc
        data = _read_bytes(target, f"training package {name}")
        if _memory_fact(target, data, name) != fact:
            _raise(f"training package artifact changed: {name}")
        contents[name] = data
    raw = _load_array_bytes(contents["raw"], "raw")
    missing_mask = _load_array_bytes(contents["raw_missing_mask"], "raw_missing_mask")
    labels = _load_array_bytes(contents["labels"], "labels")
    weights = _load_array_bytes(contents["weights"], "weights")
    transformed = _load_array_bytes(contents["transformed"], "transformed")
    if raw.dtype != np.float64 or labels.dtype != np.int8 or weights.dtype != np.float64 or transformed.dtype != np.float64 or missing_mask.dtype != np.bool_:
        _raise("training package array dtypes are invalid")
    if raw.shape != (rows, len(LOCKED_FEATURES)) or transformed.shape != raw.shape or missing_mask.shape != raw.shape or labels.shape != (rows,) or weights.shape != (rows,):
        _raise("training package array shapes are invalid")
    for value in (raw, missing_mask, labels, weights, transformed):
        value.setflags(write=False)
    keys_payload = _load_json_bytes(contents["keys"], "training package keys")
    prep_payload = _load_json_bytes(contents["preprocessing"], "training package preprocessing")
    if set(keys_payload) != {"keys"} or set(prep_payload) != {"preprocessing"}:
        _raise("training package JSON files are invalid")
    keys = keys_payload["keys"]
    prep = prep_payload["preprocessing"]
    if not isinstance(keys, list) or len(keys) != rows or not isinstance(prep, Mapping):
        _raise("training package JSON dimensions are invalid")
    return {
        "package": package,
        "package_manifest_sha256": package_digest,
        "package_manifest_fact": _memory_fact(package_path, package_bytes, "training package manifest"),
        "raw": raw,
        "raw_missing_mask": missing_mask,
        "labels": labels,
        "weights": weights,
        "transformed": transformed,
        "keys": [tuple(item) for item in keys],
        "preprocessing": dict(prep),
        "sampling": list(package["sampling"]),
        "columns": list(LOCKED_FEATURES),
        "run_id": str(package["run_id"]),
    }


def load_approved_training_package(
    package_dir: Path | str,
    approval: Mapping[str, object],
    *,
    expected_inputs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Consume the exact package that an independent check approved.

    ``verify_training_package`` performs the independent reconstruction.  This
    function is the fit boundary: it validates the approval record and all
    bound input facts, then loads the package only when its manifest still has
    the approved digest.  A package changed between those two operations is
    therefore rejected before estimator construction.
    """
    if not isinstance(approval, Mapping):
        _raise("training approval is not an object")
    actual_keys = set(approval)
    if not _BASE_APPROVAL_KEYS.issubset(actual_keys) or actual_keys - (_BASE_APPROVAL_KEYS | _OPTIONAL_APPROVAL_KEYS):
        _raise("training approval keys are invalid")
    version = approval.get("version")
    if approval.get("status") != "ready_for_fit" or version not in {APPROVAL_VERSION, REFERENCE_APPROVAL_VERSION}:
        _raise("training approval is not ready_for_fit")
    run_id = approval.get("run_id")
    if type(run_id) is not str or not run_id:
        _raise("training approval run_id is invalid")
    rows = approval.get("rows")
    positives = approval.get("positive_rows")
    negatives = approval.get("negative_rows")
    if type(rows) is not int or rows <= 0 or type(positives) is not int or positives <= 0 or type(negatives) is not int or negatives <= 0 or positives + negatives != rows:
        _raise("training approval class counts are invalid")
    if approval.get("columns") != list(LOCKED_FEATURES):
        _raise("training approval feature order is invalid")
    checks = approval.get("checks")
    expected_checks = _APPROVAL_CHECK_KEYS | ({"panel_reference"} if version == REFERENCE_APPROVAL_VERSION else set())
    if not isinstance(checks, Mapping) or set(checks) != expected_checks or any(value is not True for value in checks.values()):
        _raise("training approval checks are incomplete")
    panel_reference = approval.get("panel_reference")
    if version == REFERENCE_APPROVAL_VERSION:
        if not isinstance(panel_reference, Mapping) or panel_reference.get("status") != "pass" or panel_reference.get("version") != "panel-reference-v1":
            _raise("training approval is missing a passing panel reference")
    elif panel_reference is not None:
        _raise("training-check-v1 approval must not contain a panel reference")
    try:
        directory = bound_path(package_dir, "approved training package", must_exist=False)
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    if not directory.is_dir():
        _raise(f"approved training package directory is missing: {directory}")
    package_fact = approval.get("package")
    if type(package_fact) is not str or not package_fact:
        _raise("training approval package path is invalid")
    try:
        approved_package_path = bound_path(package_fact, "approved training package", must_exist=False)
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    if approved_package_path != directory:
        _raise("training approval package path differs from worker package")
    digest = approval.get("package_manifest_sha256")
    if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(char not in "0123456789abcdef" for char in digest):
        _raise("training approval package manifest digest is invalid")
    raw_package_fact = approval.get("package_manifest_fact")
    if raw_package_fact is not None:
        package_fact_record = _strict_fact(raw_package_fact, "training approval package manifest")
        expected_package_path = directory / "package.json"
        try:
            bound_package_path = bound_path(package_fact_record["path"], "training approval package manifest", must_exist=False)
        except ArtifactError as exc:
            raise TrainingCheckError(str(exc)) from exc
        if bound_package_path != expected_package_path or package_fact_record["sha256"] != digest:
            _raise("training approval package manifest fact differs from digest or package path")
    raw_inputs = approval.get("inputs")
    if expected_inputs is not None:
        if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(expected_inputs):
            _raise("training approval input facts are incomplete")
        for name, expected in expected_inputs.items():
            if type(name) is not str or not name:
                _raise("training approval input name is invalid")
            expected_fact = _strict_fact(expected, f"expected training input {name}")
            approved_fact = _strict_fact(raw_inputs.get(name), f"training approval input {name}")
            if approved_fact != expected_fact:
                _raise(f"training approval input differs: {name}")
            if _fact(expected_fact["path"], f"training input {name}") != expected_fact:
                _raise(f"training input changed after approval: {name}")
    elif raw_inputs is not None:
        if not isinstance(raw_inputs, Mapping) or any(type(name) is not str or not name for name in raw_inputs):
            _raise("training approval input facts are invalid")
        for name, raw_fact in raw_inputs.items():
            fact = _strict_fact(raw_fact, f"training approval input {name}")
            if _fact(fact["path"], f"training input {name}") != fact:
                _raise(f"training input changed after approval: {name}")

    package = load_training_package(
        directory,
        expected_package_manifest_sha256=digest,
    )
    if raw_package_fact is not None and package.get("package_manifest_fact") != package_fact_record:
        _raise("loaded package manifest differs from approved bytes")
    manifest = package.get("package")
    if not isinstance(manifest, Mapping):
        _raise("loaded training package manifest is invalid")
    if str(package.get("run_id")) != run_id or package.get("columns") != list(LOCKED_FEATURES):
        _raise("training package identity differs from approval")
    if int(manifest.get("rows", -1)) != rows or int(manifest.get("positive_rows", -1)) != positives or int(manifest.get("negative_rows", -1)) != negatives:
        _raise("training package counts differ from approval")
    return package


def _sampling_hash(date_text: str, serial: str) -> str:
    return hashlib.sha256(f"fit-v1|20260913|{date_text}|{serial}".encode()).hexdigest()


def _value(row: sqlite3.Row, name: str) -> float:
    value = row[name]
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingCheckError(f"training feature {name} is not numeric") from exc
    if not math.isfinite(number):
        raise TrainingCheckError(f"training feature {name} is not finite")
    return number


def _preprocess(raw: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    valid = np.isfinite(raw)
    total = float(weights.sum())
    if raw.ndim != 2 or len(raw) != len(weights) or total <= 0:
        _raise("independent training matrix or weights are invalid")
    means = np.zeros(raw.shape[1], dtype=np.float64)
    observed = valid.sum(axis=0).astype(int)
    for index in range(raw.shape[1]):
        if int(observed[index]):
            means[index] = np.sum(weights[valid[:, index]] * raw[valid[:, index], index]) / np.sum(weights[valid[:, index]])
    filled = np.where(valid, raw, means)
    center = np.sum(filled * weights[:, None], axis=0) / total
    variance = np.sum(((filled - center) ** 2) * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    transformed = (filled - center) / scale
    if not np.isfinite(transformed).all():
        _raise("independent preprocessing produced non-finite values")
    return transformed, {
        "imputation_mean": means.tolist(),
        "standardization_mean": center.tolist(),
        "standardization_scale": scale.tolist(),
        "observed_count": observed.tolist(),
        "weighted_total": total,
    }


def _reconstruct(feature_db: Path, label_db: Path, run_id: str) -> dict[str, object]:
    try:
        feature_conn = sqlite3.connect(feature_db.as_uri() + "?mode=ro&immutable=1", uri=True)
        label_conn = sqlite3.connect(label_db.as_uri() + "?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as exc:
        raise TrainingCheckError("cannot open training databases") from exc
    feature_conn.row_factory = sqlite3.Row
    label_conn.row_factory = sqlite3.Row
    try:
        feature_columns = {str(row[1]) for row in feature_conn.execute("PRAGMA table_info(feature_rows)")}
        if not set(LOCKED_FEATURES).issubset(feature_columns):
            _raise("feature database is missing locked columns")
        matrix: list[list[float]] = []
        labels: list[int] = []
        weights: list[float] = []
        keys: list[tuple[str, str]] = []
        sampling: list[dict[str, object]] = []
        any_rows = False
        rows = feature_conn.execute("SELECT * FROM feature_rows ORDER BY decision_date,serial_number")
        for date_text, date_iter in itertools.groupby(rows, key=lambda row: str(row["decision_date"])):
            any_rows = True
            date_rows = list(date_iter)
            label_rows = label_conn.execute(
                "SELECT * FROM label_flow WHERE run_id=? AND decision_date=? AND eligible=1 ORDER BY serial_number",
                (run_id, date_text),
            ).fetchall()
            by_serial = {str(row["serial_number"]): row for row in label_rows}
            if {str(row["serial_number"]) for row in date_rows} != set(by_serial):
                _raise(f"feature/label eligibility differs on {date_text}")
            negatives = [row for row in date_rows if str(by_serial[str(row["serial_number"])] ["status"]) == "negative_observed"]
            positives = [row for row in date_rows if str(by_serial[str(row["serial_number"])] ["status"]).startswith("positive")]
            ranked = sorted(((_sampling_hash(date_text, str(row["serial_number"])), str(row["serial_number"]), row) for row in negatives), key=lambda item: (item[0], item[1]))
            sample_count = min(len(ranked), max(1, len(ranked) // 20)) if ranked else 0
            selected = {serial for _digest, serial, _row in ranked[:sample_count]}
            inverse = len(ranked) / sample_count if sample_count else 0.0
            for row in date_rows:
                serial = str(row["serial_number"])
                label = by_serial.get(serial)
                if label is None or str(label["status"]) not in KNOWN_STATUSES:
                    continue
                target = int(label["label"])
                if target == 0 and serial not in selected:
                    continue
                matrix.append([_value(row, name) for name in LOCKED_FEATURES])
                labels.append(target)
                weights.append(1.0 if target == 1 else inverse)
                keys.append((date_text, serial))
            sampling.append({"date": date_text, "eligible": len(date_rows), "positive": len(positives), "negative": len(negatives), "sampled_negative": sample_count, "inverse_weight": inverse})
        if not any_rows:
            _raise("no feature rows for independent training check")
        raw = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int8)
        sample_weights = np.asarray(weights, dtype=np.float64)
        if set(target.tolist()) != {0, 1}:
            _raise("independent training sample must contain both classes")
        sample_weights /= sample_weights.mean()
        transformed, prep = _preprocess(raw, sample_weights)
        return {"raw": raw, "labels": target, "weights": sample_weights, "keys": keys, "transformed": transformed, "preprocessing": prep, "sampling": sampling}
    finally:
        feature_conn.close()
        label_conn.close()


def _same_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        _raise(f"training {label} shape or dtype differs from independent reconstruction")
    if actual.dtype.kind == "f":
        if not np.array_equal(actual, expected, equal_nan=True):
            if not np.allclose(actual, expected, atol=ABS_TOL, rtol=REL_TOL, equal_nan=True):
                _raise(f"training {label} differs from independent reconstruction")
    elif not np.array_equal(actual, expected):
        _raise(f"training {label} differs from independent reconstruction")


def verify_training_package(
    feature_db: Path | str,
    label_db: Path | str,
    run_id: str,
    package_dir: Path | str,
    *,
    progress_callback: Callable[[], int] | None = None,
    reference_check: Mapping[str, object] | None = None,
    expected_inputs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Reconstruct and compare every package component before fitting."""
    try:
        feature_path = bound_path(feature_db, "training feature database", must_exist=True)
        label_path = bound_path(label_db, "training label database", must_exist=True)
    except ArtifactError as exc:
        raise TrainingCheckError(str(exc)) from exc
    normalised_inputs: dict[str, dict[str, object]] | None = None
    if expected_inputs is not None:
        if not isinstance(expected_inputs, Mapping) or "package_manifest" not in expected_inputs:
            _raise("expected training inputs must include package_manifest")
        normalised_inputs = {}
        for name, raw_fact in expected_inputs.items():
            if type(name) is not str or not name:
                _raise("expected training input name is invalid")
            normalised_inputs[name] = _strict_fact(raw_fact, f"expected training input {name}")
            if _fact(normalised_inputs[name]["path"], f"expected training input {name}") != normalised_inputs[name]:
                _raise(f"training input changed before independent check: {name}")
    package = load_training_package(package_dir)
    if normalised_inputs is not None:
        package_fact = package.get("package_manifest_fact")
        if package_fact != normalised_inputs["package_manifest"]:
            _raise("training package manifest changed before independent check")
    if str(package["run_id"]) != str(run_id):
        _raise("training package run_id differs from worker run_id")
    if progress_callback is not None and progress_callback():
        _raise("training package check cancelled")
    expected = _reconstruct(feature_path, label_path, str(run_id))
    _same_array(package["raw"], expected["raw"], "raw matrix")
    _same_array(package["raw_missing_mask"], (~np.isfinite(expected["raw"])).astype(np.bool_), "missing mask")
    _same_array(package["labels"], expected["labels"], "labels")
    _same_array(package["weights"], expected["weights"], "weights")
    _same_array(package["transformed"], expected["transformed"], "transformed matrix")
    if list(package["keys"]) != list(expected["keys"]):
        _raise("training keys differ from independent reconstruction")
    if package["preprocessing"] != expected["preprocessing"]:
        # JSON numbers can be represented with an insignificant formatting
        # difference; compare each field numerically while keeping dimensions
        # and observed counts exact.
        for field in ("imputation_mean", "standardization_mean", "standardization_scale"):
            actual = np.asarray(package["preprocessing"].get(field), dtype=np.float64)
            want = np.asarray(expected["preprocessing"].get(field), dtype=np.float64)
            _same_array(actual, want, f"preprocessing.{field}")
        if package["preprocessing"].get("observed_count") != expected["preprocessing"].get("observed_count"):
            _raise("training preprocessing observed counts differ")
        if not math.isclose(float(package["preprocessing"].get("weighted_total")), float(expected["preprocessing"].get("weighted_total")), rel_tol=REL_TOL, abs_tol=ABS_TOL):
            _raise("training preprocessing weighted total differs")
    if package["sampling"] != expected["sampling"]:
        _raise("training sampling statistics differ from independent reconstruction")
    if reference_check is not None:
        if not isinstance(reference_check, Mapping) or reference_check.get("status") != "pass" or reference_check.get("version") != "panel-reference-v1":
            _raise("panel reference check is not passing")
    if normalised_inputs is not None:
        for name, expected_fact in normalised_inputs.items():
            if name == "package_manifest":
                current_fact = _fact(expected_fact["path"], "training package manifest")
                if current_fact != package.get("package_manifest_fact"):
                    _raise("training package manifest changed during independent check")
            elif _fact(expected_fact["path"], f"training input {name}") != expected_fact:
                _raise(f"training input changed during independent check: {name}")
    labels = package["labels"]
    package_path = bound_path(package_dir, "training package")
    approval_checks = {
        "package_internal_facts": True,
        "keys": True,
        "labels": True,
        "weights": True,
        "raw_values_and_missingness": True,
        "preprocessing": True,
        "transformed_matrix": True,
        "sampling": True,
    }
    approval = {
        "status": "ready_for_fit",
        "version": REFERENCE_APPROVAL_VERSION if reference_check is not None else APPROVAL_VERSION,
        "run_id": str(run_id),
        "rows": int(len(labels)),
        "positive_rows": int((labels == 1).sum()),
        "negative_rows": int((labels == 0).sum()),
        "columns": list(LOCKED_FEATURES),
        "checks": approval_checks,
        "package": str(package_path),
        "package_manifest_sha256": package["package_manifest_sha256"],
        "package_manifest_fact": package["package_manifest_fact"],
    }
    if reference_check is not None:
        approval["checks"] = {**approval_checks, "panel_reference": True}
        approval["panel_reference"] = dict(reference_check)
    if normalised_inputs is not None:
        approval["inputs"] = normalised_inputs
    return approval


__all__ = [
    "ABS_TOL", "APPROVAL_VERSION", "REFERENCE_APPROVAL_VERSION", "PACKAGE_VERSION", "TrainingCheckError",
    "load_approved_training_package", "load_training_package", "verify_training_package",
    "write_training_package",
]
