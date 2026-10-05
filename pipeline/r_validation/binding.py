"""R0 binding: parameter approval from the source model database.

Independent read-only SQL cross-checks the frozen full-parameter JSON
(``examples/small_replay/current_lr.json``) against the four source tables in
``data/derived/simple_baseline_v2/model_results.sqlite``.  Any mismatch —
feature order, coefficient, imputation mean, standardization mean/scale,
observed count, intercept, feature count, run status, or the recorded source
database hash — is a hard refusal.  A metadata-only summary (payload with no
coefficients) can never pass because the check requires every coefficient
field to be present and equal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]


class BindingError(RuntimeError):
    """The model binding between source database and frozen parameters failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BindingError(f"{label} must be a number, got {value!r}")
    return float(value)


def approve_parameters(
    source_database: Path | str,
    full_parameters: Path | str,
    *,
    expected_source_sha256: str | None = None,
    expected_model: str = "current_lr",
    float_abs_tol: float = 1e-12,
    float_rel_tol: float = 1e-10,
) -> dict[str, Any]:
    """Approve the frozen parameters by field-level SQL cross-check.

    Returns the approved parameter document (the caller copies it into the R
    binding package).  Raises :class:`BindingError` on any mismatch.
    """
    source_database = Path(source_database)
    full_parameters = Path(full_parameters)
    if not source_database.is_file():
        raise BindingError(f"source model database is missing: {source_database}")
    if not full_parameters.is_file():
        raise BindingError(f"full parameter file is missing: {full_parameters}")

    actual_source_sha = sha256_file(source_database)
    if expected_source_sha256 and actual_source_sha != expected_source_sha256:
        raise BindingError(
            f"source database SHA differs: {actual_source_sha} != {expected_source_sha256}"
        )

    params = json.loads(full_parameters.read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise BindingError("full parameter file must be a JSON object")

    model = params.get("model")
    if model != expected_model:
        raise BindingError(f"parameter model is {model!r}, expected {expected_model!r}")

    status = params.get("model_run_status")
    if status != "reused_v1":
        raise BindingError(f"model_run_status is {status!r}, expected 'reused_v1' (no refit)")

    features = params.get("features")
    if not isinstance(features, list) or not features:
        raise BindingError("full parameters have no feature list")
    if len(features) != params.get("feature_count"):
        raise BindingError("feature_count does not match the feature list length")
    for i, f in enumerate(features):
        if f.get("position") != i:
            raise BindingError(f"feature position {i} out of order")
        if "coefficient" not in f:
            raise BindingError(
                f"feature {f.get('feature_name')} has no coefficient; "
                "a metadata-only summary can never be approved"
            )

    connection = sqlite3.connect(
        source_database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        try:
            connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise BindingError(f"source model database is unreadable: {exc}") from exc
        run = connection.execute(
            "SELECT model, status, feature_count, intercept FROM model_runs WHERE model = ?",
            (model,),
        ).fetchone()
        if run is None:
            raise BindingError(f"model_runs has no row for {model}")
        if run["status"] != status:
            raise BindingError(f"run status differs: {run['status']!r} != {status!r}")
        if run["feature_count"] != len(features):
            raise BindingError(
                f"source feature_count {run['feature_count']} != parameter {len(features)}"
            )
        intercept = _as_float(run["intercept"], "source intercept")
        param_intercept = _as_float(params.get("intercept"), "parameter intercept")
        if abs(intercept - param_intercept) > float_abs_tol + float_rel_tol * abs(intercept):
            raise BindingError(f"intercept differs: {intercept} != {param_intercept}")

        src_map = {
            row["position"]: dict(row)
            for row in connection.execute(
                "SELECT position, feature_name, source_name, is_missing_indicator "
                "FROM model_feature_map WHERE model = ? ORDER BY position",
                (model,),
            )
        }
        src_pre = {
            row["feature_name"]: dict(row)
            for row in connection.execute(
                "SELECT feature_name, imputation_mean, standardization_mean, "
                "standardization_scale, observed_count FROM preprocessing_stats WHERE model = ?",
                (model,),
            )
        }
        src_coef = {
            row["feature_name"]: float(row["coefficient"])
            for row in connection.execute(
                "SELECT feature_name, coefficient FROM model_coefficients WHERE model = ?",
                (model,),
            )
        }
    finally:
        connection.close()

    if len(src_map) != len(features):
        raise BindingError(f"source feature map has {len(src_map)} rows, expected {len(features)}")

    checked = 0
    for f in features:
        position = f["position"]
        name = f.get("feature_name")
        src = src_map.get(position)
        if src is None or src["feature_name"] != name:
            raise BindingError(f"feature map differs at position {position}")
        if bool(src["is_missing_indicator"]) != bool(f.get("is_missing_indicator")):
            raise BindingError(f"is_missing_indicator differs at position {position}")
        if src["source_name"] != f.get("source_name"):
            raise BindingError(f"source_name differs at position {position}")

        pre = src_pre.get(name)
        if pre is None:
            raise BindingError(f"preprocessing_stats missing for {name}")
        for key in ("imputation_mean", "standardization_mean", "standardization_scale"):
            a = _as_float(pre[key], f"source {name}.{key}")
            b = _as_float(f.get(key), f"parameter {name}.{key}")
            if abs(a - b) > float_abs_tol + float_rel_tol * abs(a):
                raise BindingError(f"{key} differs for {name}: {a} != {b}")
        if pre["observed_count"] != f.get("observed_count"):
            raise BindingError(f"observed_count differs for {name}")

        coef = _as_float(f.get("coefficient"), f"parameter {name}.coefficient")
        src_coef_value = src_coef.get(name)
        if src_coef_value is None:
            raise BindingError(f"model_coefficients missing for {name}")
        if abs(coef - src_coef_value) > float_abs_tol + float_rel_tol * abs(coef):
            raise BindingError(f"coefficient differs for {name}: {coef} != {src_coef_value}")
        checked += 1

    approved = json.loads(json.dumps(params, ensure_ascii=False, allow_nan=False))
    approved["approval"] = {
        "schema": "r0-parameter-approval-v1",
        "source_database": str(source_database),
        "source_database_sha256": actual_source_sha,
        "full_parameters_sha256": sha256_file(full_parameters),
        "checked_features": checked,
        "checked_tables": ["model_runs", "model_feature_map", "preprocessing_stats", "model_coefficients"],
    }
    return approved


def verify_approved_package(package_path: Path | str, *, source_database: Path | str) -> dict[str, Any]:
    """Re-run the field check against an already-copied approved package."""
    package_path = Path(package_path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    approval = payload.get("approval")
    if not isinstance(approval, dict):
        raise BindingError("approved package has no approval record")
    return approve_parameters(
        source_database,
        package_path,
        expected_source_sha256=approval.get("source_database_sha256"),
        expected_model=expected_model_of(payload),
    )


def expected_model_of(payload: Mapping[str, Any]) -> str:
    model = payload.get("model")
    if not isinstance(model, str):
        raise BindingError("approved package has no model")
    return model
