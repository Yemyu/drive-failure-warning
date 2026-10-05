"""Locked configuration and stage-boundary validation for real runs.

The small synthetic fixtures used by the historical unit tests may continue to
use a compact configuration.  A panel published by the real panel builder is
always required to use the locked configuration below; this prevents a real
run from merely recording a configuration hash while continuing to use module
constants or command-line dates.
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import json
from pathlib import Path
import sqlite3
from typing import Any

from .artifacts import ArtifactError, bound_path, sha256_file


class ContractError(RuntimeError):
    """A locked configuration, date, or panel contract is invalid."""


LOCKED_PROFILE = "backblaze_2023_st4000dm000"
LOCKED_VERSION = "repro-current-v1"
LOCKED_MODEL = "ST4000DM000"
LOCKED_FEATURES = (
    "smart_5_current_log1p", "smart_5_current_missing", "smart_5_current_nonzero",
    "smart_9_current_log1p", "smart_9_current_missing",
    "smart_187_current_log1p", "smart_187_current_missing", "smart_187_current_nonzero",
    "smart_188_current_nonzero", "smart_188_current_missing",
    "smart_197_current_log1p", "smart_197_current_missing", "smart_197_current_nonzero",
    "smart_198_current_log1p", "smart_198_current_missing", "smart_198_current_nonzero",
)
LOCKED_DATES = {
    "train_start": "2023-01-15",
    "train_end": "2023-06-23",
    "train_cutoff": "2023-06-30",
    "train_event_start": "2023-01-22",
    "train_event_end": "2023-06-24",
    "eval_start": "2023-07-01",
    "eval_end": "2023-09-23",
    "eval_cutoff": "2023-09-30",
    "event_start": "2023-07-08",
    "event_end": "2023-09-24",
}


def _reject_constant(value: str) -> Any:
    raise ContractError(f"JSON constant is not allowed: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path | str, label: str) -> dict[str, Any]:
    target = bound_path(path, label, must_exist=True)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid {label}: {target}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object")
    return payload


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unknown:
            detail.append("unknown=" + ",".join(unknown))
        raise ContractError(f"{label} keys invalid ({'; '.join(detail)})")


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ContractError(f"{label} must be an integer")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be boolean")
    return value


def _strict_float(value: Any, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ContractError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{label} must be finite")
    return number


def validate_locked_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every field of the real profile and return a plain copy."""
    expected_top = {"version", "profile", "task", "dates", "features", "sampling", "preprocessing", "estimator", "selection", "limits", "comparison"}
    _exact_keys(payload, expected_top, "locked configuration")
    if payload["version"] != LOCKED_VERSION or payload["profile"] != LOCKED_PROFILE:
        raise ContractError("configuration version or profile is not the locked real profile")

    task = payload["task"]
    if not isinstance(task, Mapping):
        raise ContractError("task must be an object")
    _exact_keys(task, {"model", "horizon_days", "history_days", "min_history", "event", "unknown_policy", "positive_with_gap", "smart_decline"}, "task")
    if task["model"] != LOCKED_MODEL or _strict_int(task["horizon_days"], "task.horizon_days") != 7 or _strict_int(task["history_days"], "task.history_days") != 14 or _strict_int(task["min_history"], "task.min_history") != 12:
        raise ContractError("task values differ from locked H7 protocol")
    if task["event"] != "first_observed_failure" or task["unknown_policy"] != "exclude_from_fit_keep_in_scoring" or task["positive_with_gap"] != "include" or task["smart_decline"] != "retain_raw_values_audit_as_of":
        raise ContractError("task policy differs from locked protocol")

    dates = payload["dates"]
    if not isinstance(dates, Mapping):
        raise ContractError("dates must be an object")
    _exact_keys(dates, set(LOCKED_DATES), "dates")
    for key, value in LOCKED_DATES.items():
        if type(dates[key]) is not str or dates[key] != value:
            raise ContractError(f"dates.{key} differs from locked protocol")

    features = payload["features"]
    if features != list(LOCKED_FEATURES):
        raise ContractError("feature order differs from locked current_lr protocol")

    sampling = payload["sampling"]
    if not isinstance(sampling, Mapping):
        raise ContractError("sampling must be an object")
    _exact_keys(sampling, {"negative_divisor", "negative_minimum_if_nonempty", "all_positives", "sha256_prefix", "weights"}, "sampling")
    if _strict_int(sampling["negative_divisor"], "sampling.negative_divisor") != 20 or _strict_int(sampling["negative_minimum_if_nonempty"], "sampling.negative_minimum_if_nonempty") != 1 or _strict_bool(sampling["all_positives"], "sampling.all_positives") is not True or sampling["sha256_prefix"] != "fit-v1|20260913" or sampling["weights"] != "inverse_probability_normalized_mean_one":
        raise ContractError("sampling differs from locked protocol")

    preprocessing = payload["preprocessing"]
    if not isinstance(preprocessing, Mapping):
        raise ContractError("preprocessing must be an object")
    _exact_keys(preprocessing, {"fit_population", "imputation", "all_missing_mean", "variance", "minimum_scale", "fallback_scale", "dtype"}, "preprocessing")
    if preprocessing["fit_population"] != "weighted_training_sample" or preprocessing["imputation"] != "weighted_observed_mean" or _strict_float(preprocessing["all_missing_mean"], "preprocessing.all_missing_mean") != 0.0 or preprocessing["variance"] != "weighted_population" or _strict_float(preprocessing["minimum_scale"], "preprocessing.minimum_scale") != 1e-12 or _strict_float(preprocessing["fallback_scale"], "preprocessing.fallback_scale") != 1.0 or preprocessing["dtype"] != "float64":
        raise ContractError("preprocessing differs from locked protocol")

    estimator = payload["estimator"]
    if not isinstance(estimator, Mapping):
        raise ContractError("estimator must be an object")
    _exact_keys(estimator, {"model", "penalty", "solver", "C", "tol", "max_iter", "fit_intercept", "class_weight", "random_state"}, "estimator")
    if estimator["model"] != "current_lr" or estimator["penalty"] != "l2" or estimator["solver"] != "lbfgs" or _strict_float(estimator["C"], "estimator.C") != 1.0 or _strict_float(estimator["tol"], "estimator.tol") != 1e-6 or _strict_int(estimator["max_iter"], "estimator.max_iter") != 1000 or _strict_bool(estimator["fit_intercept"], "estimator.fit_intercept") is not True or estimator["class_weight"] is not None or _strict_int(estimator["random_state"], "estimator.random_state") != 20260913:
        raise ContractError("estimator differs from locked protocol")

    selection = payload["selection"]
    if not isinstance(selection, Mapping):
        raise ContractError("selection must be an object")
    _exact_keys(selection, {"methods", "budget_denominator", "cooldown_days", "tie_prefix", "reset_cooldown_at_eval_start", "rule_requires_positive_signal"}, "selection")
    if selection["methods"] != ["current_lr", "smart_nonzero"] or _strict_int(selection["budget_denominator"], "selection.budget_denominator") != 1000 or _strict_int(selection["cooldown_days"], "selection.cooldown_days") != 7 or selection["tie_prefix"] != "drive-v1|20260912" or _strict_bool(selection["reset_cooldown_at_eval_start"], "selection.reset_cooldown_at_eval_start") is not True or _strict_bool(selection["rule_requires_positive_signal"], "selection.rule_requires_positive_signal") is not True:
        raise ContractError("selection differs from locked protocol")

    limits = payload["limits"]
    if not isinstance(limits, Mapping):
        raise ContractError("limits must be an object")
    _exact_keys(limits, {"max_rss_bytes", "max_output_bytes", "initial_free_bytes", "min_free_bytes", "max_elapsed_seconds", "fit_timeout_seconds", "poll_seconds", "numerical_threads"}, "limits")
    expected_limits = {"max_rss_bytes": 3 * 1024**3, "max_output_bytes": 12 * 1024**3, "initial_free_bytes": 14 * 1024**3, "min_free_bytes": 2 * 1024**3, "max_elapsed_seconds": 10800, "fit_timeout_seconds": 900, "poll_seconds": 0.2, "numerical_threads": 1}
    for key, expected in expected_limits.items():
        value = limits[key]
        if isinstance(expected, float):
            if _strict_float(value, f"limits.{key}") != expected:
                raise ContractError(f"limits.{key} differs from locked protocol")
        elif _strict_int(value, f"limits.{key}") != expected:
            raise ContractError(f"limits.{key} differs from locked protocol")

    comparison = payload["comparison"]
    if not isinstance(comparison, Mapping):
        raise ContractError("comparison must be an object")
    _exact_keys(comparison, {"abs_tol", "rel_tol", "keys_and_nulls"}, "comparison")
    if _strict_float(comparison["abs_tol"], "comparison.abs_tol") != 1e-12 or _strict_float(comparison["rel_tol"], "comparison.rel_tol") != 1e-10 or comparison["keys_and_nulls"] != "exact":
        raise ContractError("comparison differs from locked protocol")
    return dict(payload)


def _metadata(path: Path) -> dict[str, str]:
    try:
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            result: dict[str, str] = {}
            for key, value in connection.execute("SELECT key,value FROM metadata"):
                if type(key) is not str or not key or type(value) is not str:
                    raise ContractError(f"panel metadata contains an invalid key/value: {path}")
                if key in result:
                    raise ContractError(f"panel metadata contains a duplicate key: {key}")
                result[key] = value
            return result
    except sqlite3.OperationalError as exc:
        # A small synthetic fixture may omit the metadata table entirely, but
        # it is still rejected by require_panel_contract until an explicit
        # fixture marker is present.
        if "no such table: metadata" in str(exc).lower():
            return {}
        raise ContractError(f"cannot read panel metadata: {path}") from exc
    except (OSError, sqlite3.Error) as exc:
        raise ContractError(f"cannot read panel metadata: {path}") from exc


def is_real_panel(path: Path | str) -> bool:
    try:
        target = bound_path(path, "panel", must_exist=True)
        metadata = _metadata(target)
    except (ArtifactError, ContractError):
        return False
    return metadata.get("build_status") == "panel_complete" and metadata.get("serial_model_registry_scope") == "full_source_rows"


def require_panel_contract(path: Path | str, *, role: str) -> dict[str, str]:
    target = bound_path(path, f"{role} panel", must_exist=True)
    metadata = _metadata(target)
    if not metadata:
        raise ContractError(f"{role} panel metadata is empty; explicit fixture or real panel marker is required")
    fixture_profile = metadata.get("fixture_profile")
    fixture_id = metadata.get("fixture_id")
    if (
        metadata.get("build_status") == "fixture"
        and fixture_profile == "synthetic_fixture_v1"
        and fixture_id
        and metadata.get("serial_model_registry_scope") == "fixture_rows"
    ):
        return {
            "mode": "synthetic_fixture", "path": str(target),
            "fixture_profile": fixture_profile, "fixture_id": fixture_id,
            "build_status": metadata["build_status"],
            "serial_model_registry_scope": metadata["serial_model_registry_scope"],
        }
    for key, expected in (("build_status", "panel_complete"), ("serial_model_registry_scope", "full_source_rows")):
        if metadata.get(key) != expected:
            raise ContractError(f"{role} panel metadata {key}={metadata.get(key)!r}, expected {expected!r}")
    if metadata.get("selected_model") not in (None, LOCKED_MODEL):
        raise ContractError(f"{role} panel selected model is not {LOCKED_MODEL}")
    return {"mode": "real_panel", "path": str(target), "build_status": metadata["build_status"], "serial_model_registry_scope": metadata["serial_model_registry_scope"], "selected_model": metadata.get("selected_model", LOCKED_MODEL)}


def load_config_for_bindings(path: Path | str, bindings: Mapping[str, Path | str]) -> tuple[Path, dict[str, Any], bool]:
    target = bound_path(path, "stage config", must_exist=True)
    payload = load_json(target, "stage config")
    panel_values = [value for name, value in bindings.items() if "panel" in str(name)]
    if not panel_values:
        raise ContractError("stage bindings must include a panel")
    contracts = [require_panel_contract(value, role="bound") for value in panel_values]
    modes = {str(item["mode"]) for item in contracts}
    profile = payload.get("profile")
    if modes == {"real_panel"}:
        if profile != "backblaze_2023_st4000dm000":
            raise ContractError("real panels require the locked configuration profile")
        payload = validate_locked_config(payload)
        return target, payload, True
    if modes == {"synthetic_fixture"}:
        if profile != "synthetic_fixture_v1":
            raise ContractError("synthetic fixture panels require synthetic_fixture_v1 profile")
        return target, payload, False
    raise ContractError("stage cannot mix real panels and synthetic fixtures")


def validate_dates(
    values: Mapping[str, str],
    *,
    role: str,
    config: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, dt.date]:
    required_by_role = {
        "train": ("train_start", "train_end", "train_cutoff"),
        "score": ("eval_start", "eval_end", "eval_cutoff"),
        "evaluate": ("eval_start", "eval_end", "eval_cutoff", "event_start", "event_end"),
        "parent": ("train_start", "train_end", "train_cutoff", "eval_start", "eval_end", "eval_cutoff", "event_start", "event_end"),
    }
    if role not in required_by_role:
        raise ContractError(f"unknown date role: {role}")
    parsed: dict[str, dt.date] = {}
    for key in required_by_role[role]:
        value = values.get(key)
        if not isinstance(value, str):
            raise ContractError(f"{role} dates missing {key}")
        try:
            parsed[key] = dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ContractError(f"{role} date {key} is not ISO YYYY-MM-DD") from exc
    if role in {"train", "parent"} and parsed["train_start"] > parsed["train_end"]:
        raise ContractError("train_start must be on or before train_end")
    if role in {"train", "parent"} and parsed["train_end"] + dt.timedelta(days=7) > parsed["train_cutoff"]:
        raise ContractError("training outcome cutoff does not provide seven days")
    if role in {"score", "evaluate", "parent"} and parsed["eval_start"] > parsed["eval_end"]:
        raise ContractError("eval_start must be on or before eval_end")
    if role in {"score", "evaluate", "parent"} and parsed["eval_end"] + dt.timedelta(days=7) > parsed["eval_cutoff"]:
        raise ContractError("evaluation outcome cutoff does not provide seven days")
    if role == "parent" and parsed["train_cutoff"] >= parsed["eval_start"]:
        raise ContractError("training outcome cutoff must be before evaluation start")
    if role in {"evaluate", "parent"}:
        if parsed["event_start"] > parsed["event_end"]:
            raise ContractError("event_start must be on or before event_end")
        if strict and (parsed["event_start"] != parsed["eval_start"] + dt.timedelta(days=7) or parsed["event_end"] != parsed["eval_end"] + dt.timedelta(days=1)):
            raise ContractError("event range differs from locked opportunity interval")
    if strict:
        if config is None or not isinstance(config.get("dates"), Mapping):
            raise ContractError("locked dates are unavailable in configuration")
        for key in required_by_role[role]:
            if values.get(key) != config["dates"].get(key):
                raise ContractError(f"{role} {key} does not match locked configuration")
    return parsed


def validate_worker_spec(role: str, spec: Mapping[str, Any], config: Mapping[str, Any], *, strict: bool) -> None:
    if not isinstance(spec, Mapping):
        raise ContractError("worker specification must be an object")
    values = {key: value for key, value in spec.items() if key.endswith(("start", "end", "cutoff")) or key in {"event_start", "event_end"}}
    validate_dates(values, role=role, config=config, strict=strict)
    if role == "train" and strict:
        timeout = spec.get("fit_timeout_seconds", config["limits"]["fit_timeout_seconds"])
        if type(timeout) not in (int, float) or float(timeout) != float(config["limits"]["fit_timeout_seconds"]):
            raise ContractError("fit timeout differs from locked configuration")


def config_facts(path: Path | str) -> dict[str, str | int]:
    target = bound_path(path, "stage config", must_exist=True)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)}


__all__ = [
    "ContractError", "LOCKED_FEATURES", "LOCKED_DATES", "LOCKED_MODEL", "LOCKED_PROFILE",
    "config_facts", "is_real_panel", "load_config_for_bindings", "load_json",
    "require_panel_contract", "validate_dates", "validate_locked_config", "validate_worker_spec",
]
