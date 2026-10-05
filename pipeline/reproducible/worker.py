"""Independent stage workers used while the full run is being split."""

from __future__ import annotations

import json
import itertools
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading
import time
from typing import Mapping

from . import current
from .artifacts import artifact_facts, bound_path, verify_manifest, write_json_exclusive, write_manifest
from .environment import module_identity_record, project_module_identity
from .cancellation import CancellationRequested, CancellationToken, attach_sqlite_progress
from .contract import ContractError, config_facts, load_config_for_bindings, load_json, require_panel_contract, validate_worker_spec
from .environment import ModuleIdentityError, write_snapshot
from .input_binding import BindingError, verify_binding_for_spec
from .process_contract import authorize_inputs
from .resource_guard import ResourceViolation, assert_fixture_controls_allowed
from .evaluation_summary import summarize_lead_metrics
from .panel_reference import verify_training_reference
from .training_check import load_approved_training_package, verify_training_package, write_training_package


class WorkerError(RuntimeError):
    """A worker specification or stage execution failed."""


def _source_module_record() -> dict[str, object]:
    """Record where the imported ``pipeline`` modules actually came from.

    Under a supervised run the snapshot manifest is bound, so a module with the
    same name that resolves outside the snapshot raises instead of being
    skipped.  Unsupervised callers still record the paths and digests they used.
    """
    if os.environ.get("REPRO_FIXTURE_INJECT") == "foreign_module":
        # Fixture: genuinely import a project module whose source file lives
        # outside the bound snapshot, so the identity check sees a module that
        # was really loaded rather than a fabricated object.
        import importlib.util

        outside = os.environ.get("REPRO_FIXTURE_FOREIGN_MODULE", "")
        if outside:
            spec = importlib.util.spec_from_file_location(
                "pipeline.reproducible.foreign_probe", outside
            )
            if spec is None or spec.loader is None:
                raise ModuleIdentityError(f"cannot load fixture module: {outside}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["pipeline.reproducible.foreign_probe"] = module
    record = module_identity_record()
    if record is not None:
        return record
    identity = project_module_identity()
    return {
        "schema": "project-module-identity-v1",
        "code_root": identity["code_root"],
        "module_count": len(identity["modules"]),
        "modules": identity["modules"],
        "violations": identity["violations"],
        "snapshot_bound": False,
    }


def _stage_manifest(path: Path | str, body: Mapping[str, object], *, artifacts: Mapping[str, object] | None = None) -> dict[str, object]:
    """Publish a stage manifest that also records its module provenance."""
    payload = dict(body)
    payload["source_modules"] = _source_module_record()
    return write_manifest(path, payload, artifacts=artifacts)


def _normalise_cancellation(exc: BaseException, token: CancellationToken) -> BaseException:
    """Turn SQLite's interrupt sentinel into the worker's stable stop error."""
    if token.requested and isinstance(exc, sqlite3.OperationalError):
        return CancellationRequested("stage cancellation requested during SQLite work")
    return exc


def _load_spec(path: Path | str) -> dict[str, object]:
    try:
        return load_json(path, "worker specification")
    except ContractError as exc:
        raise WorkerError(str(exc)) from exc


def _required_text(spec: Mapping[str, object], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise WorkerError(f"worker specification missing {key}")
    return value


def _semantic_bindings(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise WorkerError("worker specification bindings must be an object")
    if any(type(key) is not str or type(value) is not str for key, value in raw.items()):
        raise WorkerError("worker specification binding names and paths must be strings")
    return dict(raw)


def _stage_binding(
    spec: Mapping[str, object],
    role: str,
    config_path: Path,
    bindings: Mapping[str, str],
    strict: bool,
) -> tuple[dict[str, object] | None, Path | None]:
    raw_path = spec.get("binding_file")
    if raw_path is None:
        raise WorkerError(f"{role} worker requires an input binding file")
    if type(raw_path) is not str or not raw_path:
        raise WorkerError("worker specification binding_file must be a path")
    try:
        binding_path = bound_path(raw_path, f"{role} input binding", must_exist=True)
        payload = verify_binding_for_spec(binding_path, role, config_path, bindings)
    except (BindingError, ContractError) as exc:
        raise WorkerError(str(exc)) from exc
    return payload, binding_path


def _fit_with_timeout(function, seconds: float, token: CancellationToken):
    if seconds <= 0:
        raise WorkerError("fit timeout must be positive")
    if hasattr(signal, "SIGALRM"):
        previous = signal.getsignal(signal.SIGALRM)

        def alarm(_signum, _frame):
            token.request()
            raise WorkerError(f"fit timed out after {seconds}s")

        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return function()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    timer = threading.Timer(seconds, token.request)
    timer.daemon = True
    timer.start()
    try:
        result = function()
        if token.requested:
            raise WorkerError(f"fit timed out after {seconds}s")
        return result
    finally:
        timer.cancel()


def train_worker(spec_path: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Fit current_lr in a separate process from the caller.

    The child accepts only a training panel and a configuration file.  Labels,
    features, samples, and the model are written below its new output directory;
    no evaluation or Q3 path is part of this worker's input contract.
    """
    spec = _load_spec(spec_path)
    token = CancellationToken()
    token.install()
    raw_bindings = spec.get("bindings")
    bindings = _semantic_bindings(raw_bindings)
    access = authorize_inputs("train", bindings)
    train_panel = bound_path(bindings["train_panel"], "training panel", must_exist=True)
    try:
        config_path, config, strict = _load_config(bindings)
        panel_contract = require_panel_contract(train_panel, role="training")
        _validate_stage("train", spec, config, strict)
        binding, binding_path = _stage_binding(spec, "train", config_path, bindings, strict)
    except (ContractError, WorkerError) as exc:
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(str(exc)) from exc
    output = bound_path(output_dir, "training worker output")
    if output.exists():
        raise WorkerError(f"training worker output already exists: {output}")
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()
    environment_path = output / "environment.json"
    run_id = str(spec.get("run_id", "worker_train_h7_v1"))
    train_start = _required_text(spec, "train_start")
    train_end = _required_text(spec, "train_end")
    train_cutoff = _required_text(spec, "train_cutoff")
    try:
        fit_timeout_seconds = float(spec.get("fit_timeout_seconds", 900))
    except (TypeError, ValueError) as exc:
        raise WorkerError("fit timeout must be numeric") from exc
    if fit_timeout_seconds <= 0:
        raise WorkerError("fit timeout must be positive")
    try:
        write_snapshot(environment_path)
        labels_path = work / "train_labels.sqlite"
        features_path = work / "train_features.sqlite"
        labels = current.build_labels(
            [train_panel], labels_path, start=train_start, end=train_end,
            dataset_end=train_cutoff, run_id=run_id, progress_callback=token.progress,
        )
        features = current.build_current_features(
            train_panel, features_path, start=train_start, end=train_end,
            dataset_end=train_cutoff, progress_callback=token.progress,
        )
        reference = verify_training_reference(
            train_panel, labels_path, features_path,
            run_id=run_id, start=train_start, end=train_end, cutoff=train_cutoff,
            progress_callback=token.progress,
        )
        calendar_path = output / "qualification_calendar.json"
        write_json_exclusive(calendar_path, {
            "status": "complete",
            "scope": "qualification_calendar_v1",
            "stage": "train",
            "features": features["qualification_calendar"],
        })
        prepared = current._prepare_training_arrays(
            features_path, labels_path, run_id, progress_callback=token.progress,
        )
        training_package_dir = work / "training_input"
        training_package = write_training_package(training_package_dir, prepared)
        # Capture the bound inputs before the independent package check.  The
        # checker compares these facts before and after reconstruction so a
        # replacement cannot become the approved package by being re-read at
        # the end of the check.
        approval_inputs = {
            "config": artifact_facts(config_path),
            "train_panel": artifact_facts(train_panel),
            "train_labels": artifact_facts(labels_path),
            "train_features": artifact_facts(features_path),
            "package_manifest": artifact_facts(training_package_dir / "package.json"),
        }
        fit_check = verify_training_package(
            features_path, labels_path, run_id, training_package_dir,
            progress_callback=token.progress,
            reference_check=reference,
            expected_inputs=approval_inputs,
        )
        fit_check_path = output / "fit_check.json"
        write_json_exclusive(fit_check_path, fit_check)
        approved = load_approved_training_package(
            training_package_dir, fit_check, expected_inputs=approval_inputs,
        )
        payload, arrays = _fit_with_timeout(
            lambda: current._fit_prepared(approved, progress_callback=token.progress),
            fit_timeout_seconds,
            token,
        )
        model_path = output / "model.json"
        samples_path = output / "train_samples.json"
        write_json_exclusive(model_path, payload)
        samples = {
            "status": "complete",
            "columns": list(current.CURRENT_COLUMNS),
            "keys": [[date, serial] for date, serial in arrays["keys"]],
            "labels": [int(value) for value in arrays["labels"].tolist()],
            "weights": [float(value) for value in arrays["weights"].tolist()],
        }
        write_json_exclusive(samples_path, samples)
        body = {
            "status": "complete",
            "role": "train",
            "scope": "worker_train_v1",
            "run_id": run_id,
            "input_contract": access,
            "input_binding": binding,
            "config": {**config_facts(config_path), "strict_locked": strict},
            "panel_contract": panel_contract,
            "environment": {"path": str(environment_path.relative_to(current.ROOT)), "sha256": artifact_facts(environment_path)["sha256"]},
            "training": {"labels": labels, "features": features, "model": payload, "fit_check": fit_check, "training_package": training_package},
        }
        train_artifacts = {
            "train_panel": train_panel,
            "config": config_path,
            "train_labels": labels_path,
            "train_features": features_path,
            "qualification_calendar": calendar_path,
            "environment": environment_path,
            "model": model_path,
            "train_samples": samples_path,
            "fit_check": fit_check_path,
            "training_package": training_package_dir / "package.json",
        }
        for package_name in ("raw", "raw_missing_mask", "labels", "weights", "transformed", "keys", "preprocessing"):
            train_artifacts[f"training_package_{package_name}"] = training_package_dir / (f"{package_name}.npy" if package_name not in {"keys", "preprocessing"} else f"{package_name}.json")
        if binding_path is not None:
            train_artifacts["input_binding"] = binding_path
        manifest = _stage_manifest(
            output / "train_manifest.json",
            body,
            artifacts=train_artifacts,
        )
        return manifest
    except BaseException as caught:
        exc = _normalise_cancellation(caught, token)
        write_json_exclusive(
            output / "failure.json",
            {"status": "failed", "role": "train", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise exc
    finally:
        token.restore()


def _load_config(bindings: Mapping[str, str]) -> tuple[Path, Mapping[str, object], bool]:
    try:
        return load_config_for_bindings(bindings["config"], bindings)
    except (ContractError, KeyError) as exc:
        raise WorkerError(str(exc)) from exc


def _validate_stage(role: str, spec: Mapping[str, object], config: Mapping[str, object], strict: bool) -> None:
    try:
        validate_worker_spec(role, spec, config, strict=strict)
    except ContractError as exc:
        raise WorkerError(str(exc)) from exc


def _validate_model_payload(model: Mapping[str, object], *, strict: bool, config: Mapping[str, object]) -> None:
    if model.get("model") != "current_lr" or model.get("fit_mode") != "fitted":
        raise WorkerError("model must be a fitted current_lr payload")
    columns = model.get("columns")
    coef = model.get("coef")
    preprocessing = model.get("preprocessing")
    if not isinstance(columns, list) or not isinstance(coef, list) or not isinstance(preprocessing, Mapping):
        raise WorkerError("model payload is missing columns, coefficients, or preprocessing")
    expected_columns = list(config.get("features", current.CURRENT_COLUMNS))
    if columns != expected_columns or len(coef) != len(expected_columns):
        raise WorkerError("model feature order or dimension does not match configuration")
    try:
        numbers = [float(value) for value in coef] + [float(model["intercept"])]
        means = [float(value) for value in preprocessing["imputation_mean"]]
        centers = [float(value) for value in preprocessing["standardization_mean"]]
        scales = [float(value) for value in preprocessing["standardization_scale"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("model preprocessing is not numeric") from exc
    if len(means) != len(expected_columns) or len(centers) != len(expected_columns) or len(scales) != len(expected_columns):
        raise WorkerError("model preprocessing dimension does not match configuration")
    if not all(math.isfinite(value) for value in numbers + means + centers + scales) or not all(value > 0 for value in scales):
        raise WorkerError("model coefficients and preprocessing must be finite with positive scales")
    count_values = [model.get(field) for field in ("training_rows", "positive_rows", "negative_rows")]
    if any(value is not None for value in count_values):
        if any(type(value) is not int or value < 0 for value in count_values):
            raise WorkerError("model training counts must be non-negative integers")
        if count_values[0] != count_values[1] + count_values[2] or count_values[1] <= 0 or count_values[2] <= 0:
            raise WorkerError("model training counts are inconsistent or missing a class")
    params = model.get("params")
    if params is not None:
        if not isinstance(params, Mapping):
            raise WorkerError("model estimator parameters must be an object")
        for key in ("penalty", "solver"):
            if type(params.get(key)) is not str or not params[key]:
                raise WorkerError(f"model estimator parameter is invalid: {key}")
        for key in ("C", "tol"):
            value = params.get(key)
            if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise WorkerError(f"model estimator parameter is invalid: {key}")
        for key in ("max_iter", "random_state"):
            if type(params.get(key)) is not int or (key == "max_iter" and params[key] <= 0):
                raise WorkerError(f"model estimator parameter is invalid: {key}")
        if type(params.get("fit_intercept")) is not bool:
            raise WorkerError("model estimator fit_intercept must be boolean")
        if params.get("class_weight") is not None and not isinstance(params.get("class_weight"), Mapping):
            raise WorkerError("model estimator class_weight is invalid")
    if strict:
        if not isinstance(params, Mapping):
            raise WorkerError("real model is missing estimator parameters")
        expected = config["estimator"]
        for key in ("penalty", "solver", "C", "tol", "max_iter", "fit_intercept", "class_weight", "random_state"):
            if params.get(key) != expected.get(key):
                raise WorkerError(f"model estimator parameter differs: {key}")


def _write_score_db(
    features_path: Path,
    output_path: Path,
    model: Mapping[str, object],
    *,
    progress_callback=None,
) -> int:
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists() or output_path.exists():
        raise WorkerError(f"score output already exists: {output_path}")
    connection = sqlite3.connect(partial)
    try:
        connection.execute(
            "CREATE TABLE scores(decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, "
            "score REAL NOT NULL, rule_score INTEGER NOT NULL, tie_break_sha256 TEXT NOT NULL, "
            "PRIMARY KEY(decision_date, serial_number))"
        )
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if progress_callback is not None:
            attach_sqlite_progress(connection, progress_callback)
        connection.execute("INSERT INTO metadata VALUES ('status', 'running')")
        rows = []
        count = 0
        for row in current._rule_and_model_scores(features_path, model, progress_callback=progress_callback):
            if progress_callback is not None and progress_callback():
                raise RuntimeError("score write cancelled")
            score = float(row["score"])
            if not math.isfinite(score):
                raise WorkerError("score is not finite")
            rows.append((row["decision_date"], row["serial_number"], score, int(row["rule_score"]), row["tie_break_sha256"]))
            if len(rows) >= 5000:
                connection.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?)", rows)
                count += len(rows)
                rows.clear()
        if rows:
            connection.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?)", rows)
            count += len(rows)
        connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", [("status", "complete"), ("score_rows", str(count))])
        connection.commit()
        connection.close()
        os.replace(partial, output_path)
        return count
    except BaseException:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        connection.close()
        raise


def _selection_from_score_db(
    score_path: Path,
    method: str,
    *,
    calendar_rows: list[Mapping[str, object]] | None = None,
    progress_callback=None,
) -> dict[str, object]:
    connection = sqlite3.connect(score_path)
    connection.row_factory = sqlite3.Row
    if progress_callback is not None:
        attach_sqlite_progress(connection, progress_callback)
    try:
        rows = connection.execute(
            "SELECT decision_date, serial_number, score, rule_score, tie_break_sha256 "
            "FROM scores ORDER BY decision_date, serial_number"
        )
        last_alert: dict[str, object] = {}
        alerts: list[dict[str, object]] = []
        daily_by_date: dict[str, dict[str, object]] = {}
        for date_text, group in itertools.groupby(rows, key=lambda row: str(row["decision_date"])):
            date_rows = [dict(row) for row in group]
            day = current.dt.date.fromisoformat(date_text)
            selected, excluded = current._select(date_rows, method, last_alert, day)
            daily_by_date[date_text] = {
                "date": date_text,
                "eligible": len(date_rows),
                "budget": current.ceil_budget(len(date_rows), current.BUDGET_DENOMINATOR),
                "cooldown_excluded": excluded,
                "alerts": len(selected),
            }
            for row in selected:
                # Keep the model score for audit, but make the score that
                # actually drove ranking explicit for the rule method.
                ranking_score = float(row["score"]) if method == "current_lr" else int(row["rule_score"])
                alerts.append(dict(row, method=method, ranking_score=ranking_score))
        if calendar_rows is None:
            daily = [daily_by_date[key] for key in sorted(daily_by_date)]
        else:
            daily = []
            for raw_calendar in calendar_rows:
                date_text = str(raw_calendar["date"])
                expected = int(raw_calendar["eligible_rows"])
                entry = daily_by_date.pop(date_text, None)
                if entry is None:
                    if expected != 0:
                        raise WorkerError(f"score rows missing for non-empty qualification day: {date_text}")
                    entry = {
                        "date": date_text, "eligible": 0, "budget": 0,
                        "cooldown_excluded": 0, "alerts": 0,
                    }
                elif int(entry["eligible"]) != expected:
                    raise WorkerError(f"score rows do not match qualification calendar: {date_text}")
                daily.append(entry)
            if daily_by_date:
                raise WorkerError("score rows contain dates absent from qualification calendar")
        return {"method": method, "alerts": alerts, "daily": daily}
    finally:
        connection.close()


def score_worker(spec_path: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Score an evaluation panel without receiving any outcome database."""
    spec = _load_spec(spec_path)
    token = CancellationToken()
    token.install()
    raw_bindings = spec.get("bindings")
    bindings = _semantic_bindings(raw_bindings)
    access = authorize_inputs("score", bindings)
    config_path, config, strict = _load_config(bindings)
    eval_panel = bound_path(bindings["eval_panel"], "evaluation panel", must_exist=True)
    history_panel = bound_path(bindings["history_panel"], "history panel", must_exist=True)
    model_path = bound_path(bindings["model"], "model JSON", must_exist=True)
    train_manifest_path = None
    train_manifest = None
    if strict:
        raw_train_manifest = bindings.get("train_manifest")
        if raw_train_manifest is None:
            raise WorkerError("real scoring requires the closed training manifest")
        train_manifest_path = bound_path(raw_train_manifest, "training manifest", must_exist=True)
        try:
            train_manifest = verify_manifest(train_manifest_path, expected_status="complete")
        except Exception as exc:
            raise WorkerError(f"invalid training manifest: {exc}") from exc
    try:
        model = load_json(model_path, "model JSON")
    except ContractError as exc:
        raise WorkerError(f"invalid model JSON: {model_path}") from exc
    try:
        eval_contract = require_panel_contract(eval_panel, role="evaluation")
        history_contract = require_panel_contract(history_panel, role="history")
        _validate_stage("score", spec, config, strict)
        binding, binding_path = _stage_binding(spec, "score", config_path, bindings, strict)
        _validate_model_payload(model, strict=strict, config=config)
        if strict:
            model_facts = train_manifest.get("verified_artifacts", {}).get("model") if isinstance(train_manifest, Mapping) else None
            config_facts_in_train = train_manifest.get("config", {}) if isinstance(train_manifest, Mapping) else {}
            if not isinstance(model_facts, Mapping) or model_facts.get("sha256") != artifact_facts(model_path)["sha256"]:
                raise WorkerError("score model is not the model published by training")
            if not isinstance(config_facts_in_train, Mapping) or config_facts_in_train.get("sha256") != config_facts(config_path)["sha256"]:
                raise WorkerError("score configuration differs from training configuration")
        if strict:
            identity = current.verify_cross_quarter_identity(history_panel, eval_panel)
            if identity.get("status") != "pass":
                raise WorkerError("cross-quarter identity did not pass before scoring")
        else:
            identity = None
    except (ContractError, WorkerError) as exc:
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(str(exc)) from exc
    output = bound_path(output_dir, "score worker output")
    if output.exists():
        raise WorkerError(f"score worker output already exists: {output}")
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()
    environment_path = output / "environment.json"
    eval_start = _required_text(spec, "eval_start")
    eval_end = _required_text(spec, "eval_end")
    eval_cutoff = _required_text(spec, "eval_cutoff")
    try:
        write_snapshot(environment_path)
        features_path = work / "eval_features.sqlite"
        features = current.build_current_features(
            eval_panel, features_path, start=eval_start, end=eval_end,
            dataset_end=eval_cutoff, history_panels=[history_panel], progress_callback=token.progress,
        )
        calendar_path = output / "qualification_calendar.json"
        write_json_exclusive(calendar_path, {
            "status": "complete",
            "scope": "qualification_calendar_v1",
            "stage": "score",
            "features": features["qualification_calendar"],
        })
        scores_path = output / "scores.sqlite"
        score_rows = _write_score_db(features_path, scores_path, model, progress_callback=token.progress)
        selection_paths = {}
        for method in ("current_lr", "smart_nonzero"):
            if token.progress():
                raise RuntimeError("selection cancelled")
            name = f"{method}_selection.json"
            path = output / name
            selection = _selection_from_score_db(
                scores_path,
                method,
                calendar_rows=features["qualification_calendar"]["days"],
                progress_callback=token.progress,
            )
            write_json_exclusive(path, selection)
            selection_paths[method] = path
        selection_artifacts = {
            "full_scores": scores_path,
            "current_lr_selection": selection_paths["current_lr"],
            "smart_nonzero_selection": selection_paths["smart_nonzero"],
            "model": model_path,
            "qualification_calendar": calendar_path,
        }
        if train_manifest_path is not None:
            selection_artifacts["train_manifest"] = train_manifest_path
        selection_manifest = _stage_manifest(
            output / "selection_manifest.json",
            {
                "status": "closed",
                "role": "score",
                "scope": "worker_score_v1",
                "score_rows": score_rows,
                "methods": ["current_lr", "smart_nonzero"],
                "selection_close_order": ["current_lr", "smart_nonzero"],
                "model_sha256": artifact_facts(model_path)["sha256"],
                "qualification_calendar_days": len(features["qualification_calendar"]["days"]),
            },
            artifacts=selection_artifacts,
        )
        score_artifacts = {
            "eval_panel": eval_panel,
            "history_panel": history_panel,
            "model": model_path,
            "config": config_path,
            "eval_features": features_path,
            "qualification_calendar": calendar_path,
            "environment": environment_path,
            "full_scores": scores_path,
            "selection_manifest": output / "selection_manifest.json",
            "current_lr_selection": selection_paths["current_lr"],
            "smart_nonzero_selection": selection_paths["smart_nonzero"],
        }
        if train_manifest_path is not None:
            score_artifacts["train_manifest"] = train_manifest_path
        if binding_path is not None:
            score_artifacts["input_binding"] = binding_path
        return _stage_manifest(
            output / "score_manifest.json",
            {
                "status": "complete",
                "role": "score",
                "scope": "worker_score_v1",
                "input_contract": access,
                "input_binding": binding,
                "config": {**config_facts(config_path), "strict_locked": strict},
                "eval_panel_contract": eval_contract,
                "history_panel_contract": history_contract,
                "cross_quarter_identity": identity,
                "environment": {"path": str(environment_path.relative_to(current.ROOT)), "sha256": artifact_facts(environment_path)["sha256"]},
                "features": features,
                "score_rows": score_rows,
                "selection_manifest_hash": selection_manifest["manifest_hash"],
            },
            artifacts=score_artifacts,
        )
    except BaseException as caught:
        exc = _normalise_cancellation(caught, token)
        write_json_exclusive(
            output / "failure.json",
            {"status": "failed", "role": "score", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise exc
    finally:
        token.restore()


def _manifest_artifact_path(manifest_path: Path, manifest: Mapping[str, object], name: str) -> Path:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or name not in raw_artifacts:
        raise WorkerError(f"manifest is missing artifact {name}")
    raw = raw_artifacts[name]
    if not isinstance(raw, Mapping):
        raise WorkerError(f"invalid artifact facts for {name}")
    path = bound_path(str(raw.get("path", "")), f"manifest artifact {name}", must_exist=True)
    if not path.is_relative_to(manifest_path.parent):
        raise WorkerError(f"manifest artifact escapes its stage directory: {name}")
    return path


def _average_precision(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    order: str,
) -> tuple[float | None, int, int]:
    """Compute grouped average precision over the complete known score pool."""
    known_total = int(connection.execute(
        "SELECT COUNT(*) FROM scores s JOIN outcomes.label_flow l "
        "ON l.run_id=? AND l.decision_date=s.decision_date AND l.serial_number=s.serial_number "
        "WHERE l.eligible=1 AND l.label IN (0,1)", (run_id,)
    ).fetchone()[0])
    positive_total = int(connection.execute(
        "SELECT COUNT(*) FROM scores s JOIN outcomes.label_flow l "
        "ON l.run_id=? AND l.decision_date=s.decision_date AND l.serial_number=s.serial_number "
        "WHERE l.eligible=1 AND l.label=1", (run_id,)
    ).fetchone()[0])
    if positive_total == 0:
        return None, known_total, positive_total
    rows = connection.execute(
        "SELECT s.score, s.rule_score, l.label FROM scores s "
        "JOIN outcomes.label_flow l ON l.run_id=? AND l.decision_date=s.decision_date "
        "AND l.serial_number=s.serial_number WHERE l.eligible=1 AND l.label IN (0,1) "
        f"ORDER BY {order}", (run_id,)
    )
    seen = 0
    positives = 0
    average_precision = 0.0
    group_value: object = object()
    group_total = 0
    group_positive = 0
    for score, rule_score, label in rows:
        value = score if order.startswith("s.score") else rule_score
        if group_total and value != group_value:
            seen += group_total
            positives += group_positive
            average_precision += (group_positive / positive_total) * (positives / seen)
            group_total = 0
            group_positive = 0
        group_value = value
        group_total += 1
        group_positive += int(label == 1)
    if group_total:
        seen += group_total
        positives += group_positive
        average_precision += (group_positive / positive_total) * (positives / seen)
    return average_precision, known_total, positive_total


def _full_pool_metrics(score_path: Path, label_path: Path, run_id: str) -> dict[str, object]:
    """Join the complete score pool to outcomes only inside evaluation."""
    scores = current._open_readonly(score_path)
    try:
        scores.execute("ATTACH DATABASE ? AS outcomes", (label_path.as_uri() + "?mode=ro&immutable=1",))
        current_ap, known_total, positive_total = _average_precision(
            scores, run_id, order="s.score DESC, s.tie_break_sha256 ASC, s.serial_number ASC"
        )
        rule_ap, rule_known_total, rule_positive_total = _average_precision(
            scores, run_id, order="s.rule_score DESC, s.tie_break_sha256 ASC, s.serial_number ASC"
        )
        if (known_total, positive_total) != (rule_known_total, rule_positive_total):
            raise WorkerError("score/outcome join changed between AP methods")
        return {
            "known_pool_rows": known_total,
            "positive_pool_rows": positive_total,
            "current_lr_known_average_precision": current_ap,
            "smart_nonzero_known_average_precision": rule_ap,
        }
    finally:
        scores.close()


def _augment_selection_metrics(summary: Mapping[str, object], full_pool: Mapping[str, object]) -> dict[str, object]:
    metrics = dict(summary["metrics"])
    total = int(metrics["alerts"])
    known_hit = int(metrics["known_hit_alerts"])
    known_no_hit = int(metrics["known_no_hit_alerts"])
    unknown = int(metrics["unknown_alerts"])
    eligible = int(metrics["eligible_rows"])
    metrics.update({
        "known_precision": known_hit / (known_hit + known_no_hit) if known_hit + known_no_hit else None,
        "precision_lower_bound": known_hit / total if total else None,
        "precision_upper_bound": (known_hit + unknown) / total if total else None,
        "unknown_ratio": unknown / total if total else None,
        "known_misses_per_1000_eligible_days": 1000.0 * known_no_hit / eligible if eligible else None,
        "known_pool_rows": full_pool["known_pool_rows"],
        "positive_pool_rows": full_pool["positive_pool_rows"],
        "full_pool_average_precision": full_pool["current_lr_known_average_precision"] if metrics["method"] == "current_lr" else full_pool["smart_nonzero_known_average_precision"],
    })
    return metrics


def evaluate_worker(spec_path: Path | str, output_dir: Path | str) -> dict[str, object]:
    """Evaluate closed selections after verifying their score manifest."""
    spec = _load_spec(spec_path)
    token = CancellationToken()
    token.install()
    raw_bindings = spec.get("bindings")
    bindings = _semantic_bindings(raw_bindings)
    access = authorize_inputs("evaluate", bindings)
    config_path, config, strict = _load_config(bindings)
    eval_panel = bound_path(bindings["eval_panel"], "evaluation panel", must_exist=True)
    history_panel = bound_path(bindings["history_panel"], "history panel", must_exist=True)
    selection_manifest_path = bound_path(bindings["selection_manifest"], "selection manifest", must_exist=True)
    score_manifest_path = bound_path(bindings["score_manifest"], "score manifest", must_exist=True)
    if selection_manifest_path == score_manifest_path:
        raise WorkerError("selection and score manifests must be different files")
    selection_manifest = verify_manifest(selection_manifest_path, expected_status="closed")
    score_manifest = verify_manifest(score_manifest_path, expected_status="complete")
    if score_manifest.get("selection_manifest_hash") != selection_manifest.get("manifest_hash"):
        raise WorkerError("score manifest does not bind the closed selection manifest")
    try:
        eval_contract = require_panel_contract(eval_panel, role="evaluation")
        history_contract = require_panel_contract(history_panel, role="history")
        _validate_stage("evaluate", spec, config, strict)
        binding, binding_path = _stage_binding(spec, "evaluate", config_path, bindings, strict)
        if strict:
            score_config = score_manifest.get("config")
            if not isinstance(score_config, Mapping) or score_config.get("sha256") != config_facts(config_path)["sha256"]:
                raise WorkerError("evaluation configuration differs from scoring configuration")
            score_model = score_manifest.get("verified_artifacts", {}).get("model") if isinstance(score_manifest.get("verified_artifacts"), Mapping) else None
            selection_model = selection_manifest.get("verified_artifacts", {}).get("model") if isinstance(selection_manifest.get("verified_artifacts"), Mapping) else None
            if not isinstance(score_model, Mapping) or not isinstance(selection_model, Mapping) or score_model.get("sha256") != selection_model.get("sha256"):
                raise WorkerError("selection and score manifests bind different models")
        if strict:
            identity = current.verify_cross_quarter_identity(history_panel, eval_panel)
            if identity.get("status") != "pass":
                raise WorkerError("cross-quarter identity did not pass before evaluation")
        else:
            identity = None
    except (ContractError, WorkerError) as exc:
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(str(exc)) from exc
    selection_paths = {
        method: _manifest_artifact_path(selection_manifest_path, selection_manifest, f"{method}_selection")
        for method in ("current_lr", "smart_nonzero")
    }
    full_scores_path = _manifest_artifact_path(score_manifest_path, score_manifest, "full_scores")
    output = bound_path(output_dir, "evaluation worker output")
    if output.exists():
        raise WorkerError(f"evaluation worker output already exists: {output}")
    output.mkdir(parents=True)
    work = output / "work"
    work.mkdir()
    environment_path = output / "environment.json"
    run_id = str(spec.get("run_id", "worker_eval_h7_v1"))
    eval_start = _required_text(spec, "eval_start")
    eval_end = _required_text(spec, "eval_end")
    eval_cutoff = _required_text(spec, "eval_cutoff")
    event_start = _required_text(spec, "event_start")
    event_end = _required_text(spec, "event_end")
    try:
        write_snapshot(environment_path)
        labels_path = work / "eval_labels.sqlite"
        labels = current.build_labels(
            [history_panel, eval_panel], labels_path, start=eval_start,
            end=eval_end, dataset_end=eval_cutoff, run_id=run_id, progress_callback=token.progress,
        )
        events = current._event_map(
            [history_panel, eval_panel],
            dataset_end=current.dt.date.fromisoformat(eval_cutoff),
            start=current.dt.date.fromisoformat(event_start),
            end=current.dt.date.fromisoformat(event_end),
            score_start=current.dt.date.fromisoformat(eval_start),
            score_end=current.dt.date.fromisoformat(eval_end),
            progress_callback=token.progress,
        )
        alert_paths: dict[str, Path] = {}
        summaries: dict[str, object] = {}
        full_pool = _full_pool_metrics(full_scores_path, labels_path, run_id)
        for method, selection_path in selection_paths.items():
            if token.progress():
                raise RuntimeError("evaluation cancelled")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if not isinstance(selection, Mapping) or selection.get("method") != method:
                raise WorkerError(f"selection method mismatch: {method}")
            alerts, summary = current._evaluate_selection(
                dict(selection), labels_path, run_id,
                {key: dict(value) for key, value in events.items()},
                progress_callback=token.progress,
            )
            summary["metrics"] = _augment_selection_metrics(summary, full_pool)
            summary["metrics"].update(
                summarize_lead_metrics(
                    alerts,
                    summary["events"],
                    event_start=event_start,
                    event_end=event_end,
                )
            )
            alert_path = output / f"{method}_alerts.json"
            write_json_exclusive(alert_path, {"status": "complete", "method": method, "alerts": alerts})
            alert_paths[method] = alert_path
            summaries[method] = summary
        metrics_path = output / "metrics.json"
        write_json_exclusive(metrics_path, {
            "status": "complete",
            "scope": "evaluation_metrics_v2",
            "event_window": {"start": event_start, "end": event_end},
            "methods": {method: summary["metrics"] for method, summary in summaries.items()},
            "event_opportunities": {method: summary["events"] for method, summary in summaries.items()},
        })
        return _stage_manifest(
            output / "evaluation_manifest.json",
            {
                "status": "complete",
                "role": "evaluate",
                "scope": "worker_evaluate_v2",
                "run_id": run_id,
                "input_contract": access,
                "input_binding": binding,
                "config": {**config_facts(config_path), "strict_locked": strict},
                "eval_panel_contract": eval_contract,
                "history_panel_contract": history_contract,
                "cross_quarter_identity": identity,
                "environment": {"path": str(environment_path.relative_to(current.ROOT)), "sha256": artifact_facts(environment_path)["sha256"]},
                "labels": labels,
                "event_count": len(events),
                "summaries": summaries,
                "full_pool": full_pool,
                "selection_manifest_hash": selection_manifest["manifest_hash"],
                "score_manifest_hash": score_manifest["manifest_hash"],
            },
            artifacts={
                "eval_panel": eval_panel,
                "history_panel": history_panel,
                "config": config_path,
                "environment": environment_path,
                "eval_labels": labels_path,
                "selection_manifest": selection_manifest_path,
                "score_manifest": score_manifest_path,
                "full_scores": full_scores_path,
                "current_lr_alerts": alert_paths["current_lr"],
                "smart_nonzero_alerts": alert_paths["smart_nonzero"],
                "metrics": metrics_path,
                **({"input_binding": binding_path} if binding_path is not None else {}),
            },
        )
    except BaseException as caught:
        exc = _normalise_cancellation(caught, token)
        write_json_exclusive(
            output / "failure.json",
            {"status": "failed", "role": "evaluate", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise exc
    finally:
        token.restore()


def run_worker(role: str, spec_path: Path | str, output_dir: Path | str) -> dict[str, object]:
    # A fit or a scoring stage must never start while test controls are set
    # for a profile that is not a synthetic fixture.
    try:
        declared_profile = load_json(Path(spec_path), "worker spec").get("config_profile")
    except (OSError, json.JSONDecodeError, WorkerError):
        declared_profile = None
    assert_fixture_controls_allowed(declared_profile)
    if role == "score":
        return score_worker(spec_path, output_dir)
    if role == "evaluate":
        return evaluate_worker(spec_path, output_dir)
    if role != "train":
        raise WorkerError(f"worker role is not implemented yet: {role}")
    return train_worker(spec_path, output_dir)


__all__ = ["WorkerError", "evaluate_worker", "run_worker", "score_worker", "train_worker"]
