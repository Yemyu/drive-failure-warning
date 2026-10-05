"""Build the three locked simple baselines on the frozen training inputs.

The program intentionally keeps the model stage separate from the historical
rule replay.  It reads the verified feature and label databases through
immutable SQLite connections, samples only known outcomes for fitting, fits
three fixed L2 logistic regressions, then scores the complete eligible pool
and applies the already locked top-k/cooldown replay.  It never writes to an
input database and never reads Q3/Q4.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import time
import warnings
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

# Keep numerical libraries single-threaded and avoid an accidental process
# explosion on a laptop.  These are set before numpy/sklearn is imported.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
PANEL_DB = ROOT / "data/derived/panel_q1q2_verified_v1.sqlite"
FEATURE_DB = ROOT / "data/derived/features_train_v1.sqlite"
REPLAY_DB = ROOT / "data/derived/rule_replay_train_v3.sqlite"
CONFIG_PATH = ROOT / "configs/simple_baseline_v1.json"
FREEZE_PATH = ROOT / "evidence/q2/fit_review_v1/review_v1.json"
OUTPUT_DIR = ROOT / "data/derived/simple_baseline_v1"
OUTPUT_DB = OUTPUT_DIR / "model_results.sqlite"
EVIDENCE_DIR = ROOT / "evidence/simple_baseline_v1"

LABEL_RUN_ID = "train_q1q2_verified_h7_v1"
LABEL_SPLIT = "train"
HORIZON = 7
MODEL_NAME = "ST4000DM000"
SCORE_START = dt.date(2023, 1, 15)
SCORE_END = dt.date(2023, 6, 23)
DATASET_END = dt.date(2023, 6, 30)
EVENT_START = dt.date(2023, 1, 22)
EVENT_END = dt.date(2023, 6, 24)
COOLDOWN_DAYS = 7
BUDGET_DENOMINATOR = 1000
CHECK_EVERY_ROWS = 10_000
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
MAX_OWNED_BYTES = 4 * 1024 * 1024 * 1024
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
FIT_SECONDS_PER_MODEL = 15 * 60
TOTAL_FIT_SECONDS = 45 * 60
STAGE_SECONDS = 90 * 60
MAX_TRAIN_ROWS = 200_000

MODEL_ORDER = ("age_lr", "current_lr", "history_lr")
CURRENT_MISSING_COLUMNS = tuple(
    f"smart_{field}_current_missing" for field in (5, 9, 187, 188, 197, 198)
)
KNOWN_STATUSES = {"positive_observed", "positive_with_gap", "negative_observed"}


class BaselineStopped(RuntimeError):
    """A declared input, reproducibility, numerical or resource failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def owned_bytes() -> int:
    total = 0
    for base in (OUTPUT_DIR, EVIDENCE_DIR):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def resource_snapshot() -> dict[str, int]:
    return {
        "rss_max_bytes": rss_bytes(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "owned_bytes": owned_bytes(),
    }


def check_resources(start: float, *, label: str) -> dict[str, int]:
    snapshot = resource_snapshot()
    if snapshot["rss_max_bytes"] >= MAX_RSS_BYTES:
        raise BaselineStopped(f"RSS limit during {label}: {snapshot}")
    if snapshot["owned_bytes"] >= MAX_OWNED_BYTES:
        raise BaselineStopped(f"project increment limit during {label}: {snapshot}")
    if snapshot["free_bytes"] < MIN_FREE_BYTES:
        raise BaselineStopped(f"free-space limit during {label}: {snapshot}")
    if time.monotonic() - start >= STAGE_SECONDS:
        raise BaselineStopped(f"stage timeout during {label}: {snapshot}")
    return snapshot


def connect_immutable(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise BaselineStopped(f"missing input database: {path}")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise BaselineStopped(f"non-empty immutable sidecar: {sidecar}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key,value FROM metadata")}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineStopped(f"cannot read JSON {path}: {exc}") from exc


def verify_freeze(config: dict) -> dict:
    freeze = read_json(FREEZE_PATH)
    if freeze.get("status") != "pass_for_frozen_inputs":
        raise BaselineStopped(f"fit freeze is not passing: {freeze.get('status')}")
    if freeze.get("next_stage_permission") != "conditional_local_simple_baseline_fit_after_contract_gates":
        raise BaselineStopped("fit freeze does not grant the locked local baseline stage")
    if freeze.get("q3_q4_access") or freeze.get("remote_setup"):
        raise BaselineStopped("fit freeze unexpectedly enables Q3/Q4 or remote setup")
    snapshots = freeze.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        raise BaselineStopped("fit freeze has no input snapshots")
    mismatches = []
    for relative, expected in snapshots.items():
        path = ROOT / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise BaselineStopped(f"frozen input changed: {mismatches[:3]}")
    if config.get("status") != "design_locked_not_executed":
        raise BaselineStopped("baseline config status must remain design_locked_not_executed before run")
    return freeze


def verify_databases(panel: sqlite3.Connection, feature: sqlite3.Connection, replay: sqlite3.Connection, config: dict) -> dict:
    panel_meta = metadata(panel)
    feature_meta = metadata(feature)
    replay_meta = metadata(replay)
    if panel_meta.get("training_approval") != "false" or feature_meta.get("training_approval") != "false":
        raise BaselineStopped("input training_approval is not false")
    if panel_meta.get("build_status") != "panel_complete" or feature_meta.get("feature_status") != "complete":
        raise BaselineStopped("panel or feature input is not complete")
    required = (
        ("panel.build_manifest_hash", panel_meta.get("build_manifest_hash")),
        ("panel.label_build_manifest_hash", panel_meta.get("label_build_manifest_hash")),
        ("panel.label_source_manifest_hash", panel_meta.get("label_source_manifest_hash")),
        ("feature.source_panel_manifest_hash", feature_meta.get("source_panel_manifest_hash")),
        ("feature.feature_dictionary_hash", feature_meta.get("feature_dictionary_hash")),
        ("feature.feature_config_hash", feature_meta.get("feature_config_hash")),
        ("feature.feature_code_sha256", feature_meta.get("feature_code_sha256")),
    )
    for name, value in required:
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise BaselineStopped(f"missing provenance hash: {name}")
    if feature_meta.get("source_panel_manifest_hash") != panel_meta.get("build_manifest_hash"):
        raise BaselineStopped("feature does not bind to panel build manifest")
    run = panel.execute(
        "SELECT * FROM label_runs WHERE run_id=? AND split=? AND horizon_days=?",
        (LABEL_RUN_ID, LABEL_SPLIT, HORIZON),
    ).fetchone()
    if run is None or run["status"] != "complete":
        raise BaselineStopped("locked H=7 label run is missing or incomplete")
    if tuple(run[key] for key in ("start_date", "end_date", "dataset_end")) != (
        SCORE_START.isoformat(), SCORE_END.isoformat(), DATASET_END.isoformat()
    ):
        raise BaselineStopped("label dates differ from the locked training range")
    for key in ("config_hash", "code_hash", "manifest_hash"):
        value = str(run[key] or "")
        if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise BaselineStopped(f"invalid label run {key}")
    if run["manifest_hash"] != panel_meta.get("label_source_manifest_hash"):
        raise BaselineStopped("label run source does not match panel source")
    if panel_meta.get("label_build_manifest_hash") != panel_meta.get("build_manifest_hash"):
        raise BaselineStopped("label build manifest does not match panel build")
    if replay_meta.get("training_approval") not in (None, "false"):
        raise BaselineStopped("replay training approval unexpectedly true")
    feature_columns = [row["name"] for row in feature.execute("PRAGMA table_info(feature_rows)")]
    expected = [str(name) for name in config["models"]["history_lr"]]
    all_expected = {name for values in config["models"].values() for name in values}
    forbidden = {"failure", "label", "first_failure_date", "future_observations", "serial_number", "decision_date", "model", "tie_break_sha256"}
    required_key_columns = {"decision_date", "serial_number", "model", "tie_break_sha256"}
    if not all_expected.issubset(set(feature_columns)) or not required_key_columns.issubset(set(feature_columns)) or all_expected & forbidden:
        raise BaselineStopped("feature input schema does not satisfy the locked whitelist")
    if len(set(expected)) != len(expected):
        raise BaselineStopped("history feature whitelist contains duplicates")
    feature_count = int(feature.execute("SELECT COUNT(*) FROM feature_rows").fetchone()[0])
    if feature_count != 2_860_459:
        raise BaselineStopped(f"feature row count changed: {feature_count}")
    return {
        "panel_meta": panel_meta,
        "feature_meta": feature_meta,
        "replay_meta": replay_meta,
        "label_run": dict(run),
        "feature_columns": feature_columns,
        "feature_count": feature_count,
    }


def sampling_hash(date_text: str, serial: str) -> str:
    return hashlib.sha256(f"fit-v1|20260913|{date_text}|{serial}".encode("utf-8")).hexdigest()


def sample_negative_indices(date_text: str, serials: Sequence[str], negative_count: int) -> tuple[set[str], str]:
    sample_count = min(negative_count, max(1, negative_count // 20)) if negative_count else 0
    ranked = sorted((sampling_hash(date_text, serial), serial) for serial in serials)
    selected = {serial for _digest, serial in ranked[:sample_count]}
    digest = hashlib.sha256("".join(f"{h}:{s}\n" for h, s in ranked[:sample_count]).encode("utf-8")).hexdigest()
    return selected, digest


def row_value(row: sqlite3.Row, name: str) -> float:
    value = row[name]
    if value is None:
        return float("nan")
    number = float(value)
    if not math.isfinite(number):
        raise BaselineStopped(f"non-finite feature value in {name}")
    return number


def collect_training_sample(panel: sqlite3.Connection, feature: sqlite3.Connection, config: dict, start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    history_columns = list(config["models"]["history_lr"])
    dates = [str(row[0]) for row in feature.execute("SELECT DISTINCT decision_date FROM feature_rows ORDER BY decision_date")]
    label_sql = (
        "SELECT serial_number,label,status,first_failure_date,eligible FROM label_flow "
        "WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? ORDER BY serial_number"
    )
    feature_sql = "SELECT * FROM feature_rows WHERE decision_date=? ORDER BY serial_number"
    matrix: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    stats: list[dict] = []
    for date_text in dates:
        label_rows = panel.execute(label_sql, (LABEL_RUN_ID, LABEL_SPLIT, HORIZON, date_text)).fetchall()
        labels_by_serial = {str(row["serial_number"]): row for row in label_rows if int(row["eligible"]) == 1}
        rows = feature.execute(feature_sql, (date_text,)).fetchall()
        if len(labels_by_serial) != len(rows):
            raise BaselineStopped(f"feature/eligible label count differs on {date_text}")
        negative_serials = [serial for serial, row in labels_by_serial.items() if row["label"] == 0 and row["status"] == "negative_observed"]
        positive_serials = [serial for serial, row in labels_by_serial.items() if row["label"] == 1 and row["status"] in {"positive_observed", "positive_with_gap"}]
        selected_negative, sample_digest = sample_negative_indices(date_text, negative_serials, len(negative_serials))
        negative_weight = float(len(negative_serials) / len(selected_negative)) if selected_negative else 0.0
        sampled = 0
        for row in rows:
            serial = str(row["serial_number"])
            label_row = labels_by_serial.get(serial)
            if label_row is None:
                raise BaselineStopped(f"feature key has no eligible label on {date_text}/{serial}")
            status = str(label_row["status"])
            if status not in KNOWN_STATUSES:
                continue
            if status.startswith("positive"):
                include, weight, target = True, 1.0, 1
            else:
                include, weight, target = serial in selected_negative, negative_weight, 0
                sampled += int(include)
            if include:
                matrix.append([row_value(row, name) for name in history_columns])
                labels.append(target)
                weights.append(weight)
        stats.append({
            "decision_date": date_text,
            "eligible_count": len(rows),
            "positive_count": len(positive_serials),
            "negative_count": len(negative_serials),
            "sampled_negative_count": len(selected_negative),
            "negative_inverse_weight": negative_weight,
            "sample_key_sha256": sample_digest,
            "known_unknown_excluded": len(rows) - len(positive_serials) - len(negative_serials),
        })
        if len(matrix) >= MAX_TRAIN_ROWS:
            raise BaselineStopped(f"training sample exceeds {MAX_TRAIN_ROWS} rows")
        if len(stats) % 5 == 0:
            check_resources(start, label=f"training sample after {date_text}")
    if not matrix or len(set(labels)) != 2:
        raise BaselineStopped("training sample does not contain both classes")
    raw = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    raw_weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0):
        raise BaselineStopped("invalid training weights")
    normalized = raw_weights / float(raw_weights.mean())
    return raw, y, normalized, {"dates": stats, "rows": len(labels), "positive_rows": int(y.sum()), "negative_rows": int((y == 0).sum())}, history_columns


def weighted_preprocess(raw: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict]:
    if raw.ndim != 2 or len(raw) != len(weights):
        raise BaselineStopped("invalid preprocessing matrix shape")
    weighted_total = float(weights.sum())
    if weighted_total <= 0:
        raise BaselineStopped("zero preprocessing weight")
    valid = np.isfinite(raw)
    observed = valid.sum(axis=0).astype(int)
    means = np.zeros(raw.shape[1], dtype=np.float64)
    for j in range(raw.shape[1]):
        if observed[j]:
            means[j] = float(np.sum(weights[valid[:, j]] * raw[valid[:, j], j]) / np.sum(weights[valid[:, j]]))
    filled = np.where(valid, raw, means)
    center = np.sum(filled * weights[:, None], axis=0) / weighted_total
    variance = np.sum(((filled - center) ** 2) * weights[:, None], axis=0) / weighted_total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[(~np.isfinite(scale)) | (scale < 1e-12)] = 1.0
    transformed = (filled - center) / scale
    if not np.isfinite(transformed).all():
        raise BaselineStopped("preprocessing produced non-finite values")
    return transformed, {
        "imputation_mean": means.tolist(),
        "standardization_mean": center.tolist(),
        "standardization_scale": scale.tolist(),
        "observed_count": observed.tolist(),
        "weighted_total": weighted_total,
    }


def apply_preprocess(raw: np.ndarray, params: Mapping[str, Sequence[float]]) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    means = np.asarray(params["imputation_mean"], dtype=np.float64)
    center = np.asarray(params["standardization_mean"], dtype=np.float64)
    scale = np.asarray(params["standardization_scale"], dtype=np.float64)
    return (np.where(np.isfinite(values), values, means) - center) / scale


def _fit_worker(connection, x: np.ndarray, y: np.ndarray, weights: np.ndarray, params: dict) -> None:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", tol=1e-6,
                max_iter=1000, fit_intercept=True, class_weight=None,
                random_state=20260913,
            )
            model.fit(x, y, sample_weight=weights)
        convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        if convergence:
            raise BaselineStopped("; ".join(convergence))
        payload = {
            "coef": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
            "n_iter": int(model.n_iter_[0]),
            "params": model.get_params(deep=True),
            "warnings": [str(item.message) for item in caught],
        }
        connection.send((True, payload))
    except Exception as exc:  # pragma: no cover - exercised through parent error path
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def fit_one_model(name: str, raw: np.ndarray, y: np.ndarray, weights: np.ndarray, indices: Sequence[int], start: float) -> tuple[dict, dict, np.ndarray]:
    selected = raw[:, list(indices)]
    transformed, prep = weighted_preprocess(selected, weights)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    parent, child = ctx.Pipe(False)
    process = ctx.Process(target=_fit_worker, args=(child, transformed, y, weights, {"model": name}))
    process.start()
    child.close()
    started = time.monotonic()
    payload = None
    try:
        while process.is_alive() or parent.poll():
            if parent.poll(0.5):
                ok, value = parent.recv()
                if not ok:
                    raise BaselineStopped(f"{name} fit failed: {value}")
                payload = value
                break
            if time.monotonic() - started >= FIT_SECONDS_PER_MODEL:
                process.terminate()
                process.join(5)
                raise BaselineStopped(f"{name} fit exceeded {FIT_SECONDS_PER_MODEL}s")
            check_resources(start, label=f"fit {name}")
        process.join(5)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    if payload is None:
        raise BaselineStopped(f"{name} fit exited without result (code={process.exitcode})")
    coefficient = np.asarray(payload["coef"], dtype=np.float64)
    intercept = float(payload["intercept"])
    if not np.isfinite(coefficient).all() or not math.isfinite(intercept):
        raise BaselineStopped(f"{name} produced non-finite coefficients")
    scores = transformed @ coefficient + intercept
    if not np.isfinite(scores).all():
        raise BaselineStopped(f"{name} produced non-finite training scores")
    payload["preprocessing"] = prep
    payload["feature_count"] = len(indices)
    return payload, prep, scores


def event_opportunities(panel: sqlite3.Connection) -> dict[str, dict]:
    result: dict[str, dict] = {}
    cursor = panel.execute("SELECT serial_number,date,failure FROM daily WHERE model=? ORDER BY serial_number,date", (MODEL_NAME,))
    current = None
    rows: list[tuple[str, int]] = []

    def flush(serial: str | None, device_rows: list[tuple[str, int]]) -> None:
        if serial is None:
            return
        dates = {dt.date.fromisoformat(day) for day, _flag in device_rows}
        failures = sorted(dt.date.fromisoformat(day) for day, flag in device_rows if flag)
        if not failures:
            return
        first = failures[0]
        if not EVENT_START <= first <= EVENT_END:
            return
        candidates = []
        for offset in range(1, HORIZON + 1):
            decision = first - dt.timedelta(days=offset)
            history = sum(decision - dt.timedelta(days=i) in dates for i in range(14))
            if SCORE_START <= decision <= SCORE_END and decision in dates and history >= 12:
                candidates.append(decision)
        result[serial] = {
            "event_key": serial,
            "first_failure_date": first.isoformat(),
            "opportunity": int(bool(candidates)),
            "opportunity_dates": [item.isoformat() for item in sorted(candidates)],
        }

    for row in cursor:
        serial = str(row["serial_number"])
        if current is None:
            current = serial
        if serial != current:
            flush(current, rows)
            rows = []
            current = serial
        rows.append((str(row["date"]), int(row["failure"] or 0)))
    flush(current, rows)
    return result


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    cumulative = np.cumsum(y)
    positions = np.arange(1, len(y) + 1)
    precision = cumulative / positions
    return float(np.sum(precision[y == 1]) / positives)


def quantile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE model_runs(
        model TEXT PRIMARY KEY, status TEXT NOT NULL, feature_count INTEGER NOT NULL,
        training_rows INTEGER NOT NULL, positive_training_rows INTEGER NOT NULL,
        negative_training_rows INTEGER NOT NULL, n_iter INTEGER, intercept REAL,
        average_precision_known REAL, warning_json TEXT NOT NULL, params_json TEXT NOT NULL
    );
    CREATE TABLE sampling_stats(
        decision_date TEXT PRIMARY KEY, eligible_count INTEGER NOT NULL,
        positive_count INTEGER NOT NULL, negative_count INTEGER NOT NULL,
        sampled_negative_count INTEGER NOT NULL, negative_inverse_weight REAL NOT NULL,
        sample_key_sha256 TEXT NOT NULL, known_unknown_excluded INTEGER NOT NULL
    );
    CREATE TABLE preprocessing_stats(
        model TEXT NOT NULL, feature_name TEXT NOT NULL, imputation_mean REAL NOT NULL,
        standardization_mean REAL NOT NULL, standardization_scale REAL NOT NULL,
        observed_count INTEGER NOT NULL, PRIMARY KEY(model, feature_name)
    );
    CREATE TABLE model_coefficients(model TEXT NOT NULL, feature_name TEXT NOT NULL, coefficient REAL NOT NULL, PRIMARY KEY(model, feature_name));
    CREATE TABLE model_scores(
        decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, tie_break_sha256 TEXT NOT NULL,
        status TEXT NOT NULL, label INTEGER, first_failure_date TEXT,
        age_lr_score REAL NOT NULL, current_lr_score REAL NOT NULL, history_lr_score REAL NOT NULL,
        PRIMARY KEY(decision_date, serial_number)
    );
    CREATE TABLE model_alerts(
        model TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL,
        score REAL NOT NULL, tie_break_sha256 TEXT NOT NULL, status TEXT NOT NULL, label INTEGER,
        first_failure_date TEXT, event_key TEXT, event_hit INTEGER NOT NULL, lead_days INTEGER,
        PRIMARY KEY(model, decision_date, serial_number)
    );
    CREATE TABLE model_daily(
        model TEXT NOT NULL, decision_date TEXT NOT NULL, eligible_count INTEGER NOT NULL,
        budget_k INTEGER NOT NULL, cooldown_excluded INTEGER NOT NULL, alerts_count INTEGER NOT NULL,
        known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL, unknown_alerts INTEGER NOT NULL,
        PRIMARY KEY(model, decision_date)
    );
    CREATE TABLE model_event_summary(
        model TEXT NOT NULL, event_key TEXT NOT NULL, first_failure_date TEXT NOT NULL,
        opportunity INTEGER NOT NULL, hit INTEGER NOT NULL, earliest_alert_date TEXT,
        earliest_lead_days INTEGER, PRIMARY KEY(model,event_key)
    );
    CREATE TABLE model_strata(
        model TEXT NOT NULL, stratum_type TEXT NOT NULL, stratum TEXT NOT NULL,
        eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
        known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL,
        unknown_alerts INTEGER NOT NULL, PRIMARY KEY(model,stratum_type,stratum)
    );
    CREATE TABLE model_metrics(
        model TEXT PRIMARY KEY, eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
        known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL, unknown_alerts INTEGER NOT NULL,
        unknown_alert_ratio REAL NOT NULL, precision_lower_bound REAL NOT NULL, precision_upper_bound REAL NOT NULL,
        known_outcome_precision REAL, confirmed_no_hit_per_1000_device_days REAL NOT NULL,
        event_total INTEGER NOT NULL, event_opportunity_total INTEGER NOT NULL, event_hits INTEGER NOT NULL,
        event_recall_at_opportunity REAL NOT NULL, all_event_capture REAL NOT NULL,
        early_event_hits_ge2_days INTEGER NOT NULL, early_event_hits_ge3_days INTEGER NOT NULL,
        early_recall_at_opportunity_ge2_days REAL NOT NULL, early_recall_at_opportunity_ge3_days REAL NOT NULL,
        hit_only_ratio_ge2_days REAL, hit_only_ratio_ge3_days REAL,
        earliest_lead_count INTEGER NOT NULL, earliest_lead_median REAL,
        earliest_lead_q25 REAL, earliest_lead_q75 REAL, repeated_alert_devices INTEGER NOT NULL,
        max_alerts_per_device INTEGER NOT NULL, minimum_alert_gap_days INTEGER,
        average_precision_known REAL, metric_scope TEXT NOT NULL
    );
    """)


def select_alert_indices(scores: np.ndarray, rows: Sequence[sqlite3.Row], decision: dt.date, last_alert: dict[str, dt.date]) -> tuple[list[int], int]:
    ranked = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), str(rows[i]["tie_break_sha256"]), str(rows[i]["serial_number"])))
    cooldown_excluded = 0
    available: list[int] = []
    for index in ranked:
        serial = str(rows[index]["serial_number"])
        previous = last_alert.get(serial)
        if previous is not None and (decision - previous).days <= COOLDOWN_DAYS:
            cooldown_excluded += 1
        else:
            available.append(index)
    budget = (len(rows) + BUDGET_DENOMINATOR - 1) // BUDGET_DENOMINATOR if rows else 0
    selected = available[:budget]
    for index in selected:
        last_alert[str(rows[index]["serial_number"])] = decision
    return selected, cooldown_excluded


def build_output(panel: sqlite3.Connection, feature: sqlite3.Connection, config: dict, input_facts: dict, raw: np.ndarray, y: np.ndarray, weights: np.ndarray, training_stats: dict, history_columns: list[str], model_payloads: dict[str, dict], start: float) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_DB.with_name(OUTPUT_DB.name + ".partial")
    if OUTPUT_DB.exists():
        raise BaselineStopped(f"refusing to overwrite completed output: {OUTPUT_DB}")
    if partial.exists():
        raise BaselineStopped(f"preserving existing failed partial output: {partial}")
    connection = sqlite3.connect(partial)
    connection.row_factory = sqlite3.Row
    try:
        create_schema(connection)
        config_hash = canonical_hash(config)
        for key, value in {
            "run_status": "running", "config_hash": config_hash,
            "config_path": str(CONFIG_PATH), "input_freeze_path": str(FREEZE_PATH),
            "panel_db_sha256": sha256_file(PANEL_DB), "feature_db_sha256": sha256_file(FEATURE_DB),
            "replay_db_sha256": sha256_file(REPLAY_DB), "training_approval": "false",
            "score_start": SCORE_START.isoformat(), "score_end": SCORE_END.isoformat(),
            "label_cutoff": DATASET_END.isoformat(), "horizon_days": str(HORIZON),
            "code_sha256": sha256_file(Path(__file__)), "environment_python": os.sys.executable,
        }.items():
            connection.execute("INSERT INTO metadata VALUES (?,?)", (key, value))
        for row in training_stats["dates"]:
            connection.execute("INSERT INTO sampling_stats VALUES (?,?,?,?,?,?,?,?)", (
                row["decision_date"], row["eligible_count"], row["positive_count"], row["negative_count"],
                row["sampled_negative_count"], row["negative_inverse_weight"], row["sample_key_sha256"], row["known_unknown_excluded"],
            ))
        for model in MODEL_ORDER:
            indices = [history_columns.index(name) for name in config["models"][model]]
            payload = model_payloads[model]
            prep = payload["preprocessing"]
            for name, mean, center, scale, observed in zip(config["models"][model], prep["imputation_mean"], prep["standardization_mean"], prep["standardization_scale"], prep["observed_count"]):
                connection.execute("INSERT INTO preprocessing_stats VALUES (?,?,?,?,?,?)", (model, name, mean, center, scale, observed))
            for name, coefficient in zip(config["models"][model], payload["coef"]):
                connection.execute("INSERT INTO model_coefficients VALUES (?,?,?)", (model, name, coefficient))
            known_scores = payload.pop("known_scores", np.empty(0, dtype=float))
            known_labels = payload.pop("known_labels", np.empty(0, dtype=np.int8))
            ap = average_precision(np.asarray(known_scores), np.asarray(known_labels))
            connection.execute("INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                model, "fitted", len(indices), training_stats["rows"], training_stats["positive_rows"], training_stats["negative_rows"],
                payload["n_iter"], payload["intercept"], ap, json.dumps(payload.get("warnings", []), ensure_ascii=False), json.dumps(payload.get("params", {}), ensure_ascii=False, sort_keys=True),
            ))

        event_map = event_opportunities(panel)
        method_alerts: dict[str, list[dict]] = {model: [] for model in MODEL_ORDER}
        method_events: dict[str, dict[str, dict]] = {model: {key: dict(value, hit=0, earliest_alert_date=None, earliest_lead_days=None) for key, value in event_map.items()} for model in MODEL_ORDER}
        method_daily: dict[str, list[dict]] = {model: [] for model in MODEL_ORDER}
        method_strata: dict[str, dict[tuple[str, str], dict[str, int]]] = {model: defaultdict(lambda: {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0}) for model in MODEL_ORDER}
        last_alert = {model: {} for model in MODEL_ORDER}
        score_arrays = {model: [] for model in MODEL_ORDER}
        score_labels = {model: [] for model in MODEL_ORDER}
        feature_sql = "SELECT * FROM feature_rows WHERE decision_date=? ORDER BY serial_number"
        label_sql = "SELECT serial_number,label,status,first_failure_date FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND decision_date=? AND eligible=1"
        dates = [str(row[0]) for row in feature.execute("SELECT DISTINCT decision_date FROM feature_rows ORDER BY decision_date")]
        for day_index, date_text in enumerate(dates, 1):
            day = dt.date.fromisoformat(date_text)
            rows = feature.execute(feature_sql, (date_text,)).fetchall()
            labels_by_serial = {str(row["serial_number"]): row for row in panel.execute(label_sql, (LABEL_RUN_ID, LABEL_SPLIT, HORIZON, date_text))}
            if len(rows) != len(labels_by_serial):
                raise BaselineStopped(f"score feature/label count differs on {date_text}")
            raw_day = np.asarray([[row_value(row, name) for name in history_columns] for row in rows], dtype=np.float64)
            score_by_model: dict[str, np.ndarray] = {}
            for model in MODEL_ORDER:
                payload = model_payloads[model]
                x = apply_preprocess(raw_day[:, [history_columns.index(name) for name in config["models"][model]]], payload["preprocessing"])
                score_by_model[model] = x @ np.asarray(payload["coef"], dtype=float) + float(payload["intercept"])
                if not np.isfinite(score_by_model[model]).all():
                    raise BaselineStopped(f"non-finite {model} score on {date_text}")
                for index, row in enumerate(rows):
                    label_row = labels_by_serial[str(row["serial_number"])]
                    label_value = label_row["label"]
                    if label_value in (0, 1):
                        score_labels[model].append(int(label_value))
                        score_arrays[model].append(float(score_by_model[model][index]))
            for index, row in enumerate(rows):
                label_row = labels_by_serial.get(str(row["serial_number"]))
                if label_row is None:
                    raise BaselineStopped(f"missing label for score key {date_text}/{row['serial_number']}")
                connection.execute("INSERT INTO model_scores VALUES (?,?,?,?,?,?,?,?,?)", (
                    date_text, str(row["serial_number"]), str(row["tie_break_sha256"]), str(label_row["status"]), label_row["label"], label_row["first_failure_date"],
                    float(score_by_model["age_lr"][index]), float(score_by_model["current_lr"][index]), float(score_by_model["history_lr"][index]),
                ))
                current_missing = "any_current_missing" if any(int(row[col] or 0) for col in CURRENT_MISSING_COLUMNS) else "no_current_missing"
                history_bucket = str(int(row["history_observations_14"]))
                for model in MODEL_ORDER:
                    for stype, sval in (("month", date_text[:7]), ("current_missing", current_missing), ("history_observations_14", history_bucket)):
                        method_strata[model][(stype, sval)]["eligible"] += 1
            for model in MODEL_ORDER:
                selected, cooldown_excluded = select_alert_indices(score_by_model[model], rows, day, last_alert[model])
                daily = {"eligible": len(rows), "budget": (len(rows) + 999) // 1000 if rows else 0, "cooldown": cooldown_excluded, "alerts": len(selected), "known_hit": 0, "known_no_hit": 0, "unknown": 0}
                for index in selected:
                    row = rows[index]
                    label_row = labels_by_serial[str(row["serial_number"])]
                    label_value = label_row["label"]
                    if label_value == 1: daily["known_hit"] += 1
                    elif label_value == 0: daily["known_no_hit"] += 1
                    else: daily["unknown"] += 1
                    failure = label_row["first_failure_date"]
                    event_key = None
                    event_hit = 0
                    lead_days = None
                    if failure:
                        failure_day = dt.date.fromisoformat(str(failure))
                        if EVENT_START <= failure_day <= EVENT_END:
                            event_key = str(row["serial_number"])
                            delta = (failure_day - day).days
                            if 1 <= delta <= HORIZON:
                                event_hit = 1
                                lead_days = delta
                                record = method_events[model].get(event_key)
                                if record is not None and (record["earliest_lead_days"] is None or delta > 0 and delta < record["earliest_lead_days"]):
                                    record["hit"] = 1
                                    record["earliest_alert_date"] = date_text
                                    record["earliest_lead_days"] = delta
                    method_alerts[model].append({"decision_date": date_text, "serial_number": str(row["serial_number"]), "score": float(score_by_model[model][index]), "tie_break_sha256": str(row["tie_break_sha256"]), "status": str(label_row["status"]), "label": label_value, "first_failure_date": failure, "event_key": event_key, "event_hit": event_hit, "lead_days": lead_days})
                    current_missing = "any_current_missing" if any(int(row[col] or 0) for col in CURRENT_MISSING_COLUMNS) else "no_current_missing"
                    history_bucket = str(int(row["history_observations_14"]))
                    for stype, sval in (("month", date_text[:7]), ("current_missing", current_missing), ("history_observations_14", history_bucket)):
                        bucket = method_strata[model][(stype, sval)]
                        bucket["alerts"] += 1
                        if label_value == 1: bucket["known_hit"] += 1
                        elif label_value == 0: bucket["known_no_hit"] += 1
                        else: bucket["unknown"] += 1
                method_daily[model].append({"date": date_text, **daily})
            if day_index % 5 == 0:
                connection.commit()
                check_resources(start, label=f"scoring after {date_text}")
        for model in MODEL_ORDER:
            for row in method_alerts[model]:
                connection.execute("INSERT INTO model_alerts VALUES (?,?,?,?,?,?,?,?,?,?,?)", (model, row["decision_date"], row["serial_number"], row["score"], row["tie_break_sha256"], row["status"], row["label"], row["first_failure_date"], row["event_key"], row["event_hit"], row["lead_days"]))
            for row in method_daily[model]:
                connection.execute("INSERT INTO model_daily VALUES (?,?,?,?,?,?,?,?,?)", (model, row["date"], row["eligible"], row["budget"], row["cooldown"], row["alerts"], row["known_hit"], row["known_no_hit"], row["unknown"]))
            for event_key, event in method_events[model].items():
                connection.execute("INSERT INTO model_event_summary VALUES (?,?,?,?,?,?,?)", (model, event_key, event["first_failure_date"], event["opportunity"], event["hit"], event["earliest_alert_date"], event["earliest_lead_days"]))
            for (stype, sval), bucket in method_strata[model].items():
                connection.execute("INSERT INTO model_strata VALUES (?,?,?,?,?,?,?,?)", (model, stype, sval, bucket["eligible"], bucket["alerts"], bucket["known_hit"], bucket["known_no_hit"], bucket["unknown"]))
            alerts = method_alerts[model]
            known_hit = sum(row["label"] == 1 for row in alerts)
            known_no_hit = sum(row["label"] == 0 for row in alerts)
            unknown = sum(row["label"] is None for row in alerts)
            events = list(method_events[model].values())
            opportunity_events = [event for event in events if event["opportunity"]]
            hit_events = [event for event in events if event["hit"]]
            opportunity_hits = [event for event in opportunity_events if event["hit"]]
            leads = [int(event["earliest_lead_days"]) for event in opportunity_hits if event["earliest_lead_days"] is not None]
            ge2 = sum(value >= 2 for value in leads)
            ge3 = sum(value >= 3 for value in leads)
            by_device = Counter(row["serial_number"] for row in alerts)
            gaps = []
            for serial in by_device:
                days = sorted(dt.date.fromisoformat(row["decision_date"]) for row in alerts if row["serial_number"] == serial)
                gaps.extend((right - left).days for left, right in zip(days, days[1:]))
            known_scores = np.asarray(score_arrays[model], dtype=float)
            known_labels = np.asarray(score_labels[model], dtype=np.int8)
            ap = average_precision(known_scores, known_labels)
            metric = (model, sum(row["eligible"] for row in method_daily[model]), len(alerts), known_hit, known_no_hit, unknown,
                unknown / len(alerts) if alerts else 0.0, known_hit / len(alerts) if alerts else 0.0, (known_hit + unknown) / len(alerts) if alerts else 0.0,
                known_hit / (known_hit + known_no_hit) if known_hit + known_no_hit else None, known_no_hit / sum(row["eligible"] for row in method_daily[model]) * 1000 if method_daily[model] else 0.0,
                len(events), len(opportunity_events), len(opportunity_hits), len(opportunity_hits) / len(opportunity_events) if opportunity_events else 0.0,
                len(hit_events) / len(events) if events else 0.0, ge2, ge3, ge2 / len(opportunity_events) if opportunity_events else 0.0, ge3 / len(opportunity_events) if opportunity_events else 0.0,
                ge2 / len(leads) if leads else None, ge3 / len(leads) if leads else None, len(leads), quantile(leads, .5), quantile(leads, .25), quantile(leads, .75), sum(count > 1 for count in by_device.values()), max(by_device.values(), default=0), min(gaps) if gaps else None, ap,
                "training-period known-outcome diagnostic; unknown outcomes excluded from AP")
            connection.execute("INSERT INTO model_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", metric)
        connection.execute("UPDATE metadata SET value='complete' WHERE key='run_status'")
        connection.commit()
        connection.close()
        os.replace(partial, OUTPUT_DB)
    except Exception:
        connection.commit()
        connection.close()
        raise
    return {"status": "pass", "output_db": str(OUTPUT_DB), "training_rows": training_stats["rows"], "eligible_rows": 2_860_459, "alerts_per_model": {model: len(method_alerts[model]) for model in MODEL_ORDER}, "elapsed_seconds": round(time.monotonic() - start, 3)}


def completed_output_reentry(config: dict) -> dict | None:
    """Return a read-only completion result for a matching final output."""
    if not OUTPUT_DB.exists():
        return None
    connection = connect_immutable(OUTPUT_DB)
    try:
        meta = metadata(connection)
        expected_config = canonical_hash(config)
        if meta.get("run_status") != "complete":
            raise BaselineStopped("existing baseline output is not complete")
        if meta.get("config_hash") != expected_config:
            raise BaselineStopped("existing baseline output config does not match the locked config")
        model_count = int(connection.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0])
        score_count = int(connection.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0])
        alert_count = int(connection.execute("SELECT COUNT(*) FROM model_alerts").fetchone()[0])
        if model_count != len(MODEL_ORDER) or score_count != 2_860_459 or alert_count != 8_847:
            raise BaselineStopped("existing baseline output is incomplete")
        return {
            "status": "already_complete",
            "output_db": str(OUTPUT_DB),
            "config_hash": expected_config,
            "model_count": model_count,
            "score_count": score_count,
            "alert_count": alert_count,
            "mutation": "read_only",
        }
    finally:
        connection.close()


def run(config_path: Path = CONFIG_PATH) -> dict:
    start = time.monotonic()
    config = read_json(config_path)
    freeze = verify_freeze(config)
    check_resources(start, label="preflight")
    reentry = completed_output_reentry(config)
    if reentry is not None:
        return reentry
    panel = feature = replay = None
    try:
        panel = connect_immutable(PANEL_DB)
        feature = connect_immutable(FEATURE_DB)
        replay = connect_immutable(REPLAY_DB)
        facts = verify_databases(panel, feature, replay, config)
        raw, y, weights, sampling, history_columns = collect_training_sample(panel, feature, config, start)
        training_payloads: dict[str, dict] = {}
        for model in MODEL_ORDER:
            indices = [history_columns.index(name) for name in config["models"][model]]
            payload, _prep, _scores = fit_one_model(model, raw, y, weights, indices, start)
            known_mask = np.isin(y, (0, 1))
            transformed = apply_preprocess(raw[:, indices], payload["preprocessing"])
            all_scores = transformed @ np.asarray(payload["coef"], dtype=float) + float(payload["intercept"])
            payload["known_scores"] = all_scores[known_mask]
            payload["known_labels"] = y[known_mask]
            training_payloads[model] = payload
        result = build_output(panel, feature, config, facts, raw, y, weights, sampling, history_columns, training_payloads, start)
        evidence = {
            "status": "pass", "stage": "simple_baseline_v1", "result": result,
            "config_hash": canonical_hash(config), "freeze_sha256": sha256_file(FREEZE_PATH),
            "config_sha256": sha256_file(config_path), "code_sha256": sha256_file(Path(__file__)),
            "environment": {"python": os.sys.executable, "python_version": os.sys.version, "numpy": np.__version__},
            "training": {"rows": sampling["rows"], "positive_rows": sampling["positive_rows"], "negative_rows": sampling["negative_rows"], "unknown_excluded": sum(row["known_unknown_excluded"] for row in sampling["dates"])},
            "models": {model: {"features": config["models"][model], "n_iter": training_payloads[model]["n_iter"], "intercept": training_payloads[model]["intercept"], "warnings": training_payloads[model]["warnings"]} for model in MODEL_ORDER},
            "training_approval": False, "q3_q4_access": False, "remote_setup": False,
        }
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        evidence_path = EVIDENCE_DIR / "run_v1.json"
        if evidence_path.exists():
            raise BaselineStopped(f"refusing to overwrite evidence: {evidence_path}")
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return evidence
    except Exception as exc:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        attempt = EVIDENCE_DIR / f"attempt_failed_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        attempt.write_text(json.dumps({"status": "fail", "stage": "simple_baseline_v1", "reason": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.monotonic() - start, 3), "training_approval": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        for connection in (panel, feature, replay):
            if connection is not None:
                connection.close()


def probe() -> dict:
    """Small, dependency-backed checks used before touching real data."""
    raw = np.asarray([[1.0, np.nan, 3.0], [2.0, 4.0, 3.0], [3.0, 6.0, 3.0]], dtype=float)
    weights = np.asarray([1.0, 2.0, 1.0])
    transformed, prep = weighted_preprocess(raw, weights)
    if not np.isfinite(transformed).all() or prep["observed_count"] != [3, 2, 3]:
        raise BaselineStopped("preprocessing probe failed")
    selected, digest = sample_negative_indices("2023-01-15", ["b", "a", "c", "d"], 4)
    expected_selected, _ = sample_negative_indices("2023-01-15", ["b", "a", "c", "d"], 4)
    if selected != expected_selected or len(selected) != 1 or len(digest) != 64:
        raise BaselineStopped("sampling probe failed")
    rows = [{"serial_number": "a", "tie_break_sha256": "0" * 64}, {"serial_number": "b", "tie_break_sha256": "1" * 64}]
    scores = np.asarray([2.0, 1.0])
    last = {}
    first, _ = select_alert_indices(scores, rows, dt.date(2023, 1, 1), last)
    second, _ = select_alert_indices(scores, rows, dt.date(2023, 1, 8), last)
    third, _ = select_alert_indices(scores, rows, dt.date(2023, 1, 9), last)
    if first != [0] or second != [1] or third != [0]:
        raise BaselineStopped("cooldown probe failed")
    return {"status": "pass", "checks": ["weighted_missing_and_constant", "deterministic_sampling", "t+1/t+7/t+8_cooldown"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    result = probe() if args.probe else run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
