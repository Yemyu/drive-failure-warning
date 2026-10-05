"""Fit one fixed histogram gradient boosting baseline and replay Q3.

The score pass reads only feature columns and the frozen control scores.  Q3
labels are opened only after the score database has been committed and closed.
The run is deliberately separate from the historical baseline executor so a
new model cannot alter the frozen results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import resource
import shutil
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from typing import Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/nonlinear_baseline_v1.json"
MODEL_NAME = "history_hgb_v1"
CONTROL_MODELS = ("current_lr", "history_lr")
ALL_MODELS = CONTROL_MODELS + (MODEL_NAME,)
HORIZON = 7
MODEL_TYPE = "ST4000DM000"
SCORE_START = dt.date(2023, 7, 1)
SCORE_END = dt.date(2023, 9, 23)
EVENT_START = dt.date(2023, 7, 8)
EVENT_END = dt.date(2023, 9, 24)
TRAIN_START = dt.date(2023, 1, 15)
TRAIN_END = dt.date(2023, 6, 23)
DATASET_END = dt.date(2023, 6, 30)
Q3_LABEL_RUN = "validation_q3_h7_v1"
Q3_SPLIT = "validation"
TRAIN_LABEL_RUN = "train_q1q2_verified_h7_v1"
TRAIN_SPLIT = "train"
CHECK_EVERY = 10_000


class ExperimentStopped(RuntimeError):
    """A declared input, resource, numerical, or reproducibility failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentStopped(f"cannot read JSON {path}: {exc}") from exc


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def process_rss_bytes(pid: int) -> int:
    if pid == os.getpid():
        return rss_bytes()
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.is_file():
        for line in proc_status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], check=False, capture_output=True, text=True, timeout=1)
        values = result.stdout.strip().split()
        return int(values[-1]) * 1024 if values else 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def connect_immutable(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ExperimentStopped(f"missing input database: {path}")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ExperimentStopped(f"non-empty immutable sidecar: {sidecar}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key,value FROM metadata")}


def qident(name: str) -> str:
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(char.isalnum() or char == "_" for char in name):
        raise ExperimentStopped(f"unsafe SQL column name: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def owned_bytes(data_dir: Path, evidence_dir: Path) -> int:
    total = 0
    for base in (data_dir, evidence_dir):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def resource_snapshot(data_dir: Path, evidence_dir: Path, child_pid: int | None = None) -> dict[str, int]:
    parent = rss_bytes()
    child = process_rss_bytes(child_pid) if child_pid else 0
    return {
        "rss_max_bytes": parent,
        "child_rss_bytes": child,
        "total_rss_bytes": parent + child,
        "owned_bytes": owned_bytes(data_dir, evidence_dir),
        "free_bytes": shutil.disk_usage(ROOT).free,
    }


def check_resources(start: float, config: Mapping[str, object], data_dir: Path, evidence_dir: Path, *, label: str, child_pid: int | None = None) -> dict[str, int]:
    snapshot = resource_snapshot(data_dir, evidence_dir, child_pid)
    resources = config["resources"]
    if snapshot["total_rss_bytes"] >= int(resources["rss_limit_bytes"]):
        raise ExperimentStopped(f"RSS limit during {label}: {snapshot}")
    if snapshot["owned_bytes"] >= int(resources["new_project_bytes_limit"]):
        raise ExperimentStopped(f"output limit during {label}: {snapshot}")
    if snapshot["free_bytes"] < int(resources["minimum_free_bytes"]):
        raise ExperimentStopped(f"free-space limit during {label}: {snapshot}")
    elapsed = time.monotonic() - start
    if elapsed >= int(resources["stage_seconds"]):
        raise ExperimentStopped(f"stage timeout during {label}: {snapshot}")
    return {"label": label, "elapsed_seconds": round(elapsed, 3), **snapshot}


def sampling_hash(date_text: str, serial: str) -> str:
    return hashlib.sha256(f"fit-v1|20260913|{date_text}|{serial}".encode("utf-8")).hexdigest()


def sample_negative_indices(date_text: str, serials: Sequence[str]) -> tuple[set[str], str, float]:
    sample_count = min(len(serials), max(1, len(serials) // 20)) if serials else 0
    ranked = sorted((sampling_hash(date_text, serial), serial) for serial in serials)
    selected = {serial for _digest, serial in ranked[:sample_count]}
    digest = hashlib.sha256("".join(f"{digest}:{serial}\n" for digest, serial in ranked[:sample_count]).encode("utf-8")).hexdigest()
    inverse_weight = float(len(serials) / sample_count) if sample_count else 0.0
    return selected, digest, inverse_weight


def load_feature_map(model_db: sqlite3.Connection, feature_db: sqlite3.Connection) -> tuple[list[dict], list[str], dict[str, np.ndarray]]:
    rows = model_db.execute("SELECT * FROM model_feature_map WHERE model='history_lr' ORDER BY position").fetchall()
    if len(rows) != 122 or [int(row["position"]) for row in rows] != list(range(122)):
        raise ExperimentStopped("history model feature map is not the locked 122-column map")
    columns = {str(row["name"]) for row in feature_db.execute("PRAGMA table_info(feature_rows)")}
    mapping = []
    for row in rows:
        feature = str(row["feature_name"])
        source = str(row["source_name"])
        if source not in columns:
            raise ExperimentStopped(f"feature map source missing: {source}")
        mapping.append({"feature_name": feature, "source_name": source, "is_missing_indicator": int(row["is_missing_indicator"])})
    stats_rows = model_db.execute("SELECT * FROM preprocessing_stats WHERE model='history_lr'").fetchall()
    stats = {str(row["feature_name"]): row for row in stats_rows}
    if set(stats) != {item["feature_name"] for item in mapping}:
        raise ExperimentStopped("preprocessing statistics do not match feature map")
    vectors = {}
    for key in ("imputation_mean", "standardization_mean", "standardization_scale", "observed_count"):
        values = []
        for item in mapping:
            row = stats[item["feature_name"]]
            value = float(row[key])
            if not math.isfinite(value) or (key == "standardization_scale" and value <= 0):
                raise ExperimentStopped(f"invalid preprocessing value for {item['feature_name']}")
            values.append(value)
        vectors[key] = np.asarray(values, dtype=np.float64)
    return mapping, [item["source_name"] for item in mapping], vectors


def row_matrix(rows: Sequence[Mapping[str, object]], mapping: Sequence[Mapping[str, object]]) -> np.ndarray:
    matrix = np.empty((len(rows), len(mapping)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, item in enumerate(mapping):
            value = row[item["source_name"]]
            if item["is_missing_indicator"]:
                feature_name = str(item["feature_name"])
                source_name = str(item["source_name"])
                if feature_name == source_name:
                    # The six current-missing columns are already 0/1 values.
                    # Only the appended ``__missing`` columns are derived from
                    # whether their source value is NULL.
                    if value is None:
                        raise ExperimentStopped(f"physical missing flag is NULL: {source_name}")
                    number = float(value)
                    if not math.isfinite(number) or number not in (0.0, 1.0):
                        raise ExperimentStopped(f"physical missing flag is not 0/1: {source_name}")
                    matrix[i, j] = number
                else:
                    if feature_name != source_name + "__missing":
                        raise ExperimentStopped(f"missing indicator mapping is ambiguous: {feature_name} <- {source_name}")
                    matrix[i, j] = 1.0 if value is None else 0.0
            elif value is None:
                matrix[i, j] = np.nan
            else:
                matrix[i, j] = float(value)
                if not math.isfinite(matrix[i, j]):
                    raise ExperimentStopped(f"non-finite value in {item['source_name']}")
    return matrix


def apply_preprocess(raw: np.ndarray, vectors: Mapping[str, np.ndarray]) -> np.ndarray:
    result = (np.where(np.isfinite(raw), raw, vectors["imputation_mean"]) - vectors["standardization_mean"]) / vectors["standardization_scale"]
    if not np.isfinite(result).all():
        raise ExperimentStopped("preprocessing produced non-finite values")
    return result


def load_training_sample(panel: sqlite3.Connection, feature: sqlite3.Connection, model_db: sqlite3.Connection, mapping: Sequence[Mapping[str, object]], sources: Sequence[str], config: Mapping[str, object], start: float, data_dir: Path, evidence_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    source_sql = ",".join(qident(name) for name in dict.fromkeys(sources))
    dates = [str(row[0]) for row in feature.execute("SELECT DISTINCT decision_date FROM feature_rows WHERE model=? ORDER BY decision_date", (MODEL_TYPE,))]
    if dates != [str(TRAIN_START + dt.timedelta(days=i)) for i in range((TRAIN_END - TRAIN_START).days + 1)]:
        raise ExperimentStopped("training feature dates differ from locked range")
    expected = config["scope"]
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    sampling_rows: list[dict] = []
    for index, date_text in enumerate(dates):
        feature_rows = feature.execute(f"SELECT serial_number,{source_sql} FROM feature_rows WHERE model=? AND decision_date=? ORDER BY serial_number", (MODEL_TYPE, date_text)).fetchall()
        label_rows = panel.execute("SELECT serial_number,label,status,eligible FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND model=? AND decision_date=? ORDER BY serial_number", (TRAIN_LABEL_RUN, TRAIN_SPLIT, HORIZON, MODEL_TYPE, date_text)).fetchall()
        labels_by_serial = {str(row["serial_number"]): row for row in label_rows if int(row["eligible"]) == 1}
        serials = [str(row["serial_number"]) for row in feature_rows]
        if set(serials) != set(labels_by_serial):
            raise ExperimentStopped(f"feature/label key mismatch on {date_text}")
        negatives = [serial for serial in serials if labels_by_serial[serial]["label"] == 0 and labels_by_serial[serial]["status"] == "negative_observed"]
        positives = [serial for serial in serials if labels_by_serial[serial]["label"] == 1 and labels_by_serial[serial]["status"] in ("positive_observed", "positive_with_gap")]
        selected, digest, inverse_weight = sample_negative_indices(date_text, negatives)
        frozen = model_db.execute("SELECT * FROM sampling_stats WHERE decision_date=?", (date_text,)).fetchone()
        if frozen is None or (int(frozen["eligible_count"]), int(frozen["positive_count"]), int(frozen["negative_count"]), int(frozen["sampled_negative_count"]), str(frozen["sample_key_sha256"])) != (len(serials), len(positives), len(negatives), len(selected), digest):
            raise ExperimentStopped(f"frozen sampling statistics differ on {date_text}")
        selected_feature_rows: list[sqlite3.Row] = []
        day_labels: list[int] = []
        day_weights: list[float] = []
        for row in feature_rows:
            serial = str(row["serial_number"])
            label_row = labels_by_serial[serial]
            status = str(label_row["status"])
            if status in ("positive_observed", "positive_with_gap"):
                selected_feature_rows.append(row); day_labels.append(1); day_weights.append(1.0)
            elif status == "negative_observed":
                if serial in selected:
                    selected_feature_rows.append(row); day_labels.append(0); day_weights.append(inverse_weight)
            elif status not in ("gap_or_exit_censored", "history_insufficient"):
                raise ExperimentStopped(f"unexpected training label status {status}")
        if selected_feature_rows:
            matrices.append(row_matrix(selected_feature_rows, mapping))
            labels.extend(day_labels)
            weights.extend(day_weights)
        sampling_rows.append({
            "decision_date": date_text, "eligible_count": len(serials), "positive_count": len(positives),
            "negative_count": len(negatives), "sampled_negative_count": len(selected),
            "negative_inverse_weight": inverse_weight, "sample_key_sha256": digest,
            "known_unknown_excluded": len(serials) - len(positives) - len(negatives),
        })
        if index % 5 == 0:
            check_resources(start, config, data_dir, evidence_dir, label=f"training sample after {date_text}")
    raw = np.concatenate(matrices, axis=0).astype(np.float64, copy=False) if matrices else np.empty((0, len(mapping)), dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    raw_weights = np.asarray(weights, dtype=np.float64)
    if len(raw) != int(expected["expected_train_rows"]) or int(y.sum()) != int(expected["expected_positive_training_rows"]) or int((y == 0).sum()) != int(expected["expected_negative_training_rows"]):
        raise ExperimentStopped(f"training sample counts differ: rows={len(raw)}, positive={int(y.sum())}, negative={int((y == 0).sum())}")
    normalized = raw_weights / float(raw_weights.mean())
    if not np.isfinite(normalized).all() or np.any(normalized <= 0):
        raise ExperimentStopped("invalid normalized training weights")
    return raw, y, normalized, sampling_rows


def fit_worker(connection, x: np.ndarray, y: np.ndarray, weights: np.ndarray, params: dict) -> None:
    try:
        model = HistGradientBoostingClassifier(**params)
        model.fit(x, y, sample_weight=weights)
        connection.send((True, pickle.dumps(model, protocol=5)))
    except Exception as exc:  # pragma: no cover - parent verifies the failure path
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def fit_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray, params: dict, config: Mapping[str, object], start: float, data_dir: Path, evidence_dir: Path, resource_log: list[dict]) -> tuple[HistGradientBoostingClassifier, float]:
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    parent, child = context.Pipe(False)
    process = context.Process(target=fit_worker, args=(child, x, y, weights, params))
    fit_start = time.monotonic()
    process.start()
    result = None
    try:
        while process.is_alive() or parent.poll():
            resource_log.append(check_resources(start, config, data_dir, evidence_dir, label="gradient boosting fit", child_pid=process.pid))
            if time.monotonic() - fit_start >= int(config["resources"]["fit_seconds"]):
                process.terminate()
                process.join(timeout=5)
                raise ExperimentStopped("gradient boosting fit timeout")
            if parent.poll(0.25):
                result = parent.recv()
                break
        process.join(timeout=10)
        if result is None:
            if parent.poll():
                result = parent.recv()
            else:
                raise ExperimentStopped(f"gradient boosting worker exited with code {process.exitcode}")
        if not result[0]:
            raise ExperimentStopped(f"gradient boosting fit failed: {result[1]}")
        model = pickle.loads(result[1])
        if not isinstance(model, HistGradientBoostingClassifier):
            raise ExperimentStopped("fit worker returned an unexpected model")
        return model, time.monotonic() - fit_start
    finally:
        if process.is_alive():
            process.terminate(); process.join(timeout=5)
        parent.close()


def rank_indices(rows: Sequence[Mapping[str, object]], scores: Sequence[float]) -> list[int]:
    return sorted(range(len(rows)), key=lambda i: (-float(scores[i]), str(rows[i]["tie_break_sha256"]), str(rows[i]["serial_number"])))


def select_indices(rows: Sequence[Mapping[str, object]], scores: Sequence[float], decision: dt.date, last_alert: dict[str, dt.date], budget_denominator: int = 1000, cooldown_days: int = 7) -> tuple[list[int], int]:
    available: list[int] = []
    excluded = 0
    for index in rank_indices(rows, scores):
        serial = str(rows[index]["serial_number"])
        previous = last_alert.get(serial)
        if previous is not None and 1 <= (decision - previous).days <= cooldown_days:
            excluded += 1
        else:
            available.append(index)
    budget = (len(rows) + budget_denominator - 1) // budget_denominator if rows else 0
    selected = available[:budget]
    for index in selected:
        last_alert[str(rows[index]["serial_number"])] = decision
    return selected, excluded


def average_precision_grouped(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    if len(scores) != len(labels):
        raise ExperimentStopped("AP inputs differ in length")
    positives = int(sum(labels))
    if positives == 0:
        return None
    pairs = sorted(zip((float(x) for x in scores), (int(y) for y in labels)), key=lambda item: -item[0])
    position = seen = true = 0
    result = 0.0
    while position < len(pairs):
        end = position + 1
        while end < len(pairs) and pairs[end][0] == pairs[position][0]:
            end += 1
        group_positive = sum(label for _score, label in pairs[position:end])
        seen += end - position
        true += group_positive
        result += (true / seen) * (group_positive / positives)
        position = end
    return float(result)


def linear_quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(serials: Sequence[str], opportunities: Mapping[str, int], left: Mapping[str, int], right: Mapping[str, int], seed: int, replicates: int) -> dict:
    ordered = sorted({str(serial) for serial in serials}, key=lambda value: value.encode("utf-8"))
    if not ordered:
        raise ExperimentStopped("bootstrap has no devices")
    denominator = np.asarray([int(opportunities.get(serial, 0)) for serial in ordered], dtype=np.int64)
    left_values = np.asarray([int(left.get(serial, 0)) for serial in ordered], dtype=np.int64)
    right_values = np.asarray([int(right.get(serial, 0)) for serial in ordered], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    differences: list[float] = []
    zero = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(ordered), size=len(ordered), dtype=np.int64)
        denom = int(denominator[indices].sum())
        if denom == 0:
            zero += 1
        else:
            differences.append(float(right_values[indices].sum() / denom - left_values[indices].sum() / denom))
    return {
        "seed": seed, "replicates": replicates, "device_count": len(ordered),
        "valid_replicates": len(differences), "zero_opportunity_replicates": zero,
        "quantile_025": linear_quantile(differences, 0.025), "quantile_975": linear_quantile(differences, 0.975),
        "mean_difference": float(np.mean(differences)) if differences else None,
        "differences": differences,
    }


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE model_runs(model TEXT PRIMARY KEY, status TEXT NOT NULL, feature_count INTEGER NOT NULL,
        training_rows INTEGER NOT NULL, positive_training_rows INTEGER NOT NULL, negative_training_rows INTEGER NOT NULL,
        fit_seconds REAL NOT NULL, model_pickle_sha256 TEXT NOT NULL, params_json TEXT NOT NULL,
        average_precision_known REAL);
    CREATE TABLE model_feature_map(model TEXT NOT NULL, position INTEGER NOT NULL, feature_name TEXT NOT NULL,
        source_name TEXT NOT NULL, is_missing_indicator INTEGER NOT NULL, PRIMARY KEY(model,position));
    CREATE TABLE preprocessing_stats(model TEXT NOT NULL, position INTEGER NOT NULL, feature_name TEXT NOT NULL,
        imputation_mean REAL NOT NULL, standardization_mean REAL NOT NULL, standardization_scale REAL NOT NULL,
        observed_count INTEGER NOT NULL, PRIMARY KEY(model,position));
    CREATE TABLE sampling_stats(decision_date TEXT PRIMARY KEY, eligible_count INTEGER NOT NULL, positive_count INTEGER NOT NULL,
        negative_count INTEGER NOT NULL, sampled_negative_count INTEGER NOT NULL, negative_inverse_weight REAL NOT NULL,
        sample_key_sha256 TEXT NOT NULL, known_unknown_excluded INTEGER NOT NULL);
    CREATE TABLE model_scores(decision_date TEXT NOT NULL, serial_number TEXT NOT NULL, tie_break_sha256 TEXT NOT NULL,
        current_lr_score REAL NOT NULL, history_lr_score REAL NOT NULL, history_hgb_score REAL NOT NULL,
        history_observations_14 INTEGER NOT NULL, observed_days_7 INTEGER NOT NULL, current_missing_any INTEGER NOT NULL,
        decrease_seen_any INTEGER NOT NULL, window_crosses_decrease_w7_any INTEGER NOT NULL,
        window_crosses_decrease_w14_any INTEGER NOT NULL, smart_nonzero_signal_count INTEGER NOT NULL,
        smart_187_signal INTEGER NOT NULL, PRIMARY KEY(decision_date,serial_number));
    CREATE TABLE model_alerts(model TEXT NOT NULL, decision_date TEXT NOT NULL, serial_number TEXT NOT NULL,
        score REAL NOT NULL, tie_break_sha256 TEXT NOT NULL, selected_rank INTEGER NOT NULL,
        status TEXT, label INTEGER, first_failure_date TEXT, event_key TEXT, event_hit INTEGER,
        lead_days INTEGER, PRIMARY KEY(model,decision_date,serial_number));
    CREATE TABLE model_daily(model TEXT NOT NULL, decision_date TEXT NOT NULL, eligible_count INTEGER NOT NULL,
        budget_k INTEGER NOT NULL, cooldown_excluded INTEGER NOT NULL, alerts_count INTEGER NOT NULL,
        known_hit_alerts INTEGER, known_no_hit_alerts INTEGER, unknown_alerts INTEGER,
        PRIMARY KEY(model,decision_date));
    CREATE TABLE model_event_summary(model TEXT NOT NULL, event_key TEXT NOT NULL, first_failure_date TEXT NOT NULL,
        opportunity INTEGER NOT NULL, hit INTEGER NOT NULL, earliest_alert_date TEXT, earliest_lead_days INTEGER,
        PRIMARY KEY(model,event_key));
    CREATE TABLE model_metrics(model TEXT PRIMARY KEY, eligible_device_days INTEGER NOT NULL, alerts INTEGER NOT NULL,
        known_hit_alerts INTEGER NOT NULL, known_no_hit_alerts INTEGER NOT NULL, unknown_alerts INTEGER NOT NULL,
        unknown_alert_ratio REAL NOT NULL, precision_lower_bound REAL NOT NULL, precision_upper_bound REAL NOT NULL,
        known_outcome_precision REAL, confirmed_no_hit_per_1000_device_days REAL NOT NULL,
        event_total INTEGER NOT NULL, event_opportunity_total INTEGER NOT NULL, event_hits INTEGER NOT NULL,
        event_recall_at_opportunity REAL NOT NULL, early_event_hits_ge2_days INTEGER NOT NULL,
        early_event_hits_ge3_days INTEGER NOT NULL, earliest_lead_count INTEGER NOT NULL,
        earliest_lead_median REAL, earliest_lead_q25 REAL, earliest_lead_q75 REAL,
        repeated_alert_devices INTEGER NOT NULL, max_alerts_per_device INTEGER NOT NULL,
        minimum_alert_gap_days INTEGER, average_precision_known REAL, metric_scope TEXT NOT NULL);
    CREATE TABLE resource_log(label TEXT NOT NULL, elapsed_seconds REAL NOT NULL, rss_max_bytes INTEGER NOT NULL,
        child_rss_bytes INTEGER NOT NULL, total_rss_bytes INTEGER NOT NULL, owned_bytes INTEGER NOT NULL, free_bytes INTEGER NOT NULL);
    """)


def load_q3_day_features(q3: sqlite3.Connection, date_text: str, sources: Sequence[str]) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    source_sql = ",".join(qident(name) for name in dict.fromkeys(sources))
    controls = q3.execute("SELECT serial_number,tie_break_sha256,current_lr_score,history_lr_score,history_observations_14,observed_days_7,current_missing_any,decrease_seen_any,window_crosses_decrease_w7_any,window_crosses_decrease_w14_any,smart_nonzero_signal_count,smart_187_signal FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
    features = q3.execute(f"SELECT serial_number,tie_break_sha256,{source_sql} FROM feature_rows WHERE model=? AND decision_date=? ORDER BY serial_number", (MODEL_TYPE, date_text)).fetchall()
    if len(controls) != len(features) or [str(row["serial_number"]) for row in controls] != [str(row["serial_number"]) for row in features]:
        raise ExperimentStopped(f"Q3 feature/control key mismatch on {date_text}")
    if any(str(a["tie_break_sha256"]) != str(b["tie_break_sha256"]) for a, b in zip(controls, features)):
        raise ExperimentStopped(f"Q3 tie-break mismatch on {date_text}")
    return controls, features


def run_score_pass(q3: sqlite3.Connection, model_db: sqlite3.Connection, output: sqlite3.Connection, model: HistGradientBoostingClassifier, mapping: Sequence[Mapping[str, object]], sources: Sequence[str], config: Mapping[str, object], start: float, data_dir: Path, evidence_dir: Path, resource_log: list[dict]) -> dict:
    dates = [str(SCORE_START + dt.timedelta(days=i)) for i in range((SCORE_END - SCORE_START).days + 1)]
    frozen_alerts = {(str(row["model"]), str(row["decision_date"]), str(row["serial_number"])) for row in q3.execute("SELECT model,decision_date,serial_number FROM model_alerts WHERE model IN ('current_lr','history_lr')")}
    last_alert = {name: {} for name in ALL_MODELS}
    score_count = 0
    selection_counts = Counter()
    for day_index, date_text in enumerate(dates):
        controls, features = load_q3_day_features(q3, date_text, sources)
        raw = row_matrix(features, mapping)
        x = apply_preprocess(raw, PREPROCESS_VECTORS)
        tree_scores = model.predict_proba(x)[:, 1].astype(np.float64)
        if not np.isfinite(tree_scores).all():
            raise ExperimentStopped(f"non-finite tree score on {date_text}")
        rows = [dict(row) for row in controls]
        for row, score in zip(rows, tree_scores):
            row["history_hgb_score"] = float(score)
        output.executemany("INSERT INTO model_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
            date_text, row["serial_number"], row["tie_break_sha256"], float(row["current_lr_score"]), float(row["history_lr_score"]), float(row["history_hgb_score"]),
            int(row["history_observations_14"]), int(row["observed_days_7"]), int(row["current_missing_any"]), int(row["decrease_seen_any"]),
            int(row["window_crosses_decrease_w7_any"]), int(row["window_crosses_decrease_w14_any"]), int(row["smart_nonzero_signal_count"]), int(row["smart_187_signal"]),
        ) for row in rows])
        score_count += len(rows)
        score_arrays = {"current_lr": np.asarray([float(row["current_lr_score"]) for row in rows]), "history_lr": np.asarray([float(row["history_lr_score"]) for row in rows]), MODEL_NAME: tree_scores}
        for name in ALL_MODELS:
            selected, excluded = select_indices(rows, score_arrays[name], dt.date.fromisoformat(date_text), last_alert[name], int(config["scope"]["budget_denominator"]), int(config["scope"]["cooldown_days"]))
            actual = {(name, date_text, str(rows[index]["serial_number"])) for index in selected}
            if name in CONTROL_MODELS and actual != {key for key in frozen_alerts if key[0] == name and key[1] == date_text}:
                raise ExperimentStopped(f"frozen {name} alert list differs on {date_text}")
            selection_counts[name] += len(selected)
            for rank, index in enumerate(selected, start=1):
                row = rows[index]
                output.execute("INSERT INTO model_alerts(model,decision_date,serial_number,score,tie_break_sha256,selected_rank) VALUES (?,?,?,?,?,?)", (name, date_text, row["serial_number"], float(score_arrays[name][index]), row["tie_break_sha256"], rank))
            output.execute("INSERT INTO model_daily(model,decision_date,eligible_count,budget_k,cooldown_excluded,alerts_count) VALUES (?,?,?,?,?,?)", (name, date_text, len(rows), (len(rows) + 999) // 1000, excluded, len(selected)))
        if day_index % 5 == 0:
            output.commit()
            resource_log.append(check_resources(start, config, data_dir, evidence_dir, label=f"Q3 score after {date_text}"))
    expected_rows = int(config["scope"]["expected_q3_eligible_rows"])
    if score_count != expected_rows or any(selection_counts[name] != 1509 for name in ALL_MODELS):
        raise ExperimentStopped(f"score or alert counts differ: scores={score_count}, selections={dict(selection_counts)}")
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('score_status','complete')")
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('selection_status','complete')")
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('outcome_access_during_score','false')")
    output.commit()
    return {"score_rows": score_count, "alert_rows": dict(selection_counts), "score_dates": len(dates), "control_lists_matched": len(dates) * len(CONTROL_MODELS)}


def evaluate(output: sqlite3.Connection, q3: sqlite3.Connection, config: Mapping[str, object], start: float, data_dir: Path, evidence_dir: Path, resource_log: list[dict]) -> dict:
    if metadata(output).get("score_status") != "complete":
        raise ExperimentStopped("evaluation opened before score pass was complete")
    events_rows = q3.execute("SELECT serial_number,MIN(first_failure_date) AS first_failure_date,MAX(CASE WHEN eligible=1 AND label=1 THEN 1 ELSE 0 END) AS opportunity FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND model=? AND first_failure_date BETWEEN ? AND ? GROUP BY serial_number ORDER BY serial_number", (Q3_LABEL_RUN, Q3_SPLIT, HORIZON, MODEL_TYPE, EVENT_START.isoformat(), EVENT_END.isoformat())).fetchall()
    events = {str(row["serial_number"]): {"first_failure_date": str(row["first_failure_date"]), "opportunity": int(row["opportunity"])} for row in events_rows}
    if (len(events), sum(value["opportunity"] for value in events.values())) != (int(config["scope"]["expected_q3_events"]), int(config["scope"]["expected_q3_opportunities"])):
        raise ExperimentStopped("Q3 event denominator differs from locked scope")
    dates = [str(SCORE_START + dt.timedelta(days=i)) for i in range((SCORE_END - SCORE_START).days + 1)]
    alerts_by_model: dict[str, list[dict]] = {name: [] for name in ALL_MODELS}
    known_scores = {name: [] for name in ALL_MODELS}
    score_columns = {"current_lr": "current_lr_score", "history_lr": "history_lr_score", MODEL_NAME: "history_hgb_score"}
    known_labels: list[int] = []
    event_hits: dict[str, dict[str, int]] = {name: {} for name in ALL_MODELS}
    strata: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"eligible": 0, "alerts": 0, "known_hit": 0, "known_no_hit": 0, "unknown": 0})
    for day_index, date_text in enumerate(dates):
        labels = {str(row["serial_number"]): row for row in q3.execute("SELECT serial_number,status,label,first_failure_date FROM label_flow WHERE run_id=? AND split=? AND horizon_days=? AND model=? AND decision_date=? AND eligible=1", (Q3_LABEL_RUN, Q3_SPLIT, HORIZON, MODEL_TYPE, date_text))}
        score_rows = output.execute("SELECT * FROM model_scores WHERE decision_date=? ORDER BY serial_number", (date_text,)).fetchall()
        if len(score_rows) != len(labels):
            raise ExperimentStopped(f"Q3 label key count differs on {date_text}")
        score_by_serial = {str(row["serial_number"]): row for row in score_rows}
        for row in score_rows:
            serial = str(row["serial_number"])
            label = labels.get(serial)
            if label is None:
                raise ExperimentStopped(f"missing Q3 label key on {date_text}/{serial}")
            for typ, value in (("month", date_text[:7]), ("current_missing", str(int(row["current_missing_any"]))), ("history_observations_14", str(int(row["history_observations_14"])))):
                strata[("all", typ, value)]["eligible"] += 1
            if label["label"] in (0, 1):
                known_labels.append(int(label["label"]))
                for name in ALL_MODELS:
                    known_scores[name].append(float(row[score_columns[name]]))
        for name in ALL_MODELS:
            selected = output.execute("SELECT * FROM model_alerts WHERE model=? AND decision_date=? ORDER BY selected_rank", (name, date_text)).fetchall()
            updates = []
            day_hit = day_no_hit = day_unknown = 0
            for row in selected:
                serial = str(row["serial_number"]); label = labels[serial]
                label_value = None if label["label"] is None else int(label["label"])
                if label_value == 1: day_hit += 1
                elif label_value == 0: day_no_hit += 1
                else: day_unknown += 1
                failure = label["first_failure_date"]
                event_key = None; hit = 0; lead = None
                if serial in events and failure is not None and str(failure) == events[serial]["first_failure_date"]:
                    delta = (dt.date.fromisoformat(str(failure)) - dt.date.fromisoformat(date_text)).days
                    if events[serial]["opportunity"] and 1 <= delta <= HORIZON:
                        event_key = serial; hit = 1; lead = delta
                updates.append((str(label["status"]), label_value, None if failure is None else str(failure), event_key, hit, lead, name, date_text, serial))
                alerts_by_model[name].append({"model": name, "decision_date": date_text, "serial_number": serial, "score": float(row["score"]), "label": label_value, "event_key": event_key, "event_hit": hit, "lead_days": lead})
                score_row = score_by_serial[serial]
                for typ, value in (("month", date_text[:7]), ("current_missing", str(int(score_row["current_missing_any"]))), ("history_observations_14", str(int(score_row["history_observations_14"])))):
                    item = strata[(name, typ, value)]; item["alerts"] += 1
                    if label_value == 1: item["known_hit"] += 1
                    elif label_value == 0: item["known_no_hit"] += 1
                    else: item["unknown"] += 1
            output.executemany("UPDATE model_alerts SET status=?,label=?,first_failure_date=?,event_key=?,event_hit=?,lead_days=? WHERE model=? AND decision_date=? AND serial_number=?", updates)
            output.execute("UPDATE model_daily SET known_hit_alerts=?,known_no_hit_alerts=?,unknown_alerts=? WHERE model=? AND decision_date=?", (day_hit, day_no_hit, day_unknown, name, date_text))
        if day_index % 5 == 0:
            output.commit()
            resource_log.append(check_resources(start, config, data_dir, evidence_dir, label=f"Q3 evaluation after {date_text}"))
    if len(known_labels) != int(config["scope"]["expected_q3_known_rows"]) or sum(known_labels) != int(config["scope"]["expected_q3_positive_rows"]):
        raise ExperimentStopped(f"known Q3 outcome count differs: {len(known_labels)}, positives={sum(known_labels)}")
    metrics: dict[str, dict] = {}
    hit_maps: dict[str, dict[str, int]] = {}
    for name in ALL_MODELS:
        rows = alerts_by_model[name]
        known_hit = sum(row["label"] == 1 for row in rows); known_no_hit = sum(row["label"] == 0 for row in rows); unknown = sum(row["label"] is None for row in rows)
        by_event: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["event_key"] is not None: by_event[row["event_key"]].append(row)
        hit_maps[name] = {key: int(any(row["event_hit"] for row in by_event.get(key, []))) for key in events}
        leads = [max(int(row["lead_days"]) for row in by_event[key]) for key in by_event if by_event[key]]
        event_total = len(events); opportunity_total = sum(value["opportunity"] for value in events.values()); hits = sum(hit_maps[name].get(key, 0) for key, value in events.items() if value["opportunity"])
        device_counts = Counter(row["serial_number"] for row in rows)
        gaps = []
        for serial in device_counts:
            day_values = sorted(dt.date.fromisoformat(row["decision_date"]) for row in rows if row["serial_number"] == serial)
            gaps.extend((right - left).days for left, right in zip(day_values, day_values[1:]))
        scores = known_scores[name]
        ap = average_precision_grouped(scores, known_labels)
        metrics[name] = {
            "eligible_device_days": int(output.execute("SELECT SUM(eligible_count) FROM model_daily WHERE model=?", (name,)).fetchone()[0]),
            "alerts": len(rows), "known_hit_alerts": known_hit, "known_no_hit_alerts": known_no_hit, "unknown_alerts": unknown,
            "unknown_alert_ratio": unknown / len(rows), "precision_lower_bound": known_hit / len(rows), "precision_upper_bound": (known_hit + unknown) / len(rows),
            "known_outcome_precision": known_hit / (known_hit + known_no_hit) if known_hit + known_no_hit else None,
            "confirmed_no_hit_per_1000_device_days": known_no_hit / (int(output.execute("SELECT SUM(eligible_count) FROM model_daily WHERE model=?", (name,)).fetchone()[0]) or 1) * 1000,
            "event_total": event_total, "event_opportunity_total": opportunity_total, "event_hits": hits, "event_recall_at_opportunity": hits / opportunity_total,
            "early_event_hits_ge2_days": sum(hit_maps[name].get(key, 0) and max(int(row["lead_days"]) for row in by_event[key]) >= 2 for key in by_event),
            "early_event_hits_ge3_days": sum(hit_maps[name].get(key, 0) and max(int(row["lead_days"]) for row in by_event[key]) >= 3 for key in by_event),
            "earliest_lead_count": len(leads), "earliest_lead_median": linear_quantile(leads, .5), "earliest_lead_q25": linear_quantile(leads, .25), "earliest_lead_q75": linear_quantile(leads, .75),
            "repeated_alert_devices": sum(value > 1 for value in device_counts.values()), "max_alerts_per_device": max(device_counts.values(), default=0), "minimum_alert_gap_days": min(gaps) if gaps else None,
            "average_precision_known": ap,
        }
        for key, value in events.items():
            event_rows = by_event.get(key, [])
            first = min(event_rows, key=lambda row: row["decision_date"]) if event_rows else None
            output.execute("INSERT INTO model_event_summary VALUES (?,?,?,?,?,?,?)", (name, key, value["first_failure_date"], value["opportunity"], hit_maps[name].get(key, 0), first["decision_date"] if first else None, first["lead_days"] if first else None))
        metric = metrics[name]
        output.execute("INSERT INTO model_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (name, metric["eligible_device_days"], metric["alerts"], metric["known_hit_alerts"], metric["known_no_hit_alerts"], metric["unknown_alerts"], metric["unknown_alert_ratio"], metric["precision_lower_bound"], metric["precision_upper_bound"], metric["known_outcome_precision"], metric["confirmed_no_hit_per_1000_device_days"], metric["event_total"], metric["event_opportunity_total"], metric["event_hits"], metric["event_recall_at_opportunity"], metric["early_event_hits_ge2_days"], metric["early_event_hits_ge3_days"], metric["earliest_lead_count"], metric["earliest_lead_median"], metric["earliest_lead_q25"], metric["earliest_lead_q75"], metric["repeated_alert_devices"], metric["max_alerts_per_device"], metric["minimum_alert_gap_days"], metric["average_precision_known"], "Q3 development after error-informed model choice; unknown excluded from AP"))
    serials = [str(row[0]) for row in output.execute("SELECT DISTINCT serial_number FROM model_scores")]
    opportunities = {key: int(value["opportunity"]) for key, value in events.items()}
    bootstraps = {}
    for left in CONTROL_MODELS:
        bootstraps[f"{MODEL_NAME}_minus_{left}"] = paired_bootstrap(serials, opportunities, hit_maps[left], hit_maps[MODEL_NAME], int(config["gate"]["bootstrap_seed"]), int(config["gate"]["bootstrap_replicates"]))
    current = metrics["current_lr"]; history = metrics["history_lr"]; candidate = metrics[MODEL_NAME]
    candidate_current = bootstraps[f"{MODEL_NAME}_minus_current_lr"]
    gate = {
        "recall_gain_vs_current_at_least_5pp": candidate["event_recall_at_opportunity"] - current["event_recall_at_opportunity"] >= float(config["gate"]["minimum_recall_gain_vs_current"]),
        "paired_current_interval_lower_strictly_above_zero": candidate_current["valid_replicates"] >= int(config["gate"]["bootstrap_min_valid"]) and candidate_current["quantile_025"] is not None and candidate_current["quantile_025"] > float(config["gate"]["paired_interval_lower_strictly_above"]),
        "hits_more_than_history": candidate["event_hits"] >= history["event_hits"] + int(config["gate"]["minimum_hits_vs_history"]),
        "known_no_hit_alerts_within_current": candidate["known_no_hit_alerts"] <= int(config["gate"]["maximum_known_no_hit_alerts"]),
        "unknown_ratio_increase_within_current": candidate["unknown_alert_ratio"] - current["unknown_alert_ratio"] <= float(config["gate"]["maximum_unknown_ratio_increase_vs_current"]),
        "opportunity_events_at_least_minimum": candidate["event_opportunity_total"] >= int(config["gate"]["minimum_opportunity_events"]),
    }
    comparison = {"candidate": MODEL_NAME, "current": "current_lr", "history": "history_lr", "gate": gate, "advance_to_independent_validation": bool(all(gate.values())), "interpretation": "Q3 is a development reference after error-informed selection; this does not establish independent generalization."}
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('evaluation_status','complete')")
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('model_adoption',?)", ("candidate_for_independent_validation" if comparison["advance_to_independent_validation"] else "do_not_advance",))
    output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('run_status','complete')")
    output.commit()
    return {"metrics": metrics, "bootstrap": bootstraps, "comparison": comparison, "known_rows": len(known_labels), "positive_rows": sum(known_labels), "events": {"total": len(events), "opportunities": sum(value["opportunity"] for value in events.values())}}


PREPROCESS_VECTORS: dict[str, np.ndarray] = {}


def finalize_partial(config_path: Path, attempt_dir: str) -> dict:
    """Finish a score-complete partial after a non-data implementation stop."""
    config = read_json(config_path)
    if not attempt_dir.startswith("attempt_") or "/" in attempt_dir or "\\" in attempt_dir:
        raise ExperimentStopped("attempt_dir must be a simple attempt_N name")
    data_relative = str(config["outputs"]["data_dir"]).replace("attempt_001", attempt_dir)
    evidence_relative = str(config["outputs"]["evidence_dir"]).replace("attempt_001", attempt_dir)
    data_dir = ROOT / data_relative
    evidence_dir = ROOT / evidence_relative
    partial = data_dir / "nonlinear_baseline.sqlite.partial"
    final = data_dir / "nonlinear_baseline.sqlite"
    if final.exists() or not partial.is_file():
        raise ExperimentStopped("partial finalization target is missing or already finalized")
    q3_panel_path = ROOT / str(config["inputs"]["q3_panel_db"])
    expected_q3_panel = str(config["input_sha256"][str(config["inputs"]["q3_panel_db"])])
    if sha256_file(q3_panel_path) != expected_q3_panel:
        raise ExperimentStopped("Q3 panel SHA differs during partial finalization")
    start = time.monotonic(); resource_log: list[dict] = []
    output = sqlite3.connect(partial); output.row_factory = sqlite3.Row
    q3_panel = connect_immutable(q3_panel_path)
    try:
        meta = metadata(output)
        if meta.get("score_status") != "complete" or meta.get("selection_status") != "complete":
            raise ExperimentStopped("partial is not score-complete")
        counts = {
            "score_rows": int(output.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]),
            "alert_rows": {name: int(output.execute("SELECT COUNT(*) FROM model_alerts WHERE model=?", (name,)).fetchone()[0]) for name in ALL_MODELS},
        }
        if counts["score_rows"] != int(config["scope"]["expected_q3_eligible_rows"]) or any(value != 1509 for value in counts["alert_rows"].values()):
            raise ExperimentStopped(f"partial score counts differ: {counts}")
        resource_log.append(check_resources(start, config, data_dir, evidence_dir, label="partial finalization preflight"))
        evaluation = evaluate(output, q3_panel, config, start, data_dir, evidence_dir, resource_log)
        output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('config_hash',?)", (canonical_hash(config),))
        output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('code_sha256',?)", (sha256_file(Path(__file__)),))
        output.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('q3_panel_db_sha256',?)", (expected_q3_panel,))
        output.execute("UPDATE model_runs SET average_precision_known=? WHERE model=?", (evaluation["metrics"][MODEL_NAME]["average_precision_known"], MODEL_NAME))
        for item in resource_log:
            output.execute("INSERT INTO resource_log VALUES (?,?,?,?,?,?,?)", (item["label"], item["elapsed_seconds"], item["rss_max_bytes"], item["child_rss_bytes"], item["total_rss_bytes"], item["owned_bytes"], item["free_bytes"]))
        output.commit(); output.close(); output = None
        os.replace(partial, final)
        evidence = {
            "status": "pass", "experiment": MODEL_NAME, "attempt_dir": data_relative, "recovered_score_partial": True,
            "recovery_note": "score pass was complete; the original run stopped before evaluation because the Q3 label input was not bound in its configuration",
            "config_hash": canonical_hash(config), "code_sha256": sha256_file(Path(__file__)),
            "output_db": str(final.relative_to(ROOT)), "output_db_sha256": sha256_file(final),
            "model_pickle_sha256": meta.get("model_pickle_sha256"), "score_pass": counts, "evaluation": evaluation, "resources": resource_log,
            "environment": {"python": os.sys.executable, "python_version": os.sys.version, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
            "q3_is_development_reference": True, "training_approval": False, "q4_access": False, "remote_setup": False,
        }
        evidence_path = evidence_dir / "run.json"
        if evidence_path.exists():
            raise ExperimentStopped(f"refusing to overwrite evidence: {evidence_path}")
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return evidence
    except Exception:
        if output is not None:
            output.close()
        raise
    finally:
        q3_panel.close()


def run(config_path: Path = CONFIG_PATH, attempt_dir: str | None = None) -> dict:
    global PREPROCESS_VECTORS
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if config.get("status") != "execution_locked" or config.get("fit_allowed") is not True or config.get("q4_access") or config.get("remote_setup"):
        raise ExperimentStopped("nonlinear experiment is not in the locked local execution state")
    config_hash = canonical_hash(config)
    input_paths = {str(path): ROOT / path for path in config["input_sha256"]}
    for relative, path in input_paths.items():
        if not path.is_file() or sha256_file(path) != config["input_sha256"][relative]:
            raise ExperimentStopped(f"input SHA differs: {relative}")
    data_relative = str(config["outputs"]["data_dir"])
    evidence_relative = str(config["outputs"]["evidence_dir"])
    if attempt_dir is not None:
        if not attempt_dir.startswith("attempt_") or "/" in attempt_dir or "\\" in attempt_dir:
            raise ExperimentStopped("attempt_dir must be a simple attempt_N name")
        data_relative = data_relative.replace("attempt_001", attempt_dir)
        evidence_relative = evidence_relative.replace("attempt_001", attempt_dir)
    data_dir = ROOT / data_relative
    evidence_dir = ROOT / evidence_relative
    if data_dir.exists() or evidence_dir.exists():
        raise ExperimentStopped("nonlinear attempt output already exists; refusing overwrite")
    data_dir.parent.mkdir(parents=True, exist_ok=True); data_dir.mkdir()
    evidence_dir.parent.mkdir(parents=True, exist_ok=True); evidence_dir.mkdir()
    start = time.monotonic(); resource_log: list[dict] = []
    panel = feature = model_db = q3 = q3_panel = output = None
    partial = data_dir / "nonlinear_baseline.sqlite.partial"
    pickle_path = data_dir / "history_hgb_v1.pkl"
    try:
        resource_log.append(check_resources(start, config, data_dir, evidence_dir, label="preflight"))
        panel = connect_immutable(input_paths["data/derived/panel_q1q2_verified_v1.sqlite"])
        feature = connect_immutable(input_paths["data/derived/features_train_v1.sqlite"])
        model_db = connect_immutable(input_paths["data/derived/simple_baseline_v2/model_results.sqlite"])
        q3 = connect_immutable(input_paths["data/derived/q3_scoring_amended_v2/attempt_001/q3_scoring_amended_v2.sqlite"])
        q3_panel = connect_immutable(input_paths["data/derived/q3_validation_v1/panel_q3.sqlite"])
        feature_meta = metadata(feature); model_meta = metadata(model_db); q3_meta = metadata(q3); q3_panel_meta = metadata(q3_panel)
        if feature_meta.get("feature_status") != "complete" or model_meta.get("run_status") != "complete" or q3_meta.get("run_status") != "complete" or q3_panel_meta.get("panel_status") != "complete":
            raise ExperimentStopped("one or more frozen input databases is incomplete")
        mapping, sources, vectors = load_feature_map(model_db, feature)
        PREPROCESS_VECTORS = vectors
        raw, y, weights, sampling_rows = load_training_sample(panel, feature, model_db, mapping, sources, config, start, data_dir, evidence_dir)
        x = apply_preprocess(raw, vectors)
        params = {key: value for key, value in dict(config["classifier"]).items() if key != "class"}
        params["max_depth"] = None if params["max_depth"] is None else int(params["max_depth"])
        params["categorical_features"] = None
        params["monotonic_cst"] = None
        params["interaction_cst"] = None
        params["validation_fraction"] = None
        model, fit_seconds = fit_model(x, y, weights, params, config, start, data_dir, evidence_dir, resource_log)
        pickle_path.write_bytes(pickle.dumps(model, protocol=5))
        pickle_sha = sha256_file(pickle_path)
        output = sqlite3.connect(partial)
        output.row_factory = sqlite3.Row
        create_schema(output)
        metadata_values = {
            "run_status": "score_running", "score_status": "pending", "selection_status": "pending", "evaluation_status": "pending",
            "experiment": MODEL_NAME, "config_hash": config_hash, "config_path": str(config_path.relative_to(ROOT)), "code_sha256": sha256_file(Path(__file__)),
            "feature_db_sha256": sha256_file(input_paths["data/derived/features_train_v1.sqlite"]), "panel_db_sha256": sha256_file(input_paths["data/derived/panel_q1q2_verified_v1.sqlite"]),
            "model_db_sha256": sha256_file(input_paths["data/derived/simple_baseline_v2/model_results.sqlite"]), "q3_db_sha256": sha256_file(input_paths["data/derived/q3_scoring_amended_v2/attempt_001/q3_scoring_amended_v2.sqlite"]),
            "training_approval": "false", "q4_access": "false", "remote_setup": "false", "outcome_access_during_score": "false",
            "environment_python": os.sys.executable, "environment_numpy": np.__version__, "environment_sklearn": __import__("sklearn").__version__,
            "model_pickle_sha256": pickle_sha, "training_rows": str(len(y)), "positive_training_rows": str(int(y.sum())), "negative_training_rows": str(int((y == 0).sum())),
        }
        output.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", metadata_values.items())
        output.executemany("INSERT INTO model_feature_map VALUES (?,?,?,?,?)", [(MODEL_NAME, i, item["feature_name"], item["source_name"], item["is_missing_indicator"]) for i, item in enumerate(mapping)])
        output.executemany("INSERT INTO preprocessing_stats VALUES (?,?,?,?,?,?,?)", [(MODEL_NAME, i, item["feature_name"], float(vectors["imputation_mean"][i]), float(vectors["standardization_mean"][i]), float(vectors["standardization_scale"][i]), int(vectors["observed_count"][i])) for i, item in enumerate(mapping)])
        output.executemany("INSERT INTO sampling_stats VALUES (?,?,?,?,?,?,?,?)", [(row["decision_date"], row["eligible_count"], row["positive_count"], row["negative_count"], row["sampled_negative_count"], row["negative_inverse_weight"], row["sample_key_sha256"], row["known_unknown_excluded"]) for row in sampling_rows])
        output.execute("INSERT INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?)", (MODEL_NAME, "fitted", len(mapping), len(y), int(y.sum()), int((y == 0).sum()), fit_seconds, pickle_sha, json.dumps(params, sort_keys=True), None))
        output.commit()
        output.close(); output = None
        output = sqlite3.connect(partial)
        output.row_factory = sqlite3.Row
        score_result = run_score_pass(q3, model_db, output, model, mapping, sources, config, start, data_dir, evidence_dir, resource_log)
        output.close(); output = None
        output = sqlite3.connect(partial)
        output.row_factory = sqlite3.Row
        evaluation_result = evaluate(output, q3_panel, config, start, data_dir, evidence_dir, resource_log)
        for item in resource_log:
            output.execute("INSERT INTO resource_log VALUES (?,?,?,?,?,?,?)", (item["label"], item["elapsed_seconds"], item["rss_max_bytes"], item["child_rss_bytes"], item["total_rss_bytes"], item["owned_bytes"], item["free_bytes"]))
        output.execute("UPDATE model_runs SET average_precision_known=? WHERE model=?", (evaluation_result["metrics"][MODEL_NAME]["average_precision_known"], MODEL_NAME))
        output.commit(); output.close(); output = None
        os.replace(partial, data_dir / "nonlinear_baseline.sqlite")
        evidence = {
            "status": "pass", "experiment": MODEL_NAME, "attempt_dir": data_relative, "config_hash": config_hash, "code_sha256": sha256_file(Path(__file__)),
            "output_db": str((data_dir / "nonlinear_baseline.sqlite").relative_to(ROOT)), "output_db_sha256": sha256_file(data_dir / "nonlinear_baseline.sqlite"), "model_pickle_sha256": pickle_sha,
            "training": {"rows": len(y), "positive_rows": int(y.sum()), "negative_rows": int((y == 0).sum()), "sampling_rows": len(sampling_rows), "preprocessing_feature_count": len(mapping), "fit_seconds": fit_seconds},
            "score_pass": score_result, "evaluation": evaluation_result, "resources": resource_log,
            "environment": {"python": os.sys.executable, "python_version": os.sys.version, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
            "q3_is_development_reference": True, "training_approval": False, "q4_access": False, "remote_setup": False,
        }
        (evidence_dir / "run.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return evidence
    except Exception as exc:
        if output is not None:
            output.close()
        failure = {"status": "fail", "experiment": MODEL_NAME, "attempt_dir": data_relative, "reason": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.monotonic() - start, 3), "training_approval": False}
        failure_path = evidence_dir / "failure.json"
        if not failure_path.exists():
            failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        for connection in (panel, feature, model_db, q3, q3_panel, output):
            if connection is not None:
                connection.close()


def probe() -> dict:
    rows = [{"serial_number": "b", "tie_break_sha256": "b"}, {"serial_number": "a", "tie_break_sha256": "a"}]
    assert rank_indices(rows, [1.0, 1.0]) == [1, 0]
    last: dict[str, dt.date] = {}
    assert select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 1), last, budget_denominator=2, cooldown_days=7)[0] == [0]
    assert select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 8), last, budget_denominator=2, cooldown_days=7)[0] == [1]
    assert select_indices(rows, [2.0, 1.0], dt.date(2023, 1, 9), last, budget_denominator=2, cooldown_days=7)[0] == [0]
    assert abs(average_precision_grouped([0.9, 0.9, 0.1], [1, 0, 0]) - 0.5) < 1e-12
    boot = paired_bootstrap(["a", "b"], {"a": 1, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 1}, 20260913, 20)
    assert boot["valid_replicates"] == 20
    return {"status": "pass", "checks": ["tie_order", "cooldown_t_plus_7_t_plus_8", "grouped_ap", "paired_bootstrap"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--attempt-dir", default=None)
    parser.add_argument("--finalize-attempt", default=None)
    args = parser.parse_args()
    if args.probe:
        result = probe()
    elif args.finalize_attempt:
        result = finalize_partial(args.config, args.finalize_attempt)
    else:
        result = run(args.config, args.attempt_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
