"""Read-only diagnostics for the frozen Q3/Q4 evaluation artifacts.

This module deliberately does not fit, rescore, select, or mutate a research
database.  It computes post-hoc ranking, probability, and alert-burden
diagnostics from sealed outputs only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import sqlite3
import sys
import time
from array import array
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Sequence
import frozen_ml_checks as independent_verifier


class PackedRows:
    """Fixed-width numeric columns; device strings are interned once."""
    def __init__(self):
        self.scores = array('d')
        self.labels = array('b')
        self.devices = array('I')
        self.names = []
        self.identities = {}

    def append(self, row):
        score, label, serial = row
        if not math.isfinite(score) or label not in (None, 0, 1):
            raise ValueError('invalid packed row')
        if serial not in self.identities:
            self.identities[serial] = len(self.names)
            self.names.append(serial)
        self.scores.append(score)
        self.labels.append(-1 if label is None else label)
        self.devices.append(self.identities[serial])

    def __len__(self):
        return len(self.scores)

    def __iter__(self):
        for score, label, device in zip(self.scores, self.labels, self.devices):
            yield score, None if label == -1 else label, self.names[device]


def batches(cursor):
    while batch := cursor.fetchmany(50000):
        yield batch

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "evidence/ml_evaluation_v1"
Q3_DB = ROOT / "data/derived/q3_scoring_amended_v2/attempt_001/q3_scoring_amended_v2.sqlite"
Q3_PANEL = ROOT / "data/derived/q3_validation_v1/panel_q3.sqlite"
Q3_MANIFEST = ROOT / "evidence/q3/scoring_amended_v2/attempt_001/q3_scoring_amended_v2_complete_manifest_v1.json"
CAPACITY_DB = ROOT / "data/derived/alert_capacity_v1/evaluation_001/capacity_evaluation.sqlite"
CAPACITY_MANIFEST = ROOT / "evidence/alert_capacity_v1/evaluation_001/capacity_complete_manifest_v1.json"
Q4_ROOT = ROOT / "evidence/r_validation_v1/attempt_001/continuation_001"
Q4_SELECTION = Q4_ROOT / "work/evaluation/selection.sqlite"
Q4_PANEL = Q4_ROOT / "work/prepared/panel.sqlite"
Q4_EVAL = Q4_ROOT / "work/evaluation/evaluation.json"
Q4_MODEL = Q4_ROOT / "work/evaluation/current_lr.json"
Q4_COMPLETE = Q4_ROOT / "r_complete.json"
Q4_LABELING = Q4_ROOT / "snapshot/pipeline/labeling.py"
PLAN = ROOT / "evidence/protocols/ml_evaluation_v1.md"
SCRIPT = Path(__file__).resolve()
SUPERVISOR = ROOT / "tools/run_frozen_ml_supervised.py"

MAX_SECONDS = 1800.0
MAX_RSS_BYTES = 2 * 1024 ** 3
MAX_OUTPUT_BYTES = 512 * 1024 ** 2

# These are protocol values, not interchangeable conveniences.  Q3 and Q4
# have different eligible device-day populations and different event
# opportunity sets; keeping them beside the computation makes accidental
# denominator reuse visible in review.
QUARTER_FACTS = {
    "q3": {
        "eligible_device_days": 1_497_209,
        "opportunity_events": 126,
        "event_start": "2023-07-08",
        "event_end": "2023-09-24",
    },
    "q4": {
        "eligible_device_days": 1_216_041,
        "opportunity_events": 86,
        "event_start": "2023-10-08",
        "event_end": "2023-12-25",
    },
}

METHODS = ("current_lr", "smart_nonzero")
Q4_EXPECTED = {"current_lr": {"alerts": 1247, "event_hits": 40}, "smart_nonzero": {"alerts": 1247, "event_hits": 35}}
Q3_EXPECTED = {"current_lr": {"alerts": 1509, "event_hits": 65}, "smart_nonzero": {"alerts": 1509, "event_hits": 47}}
BIN_EDGES = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


class AnalysisError(RuntimeError):
    pass


def safe_attempt_id(value: str) -> str:
    """Accept only a simple leaf name for a new immutable result directory."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise AnalysisError(f"unsafe attempt id: {value!r}")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise AnalysisError(f"unsafe attempt id: {value!r}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resource_snapshot(started: float, output_dir: Path | None = None) -> dict[str, Any]:
    """Return process and output facts using platform-defined RSS units."""
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other POSIX implementations report KiB.
    rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    output_bytes = 0
    if output_dir is not None and output_dir.exists():
        output_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    return {
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_source": "resource.getrusage.ru_maxrss",
        "peak_rss_raw": raw_rss,
        "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "output_bytes": output_bytes,
        "output_limit_bytes": MAX_OUTPUT_BYTES,
        "rss_limit_bytes": MAX_RSS_BYTES,
        "time_limit_seconds": MAX_SECONDS,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
    }


def enforce_resources(resources: dict[str, Any], *, include_output: bool = True) -> None:
    if float(resources["elapsed_seconds"]) > MAX_SECONDS:
        raise AnalysisError(f"resource guard: elapsed {resources['elapsed_seconds']:.3f}s > {MAX_SECONDS:.0f}s")
    if int(resources["peak_rss_bytes"]) > MAX_RSS_BYTES:
        raise AnalysisError(f"resource guard: peak RSS {resources['peak_rss_bytes']} > {MAX_RSS_BYTES}")
    if include_output and int(resources["output_bytes"]) > MAX_OUTPUT_BYTES:
        raise AnalysisError(f"resource guard: output bytes {resources['output_bytes']} > {MAX_OUTPUT_BYTES}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON object required: {path}")
    return value


def ro_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise AnalysisError(f"missing database: {path}")
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise AnalysisError(f"non-empty SQLite sidecar refuses immutable read: {sidecar}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA query_only=ON")
    return connection


def assert_metadata(connection: sqlite3.Connection, expected: dict[str, str], label: str) -> dict[str, str]:
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AnalysisError(f"{label} metadata {key}={metadata.get(key)!r}, expected {value!r}")
    return metadata


def preflight() -> dict[str, Any]:
    started = time.monotonic()
    # Review-pinned historical manifests, rather than a newly invented source
    # expectation derived from the same files that are being checked.
    for path, expected in (
        (Q3_MANIFEST, '87e6c9bcb48c6c720bf12731bebfb8f256096dbda3bc2b522ccb303862432f72'),
        (Q4_COMPLETE, '60a592608adb3a0c3b40289d3ebe664b6aea5a44fa9e5fd23f401135064dd372'),
        (CAPACITY_MANIFEST, 'c6eab6a1b20b40c1a89fec313b8b2e18267e8d63848c7581f303a5f32ee584e9'),
    ):
        if sha256(path) != expected:
            raise AnalysisError(f'historical source manifest changed: {path}')
    if not PLAN.is_file():
        raise AnalysisError("E1.0 plan is missing")
    q3_manifest = read_json(Q3_MANIFEST)
    if q3_manifest.get("status") != "complete" or q3_manifest.get("database") != str(Q3_DB.relative_to(ROOT)):
        raise AnalysisError("Q3 complete manifest is not bound to the expected complete database")
    q4_complete = read_json(Q4_COMPLETE)
    if q4_complete.get("status") != "complete" or q4_complete.get("verification") != "independent_recomputation_passed":
        raise AnalysisError("Q4 completion or independent verification is not complete")
    q4_model = read_json(Q4_MODEL)
    if (q4_model.get("model") != "current_lr" or q4_model.get("feature_count") != 16
            or len(q4_model.get("features", [])) != 16
            or q4_model.get("model_run_status") != "reused_v1"):
        raise AnalysisError("Q4 current_lr parameter binding is not the sealed 16-feature reused model")
    for feature in q4_model.get("features", []):
        for key in ("position", "feature_name", "source_name", "coefficient", "standardization_scale", "imputation_mean"):
            if key not in feature:
                raise AnalysisError(f"Q4 model feature lacks fixed-key field: {key}")
        if not all(math.isfinite(float(feature[key])) for key in ("coefficient", "standardization_scale", "imputation_mean")):
            raise AnalysisError("Q4 model contains a non-finite coefficient or preprocessing value")
        if float(feature["standardization_scale"]) <= 0:
            raise AnalysisError("Q4 model contains a non-positive standardization scale")
    if not math.isfinite(float(q4_model.get("intercept"))):
        raise AnalysisError("Q4 model intercept is not finite")
    q4_meta = {}
    selection = ro_db(Q4_SELECTION)
    try:
        q4_meta = assert_metadata(selection, {
            "phase": "complete", "start": "2023-10-01", "end": "2023-12-24",
            "outcome_cutoff": "2023-12-31", "horizon_days": "7", "history_days": "14",
            "min_history": "12", "budget_denominator": "1000", "cooldown_days": "7",
        }, "Q4 selection")
        if selection.execute("SELECT COUNT(*) FROM scores WHERE model IN ('current_lr','smart_nonzero')").fetchone()[0] != 2432082:
            raise AnalysisError("Q4 score row count is not the sealed 2,432,082")
        if selection.execute("SELECT COUNT(*) FROM selections").fetchone()[0] != 2494:
            raise AnalysisError("Q4 selection row count is not the sealed 2,494")
    finally:
        selection.close()
    q3 = ro_db(Q3_DB)
    try:
        q3_meta = assert_metadata(q3, {
            "attempt_id": "attempt_001", "model": "ST4000DM000", "score_start": "2023-07-01",
            "score_end": "2023-09-23", "label_cutoff": "2023-09-30", "score_status": "complete",
            "selection_status": "complete", "evaluation_status": "complete",
        }, "Q3 scoring")
        if int(q3_meta.get("score_rows", "0")) != 1497209:
            raise AnalysisError("Q3 score row count is not the sealed 1,497,209")
    finally:
        q3.close()
    bindings = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (Q3_DB, Q3_PANEL, Q3_MANIFEST, CAPACITY_DB, CAPACITY_MANIFEST,
                     Q4_SELECTION, Q4_PANEL, Q4_EVAL, Q4_MODEL, Q4_COMPLETE, Q4_LABELING, PLAN, SCRIPT, SUPERVISOR,
                     ROOT / 'tools/frozen_ml_checks.py',
                     ROOT / 'pipeline/reproducible/resource_guard.py',
                     ROOT / 'pipeline/reproducible/supervisor.py',
                     ROOT / 'evidence/ml_evaluation_v1/implementation_spec.md')
    }
    if bindings[str(Q3_DB.relative_to(ROOT))] != q3_manifest.get("database_sha256"):
        raise AnalysisError("Q3 database SHA differs from its complete manifest")
    q3_bindings = q3_manifest.get("input_bindings", {})
    for relative, path in {
        "data/derived/q3_validation_v1/panel_q3.sqlite": Q3_PANEL,
        "data/derived/simple_baseline_v2/model_results.sqlite": ROOT / "data/derived/simple_baseline_v2/model_results.sqlite",
    }.items():
        expected = q3_bindings.get(relative)
        if expected is not None and expected != sha256(path):
            raise AnalysisError(f"Q3 source binding differs for {relative}")
    if q3_bindings and q3_bindings.get("data/derived/q3_validation_v1/panel_q3.sqlite") != sha256(Q3_PANEL):
        raise AnalysisError("Q3 panel is not the manifest-bound label source")
    required_q4_artifacts = {"work/evaluation/selection.sqlite", "work/prepared/panel.sqlite", "work/evaluation/evaluation.json", "work/evaluation/current_lr.json", "snapshot/pipeline/labeling.py"}
    if not required_q4_artifacts.issubset(q4_complete.get("artifacts", {})):
        raise AnalysisError("Q4 completion manifest lacks a required evaluation artifact")
    for name, artifact in q4_complete.get("artifacts", {}).items():
        if name in {"work/evaluation/selection.sqlite", "work/prepared/panel.sqlite", "work/evaluation/evaluation.json", "work/evaluation/current_lr.json", "snapshot/pipeline/labeling.py"}:
            path = Q4_ROOT / artifact["path"].split("continuation_001/", 1)[-1]
            if path != {"work/evaluation/selection.sqlite": Q4_SELECTION, "work/prepared/panel.sqlite": Q4_PANEL, "work/evaluation/evaluation.json": Q4_EVAL, "work/evaluation/current_lr.json": Q4_MODEL, "snapshot/pipeline/labeling.py": Q4_LABELING}[name]:
                raise AnalysisError(f"Q4 artifact path does not resolve as expected: {name}")
            if sha256(path) != artifact.get("sha256"):
                raise AnalysisError(f"Q4 artifact SHA mismatch: {path}")
    cap_manifest = read_json(CAPACITY_MANIFEST)
    if (cap_manifest.get("status") != "complete"
            or cap_manifest.get("recovery", {}).get("status") != "complete"
            or cap_manifest.get("output_sha256", {}).get("database") != sha256(CAPACITY_DB)):
        raise AnalysisError("capacity manifest does not bind the immutable capacity database")
    q4_eval = read_json(Q4_EVAL)
    if q4_eval.get("status") != "pass":
        raise AnalysisError("Q4 evaluation status is not pass")
    score_proof = reconstruct_fixed_scores(q4_model)
    return {
        "status": "pass", "elapsed_seconds": time.monotonic() - started,
        "fixed_key_score_reconstruction": score_proof,
        "input_sha256": bindings, "q3_metadata": q3_meta, "q4_metadata": q4_meta,
        "q4_model": {"model": q4_model.get("model"), "feature_count": q4_model.get("feature_count"),
                     "model_run_status": q4_model.get("model_run_status")},
        "q4_evaluation_status": q4_eval.get("status"), "plan_sha256": sha256(PLAN),
        "post_hoc": True, "fit": 0, "q1_content_read": False,
    }


def reconstruct_fixed_scores(model):
    """Fixed first 32 date/serial keys per quarter; no new full scoring."""
    result = {}
    for quarter, path in [('q3',Q3_DB),('q4',Q4_SELECTION)]:
        db=ro_db(path)
        try:
            if quarter=='q3':
                query='''SELECT f.*,s.current_lr_score AS saved_score FROM feature_rows f
                    JOIN model_scores s USING(decision_date,serial_number)
                    ORDER BY f.decision_date,f.serial_number LIMIT 32'''
                samples=[(dict(r),r['saved_score']) for r in db.execute(query)]
            else:
                query='''SELECT f.payload_json,s.score FROM features f JOIN scores s
                    USING(decision_date,serial_number) WHERE s.model='current_lr'
                    ORDER BY f.decision_date,f.serial_number LIMIT 32'''
                samples=[(json.loads(r['payload_json']),r['score']) for r in db.execute(query)]
            if len(samples)!=32:raise AnalysisError('fixed key reconstruction missing rows')
            errors=[abs(independent_verifier.linear_score(model,f)-z) for f,z in samples]
            if max(errors)>1e-10:raise AnalysisError(f'{quarter}: saved score is not the bound linear score')
            result[quarter]={'keys':32,'ordering':'decision_date,serial_number ascending','maximum_error':max(errors)}
        finally:db.close()
    return result


def sigmoid(logit: float) -> float:
    if not math.isfinite(logit):
        raise AnalysisError("non-finite logit")
    if logit >= 0:
        z = math.exp(-logit) if logit < 745 else 0.0
        return 1.0 / (1.0 + z)
    z = math.exp(logit) if logit > -745 else 0.0
    return z / (1.0 + z)


def log_loss_from_logit(logit: float, label: int) -> float:
    if label not in (0, 1) or not math.isfinite(logit):
        raise AnalysisError("invalid log-loss input")
    return max(0.0, logit) - logit * label + math.log1p(math.exp(-abs(logit)))


def average_precision_rows(rows: Sequence[tuple[float, int | None, str]]) -> dict[str, Any]:
    import numpy as np
    if any((not math.isfinite(score)) or (label not in (None, 0, 1)) for score, label, _ in rows):
        raise AnalysisError("invalid AP row")
    dtype = np.dtype([('score', 'f8'), ('label', 'i1')])
    known = np.fromiter(((score, label) for score, label, _ in rows if label is not None), dtype=dtype)
    known.sort(order='score')
    known = known[::-1]
    positives = int(known['label'].sum())
    result = {'value':None, 'known_rows':len(known), 'positive_rows':positives,
              'unknown_rows':len(rows)-len(known), 'curve':[]}
    if not positives:
        return result
    starts = np.r_[0, np.flatnonzero(np.diff(known['score']) != 0)+1]
    counts = np.diff(np.r_[starts, len(known)])
    positive_counts = np.add.reduceat(known['label'].astype('i8'), starts)
    seen = np.cumsum(counts); hits = np.cumsum(positive_counts)
    precision = hits / seen
    result['value'] = float(np.sum((positive_counts / positives) * precision))
    # A recall query takes the first threshold reaching its target; groups
    # without new positives cannot change that answer (except the first).
    for i in np.flatnonzero((positive_counts > 0) | (starts == 0)):
        result['curve'].append({'threshold':float(known['score'][starts[i]]),
            'recall':float(hits[i]/positives), 'precision':float(precision[i]),
            'group_rows':int(counts[i]), 'group_positive':int(positive_counts[i])})
    return result


def fixed_recall_points(curve: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for i in range(101):
        target = i / 100
        point = next((item for item in curve if item["recall"] >= target), None)
        output.append({"target_recall": target, "actual_recall": point["recall"] if point else None,
                       "precision": point["precision"] if point else None,
                       "threshold": point["threshold"] if point else None})
    return output


def calibration_rows(rows: Sequence[tuple[float, int | None, str]]) -> dict[str, Any]:
    bins = []
    for index, lo in enumerate(BIN_EDGES[:-1]):
        hi = BIN_EDGES[index + 1]
        bins.append({"bin": index, "lower": lo, "upper": hi, "include_upper": index == len(BIN_EDGES) - 2,
                     "all": 0, "known": 0, "positive": 0, "unknown": 0, "sum_p_all": 0.0,
                     "sum_p_known": 0.0, "sum_brier_known": 0.0, "sum_logloss_known": 0.0,
                     "devices": set(), "positive_devices": set()})
    known_brier = known_logloss = sum_p = 0.0
    known = positive = 0
    for score, label, serial in rows:
        p = sigmoid(score)
        index = next(
            (i for i in range(len(bins))
             if p >= bins[i]["lower"] and
             (p < bins[i]["upper"] or (bins[i]["include_upper"] and p <= bins[i]["upper"]))),
            None,
        )
        if index is None:
            raise AnalysisError(f"probability outside fixed bins: {p}")
        item = bins[index]
        item["all"] += 1; item["sum_p_all"] += p; item["devices"].add(serial)
        if label is None:
            item["unknown"] += 1
            continue
        item["known"] += 1; item["sum_p_known"] += p; known += 1; positive += int(label)
        brier_value = (p - label) ** 2
        logloss_value = log_loss_from_logit(score, label)
        known_brier += brier_value; known_logloss += logloss_value; sum_p += p
        item["sum_brier_known"] += brier_value; item["sum_logloss_known"] += logloss_value
        if label:
            item["positive"] += 1; item["positive_devices"].add(serial)
    output_bins = []
    for item in bins:
        all_n = item["all"]; known_n = item["known"]
        output_bins.append({"bin": item["bin"], "lower": item["lower"], "upper": item["upper"],
                            "include_upper": item["include_upper"], "all_rows": all_n, "known_rows": known_n,
                            "positive_rows": item["positive"], "unknown_rows": item["unknown"],
                            "unknown_ratio": item["unknown"]/all_n if all_n else None,
                            "distinct_devices": len(item["devices"]), "positive_devices": len(item["positive_devices"]),
                            "mean_probability_all": item["sum_p_all"] / all_n if all_n else None,
                            "mean_probability_known": item["sum_p_known"] / known_n if known_n else None,
                            "sum_brier_known": item["sum_brier_known"],
                            "sum_logloss_known": item["sum_logloss_known"],
                            "known_rate": item["positive"] / known_n if known_n else None,
                            "all_rate_lower": item["positive"] / all_n if all_n else None,
                            "all_rate_upper": (item["positive"] + item["unknown"]) / all_n if all_n else None,
                            # The inferential warning is about distinct
                            # positive-label devices, not the number of
                            # device-day rows.  Keep the old raw-row fact for
                            # transparency, but never use it as the gate.
                            "sparse_known_rows": bool(0 < known_n < 20),
                            "sparse_positive_devices": bool(all_n > 0 and len(item["positive_devices"]) < 20),
                            "empty_bin": all_n == 0})
    prevalence = positive / known if known else None
    zero_brier = prevalence_brier = constant_logloss = None
    if known:
        zero_brier = positive / known
        prevalence_brier = prevalence * (1.0 - prevalence)
        constant_logloss = -(positive * math.log(prevalence) + (known - positive) * math.log1p(-prevalence)) / known if 0 < prevalence < 1 else 0.0
    return {"bins": output_bins, "known_rows": known, "positive_rows": positive,
            "positive_devices": len({serial for score, label, serial in rows if label == 1}),
            "unknown_rows": len(rows) - known,
            "unknown_ratio": (len(rows)-known)/len(rows) if len(rows) else None,
            "predicted_positive_known": sum_p,
            "observed_positive_known": positive,
            "predicted_observed_ratio_known": sum_p/positive if positive else None,
            "mean_probability_known": sum_p / known if known else None, "known_prevalence": prevalence,
            "brier": known_brier / known if known else None, "log_loss": known_logloss / known if known else None,
            "constant_zero_brier": zero_brier, "constant_prevalence_brier": prevalence_brier,
            "constant_prevalence_log_loss": constant_logloss}


def load_q3_rows(method: str) -> list[tuple[float, int | None, str]]:
    db = ro_db(Q3_DB)
    panel_guard = ro_db(Q3_PANEL)
    panel_guard.close()
    try:
        db.execute("ATTACH DATABASE ? AS q3_panel", (Q3_PANEL.resolve().as_uri() + "?mode=ro&immutable=1",))
        score_col = "current_lr_score" if method == "current_lr" else "smart_nonzero_signal_count"
        duplicate_labels = db.execute("""SELECT COUNT(*) FROM (
            SELECT l.decision_date, l.serial_number, COUNT(*) AS n
            FROM q3_panel.label_flow l
            WHERE l.run_id='validation_q3_h7_v1' AND l.split='validation'
              AND l.horizon_days=7 AND l.model='ST4000DM000' AND l.eligible=1
            GROUP BY l.decision_date, l.serial_number HAVING n > 1
        )""").fetchone()[0]
        if duplicate_labels:
            raise AnalysisError(f"Q3 label join has {duplicate_labels} duplicate keys")
        query = f"""SELECT s.{score_col} AS score, l.label, s.serial_number
                    FROM model_scores s JOIN q3_panel.label_flow l
                    ON l.decision_date=s.decision_date AND l.serial_number=s.serial_number
                    WHERE l.run_id='validation_q3_h7_v1' AND l.split='validation' AND l.horizon_days=7
                      AND l.model='ST4000DM000'
                      AND l.eligible=1 ORDER BY s.decision_date, s.serial_number"""
        score_count = db.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
        rows = PackedRows()
        for batch in batches(db.execute(query)):
            for row in batch:
                rows.append((float(row['score']), None if row['label'] is None else int(row['label']), str(row['serial_number'])))
        if len(rows) != score_count:
            raise AnalysisError(f"Q3 label join lost keys: scores={score_count}, joined={len(rows)}")
        if len(rows) != 1497209:
            raise AnalysisError(f"Q3 {method} row count {len(rows)} != 1497209")
        return rows
    finally:
        db.close()


def assert_q4_score_keys(selection):
    # Compare keys inside SQLite, avoiding two million-entry Python sets.
    for left, right in (('current_lr','smart_nonzero'),('smart_nonzero','current_lr')):
        mismatch = selection.execute('''SELECT decision_date,serial_number FROM scores WHERE model=?
            EXCEPT SELECT decision_date,serial_number FROM scores WHERE model=? LIMIT 1''',(left,right)).fetchone()
        if mismatch is not None:raise AnalysisError('Q4 method key sets differ')
    duplicate = selection.execute('''SELECT 1 FROM scores WHERE model IN ('current_lr','smart_nonzero')
        GROUP BY model,decision_date,serial_number HAVING COUNT(*)>1 LIMIT 1''').fetchone()
    if duplicate is not None:raise AnalysisError('duplicate Q4 score key')


def load_q4_rows() -> tuple[dict[str, list[tuple[float, int | None, str]]], dict[str, list[dict[str, Any]]]]:
    """Stream both immutable Q4 files grouped by serial, applying the sealed label function."""
    module_spec = importlib.util.spec_from_file_location("q4_snapshot_labeling", Q4_LABELING)
    if module_spec is None or module_spec.loader is None:
        raise AnalysisError("cannot load the bound Q4 labeling snapshot")
    labeling = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(labeling)
    selection = ro_db(Q4_SELECTION); panel = ro_db(Q4_PANEL)
    try:
        scores = selection.execute("SELECT serial_number,model,decision_date,score FROM scores WHERE model IN ('current_lr','smart_nonzero') ORDER BY serial_number,decision_date,model")
        daily = panel.execute("SELECT serial_number,date,model,capacity_bytes,failure,smart_5_raw,smart_9_raw,smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw FROM daily ORDER BY serial_number,date")
        selections = {(str(r["serial_number"]), str(r["decision_date"]), str(r["model"])): r for r in selection.execute("SELECT * FROM selections")}
        score_groups = groupby(scores, key=lambda r: str(r["serial_number"]))
        day_groups = groupby(daily, key=lambda r: str(r["serial_number"]))
        score_group = next(score_groups, None); day_group = next(day_groups, None)
        result = {method: PackedRows() for method in METHODS}; alerts = {method: [] for method in METHODS}
        assert_q4_score_keys(selection)
        selection_keys = {method: set() for method in METHODS}
        for key in selections:
            selection_keys[key[2]].add(key[:2])
        total = 0
        while score_group is not None:
            score_serial, score_rows_it = score_group
            # The prepared panel contains the full eligible population while
            # the sealed score table contains only the history-qualified
            # population. Advance over panel-only serials; a score serial
            # without a panel remains a hard binding error.
            while day_group is not None and day_group[0] < score_serial:
                day_group = next(day_groups, None)
            if day_group is None:
                raise AnalysisError(f"Q4 scores have no panel serial: {score_serial}")
            day_serial, day_rows_it = day_group
            if score_serial != day_serial:
                raise AnalysisError(f"Q4 serial sets differ: score={score_serial}, panel={day_serial}")
            day_rows = []
            for day_row in day_rows_it:
                if len(day_rows) >= 50000:raise AnalysisError('device history exceeds batch bound')
                day_rows.append(day_row)
            label_rows = labeling.classify_device_rows(day_rows, start="2023-10-01", end="2023-12-24", dataset_end="2023-12-31", horizon_days=7, history_days=14, min_history=12)
            labels = {str(row["decision_date"]): row for row in label_rows if int(row["eligible"]) == 1}
            independent_labels = {}
            independent_dates = independent_verifier.prepare_outcome_dates(day_rows)
            for row in score_rows_it:
                method = str(row["model"]); day = str(row["decision_date"])
                if day not in labels:
                    raise AnalysisError(f"Q4 score key has no eligible label: {score_serial}/{day}")
                label_row = labels[day]; label = label_row["label"]
                if day not in independent_labels:
                    independent_labels[day] = independent_verifier.outcome_from_prepared(independent_dates, day)
                if label != independent_labels[day]:
                    raise AnalysisError(f'independent date-set label mismatch: {score_serial}/{day}')
                result[method].append((float(row["score"]), None if label is None else int(label), score_serial))
                key = (score_serial, day, method)
                if key in selections:
                    selection_keys[method].discard((score_serial, day))
                    alerts[method].append({"serial_number": score_serial, "decision_date": day, "score": float(row["score"]),
                                           "label": None if label is None else int(label), "first_failure_date": label_row["first_failure_date"],
                                           "event_key": (f"{score_serial}|{label_row['first_failure_date']}"
                                                         if label_row["first_failure_date"] is not None else None),
                                           "event_hit": int(label == 1 and label_row["first_failure_date"] is not None and "2023-10-08" <= label_row["first_failure_date"] <= "2023-12-25")})
                total += 1
            score_group = next(score_groups, None); day_group = next(day_groups, None)
        if any(selection_keys[m] for m in METHODS):
            raise AnalysisError("Q4 selection contains a key absent from its score method")
        if any(len(result[m]) != 1216041 for m in METHODS) or any(len(alerts[m]) != 1247 for m in METHODS):
            raise AnalysisError(f"Q4 row counts differ from sealed values: scores={ {m:len(result[m]) for m in METHODS} }, alerts={ {m:len(alerts[m]) for m in METHODS} }")
        return result, alerts
    finally:
        selection.close(); panel.close()


def _event_key(row: dict[str, Any], *, event_start: str, event_end: str) -> str | None:
    """Return the stable first-failure key used for event-level accounting."""
    explicit = row.get("event_key")
    if explicit:
        return str(explicit)
    first_failure = row.get("first_failure_date")
    if first_failure and event_start <= str(first_failure) <= event_end:
        return f"{row.get('serial_number')}|{first_failure}"
    return None


def burden(
    records: Sequence[dict[str, Any]],
    *,
    event_start: str,
    event_end: str,
    eligible_device_days: int = 1_216_041,
    opportunity_events: int | None = None,
) -> dict[str, Any]:
    """Summarise alert burden with event-key de-duplication.

    ``event_hits`` is deliberately the number of distinct first-failure keys
    hit by at least one alert.  Counting alert rows would inflate recall for a
    device that is alerted repeatedly before the same failure.
    """
    if eligible_device_days <= 0:
        raise AnalysisError("eligible_device_days must be positive")
    alerts = len(records)
    unknown = sum(row["label"] is None for row in records)
    known_hits = sum(row["label"] == 1 for row in records)
    known_no = sum(row["label"] == 0 for row in records)
    hit_keys = {
        key for row in records
        if row.get("event_hit", 0) and (key := _event_key(row, event_start=event_start, event_end=event_end))
    }
    event_keys = {
        key for row in records
        if (key := _event_key(row, event_start=event_start, event_end=event_end))
    }
    counts = Counter(str(row["serial_number"]) for row in records)
    first = len(counts)
    event_hits = len(hit_keys)
    return {
        "alerts": alerts,
        "known_hit_alerts": known_hits,
        "known_no_hit_alerts": known_no,
        "unknown_alerts": unknown,
        "unknown_ratio": unknown / alerts if alerts else None,
        "event_hits": event_hits,
        "event_keys_observed": len(event_keys),
        "event_keys_missing": sum(1 for row in records if row.get("event_hit", 0) and _event_key(row, event_start=event_start, event_end=event_end) is None),
        "opportunity_events": opportunity_events,
        "event_recall": event_hits / opportunity_events if opportunity_events else None,
        "alerted_devices": len(counts),
        "first_alerts": first,
        "later_alerts": alerts - first,
        "later_alert_share": (alerts-first)/alerts if alerts else None,
        "capture_status": "captured" if event_hits else "no_captured_event",
        "repeat_alert_devices": sum(v > 1 for v in counts.values()),
        "alerts_per_event": alerts / event_hits if event_hits else None,
        "alerted_devices_per_event": len(counts) / event_hits if event_hits else None,
        "eligible_device_days": eligible_device_days,
        "alerts_per_1000_device_days": alerts / eligible_device_days * 1000 if alerts else 0.0,
        "known_precision": known_hits / (known_hits + known_no) if known_hits + known_no else None,
        "unknown_precision_lower": known_hits / alerts if alerts else None,
        "unknown_precision_upper": (known_hits + unknown) / alerts if alerts else None,
        "outside_main_event_alerts": sum(row["label"] == 1 and not row.get("event_hit", 0) for row in records),
        "outside_main_event_keys": len({
            key for row in records
            if row["label"] == 1 and not row.get("event_hit", 0)
            and (key := _event_key(row, event_start=event_start, event_end=event_end))
        }),
    }


def read_q3_alerts() -> dict[str, list[dict[str, Any]]]:
    db = ro_db(Q3_DB)
    try:
        out = {m: [] for m in METHODS}
        for row in db.execute("SELECT model,decision_date,serial_number,label,event_hit,first_failure_date,event_key FROM model_alerts WHERE model IN ('current_lr','smart_nonzero')"):
            out[str(row["model"])].append({"serial_number": str(row["serial_number"]), "decision_date": str(row["decision_date"]),
                "label": None if row["label"] is None else int(row["label"]), "event_hit": int(row["event_hit"]),
                "first_failure_date": row["first_failure_date"], "event_key": row["event_key"]})
        if any(len(out[m]) != 1509 for m in METHODS):
            raise AnalysisError("Q3 alert rows differ from the sealed 1,509 per method")
        return out
    finally:
        db.close()


def saved_ap_anchors():
    db = ro_db(Q3_DB)
    try:
        q3 = {r['model']: r['average_precision_known'] for r in db.execute(
            "SELECT model,average_precision_known FROM model_metrics WHERE model IN ('current_lr','smart_nonzero')")}
    finally:
        db.close()
    q4 = read_json(Q4_EVAL)['protocol_metrics']['methods']
    return {'q3': q3, 'q4': {m:q4[m]['average_precision_known']['value'] for m in METHODS}}


def write_csv(path, rows, fieldnames):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(out, pr_rows, bin_rows):
    """Static diagnostic figures with explicit sample counts and limits."""
    os.environ['MPLCONFIGDIR'] = str(out.parent / '.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'svg.fonttype':'none', 'font.size':10}):
        fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
        for q in ('q3','q4'):
            for method in METHODS:
                points = [r for r in pr_rows if r['quarter']==q and r['method']==method and r['precision'] is not None]
                ax.plot([r['actual_recall'] for r in points], [r['precision'] for r in points], label=f'{q.upper()} / {method}')
        ax.set(xlim=(0,1), ylim=(0,1), xlabel='Recall (known device-day outcomes)', ylabel='Precision',
               title='Frozen model PR diagnostics — fixed 101 recall queries')
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(out/'pr_curves.svg'); plt.close(fig)
        fig, axes = plt.subplots(2,2,figsize=(14,11),layout='constrained',gridspec_kw={'height_ratios':[1,1.1]})
        for col,q in enumerate(('q3','q4')):
            rows = [r for r in bin_rows if r['quarter']==q]
            ax=axes[0,col]
            points=[r for r in rows if r['known_rows']]
            ax.plot([0,1],[0,1],ls='--',color='gray',label='Perfect reliability')
            ax.plot([r['mean_probability_known'] for r in points],[r['known_rate'] for r in points],marker='o',label='Known outcomes')
            sparse=[r for r in points if r['sparse_positive_devices']]
            ax.scatter([r['mean_probability_known'] for r in sparse],[r['known_rate'] for r in sparse],marker='x',s=80,color='red',label='<20 positive devices')
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability (known)',ylabel='Observed positive fraction (known)',title=f'{q.upper()} frozen current_lr')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            table_ax=axes[1,col];table_ax.axis('off')
            cells=[]
            for r in rows:
                interval=f"[{r['lower']:g}, {r['upper']:g}{']' if r['include_upper'] else ')'}"
                flag='empty' if r['empty_bin'] else ('sparse' if r['sparse_positive_devices'] else '')
                cells.append([interval,r['all_rows'],r['known_rows'],r['unknown_rows'],r['positive_devices'],flag])
            table=table_ax.table(cellText=cells,colLabels=['Bin','All','Known','Unknown','Positive\ndevices','Flag'],loc='center',cellLoc='center',colWidths=[.25,.16,.16,.16,.15,.12])
            table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
            table_ax.set_title('Unknown outcomes excluded from rates; sparse bins are descriptive.',fontsize=9)
        fig.suptitle('Probability reliability — no calibrator fitted',fontsize=15)
        fig.savefig(out/'calibration_current_lr.svg');plt.close(fig)


def run(attempt_id: str, output_root: Path | None = None) -> Path:
    attempt_id = safe_attempt_id(attempt_id)
    expected = OUT_ROOT / f'.staging_{attempt_id}_{os.getppid()}'
    if output_root is None or output_root.resolve() != expected.resolve() or output_root.is_symlink():
        raise AnalysisError('worker requires its parent-owned project staging directory')
    root = output_root.resolve()
    out = root / attempt_id
    if out.exists():
        raise AnalysisError(f"refusing to overwrite existing attempt: {out}")
    out.mkdir(parents=True)
    started = time.monotonic()
    facts = preflight()
    alerts_by_q = {}
    metric_payload: dict[str, Any] = {"schema_version": 1, "status": "complete", "post_hoc": True, "fit": 0, "quarters": {}, "capacity": []}
    pr_csv=[]; calibration_csv=[]; burden_csv=[]
    for quarter in ('q3', 'q4'):
        if quarter == 'q3':
            rows_by_method = {m: load_q3_rows(m) for m in METHODS}
            alerts_by_q[quarter] = read_q3_alerts()
        else:
            rows_by_method, alerts_by_q[quarter] = load_q4_rows()
        metric_payload["quarters"][quarter] = {}
        for method in METHODS:
            rows = rows_by_method[method]
            ap = average_precision_rows(rows); pr_points = fixed_recall_points(ap["curve"])
            metric_payload["quarters"][quarter][method] = {"rows": len(rows), "known_rows": ap["known_rows"], "positive_rows": ap["positive_rows"], "unknown_rows": ap["unknown_rows"], "positive_rate_known": ap["positive_rows"] / ap["known_rows"] if ap["known_rows"] else None, "average_precision_known": ap["value"]}
            item = metric_payload["quarters"][quarter][method]
            item['distinct_devices'] = len({serial for _, _, serial in rows})
            item['ap_over_prevalence'] = ap['value']/item['positive_rate_known'] if item['positive_rate_known'] else None
            item['evaluation_status'] = 'evaluable' if ap['positive_rows'] else 'no_known_positive'
            for point in pr_points: pr_csv.append({"quarter": quarter, "method": method, **point})
            if method == "current_lr":
                cal = calibration_rows(rows); metric_payload["quarters"][quarter][method]["calibration"] = {k:v for k,v in cal.items() if k != "bins"}
                for item in cal["bins"]: calibration_csv.append({"quarter": quarter, "method": method, **item})
        del rows, rows_by_method, ap
    for quarter, methods_alerts in alerts_by_q.items():
        for method in METHODS:
            qfacts = QUARTER_FACTS[quarter]
            b = burden(methods_alerts[method], event_start=qfacts["event_start"], event_end=qfacts["event_end"],
                       eligible_device_days=qfacts["eligible_device_days"],
                       opportunity_events=qfacts["opportunity_events"])
            metric_payload["quarters"][quarter][method]["burden"] = b
            burden_csv.append({"quarter": quarter, "method": method, **b})
    cap = ro_db(CAPACITY_DB)
    try:
        metric_payload['capacity_burden'] = []
        for row in cap.execute("SELECT * FROM metrics WHERE model='current_lr' ORDER BY denominator"):
            d=dict(row); metric_payload["capacity"].append(d)
            records = [dict(r) for r in cap.execute("SELECT * FROM alert_outcomes WHERE model='current_lr' AND denominator=? ORDER BY decision_date,serial_number",(d['denominator'],))]
            b = {'denominator':d['denominator'], **burden(records, **QUARTER_FACTS['q3'])}
            metric_payload['capacity_burden'].append(b)
            burden_csv.append({'quarter':'q3_capacity','method':'current_lr',**b})
        metric_payload["capacity_deltas"]=[]
        for row in cap.execute("SELECT * FROM capacity_deltas WHERE model='current_lr' ORDER BY denominator"):
            # The sealed table contains event-key lists for internal audit. Do
            # not publish device identifiers in this diagnostic export; retain
            # only aggregate delta counts and efficiency fields.
            delta = dict(row)
            delta.pop("added_event_keys_json", None)
            delta.pop("lost_event_keys_json", None)
            metric_payload["capacity_deltas"].append(delta)
    finally: cap.close()
    for q in ("q3","q4"):
        metric_payload["quarters"][q]["current_lr"]["calibration_bins"]=[r for r in calibration_csv if r["quarter"]==q]
    metric_payload["facts"]={"q3_anchor":Q3_EXPECTED,"q4_anchor":Q4_EXPECTED,"q4_unknown_alerts":{"current_lr":38,"smart_nonzero":27},"q4_known_positive_scoring_rows":609,"q4_opportunity_events":86,"q3_opportunity_events":126,
                             "quarter_facts": QUARTER_FACTS}
    metric_payload["preflight"]=facts
    metric_payload["method_scope"]={"primary":list(METHODS),"calibration":["current_lr"],"post_hoc":True,"known_outcomes_only":True}
    write_csv(out/"pr_curve.csv",pr_csv,["quarter","method","target_recall","actual_recall","precision","threshold"])
    write_csv(out/"calibration_bins.csv",calibration_csv,["quarter","method","bin","lower","upper","include_upper","all_rows","known_rows","positive_rows","unknown_rows","unknown_ratio","distinct_devices","positive_devices","mean_probability_all","mean_probability_known","sum_brier_known","sum_logloss_known","known_rate","all_rate_lower","all_rate_upper","sparse_known_rows","sparse_positive_devices","empty_bin"])
    write_csv(out/"burden.csv",burden_csv,sorted({key for row in burden_csv for key in row}))
    write_figures(out, pr_csv, calibration_csv)
    facts["code_sha256"] = sha256(SCRIPT)
    facts["command"] = list(sys.argv)
    from importlib.metadata import version
    facts["environment"] = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
                            "numpy":version('numpy'), "matplotlib":version('matplotlib')}
    (out/"input_manifest.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    (out/"metrics.json").write_text(json.dumps(metric_payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    checks_payload = {
        "schema": "frozen-ml-evaluation-checks-v1",
        "status": "pass",
        "fixture": fixture(),
        "anchors": {"q3": Q3_EXPECTED, "q4": Q4_EXPECTED,
                    "q4_unknown_alerts": {"current_lr": 38, "smart_nonzero": 27}},
        "row_counts": {q: {m: metric_payload['quarters'][q][m]['rows'] for m in METHODS} for q in ('q3','q4')},
        "known_positive_scoring_rows": {q: {m: metric_payload["quarters"][q][m]["positive_rows"] for m in METHODS} for q in ("q3", "q4")},
        "output_contract": ["input_manifest.json", "metrics.json", "calibration_bins.csv", "pr_curve.csv", "burden.csv", "pr_curves.svg", "calibration_current_lr.svg", "CHECKS.json", "summary.json"],
    }
    (out/"CHECKS.json").write_text(json.dumps(checks_payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    resources = resource_snapshot(started, out)
    enforce_resources(resources)
    facts["resources"] = resources
    metric_payload["resources"] = resources
    metric_payload["elapsed_seconds"] = resources["elapsed_seconds"]
    (out/"input_manifest.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    (out/"metrics.json").write_text(json.dumps(metric_payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    resources = resource_snapshot(started, out)
    enforce_resources(resources)
    checks_payload["resources"] = {"status": "pass", **resources}
    (out/"CHECKS.json").write_text(json.dumps(checks_payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    resources = resource_snapshot(started, out)
    enforce_resources(resources)
    output_hashes={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name != "summary.json"}
    summary={"schema":"frozen-ml-evaluation-v1","status":"complete","attempt_id":attempt_id,"post_hoc":True,"real_fit":0,"q1_content_read":False,"outputs":output_hashes,"metrics_sha256":sha256(out/"metrics.json"),"input_manifest_sha256":sha256(out/"input_manifest.json"),"checks_sha256":sha256(out/"CHECKS.json"),"resources":dict(resources),"elapsed_seconds":resources["elapsed_seconds"],"notes":["AP and Q3 capacity diagnostics existed before this attempt; this attempt adds frozen LR probability reliability and a unified burden export.","Calibration is diagnostic only; no calibrator was fitted.","Summary output bytes include this completion marker."]}
    # ``summary.json`` carries the byte total of the complete directory.  Its
    # own length depends on that number, so solve the tiny decimal fixed point
    # before publishing it instead of reporting a pre-summary total.
    base_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file() and p.name != "summary.json")
    guess = base_bytes
    for _ in range(8):
        summary["resources"]["output_bytes"] = guess
        payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        total = base_bytes + len(payload.encode("utf-8"))
        if total == guess:
            break
        guess = total
    if guess > MAX_OUTPUT_BYTES:
        raise AnalysisError("resource guard: projected output including summary exceeds limit")
    summary["resources"]["output_bytes"] = guess
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    final_resources = resource_snapshot(started, out)
    final_resources["output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    try:
        enforce_resources(final_resources)
    except AnalysisError:
        (out/"summary.json").unlink(missing_ok=True)
        raise
    return out


def audit(out: Path) -> dict[str, Any]:
    metrics=read_json(out/"metrics.json"); summary=read_json(out/"summary.json"); facts=read_json(out/"input_manifest.json"); checks_file=read_json(out/"CHECKS.json")
    verified = preflight()
    independent_verifier.check_artifacts(out, summary, facts, verified['input_sha256'], ROOT)
    if metrics.get("status")!="complete" or summary.get("status")!="complete" or not metrics.get("post_hoc"):
        raise AnalysisError("output completion markers invalid")
    if summary.get("attempt_id") != out.name or not safe_attempt_id(str(summary.get("attempt_id"))):
        raise AnalysisError("summary attempt id is not bound to its output directory")
    if checks_file.get("status") != "pass" or summary.get("checks_sha256") != sha256(out/"CHECKS.json"):
        raise AnalysisError("CHECKS.json is missing or not bound by summary")
    for name, expected in summary.get("outputs", {}).items():
        path = out / name
        if not path.is_file() or sha256(path) != expected:
            raise AnalysisError(f"output hash mismatch: {name}")
    for path, expected in facts["input_sha256"].items():
        if sha256(ROOT/path)!=expected: raise AnalysisError(f"input changed: {path}")
    checks={"q3":Q3_EXPECTED,"q4":Q4_EXPECTED}
    for q, expected in checks.items():
        for method, anchor in expected.items():
            burden_value=metrics["quarters"][q][method]["burden"]
            if burden_value["alerts"]!=anchor["alerts"] or burden_value["event_hits"]!=anchor["event_hits"]:
                raise AnalysisError(f"anchor mismatch {q}/{method}: {burden_value}")
            cal=metrics["quarters"][q][method]["calibration"] if method=="current_lr" else None
            if cal and cal["known_rows"]+cal["unknown_rows"]!=metrics["quarters"][q][method]["rows"]:
                raise AnalysisError(f"calibration count mismatch {q}")
            if q == "q4" and burden_value["unknown_alerts"] != {"current_lr": 38, "smart_nonzero": 27}[method]:
                raise AnalysisError(f"Q4 unknown-alert anchor mismatch {q}/{method}")
    if metrics["quarters"]["q4"]["current_lr"]["positive_rows"] != 609:
        raise AnalysisError("Q4 known positive scoring-row anchor mismatch")
    pr= list(csv.DictReader((out/"pr_curve.csv").open(encoding="utf-8")))
    if len(pr)!=404: raise AnalysisError(f"PR curve row count {len(pr)} != 404")
    bins=list(csv.DictReader((out/"calibration_bins.csv").open(encoding="utf-8")))
    if len(bins)!=18: raise AnalysisError(f"calibration bin row count {len(bins)} != 18")
    if {row.get("sparse_positive_devices") for row in bins} - {"True", "False", ""}:
        raise AnalysisError("calibration sparse-positive field is malformed")

    # Independent numerical audit.  This deliberately re-derives the
    # formulas from immutable source rows rather than trusting metrics.json,
    # CHECKS.json, or the production calibration result.  It is what catches
    # a self-consistent but false metric whose output hash was re-written.
    source_alerts = {}
    independent_checks = 0
    ap_anchors = saved_ap_anchors()
    for quarter in ("q3", "q4"):
        if quarter == 'q3':
            rows_by_method = {m: load_q3_rows(m) for m in METHODS}
            source_alerts[quarter] = read_q3_alerts()
        else:
            rows_by_method, source_alerts[quarter] = load_q4_rows()
        qfacts = QUARTER_FACTS[quarter]
        for method in METHODS:
            rows = rows_by_method[method]
            ranked = independent_verifier.ranking(rows)
            stored = metrics['quarters'][quarter][method]
            independent_verifier.require_equal(stored['average_precision_known'],ranked['ap'],f'{quarter}/{method}/AP')
            if ap_anchors[quarter][method] is not None:
                independent_verifier.require_equal(ranked['ap'],ap_anchors[quarter][method],f'{quarter}/{method}/original AP anchor')
            for field,key in [('known_rows','known'),('positive_rows','positive'),('unknown_rows','unknown')]:
                independent_verifier.require_equal(stored[field],ranked[key],f'{quarter}/{method}/{field}')
            rate = ranked['positive']/ranked['known'] if ranked['known'] else None
            extra = {'positive_rate_known':rate,'distinct_devices':len({serial for _,_,serial in rows}),
                     'ap_over_prevalence':ranked['ap']/rate if rate else None,
                     'evaluation_status':'evaluable' if ranked['positive'] else 'no_known_positive'}
            for field, value in extra.items():
                independent_verifier.require_equal(stored[field],value,f'{quarter}/{method}/{field}')
            expected_points=[{'quarter':quarter,'method':method,**point} for point in ranked['points']]
            selected=[r for r in pr if r['quarter']==quarter and r['method']==method]
            independent_verifier.compare_csv(selected,expected_points,f'{quarter}/{method}/PR')
            independent_checks += 4
            if method == "current_lr":
                computed = independent_verifier.calibration(rows)
                stored_cal = metrics["quarters"][quarter][method]["calibration"]
                independent_verifier.require_equal(stored_cal, {k:v for k,v in computed.items() if k != 'bins'}, f'{quarter}/calibration')
                saved_bins = metrics['quarters'][quarter][method]['calibration_bins']
                csv_bins = [r for r in bins if r['quarter']==quarter and r['method']==method]
                if len(saved_bins) != 9 or len(csv_bins) != 9:
                    raise AnalysisError('missing or duplicate calibration bins')
                for expected, saved, csv_row in zip(computed['bins'], saved_bins, csv_bins):
                    expected = {'quarter':quarter, 'method':method, **expected}
                    independent_verifier.require_equal(saved, expected, f'{quarter}/JSON bin')
                    if set(csv_row) != set(expected):
                        raise AnalysisError('CSV bin fields differ')
                    parsed = {k:independent_verifier.csv_value(csv_row[k],v) for k,v in expected.items()}
                    independent_verifier.require_equal(parsed, expected, f'{quarter}/CSV bin')
                independent_checks += 2
        for method in METHODS:
            actual = independent_verifier.alert_burden(source_alerts[quarter][method], **qfacts)
            stored = metrics['quarters'][quarter][method]['burden']
            independent_verifier.require_equal(stored,actual,f'{quarter}/{method}/burden')
            independent_checks += len(actual)
        del rows, rows_by_method, ranked

    with (out/'burden.csv').open(encoding='utf-8') as stream:
        burden_rows=list(csv.DictReader(stream))
    expected_burden=[]
    for q in ('q3','q4'):
        for m in METHODS:
            expected_burden.append({'quarter':q,'method':m,**independent_verifier.alert_burden(source_alerts[q][m],**QUARTER_FACTS[q])})
    capacity=ro_db(CAPACITY_DB)
    try:
        saved_capacity=[dict(r) for r in capacity.execute("SELECT * FROM metrics WHERE model='current_lr' ORDER BY denominator")]
        independent_verifier.require_equal(metrics['capacity'],saved_capacity,'capacity')
        capacity_burden = []
        for row in saved_capacity:
            records = [dict(r) for r in capacity.execute("SELECT * FROM alert_outcomes WHERE model='current_lr' AND denominator=? ORDER BY decision_date,serial_number",(row['denominator'],))]
            b = {'denominator':row['denominator'], **independent_verifier.alert_burden(records, **QUARTER_FACTS['q3'])}
            capacity_burden.append(b)
            expected_burden.append({'quarter':'q3_capacity','method':'current_lr',**b})
        independent_verifier.require_equal(metrics['capacity_burden'],capacity_burden,'capacity burden')
        deltas=[]
        base={r[0] for r in capacity.execute("SELECT event_key FROM event_summary WHERE model='current_lr' AND denominator=1000 AND opportunity=1 AND hit=1")}
        for row in capacity.execute("SELECT * FROM capacity_deltas WHERE model='current_lr' ORDER BY denominator"):
            delta=dict(row)
            keys={r[0] for r in capacity.execute("SELECT event_key FROM event_summary WHERE model='current_lr' AND denominator=? AND opportunity=1 AND hit=1",(delta['denominator'],))}
            for field,expected in [('added_event_count',len(keys-base)),('lost_event_count',len(base-keys)),('event_delta',len(keys)-len(base))]:
                independent_verifier.require_equal(delta[field],expected,f'capacity/{field}')
            for field,expected in [('added_event_keys_json',keys-base),('lost_event_keys_json',base-keys)]:
                if set(json.loads(delta.pop(field)))!=expected:raise AnalysisError(f'capacity {field} differs')
            deltas.append(delta)
        independent_verifier.require_equal(metrics['capacity_deltas'],deltas,'capacity deltas')
    finally:capacity.close()
    columns=set().union(*(r.keys() for r in expected_burden))
    normalized=[{key:r.get(key) for key in columns} for r in expected_burden]
    independent_verifier.compare_csv(burden_rows,normalized,'burden CSV')

    actual_output_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    resources = summary.get("resources", {})
    if int(resources.get("output_bytes", -1)) != actual_output_bytes:
        raise AnalysisError(f"summary output_bytes {resources.get('output_bytes')} != actual {actual_output_bytes}")
    enforce_resources({**resources, "output_bytes": actual_output_bytes}, include_output=True)
    for path, expected in facts['input_sha256'].items():
        if sha256(ROOT/path) != expected:raise AnalysisError(f'input changed during audit: {path}')
    return {"status":"pass","checked_input_hashes":len(facts["input_sha256"]),"pr_rows":len(pr),"calibration_rows":len(bins),"anchor_checks":4,"independent_checks":independent_checks,"resource_status":"pass"}


def fixture() -> dict[str, Any]:
    rows=[(2.0,1,"a"),(2.0,0,"b"),(1.0,None,"c"),(0.0,0,"d")]
    ap=average_precision_rows(rows)
    # The positive is tied with one negative; grouped ties contribute the
    # group precision (1/2) rather than receiving an arbitrary rank bonus.
    if abs(ap["value"]-0.5)>1e-12: raise AnalysisError("fixture AP tie handling failed")
    cal=calibration_rows([(1000.0,1,"a"),(-1000.0,0,"b")])
    if cal["known_rows"]!=2 or not math.isfinite(cal["log_loss"]): raise AnalysisError("fixture extreme logit failed")
    return {"status":"pass","checks":["ties grouped","unknown retained","extreme logits stable","fixed bins"]}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("preflight","fixture","run","audit","_worker")); parser.add_argument("--attempt-id",default="attempt_001"); parser.add_argument("--path",type=Path); parser.add_argument("--staged-root",type=Path)
    args=parser.parse_args()
    try:
        if args.mode=="preflight": print(json.dumps(preflight(),ensure_ascii=False,indent=2))
        elif args.mode=="fixture": print(json.dumps(fixture(),ensure_ascii=False,indent=2))
        elif args.mode=="run":
            if args.staged_root is not None:
                raise AnalysisError('public run cannot override staging root')
            from run_frozen_ml_supervised import supervise
            print(json.dumps(supervise(args.attempt_id)))
        elif args.mode=="_worker": print(run(args.attempt_id, args.staged_root))
        else: print(json.dumps(audit((args.path or OUT_ROOT/args.attempt_id).resolve()),ensure_ascii=False,indent=2))
        return 0
    except (AnalysisError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
