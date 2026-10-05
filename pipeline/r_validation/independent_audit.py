"""Independent checks for the small R validation chain.

This module intentionally does not import the production feature builder,
scoring function, or selection helper.  It reconstructs the current-feature
subset, score, rank, cooldown, and label counts from the panel and approved
parameter payload so that a corrupt intermediate cannot certify itself.
"""

from __future__ import annotations

import datetime as dt
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping


SMART_FIELDS = (5, 9, 187, 188, 197, 198)
NONZERO_FIELDS = (5, 187, 188, 197, 198)
MODEL = "ST4000DM000"
HISTORY_DAYS = 14
MIN_HISTORY = 12
HORIZON = 7
COOLDOWN_DAYS = 7
BUDGET_DENOMINATOR = 1000
RANDOM_SALT = "drive-v1|20260912"


class IndependentAuditError(RuntimeError):
    """A production intermediate differs from an independent reconstruction."""


def _date(value: object) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise IndependentAuditError(f"invalid date: {value!r}") from exc


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IndependentAuditError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise IndependentAuditError(f"{label} is not finite")
    return result


def _tie_break(decision_date: str, serial: str) -> str:
    return hashlib.sha256(
        f"{RANDOM_SALT}|{decision_date}|{serial}".encode("utf-8")
    ).hexdigest()


def _current_features(rows: list[dict[str, Any]], decision_date: str) -> dict[str, Any] | None:
    """Rebuild the 16 current SMART features from raw panel rows."""
    decision = _date(decision_date)
    by_date = {_date(row["date"]): row for row in rows if _date(row["date"]) <= decision}
    current = by_date.get(decision)
    if current is None or int(current["failure"]) != 0:
        return None
    first_failure = min(
        (_date(row["date"]) for row in rows if int(row["failure"]) == 1),
        default=None,
    )
    if first_failure is not None and first_failure <= decision:
        return None
    history = {
        day: by_date[day]
        for day in (decision - dt.timedelta(days=offset) for offset in range(HISTORY_DAYS))
        if day in by_date
    }
    if len(history) < MIN_HISTORY:
        return None
    result: dict[str, Any] = {}
    for field in (5, 9, 187, 197, 198):
        value = current[f"smart_{field}_raw"]
        missing = int(value is None)
        result[f"smart_{field}_current_log1p"] = None if missing else math.log1p(int(value))
        result[f"smart_{field}_current_missing"] = missing
        result[f"smart_{field}_current_nonzero"] = None if missing else int(int(value) > 0)
    value188 = current["smart_188_raw"]
    result["smart_188_current_nonzero"] = None if value188 is None else int(int(value188) > 0)
    result["smart_188_current_missing"] = int(value188 is None)
    result["smart_nonzero_signal_count"] = sum(
        int(current[f"smart_{field}_raw"] is not None and int(current[f"smart_{field}_raw"]) > 0)
        for field in NONZERO_FIELDS
    )
    result["smart_187_signal"] = int(
        current["smart_187_raw"] is not None and int(current["smart_187_raw"]) > 0
    )
    result["decision_date"] = decision_date
    result["serial_number"] = str(current["serial_number"])
    result["model"] = str(current["model"])
    return result


def _score(features: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    total = _finite(model.get("intercept"), "intercept")
    for item in model.get("features", []):
        name = str(item["feature_name"])
        value = features.get(name)
        if value is None:
            value = item["imputation_mean"]
        transformed = (_finite(value, name) - _finite(item["standardization_mean"], f"{name}.mean")) / _finite(item["standardization_scale"], f"{name}.scale")
        total += transformed * _finite(item["coefficient"], f"{name}.coefficient")
    return _finite(total, "score")


def _close(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_reference(
    event_keys: list[str],
    current_hits: Mapping[str, int],
    smart_hits: Mapping[str, int],
    device_flags: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Independent local bootstrap reconstruction for the sealed metrics."""
    import numpy as np

    ordered = sorted((str(serial) for serial in device_flags), key=lambda value: value.encode("utf-8"))
    opportunity = [int(device_flags[serial]["opportunity"]) for serial in ordered]
    current = [int(device_flags[serial]["current_hit"]) for serial in ordered]
    smart = [int(device_flags[serial]["smart_hit"]) for serial in ordered]
    generator = np.random.Generator(np.random.PCG64(20260913))
    differences: list[float] = []
    empty = 0
    for _ in range(2000):
        if not ordered:
            empty += 1
            continue
        index = generator.integers(0, len(ordered), size=len(ordered))
        denominator = sum(opportunity[item] for item in index)
        if denominator == 0:
            empty += 1
            continue
        differences.append(
            (sum(opportunity[item] * current[item] for item in index)
             - sum(opportunity[item] * smart[item] for item in index))
            / denominator * 100.0
        )
    point_denominator = sum(opportunity)
    result: dict[str, Any] = {
        "valid_replicates": len(differences),
        "empty_replicates": empty,
        "seed": 20260913,
        "replicates_requested": 2000,
        "direction": "current_minus_smart",
        "roster_mode": "all_eligible_devices",
        "roster_devices": len(ordered),
        "opportunity_events": point_denominator,
        "point_estimate_pp": (
            (sum(opportunity[i] * current[i] for i in range(len(ordered)))
             - sum(opportunity[i] * smart[i] for i in range(len(ordered))))
            / point_denominator * 100.0
            if point_denominator else None
        ),
        "differences": differences,
    }
    result["interval"] = (
        [_quantile(differences, 0.025), _quantile(differences, 0.975)]
        if point_denominator >= 100 and len(differences) >= 1900 else None
    )
    if point_denominator < 100:
        result["insufficient_events"] = True
        result["minimum_events"] = 100
    elif len(differences) < 1900:
        result["insufficient_valid_replicates"] = True
    else:
        result["interval_pp"] = list(result["interval"])
    return result


def _manual_label(
    rows: list[dict[str, Any]],
    decision: dt.date,
    *,
    horizon: int = HORIZON,
    outcome_cutoff: dt.date | None = None,
) -> int | None:
    dates = {_date(row["date"]) for row in rows}
    failures = sorted(_date(row["date"]) for row in rows if int(row["failure"]) == 1)
    first_failure = failures[0] if failures else None
    if first_failure is not None and first_failure <= decision:
        return None
    horizon_days = [decision + dt.timedelta(days=i) for i in range(1, horizon + 1)]
    if first_failure is not None and first_failure in horizon_days:
        if outcome_cutoff is not None and first_failure > outcome_cutoff:
            return None
        return 1
    if all(day in dates for day in horizon_days) and not any(day in failures for day in horizon_days):
        if outcome_cutoff is not None and horizon_days[-1] > outcome_cutoff:
            return None
        return 0
    return None


def audit_selection(
    selection: Path,
    panel: Path,
    evaluation: Mapping[str, Any],
    audit_path: Path | None,
    *,
    model: Mapping[str, Any],
    source_path: Path | None = None,
    historical_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the complete source-to-metrics chain independently.

    Expected keys are generated from the raw panel and the declared metadata
    date range, never from the stored feature or score tables.  This is what
    makes missing eligible rows, missing scores, and a changed denominator
    observable.
    """
    panel_connection = sqlite3.connect(panel.as_uri() + "?mode=ro&immutable=1", uri=True)
    selection_connection = sqlite3.connect(selection.as_uri() + "?mode=ro&immutable=1", uri=True)
    panel_connection.row_factory = sqlite3.Row
    selection_connection.row_factory = sqlite3.Row
    try:
        from collections.abc import Mapping as ABCMapping
        from functools import lru_cache

        class PanelRows(ABCMapping):
            def __init__(self):
                self._read = lru_cache(maxsize=32)(self._read)
                self.serials = tuple(str(row[0]) for row in panel_connection.execute(
                    "SELECT DISTINCT serial_number FROM daily ORDER BY serial_number"))

            def __iter__(self):
                return iter(self.serials)

            def __len__(self):
                return len(self.serials)

            def __getitem__(self, serial):
                return self._read(serial)

            def _read(self, serial):
                return [dict(row) for row in panel_connection.execute(
                    "SELECT date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_9_raw,"
                    "smart_187_raw,smart_188_raw,smart_197_raw,smart_198_raw "
                    "FROM daily WHERE serial_number=? ORDER BY date", (serial,))]

        raw_by_serial = PanelRows()
        if source_path is not None:
            source_rows: list[tuple[str, str, str, int, int, tuple[Any, ...]]] = []
            with Path(source_path).open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    source_rows.append((
                        str(row["date"]), str(row["serial_number"]), str(row["model"]),
                        int(row["capacity_bytes"]), int(row["failure"]),
                        tuple(None if row[f"smart_{field}_raw"] == "" else int(row[f"smart_{field}_raw"])
                              for field in SMART_FIELDS),
                    ))
            if historical_inputs:
                # Reconstruct the union from bound inputs, independently of
                # the production merge and its receipt counts.
                import hashlib
                metadata_for_history = dict(selection_connection.execute('SELECT key,value FROM metadata'))
                cutoff = metadata_for_history['start']
                first = (_date(cutoff) - dt.timedelta(days=13)).isoformat()
                for role, binding in historical_inputs.items():
                    history_path = Path(binding['path']).resolve()
                    def digest_history():
                        digest = hashlib.sha256()
                        with history_path.open('rb') as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                digest.update(block)
                        return digest.hexdigest()
                    if digest_history() != binding['sha256']:
                        raise IndependentAuditError(f'historical input SHA mismatch: {role}')
                    history = sqlite3.connect(history_path.as_uri()+'?mode=ro&immutable=1', uri=True)
                    try:
                        fields = 'date,serial_number,model,capacity_bytes,failure,' + ','.join(f'smart_{field}_raw' for field in SMART_FIELDS)
                        for row in history.execute(f'SELECT {fields} FROM daily WHERE model=? AND date<?', ('ST4000DM000', cutoff)):
                            if row[0] >= first or row[4] == 1:
                                source_rows.append((row[0], row[1], row[2], row[3], row[4], tuple(row[5:])))
                    finally:
                        history.close()
                    if digest_history() != binding['sha256']:
                        raise IndependentAuditError(f'historical input changed during audit: {role}')
            panel_rows = []
            for serial in sorted(raw_by_serial):
                panel_rows.extend(
                    (str(row["date"]), str(row["serial_number"]), str(row["model"]),
                     int(row["capacity_bytes"]) if "capacity_bytes" in row else 0,
                     int(row["failure"]), tuple(row[f"smart_{field}_raw"] for field in SMART_FIELDS))
                    for row in raw_by_serial[serial]
                )
            if sorted(source_rows) != sorted(panel_rows):
                raise IndependentAuditError("source CSV and panel rows differ")
        metadata = {
            str(row["key"]): str(row["value"])
            for row in selection_connection.execute("SELECT key,value FROM metadata")
        }
        try:
            score_start = _date(metadata["start"])
            score_end = _date(metadata["end"])
            horizon = int(metadata.get("horizon_days", HORIZON))
            outcome_cutoff = _date(metadata.get("outcome_cutoff", score_end.isoformat()))
        except (KeyError, ValueError) as exc:
            raise IndependentAuditError("selection metadata lacks protocol date fields") from exc
        event_start = score_start + dt.timedelta(days=horizon)
        event_end = score_end + dt.timedelta(days=1)
        all_serials = tuple(sorted(raw_by_serial))
        expected_features = set()
        model_names = [str(item["feature_name"]) for item in model["features"]]
        method_inputs = {"current_lr": {}, "smart_nonzero": {}}
        for serial in all_serials:
            stored_features = {str(row[0]): json.loads(row[1]) for row in selection_connection.execute(
                "SELECT decision_date,payload_json FROM features WHERE serial_number=?", (serial,))}
            stored_scores = {(str(row["model"]), str(row["decision_date"])): row
                             for row in selection_connection.execute(
                                 "SELECT * FROM scores WHERE serial_number=?", (serial,))}
            computed_days = set()
            for offset in range((score_end - score_start).days + 1):
                decision = (score_start + dt.timedelta(days=offset)).isoformat()
                computed = _current_features(raw_by_serial[serial], decision)
                if computed is None:
                    continue
                key = (decision, serial)
                expected_features.add(key)
                computed_days.add(decision)
                stored = stored_features.get(decision)
                if stored is None:
                    raise IndependentAuditError("feature key set differs from raw panel reconstruction")
                for name in model_names:
                    if not _close(stored.get(name), computed.get(name)):
                        raise IndependentAuditError(f"feature mismatch: {key}/{name}")
                expected_signal = int(computed["smart_nonzero_signal_count"])
                for method in method_inputs:
                    row = stored_scores.get((method, decision))
                    if row is None:
                        raise IndependentAuditError(f"{method} score key set differs from eligible source keys")
                    if int(row["smart_nonzero_signal_count"]) != expected_signal:
                        raise IndependentAuditError(f"SMART signal count mismatch: {key}")
                    expected_score = _score(computed, model) if method == "current_lr" else expected_signal
                    if not _close(row["score"], expected_score):
                        raise IndependentAuditError(f"{'SMART ' if method == 'smart_nonzero' else ''}score mismatch: {key}")
                    if str(row["tie_break_sha256"]) != _tie_break(*key):
                        raise IndependentAuditError(f"tie break mismatch: {key}")
                    method_inputs[method][key] = (float(row["score"]), expected_signal)
            if set(stored_features) != computed_days:
                raise IndependentAuditError("feature key set differs from raw panel reconstruction")
            if set(stored_scores) != {(method, day) for method in method_inputs for day in computed_days}:
                raise IndependentAuditError("score key set differs from eligible source keys")
        if selection_connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] != len(expected_features):
            raise IndependentAuditError("feature key set differs from raw panel reconstruction")
        if selection_connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0] != 2 * len(expected_features):
            raise IndependentAuditError("score key set differs from eligible source keys")
        keys_by_day = {}
        for key in expected_features:
            keys_by_day.setdefault(key[0], []).append(key)

        actual_selected_by_method: dict[str, list[tuple[str, str, int]]] = {}
        expected_daily: dict[tuple[str, str], dict[str, int]] = {}
        for method in method_inputs:
            selected_rows = [dict(row) for row in selection_connection.execute(
                "SELECT model,decision_date,serial_number,score,tie_break_sha256,selected_rank "
                "FROM selections WHERE model=? ORDER BY decision_date,selected_rank", (method,)
            )]
            last_alert: dict[str, dt.date] = {}
            expected_selected: list[tuple[str, str, int]] = []
            # Include every declared scoring date, including legitimate days
            # with zero eligible devices.  Those rows are part of the budget
            # contract and must not disappear merely because no feature key
            # was stored for that day.
            declared_dates = [
                (score_start + dt.timedelta(days=offset)).isoformat()
                for offset in range((score_end - score_start).days + 1)
            ]
            for decision in declared_dates:
                all_candidates = keys_by_day.get(decision, [])
                candidates = list(all_candidates)
                if method == "current_lr":
                    ranked = sorted(candidates, key=lambda key: (-method_inputs[method][key][0], _tie_break(*key), key[1]))
                else:
                    ranked = sorted(
                        [key for key in candidates if method_inputs[method][key][1] > 0],
                        key=lambda key: (-method_inputs[method][key][1], _tie_break(*key), key[1]),
                    )
                budget = math.ceil(len(all_candidates) / BUDGET_DENOMINATOR) if all_candidates else 0
                rank = 0
                selected_for_day = 0
                cooldown_excluded = 0
                for key in ranked:
                    previous = last_alert.get(key[1])
                    day = _date(decision)
                    if previous is not None and (day - previous).days <= COOLDOWN_DAYS:
                        cooldown_excluded += 1
                        continue
                    if selected_for_day >= budget:
                        continue
                    selected_for_day += 1
                    rank += 1
                    last_alert[key[1]] = day
                    expected_selected.append((key[0], key[1], rank))
                expected_daily[(method, decision)] = {
                    "eligible_count": len(all_candidates),
                    "budget_k": budget,
                    "signal_count": (
                        len(all_candidates)
                        if method == "current_lr"
                        else sum(method_inputs[method][key][1] > 0 for key in all_candidates)
                    ),
                    "cooldown_excluded": cooldown_excluded,
                    "alerts_count": sum(item[0] == decision for item in expected_selected),
                }
            actual = [
                (str(row["decision_date"]), str(row["serial_number"]), int(row["selected_rank"]))
                for row in selected_rows
            ]
            if actual != expected_selected:
                raise IndependentAuditError(f"{method} selection or cooldown differs from reconstruction")
            actual_selected_by_method[method] = actual
        actual_daily_keys = set()
        for row in selection_connection.execute(
            "SELECT model,decision_date,eligible_count,budget_k,signal_count,cooldown_excluded,alerts_count FROM daily"
        ):
            key = (str(row["model"]), str(row["decision_date"]))
            actual_daily_keys.add(key)
            expected = expected_daily.get(key)
            if expected is None or any(int(row[name]) != value for name, value in expected.items()):
                actual_values = {name: int(row[name]) for name in expected} if expected else dict(row)
                raise IndependentAuditError(
                    f"daily budget/stat row mismatch: {key}; expected={expected}; actual={actual_values}"
                )
        if actual_daily_keys != set(expected_daily):
            raise IndependentAuditError("daily key set differs from protocol calendar")

        counts: dict[str, dict[str, int]] = {
            method: {"alerts": 0, "known_hit_alerts": 0, "known_no_hit_alerts": 0, "unknown_alerts": 0}
            for method in ("current_lr", "smart_nonzero")
        }
        for row in selection_connection.execute(
            "SELECT model,decision_date,serial_number FROM selections ORDER BY model,decision_date,selected_rank"
        ):
            method = str(row["model"])
            label = _manual_label(
                raw_by_serial[str(row["serial_number"])], _date(row["decision_date"]),
                horizon=horizon, outcome_cutoff=outcome_cutoff,
            )
            counts[method]["alerts"] += 1
            if label == 1:
                counts[method]["known_hit_alerts"] += 1
            elif label == 0:
                counts[method]["known_no_hit_alerts"] += 1
            else:
                counts[method]["unknown_alerts"] += 1
        event_rows: dict[str, dict[str, Any]] = {}
        for serial, rows in raw_by_serial.items():
            failures = sorted(_date(row["date"]) for row in rows if int(row["failure"]) == 1)
            if not failures or not (event_start <= failures[0] <= event_end):
                continue
            failure = failures[0]
            opportunity = int(any(
                ((failure - dt.timedelta(days=lead)).isoformat(), serial) in expected_features
                for lead in range(1, horizon + 1)
            ))
            event_rows[serial] = {"first_failure_date": failure.isoformat(), "opportunity": opportunity}
        for method, count in counts.items():
            recorded = evaluation.get("methods", {}).get(method, {})
            for key, value in count.items():
                if int(recorded.get(key, -1)) != value:
                    raise IndependentAuditError(f"label count mismatch: {method}/{key}")
            selected = actual_selected_by_method[method]
            hit_count = 0
            for serial, event in event_rows.items():
                if event["opportunity"] and any(
                    item[1] == serial and 1 <= (_date(event["first_failure_date"]) - _date(item[0])).days <= horizon
                    for item in selected
                ):
                    hit_count += 1
            if int(recorded.get("event_total", -1)) != len(event_rows):
                raise IndependentAuditError(f"event total mismatch: {method}")
            if int(recorded.get("event_opportunity_total", -1)) != sum(item["opportunity"] for item in event_rows.values()):
                raise IndependentAuditError(f"event opportunity mismatch: {method}")
            if int(recorded.get("event_hits", -1)) != hit_count:
                raise IndependentAuditError(f"event hit mismatch: {method}")
        protocol = evaluation.get("protocol_metrics")
        if not isinstance(protocol, Mapping):
            raise IndependentAuditError("evaluation has no protocol metrics")
        method_leads: dict[str, list[int]] = {"current_lr": [], "smart_nonzero": []}
        method_event_hits: dict[str, dict[str, int]] = {
            "current_lr": {}, "smart_nonzero": {}
        }
        for method, selected in actual_selected_by_method.items():
            for serial, event in event_rows.items():
                choices = [
                    (_date(event["first_failure_date"]) - _date(item[0])).days
                    for item in selected
                    if item[1] == serial
                    and 1 <= (_date(event["first_failure_date"]) - _date(item[0])).days <= horizon
                ]
                method_event_hits[method][serial] = int(bool(event["opportunity"] and choices))
                if event["opportunity"] and choices:
                    method_leads[method].append(max(choices))
        event_keys = sorted(event_rows)
        eligible_serials = sorted(
            {key[1] for key in expected_features}, key=lambda value: value.encode("utf-8")
        )
        device_flags = {
            serial: {
                "opportunity": int(event_rows.get(serial, {}).get("opportunity", 0)),
                "current_hit": int(method_event_hits["current_lr"].get(serial, 0)),
                "smart_hit": int(method_event_hits["smart_nonzero"].get(serial, 0)),
            }
            for serial in eligible_serials
        }
        bootstrap = _bootstrap_reference(
            event_keys,
            {serial: device_flags.get(serial, {}).get("current_hit", 0) for serial in event_keys},
            {serial: device_flags.get(serial, {}).get("smart_hit", 0) for serial in event_keys},
            device_flags,
        )
        recorded_bootstrap = protocol.get("bootstrap")
        if not isinstance(recorded_bootstrap, Mapping):
            raise IndependentAuditError("protocol metrics have no bootstrap result")
        for key in ("valid_replicates", "empty_replicates", "seed", "replicates_requested",
                    "direction", "roster_mode", "roster_devices", "opportunity_events"):
            if recorded_bootstrap.get(key) != bootstrap.get(key):
                raise IndependentAuditError(f"bootstrap metric mismatch: {key}")
        if len(recorded_bootstrap.get("differences", [])) != len(bootstrap["differences"]):
            raise IndependentAuditError("bootstrap difference vector length mismatch")
        for actual_value, expected_value in zip(recorded_bootstrap["differences"], bootstrap["differences"]):
            if not _close(actual_value, expected_value):
                raise IndependentAuditError("bootstrap difference vector mismatch")
        if recorded_bootstrap.get("interval") != bootstrap.get("interval"):
            raise IndependentAuditError("bootstrap interval mismatch")
        current_recorded = protocol.get("methods", {}).get("current_lr", {})
        smart_recorded = protocol.get("methods", {}).get("smart_nonzero", {})
        for method, recorded_method in (("current_lr", current_recorded), ("smart_nonzero", smart_recorded)):
            if not isinstance(recorded_method, Mapping):
                raise IndependentAuditError(f"protocol method metrics missing: {method}")
            expected_leads = method_leads[method]
            # Independent score-threshold AP; no production metric helper.
            groups: dict[float, list[int]] = {}
            unknown_scores = 0
            for (day, serial), values in method_inputs[method].items():
                label = _manual_label(raw_by_serial[serial], _date(day), horizon=horizon, outcome_cutoff=outcome_cutoff)
                if label is None:
                    unknown_scores += 1
                else:
                    group = groups.setdefault(values[0], [0, 0])
                    group[0] += 1
                    group[1] += label
            known = sum(value[0] for value in groups.values())
            positives = sum(value[1] for value in groups.values())
            ap = None
            if positives:
                cumulative_rows = cumulative_hits = 0
                ap = 0.0
                for score in sorted(groups, reverse=True):
                    size, hits = groups[score]
                    cumulative_rows += size
                    cumulative_hits += hits
                    ap += hits * cumulative_hits / (positives * cumulative_rows)
            expected_ap = {"scope": "all_eligible_scoring_rows_known_labels", "known_rows": known,
                           "positive_rows": positives, "unknown_rows": unknown_scores, "value": ap}
            alert_devices: dict[str, int] = {}
            outside_alerts = []
            for day, serial, _rank in actual_selected_by_method[method]:
                alert_devices[serial] = alert_devices.get(serial, 0) + 1
                failures = sorted(_date(row["date"]) for row in raw_by_serial[serial]
                                  if int(row["failure"]) == 1 and _date(row["date"]) <= outcome_cutoff)
                if failures and 1 <= (failures[0] - _date(day)).days <= horizon and not event_start <= failures[0] <= event_end:
                    outside_alerts.append({"decision_date": day, "serial_number": serial,
                                           "first_failure_date": failures[0].isoformat()})
            alerts = counts[method]["alerts"]
            total = len(expected_features)
            expected_burden = {
                "eligible_device_days": total, "alerts": alerts,
                "alerted_devices": len(alert_devices), "first_alerts": len(alert_devices),
                "later_alerts": alerts - len(alert_devices),
                "repeat_alert_devices": sum(value >= 2 for value in alert_devices.values()),
                "alerts_per_1000_device_days": alerts * 1000 / total if total else None,
                "confirmed_no_hit_per_1000_device_days": counts[method]["known_no_hit_alerts"] * 1000 / total if total else None,
            }
            for field, expected_metric in (("average_precision_known", expected_ap), ("burden", expected_burden)):
                actual_metric = recorded_method.get(field)
                if not isinstance(actual_metric, Mapping) or set(actual_metric) != set(expected_metric):
                    raise IndependentAuditError(f"secondary metric fields mismatch: {method}/{field}")
                for key, value in expected_metric.items():
                    actual = actual_metric[key]
                    valid = (type(actual) is int and actual == value) if type(value) is int else (
                        actual == value if isinstance(value, str) else _close(actual, value))
                    if not valid:
                        raise IndependentAuditError(f"secondary metric mismatch: {method}/{field}/{key}")
            if (recorded_method.get("outside_main_event_alert_count") != len(outside_alerts) or
                    recorded_method.get("outside_main_event_alerts") != sorted(outside_alerts, key=lambda row: (row["decision_date"], row["serial_number"]))):
                raise IndependentAuditError(f"outside event alerts mismatch: {method}")
            for field in ("eligible_device_days", "first_alerts", "later_alerts", "repeat_alert_devices"):
                if evaluation["methods"][method].get(field) != expected_burden[field]:
                    raise IndependentAuditError(f"legacy burden mismatch: {method}/{field}")
            opportunities = sum(item["opportunity"] for item in event_rows.values())
            actual_lead_recall = recorded_method.get("lead_recall")
            if not isinstance(actual_lead_recall, Mapping):
                raise IndependentAuditError(f"lead recall missing: {method}")
            expected_counts = {"opportunity_events": opportunities,
                               "ge2_hits": sum(value >= 2 for value in expected_leads),
                               "ge3_hits": sum(value >= 3 for value in expected_leads)}
            if any(type(actual_lead_recall.get(key)) is not int or actual_lead_recall[key] != value
                   for key, value in expected_counts.items()):
                raise IndependentAuditError(f"lead recall count mismatch: {method}")
            for days in (2, 3):
                expected_ratio = expected_counts[f"ge{days}_hits"] / opportunities if opportunities else None
                if not _close(actual_lead_recall.get(f"ge{days}_recall"), expected_ratio):
                    raise IndependentAuditError(f"lead recall ratio mismatch: {method}")
            expected_stats = {
                "count": len(expected_leads),
                "median": _quantile(expected_leads, 0.5),
                "q25": _quantile(expected_leads, 0.25),
                "q75": _quantile(expected_leads, 0.75),
            }
            actual_stats = recorded_method.get("lead_statistics")
            if not isinstance(actual_stats, Mapping) or any(
                not _close(actual_stats.get(key), value) for key, value in expected_stats.items()
            ):
                raise IndependentAuditError(f"lead statistics mismatch: {method}")
            precision = recorded_method.get("precision")
            if not isinstance(precision, Mapping):
                raise IndependentAuditError(f"precision metrics missing: {method}")
            alerts = counts[method]["alerts"]
            known_hits = counts[method]["known_hit_alerts"]
            unknown = counts[method]["unknown_alerts"]
            expected_precision = {
                "lower": known_hits / alerts if alerts else None,
                "upper": (known_hits + unknown) / alerts if alerts else None,
                "known_outcome": known_hits / (alerts - unknown) if alerts - unknown > 0 else None,
            }
            if any(not _close(precision.get(key), value) for key, value in expected_precision.items()):
                raise IndependentAuditError(f"precision metrics mismatch: {method}")
        expected_delta = None
        opportunity_total = sum(int(item["opportunity"]) for item in event_rows.values())
        if opportunity_total:
            expected_delta = (
                sum(method_event_hits["current_lr"].values())
                - sum(method_event_hits["smart_nonzero"].values())
            ) / opportunity_total * 100.0
        if not _close(protocol.get("delta_recall_pp"), expected_delta):
            raise IndependentAuditError("delta recall metric mismatch")
        eligible_total = len(expected_features)
        unknown_total = sum(
            _manual_label(raw_by_serial[serial], _date(day), horizon=horizon,
                          outcome_cutoff=outcome_cutoff) is None
            for day, serial in sorted(expected_features, key=lambda key: (key[1], key[0]))
        )
        expected_quality_status = (
            "not_evaluable" if not eligible_total
            else ("failed" if unknown_total * 5 > eligible_total else "pass")
        )
        quality = protocol.get("quality_gate", {})
        expected_quality = {
            "scope": "common_eligible_scoring_rows", "eligible_rows": eligible_total,
            "unknown_rows": unknown_total, "status": expected_quality_status,
        }
        if any(quality.get(key) != value for key, value in expected_quality.items()):
            raise IndependentAuditError("quality gate mismatch")
        if not _close(quality.get("unknown_ratio_max"), 0.2) or not _close(
            quality.get("unknown_ratio"), unknown_total / eligible_total if eligible_total else None
        ):
            raise IndependentAuditError("quality gate ratio mismatch")
        if protocol.get("status") != {
            "failed": "data_quality_failed", "not_evaluable": "not_evaluable", "pass": "pass",
        }[expected_quality_status]:
            raise IndependentAuditError("quality gate result status mismatch")
        ca, sa = counts["current_lr"]["alerts"], counts["smart_nonzero"]["alerts"]
        cu, su = counts["current_lr"]["unknown_alerts"], counts["smart_nonzero"]["unknown_alerts"]
        for method in ("current_lr", "smart_nonzero"):
            n, u = counts[method]["alerts"], counts[method]["unknown_alerts"]
            if not _close(protocol.get("unknown_alert_ratio", {}).get(method), u / n if n else None):
                raise IndependentAuditError("unknown alert ratio mismatch")
        hit_difference = sum(method_event_hits["current_lr"].values()) - sum(method_event_hits["smart_nonzero"].values())
        expected_conditions = {
            "minimum_events": opportunity_total >= 100,
            "minimum_valid_replicates": bootstrap["valid_replicates"] >= 1900,
            "delta_at_least_5pp": None if not opportunity_total else 100 * hit_difference >= 5 * opportunity_total,
            "interval_lower_above_zero": None if bootstrap["interval"] is None else bootstrap["interval"][0] > 0,
            "unknown_increase_at_most_2pp": None if not ca or not sa else 100 * (cu * sa - su * ca) <= 2 * ca * sa,
            "quality_passed": expected_quality_status == "pass",
        }
        expected_gate = {
            "minimum_events": 100, "minimum_valid_replicates": 1900,
            "conditions": expected_conditions,
            "numerical_conditions_met": all(value is True for value in expected_conditions.values()),
            "unmet_conditions": [key for key, value in expected_conditions.items() if value is False],
            "unavailable_conditions": [key for key, value in expected_conditions.items() if value is None],
            "complete_acceptance": "pending", "support_gain": False,
            "reason": "complete execution and independent acceptance required before a gain claim",
        }
        if json.dumps(protocol.get("interpretation_gate"), sort_keys=True) != json.dumps(expected_gate, sort_keys=True):
            raise IndependentAuditError("interpretation gate mismatch")
        publication_blocks = []
        if opportunity_total < 100:
            publication_blocks.append("insufficient_events")
        if bootstrap["valid_replicates"] < 1900:
            publication_blocks.append("insufficient_valid_replicates")
        if expected_quality_status != "pass":
            publication_blocks.append("data_quality_not_passed")
        publication_blocks.append("complete_acceptance_pending")
        if protocol.get("interval_publication") != {
            "status": "withheld", "interval_pp": None,
            "computed_interval_scope": "internal_audit_only", "blocking_reasons": publication_blocks,
        }:
            raise IndependentAuditError("interval publication mismatch")
        result = {
            "status": "pass",
            "independent_source_rows": sum(len(rows) for rows in raw_by_serial.values()),
            "independent_feature_rows": len(expected_features),
            "independent_score_rows": {method: len(values) for method, values in method_inputs.items()},
            "independent_selection_rows": {method: len(values) for method, values in actual_selected_by_method.items()},
            "independent_counts": counts,
            "features_scores_selections_verified": True,
            "evaluation_matches": True,
            "event_rows": len(event_rows),
            "source_panel_verified": source_path is not None,
            "protocol_metrics_verified": True,
        }
    finally:
        selection_connection.close()
        panel_connection.close()
    if audit_path is not None:
        audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
