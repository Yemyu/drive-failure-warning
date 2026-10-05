"""R0 synthetic acceptance suite.

Runs the R0 必测矩阵 over a small synthetic daily-source input using the real
protocol engine functions, through real file reads and content-different
inputs.  No Q4 content is involved: all dates use the synthetic quarter
2023-10-01..12-31 exactly as the protocol fixes them.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pipeline.r_validation import (
    ProtocolError,
    alert_precision,
    apply_cooldown,
    capacity_for,
    eligible_dates,
    event_window,
    label_for,
    lead_statistics,
    opportunity_events,
    paired_bootstrap_difference,
)


class CaseFailure(AssertionError):
    pass


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _run_case(name: str, results: dict[str, Any]) -> None:
    print(f"[suite] {name}: ok")


def run_synthetic_suite(output: Path, *, root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {"status": "pass", "cases": {}}

    def record(name: str, payload: dict[str, Any]) -> None:
        results["cases"][name] = payload
        _run_case(name, payload)

    eval_start, eval_end = _date("2023-10-01"), _date("2023-12-24")
    event_start, event_end = event_window(eval_start, eval_end)
    if (event_start, event_end) != (_date("2023-10-08"), _date("2023-12-25")):
        raise CaseFailure("event window derivation differs from the protocol")

    # --- dates: 10-08 in, 10-07 out; 12-25 in, 12-26 out; 12-24 + 7 = 12-31
    record("dates", {
        "event_start": event_start.isoformat(),
        "event_end": event_end.isoformat(),
        "outcome_coverage": (event_end + dt.timedelta(days=6)).isoformat(),
        "first_score_plus_7": (eval_start + dt.timedelta(days=7)).isoformat(),
    })

    # --- eligibility: 14-day window with >= 12 observations; cross-quarter
    # bridge history (09-18..09-30) makes 10-01 eligible for a daily device.
    observed = {_date("2023-09-18") + dt.timedelta(days=i) for i in range(13)}
    observed |= {_date("2023-10-01") + dt.timedelta(days=i) for i in range(30)}
    days = eligible_dates(_date("2023-10-01"), _date("2023-10-10"), observed, "S1")
    if days[0] != _date("2023-10-01"):
        raise CaseFailure("bridge history did not make 10-01 eligible")
    sparse = {d for d in observed if d >= _date("2023-10-03")}
    if eligible_dates(_date("2023-10-01"), _date("2023-10-10"), sparse, "S1"):
        raise CaseFailure("a device without bridge history became eligible too early")
    absent_today = set(observed)
    absent_today.remove(_date("2023-10-01"))
    if eligible_dates(_date("2023-10-01"), _date("2023-10-01"), absent_today, "S1"):
        raise CaseFailure("a missing current-day row became eligible")
    record("eligibility", {"first_eligible": days[0].isoformat(), "eligible_days": len(days),
                            "missing_current_day_rejected": True})

    # --- cooldown: 10-01 alert -> 10-08 rejected (7 days), 10-09 allowed (8)
    kept, rejected = apply_cooldown(
        [(_date("2023-10-01"), "A"), (_date("2023-10-08"), "A"), (_date("2023-10-09"), "A")]
    )
    if [(d, s) for d, s in kept] != [(_date("2023-10-01"), "A"), (_date("2023-10-09"), "A")]:
        raise CaseFailure(f"cooldown behaviour differs: {kept}")
    record("cooldown", {"kept": [(d.isoformat(), s) for d, s in kept],
                        "rejected": rejected})

    # --- capacity: k = ceil(N/1000); N=0 -> 0
    cap = capacity_for({_date("2023-10-01"): 1500, _date("2023-10-02"): 0})
    if cap != {_date("2023-10-01"): 2, _date("2023-10-02"): 0}:
        raise CaseFailure("capacity rule differs")
    record("capacity", {d.isoformat(): v for d, v in cap.items()})

    # --- labels: positive_with_gap / negative_observed / unknown; first
    # failure on the decision date itself is not labelle.
    decision = _date("2023-10-10")
    observed_after = {decision + dt.timedelta(days=i) for i in range(1, 8)}
    if label_for(_date("2023-10-12"), decision, observed_after,
                 outcome_cutoff=_date("2023-12-31")) != "positive_with_gap":
        raise CaseFailure("in-window failure must be positive_with_gap")
    if label_for(_date("2023-10-17"), decision, observed_after,
                 outcome_cutoff=_date("2023-12-31")) != "positive_with_gap":
        raise CaseFailure("failure on the last horizon day must be positive_with_gap")
    if label_for(_date("2023-10-18"), decision, observed_after,
                 outcome_cutoff=_date("2023-12-31")) != "negative_observed":
        raise CaseFailure("failure on day 8 is outside this decision horizon")
    if label_for(None, decision, observed_after, outcome_cutoff=_date("2023-12-31")) != "negative_observed":
        raise CaseFailure("full follow-up without failure must be negative_observed")
    partial = {decision + dt.timedelta(days=i) for i in range(1, 5)}
    if label_for(None, decision, partial, outcome_cutoff=_date("2023-12-31")) != "unknown":
        raise CaseFailure("partial follow-up must be unknown")
    try:
        label_for(decision, decision, observed_after, outcome_cutoff=_date("2023-12-31"))
    except ProtocolError:
        pass
    else:
        raise CaseFailure("first failure on the decision date must not be labelle")
    record("labels", {"positive": "in-window failure", "negative": "full follow-up",
                      "unknown": "partial follow-up", "same_day_failure": "refused"})

    # --- opportunity: shared denominator independent of alerts
    first_failures = {"S1": _date("2023-10-08"), "S2": _date("2023-10-08"),
                      "S3": _date("2023-12-26"), "S4": _date("2023-10-05")}
    eligibility = {"S1": [eval_start], "S2": [eval_start], "S3": [eval_start],
                   "S4": [eval_start]}
    events = opportunity_events(first_failures, event_start=event_start,
                                event_end=event_end, eligibility=eligibility)
    if set(events) != {"S1", "S2"}:
        raise CaseFailure("opportunity set differs (S3 out of window, S4 before it)")
    record("opportunity", {"events": sorted(events),
                           "excluded": {"S3": "outside event window", "S4": "failure before it"}})

    # --- statistics: [1,3,7] with 4 opportunities
    stats = lead_statistics([1, 3, 7])
    if (stats["median"], stats["q25"], stats["q75"]) != (3.0, 2.0, 5.0):
        raise CaseFailure("lead quantiles differ")
    precision = alert_precision(alerts=10, known_hits=2, unknown_alerts=3)
    if (precision["lower"], precision["upper"], precision["known_outcome"]) != (0.2, 0.5, 2 / 7):
        raise CaseFailure("precision bounds differ")
    record("statistics", {"leads": stats, "precision": precision})

    # --- bootstrap: full eligible roster, paired direction, denominator
    # recomputed per replicate, and swapping methods flips the sign.
    events_list = ["A", "B", "C", "D"]
    current_hits = {"A": 1, "B": 1, "C": 0, "D": 0}
    smart_hits = {"A": 0, "B": 0, "C": 0, "D": 0}
    device_flags = {
        "A": {"opportunity": 1, "current_hit": 1, "smart_hit": 0},
        "B": {"opportunity": 1, "current_hit": 1, "smart_hit": 0},
        "C": {"opportunity": 1, "current_hit": 0, "smart_hit": 0},
        "D": {"opportunity": 1, "current_hit": 0, "smart_hit": 0},
        "E": {"opportunity": 0, "current_hit": 0, "smart_hit": 0},
        "F": {"opportunity": 0, "current_hit": 0, "smart_hit": 0},
    }
    forward = paired_bootstrap_difference(events_list, current_hits, smart_hits,
                                          device_flags=device_flags, minimum_events=0)
    backward_flags = {serial: {**item, "current_hit": item["smart_hit"],
                               "smart_hit": item["current_hit"]}
                      for serial, item in device_flags.items()}
    backward = paired_bootstrap_difference(events_list, smart_hits, current_hits,
                                           device_flags=backward_flags, minimum_events=0)
    if forward["point_estimate_pp"] != 50.0 or backward["point_estimate_pp"] != -50.0:
        raise CaseFailure("bootstrap point estimates differ from 2/4 vs 0/4")
    if forward["roster_mode"] != "all_eligible_devices" or forward["roster_devices"] != 6:
        raise CaseFailure("bootstrap did not use the complete eligible roster")
    if len(forward["differences"]) != forward["valid_replicates"]:
        raise CaseFailure("bootstrap difference vector was not retained")
    record("bootstrap", {"forward": forward, "backward": backward,
                          "non_event_devices_in_roster": 2})

    # --- insufficient events gate: the arithmetic fixture may exercise CI with
    # an explicit opt-out, while the protocol gate itself must suppress it.
    gated = paired_bootstrap_difference(events_list, current_hits, smart_hits,
                                        device_flags=device_flags)
    if gated.get("interval") is not None or not gated.get("insufficient_events"):
        raise CaseFailure("fewer than 100 opportunities must not publish a CI")
    record("insufficient_events", {
        "opportunity_events": len(events_list),
        "interval": gated.get("interval"),
        "minimum_events": gated.get("minimum_events"),
    })

    # --- perturbation: a later-recorded first-failure change alters the
    # forward view (earliest possible lead 5 -> 7) while decisions up to t
    # stay untouched.
    pert_eligibility = dict(eligibility)
    pert_eligibility["S2"] = [eval_start, _date("2023-10-05")]
    events1 = opportunity_events(first_failures, event_start=event_start,
                                 event_end=event_end, eligibility=pert_eligibility)
    pert_failure = dict(first_failures)
    pert_failure["S2"] = _date("2023-10-12")
    events2 = opportunity_events(pert_failure, event_start=event_start,
                                 event_end=event_end, eligibility=pert_eligibility)
    # original failure 10-08 can use the 10-01 decision (lead 7); after the
    # perturbation the 10-01 decision is 11 days before the failure and falls
    # out of the 1..7 candidate window, so the earliest eligible decision
    # moves to 10-05 (lead 7 from that day).
    if events1["S2"]["first_eligible_decision"] != "2023-10-01" or \
            events2["S2"]["first_eligible_decision"] != "2023-10-05":
        raise CaseFailure(
            f"perturbation not observed: {events1['S2']} -> {events2['S2']}"
        )
    record("perturbation", {"before": events1["S2"], "after": events2["S2"],
                            "note": "decisions on/before 10-05 unchanged; only the forward view moved"})

    results["status"] = "pass"
    (output / "synthetic_suite.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return results


def run_independent_audit(attempt: Path, *, root: Path) -> dict[str, Any]:
    """Reopen and independently recompute each recorded synthetic chain.

    Saved pass flags are never accepted as evidence of current file contents.
    This command is read-only; historical evidence is not rewritten.
    """
    from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError
    from pipeline.r_validation.history import file_sha
    import sqlite3

    root = Path(root).resolve()
    attempt = Path(attempt).resolve()
    if not attempt.is_relative_to(root):
        return {"status": "fail", "reason": "audit attempt must stay inside project"}
    checked = {}
    try:
        suite = json.loads((attempt / 'synthetic_suite.json').read_text())
        if suite['cases']['statistics']['leads']['median'] != 3.0:
            raise ValueError('independent median mismatch')
        names = ['cli_chain']
        if 'protocol_chain' in suite:
            names += ['protocol_chain_a/cli_chain', 'protocol_chain_b/cli_chain']
        for name in names:
            chain = attempt / name
            record = json.loads((chain / 'chain_results.json').read_text())
            if record.get('profile') != 'synthetic':
                raise ValueError('synthetic audit cannot certify a production profile')
            paths = {key: chain / filename for key, filename in {
                'source_sha256': 'daily.csv', 'panel_sha256': 'panel.sqlite',
                'selection_sha256': 'selection.sqlite', 'evaluation_sha256': 'evaluation.json',
            }.items()}
            for key, path in paths.items():
                if not path.resolve().is_relative_to(attempt):
                    raise ValueError('audit input escapes attempt')
                if file_sha(path) != record.get(key):
                    raise ValueError(f'{name}: {key} mismatch')
            model_path = chain / 'current_lr.json'
            if file_sha(model_path) != file_sha(root / 'examples/small_replay/current_lr.json'):
                raise ValueError('synthetic model differs from frozen parameters')
            evaluation = json.loads(paths['evaluation_sha256'].read_text())
            history = record.get('historical_inputs')
            bindings = None
            if history:
                bindings = {}
                for item in history['inputs']:
                    path = Path(item['path']).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError('historical input escapes project')
                    bindings[item['role']] = {'path': str(path), 'sha256': item['sha256']}
            # Recompute from actual source/panel/parameters, ignoring saved audit.json.
            checked[name] = audit_selection(
                paths['selection_sha256'], paths['panel_sha256'], evaluation, None,
                model=json.loads(model_path.read_text()), source_path=paths['source_sha256'],
                historical_inputs=bindings,
            )
            for key, path in paths.items():
                if file_sha(path) != record[key]:
                    raise ValueError(f'{name}: input changed during audit')
        return {'status': 'pass', 'mode': 'read_only_recomputed', 'chains': checked}
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, IndependentAuditError) as exc:
        return {'status': 'fail', 'reason': str(exc), 'chains_checked': list(checked)}

