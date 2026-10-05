"""R0 core: protocol engine for the synthetic acceptance pipeline.

Implements the date/eligibility/cooldown/label/opportunity/statistics rules of
``evidence/protocols/r_validation_v1.md`` §4—§7 as pure functions over small in-memory inputs,
so the R0 acceptance matrix can exercise the real rules without Q4 content.

The production entry points (``run``/``audit`` on the CLI) are gated behind a
release record bound to an approved protocol lock; see ``release.py``.
"""

from __future__ import annotations

import datetime as dt
import math
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

HORIZON_DAYS = 7
COOLDOWN_DAYS = 7
MIN_HISTORY_OBSERVATIONS = 12
HISTORY_WINDOW_DAYS = 14


class ProtocolError(RuntimeError):
    """A synthetic input violates the protocol."""


def gain_conditions(*, opportunities: int, current_hits: int, smart_hits: int,
                    current_alerts: int, current_unknown: int,
                    smart_alerts: int, smart_unknown: int,
                    valid_replicates: int, interval: Sequence[float] | None,
                    quality_status: str) -> dict[str, Any]:
    """Evaluate §7 numerical conditions; execution acceptance is external.

    Counts make the inclusive five- and two-percentage-point boundaries exact.
    This intermediate cannot certify its own execution or authorise a claim.
    """
    for numerator, denominator in ((current_hits, opportunities), (smart_hits, opportunities),
                                   (current_unknown, current_alerts), (smart_unknown, smart_alerts)):
        if type(numerator) is not int or type(denominator) is not int or not 0 <= numerator <= denominator:
            raise ProtocolError("invalid gain counts")
    if type(valid_replicates) is not int or not 0 <= valid_replicates <= 2000:
        raise ProtocolError("invalid bootstrap replicate count")
    if interval is not None and (len(interval) != 2 or
            any(not math.isfinite(value) for value in interval) or interval[0] > interval[1]):
        raise ProtocolError("invalid gain interval")
    conditions = {
        "minimum_events": opportunities >= 100,
        "minimum_valid_replicates": valid_replicates >= 1900,
        "delta_at_least_5pp": None if not opportunities else (current_hits - smart_hits) * 20 >= opportunities,
        "interval_lower_above_zero": None if interval is None else interval[0] > 0,
        "unknown_increase_at_most_2pp": None if not current_alerts or not smart_alerts else (
            Fraction(current_unknown, current_alerts) - Fraction(smart_unknown, smart_alerts) <= Fraction(1, 50)
        ),
        "quality_passed": quality_status == "pass",
    }
    return {
        "minimum_events": 100, "minimum_valid_replicates": 1900,
        "conditions": conditions,
        "numerical_conditions_met": all(value is True for value in conditions.values()),
        "unmet_conditions": [key for key, value in conditions.items() if value is False],
        "unavailable_conditions": [key for key, value in conditions.items() if value is None],
        "complete_acceptance": "pending", "support_gain": False,
        "reason": "complete execution and independent acceptance required before a gain claim",
    }


def interval_publication(bootstrap: Mapping[str, Any], quality_status: str) -> dict[str, Any]:
    """The evaluation intermediate has no authority to certify full execution."""
    reasons = []
    if bootstrap["opportunity_events"] < 100:
        reasons.append("insufficient_events")
    if bootstrap["valid_replicates"] < 1900:
        reasons.append("insufficient_valid_replicates")
    if quality_status != "pass":
        reasons.append("data_quality_not_passed")
    reasons.append("complete_acceptance_pending")
    return {"status": "withheld", "interval_pp": None,
            "computed_interval_scope": "internal_audit_only", "blocking_reasons": reasons}


def known_average_precision(pairs: Sequence[tuple[float, int | None]]) -> dict[str, Any]:
    """Full eligible score pool, unknown excluded, ties grouped as one threshold."""
    if any(not math.isfinite(score) or label not in (None, 0, 1) for score, label in pairs):
        raise ProtocolError("invalid AP input")
    known = sorted(((score, label) for score, label in pairs if label is not None), reverse=True)
    positives = sum(label for _, label in known)
    value = None
    if positives:
        seen = hits = position = 0
        value = 0.0
        while position < len(known):
            end = position + 1
            while end < len(known) and known[end][0] == known[position][0]:
                end += 1
            added = sum(label for _, label in known[position:end])
            seen += end - position
            hits += added
            value += added / positives * hits / seen
            position = end
    return {"scope": "all_eligible_scoring_rows_known_labels", "known_rows": len(known),
            "positive_rows": positives, "unknown_rows": len(pairs) - len(known), "value": value}


def lead_recall(leads: Sequence[int], opportunities: int) -> dict[str, Any]:
    """Recall with at least two/three days' notice, over all opportunities."""
    if type(opportunities) is not int or opportunities < len(leads) or opportunities < 0:
        raise ProtocolError("invalid lead opportunity denominator")
    if any(type(value) is not int or not 1 <= value <= HORIZON_DAYS for value in leads):
        raise ProtocolError("invalid event lead")
    counts = {f"ge{days}_hits": sum(value >= days for value in leads) for days in (2, 3)}
    return {"opportunity_events": opportunities, **counts,
            **{f"ge{days}_recall": counts[f"ge{days}_hits"] / opportunities if opportunities else None
               for days in (2, 3)}}


def parse_date(value: object) -> dt.date:
    if not isinstance(value, str):
        raise ProtocolError(f"date must be an ISO string, got {value!r}")
    parsed = dt.date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ProtocolError(f"date must be canonical YYYY-MM-DD, got {value!r}")
    return parsed


def event_window(eval_start: dt.date, eval_end: dt.date) -> tuple[dt.date, dt.date]:
    """event_start = eval_start + H; event_end = eval_end + 1 (protocol §4)."""
    if eval_start > eval_end:
        raise ProtocolError("eval_start after eval_end")
    return eval_start + dt.timedelta(days=HORIZON_DAYS), eval_end + dt.timedelta(days=1)


def eligible_dates(
    score_start: dt.date,
    score_end: dt.date,
    observed_days: set[dt.date],
    serial: str,
    *,
    first_failure: dt.date | None = None,
) -> list[dt.date]:
    """Days t in the scoring window where the device has a valid 14-day window.

    A day is eligible when the window [t-13, t] contains at least
    ``MIN_HISTORY_OBSERVATIONS`` observed days.  Missing days are never
    backfilled as observations.  A first failure on or before t removes the
    device from scoring eligibility from that day onward (protocol §4).
    """
    if score_start > score_end:
        raise ProtocolError("score_start after score_end")
    eligible: list[dt.date] = []
    for offset in range((score_end - score_start).days + 1):
        t = score_start + dt.timedelta(days=offset)
        if first_failure is not None and t >= first_failure:
            continue
        # A valid trailing window does not imply a current-day observation.
        # Missing ``t`` must remain missing and cannot become a score row.
        if t not in observed_days:
            continue
        window = {t - dt.timedelta(days=back) for back in range(HISTORY_WINDOW_DAYS)}
        if len(window & observed_days) < MIN_HISTORY_OBSERVATIONS:
            continue
        eligible.append(t)
    return eligible


def capacity_for(day_counts: Mapping[dt.date, int]) -> dict[dt.date, int]:
    """k_t = ceil(N_t / 1000); N_t = 0 -> k_t = 0 (protocol §5)."""
    return {day: math.ceil(count / 1000) if count else 0 for day, count in day_counts.items()}


def apply_cooldown(
    candidates: Iterable[tuple[dt.date, str]],
    *,
    cooldown_days: int = COOLDOWN_DAYS,
) -> list[tuple[dt.date, str]]:
    """Greedy cooldown with the tie prefix already applied by the caller.

    ``candidates`` is an ordered (by score, tie prefix, serial) stream of
    daily selections.  A device is rejected when the date difference to its
    previous alert is <= ``cooldown_days`` (protocol §5: 10-01 alert -> the
    next possible alert is 10-09, a difference of 8 days).
    """
    last_alert: dict[str, dt.date] = {}
    kept: list[tuple[dt.date, str]] = []
    rejected: list[dict[str, object]] = []
    for day, serial in candidates:
        previous = last_alert.get(serial)
        if previous is not None and (day - previous).days <= cooldown_days:
            rejected.append({"date": day.isoformat(), "serial": serial,
                             "days_since": (day - previous).days})
            continue
        last_alert[serial] = day
        kept.append((day, serial))
    return kept, rejected


def label_for(
    first_failure: dt.date | None,
    decision_date: dt.date,
    observed_after: set[dt.date],
    *,
    outcome_cutoff: dt.date,
) -> str:
    """Label a decision: positive_with_gap / negative_observed / unknown.

    A failure on the decision date itself is not scored (the device left the
    eligible pool).  ``observed_after`` holds the days in t+1..t+7 that the
    device was actually observed in the source (presence, not failure).
    """
    if first_failure is not None and first_failure <= decision_date:
        raise ProtocolError("decision date on/after first failure must not be labelled")
    horizon = [decision_date + dt.timedelta(days=d) for d in range(1, HORIZON_DAYS + 1)]
    if first_failure is not None and decision_date < first_failure <= horizon[-1]:
        if first_failure <= outcome_cutoff:
            return "positive_with_gap"
        return "unknown"
    observed_horizon = set(horizon) & observed_after
    if horizon[-1] <= outcome_cutoff and len(observed_horizon) == HORIZON_DAYS:
        return "negative_observed"
    return "unknown"


def opportunity_events(
    first_failures: Mapping[str, dt.date],
    *,
    event_start: dt.date,
    event_end: dt.date,
    eligibility: Mapping[str, Sequence[dt.date]],
) -> dict[str, dict[str, Any]]:
    """Opportunity events: first failure inside the main event window with at
    least one eligible decision day in the preceding 1..H days.

    The denominator depends only on shared scoring eligibility — never on any
    method having alerted, being in cooldown, its rank, or signal positivity.
    """
    events: dict[str, dict[str, Any]] = {}
    for serial, failure in sorted(first_failures.items(), key=lambda p: p[1].isoformat()):
        if not (event_start <= failure <= event_end):
            continue
        days = eligibility.get(serial, ())
        candidates = [
            day
            for day in days
            if day < failure and 1 <= (failure - day).days <= HORIZON_DAYS
        ]
        if not candidates:
            continue
        events[serial] = {
            "first_failure_date": failure.isoformat(),
            "first_eligible_decision": min(candidates).isoformat(),
            "earliest_possible_lead": (failure - min(candidates)).days,
        }
    return events


def linear_quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def lead_statistics(leads: Sequence[int]) -> dict[str, Any]:
    """Lead-time quantiles over captured events only (protocol §7)."""
    leads = [int(v) for v in leads]
    return {
        "count": len(leads),
        "median": linear_quantile(leads, 0.5),
        "q25": linear_quantile(leads, 0.25),
        "q75": linear_quantile(leads, 0.75),
    }


def alert_precision(
    alerts: int, known_hits: int, unknown_alerts: int
) -> dict[str, float | None]:
    """Precision lower/upper bounds keep unknown in the denominator; the
    known-outcome precision is reported separately.  A=0 or A-U=0 are null."""
    if alerts < 0 or known_hits < 0 or unknown_alerts < 0:
        raise ProtocolError("negative alert counts")
    if known_hits + unknown_alerts > alerts:
        raise ProtocolError("H + U exceeds A")
    lower = known_hits / alerts if alerts else None
    upper = (known_hits + unknown_alerts) / alerts if alerts else None
    known_denominator = alerts - unknown_alerts
    known_precision = (
        known_hits / known_denominator if known_denominator > 0 else None
    )
    return {"lower": lower, "upper": upper, "known_outcome": known_precision}


def paired_bootstrap_difference(
    events: Sequence[str],
    current_hits: Mapping[str, int],
    smart_hits: Mapping[str, int],
    *,
    device_flags: Mapping[str, Mapping[str, int | bool]] | None = None,
    seed: int = 20260913,
    replicates: int = 2000,
    min_valid: int = 1900,
    minimum_events: int = 100,
) -> dict[str, Any]:
    """Paired device-level bootstrap of Δ = recall(current) − recall(smart).

    The protocol form passes ``device_flags`` containing every device with at
    least one eligible scoring day.  Each item has binary ``opportunity``,
    ``current_hit`` and ``smart_hit`` flags.  The same device indices are
    sampled for both methods, and every replicate recomputes its opportunity
    denominator.  Omitting ``device_flags`` preserves the old event-only
    arithmetic fixture for compatibility, but is marked as legacy and is not
    the R1 production path.
    """
    if replicates <= 0 or min_valid <= 0 or minimum_events < 0:
        raise ProtocolError("bootstrap counts must be positive")
    if min_valid > replicates:
        raise ProtocolError("min_valid cannot exceed replicates")
    if device_flags is None:
        ordered = sorted((str(serial) for serial in events), key=lambda s: s.encode("utf-8"))
        roster_mode = "events_only_legacy"
        opportunity = [1] * len(ordered)
        current = [int(current_hits.get(serial, 0)) for serial in ordered]
        smart = [int(smart_hits.get(serial, 0)) for serial in ordered]
    else:
        if not isinstance(device_flags, Mapping):
            raise ProtocolError("device_flags must contain the eligible device roster")
        ordered = sorted((str(serial) for serial in device_flags), key=lambda s: s.encode("utf-8"))
        roster_mode = "all_eligible_devices"
        opportunity, current, smart = [], [], []
        for serial in ordered:
            item = device_flags[serial]
            if not isinstance(item, Mapping):
                raise ProtocolError(f"device flags are not an object: {serial}")
            values = {
                "opportunity": int(item.get("opportunity", 0)),
                "current_hit": int(item.get("current_hit", current_hits.get(serial, 0))),
                "smart_hit": int(item.get("smart_hit", smart_hits.get(serial, 0))),
            }
            if any(value not in (0, 1) for value in values.values()):
                raise ProtocolError(f"device flags must be binary: {serial}")
            if values["opportunity"] == 0 and (values["current_hit"] or values["smart_hit"]):
                raise ProtocolError(f"a non-opportunity device cannot be a hit: {serial}")
            opportunity.append(values["opportunity"])
            current.append(values["current_hit"])
            smart.append(values["smart_hit"])
    if not ordered:
        return {"valid_replicates": 0, "empty_replicates": replicates,
                "differences": [], "interval": None, "seed": seed,
                "replicates_requested": replicates, "roster_mode": roster_mode,
                "direction": "current_minus_smart", "roster_devices": 0,
                "opportunity_events": 0, "point_estimate_pp": None,
                "insufficient_events": True, "minimum_events": minimum_events}
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(seed))
    total = len(ordered)
    opportunity_total = sum(opportunity)
    current_total = sum(flag * hit for flag, hit in zip(opportunity, current))
    smart_total = sum(flag * hit for flag, hit in zip(opportunity, smart))
    differences: list[float] = []
    empty = 0
    for _ in range(replicates):
        index = generator.integers(0, total, size=total)
        denominator = sum(opportunity[i] for i in index)
        if denominator == 0:
            empty += 1
            continue
        current_sum = sum(opportunity[i] * current[i] for i in index)
        smart_sum = sum(opportunity[i] * smart[i] for i in index)
        differences.append((current_sum - smart_sum) / denominator * 100.0)
    valid = len(differences)
    result: dict[str, Any] = {
        "valid_replicates": valid,
        "empty_replicates": replicates - valid,
        "seed": seed,
        "replicates_requested": replicates,
        "direction": "current_minus_smart",
        "roster_mode": roster_mode,
        "roster_devices": total,
        "opportunity_events": opportunity_total,
        "point_estimate_pp": ((current_total - smart_total) / opportunity_total * 100.0
                               if opportunity_total else None),
        "differences": list(differences),
    }
    if opportunity_total < minimum_events:
        result["interval"] = None
        result["insufficient_events"] = True
        result["minimum_events"] = minimum_events
        return result
    if valid < min_valid:
        result["interval"] = None
        result["insufficient_valid_replicates"] = True
        return result
    ordered_differences = sorted(differences)
    lower = linear_quantile(ordered_differences, 0.025)
    upper = linear_quantile(ordered_differences, 0.975)
    result["interval"] = [lower, upper]
    result["interval_pp"] = [lower, upper]
    return result
