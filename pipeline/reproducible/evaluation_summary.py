"""Independent aggregation of closed alert selections.

This module consumes only the alert rows and the event ledger produced by the
evaluation stage.  It does not read labels, scores, or source data and never
changes the selected alert list.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any


HORIZON_DAYS = 7


class EvaluationSummaryError(RuntimeError):
    """The closed evaluation rows cannot support a deterministic summary."""


def _fail(message: str) -> None:
    raise EvaluationSummaryError(message)


def _date(value: object, label: str) -> dt.date:
    if type(value) is not str:
        _fail(f"{label} must be an ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise EvaluationSummaryError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        _fail(f"{label} must use canonical YYYY-MM-DD")
    return parsed


def _range(value: dt.date | str, label: str) -> dt.date:
    return value if isinstance(value, dt.date) else _date(value, label)


def _quantile(values: Sequence[int], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _strict_lead(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= HORIZON_DAYS:
        _fail(f"{label} must be an integer from 1 to {HORIZON_DAYS}")
    return value


def summarize_lead_metrics(
    alerts: Sequence[Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
    *,
    event_start: dt.date | str,
    event_end: dt.date | str,
) -> dict[str, Any]:
    """Return the locked lead-time and outside-event metrics.

    ``events`` is the per-device opportunity ledger.  Its ``earliest_lead``
    value is the largest lead (the earliest alert) among valid alerts for that
    event.  The denominator for the ``>=2`` and ``>=3`` measures is every
    event with an opportunity, including missed events.
    """
    start = _range(event_start, "event_start")
    end = _range(event_end, "event_end")
    if start > end:
        _fail("event_start must be on or before event_end")
    if not isinstance(events, Mapping):
        _fail("event ledger must be an object")
    if not isinstance(alerts, Sequence) or isinstance(alerts, (str, bytes)):
        _fail("alerts must be a sequence")
    opportunity_count = 0
    lead_days: list[int] = []
    for event_key, raw_event in events.items():
        if not isinstance(event_key, str) or not event_key:
            _fail("event key is invalid")
        if not isinstance(raw_event, Mapping):
            _fail(f"event {event_key} is not an object")
        opportunity = raw_event.get("opportunity")
        hit = raw_event.get("hit")
        if type(opportunity) is not int or opportunity not in (0, 1) or type(hit) is not int or hit not in (0, 1):
            _fail(f"event {event_key} opportunity/hit flags are invalid")
        if opportunity:
            opportunity_count += 1
        raw_lead = raw_event.get("earliest_lead_days")
        if hit:
            if not opportunity:
                _fail(f"event {event_key} is marked hit without an opportunity")
            lead = _strict_lead(raw_lead, f"event {event_key} earliest_lead_days")
            earliest_date = raw_event.get("earliest_alert_date")
            failure_date = raw_event.get("first_failure_date")
            if earliest_date is None or failure_date is None:
                _fail(f"event {event_key} is missing its earliest alert or failure date")
            earliest = _date(earliest_date, f"event {event_key} earliest_alert_date")
            failure = _date(failure_date, f"event {event_key} first_failure_date")
            if (failure - earliest).days != lead:
                _fail(f"event {event_key} earliest lead does not match its dates")
            opportunity_dates = raw_event.get("opportunity_dates")
            if opportunity_dates is not None:
                if not isinstance(opportunity_dates, list) or earliest_date not in opportunity_dates:
                    _fail(f"event {event_key} earliest alert is outside its opportunity dates")
            lead_days.append(lead)
        elif raw_lead is not None:
            _fail(f"event {event_key} has a lead without a hit")

    outside: list[dict[str, Any]] = []
    seen_alert_keys: set[tuple[str, str, str]] = set()
    for index, raw_alert in enumerate(alerts):
        if not isinstance(raw_alert, Mapping):
            _fail(f"alert {index} is not an object")
        decision_text = raw_alert.get("decision_date")
        serial = raw_alert.get("serial_number")
        failure_text = raw_alert.get("first_failure_date")
        if type(serial) is not str or not serial:
            _fail(f"alert {index} serial_number is invalid")
        method = raw_alert.get("method")
        if type(method) is not str or not method:
            _fail(f"alert {index} method is invalid")
        decision = _date(decision_text, f"alert {index} decision_date")
        key = (method, decision_text, serial)
        if key in seen_alert_keys:
            _fail(f"duplicate alert key in closed selection: {key}")
        seen_alert_keys.add(key)
        if failure_text is None:
            continue
        failure = _date(failure_text, f"alert {index} first_failure_date")
        lead = (failure - decision).days
        if 1 <= lead <= HORIZON_DAYS and not start <= failure <= end:
            outside.append({
                "method": method,
                "decision_date": decision_text,
                "serial_number": serial,
                "first_failure_date": failure_text,
                "lead_days": lead,
            })

    outside.sort(key=lambda item: (item["method"], item["decision_date"], item["serial_number"]))
    captured = len(lead_days)
    at_least_two = sum(value >= 2 for value in lead_days)
    at_least_three = sum(value >= 3 for value in lead_days)
    return {
        "lead_days_median": _quantile(lead_days, 0.5),
        "lead_days_q25": _quantile(lead_days, 0.25),
        "lead_days_q75": _quantile(lead_days, 0.75),
        "events_lead_ge_2": at_least_two,
        "events_lead_ge_3": at_least_three,
        "opportunity_recall_lead_ge_2": at_least_two / opportunity_count if opportunity_count else None,
        "opportunity_recall_lead_ge_3": at_least_three / opportunity_count if opportunity_count else None,
        "outside_main_event_alerts": outside,
        "outside_main_event_alert_count": len(outside),
        "lead_days_captured_events": captured,
        "lead_days_values": lead_days,
    }


__all__ = ["EvaluationSummaryError", "summarize_lead_metrics"]
