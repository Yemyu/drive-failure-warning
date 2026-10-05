"""Leakage-aware first-event labels for device-day records."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def classify_device_rows(
    rows: Iterable[Mapping],
    *,
    start: str,
    end: str,
    dataset_end: str,
    horizon_days: int = 7,
    history_days: int = 14,
    min_history: int = 12,
) -> list[dict]:
    """Classify every pre-first-event decision row in a date range.

    A row is eligible only when its past calendar window contains at least
    ``min_history`` observed records. A positive label needs only an observed
    first failure inside the future window. A negative label requires all future
    calendar days to be observed and failure-free. All other outcomes stay
    unknown, preserving right censoring and exit gaps.
    """

    lo = _date(start)
    hi = _date(end)
    source_end = _date(dataset_end)
    if lo > hi:
        raise ValueError("start must be on or before end")
    if hi > source_end:
        raise ValueError("end must not be after dataset_end")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    if not 1 <= min_history <= history_days:
        raise ValueError("min_history must be between 1 and history_days")

    # The source cutoff is part of the label contract.  A caller may have a
    # larger panel available, but rows after dataset_end are never allowed to
    # decide a first failure, an observed future day, or a post-failure row.
    materialized = []
    for row in rows:
        item = dict(row)
        row_date = _date(item["date"])
        if row_date > source_end:
            continue
        failure = int(item["failure"])
        if failure not in (0, 1):
            raise ValueError(f"failure must be 0 or 1: {item['failure']!r}")
        materialized.append(item)
    materialized.sort(key=lambda row: row["date"])
    by_date = {row["date"]: row for row in materialized}
    if len(by_date) != len(materialized):
        raise ValueError("duplicate serial/date rows")
    first_failure = next((row for row in materialized if int(row["failure"]) == 1), None)
    first_failure_date = first_failure["date"] if first_failure else None
    output = []
    for row in materialized:
        decision = _date(row["date"])
        if decision < lo or decision > hi:
            continue
        if first_failure_date and row["date"] > first_failure_date:
            status = "post_failure"
            output.append(_flow(row, first_failure_date, status, None, 0, 0))
            continue
        if int(row["failure"]) == 1:
            output.append(_flow(row, first_failure_date, "same_day_failure", None, 0, 0))
            continue
        history_dates = [
            decision - dt.timedelta(days=offset)
            for offset in range(history_days)
            if (decision - dt.timedelta(days=offset)).isoformat() in by_date
        ]
        history_count = len(history_dates)
        future_dates = [decision + dt.timedelta(days=offset) for offset in range(1, horizon_days + 1)]
        observed_future = [future.isoformat() in by_date for future in future_dates]
        future_count = sum(observed_future)
        if history_count < min_history:
            output.append(_flow(row, first_failure_date, "history_insufficient", None, history_count, future_count))
            continue
        failure_in_window = first_failure_date and decision < _date(first_failure_date) <= decision + dt.timedelta(days=horizon_days)
        if failure_in_window:
            status = "positive_observed" if all(observed_future[: max(0, (_date(first_failure_date) - decision).days - 1)]) else "positive_with_gap"
            output.append(_flow(row, first_failure_date, status, 1, history_count, future_count))
        elif all(observed_future):
            output.append(_flow(row, first_failure_date, "negative_observed", 0, history_count, future_count))
        elif decision + dt.timedelta(days=horizon_days) > source_end:
            output.append(_flow(row, first_failure_date, "end_censored", None, history_count, future_count))
        else:
            output.append(_flow(row, first_failure_date, "gap_or_exit_censored", None, history_count, future_count))
    return output


def _flow(row: Mapping, first_failure_date: str | None, status: str, label: int | None, history: int, future: int) -> dict:
    return {
        "decision_date": row["date"],
        "serial_number": row["serial_number"],
        "model": row["model"],
        "capacity_bytes": row["capacity_bytes"],
        "first_failure_date": first_failure_date,
        "label": label,
        "status": status,
        "eligible": int(status in {"positive_observed", "positive_with_gap", "negative_observed", "end_censored", "gap_or_exit_censored"}),
        "history_observations": history,
        "future_observations": future,
    }
