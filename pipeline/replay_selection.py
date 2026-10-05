"""Deterministic ranking, budget, and cooldown helpers for replays.

The functions in this module do not access SQLite or labels.  They accept a
list of feature-like mappings and return a ranked candidate list plus the
selected alert rows, so the boundary can be checked with small hand-built
fixtures before a real replay is run.  ``select_alerts`` updates the supplied
cooldown mapping for selected rows; ranking and summary helpers remain pure.
"""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Mapping, MutableMapping, Sequence


METHODS = ("random", "smart_nonzero", "smart187")


def ceil_budget(n: int, denominator: int = 1000) -> int:
    if n < 0 or denominator <= 0:
        raise ValueError("n must be non-negative and denominator must be positive")
    return (n + denominator - 1) // denominator if n else 0


def rank_candidates(rows: Sequence[Mapping[str, object]], method: str) -> list[Mapping[str, object]]:
    """Rank only the candidates that can produce an alert for ``method``."""
    if method == "random":
        candidates = list(rows)
        return sorted(candidates, key=lambda row: (row["tie_break_sha256"], row["serial_number"]))
    if method == "smart_nonzero":
        candidates = [row for row in rows if int(row["smart_nonzero_signal_count"]) > 0]
        return sorted(
            candidates,
            key=lambda row: (
                -int(row["smart_nonzero_signal_count"]),
                row["tie_break_sha256"],
                row["serial_number"],
            ),
        )
    if method == "smart187":
        candidates = [row for row in rows if int(row["smart_187_signal"]) > 0]
        return sorted(candidates, key=lambda row: (row["tie_break_sha256"], row["serial_number"]))
    if method == "current_lr":
        return sorted(
            rows,
            key=lambda row: (-float(row["score"]), row["tie_break_sha256"], row["serial_number"]),
        )
    raise ValueError(f"unknown replay method: {method}")


def select_alerts(
    rows: Sequence[Mapping[str, object]],
    method: str,
    decision_date: dt.date,
    last_alert_date: MutableMapping[str, dt.date],
    *,
    denominator: int = 1000,
    cooldown_days: int = 7,
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    """Select at most K candidates while applying a device cooldown.

    ``cooldown_excluded`` counts every method candidate that is in cooldown,
    including candidates below the Kth selected row.  That makes the metric a
    diagnostic of the whole signal pool rather than an artefact of the budget
    cutoff.  The mapping is updated only for selected rows.
    """
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    ranked = rank_candidates(rows, method)
    budget_k = ceil_budget(len(rows), denominator)
    cooldown_excluded = 0
    eligible_ranked: list[Mapping[str, object]] = []
    for row in ranked:
        serial = str(row["serial_number"])
        previous = last_alert_date.get(serial)
        if previous is not None and (decision_date - previous).days <= cooldown_days:
            cooldown_excluded += 1
            continue
        eligible_ranked.append(row)
    selected = eligible_ranked[:budget_k]
    for row in selected:
        last_alert_date[str(row["serial_number"])] = decision_date
    return selected, {
        "eligible_count": len(rows),
        "budget_k": budget_k,
        "signal_count": len(ranked),
        "cooldown_excluded": cooldown_excluded,
        "alerts_count": len(selected),
    }


def median(values: Sequence[float | int]) -> float | int | None:
    """Return the ordinary median, averaging the two middle values when even."""
    if not values:
        return None
    return statistics.median(values)


def quantile(values: Sequence[float | int], probability: float) -> float | int | None:
    """Linear-interpolation quantile using position ``(n - 1) * p``."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return value if isinstance(value, float) else int(value)


def minimum_gap(dates: Sequence[dt.date]) -> int | None:
    if len(dates) < 2:
        return None
    ordered = sorted(dates)
    return min((right - left).days for left, right in zip(ordered, ordered[1:]))
