"""R validation package (protocol R1.0, R0 scope)."""

from .binding import BindingError, approve_parameters, verify_approved_package
from .engine import (
    ProtocolError,
    alert_precision,
    apply_cooldown,
    capacity_for,
    eligible_dates,
    event_window,
    label_for,
    lead_statistics,
    linear_quantile,
    opportunity_events,
    paired_bootstrap_difference,
)
from .release import ReleaseError, build_release, require_release

__all__ = [
    "BindingError",
    "ProtocolError",
    "ReleaseError",
    "approve_parameters",
    "verify_approved_package",
    "alert_precision",
    "apply_cooldown",
    "capacity_for",
    "eligible_dates",
    "event_window",
    "label_for",
    "lead_statistics",
    "linear_quantile",
    "opportunity_events",
    "paired_bootstrap_difference",
    "build_release",
    "require_release",
]
