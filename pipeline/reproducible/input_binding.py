"""Immutable semantic bindings for isolated pipeline stages.

The process access contract says which *names* a worker may receive.  This
module adds the second half of that boundary: a signed-by-content record of
the exact files, panel roles, profile, and locked configuration that those
names referred to when the parent launched the stage.
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
from pathlib import Path
import sqlite3
from typing import Any

from .artifacts import ArtifactError, artifact_facts, bound_path, write_json_exclusive
from .contract import (
    ContractError,
    LOCKED_PROFILE,
    config_facts,
    load_json,
    require_panel_contract,
    validate_locked_config,
)
from .process_contract import ROLE_INPUTS, ROLE_REQUIRED


BINDING_VERSION = "input-binding-v1"
FIXTURE_PROFILE = "synthetic_fixture_v1"
REAL_CONFIG_SHA256 = "275ce0d764d06a56c07dd95069404d85824fb3f0ad99afb1f54b8b06e9f20e20"
REAL_PANEL_FACTS: dict[str, tuple[int, str]] = {
    # Q1/Q2 is used as the training panel and as history during Q3 scoring.
    "train_panel": (
        860_471_296,
        "62b7a5bf32f4528d3c53ac110cf37dddb5170539e7b51816e83b4e5aa1e885a2",
    ),
    "history_panel": (
        860_471_296,
        "62b7a5bf32f4528d3c53ac110cf37dddb5170539e7b51816e83b4e5aa1e885a2",
    ),
    "eval_panel": (
        446_214_144,
        "c9e35f8685e384cba3e45d556a621283128b8554b7df84db3a5c79b73a417b4f",
    ),
}

_TOP_KEYS = {"version", "role", "profile", "config", "inputs", "panels"}
_FACT_KEYS = {"path", "bytes", "sha256"}
_REAL_PANEL_CALENDARS = {
    "train_panel": (dt.date(2023, 1, 1), dt.date(2023, 6, 30), 181),
    "history_panel": (dt.date(2023, 1, 1), dt.date(2023, 6, 30), 181),
    "eval_panel": (dt.date(2023, 7, 1), dt.date(2023, 9, 30), 92),
}


class BindingError(RuntimeError):
    """A semantic input binding is malformed, stale, or not permitted."""


def _as_path(value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise BindingError(f"{label} must be a path")
    try:
        return bound_path(value, label, must_exist=True)
    except ArtifactError as exc:
        raise BindingError(str(exc)) from exc


def _fact(path: Path | str, label: str) -> dict[str, object]:
    try:
        return dict(artifact_facts(path))
    except ArtifactError as exc:
        raise BindingError(str(exc)) from exc


def _strict_fact(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FACT_KEYS:
        raise BindingError(f"{label} facts are invalid")
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if type(path) is not str or not path:
        raise BindingError(f"{label}.path must be a non-empty string")
    if type(size) is not int or size < 0:
        raise BindingError(f"{label}.bytes must be a non-negative integer")
    if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(c not in "0123456789abcdef" for c in digest):
        raise BindingError(f"{label}.sha256 must be a lowercase SHA256 string")
    return {"path": path, "bytes": size, "sha256": digest}


def _canonical_path(value: object, label: str) -> str:
    return str(_as_path(value, label))


def _panel_bindings(bindings: Mapping[str, Path | str]) -> dict[str, Path | str]:
    result = {str(name): value for name, value in bindings.items() if "panel" in str(name)}
    if not result:
        raise BindingError("semantic bindings must include at least one panel")
    return result


def _profile_and_contracts(bindings: Mapping[str, Path | str]) -> tuple[str, dict[str, dict[str, Any]]]:
    panels = _panel_bindings(bindings)
    contracts: dict[str, dict[str, Any]] = {}
    modes: set[str] = set()
    for name, value in sorted(panels.items()):
        try:
            contract = require_panel_contract(value, role=name)
        except (ArtifactError, ContractError) as exc:
            raise BindingError(str(exc)) from exc
        if contract.get("mode") == "real_panel":
            contract = {**contract, "identity": _verify_real_panel_contents(_as_path(value, name), name)}
        contracts[name] = contract
        modes.add(str(contract.get("mode")))
    if modes == {"real_panel"}:
        return LOCKED_PROFILE, contracts
    if modes == {"synthetic_fixture"}:
        return FIXTURE_PROFILE, contracts
    raise BindingError("semantic bindings cannot mix real panels and synthetic fixtures")


def _verify_real_panel_contents(path: Path, role: str) -> dict[str, object]:
    """Check the retained panel's calendar, source counts, and model identity."""
    calendar = _REAL_PANEL_CALENDARS.get(role)
    if calendar is None:
        return {"checked": False, "reason": "no retained calendar for role"}
    start, end, expected_days = calendar
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise BindingError(f"{role} panel has a non-empty SQLite sidecar: {sidecar.name}")
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise BindingError(f"cannot open {role} panel for identity checks") from exc
    try:
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required_tables = {"daily", "member_counts", "serial_model_registry", "metadata"}
        if not required_tables.issubset(tables):
            raise BindingError(f"{role} panel is missing required identity tables")
        required_daily = {"date", "serial_number", "model", "failure", "smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"}
        required_counts = {"date", "source_member", "source_sha256", "source_rows", "selected_rows", "failure_rows", "schema_columns", "schema_sha256"}
        daily_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(daily)")}
        count_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(member_counts)")}
        if not required_daily.issubset(daily_columns) or not required_counts.issubset(count_columns):
            raise BindingError(f"{role} panel is missing required identity columns")
        expected_dates = {(start + dt.timedelta(days=offset)).isoformat() for offset in range(expected_days)}
        count_dates = {str(row[0]) for row in connection.execute("SELECT date FROM member_counts ORDER BY date")}
        daily_dates = {str(row[0]) for row in connection.execute("SELECT DISTINCT date FROM daily ORDER BY date")}
        if count_dates != expected_dates:
            raise BindingError(f"{role} panel source date coverage is incomplete")
        if not daily_dates.issubset(expected_dates):
            raise BindingError(f"{role} panel has daily rows outside its source calendar")
        target_rows = int(connection.execute("SELECT COUNT(*) FROM daily WHERE model=?", ("ST4000DM000",)).fetchone()[0])
        if target_rows <= 0:
            raise BindingError(f"{role} panel contains no ST4000DM000 rows")
        duplicate = connection.execute("SELECT 1 FROM daily GROUP BY date,serial_number HAVING COUNT(*)>1 LIMIT 1").fetchone()
        if duplicate is not None:
            raise BindingError(f"{role} panel contains duplicate date/serial keys")
        mismatch = connection.execute(
            "SELECT m.date,m.selected_rows,COUNT(d.serial_number),m.failure_rows,COALESCE(SUM(d.failure),0) "
            "FROM member_counts m LEFT JOIN daily d ON d.date=m.date "
            "GROUP BY m.date,m.selected_rows,m.failure_rows "
            "HAVING m.selected_rows<>COUNT(d.serial_number) OR m.failure_rows<>COALESCE(SUM(d.failure),0) LIMIT 1"
        ).fetchone()
        if mismatch is not None:
            raise BindingError(f"{role} panel daily/member_counts counts differ on {mismatch[0]}")
        bad_count = connection.execute(
            "SELECT 1 FROM member_counts WHERE source_rows<0 OR selected_rows<0 OR failure_rows<0 "
            "OR source_member='' OR source_sha256='' OR length(source_sha256)<>64 LIMIT 1"
        ).fetchone()
        if bad_count is not None:
            raise BindingError(f"{role} panel contains invalid source count facts")
        registry_bad = connection.execute(
            "SELECT 1 FROM serial_model_registry WHERE model=? AND source_scope<>? LIMIT 1",
            ("ST4000DM000", "full_source_member"),
        ).fetchone()
        if registry_bad is not None:
            raise BindingError(f"{role} panel target identity registry is not full-source")
        return {
            "checked": True,
            "source_days": len(count_dates),
            "daily_dates": len(daily_dates),
            "daily_rows": int(connection.execute("SELECT COUNT(*) FROM daily").fetchone()[0]),
        }
    except sqlite3.Error as exc:
        raise BindingError(f"cannot inspect {role} panel identity tables") from exc
    finally:
        connection.close()


def _check_known_facts(
    profile: str,
    config_path: Path,
    bindings: Mapping[str, Path | str],
    contracts: Mapping[str, Mapping[str, str]],
) -> None:
    if profile == LOCKED_PROFILE:
        try:
            payload = load_json(config_path, "locked configuration")
            validate_locked_config(payload)
        except ContractError as exc:
            raise BindingError(str(exc)) from exc
        config_digest = config_facts(config_path)["sha256"]
        if config_digest != REAL_CONFIG_SHA256:
            raise BindingError("real profile configuration SHA256 is not the locked file")
        for name, contract in contracts.items():
            if str(contract.get("mode")) != "real_panel":
                raise BindingError(f"{name} is not a real panel")
            expected = REAL_PANEL_FACTS.get(name)
            if expected is None:
                continue
            facts = _fact(bindings[name], name)
            if (facts["bytes"], facts["sha256"]) != expected:
                raise BindingError(f"{name} does not match the retained real panel identity")
    elif profile == FIXTURE_PROFILE:
        if any(str(contract.get("mode")) != "synthetic_fixture" for contract in contracts.values()):
            raise BindingError("fixture profile contains a non-fixture panel")
    else:
        raise BindingError(f"unknown input binding profile: {profile}")


def _validate_semantic_names(role: str, bindings: Mapping[str, Path | str]) -> None:
    if role not in ROLE_INPUTS:
        raise BindingError(f"unknown process role: {role}")
    names = set(bindings)
    unknown = sorted(names - ROLE_INPUTS[role])
    if unknown:
        raise BindingError(f"{role} binding contains disallowed names: {', '.join(unknown)}")
    missing = sorted(ROLE_REQUIRED[role] - names)
    if missing:
        raise BindingError(f"{role} binding is missing: {', '.join(missing)}")
    if "config" not in names:
        raise BindingError(f"{role} binding is missing config")
    for name, value in bindings.items():
        if type(name) is not str or not name:
            raise BindingError("semantic binding names must be non-empty strings")
        if not isinstance(value, (str, Path)):
            raise BindingError(f"{name} binding must be a path")


def _build_payload(role: str, config_path: Path, bindings: Mapping[str, Path | str]) -> dict[str, object]:
    _validate_semantic_names(role, bindings)
    profile, contracts = _profile_and_contracts(bindings)
    _check_known_facts(profile, config_path, bindings, contracts)
    inputs = {
        name: _fact(value, f"{role}.{name}")
        for name, value in sorted(bindings.items())
        if name != "config"
    }
    return {
        "version": BINDING_VERSION,
        "role": role,
        "profile": profile,
        "config": _fact(config_path, "binding configuration"),
        "inputs": inputs,
        "panels": contracts,
    }


def create_binding_file(
    path: Path | str,
    role: str,
    config_path: Path | str,
    bindings: Mapping[str, Path | str],
) -> dict[str, object]:
    """Create a fresh semantic binding file and return its payload."""
    try:
        config = _as_path(config_path, "binding configuration")
        payload = _build_payload(role, config, bindings)
        write_json_exclusive(path, payload)
        return payload
    except (ArtifactError, BindingError) as exc:
        if isinstance(exc, BindingError):
            raise
        raise BindingError(str(exc)) from exc


def load_binding_file(path: Path | str, *, expected_role: str | None = None) -> dict[str, object]:
    """Load and revalidate a binding file against the current filesystem."""
    try:
        payload = load_json(path, "input binding")
    except ContractError as exc:
        raise BindingError(str(exc)) from exc
    if set(payload) != _TOP_KEYS:
        raise BindingError("input binding keys are invalid")
    if payload.get("version") != BINDING_VERSION:
        raise BindingError("input binding version is unsupported")
    role = payload.get("role")
    if type(role) is not str or not role:
        raise BindingError("input binding role is invalid")
    if expected_role is not None and role != expected_role:
        raise BindingError(f"input binding role differs: {role} != {expected_role}")
    profile = payload.get("profile")
    if profile not in (LOCKED_PROFILE, FIXTURE_PROFILE):
        raise BindingError("input binding profile is invalid")
    config = _strict_fact(payload.get("config"), "input binding config")
    inputs = payload.get("inputs")
    panels = payload.get("panels")
    if not isinstance(inputs, Mapping) or not isinstance(panels, Mapping):
        raise BindingError("input binding inputs and panels must be objects")
    if any(type(key) is not str for key in inputs) or any(type(key) is not str for key in panels):
        raise BindingError("input binding names must be strings")
    config_path = _as_path(config["path"], "bound configuration")
    actual_config = _fact(config_path, "bound configuration")
    if actual_config != config:
        raise BindingError("bound configuration changed after binding was written")
    bindings: dict[str, Path] = {"config": config_path}
    for name, raw_fact in inputs.items():
        fact = _strict_fact(raw_fact, f"input binding {name}")
        target = _as_path(fact["path"], f"bound {name}")
        if _fact(target, name) != fact:
            raise BindingError(f"bound input changed: {name}")
        bindings[name] = target
    _validate_semantic_names(role, bindings)
    inferred, contracts = _profile_and_contracts(bindings)
    if inferred != profile:
        raise BindingError(f"input binding profile does not match panel metadata: {profile} != {inferred}")
    if dict(panels) != contracts:
        raise BindingError("panel contracts changed after binding was written")
    _check_known_facts(profile, config_path, bindings, contracts)
    return {
        "version": BINDING_VERSION,
        "role": role,
        "profile": profile,
        "config": config,
        "inputs": {str(name): _strict_fact(value, f"input binding {name}") for name, value in sorted(inputs.items())},
        "panels": {str(name): dict(value) for name, value in sorted(panels.items())},
    }


def verify_binding_for_spec(
    path: Path | str,
    role: str,
    config_path: Path | str,
    bindings: Mapping[str, Path | str],
) -> dict[str, object]:
    """Require a binding file to describe exactly the worker's semantic spec."""
    payload = load_binding_file(path, expected_role=role)
    expected_config = _canonical_path(config_path, "spec configuration")
    actual_config = str(_as_path(payload["config"]["path"], "bound configuration"))
    if actual_config != expected_config:
        raise BindingError("input binding configuration path differs from worker spec")
    expected_names = set(bindings)
    bound_names = {"config", *payload["inputs"]}
    if expected_names != bound_names:
        raise BindingError("input binding semantic names differ from worker spec")
    for name, value in bindings.items():
        expected = _canonical_path(value, f"spec {name}")
        actual = expected_config if name == "config" else str(_as_path(payload["inputs"][name]["path"], f"bound {name}"))
        if actual != expected:
            raise BindingError(f"input binding path differs for {name}")
    return payload


__all__ = [
    "BINDING_VERSION", "BindingError", "FIXTURE_PROFILE", "REAL_CONFIG_SHA256",
    "REAL_PANEL_FACTS", "create_binding_file", "load_binding_file",
    "verify_binding_for_spec",
]
