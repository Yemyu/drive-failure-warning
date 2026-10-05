"""Protocol-shaped synthetic input for the R0 acceptance chain.

The checked-in small replay remains a January demonstration.  This module
builds a separate, local-only source with non-demo identities and October
dates so the R runner is exercised with the protocol's date and roster
semantics.  It never reads a Backblaze file.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from pipeline.r_validation.stage_runner import run_stage
from pipeline.r_validation.cli_fixture import _write_csv
from tools import run_small_replay as replay


class ProtocolFixtureError(RuntimeError):
    """The protocol-shaped synthetic chain did not meet its own checks."""


def _protocol_rows() -> list[dict[str, Any]]:
    rows, _ = replay._parse_source(replay.SOURCE_CSV, strict=True)
    shift = dt.date(2023, 10, 1) - dt.date(2023, 1, 1)
    transformed: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        original_date = dt.date.fromisoformat(str(row["date"]))
        row["date"] = (original_date + shift).isoformat()
        row["serial_number"] = str(row["serial_number"]).replace("demo-", "synthetic-")
        # A legal SMART9 decline is kept as an input diagnostic.  The R
        # protocol explicitly preserves the non-negative raw value; the
        # feature builder receives allow_smart_decreases=True for this chain.
        if row["serial_number"] == "synthetic-A" and row["date"] == "2023-10-20":
            row["smart_9_raw"] = 1
        transformed.append(row)
    transformed.sort(key=lambda item: (str(item["date"]), str(item["serial_number"])))
    return transformed


def _spec(rows: list[dict[str, Any]]) -> dict[str, Any]:
    serials = tuple(sorted({str(row["serial_number"]) for row in rows}))
    return {
        # The fixture has a shortened observation tail but uses the R calendar
        # and event derivation.  The production CLI supplies the full Q4
        # values after a lock-bound release is approved.
        "score_start": "2023-10-01",
        "score_end": "2023-10-24",
        "event_start": "2023-10-08",
        "event_end": "2023-10-25",
        "outcome_cutoff": "2023-11-04",
        "horizon_days": 7,
        "serials": serials,
        "allow_smart_decreases": True,
    }


def run_protocol_fixture(output: Path, *, root: Path) -> dict[str, Any]:
    """Run two content-different protocol-shaped source copies through R."""
    output = Path(output).resolve()
    input_dir = output / "protocol_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = _protocol_rows()
    source_a = input_dir / "source_a.csv"
    _write_csv(source_a, rows)

    # Change only a record after the last scoring day.  The two complete
    # chains must have identical sealed selections while their source bytes
    # and panel hashes differ.
    source_b_rows = [dict(row) for row in rows]
    future = [row for row in source_b_rows if str(row["date"]) > "2023-10-24"]
    if not future:
        raise ProtocolFixtureError("protocol fixture has no future perturbation row")
    future[0]["smart_5_raw"] = int(future[0]["smart_5_raw"] or 0) + 777
    source_b = input_dir / "source_b.csv"
    _write_csv(source_b, source_b_rows)

    spec = _spec(rows)
    model = root / "examples/small_replay/current_lr.json"
    first = run_stage(
        output / "protocol_chain_a",
        root=root,
        profile="synthetic",
        source_csv=source_a,
        model_json=model,
        spec=spec,
        exercise_future_guards=True,
    )
    second = run_stage(
        output / "protocol_chain_b",
        root=root,
        profile="synthetic",
        source_csv=source_b,
        model_json=model,
        spec=spec,
        exercise_future_guards=False,
    )
    if first["selection_sha256"] != second["selection_sha256"]:
        raise ProtocolFixtureError("future-only source change altered sealed selections")
    for name, result in (("a", first), ("b", second)):
        if result.get("status") != "pass" or not result.get("independent_audit", {}).get("source_panel_verified"):
            raise ProtocolFixtureError(f"protocol chain {name} did not pass source/panel audit")
    payload = {
        "status": "pass",
        "scope": "r_protocol_shaped_synthetic",
        "source_serials": list(spec["serials"]),
        "score_start": spec["score_start"],
        "score_end": spec["score_end"],
        "event_start": spec["event_start"],
        "event_end": spec["event_end"],
        "outcome_cutoff": spec["outcome_cutoff"],
        "horizon_days": spec["horizon_days"],
        "allow_smart_decreases": True,
        "legal_smart_decrease_preserved": True,
        "source_bytes_differ": source_a.read_bytes() != source_b.read_bytes(),
        "future_perturbation_preserved_selection": True,
        "chain_a": first,
        "chain_b": second,
    }
    (output / "protocol_chain.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


__all__ = ["ProtocolFixtureError", "run_protocol_fixture"]
