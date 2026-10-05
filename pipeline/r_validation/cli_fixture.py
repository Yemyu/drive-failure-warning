"""File-backed R0 chain using the project's frozen replay implementation.

The fixture deliberately calls the same feature, score, selection and label
functions used by ``tools/run_small_replay.py``. It is still synthetic and
small; it proves the stage boundary and as-of access contract, not Q4 model
performance.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable


class FixtureChainError(RuntimeError):
    """The file-backed synthetic chain violated the stage contract."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_panel(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Materialise a panel with the same daily columns consumed by replay."""

    from tools import run_small_replay as replay

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        columns = ", ".join(
            f"{name} {'INTEGER' if name in {'capacity_bytes', 'failure'} or name.startswith('smart_') else 'TEXT'}"
            for name in replay.SOURCE_COLUMNS
        )
        connection.execute(
            f"CREATE TABLE daily ({columns}, PRIMARY KEY (date, serial_number))"
        )
        placeholders = ",".join("?" for _ in replay.SOURCE_COLUMNS)
        connection.executemany(
            f"INSERT INTO daily ({','.join(replay.SOURCE_COLUMNS)}) VALUES ({placeholders})",
            [tuple(row[name] for name in replay.SOURCE_COLUMNS) for row in rows],
        )
        connection.commit()
    finally:
        connection.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from tools import run_small_replay as replay

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=replay.SOURCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: "" if row[name] is None else row[name]
                for name in replay.SOURCE_COLUMNS
            })


def _run_score(panel: Path, output_db: Path, rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    from tools import run_small_replay as replay

    access: list[dict[str, Any]] = []
    connection = sqlite3.connect(panel)
    connection.row_factory = sqlite3.Row
    try:
        replay._score_and_select(connection, rows, model, output_db, access)
    finally:
        connection.close()
    score_access = [item for item in access if item.get("phase") == "score"]
    if not score_access or any(
        item.get("max_date") is not None and item["max_date"] > item["decision_date"]
        for item in score_access
    ):
        raise FixtureChainError("score stage accessed a future row")
    return access


def _seal(selection: Path, seal: Path) -> str:
    digest = _sha(selection)
    seal.write_text(
        json.dumps({
            "schema": "r0-fixture-seal-v2",
            "selection_sha256": digest,
            "selection_closed": True,
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def _manual_label(rows: list[dict[str, Any]], decision: dt.date) -> int | None:
    dates = {dt.date.fromisoformat(str(row["date"])) for row in rows}
    failures = sorted(
        dt.date.fromisoformat(str(row["date"]))
        for row in rows if int(row["failure"]) == 1
    )
    first_failure = failures[0] if failures else None
    if first_failure is not None and first_failure <= decision:
        return None
    horizon = [decision + dt.timedelta(days=i) for i in range(1, 8)]
    if first_failure is not None and first_failure in horizon:
        return 1
    if all(day in dates for day in horizon) and not any(day in failures for day in horizon):
        return 0
    return None


def _independent_audit(selection: Path, panel: Path, evaluation: dict[str, Any], audit: Path) -> dict[str, Any]:
    """Recompute selected labels without calling the production evaluator."""

    from tools import run_small_replay as replay

    connection = sqlite3.connect(panel)
    connection.row_factory = sqlite3.Row
    selected = sqlite3.connect(selection.as_uri() + "?mode=ro&immutable=1", uri=True)
    selected.row_factory = sqlite3.Row
    try:
        rows_by_serial: dict[str, list[dict[str, Any]]] = {}
        for serial in replay.SERIALS:
            rows_by_serial[serial] = [
                dict(row) for row in connection.execute(
                    "SELECT date,serial_number,model,capacity_bytes,failure," +
                    ",".join(f"smart_{field}_raw" for field in replay.SMART_FIELDS) +
                    " FROM daily WHERE serial_number=? ORDER BY date", (serial,)
                )
            ]
        counts: dict[str, dict[str, int]] = {
            method: {"alerts": 0, "known_hit_alerts": 0, "known_no_hit_alerts": 0, "unknown_alerts": 0}
            for method in replay.METHODS
        }
        for row in selected.execute(
            "SELECT model,decision_date,serial_number FROM selections ORDER BY model,decision_date,selected_rank"
        ):
            method = str(row["model"])
            label = _manual_label(rows_by_serial[str(row["serial_number"])], dt.date.fromisoformat(row["decision_date"]))
            counts[method]["alerts"] += 1
            if label == 1:
                counts[method]["known_hit_alerts"] += 1
            elif label == 0:
                counts[method]["known_no_hit_alerts"] += 1
            else:
                counts[method]["unknown_alerts"] += 1
        for method, count in counts.items():
            recorded = evaluation.get("methods", {}).get(method, {})
            for key, value in count.items():
                if int(recorded.get(key, -1)) != value:
                    raise FixtureChainError(f"independent audit mismatch: {method}/{key}")
        result = {
            "status": "pass",
            "independent_counts": counts,
            "evaluation_matches": True,
            "selection_sha256": _sha(selection),
        }
    finally:
        selected.close()
        connection.close()
    audit.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _legacy_cli_fixture_chain(output: Path, *, root: Path) -> dict[str, Any]:
    """Historical pre-runner chain retained only for old evidence readers."""

    del root
    from tools import run_small_replay as replay

    chain = output / "cli_chain"
    chain.mkdir(parents=True, exist_ok=True)
    for name in ("daily.csv", "current_lr.json", "expected.json", "manifest.json"):
        shutil.copy2(replay.FIXTURE_ROOT / name, chain / name)

    # Consume the copied run-local files, rather than merely placing them in
    # the evidence directory.  The loader still enforces the frozen fixture
    # schema and row/date/count boundaries.
    source_rows, _ = replay._parse_source(chain / "daily.csv")
    frozen_model = replay._load_model()
    copied_model = replay._json_read(chain / "current_lr.json")
    if copied_model != frozen_model:
        raise FixtureChainError("run-local model copy differs from frozen model")
    # Validation above proves the copied bytes are the approved payload; the
    # actual scorer consumes that run-local object, so the evidence path is
    # the object used by this chain rather than the source template path.
    model = copied_model
    expected = replay._json_read(chain / "expected.json")
    panel = chain / "panel.sqlite"
    _write_panel(panel, source_rows)
    selection = chain / "selection.sqlite"
    access = _run_score(panel, selection, source_rows, model)
    score_access = [item for item in access if item.get("phase") == "score"]

    # Modify a record strictly after the last decision day. The same
    # production scorer must produce byte-identical selections.
    perturbed_rows = [dict(row) for row in source_rows]
    changed = False
    for row in perturbed_rows:
        if row["date"] == "2023-01-29" and row["serial_number"] == "demo-A":
            row["smart_5_raw"] = int(row["smart_5_raw"] or 0) + 999
            changed = True
            break
    if not changed:
        raise FixtureChainError("future perturbation target is missing")
    perturbed_source = chain / "daily_perturbed.csv"
    _write_csv(perturbed_source, perturbed_rows)
    perturbed_panel = chain / "panel_perturbed.sqlite"
    _write_panel(perturbed_panel, perturbed_rows)
    perturbed_selection = chain / "selection_perturbed.sqlite"
    _run_score(perturbed_panel, perturbed_selection, perturbed_rows, model)
    if _sha(selection) != _sha(perturbed_selection):
        raise FixtureChainError("future value perturbation changed an as-of selection")

    # Delete a future row in a second panel. The score stage does not depend
    # on data after its decision dates, so this must also preserve the output.
    deleted_rows = [
        row for row in source_rows
        if not (row["date"] == "2023-01-30" and row["serial_number"] == "demo-B")
    ]
    deleted_panel = chain / "panel_future_row_deleted.sqlite"
    _write_panel(deleted_panel, deleted_rows)
    deleted_selection = chain / "selection_future_row_deleted.sqlite"
    _run_score(deleted_panel, deleted_selection, deleted_rows, model)
    if _sha(selection) != _sha(deleted_selection):
        raise FixtureChainError("future row deletion changed an as-of selection")

    seal = chain / "selection_seal.json"
    selection_sha = _seal(selection, seal)
    outcome_marker = chain / "outcome_access_after_seal.json"
    outcome_marker.write_text(json.dumps({
        "selection_sha256": selection_sha,
        "opened_after_selection_seal": True,
        "source": "panel.sqlite",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    panel_connection = sqlite3.connect(panel)
    panel_connection.row_factory = sqlite3.Row
    try:
        evaluation = replay._evaluate(
            selection, panel_connection, selection_sha, expected, access
        )
    finally:
        panel_connection.close()
    evaluation_path = chain / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = _independent_audit(selection, panel, evaluation, chain / "audit.json")
    if audit.get("status") != "pass":
        raise FixtureChainError("independent fixture audit failed")
    if not evaluation.get("evaluation_opened_after_selection_close"):
        raise FixtureChainError("evaluation did not open after selection close")

    result = {
        "status": "pass",
        "stages": ["source", "panel", "score", "seal", "evaluate", "audit"],
        "source_sha256": _sha(chain / "daily.csv"),
        "panel_sha256": _sha(panel),
        "selection_sha256": selection_sha,
        "evaluation_sha256": _sha(evaluation_path),
        "audit_sha256": _sha(chain / "audit.json"),
        "model_binding": {
            "model": "current_lr",
            "parameters": "cli_chain/current_lr.json",
            "feature_count": len(model["features"]),
        },
        "score_access": {
            "query_count": len(score_access),
            "all_max_date_at_or_before_decision": True,
        },
        "future_perturbation_preserved_selection": True,
        "future_deletion_preserved_selection": True,
        "outcome_access_after_seal": True,
        "evaluation": {
            "methods": evaluation.get("methods", {}),
            "evaluation_opened_after_selection_close": evaluation.get("evaluation_opened_after_selection_close"),
        },
        "audit": audit,
    }
    (chain / "chain_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_cli_fixture_chain(output: Path, *, root: Path) -> dict[str, Any]:
    """Run the shared parameterized stage runner under the synthetic profile."""
    from pipeline.r_validation.stage_runner import run_stage

    return run_stage(
        output,
        root=root,
        profile="synthetic",
        spec={"serials": tuple(f"demo-{letter}" for letter in "ABCDEFGHIJKL")},
        exercise_future_guards=True,
    )


def _independent_audit(selection: Path, panel: Path, evaluation: dict[str, Any], audit: Path) -> dict[str, Any]:
    """Compatibility entry that delegates to the independent reconstruction."""
    from pipeline.r_validation.independent_audit import audit_selection
    from tools import run_small_replay as replay

    return audit_selection(selection, panel, evaluation, audit, model=replay._load_model())
