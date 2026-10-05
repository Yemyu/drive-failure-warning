"""Small full-calendar Q1 fixture used only for the implementation gate.

The fixture has the real Q1 calendar (91 raw days, 84 scoring days, leap
day, and a 13-day December bridge) but synthetic rows and identities.  It is
never a substitute for the Backblaze release.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
import sqlite3
import zipfile

from pipeline.r_validation.cli_fixture import _write_panel
from pipeline.r_validation.history import file_sha
from tools import run_small_replay as replay

Q1_START = dt.date(2024, 1, 1)
Q1_END = dt.date(2024, 3, 31)
PREFIX = "q1_fixture"
SERIALS = tuple(f"q1-synthetic-{i:02d}" for i in range(12))


def _row(day: dt.date, serial: str, *, failure: int = 0, smart_signal: int = 1) -> dict:
    values = {name: None for name in replay.SOURCE_COLUMNS}
    values.update({"date": day.isoformat(), "serial_number": serial,
                   "model": replay.MODEL, "capacity_bytes": 4000787030016,
                   "failure": failure})
    for field in replay.SMART_FIELDS:
        values[f"smart_{field}_raw"] = smart_signal
    values["smart_9_raw"] = 100
    return values


def _history(path: Path, serials: tuple[str, ...], *, start: dt.date,
             days: int, prior_failure: str | None = None) -> None:
    rows = []
    for serial in serials:
        if prior_failure and serial == serials[0]:
            rows.append(_row(dt.date(2023, 12, 10), serial, failure=1))
        for offset in range(days):
            rows.append(_row(start + dt.timedelta(days=offset), serial))
    _write_panel(path, rows)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE serial_model_registry(serial_number TEXT PRIMARY KEY, model TEXT NOT NULL)")
        connection.executemany("INSERT INTO serial_model_registry VALUES (?, ?)",
                               [(serial, replay.MODEL) for serial in serials])
        connection.commit()
    finally:
        connection.close()


def _archive(path: Path, *, future_variant: bool = False) -> None:
    columns = list(replay.SOURCE_COLUMNS)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{PREFIX}/.DS_Store", b"synthetic Q1 directory metadata")
        day = Q1_START
        while day <= Q1_END:
            rows = []
            for index, serial in enumerate(SERIALS):
                failure = int((index == 1 and day == dt.date(2024, 2, 29))
                              or (index == 2 and day == dt.date(2024, 3, 26)))
                signal = 1
                # This failure is outside the main event interval but inside
                # the outcome cutoff; it exercises the separate boundary.
                if future_variant and serial == SERIALS[3] and day == dt.date(2024, 3, 30):
                    signal = 777
                rows.append(_row(day, serial, failure=failure, smart_signal=signal))
            stream = __import__("io").StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: "" if row[name] is None else row[name] for name in columns})
            archive.writestr(f"{PREFIX}/{day.isoformat()}.csv", stream.getvalue())
            day += dt.timedelta(days=1)


def build_q1_inputs(root: Path, output: Path, *, future_variant: bool = False) -> dict:
    """Create a bounded Q1 source and three non-overlapping history panels."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "data_Q1_2024_fixture.zip"
    _archive(archive, future_variant=future_variant)
    history_dir = output / "history"
    history_dir.mkdir()
    groups = (SERIALS[:4], SERIALS[4:8], SERIALS[8:])
    names = ("q1q2_panel", "q3_panel", "q4_candidate_panel")
    history = {}
    starts = (dt.date(2023, 12, 19), dt.date(2023, 7, 1), dt.date(2023, 10, 1))
    lengths = (13, 92, 79)
    for name, serials, start, days in zip(names, groups, starts, lengths):
        path = history_dir / f"{name}.sqlite"
        _history(path, serials, start=start, days=days, prior_failure=name == "q1q2_panel")
        history[name] = {"path": str(path), "sha256": file_sha(path)}
    return {
        "archive": archive,
        "history": history,
        "source": {"archive": {"path": str(archive), "sha256": file_sha(archive)},
                   "receipt": None, "prefix": PREFIX,
                   "start": Q1_START.isoformat(), "end": Q1_END.isoformat()},
    }


def q1_spec() -> dict:
    return {"score_start": "2024-01-01", "score_end": "2024-03-24",
            "event_start": "2024-01-08", "event_end": "2024-03-25",
            "outcome_cutoff": "2024-03-31", "horizon_days": 7,
            "allow_smart_decreases": True}


def synthetic_contract(inputs: dict, model_path: Path, *, profile: str = "synthetic_q1_bound_zip") -> dict:
    return {"schema": "r-bound-zip-v1", "profile": profile,
            "source": inputs["source"], "model": {"path": str(model_path), "sha256": file_sha(model_path)},
            "history": inputs["history"], "spec": q1_spec(), "authorization": None}


__all__ = ["build_q1_inputs", "q1_spec", "synthetic_contract", "Q1_START", "Q1_END", "PREFIX"]
