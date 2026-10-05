"""Create the deterministic synthetic daily records used by the small replay."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples/small_replay/daily.csv"
MODEL = "ST4000DM000"
CAPACITY = "4000787030016"
SERIALS = tuple(f"demo-{letter}" for letter in "ABCDEFGHIJKL")
START = dt.date(2023, 1, 1)
END = dt.date(2023, 2, 4)
SMART_FIELDS = (5, 9, 187, 188, 197, 198)
FIELDNAMES = ("date", "serial_number", "model", "capacity_bytes", "failure") + tuple(
    f"smart_{field}_raw" for field in SMART_FIELDS
)


def build_rows() -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    day_count = (END - START).days + 1
    for offset in range(day_count):
        day = START + dt.timedelta(days=offset)
        for serial in SERIALS:
            if serial == "demo-D" and day == dt.date(2023, 1, 22):
                continue
            if serial == "demo-E" and day > dt.date(2023, 1, 20):
                continue
            if serial == "demo-G" and day < dt.date(2023, 1, 10):
                continue
            values: dict[str, str | None] = {
                "date": day.isoformat(),
                "serial_number": serial,
                "model": MODEL,
                "capacity_bytes": CAPACITY,
                "failure": "0",
            }
            for field in SMART_FIELDS:
                values[f"smart_{field}_raw"] = "0"
            if serial == "demo-A" and day >= dt.date(2023, 1, 10):
                values["smart_5_raw"] = "10"
            if serial == "demo-B" and day >= dt.date(2023, 1, 18):
                values["smart_187_raw"] = "3"
            if serial == "demo-C" and day >= dt.date(2023, 1, 14):
                values["smart_197_raw"] = "2"
            if serial == "demo-I":
                values["smart_5_raw"] = "1"
                values["smart_187_raw"] = "1"
            if serial == "demo-J" and dt.date(2023, 1, 14) <= day <= dt.date(2023, 1, 18):
                values["smart_197_raw"] = "2"
            values["smart_9_raw"] = str(10000 + 24 * offset)
            if serial == "demo-F" and day == dt.date(2023, 1, 15):
                for field in SMART_FIELDS:
                    values[f"smart_{field}_raw"] = None
            if serial == "demo-A" and day == dt.date(2023, 1, 24):
                values["failure"] = "1"
            if serial == "demo-B" and day == dt.date(2023, 1, 28):
                values["failure"] = "1"
            if serial == "demo-C" and day == dt.date(2023, 1, 21):
                values["failure"] = "1"
            if serial == "demo-H" and day in {dt.date(2023, 1, 10), dt.date(2023, 1, 24)}:
                values["failure"] = "1"
            rows.append(values)
    rows.sort(key=lambda row: (str(row["date"]), str(row["serial_number"])))
    return rows


def write_fixture(output: Path = DEFAULT_OUTPUT) -> None:
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("output must be inside the project")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic small replay daily data")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        write_fixture(output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"wrote {output.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
