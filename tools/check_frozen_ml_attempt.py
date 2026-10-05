"""Explicit integration checks for a supplied E1 attempt; never part of unit discovery."""
import argparse
import csv
import json
from pathlib import Path
import analyze_frozen_ml as ml

def test_run_output_contract_and_anchors() -> None:
    result = ml.audit(OUT)
    assert result["status"] == "pass"
    metrics = json.loads((OUT / "metrics.json").read_text())
    assert metrics["quarters"]["q3"]["current_lr"]["burden"]["alerts"] == 1509
    assert metrics["quarters"]["q4"]["current_lr"]["burden"]["event_hits"] == 40
    assert metrics["quarters"]["q4"]["current_lr"]["unknown_rows"] == 31684


def test_output_has_no_device_identifiers() -> None:
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".svg"}:
            text = path.read_text(encoding="utf-8")
            assert "S300" not in text
            assert "Z302" not in text


def test_pr_and_calibration_row_counts() -> None:
    with (OUT / "pr_curve.csv").open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 404
    with (OUT / "calibration_bins.csv").open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 18


def test_summary_is_last_completion_marker() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["post_hoc"] is True
    assert summary["real_fit"] == 0
    assert summary["q1_content_read"] is False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    global OUT
    OUT = args.path.resolve(strict=True)
    tests = sorted((name, value) for name, value in globals().items()
                   if name.startswith("test_") and callable(value))
    for name, check in tests:
        check()
        print(f"PASS {name}")
    print(f"{len(tests)} integration checks passed")

if __name__ == "__main__":
    main()
