"""The R quality gate counts all eligible scoring rows, not alerts."""
import copy
from contextlib import closing
import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.r_validation.stage_runner import _scoring_quality, _protocol_metrics
from pipeline.r_validation.cli_fixture import run_cli_fixture_chain
from pipeline.r_validation.independent_audit import audit_selection, IndependentAuditError

ROOT = Path(__file__).resolve().parents[1]


class QualityGateTests(unittest.TestCase):
    def quality(self, unknown, total=5):
        with closing(sqlite3.connect(":memory:")) as selection, closing(sqlite3.connect(":memory:")) as panel:
            selection.executescript("CREATE TABLE metadata(key,value); CREATE TABLE features(decision_date,serial_number);")
            selection.executemany("INSERT INTO metadata VALUES (?,?)", {
                "start": "2023-10-01", "end": "2023-10-01", "outcome_cutoff": "2023-10-08",
                "horizon_days": "7", "history_days": "14", "min_history": "12",
            }.items())
            panel.execute("CREATE TABLE daily(date,serial_number,model,capacity_bytes,failure)")
            for i in range(total):
                serial = f"disk-{i}"
                selection.execute("INSERT INTO features VALUES (?,?)", ("2023-10-01", serial))
                # All devices qualify on Oct 1; some disappear immediately afterward.
                for offset in range(14 if i < unknown else 21):
                    date = (dt.date(2023, 9, 18) + dt.timedelta(days=offset)).isoformat()
                    panel.execute("INSERT INTO daily VALUES (?,?,?,?,?)", (date, serial, "ST4000DM000", 4000787030016, 0))
            return _scoring_quality(selection, panel)

    def test_exact_twenty_percent_passes(self):
        quality = self.quality(1)
        self.assertEqual((quality["unknown_rows"], quality["eligible_rows"]), (1, 5))
        self.assertEqual(quality["unknown_ratio"], 0.2)
        self.assertEqual(quality["status"], "pass")

    def test_above_twenty_percent_fails(self):
        self.assertEqual(self.quality(2)["status"], "failed")

    def test_empty_scoring_population_is_not_a_pass(self):
        quality = self.quality(0, 0)
        self.assertEqual(quality["status"], "not_evaluable")
        self.assertIsNone(quality["unknown_ratio"])

    def test_alert_ratios_do_not_control_quality_and_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            output = Path(directory)
            run_cli_fixture_chain(output, root=ROOT)
            chain = output / "cli_chain"
            evaluation = json.loads((chain / "evaluation.json").read_text())
            changed = copy.deepcopy(evaluation)
            for method in changed["methods"].values():
                method["unknown_alert_ratio"] = 1.0
            metrics = _protocol_metrics(chain / "selection.sqlite", chain / "panel.sqlite", changed)
            self.assertEqual(metrics["quality_gate"], evaluation["protocol_metrics"]["quality_gate"])
            for key, value in (("unknown_rows", 999), ("eligible_rows", 999),
                               ("scope", "alerts"), ("unknown_ratio", 0.999)):
                with self.subTest(field=key):
                    corrupted = copy.deepcopy(evaluation)
                    corrupted["protocol_metrics"]["quality_gate"][key] = value
                    with self.assertRaises(IndependentAuditError):
                        audit_selection(chain / "selection.sqlite", chain / "panel.sqlite", corrupted,
                                        output / "unexpected_audit.json",
                                        model=json.loads((chain / "current_lr.json").read_text()))


if __name__ == "__main__":
    unittest.main()
