from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
import zipfile

from pipeline.r_validation.q1_binding import validate_q1_bound_zip
from pipeline.r_validation.q1_fixture import build_q1_inputs, q1_spec, synthetic_contract
from pipeline.r_validation.release import ReleaseError

ROOT = Path(__file__).resolve().parents[1]


class Q1ProtocolTests(unittest.TestCase):
    def test_locked_calendar_and_event_window(self):
        spec = q1_spec()
        self.assertEqual(spec["score_start"], "2024-01-01")
        self.assertEqual(spec["score_end"], "2024-03-24")
        self.assertEqual(spec["event_start"], "2024-01-08")
        self.assertEqual(spec["event_end"], "2024-03-25")
        self.assertEqual(spec["outcome_cutoff"], "2024-03-31")
        self.assertEqual((dt.date.fromisoformat(spec["outcome_cutoff"]) -
                          dt.date.fromisoformat(spec["score_start"])).days + 1, 91)

    def test_q1_fixture_has_91_members_and_three_nonoverlapping_histories(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            inputs = build_q1_inputs(ROOT, Path(directory) / "inputs")
            with zipfile.ZipFile(inputs["archive"]) as archive:
                names = set(archive.namelist())
            dates = {name.rsplit("/", 1)[-1] for name in names if name.endswith(".csv")}
            self.assertEqual(len(dates), 91)
            self.assertIn("q1_fixture/.DS_Store", names)
            self.assertEqual(set(inputs["history"]), {"q1q2_panel", "q3_panel", "q4_candidate_panel"})
            validate_q1_bound_zip(synthetic_contract(inputs, ROOT / "examples/small_replay/current_lr.json"))

    def test_q1_source_prefix_is_not_accepted_as_q4(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            inputs = build_q1_inputs(ROOT, Path(directory) / "inputs")
            contract = synthetic_contract(inputs, ROOT / "examples/small_replay/current_lr.json")
            contract["source"]["prefix"] = "data_Q4_2023"
            with self.assertRaises(ReleaseError):
                validate_q1_bound_zip(contract)


if __name__ == "__main__":
    unittest.main()
