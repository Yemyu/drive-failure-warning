import json
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.reproducible import current
from pipeline.reproducible.artifacts import artifact_facts
from pipeline.reproducible import training_check
from pipeline.reproducible.contract import ContractError, load_config_for_bindings, require_panel_contract
from pipeline.reproducible.input_binding import (
    BindingError,
    create_binding_file,
    load_binding_file,
    verify_binding_for_spec,
)
from pipeline.reproducible.training_check import TrainingCheckError, load_approved_training_package, verify_training_package, write_training_package
from tests.test_reproducible_pipeline import make_panel


ROOT = Path(__file__).resolve().parents[1]


class InputBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_metadata_less_panel_is_rejected(self):
        panel = self.root / "bare.sqlite"
        with sqlite3.connect(panel) as connection:
            connection.execute("CREATE TABLE daily(date TEXT)")
        with self.assertRaisesRegex(ContractError, "explicit fixture or real panel marker"):
            require_panel_contract(panel, role="training")

    def test_binding_round_trip_and_filesystem_tamper_are_detected(self):
        panel = self.root / "panel.sqlite"
        config = self.root / "config.json"
        binding = self.root / "train.binding.json"
        make_panel(panel)
        config.write_text(json.dumps({"profile": "synthetic_fixture_v1", "model": "current_lr"}), encoding="utf-8")
        bindings = {"train_panel": panel, "config": config}
        payload = create_binding_file(binding, "train", config, bindings)
        self.assertEqual(payload["profile"], "synthetic_fixture_v1")
        loaded = load_binding_file(binding, expected_role="train")
        self.assertEqual(loaded["inputs"]["train_panel"]["sha256"], payload["inputs"]["train_panel"]["sha256"])
        self.assertEqual(verify_binding_for_spec(binding, "train", config, bindings)["role"], "train")
        with sqlite3.connect(panel) as connection:
            connection.execute("UPDATE daily SET smart_5_raw=smart_5_raw+1 WHERE serial_number='disk-b' AND date='2023-01-01'")
        with self.assertRaisesRegex(BindingError, "changed|identity"):
            load_binding_file(binding, expected_role="train")


class TrainingPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_independent_prefit_check_rejects_tampered_package(self):
        panel = self.root / "panel.sqlite"
        labels = self.root / "labels.sqlite"
        features = self.root / "features.sqlite"
        package = self.root / "training_input"
        make_panel(panel)
        current.build_labels([panel], labels, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04", run_id="test-run")
        current.build_current_features(panel, features, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04")
        prepared = current._prepare_training_arrays(features, labels, "test-run")
        write_training_package(package, prepared)
        approval = verify_training_package(features, labels, "test-run", package)
        self.assertEqual(approval["status"], "ready_for_fit")
        tampered = package / "labels.npy"
        values = __import__("numpy").load(tampered, allow_pickle=False)
        values[0] = 1 - values[0]
        with tampered.open("wb") as handle:
            __import__("numpy").save(handle, values, allow_pickle=False)
        with self.assertRaisesRegex(TrainingCheckError, "changed"):
            verify_training_package(features, labels, "test-run", package)

    def test_fixture_config_requires_explicit_profile(self):
        panel = self.root / "panel.sqlite"
        config = self.root / "config.json"
        make_panel(panel)
        config.write_text(json.dumps({"model": "current_lr"}), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "synthetic_fixture_v1"):
            load_config_for_bindings(config, {"train_panel": panel, "config": config})

    def test_approved_package_consumer_rejects_manifest_replacement(self):
        panel = self.root / "panel.sqlite"
        labels = self.root / "labels.sqlite"
        features = self.root / "features.sqlite"
        package = self.root / "training_input"
        make_panel(panel)
        current.build_labels([panel], labels, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04", run_id="approval-run")
        current.build_current_features(panel, features, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04")
        prepared = current._prepare_training_arrays(features, labels, "approval-run")
        write_training_package(package, prepared)
        approval = verify_training_package(features, labels, "approval-run", package)
        loaded = load_approved_training_package(package, approval)
        self.assertEqual(loaded["run_id"], "approval-run")
        package_json = package / "package.json"
        package_json.write_text(package_json.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(TrainingCheckError, "manifest changed"):
            load_approved_training_package(package, approval)

    def test_approved_package_decodes_the_bytes_it_checked(self):
        panel = self.root / "panel.sqlite"
        labels = self.root / "labels.sqlite"
        features = self.root / "features.sqlite"
        package = self.root / "training_input"
        make_panel(panel)
        current.build_labels([panel], labels, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04", run_id="bytes-run")
        current.build_current_features(panel, features, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04")
        prepared = current._prepare_training_arrays(features, labels, "bytes-run")
        write_training_package(package, prepared)
        approval = verify_training_package(features, labels, "bytes-run", package)
        original_read = training_check._read_bytes
        original_labels = np.load(package / "labels.npy", allow_pickle=False).copy()

        def read_then_replace(path, label):
            data = original_read(path, label)
            if Path(path).name == "labels.npy":
                replacement = np.load(io.BytesIO(data), allow_pickle=False)
                replacement[0] = 1 - replacement[0]
                with Path(path).open("wb") as handle:
                    np.save(handle, replacement, allow_pickle=False)
            return data

        with patch.object(training_check, "_read_bytes", side_effect=read_then_replace):
            loaded = load_approved_training_package(package, approval)
        self.assertEqual(loaded["labels"].tolist(), original_labels.tolist())

    def test_prefit_check_rejects_manifest_replacement_during_check(self):
        panel = self.root / "panel.sqlite"
        labels = self.root / "labels.sqlite"
        features = self.root / "features.sqlite"
        package = self.root / "training_input"
        make_panel(panel)
        current.build_labels([panel], labels, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04", run_id="race-run")
        current.build_current_features(panel, features, start="2023-01-15", end="2023-01-28", dataset_end="2023-02-04")
        prepared = current._prepare_training_arrays(features, labels, "race-run")
        write_training_package(package, prepared)
        expected_inputs = {"package_manifest": artifact_facts(package / "package.json")}
        original_read = training_check._read_bytes
        replaced = {"done": False}

        def read_then_replace(path, label):
            data = original_read(path, label)
            if Path(path).name == "package.json" and not replaced["done"]:
                replaced["done"] = True
                Path(path).write_bytes(data + b"\n")
            return data

        with patch.object(training_check, "_read_bytes", side_effect=read_then_replace):
            with self.assertRaisesRegex(TrainingCheckError, "changed"):
                verify_training_package(
                    features, labels, "race-run", package,
                    expected_inputs=expected_inputs,
                )


if __name__ == "__main__":
    unittest.main()
