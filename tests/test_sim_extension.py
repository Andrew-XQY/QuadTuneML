"""Reuse rejects altered physics, missing records and inconsistent diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import sim_data_gen as sim


class ExtensionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = sim.load_config(sim.REPO_ROOT / "simulation_config.yaml")
        self.config.update(samples=8, lattice_root=str(self.directory / "lattice"),
                           external_runner=str(self.directory / "original.madx"),
                           initial_conditions=str(self.directory / "initial.inp"))
        sources = [Path(self.config["lattice_root"]) / name for name in sim.lattice_files()]
        sources += [Path(self.config[name]) for name in ("external_runner", "initial_conditions")]
        for path in sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("! Synthetic static source used only for validation tests.\n")
        self.queue = sim.build_quad_sobol_queue(self.config, 8)
        self.previous = self.directory / "previous"
        self.previous.mkdir()
        template = sim.build_runner(self.config, self.queue[0])
        (self.previous / "runner_template.madx").write_text(template)
        records, diagnostics = [], []
        for settings in self.queue[:4]:
            row = dict(settings)
            row.update({key: [1.0 if key == "transmission" else 1e-6]*8 for key in sim.OUTPUT_KEYS})
            records.append(row)
            diag = {key: 0 for key in sim.diagnostic_columns(settings)}
            diag.update(settings, status="ok", final_particles=self.config["particles"],
                        madx_version="test-version", error="", actual_segment_names="[]")
            diagnostics.append(diag)
        pd.DataFrame(records).to_csv(self.previous / "data.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(self.previous / "diagnostics.csv", index=False)
        (self.previous / "failures.jsonl").write_text("")
        prior_config = dict(self.config, samples=4, output=str(self.previous / "data.csv"))
        self.metadata = dict(status="complete", completed=4, requested=4, failed=0,
            configuration=prior_config, physics_implementation_sha256=sim.physics_implementation_sha256(),
            generator_sha256="synthetic-test-generator", source_sha256={str(path): sim.file_sha256(path) for path in sources},
            runner_template_sha256=sim.file_sha256(self.previous / "runner_template.madx"),
            package_versions={}, madx_version="test-version", actual_segment_names=[])
        self.save_metadata()
        self.madx = patch("cpymad.madx.Madx")
        mock = self.madx.start()
        self.addCleanup(self.madx.stop)
        mock.return_value.__enter__.return_value.version = "test-version"

    def save_metadata(self):
        (self.previous / "metadata.json").write_text(json.dumps(self.metadata))

    def validate(self, config=None):
        return sim.validate_extension(config or self.config, self.previous, self.queue)

    def test_valid_complete_prefix(self):
        report = self.validate()
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["data_sha256"], sim.file_sha256(self.previous / "data.csv"))

    def test_changed_seed_or_beam_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["beam"]["alfx"] += 0.01
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.validate(changed)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.validate(dict(self.config, seed=99))

    def test_changed_input_record_rejected(self):
        path = self.previous / "data.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        frame.loc[0, "sigma_delta"] += 0.00001
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "exact deterministic Sobol prefix"):
            self.validate()

    def test_missing_diagnostic_record_rejected(self):
        path = self.previous / "diagnostics.csv"
        frame = pd.read_csv(path).iloc[:-1]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "row counts"):
            self.validate()

    def test_changed_source_rejected(self):
        (Path(self.config["lattice_root"]) / "deflectors.ele").write_text("! changed\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.validate()

    def test_loss_accounting_rejected(self):
        path = self.previous / "diagnostics.csv"
        frame = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
        frame.loc[1, "final_particles"] -= 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "particle-loss accounting"):
            self.validate()

    def test_unknown_legacy_implementation_rejected(self):
        del self.metadata["physics_implementation_sha256"]
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "Unrecognized legacy"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
