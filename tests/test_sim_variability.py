"""Repeat inputs stay fixed; empirical ranges use the saved training scale."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import sim_data_gen as sim
import simulation_variability as variability


class VariabilityTests(unittest.TestCase):
    def test_fixed_inputs_independent_particle_realizations(self):
        config = sim.load_config(sim.REPO_ROOT / "simulation_config.yaml")
        config.update(seed=20260908, particles=100)
        queue = variability.make_repeat_queue(config, anchors=2, replicates=3)
        keys = config["quad_keys"] + ["alfx", "alfy", "sigma_delta"]
        for anchor in range(2):
            group = queue[3*anchor:3*(anchor+1)]
            self.assertTrue(all([row[key] for key in keys] == [group[0][key] for key in keys] for row in group))
            self.assertEqual([row["replicate_id"] for row in group], [0, 1, 2])
        self.assertEqual([row["index"] for row in queue], list(range(6)))
        first = sim.generate_particles(config, queue[0], .0146)[0]
        second = sim.generate_particles(config, queue[1], .0146)[0]
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, sim.generate_particles(config, queue[0], .0146)[0])
        self.assertFalse(np.array_equal(first, sim.generate_particles(dict(config, seed=42), queue[0], .0146)[0]))

    def create_summary_fixture(self, root):
        directory = root / "results/study"
        directory.mkdir(parents=True)
        keys = ["q1", "q2", "q3", "q4", "alfx", "alfy", "sigma_delta"]
        records = []
        for anchor, log_x, log_y in ((0, [1, 2, 3], [-5, -4, -3]), (1, [2, 4, 6], [-2, -1, 0])):
            for replica in range(3):
                record = {key: float(anchor) for key in keys}
                record.update(anchor_id=anchor, replicate_id=replica,
                              emittance_x=[float(np.exp(log_x[replica]))]*8,
                              emittance_y=[float(np.exp(log_y[replica]))]*8)
                records.append(record)
        data_path = directory / "data.csv"
        pd.DataFrame(records).to_csv(data_path, index=False)
        pd.DataFrame(dict(lost_particles=[0]*6, nonfinite_lost_particles=[0]*6)).to_csv(directory / "diagnostics.csv", index=False)
        metadata = dict(status="complete", failed=0, variability_study=dict(anchors=2, replicates=3),
                        configuration=dict(particles=6000), input_columns=keys)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        preprocessing = dict(target_transform="natural_log_then_minmax", input_cols=keys,
            target_cols=["emittance_x", "emittance_y"], segment=7, target_log_min=[-10, -7],
            target_log_range=[2, 4], fit_rows=9600)
        prep_path = directory / "preprocessing.json"
        prep_path.write_text(json.dumps(preprocessing))
        return data_path, prep_path

    def test_scaled_band_uses_saved_training_range(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(sim, "REPO_ROOT", Path(temp)):
            data, preprocessing = self.create_summary_fixture(Path(temp))
            report = variability.summarize(data, preprocessing)
            self.assertAlmostEqual(report["spaces"]["log"]["emittance_x"]["max_observed_range"], 4)
            self.assertAlmostEqual(report["spaces"]["scaled"]["emittance_x"]["half_max_observed_range"], 1)
            self.assertAlmostEqual(report["spaces"]["scaled"]["emittance_y"]["half_max_observed_range"], .25)
            self.assertFalse(report["preprocessing"]["fitted_on_repeats"])

    def test_changed_input_within_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(sim, "REPO_ROOT", Path(temp)):
            data, preprocessing = self.create_summary_fixture(Path(temp))
            frame = pd.read_csv(data)
            frame.loc[0, "alfx"] += .1
            frame.to_csv(data, index=False)
            with self.assertRaisesRegex(ValueError, "input changes within anchor"):
                variability.summarize(data, preprocessing)


if __name__ == "__main__":
    unittest.main()
