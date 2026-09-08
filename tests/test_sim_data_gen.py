"""Physics and isolation checks; no external lattice or MAD-X process required."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sim_data_gen", SOURCE / "sim_data_gen.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = sim.load_config(SOURCE / "simulation_config.yaml")

    def test_exact_delta_pt_at_low_energy(self):
        mass, kinetic = 0.93827208816, 0.0001
        momentum = np.sqrt(kinetic * (kinetic + 2 * mass))
        beta = momentum / (mass + kinetic)
        delta = np.array([-0.01, -0.001, -1e-12, 0, 1e-12, 0.001, 0.01])
        pt = sim.delta_to_pt(delta, beta)
        np.testing.assert_allclose(sim.pt_to_delta(pt, beta), delta, rtol=3e-15, atol=1e-26)
        # Independently reconstruct energy-momentum relation in long double.
        energy = (mass + kinetic) + pt.astype(np.longdouble) * momentum
        reconstructed_p2 = (energy - mass) * (energy + mass)
        np.testing.assert_allclose(reconstructed_p2,
                                   (momentum * (1 + delta)) ** 2, rtol=2e-12)
        self.assertAlmostEqual(float(pt[4] / delta[4]), beta, places=12)
        self.assertLess(abs(pt[-2] / delta[-2]), 0.02)
        with self.assertRaises(ValueError):
            sim.delta_to_pt([-1], beta)

    def test_sampled_beam_covariance_and_dispersion(self):
        self.config["particles"] = 200000
        settings = sim.build_quad_sobol_queue(self.config, 1)[0]
        beta = 0.014598
        rays, stats = sim.generate_particles(self.config, settings, beta)
        delta = sim.pt_to_delta(rays[:, 5], beta)
        self.assertAlmostEqual(delta.std() / settings["sigma_delta"], 1, delta=0.008)
        betatron = rays[:, :4] - rays[:, 5, None] * np.array(self.config["beam"]["dispersion_pt"])
        for axis, offset in (("x", 0), ("y", 2)):
            beam = self.config["beam"]
            emit, b, a = beam["gem"+axis], beam["beta"+axis], settings["alf"+axis]
            expected = emit * np.array([[b, -a], [-a, (1+a*a)/b]])
            np.testing.assert_allclose(np.cov(betatron[:, offset:offset+2], rowvar=False, bias=True),
                                       expected, rtol=0.012, atol=0)
            self.assertAlmostEqual(stats["realized_alf"+axis] / a, 1, delta=0.015)
        np.testing.assert_array_equal(sim.generate_particles(self.config, settings, beta)[0], rays)

    def test_zero_spread_and_repeatable_sobol_prefix(self):
        queue = sim.build_quad_sobol_queue(self.config, 5)
        self.assertEqual(queue, sim.build_quad_sobol_queue(self.config, 8)[:5])
        self.assertEqual(set(queue[0]), set(self.config["quad_keys"] + ["alfx", "alfy", "sigma_delta", "index"]))
        settings = dict(queue[0], sigma_delta=0.0)
        rays, stats = sim.generate_particles(self.config, settings, 0.0146)
        self.assertTrue(np.all(rays[:, 5] == 0))
        self.assertEqual(stats["realized_sigma_delta"], 0)

    def test_runner_has_no_external_mutation_or_matching(self):
        settings = sim.build_quad_sobol_queue(self.config, 1)[0]
        text = sim.build_runner(self.config, settings)
        self.assertNotIn(self.config["lattice_root"], text)
        for forbidden in ("system,", "match,", "simplex,", "stop;", "general_lne02", "survey"):
            self.assertNotIn(forbidden, text.lower())
        self.assertIn("track, onepass=true", text)
        self.assertIn("aperture=true", text)
        self.assertIn("maxaper={1,1,1,1,1e9,1e9}", text)
        self.assertEqual(text.count("observe,"), 6)
        for key in self.config["quad_keys"]:
            self.assertIn("{}={:.17g};".format(key, settings[key]), text)
        ptc_config = dict(self.config, backend="ptc")
        self.assertIn("time=true", sim.build_runner(ptc_config, settings))
        self.assertIn("ptc_track", sim.build_runner(ptc_config, settings))

    def test_snapshot_refuses_embedded_shell_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "deflectors.ele").write_text('system, "touch outside";')
            config = copy.deepcopy(self.config)
            config["lattice_root"] = str(source)
            out = root / "out"
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "control command"):
                sim.snapshot_sources(config, out)

    def test_raw_and_centered_emittance_are_distinct(self):
        x = np.array([0.0, 1.0, 1.0, 0.0]) + 5
        px = np.array([0.0, 0.0, 1.0, 1.0]) - 2
        self.assertAlmostEqual(sim.rms_emittance(x, px, centered=True), 0.25)
        self.assertGreater(sim.rms_emittance(x, px), 0.25)

    def test_segment_identity_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trackone"
            text = "* NUMBER TURN X PX Y PY T PT S E\n"
            for i, name in enumerate(self.config["expected_segments"], 1):
                text += "#segment {} 2 4 0 {}\n".format(i, name)
                for j, values in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                    x, px = values
                    text += "{} 1 {} {} {} {} 0 0 {} 1\n".format(j+1, x, px, px, x, i)
            path.write_text(text)
            frame = sim.parse_trackone_to_df(path)
            output = sim.calculate_beam_parameters(frame, 4, self.config["expected_segments"])
            self.assertEqual(output.shape, (8, 7))
            self.assertTrue(np.all(output.transmission == 1))
            frame.attrs["segment_names"][-1] = "wrong_end"
            with self.assertRaisesRegex(ValueError, "segment names"):
                sim.calculate_beam_parameters(frame, 4, self.config["expected_segments"])
            frame.loc[0, "X"] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                sim.calculate_beam_parameters(frame)


if __name__ == "__main__":
    unittest.main()
