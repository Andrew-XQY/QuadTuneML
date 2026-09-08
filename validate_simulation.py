"""Run small, reproducible native MAD-X tracking checks without training.

Examples: python validate_simulation.py --output results/validation/native_check
The actual lattice is copied; external files are never modified or executed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cpymad.madx import Madx

import sim_data_gen as sim


def track_example(directory, geometry, rays, momentum, observation):
    directory.mkdir()
    script = "beam,particle=antiproton,pc={:.17g};\n".format(momentum) + geometry
    script += "\ntrack,onepass=true,onetable=true,dump=true,aperture=true,recloss=true;\n"
    for row in rays:
        script += "start,x={:.17g},px={:.17g},y={:.17g},py={:.17g},t={:.17g},pt={:.17g};\n".format(*row)
    script += "observe,place={};\n".format(observation)
    script += "run,turns=1,maxaper={1,1,1,1,1e9,1e9};\nendtrack;\n"
    (directory / "runner.madx").write_text(script)
    with open(directory / "madx.log", "w") as stream:
        with Madx(stdout=stream) as madx:
            with madx.chdir(str(directory)):
                madx.call("runner.madx")
                loss = madx.table.trackloss.dframe()
                loss.to_csv(directory / "trackloss.csv", index=False)
    frame = sim.parse_trackone_to_df(directory / "trackone")
    frame.attrs["recorded_losses"] = len(loss)
    return frame


def validate(config, output):
    output = Path(output).resolve()
    if (sim.REPO_ROOT / "results").resolve() not in output.parents:
        raise ValueError("Validation output must be inside repo results/")
    output.mkdir(parents=True, exist_ok=False)
    # Ask the installed MAD-X for its mass, rather than duplicating constants.
    with Madx(stdout=False) as madx:
        madx.input("beam,particle=antiproton;")
        mass = float(madx.beam.mass)
    kinetic = config["beam"]["kinetic_energy_gev"]
    momentum = np.sqrt(kinetic * (kinetic + 2 * mass))
    beta = momentum / (mass + kinetic)
    delta = np.array([0.0, 0.001, -0.001])
    pt = sim.delta_to_pt(delta, beta)
    rays = np.zeros((3, 6))
    rays[:, 0], rays[:, 1], rays[:, 5] = 1e-6, 2e-7, pt
    report = {"beta0": float(beta), "p0_gev_c": float(momentum),
              "limitations": "Checks numerical implementation, not calibration against measured beam or backend parity."}

    matrix = ("m:matrix,l=0,rm11=2,rm12=3,rm16=4,rm21=0,rm22=.5,"
              "rm33=1,rm44=1,rm55=1,rm66=1;\n"
              "line1:line=(m);\nuse,period=line1;\n")
    frame = track_example(output / "matrix", matrix, rays, momentum, "m")
    final = frame[frame.segment == frame.segment.max()]
    expected_x = 2 * rays[:, 0] + 3 * rays[:, 1] + 4 * pt
    np.testing.assert_allclose(final.X.values, expected_x, rtol=3e-10, atol=1e-15)
    np.testing.assert_allclose(final.PX.values, 0.5 * rays[:, 1], rtol=1e-10, atol=1e-16)
    report["matrix"] = {"expected_x": expected_x.tolist(), "actual_x": final.X.tolist(),
                         "max_absolute_x_error": float(np.max(abs(final.X.values-expected_x))),
                         "passed": True}

    quad = "q:quadrupole,l=1,k1=1;\nline1:line=(q);\nuse,period=line1;\n"
    frame = track_example(output / "quad", quad, rays, momentum, "q")
    final = frame[frame.segment == frame.segment.max()]
    root = np.sqrt(1 / (1 + delta))
    expected_x = np.cos(root) * rays[:, 0] + np.sin(root) / root * rays[:, 1] / (1 + delta)
    expected_px = -(1 + delta) * root * np.sin(root) * rays[:, 0] + np.cos(root) * rays[:, 1]
    np.testing.assert_allclose(final.X.values, expected_x, rtol=2e-10, atol=1e-16)
    np.testing.assert_allclose(final.PX.values, expected_px, rtol=2e-10, atol=1e-16)
    report["thick_quadrupole"] = {"delta": delta.tolist(), "expected_x": expected_x.tolist(),
        "actual_x": final.X.tolist(), "expected_px": expected_px.tolist(), "actual_px": final.PX.tolist(),
        "max_absolute_x_error": float(np.max(abs(final.X.values-expected_x))),
        "max_absolute_px_error": float(np.max(abs(final.PX.values-expected_px))), "passed": True}

    # Deliberately exceed only a finite transverse numerical guard. These are
    # valid input rays and a known map; check surviving + lost particle counts.
    loss_rays = np.zeros((3, 6))
    loss_rays[:, 0] = [1e-6, 0.4, -0.4]
    loss_map = ("m:matrix,l=0,rm11=3,rm22=1,rm33=1,rm44=1,rm55=1,rm66=1;\n"
                "line1:line=(m);\nuse,period=line1;\n")
    frame = track_example(output / "loss_accounting", loss_map, loss_rays, momentum, "m")
    survivors = int((frame.segment == frame.segment.max()).sum())
    losses = frame.attrs["recorded_losses"]
    if survivors != 1 or losses != 2 or survivors + losses != len(loss_rays):
        raise AssertionError("Native numerical-loss accounting is inconsistent")
    report["loss_accounting"] = {"initial": 3, "survivors": survivors, "losses": losses, "passed": True}

    actual = output / "actual_lattice"
    actual.mkdir()
    config = dict(config, backend="madx")
    snapshot, hashes = sim.snapshot_sources(config, actual)
    sim._initialize_worker(config, str(snapshot), str(actual))
    settings = sim.build_quad_sobol_queue(config, 1)[0]
    row, stats = sim.run_single(settings)
    if row is None:
        raise AssertionError("Actual lattice tracking failed: " + stats["error"])
    if len(row["emittance_x"]) != 8:
        raise AssertionError("Historical eight-segment output schema changed")
    report["actual_lattice"] = {"settings": settings, "diagnostics": stats,
                                 "beam_outputs": row, "source_sha256": hashes, "passed": True}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def validate_beam_input_effects(config, output):
    """Paired seeds isolate alpha/spread effects at one fixed quadrupole setting."""
    output = Path(output).resolve()
    if (sim.REPO_ROOT / "results").resolve() not in output.parents:
        raise ValueError("Validation output must be inside repo results/")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(config, backend="madx")
    snapshot, hashes = sim.snapshot_sources(config, output)
    with Madx(stdout=False) as madx:
        madx.input("beam,particle=antiproton;")
        mass = float(madx.beam.mass)
    kinetic = config["beam"]["kinetic_energy_gev"]
    beta0 = np.sqrt(kinetic * (kinetic + 2 * mass)) / (kinetic + mass)
    nominal = dict(zip(config["quad_keys"], [60.0, -40.0, 40.0, -20.0]))
    nominal.update(alfx=config["beam"]["alfx"], alfy=config["beam"]["alfy"], sigma_delta=0.0, index=0)
    cases = {"nominal_zero_spread": nominal,
             "alpha_minus_20pct": dict(nominal, alfx=0.8*nominal["alfx"], alfy=0.8*nominal["alfy"]),
             "alpha_plus_20pct": dict(nominal, alfx=1.2*nominal["alfx"], alfy=1.2*nominal["alfy"]),
             "sigma_delta_0p001": dict(nominal, sigma_delta=0.001)}
    runs, beams, records = {}, {}, []
    for name, settings in cases.items():
        work = output / name
        work.mkdir()
        sim._initialize_worker(config, str(snapshot), str(work))
        rays, initial_stats = sim.generate_particles(config, settings, beta0)
        beams[name] = rays
        row, diagnostics = sim.run_single(settings)
        if row is None:
            raise AssertionError("{} failed: {}".format(name, diagnostics["error"]))
        if diagnostics["nonfinite_rows"] or diagnostics["lost_particles"]:
            raise AssertionError("Controlled nominal case has numerical losses: " + name)
        runs[name] = {"settings": settings, "diagnostics": diagnostics, "outputs": row}
        record = dict(scenario=name, **settings, **initial_stats)
        for axis, offset in (("x", 0), ("y", 2)):
            record["initial_sigma_"+axis] = float(rays[:, offset].std())
            record["initial_cov_"+axis+"_p"+axis] = float(np.cov(rays[:, offset:offset+2], rowvar=False, bias=True)[0, 1])
            for quantity in ("sigma_", "emittance_"):
                record["final_"+quantity+axis] = row[quantity+axis][-1]
        record["transmission"] = row["transmission"][-1]
        records.append(record)

    baseline = beams["nominal_zero_spread"]
    checks = {}
    for name in ("alpha_minus_20pct", "alpha_plus_20pct"):
        ray = beams[name]
        for axis, offset in (("x", 0), ("y", 2)):
            # Same random particles: an alpha change is exactly a shear in px.
            np.testing.assert_array_equal(ray[:, offset], baseline[:, offset])
            delta_alpha = cases[name]["alf"+axis] - nominal["alf"+axis]
            expected_p = baseline[:, offset+1] - delta_alpha / config["beam"]["beta"+axis] * baseline[:, offset]
            np.testing.assert_allclose(ray[:, offset+1], expected_p, rtol=1e-12, atol=1e-17)
            np.testing.assert_allclose(runs[name]["diagnostics"]["realized_gem"+axis],
                                       runs["nominal_zero_spread"]["diagnostics"]["realized_gem"+axis], rtol=1e-12)
        checks[name+"_entrance_shear"] = True
    dispersed = beams["sigma_delta_0p001"]
    expected_transverse = baseline[:, :4] + dispersed[:, 5, None] * np.asarray(config["beam"]["dispersion_pt"])
    np.testing.assert_allclose(dispersed[:, :4], expected_transverse, rtol=1e-13, atol=1e-17)
    realized_spread = runs["sigma_delta_0p001"]["diagnostics"]["realized_sigma_delta"]
    if not np.isclose(realized_spread, 0.001, rtol=0.04):
        raise AssertionError("Generated spread disagrees with intended Gaussian RMS")
    checks["spread_conversion_and_dispersion"] = True
    for name in cases:
        if name == "nominal_zero_spread":
            continue
        changes = {key: runs[name]["outputs"][key][-1] / runs["nominal_zero_spread"]["outputs"][key][-1] - 1
                   for key in ("sigma_x", "sigma_y", "emittance_x", "emittance_y")}
        if max(abs(value) for value in changes.values()) < 1e-6:
            raise AssertionError("Input change produced no detectable downstream response: " + name)
        runs[name]["final_relative_change_from_nominal"] = changes
        checks[name+"_downstream_response"] = True
    pd.DataFrame(records).to_csv(output / "comparison.csv", index=False)
    report = {"passed": True, "checks": checks, "particles_per_case": config["particles"],
              "seed": config["seed"], "source_sha256": hashes, "cases": runs,
              "interpretation": "Paired numerical sensitivity checks at fixed quadrupoles; not validation against measured beam or a claim of operational tuning performance."}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=sim.REPO_ROOT / "simulation_config.yaml")
    parser.add_argument("--output", type=Path, default=sim.REPO_ROOT / "results/validation/native_tracking")
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--beam-input-effects", action="store_true",
                        help="Run four paired alpha/spread cases; use --particles 6000 for the pilot check")
    args = parser.parse_args()
    config = sim.load_config(args.config)
    config["particles"] = args.particles
    if args.beam_input_effects:
        report = validate_beam_input_effects(config, args.output)
        print(json.dumps(report["checks"], indent=2))
    else:
        report = validate(config, args.output)
        print(json.dumps({key: value["passed"] for key, value in report.items()
                          if isinstance(value, dict) and "passed" in value}, indent=2))


if __name__ == "__main__":
    main()
