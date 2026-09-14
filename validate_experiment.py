"""Audit completed simulation datasets without rerunning tracking or training.

Examples:
  python validate_experiment.py results/experiment_20260908_12000 \
      --expected-samples 12000 --expected-reused 5000
  python validate_experiment.py results/experiment_20260908_12000/variability \
      --expected-samples 1536 --expected-anchors 64 --expected-replicates 24

Writes validation_summary.json and column_summary.csv beside completed data.
No preprocessing, model predictions or previously calculated bands are used.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sim_data_gen as sim
from simulation_variability import make_repeat_queue


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_summary(values, column, kind, segment_index=None, segment_name=None):
    values = np.asarray(values, dtype=float)
    return dict(column=column, kind=kind, segment_index=segment_index, segment_name=segment_name,
        count=len(values), minimum=float(values.min()), maximum=float(values.max()),
        mean=float(values.mean()), median=float(np.median(values)), std=float(values.std(ddof=1)),
        p01=float(np.quantile(values, .01)), p05=float(np.quantile(values, .05)),
        p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
        zero_count=int((values == 0).sum()), nonfinite_count=int((~np.isfinite(values)).sum()))


def audit(directory, expected_samples=None, expected_reused=None,
          expected_anchors=None, expected_replicates=None):
    directory = Path(directory)
    directory = (sim.REPO_ROOT / directory).resolve() if not directory.is_absolute() else directory.resolve()
    require((sim.REPO_ROOT / "results").resolve() in directory.parents,
            "Audit output must be inside repository results/")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Do this before reading either CSV: partial records are not audit evidence.
    require(metadata.get("status") == "complete" and metadata.get("failed") == 0,
            "Simulation metadata is running, incomplete or contains failures; audit deferred")
    config = metadata["configuration"]
    count = int(metadata["requested"])
    particles = int(config["particles"])
    require(count > 0 and metadata["completed"] == count, "Completed/requested counts disagree")
    if expected_samples is not None:
        require(count == expected_samples, "Dataset size differs from the explicitly expected count")
    require(particles == 6000, "This experiment audit expects 6,000 particles per setting")
    data_path = directory / Path(config["output"]).name
    diagnostics_path = directory / "diagnostics.csv"
    for key, path in (("data_sha256", data_path), ("diagnostics_sha256", diagnostics_path)):
        require(metadata.get(key) == sim.file_sha256(path), "Completed CSV checksum mismatch: " + str(path))
    require(metadata["physics_implementation_sha256"] == sim.physics_implementation_sha256(),
            "Current simulation-function fingerprint differs from the completed run")
    require(not (directory / "failures.jsonl").read_text().strip(), "Failure log is not empty")
    frame = pd.read_csv(data_path, float_precision="round_trip")
    diagnostics = pd.read_csv(diagnostics_path, float_precision="round_trip", keep_default_na=False)
    require(len(frame) == count and len(diagnostics) == count, "CSV counts differ from completed metadata")
    inputs = config["quad_keys"] + ["alfx", "alfy", "sigma_delta"]
    require(metadata["input_columns"] == inputs, "Input feature schema changed")
    require(metadata["output_columns"] == sim.OUTPUT_KEYS, "Beam output schema changed")
    require(len(config["expected_segments"]) == 8, "Expected observation count is not eight")

    study = metadata.get("variability_study")
    if study:
        anchors, replicates = int(study["anchors"]), int(study["replicates"])
        require(count == anchors * replicates, "Study size differs from anchors times replicates")
        require(expected_anchors is None or anchors == expected_anchors, "Anchor count differs from expected")
        require(expected_replicates is None or replicates == expected_replicates, "Replicate count differs from expected")
        require(config["seed"] == study["particle_seed"], "Repeat particle seed differs from metadata")
        require(config["seed"] != study["independent_from_production_seed"], "Repeat and production seed namespaces coincide")
        require(study["fixed_within_anchor"] == inputs, "Repeated-study fixed input schema differs")
        queue = make_repeat_queue(config, anchors, replicates, study["anchor_seed"])
    else:
        require(expected_anchors is None and expected_replicates is None,
                "Anchor/replicate expectations supplied for a non-repeat dataset")
        require(metadata.get("sampling") == "sobol", "Main dataset is not identified as a Sobol sample")
        require(config["seed"] == 42, "Production seed differs from the approved seed 42")
        queue = sim.build_quad_sobol_queue(config, count)
    expected = pd.DataFrame(queue)
    require(list(frame.columns) == list(queue[0]) + sim.OUTPUT_KEYS, "Data columns differ from the expected schema")
    require(list(diagnostics.columns) == sim.diagnostic_columns(queue[0]), "Diagnostic columns differ from expected schema")
    for label, table in (("data", frame), ("diagnostics", diagnostics)):
        require(np.array_equal(table[expected.columns].to_numpy(), expected.to_numpy()),
                label + " inputs or seed indices differ from the exact deterministic queue")
    require(frame["index"].is_unique and np.array_equal(frame["index"], np.arange(count)),
            "Simulation seed indices are missing, duplicated or out of order")
    require((diagnostics.status == "ok").all() and (diagnostics.error == "").all(),
            "Non-success statuses or error text found in diagnostics")
    require((diagnostics.nonfinite_rows == 0).all(), "Nonfinite tracked-particle rows were reported")
    require((diagnostics.madx_version == metadata["madx_version"]).all(), "Mixed MAD-X runtime versions")
    for names in diagnostics.actual_segment_names:
        actual = json.loads(names)
        normalized = ["end" if name.lower() == "lne00lne01lne02$end" else name.lower() for name in actual]
        require(normalized == [name.lower() for name in config["expected_segments"]],
                "Unexpected observation labels or order in diagnostics")
    require(all(json.loads(value) == {} for value in diagnostics.nonfinite_segments),
            "A diagnostic reports a nonfinite observation segment")

    columns, arrays = [], {}
    for key in inputs:
        require(np.isfinite(frame[key]).all(), "Nonfinite physical input: " + key)
        columns.append(numeric_summary(frame[key], key, "input"))
    for key in sim.OUTPUT_KEYS:
        values = np.asarray([ast.literal_eval(value) for value in frame[key]], dtype=float)
        require(values.shape == (count, 8) and np.isfinite(values).all(),
                "Malformed or nonfinite eight-segment output: " + key)
        if key.startswith(("sigma_", "emittance_")):
            require((values > 0).all(), "Nonpositive beam size/emittance: " + key)
        arrays[key] = values
        for segment, name in enumerate(config["expected_segments"]):
            columns.append(numeric_summary(values[:, segment], key, "output", segment, name))
    transmission = arrays["transmission"]
    require(((transmission >= 0) & (transmission <= 1)).all(), "Transmission lies outside [0,1]")
    require((transmission[:, 0] == 1).all(), "Entrance particle normalization differs from one")
    require((np.diff(transmission, axis=1) <= 1e-14).all(), "Transmission increases along the line")
    require(np.allclose(transmission*particles, np.rint(transmission*particles), rtol=0, atol=1e-9),
            "Transmission does not correspond to integer particle counts")
    for key in ("final_particles", "lost_particles", "finite_lost_particles", "nonfinite_lost_particles"):
        value = diagnostics[key].to_numpy(dtype=float)
        require(np.isfinite(value).all() and (value >= 0).all() and (value == np.rint(value)).all(),
                "Invalid particle count in diagnostic " + key)
    require((diagnostics.final_particles + diagnostics.lost_particles == particles).all(),
            "Survivors plus losses do not equal 6,000")
    require((diagnostics.finite_lost_particles + diagnostics.nonfinite_lost_particles == diagnostics.lost_particles).all(),
            "Finite and nonfinite loss classifications do not sum to total losses")
    require(np.allclose(transmission[:, -1], diagnostics.final_particles/particles, rtol=0, atol=1e-15),
            "Endpoint transmission disagrees with surviving-particle diagnostics")
    for key in ("realized_sigma_delta", "realized_mean_delta", "realized_sigma_pt", "realized_alfx",
                "realized_alfy", "realized_gemx", "realized_gemy", "tracked_initial_sigma_delta",
                "seconds", "lost_particles", "nonfinite_lost_particles"):
        require(np.isfinite(diagnostics[key]).all(), "Nonfinite input/diagnostic statistic: " + key)
        columns.append(numeric_summary(diagnostics[key], key, "diagnostic"))
    require((diagnostics.realized_gemx > 0).all() and (diagnostics.realized_gemy > 0).all(),
            "Invalid realized entrance geometric emittance")
    require(np.allclose(diagnostics.realized_sigma_delta, diagnostics.tracked_initial_sigma_delta, rtol=1e-8, atol=1e-14),
            "Tracked entrance momentum spread differs from generated spread")

    expected_sources = [Path(config["lattice_root"]) / name for name in sim.lattice_files()]
    expected_sources += [Path(config[key]) for key in ("external_runner", "initial_conditions")]
    require(len(expected_sources) == 12 and set(metadata["source_sha256"]) == {str(path) for path in expected_sources},
            "Unexpected external source-file provenance set")
    for source in expected_sources:
        require(sim.file_sha256(source) == metadata["source_sha256"][str(source)],
                "External source changed after simulation: " + str(source))
        if source in expected_sources[:10]:
            archived = directory / "source_lattice" / source.relative_to(config["lattice_root"])
        else:
            archived = directory / ("original_" + source.name)
        require(sim.file_sha256(archived) == metadata["source_sha256"][str(source)],
                "Archived source snapshot checksum differs: " + str(archived))
    require(sim.file_sha256(directory / "runner_template.madx") == metadata["runner_template_sha256"],
            "Archived generated runner checksum differs")

    reused = int(metadata.get("reused", 0))
    require(expected_reused is None or reused == expected_reused, "Reused-record count differs from expected")
    require(reused + metadata.get("generated_this_run", count) == count,
            "New plus reused simulation counts differ from total")
    reuse_report = {"count": reused, "data_prefix_byte_equal": None, "diagnostic_prefix_byte_equal": None}
    if reused:
        reuse = metadata["reuse"]
        require(reuse["count"] == reused, "Reuse metadata counts disagree")
        for key, path_key, current_path in (("data_sha256", "data_path", data_path),
                                             ("diagnostics_sha256", "diagnostics_path", diagnostics_path)):
            previous_path = Path(reuse[path_key])
            require(sim.file_sha256(previous_path) == reuse[key], "Previously reused CSV has changed")
            require(current_path.read_bytes().startswith(previous_path.read_bytes()),
                    "Reused CSV prefix is not byte-for-byte identical: " + path_key)
            old = pd.read_csv(previous_path)
            require(len(old) == reused, "Reused source CSV count changed")
        require(sim.file_sha256(reuse["metadata_path"]) == reuse["metadata_sha256"], "Prior source metadata changed")
        require(sim.file_sha256(directory / "reuse_source_metadata.json") == reuse["metadata_sha256"],
                "Archived prior-run metadata differs")
        reuse_report.update(source_directory=reuse["directory"], data_prefix_byte_equal=True,
                            diagnostic_prefix_byte_equal=True, previous_inputs_retained=True)

    repeat_report = None
    if study:
        require(not reused, "Repeated simulations unexpectedly reuse production rows")
        groups = frame.groupby("anchor_id")
        require(len(groups) == anchors, "Wrong number of repeat anchor groups")
        for anchor_id, group in groups:
            require(len(group) == replicates and np.array_equal(np.sort(group.replicate_id), np.arange(replicates)),
                    "Missing or duplicate replicates at anchor " + str(anchor_id))
            require((group[inputs].nunique() == 1).all(), "A physical input changes within an anchor")
            random_stats = diagnostics.loc[group.index, ["realized_sigma_delta", "realized_alfx", "realized_alfy"]]
            require(len(random_stats.drop_duplicates()) == replicates, "Particle realizations unexpectedly repeat within an anchor")
        reference_metadata_path = Path(study["reference_run"]) / "metadata.json"
        require(sim.file_sha256(reference_metadata_path) == study["reference_metadata_sha256"],
                "Repeat-study reference provenance changed")
        reference = json.loads(reference_metadata_path.read_text())
        for key in ("particles", "backend", "quad_keys", "quad_range", "alpha_fraction", "sigma_delta_range", "beam", "observations", "expected_segments"):
            require(config[key] == reference["configuration"][key], "Repeat-study physics differs from production: " + key)
        require(metadata["source_sha256"] == reference["source_sha256"], "Repeat lattice sources differ from production")
        repeat_report = dict(anchors=anchors, replicates_per_anchor=replicates,
            fixed_physical_inputs=inputs, all_groups_fixed=True,
            particle_seed=study["particle_seed"], anchor_seed=study["anchor_seed"],
            production_seed=study["independent_from_production_seed"],
            unique_particle_seed_indices=count, unique_realization_statistics_verified=True,
            preprocessing_used=False)

    worst = diagnostics.sort_values("lost_particles", ascending=False).head(10)
    loss_summary = dict(runs_with_losses=int((diagnostics.lost_particles > 0).sum()),
        fraction_of_runs_with_losses=float((diagnostics.lost_particles > 0).mean()),
        total_lost_particles=int(diagnostics.lost_particles.sum()),
        total_nonfinite_lost_particles=int(diagnostics.nonfinite_lost_particles.sum()),
        maximum_losses_per_run=int(diagnostics.lost_particles.max()),
        minimum_endpoint_transmission=float(transmission[:, -1].min()),
        nonfinite_surviving_coordinate_rows=int(diagnostics.nonfinite_rows.sum()),
        worst_cases=worst[["index", "lost_particles", "nonfinite_lost_particles", "final_particles"]].to_dict(orient="records"),
        interpretation="Numerical-domain/broad-guard losses, not calibrated hardware or experimental transmission losses.")
    summary = dict(status="passed", dataset_kind="fixed_input_repeats" if study else "sobol_extension",
        requested=count, data_rows=len(frame), diagnostic_rows=len(diagnostics),
        particles_per_simulation=particles, observation_segments=config["expected_segments"],
        checks=dict(metadata_complete=True, data_and_diagnostic_checksums=True, schema=True,
                    deterministic_input_queue=True, finite_positive_sizes_and_emittances=True,
                    all_eight_segments=True, particle_loss_conservation=True, all_12_source_hashes_unchanged=True,
                    archived_source_hashes=True, generator_physics_fingerprint=True),
        reuse=reuse_report, repeat_study=repeat_report, numerical_losses=loss_summary,
        data_sha256=sim.file_sha256(data_path), diagnostics_sha256=sim.file_sha256(diagnostics_path),
        source_sha256=metadata["source_sha256"],
        generator_sha256=metadata["generator_sha256"], physics_implementation_sha256=metadata["physics_implementation_sha256"],
        madx_version=metadata["madx_version"], column_summary="column_summary.csv",
        scope="Data integrity, deterministic sampling, numerical finiteness and provenance. Does not establish historical-backend parity or agreement with measured beam data.")
    pd.DataFrame(columns).to_csv(directory / "column_summary.csv", index=False)
    (directory / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-reused", type=int)
    parser.add_argument("--expected-anchors", type=int)
    parser.add_argument("--expected-replicates", type=int)
    args = parser.parse_args()
    result = audit(args.directory, args.expected_samples, args.expected_reused,
                   args.expected_anchors, args.expected_replicates)
    print(json.dumps({key: result[key] for key in ("status", "dataset_kind", "data_rows", "particles_per_simulation")}, indent=2))


if __name__ == "__main__":
    main()
