"""Measure finite-particle simulation variability with seven fixed inputs.

Generate independent repeated beams at representative settings:
  python simulation_variability.py --reference-run results/pilot_20260907 \
      --output results/experiment_20260908_12000/variability/data.csv

After training, express the same repeats in the saved training coordinates:
  python simulation_variability.py --summarize-only RESULTS/variability/data.csv \
      --preprocessing RESULTS/training/preprocessing.json

Observed replicate ranges and quantiles are descriptive Monte Carlo repeatability
estimates, not confidence intervals or universal bounds on simulation/model error.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sim_data_gen as sim


TARGETS = ["emittance_x", "emittance_y"]


def make_repeat_queue(config, anchors=64, replicates=24, anchor_seed=20260909):
    if anchors < 1 or replicates < 2:
        raise ValueError("Require at least one anchor and two replicates")
    anchor_settings = sim.build_quad_sobol_queue(config, anchors, seed=anchor_seed)
    queue = []
    for anchor_id, anchor in enumerate(anchor_settings):
        for replicate_id in range(replicates):
            settings = dict(anchor)
            settings.update(index=len(queue), anchor_id=anchor_id, replicate_id=replicate_id)
            queue.append(settings)
    return queue


def checked_results_path(path):
    path = Path(path)
    path = (sim.REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if (sim.REPO_ROOT / "results").resolve() not in path.parents:
        raise ValueError("Variability artifacts must stay inside repository results/")
    return path


def summarize(data_path, preprocessing_path=None, segment=7):
    data_path = checked_results_path(data_path)
    directory = data_path.parent
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["failed"]:
        raise ValueError("Variability run is incomplete or contains failed simulations; inspect its failure log")
    if metadata.get("data_sha256") and sim.file_sha256(data_path) != metadata["data_sha256"]:
        raise ValueError("Variability data checksum differs from completed-run metadata")
    study = metadata["variability_study"]
    frame = pd.read_csv(data_path, float_precision="round_trip")
    if len(frame) != study["anchors"] * study["replicates"]:
        raise ValueError("Repeat row count disagrees with the study design")
    preprocessing = None
    if preprocessing_path:
        preprocessing_path = Path(preprocessing_path).resolve()
        preprocessing = json.loads(preprocessing_path.read_text())
        if preprocessing["target_transform"] != "natural_log_then_minmax":
            raise ValueError("Unsupported preprocessing transform")
        if preprocessing["target_cols"] != TARGETS or preprocessing["input_cols"] != metadata["input_columns"]:
            raise ValueError("Preprocessing columns do not match the variability study")
        segment = int(preprocessing["segment"])
    if not 0 <= segment < 8:
        raise ValueError("segment must lie in [0, 7]")
    values = {}
    for target in TARGETS:
        lists = [ast.literal_eval(value) for value in frame[target]]
        arrays = np.asarray(lists, dtype=float)
        if arrays.shape != (len(frame), 8) or not np.isfinite(arrays).all() or not (arrays > 0).all():
            raise ValueError("Invalid emittance output for " + target)
        values[target] = arrays[:, segment]
    physical = pd.DataFrame(values, index=frame.index)
    spaces = {"physical": physical, "log": np.log(physical)}
    if preprocessing:
        scales = np.asarray(preprocessing["target_log_range"], dtype=float)
        offsets = np.asarray(preprocessing["target_log_min"], dtype=float)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("Invalid saved target scales")
        spaces["scaled"] = (np.log(physical) - offsets) / scales
    inputs = metadata["input_columns"]
    by_anchor = frame.groupby("anchor_id", sort=True)
    for anchor_id, group in by_anchor:
        if len(group) != study["replicates"] or group.replicate_id.nunique() != study["replicates"]:
            raise ValueError("Missing or duplicated replicates for anchor " + str(anchor_id))
        if (group[inputs].nunique() != 1).any():
            raise ValueError("A beam input changes within anchor " + str(anchor_id))
    if len(by_anchor) != study["anchors"]:
        raise ValueError("Anchor count differs from the study design")

    records, global_spaces = [], {}
    for space, table in spaces.items():
        global_spaces[space] = {}
        for target in TARGETS:
            target_records, residuals = [], []
            for anchor_id, group in by_anchor:
                y = table.loc[group.index, target].to_numpy(dtype=float)
                centered = y - y.mean()
                record = dict(space=space, target=target, anchor_id=int(anchor_id),
                    replicates=len(y), mean=float(y.mean()), std=float(y.std(ddof=1)),
                    minimum=float(y.min()), maximum=float(y.max()), observed_range=float(np.ptp(y)),
                    half_observed_range=float(np.ptp(y)/2), q025=float(np.quantile(y, .025)),
                    q975=float(np.quantile(y, .975)),
                    max_absolute_centered_deviation=float(np.max(np.abs(centered))))
                record.update({key: float(group.iloc[0][key]) for key in inputs})
                records.append(record)
                target_records.append(record)
                residuals.extend(centered.tolist())
            largest = max(target_records, key=lambda record: record["observed_range"])
            residuals = np.asarray(residuals)
            global_spaces[space][target] = dict(
                max_observed_range=largest["observed_range"],
                half_max_observed_range=largest["half_observed_range"],
                max_range_anchor_id=largest["anchor_id"],
                max_absolute_centered_deviation=float(np.max(np.abs(residuals))),
                pooled_centered_q025=float(np.quantile(residuals, .025)),
                pooled_centered_q975=float(np.quantile(residuals, .975)),
                pooled_centered_rms=float(np.sqrt(np.mean(residuals**2))),
                median_anchor_std=float(np.median([row["std"] for row in target_records])))
    pd.DataFrame(records).to_csv(directory / "anchor_variability.csv", index=False)
    diagnostics = pd.read_csv(directory / "diagnostics.csv")
    summary = dict(status="complete", anchors=study["anchors"], replicates_per_anchor=study["replicates"],
        particle_count=metadata["configuration"]["particles"], segment=segment,
        input_columns=inputs, target_columns=TARGETS, spaces=global_spaces,
        data_path=str(data_path), data_sha256=sim.file_sha256(data_path),
        metadata_sha256=sim.file_sha256(directory / "metadata.json"),
        definitions={
            "half_max_observed_range": "Half of the largest observed within-anchor max-minus-min span; matches the old notebook band convention, not a universal bound or a 95% interval.",
            "max_absolute_centered_deviation": "Largest absolute replicate deviation from its anchor's replicate mean among observed runs.",
            "pooled_centered_q025_q975": "Empirical central 95% quantiles of replicate deviations from each anchor's sample mean, pooled equally over anchors; descriptive repeatability, not a confidence interval.",
            "std": "Sample standard deviation across independent Gaussian particle realizations at fixed seven inputs (ddof=1)."},
        limitations="Finite anchor/replicate study of Monte Carlo variation under the current simulator. Bounds depend on sample count and input setting. Numerical tail losses can affect survivor-emittance statistics. No measured beam, machine drift, space-charge uncertainty or ML training variability is included.",
        numerical_losses=dict(runs_with_losses=int((diagnostics.lost_particles > 0).sum()),
                              lost_particles=int(diagnostics.lost_particles.sum()),
                              nonfinite_lost_particles=int(diagnostics.nonfinite_lost_particles.sum())))
    if preprocessing:
        summary["preprocessing"] = dict(path=str(preprocessing_path), sha256=sim.file_sha256(preprocessing_path),
            target_log_min=preprocessing["target_log_min"], target_log_range=preprocessing["target_log_range"],
            fit_rows=preprocessing["fit_rows"], fitted_on_repeats=False, clipping=False)
    (directory / "variability_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def run_study(reference_dir, output, anchors=64, replicates=24, seed=20260908,
              anchor_seed=20260909, workers=2, particles=6000):
    reference_dir = Path(reference_dir).resolve()
    reference_metadata = json.loads((reference_dir / "metadata.json").read_text())
    base = reference_metadata["configuration"]
    # Reuse validation checks reference simulation compatibility; no rows reused.
    sim.validate_extension(base, reference_dir,
        sim.build_quad_sobol_queue(base, reference_metadata["requested"] + 1))
    if seed == base["seed"]:
        raise ValueError("Particle-repeat seed must differ from production seed")
    if particles != base["particles"]:
        raise ValueError("Repeat particle count must match the reference dataset")
    output = checked_results_path(output)
    config = dict(base, samples=anchors*replicates, seed=seed, workers=workers,
                  output=str(output), particles=particles)
    queue = make_repeat_queue(config, anchors, replicates, anchor_seed)
    study = dict(anchors=anchors, replicates=replicates, particle_seed=seed, anchor_seed=anchor_seed,
        anchor_design="Independent scrambled Sobol points across the same seven-dimensional input ranges",
        particle_seed_rule="numpy SeedSequence([particle_seed, index]); index=anchor_id*replicates+replicate_id",
        fixed_within_anchor=config["quad_keys"]+["alfx", "alfy", "sigma_delta"],
        independent_from_production_seed=base["seed"],
        reference_run=str(reference_dir), reference_metadata_sha256=sim.file_sha256(reference_dir / "metadata.json"))
    metadata = sim.run_batch(config, queue=queue, metadata_extra={"variability_study": study})
    pd.DataFrame([row for row in queue if row["replicate_id"] == 0]).drop(columns=["index", "replicate_id"]).to_csv(output.parent / "anchors.csv", index=False)
    if metadata["failed"]:
        raise RuntimeError("Some repeat simulations failed; raw diagnostics retained, no variability band calculated")
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=sim.REPO_ROOT / "results/pilot_20260907")
    parser.add_argument("--output", type=Path, default=sim.REPO_ROOT / "results/experiment_20260908_12000/variability/data.csv")
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--anchor-seed", type=int, default=20260909)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--particles", type=int, default=6000)
    parser.add_argument("--summarize-only", type=Path)
    parser.add_argument("--preprocessing", type=Path)
    args = parser.parse_args()
    if args.summarize_only:
        summary = summarize(args.summarize_only, args.preprocessing)
    else:
        summary = run_study(args.reference_run, args.output, args.anchors, args.replicates,
                            args.seed, args.anchor_seed, args.workers, args.particles)
        if args.preprocessing:
            summary = summarize(args.output, args.preprocessing)
    print(json.dumps({"anchors": summary["anchors"], "replicates_per_anchor": summary["replicates_per_anchor"],
                      "spaces": list(summary["spaces"])}, indent=2))


if __name__ == "__main__":
    main()
