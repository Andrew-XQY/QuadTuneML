"""Standalone data checks and publication figures; never trains or runs MAD-X.

python analyze.py --data results/pilot_20260907/data.csv \
    --training-dir results/pilot_20260907/training \
    --output-dir results/pilot_20260907/figures

PDFs retain vector graphics, embedded TrueType fonts and transparent backgrounds.
Normalized errors are percentage points of the training log-emittance range,
not relative physical-emittance errors. No predictions or axes are clipped to [0, 1].
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter

from APS_PLOT_STYLE import APS_COLORS, set_aps_double_column

BLUE, ORANGE, GREEN = APS_COLORS[4], APS_COLORS[0], APS_COLORS[2]
BEAM_COLUMNS = ["mean_x", "mean_y", "sigma_x", "sigma_y", "emittance_x", "emittance_y", "transmission"]
LABELS = {
    "alfx": r"$\alpha_x$", "alfy": r"$\alpha_y$",
    "sigma_delta": r"$\sigma_\delta = \sigma_p/p_0$",
    "emittance_x": r"$\varepsilon_x$", "emittance_y": r"$\varepsilon_y$",
    "sigma_x": r"$\sigma_x$", "sigma_y": r"$\sigma_y$",
    "klne.zqmd.0208": "ZQMD 0208", "klne.zqmf.0209": "ZQMF 0209",
    "klne.zqmd.0214": "ZQMD 0214", "klne.zqmf.0215": "ZQMF 0215",
}


def configure_style():
    set_aps_double_column(scale=1.05, legend_background=False)
    plt.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def new_figure(nrows=1, ncols=2, figsize=(6.8, 3.15), **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kwargs)
    for ax in np.atleast_1d(axes).flat:
        ax.tick_params(top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fig, axes


def panel(ax, letter, label=""):
    ax.text(0.0, 1.025, f"({letter}) {label}", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9)


def save_figure(fig, output_dir, name, caption, manifest):
    path = output_dir / f"{name}.pdf"
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04,
                metadata={"Creator": "QuadTuneML analyze.py", "Title": caption.split(".")[0]})
    plt.close(fig)
    manifest.append({"file": path.name, "caption": caption})


def load_data(path, input_cols, segment):
    """Read legacy list-valued simulation outputs and report every segment."""
    data = pd.read_csv(path)
    missing = set(input_cols + BEAM_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing simulation columns: {sorted(missing)}")
    selected = data[input_cols].apply(pd.to_numeric, errors="raise").copy()
    arrays = {}
    issues = []
    summary = []
    for col in input_cols:
        if not np.isfinite(selected[col]).all():
            issues.append(f"{col}: nonfinite input values")
    for col in BEAM_COLUMNS:
        parsed = []
        for row, cell in enumerate(data[col]):
            values = ast.literal_eval(cell) if isinstance(cell, str) else cell
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1 or len(arr) != 8:
                raise ValueError(f"Row {row}, {col}: expected eight segment values; got {arr.shape}")
            parsed.append(arr)
        arr = np.stack(parsed)
        arrays[col] = arr
        selected[col] = arr[:, segment]
        invalid = ~np.isfinite(arr)
        if invalid.any():
            issues.append(f"{col}: {int(invalid.sum())} nonfinite values across all segments")
        if col.startswith("emittance") and (arr <= 0).any():
            issues.append(f"{col}: {int((arr <= 0).sum())} nonpositive values across all segments")
        if col.startswith("sigma") and (arr < 0).any():
            issues.append(f"{col}: negative beam sizes")
        if col == "transmission" and ((arr < 0) | (arr > 1 + 1e-12)).any():
            issues.append("transmission: values outside [0, 1]")
    if "sigma_delta" in selected and (selected["sigma_delta"] < 0).any():
        issues.append("sigma_delta: negative momentum spreads")
    for col in selected.columns:
        values = selected[col].to_numpy()
        values = values[np.isfinite(values)]
        if len(values):
            q = np.quantile(values, [0, .05, .5, .95, 1])
            summary.append({"column": col, "count": len(values), "min": q[0],
                            "q05": q[1], "median": q[2], "q95": q[3], "max": q[4],
                            "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))})
    qa = {
        "data_file": str(path.resolve()), "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(data), "input_columns": input_cols, "input_dimension": len(input_cols),
        "observation_segments": 8, "selected_segment_zero_based": segment,
        "duplicate_input_rows": int(selected.duplicated(input_cols).sum()),
        "all_segment_transmission_min": float(np.min(arrays["transmission"])),
        "all_segment_transmission_max": float(np.max(arrays["transmission"])),
        "issues": issues, "checks_passed": not issues,
        "scope": "Numerical/data-contract checks; not validation against measured AEgIS beam data.",
    }
    return selected, qa, pd.DataFrame(summary)




def completed_simulation_metadata(data_path):
    """Do not read a running/failed batch as if it were a final dataset."""
    path = data_path.parent / "metadata.json"
    if not path.exists():
        return None
    metadata = read_json(path)
    if metadata.get("status") != "complete":
        raise ValueError(f"Simulation status is {metadata.get('status')!r}; final figures require a complete batch.")
    if metadata.get("failed", 0) != 0:
        raise ValueError("Simulation contains failed settings; final figures require zero failed settings.")
    if metadata.get("completed") != metadata.get("requested"):
        raise ValueError("Simulation completed/requested counts disagree.")
    return metadata


def distribution_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    q = np.quantile(values, [0, .05, .5, .95, 1])
    return dict(min=float(q[0]), q05=float(q[1]), median=float(q[2]), q95=float(q[3]),
                max=float(q[4]), mean=float(np.mean(values)), std=float(np.std(values)))


def simulation_diagnostics(data_path, input_cols, qa, metadata):
    """Cross-check full-batch accounting and the particle-generation/tracker boundary."""
    path = data_path.parent / "diagnostics.csv"
    report = {"metadata_available": metadata is not None, "diagnostics_available": path.exists()}
    qa["simulation"] = report
    issues = qa["issues"]
    if metadata is not None:
        fields = ["status", "requested", "completed", "failed", "backend", "backend_reason", "madx_version",
                  "hardware_aperture_model", "aperture_checks_enabled", "space_charge", "numerical_guards",
                  "loss_interpretation", "sigma_delta_definition", "pt_definition", "transverse_convention",
                  "source_sha256", "generator_sha256", "runner_template_sha256", "package_versions", "note"]
        report.update({key: metadata.get(key) for key in fields})
        report["metadata_sha256"] = hashlib.sha256((data_path.parent / "metadata.json").read_bytes()).hexdigest()
        if qa["rows"] != metadata["requested"]:
            issues.append("CSV row count differs from completed simulation request")
    if not path.exists():
        qa["checks_passed"] = not issues
        return
    diagnostic = pd.read_csv(path)
    report["diagnostics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["diagnostic_rows"] = len(diagnostic)
    raw = pd.read_csv(data_path, usecols=["index", "transmission"] + input_cols)
    if raw["index"].duplicated().any() or diagnostic["index"].duplicated().any():
        issues.append("Simulation indices are not unique")
    if set(raw["index"]) != set(diagnostic["index"]) or len(raw) != len(diagnostic):
        issues.append("Diagnostics and data contain different simulation indices")
    if not diagnostic["status"].eq("ok").all():
        issues.append("Diagnostics contain failed simulation settings")
    if issues:
        qa["checks_passed"] = False
        return
    diagnostic = diagnostic.set_index("index").loc[raw["index"]]
    if not np.allclose(raw[input_cols].to_numpy(), diagnostic[input_cols].to_numpy(), rtol=1e-10, atol=1e-14):
        issues.append("Diagnostic inputs do not match their simulation data rows")
    particles = metadata.get("configuration", {}).get("particles") if metadata else None
    report["particles_per_setting"] = particles
    if particles is not None:
        survivors = diagnostic["final_particles"].to_numpy()
        lost = diagnostic["lost_particles"].to_numpy()
        if not np.isfinite(np.r_[survivors, lost]).all() or (survivors < 0).any() or (lost < 0).any():
            issues.append("Invalid survivor/loss particle counts")
        if not np.all(survivors + lost == particles):
            issues.append("Final survivors plus recorded losses differ from injected particle count")
        final_transmission = np.array([ast.literal_eval(value)[-1] for value in raw["transmission"]])
        if not np.allclose(final_transmission, survivors / particles, rtol=1e-12, atol=1e-12):
            issues.append("Final transmission differs from final_particles / injected_particles")
        loss_report = dict(settings_with_losses=int((lost > 0).sum()), total_injected=int(particles * len(raw)),
                           total_survivors=int(survivors.sum()), total_lost=int(lost.sum()),
                           final_transmission=distribution_summary(final_transmission))
        for key in ["nonfinite_lost_particles", "finite_lost_particles", "nonfinite_rows"]:
            loss_report[key] = int(diagnostic[key].sum()) if key in diagnostic else None
        if {"nonfinite_lost_particles", "finite_lost_particles"}.issubset(diagnostic.columns):
            if not np.all(diagnostic["nonfinite_lost_particles"] + diagnostic["finite_lost_particles"] == lost):
                issues.append("Finite plus nonfinite loss counts differ from total recorded losses")
        if "nonfinite_rows" in diagnostic and not diagnostic["nonfinite_rows"].eq(0).all():
            issues.append("Nonfinite surviving TRACKONE rows remain")
        loss_report["interpretation"] = (metadata or {}).get("loss_interpretation",
            "Recorded tracking losses; physical hardware-aperture capture efficiency is not established.")
        report["losses"] = loss_report
    required = ["sigma_delta", "realized_sigma_delta", "tracked_initial_sigma_delta"]
    if set(required).issubset(diagnostic.columns):
        requested = diagnostic["sigma_delta"].to_numpy()
        realized = diagnostic["realized_sigma_delta"].to_numpy()
        tracked = diagnostic["tracked_initial_sigma_delta"].to_numpy()
        if not np.isfinite(np.r_[requested, realized, tracked]).all() or (np.r_[requested, realized, tracked] < 0).any():
            issues.append("Invalid requested/generated/tracked momentum-spread values")
        if not np.allclose(realized, tracked, rtol=1e-8, atol=1e-14):
            issues.append("TRACKONE initial momentum spread does not match generated particles")
        nonzero = requested > 0
        if np.any(realized[~nonzero] > 1e-14):
            issues.append("Nonzero generated momentum spread for a zero-spread request")
        report["momentum_spread"] = {
            "tracked_vs_generated_rtol": 1e-8, "tracked_vs_generated_atol": 1e-14,
            "max_tracked_generated_absolute_difference": float(np.max(np.abs(tracked-realized))),
            "realized_to_requested_rms_ratio": distribution_summary(realized[nonzero]/requested[nonzero]),
            "zero_spread_settings": int((~nonzero).sum()),
            "ratio_interpretation": "Finite-particle Gaussian RMS fluctuates around the requested ensemble spread.",
        }
    qa["checks_passed"] = not issues


def validate_predictions(predictions, data, preprocessing, metrics):
    """Reject mismatched data/training artifacts before drawing credible-looking plots."""
    rows = predictions["row_index"].to_numpy(dtype=int)
    if ((rows < 0) | (rows >= len(data))).any() or len(np.unique(rows)) != len(rows):
        raise ValueError("Test row indices must be unique positions in the supplied simulation CSV.")
    matched = data.iloc[rows]
    inputs = preprocessing["input_cols"]
    if not np.allclose(predictions[inputs], matched[inputs], rtol=1e-7, atol=1e-12):
        raise ValueError("Test inputs do not match the supplied simulation CSV.")
    for i, target in enumerate(preprocessing["target_cols"]):
        truth = predictions[f"true_{target}"].to_numpy()
        if not np.allclose(truth, matched[target], rtol=1e-7, atol=0):
            raise ValueError(f"{target}: test truth does not match the simulation observation.")
        expected_scaled = (np.log(truth) - preprocessing["target_log_min"][i]) / preprocessing["target_log_range"][i]
        scaled_truth = predictions[f"true_scaled_{target}"].to_numpy()
        scaled_prediction = predictions[f"pred_scaled_{target}"].to_numpy()
        if not np.allclose(expected_scaled, scaled_truth, rtol=1e-6, atol=1e-7):
            raise ValueError(f"{target}: scaling metadata does not reproduce test truth.")
        residual = scaled_prediction - scaled_truth
        variance = np.sum((scaled_truth - scaled_truth.mean()) ** 2)
        computed = {"MAE": np.mean(np.abs(residual)), "RMSE": np.sqrt(np.mean(residual**2)),
                    "R2": 1 - np.sum(residual**2) / variance if variance else np.nan}
        for name, value in computed.items():
            saved = metrics["scaled"][target][name]
            saved = np.nan if saved is None else saved
            if not np.isclose(value, saved, rtol=1e-5, atol=1e-7, equal_nan=True):
                raise ValueError(f"{target}: {name} disagrees with saved held-out predictions.")


def parity(pred, targets, metrics, output_dir, manifest):
    fig, axes = new_figure(ncols=len(targets))
    for i, (ax, target) in enumerate(zip(np.atleast_1d(axes), targets)):
        truth = pred[f"true_scaled_{target}"].to_numpy()
        estimate = pred[f"pred_scaled_{target}"].to_numpy()
        low, high = min(0., truth.min(), estimate.min()), max(1., truth.max(), estimate.max())
        padding = 0.025 * (high - low)
        limits = (low - padding, high + padding)
        ax.plot(limits, limits, "--", color=BLUE, lw=.9, label="Ideal")
        ax.scatter(truth, estimate, s=6, c=ORANGE, alpha=.72, edgecolors="none", label="Test samples")
        ax.set(xlim=limits, ylim=limits, aspect="equal")
        suffix = target.rsplit("_", 1)[-1]
        ax.set_xlabel(rf"Simulated $\varepsilon_{{{suffix},\mathrm{{scaled}}}}$")
        ax.set_ylabel(rf"Predicted $\varepsilon_{{{suffix},\mathrm{{scaled}}}}$")
        panel(ax, chr(97+i))
        metric = metrics["scaled"][target]
        r2_label = "undefined" if metric["R2"] is None else f"{metric['R2']:.3f}"
        ax.text(.97, .04, f"MAE = {metric['MAE']*100:.2f} pp\n" + rf"$R^2$ = {r2_label}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, .89), markerscale=1.3)
    save_figure(fig, output_dir, "parity_scaled",
                f"Held-out prediction parity ({len(pred):,} test settings). Emittances are natural-log transformed "
                "and scaled using training-set extrema. Points and axes are not clipped. "
                "MAE is in percentage points (pp) of the training log-emittance range, not a relative physical error. "
                "No simulation-variability band is shown because no matching repeated-run estimate is supplied.", manifest)


def residuals(pred, targets, output_dir, manifest):
    errors = {t: 100 * (pred[f"pred_scaled_{t}"] - pred[f"true_scaled_{t}"]) for t in targets}
    all_errors = np.concatenate([e.to_numpy() for e in errors.values()])
    bins = np.histogram_bin_edges(all_errors, bins=35)
    fig, axes = new_figure(ncols=len(targets))
    for i, (ax, target) in enumerate(zip(np.atleast_1d(axes), targets)):
        err = errors[target]
        ax.hist(err, bins=bins, color=ORANGE, alpha=.78, density=True, linewidth=.4, edgecolor="white")
        ax.axvline(0, color=".35", ls="--", lw=.9)
        ax.axvline(err.mean(), color=BLUE, lw=1.1)
        ax.set_xlabel("Predicted - simulated (pp)")
        ax.set_ylabel("Probability density (pp$^{-1}$)")
        panel(ax, chr(97+i), LABELS.get(target, target))
        ax.text(.97, .96, f"Bias = {err.mean():.2f} pp\nRMSE = {np.sqrt(np.mean(err**2)):.2f} pp",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    save_figure(fig, output_dir, "residuals_scaled",
                "Test-set residuals in percentage points of the train-fit normalized log-emittance coordinate. "
                "The dashed line is zero error and the blue line is mean bias. "
                "All test residuals are included, with identical histogram bins across targets.", manifest)


def training_curves(history, output_dir, manifest):
    fig, axes = new_figure()
    epoch = history["epoch"]
    for ax, column, ylabel, factor in zip(axes, ["loss", "mae"], ["Huber loss", "Scaled MAE (pp)"], [1, 100]):
        ax.plot(epoch, factor * history[column], color=ORANGE, label="Training", lw=1.2)
        ax.plot(epoch, factor * history[f"val_{column}"], color=BLUE, label="Validation", lw=1.2)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        if (history[column] > 0).all() and (history[f"val_{column}"] > 0).all():
            ax.set_yscale("log")
        ax.legend(loc="upper right")
    panel(axes[0], "a")
    panel(axes[1], "b")
    save_figure(fig, output_dir, "training_curves",
                "Training and validation learning curves. Loss and MAE are averages over the two normalized "
                "log-emittance outputs. These epoch curves describe the optimizer history; reported held-out "
                "metrics come from the selected best-validation checkpoint.", manifest)


def incoming_coverage(data, output_dir, manifest):
    fig, axes = new_figure(ncols=3, figsize=(6.8, 2.4))
    for i, (ax, col, color) in enumerate(zip(axes, ["alfx", "alfy", "sigma_delta"], [ORANGE, BLUE, GREEN])):
        values = data[col] * (1000 if col == "sigma_delta" else 1)
        ax.hist(values, bins=24, color=color, alpha=.78, edgecolor="white", linewidth=.45)
        ax.set_xlabel(LABELS[col] + (r" ($10^{-3}$)" if col == "sigma_delta" else ""))
        ax.set_ylabel("Simulation settings")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        panel(ax, chr(97+i))
    save_figure(fig, output_dir, "incoming_beam_coverage",
                f"Coverage of the three added scalar inputs over {len(data):,} simulation settings: incoming "
                "horizontal and vertical Twiss alpha, and relative RMS momentum spread. "
                "These are pilot sampling ranges, not a measured distribution of ELENA fills.", manifest)


def beam_distributions(data, output_dir, manifest):
    fig, axes = new_figure(2, 2, figsize=(6.8, 4.6))
    columns = ["emittance_x", "emittance_y", "sigma_x", "sigma_y"]
    for i, (ax, col) in enumerate(zip(axes.flat, columns)):
        values = data[col].to_numpy()
        factor = 1 if col.startswith("emittance") else 1000
        values = factor * values[np.isfinite(values) & (values > 0)]
        low, high = float(values.min()), float(values.max())
        bins = np.geomspace(low, high, 36) if high > low else np.linspace(low*.9, low*1.1, 12)
        ax.hist(values, bins=bins, color=ORANGE if col.endswith("x") else BLUE,
                edgecolor="white", linewidth=.35, alpha=.8)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(LABELS[col] + (" (m)" if col.startswith("emittance") else " (mm)"))
        ax.set_ylabel("Simulation settings")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        panel(ax, chr(97+i))
    save_figure(fig, output_dir, "beam_distributions",
                "Final-observation beam distributions across all pilot simulation settings; horizontal axes use "
                "logarithmic scales and equal log-width bins. Emittance uses the simulator's native "
                "RMS determinant in x/PX and y/PY coordinates (PX and PY normalized to reference momentum), "
                "with units of metres. Beam widths are centered RMS sizes. "
                "These outputs are not experimental capture efficiencies.", manifest)


def correlations(data, input_cols, output_dir, manifest):
    target_cols = ["emittance_x", "emittance_y", "sigma_x", "sigma_y"]
    matrix = data[input_cols + target_cols].corr(method="spearman").loc[input_cols, target_cols]
    fig, ax = plt.subplots(figsize=(5.1, 3.9), layout="constrained")
    im = ax.pcolormesh(np.arange(len(target_cols) + 1) - .5,
                       np.arange(len(input_cols) + 1) - .5, matrix.to_numpy(),
                       vmin=-1, vmax=1, cmap="RdBu_r", shading="flat", rasterized=False)
    ax.invert_yaxis()
    ax.set_xticks(range(len(target_cols)), [LABELS[c] for c in target_cols])
    ax.set_yticks(range(len(input_cols)), [LABELS.get(c, c) for c in input_cols])
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > .65 else "#222222", fontsize=8)
    colorbar = fig.colorbar(im, ax=ax, shrink=.9, pad=.04, label=r"Spearman $\rho$")
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    matrix.to_csv(output_dir / "spearman_correlations.csv")
    save_figure(fig, output_dir, "input_output_correlations",
                "Marginal Spearman correlations between each of the seven scalar inputs and final-observation "
                "beam outputs. Correlations summarize this sampling domain; weak marginal association does "
                "not establish that an input is unimportant or exclude nonlinear interactions.", manifest)


def sample_efficiency(frame, targets, output_dir, manifest):
    fig, axes = new_figure(ncols=len(targets), figsize=(7.1, 3.2))
    for i, (ax, target) in enumerate(zip(np.atleast_1d(axes), targets)):
        subset = frame[frame["target"] == target].sort_values("training_samples")
        if subset.empty:
            raise ValueError(f"No sample-efficiency rows for {target}")
        n = subset["training_samples"]
        ax.plot(n, 100*subset["scaled_MAE"], "o-", color=BLUE, label="MAE", ms=3.5, lw=1)
        ax.plot(n, 100*subset["scaled_RMSE"], "s-", color=ORANGE, label="RMSE", ms=3.5, lw=1)
        ax.set(xlabel="Training samples", ylabel="Scaled error (pp)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_ylim(bottom=0)
        right = ax.twinx()
        right.plot(n, subset["scaled_R2"], "^-", color=GREEN, label=r"$R^2$", ms=4, lw=1)
        right.set_ylabel(r"$R^2$", color=GREEN)
        right.tick_params(axis="y", colors=GREEN, direction="in")
        right.spines["right"].set_color(GREEN)
        right.set_ylim(top=max(1.04, subset["scaled_R2"].max()+.04))
        lines, labels = ax.get_legend_handles_labels()
        other, other_labels = right.get_legend_handles_labels()
        ax.legend(lines + other, labels + other_labels, loc="upper right", fontsize=7)
        panel(ax, chr(97+i), LABELS.get(target, target))
    save_figure(fig, output_dir, "sample_efficiency",
                "Sample-efficiency pilot with separate curves for horizontal and vertical emittance. "
                "Training subsets vary while validation and test settings are fixed. Scaled errors are "
                "evaluated in the full-training-set reference log-emittance coordinate for comparability. "
                "Each point is a single training run; no statistical error bars or saturation threshold are inferred.", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=None, help="Zero-based observation; defaults to training metadata.")
    args = parser.parse_args()
    simulation_metadata = completed_simulation_metadata(args.data)
    preprocessing = read_json(args.training_dir / "preprocessing.json")
    targets = preprocessing["target_cols"]
    segment = preprocessing["segment"] if args.segment is None else args.segment
    if segment != preprocessing["segment"]:
        parser.error("Analysis segment must match the segment used for training.")
    if not 0 <= segment < 8:
        parser.error("Segment must be between 0 and 7.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, qa, summary = load_data(args.data, preprocessing["input_cols"], segment)
    simulation_diagnostics(args.data, preprocessing["input_cols"], qa, simulation_metadata)
    (args.output_dir / "data_quality.json").write_text(json.dumps(qa, indent=2) + "\n")
    summary.to_csv(args.output_dir / "column_summary.csv", index=False)
    if qa["issues"]:
        raise ValueError("Simulation data checks failed; see data_quality.json: " + "; ".join(qa["issues"]))
    predictions = pd.read_csv(args.training_dir / "test_predictions.csv")
    metrics = read_json(args.training_dir / "metrics.json")
    history = pd.read_csv(args.training_dir / "history.csv")
    required_predictions = [f"{prefix}_{t}" for t in targets for prefix in ["true_scaled", "pred_scaled", "true", "pred"]]
    if not np.isfinite(predictions[required_predictions].to_numpy()).all():
        raise ValueError("Nonfinite held-out targets/predictions.")
    validate_predictions(predictions, data, preprocessing, metrics)
    configure_style()
    manifest = []
    parity(predictions, targets, metrics, args.output_dir, manifest)
    residuals(predictions, targets, args.output_dir, manifest)
    training_curves(history, args.output_dir, manifest)
    incoming_coverage(data, args.output_dir, manifest)
    beam_distributions(data, args.output_dir, manifest)
    correlations(data, preprocessing["input_cols"], args.output_dir, manifest)
    sweep_path = args.training_dir / "sample_efficiency.csv"
    if sweep_path.exists():
        sample_efficiency(pd.read_csv(sweep_path), targets, args.output_dir, manifest)
    (args.output_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    captions = "# Figure captions and metric definitions\n\n" + "\n\n".join(
        f"**{item['file']}**\n\n{item['caption']}" for item in manifest)
    (args.output_dir / "captions.md").write_text(captions + "\n")
    print(f"Data checks passed: {len(data):,} rows, {len(preprocessing['input_cols'])} inputs, eight observations.")
    print(f"Wrote {len(manifest)} transparent vector PDFs and data QA to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
