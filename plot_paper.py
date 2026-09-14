"""Render the two paper figures from completed training artifacts, independently.

python plot_paper.py --training-dir results/experiment_20260908_12000/training \
    --output-dir results/experiment_20260908_12000/figures --figure both \
    --variability-summary results/experiment_20260908_12000/variability/variability_summary.json \
    --knee-summary results/experiment_20260908_12000/training/knee_summary.json

Use --figure parity or efficiency to regenerate one figure. Edit plot_config.yaml
for each figure's single font multiplier, or use --font-scale to override it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd
import yaml

BLUE, ORANGE, GREEN, KNEE = '#0072B2', '#E69F00', '#009E73', '#D55E00'
TARGETS = ['emittance_x', 'emittance_y']
FONT_SIZES = {'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
              'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
              'figure.titlesize': 10, 'figure.labelsize': 9}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_style(font_scale):
    """All text sizes, including math/annotations and future ticks, share one factor."""
    if not np.isfinite(font_scale) or font_scale <= 0:
        raise ValueError('Font scale must be finite and positive.')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .7,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': .7,
        'ytick.major.width': .7, 'legend.frameon': False, 'savefig.transparent': True,
        'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.dpi': 600,
        **{key: value * font_scale for key, value in FONT_SIZES.items()}})


def make_axes(options):
    configure_style(options['font_scale'])
    fig, axes = plt.subplots(1, 2, figsize=options['figsize'], layout='constrained',
                             gridspec_kw={'wspace': options['wspace']})
    for ax in axes:
        ax.tick_params(top=True, right=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fig, axes


def variability_widths(summary, preprocessing_path):
    if summary is None:
        return None
    if summary.get('status') != 'complete':
        raise ValueError('Variability summary must describe a completed repeat run.')
    if summary.get('preprocessing', {}).get('sha256') != sha256(preprocessing_path):
        raise ValueError('Variability summary uses different/missing training preprocessing.')
    widths = {target: float(summary['spaces']['scaled'][target]['half_max_observed_range']) for target in TARGETS}
    if any(not np.isfinite(value) or value < 0 for value in widths.values()):
        raise ValueError('Variability band widths must be finite and nonnegative.')
    return widths


def plot_parity(predictions, metrics, options, widths=None):
    fig, axes = make_axes(options)
    for ax, target in zip(axes, TARGETS):
        truth = predictions[f'true_scaled_{target}'].to_numpy(dtype=float)
        estimate = predictions[f'pred_scaled_{target}'].to_numpy(dtype=float)
        if not np.isfinite(np.r_[truth, estimate]).all():
            raise ValueError('Nonfinite parity targets/predictions.')
        low, high = min(0., truth.min(), estimate.min()), max(1., truth.max(), estimate.max())
        padding = .025 * (high - low)
        limits = (low - padding, high + padding)
        x = np.linspace(*limits, 256)
        if widths is not None:
            half = widths[target]
            ax.fill_between(x, x-half, x+half, color=BLUE, alpha=.25,
                            label='Simulation variability', linewidth=0)
            ax.plot(x, x-half, color=BLUE, lw=.45, alpha=.65)
            ax.plot(x, x+half, color=BLUE, lw=.45, alpha=.65)
        ax.scatter(truth, estimate, s=options.get('marker_size', 7), color=ORANGE,
                   alpha=.75, edgecolors='none', label='Test samples')
        ax.plot(limits, limits, '--', color=BLUE, lw=.9, label='Ideal')
        ax.set(xlim=limits, ylim=limits, aspect='equal')
        plane = target[-1]
        ax.set_xlabel(rf'Simulated $\varepsilon_{{{plane},\mathrm{{scaled}}}}$')
        ax.set_ylabel(rf'Predicted $\varepsilon_{{{plane},\mathrm{{scaled}}}}$')
        m = metrics['scaled'][target]
        r2 = 'undefined' if m['R2'] is None else f"{m['R2']:.3f}"
        ax.text(.97, .035, f"MAE = {100*m['MAE']:.2f} pp\n" + rf'$R^2$ = {r2}',
                transform=ax.transAxes, ha='right', va='bottom')
    axes[0].legend(loc='upper left', borderaxespad=.45, handlelength=1.8, labelspacing=.3)
    return fig


def aggregate_efficiency(frame, protocol):
    """Aggregate validation metrics only; never consult test metrics for the curve."""
    keys = ['training_samples', 'target', 'seed']
    metrics = ['validation_scaled_MAE', 'validation_scaled_RMSE', 'validation_scaled_R2']
    required = set(keys + metrics)
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing validation sweep fields: {sorted(required - set(frame.columns))}')
    if not protocol.get('completed') or protocol.get('selection_split') != 'validation':
        raise ValueError('A completed validation-only sample-efficiency protocol is required.')
    if frame.duplicated(keys).any() or not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError('Duplicate or nonfinite sample-efficiency observations.')
    seeds = protocol['repeat_seeds']
    counts = protocol['training_counts']
    expected = {(n, t, s) for n in counts for t in TARGETS for s in seeds}
    actual = set(frame[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Sweep does not contain every declared count, target and seed.')
    if len(seeds) < 2:
        raise ValueError('At least two model seeds are needed for mean and standard deviation.')
    return frame.groupby(['target', 'training_samples'])[metrics].agg(['mean', 'std']).sort_index()



def validate_knee_summary(summary, sweep_path):
    if summary is not None and summary.get('sweep_sha256') != sha256(sweep_path):
        raise ValueError('Knee summary does not match the plotted validation sweep.')
    return summary


def plot_efficiency(aggregate, options, knee_summary=None):
    fig, axes = make_axes(options)
    right_axes = []
    marker_any = knee_summary is not None and any(knee_summary['targets'][t]['display_knee'] for t in TARGETS)
    for ax, target in zip(axes, TARGETS):
        rows = aggregate.loc[target]
        n = rows.index.to_numpy()
        for metric, color, marker, label in [('MAE', BLUE, 'o', 'MAE'), ('RMSE', ORANGE, 's', 'RMSE')]:
            mean = 100 * rows[(f'validation_scaled_{metric}', 'mean')].to_numpy()
            sd = 100 * rows[(f'validation_scaled_{metric}', 'std')].to_numpy()
            ax.plot(n, mean, marker+'-', color=color, label=label, ms=3.6, lw=1.05)
            ax.fill_between(n, mean-sd, mean+sd, color=color, alpha=.14, linewidth=0)
        ax.set_xlabel('Training samples')
        plane = target[-1]
        ax.set_ylabel(rf'$\varepsilon_{plane}$ validation error (pp)')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        right = ax.twinx()
        right_axes.append(right)
        mean = rows[('validation_scaled_R2', 'mean')].to_numpy()
        sd = rows[('validation_scaled_R2', 'std')].to_numpy()
        right.plot(n, mean, '^-', color=GREEN, label=r'$R^2$', ms=4, lw=1.05)
        right.fill_between(n, mean-sd, mean+sd, color=GREEN, alpha=.12, linewidth=0)
        right.set_ylabel(r'$R^2$', color=GREEN)
        right.tick_params(axis='y', colors=GREEN, direction='in')
        right.spines['right'].set_color(GREEN)
        right.set_ylim(top=max(1.04, np.max(mean+sd)+.04))
        if knee_summary is not None:
            item = knee_summary['targets'][target]
            if item['display_knee']:
                if item['knee'] not in n:
                    raise ValueError('Displayed knee is not a sampled training size.')
                ax.axvline(item['knee'], color=KNEE, ls='--', lw=1.05)
                ax.text(item['knee'], .03, f"n = {item['knee']:,}", color=KNEE,
                        transform=ax.get_xaxis_transform(), rotation=90, va='bottom', ha='right')
    handles, labels = axes[1].get_legend_handles_labels()
    more_handles, more_labels = right_axes[1].get_legend_handles_labels()
    handles += more_handles
    labels += more_labels
    if marker_any:
        handles.append(Line2D([], [], color=KNEE, ls='--', lw=1.05))
        labels.append('Estimated knee')
    # Attached only to the right subplot; outside its data area so all curves remain visible.
    axes[1].legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(.5, 1.015), ncol=2, borderaxespad=.15, columnspacing=1.0,
                   handlelength=1.8, labelspacing=.25)
    return fig


def save(fig, output_dir, name, caption, sources, options):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{name}.pdf'
    fig.savefig(path, transparent=True, bbox_inches='tight', pad_inches=.04,
                metadata={'Creator': 'QuadTuneML plot_paper.py', 'Title': name})
    plt.close(fig)
    sidecar = {'pdf': path.name, 'sha256': sha256(path), 'plot_options': options,
               'sources': sources, 'caption': caption, 'script_sha256': sha256(__file__)}
    path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False)+'\n')
    path.with_suffix('.caption.md').write_text(caption+'\n')
    print(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figure', choices=['parity', 'efficiency', 'both'], default='both')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('plot_config.yaml'))
    parser.add_argument('--font-scale', type=float, help='Override the font multiplier for each selected figure.')
    parser.add_argument('--variability-summary', type=Path)
    parser.add_argument('--knee-summary', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for name in ['parity', 'efficiency']:
        if args.font_scale is not None:
            config[name]['font_scale'] = args.font_scale
    if args.figure in ['parity', 'both']:
        pre_path = args.training_dir / 'preprocessing.json'
        predictions_path = args.training_dir / 'test_predictions.csv'
        metrics_path = args.training_dir / 'metrics.json'
        variability = read_json(args.variability_summary) if args.variability_summary else None
        widths = variability_widths(variability, pre_path)
        fig = plot_parity(pd.read_csv(predictions_path), read_json(metrics_path), config['parity'], widths)
        sources = {str(p.resolve()): sha256(p) for p in [pre_path, predictions_path, metrics_path]}
        caption = ('Held-out test prediction parity in the training-fit normalized natural-log emittance coordinate. '
                   'MAE is percentage points (pp) of the training log-emittance range, not relative physical error. '
                   'All points are shown without clipping predictions to [0, 1]. ')
        if variability is not None:
            sources[str(args.variability_summary.resolve())] = sha256(args.variability_summary)
            caption += (f"The simulation-variability band uses half the maximum observed within-anchor min-max range, "
                        f"separately for each target, from {variability['anchors']} fixed settings with "
                        f"{variability['replicates_per_anchor']} particle realizations each. "
                        'This is an observed finite-repeat range, not a 95% confidence interval, a true bound, '
                        'or ML predictive uncertainty. ')
        else:
            caption += 'No repeated-simulation variability summary was supplied; no band is shown. '
        save(fig, args.output_dir, 'parity_scaled', caption, sources, config['parity'])
    if args.figure in ['efficiency', 'both']:
        sweep_path = args.training_dir / 'sample_efficiency.csv'
        protocol_path = args.training_dir / 'sample_efficiency_protocol.json'
        protocol = read_json(protocol_path)
        aggregate = aggregate_efficiency(pd.read_csv(sweep_path), protocol)
        knee = validate_knee_summary(read_json(args.knee_summary) if args.knee_summary else None, sweep_path)
        fig = plot_efficiency(aggregate, config['efficiency'], knee)
        sources = {str(p.resolve()): sha256(p) for p in [sweep_path, protocol_path]}
        if args.knee_summary:
            sources[str(args.knee_summary.resolve())] = sha256(args.knee_summary)
        caption = (f"Validation sample-efficiency curves, mean plus/minus one sample standard deviation across "
                   f"{len(protocol['repeat_seeds'])} model seeds at each training size. Training subsets and "
                   'validation rows are fixed across seeds; all errors use the full-training reference log scale. '
                   'Shading describes optimization-seed variation, not a data-sampling confidence interval. '
                   'Test metrics are not used to select sample counts or knees. ')
        if knee is not None and any(knee['targets'][t]['display_knee'] for t in TARGETS):
            caption += ('Dashed vertical markers denote stability-supported estimated validation-RMSE knees; '
                        'they do not establish saturation or prove that additional data cannot help. ')
        else:
            caption += 'No stability-supported knee marker is shown; a plateau is not established. '
        save(fig, args.output_dir, 'sample_efficiency', caption, sources, config['efficiency'])


if __name__ == '__main__':
    main()
