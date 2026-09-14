"""Select a validation-curve knee without consulting the held-out test set.

Kneedle (Satopaa et al., 2011), convex/decreasing, linear sample-count axis,
S=1. Mean RMSE is made non-increasing by isotonic regression. The marker is
shown only when supported by seed and sensitivity checks; it is not a claim
that more training data cannot help. See results/experiment_20260908_12000/protocol.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.isotonic import IsotonicRegression

METRIC = 'validation_scaled_RMSE'
TARGETS = ('emittance_x', 'emittance_y')
SOURCE = 'https://www.cs.williams.edu/~jeannie/papers/kneedle-simplex11.pdf'


def candidate_knee(counts, errors, sensitivity=1.0):
    """Return a supported interior knee and the monotone fitted error curve."""
    counts, errors = np.asarray(counts, dtype=float), np.asarray(errors, dtype=float)
    smooth = IsotonicRegression(increasing=False).fit_transform(counts, errors)
    span = float(np.ptp(smooth))
    if span <= max(1e-12, float(np.max(np.abs(smooth))) * 1e-8):
        return None, smooth
    # A flat or straight curve has no identifiable knee, even with round-off.
    xnorm = (counts - counts[0]) / np.ptp(counts)
    benefit = (smooth[0] - smooth) / span
    if np.max(benefit - xnorm) <= 1e-8:
        return None, smooth
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        locator = KneeLocator(counts, smooth, S=sensitivity, curve='convex',
                              direction='decreasing', interp_method='interp1d', online=False)
    if locator.knee is None:
        return None, smooth
    index = int(np.argmin(np.abs(counts - locator.knee)))
    # Two later sampled sizes must exist to support the apparent tail.
    if index == 0 or index > len(counts) - 3:
        return None, smooth
    return int(counts[index]), smooth


def summarize_curve(frame):
    required = {'training_samples', 'target', 'seed', METRIC}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing validation-curve columns: {sorted(missing)}')
    frame = frame[list(required)].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError('Curve must contain complete, nonempty validation results.')
    if set(frame['target']) != set(TARGETS):
        raise ValueError(f'Expected exactly these targets: {TARGETS}')
    for col in ('training_samples', 'seed'):
        values = pd.to_numeric(frame[col], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{col} must contain finite integers.')
        frame[col] = values.astype(int)
    if (frame['training_samples'] <= 0).any():
        raise ValueError('Sample counts must be positive.')
    values = pd.to_numeric(frame[METRIC], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Validation RMSE must be finite and nonnegative.')
    frame[METRIC] = values
    if frame.duplicated(['training_samples', 'target', 'seed']).any():
        raise ValueError('Duplicate count/target/seed results.')
    counts = sorted(frame['training_samples'].unique().tolist())
    seeds = sorted(frame['seed'].unique().tolist())
    if len(counts) < 5 or len(seeds) < 3:
        raise ValueError('Need at least five sizes and three model seeds for knee stability.')
    expected = {(n, t, s) for n in counts for t in TARGETS for s in seeds}
    actual = set(frame[['training_samples', 'target', 'seed']].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Every target and seed must share the same complete sample-count grid.')
    summary = {
        'method': 'Kneedle on isotonic non-increasing mean validation RMSE; linear sample-count axis',
        'metric': METRIC, 'source': SOURCE, 'sensitivity': 1.0,
        'sensitivity_checks': [0.5, 1.0, 2.0], 'model_seeds': seeds,
        'stability_rule': 'At least two thirds of seeds and two of three sensitivities within one grid step of the mean knee; at least two later counts.',
        'interpretation': 'Estimated diminishing-returns knee, not proof of saturation, optimal dataset size, or zero benefit from more data.',
        'selection_data': 'Validation results only. No held-out test predictions or metrics used.',
        'targets': {},
    }
    for target in TARGETS:
        table = frame[frame.target == target].pivot(index='training_samples', columns='seed', values=METRIC).loc[counts, seeds]
        mean = table.mean(axis=1).to_numpy()
        sd = table.std(axis=1, ddof=1).to_numpy()
        knee, smooth = candidate_knee(counts, mean)
        per_seed = {str(seed): candidate_knee(counts, table[seed].to_numpy())[0] for seed in seeds}
        sensitivity = {str(s): candidate_knee(counts, mean, s)[0] for s in (0.5, 1.0, 2.0)}
        def close(other):
            return knee is not None and other is not None and abs(counts.index(other) - counts.index(knee)) <= 1
        seed_support = sum(close(value) for value in per_seed.values())
        sensitivity_support = sum(close(value) for value in sensitivity.values())
        stable = knee is not None and seed_support >= int(np.ceil(2 * len(seeds) / 3)) and sensitivity_support >= 2
        knee_error = None if knee is None else float(smooth[counts.index(knee)])
        # Zero is a valid RMSE. Relative improvement from zero is undefined,
        # even when the whole fitted tail is also zero; keep JSON finite.
        remaining = None if knee_error is None or knee_error == 0 else float((knee_error - smooth[-1]) / knee_error)
        summary['targets'][target] = {
            'knee': knee, 'display_knee': stable,
            'status': 'detected' if stable else ('not_detected' if knee is None else 'unstable'),
            'label': 'Estimated knee', 'seed_knees': per_seed, 'sensitivity_knees': sensitivity,
            'supporting_seeds': seed_support, 'supporting_sensitivities': sensitivity_support,
            'remaining_relative_improvement': remaining,
            'remaining_improvement_definition': 'Fractional decline in isotonic mean validation RMSE from candidate knee to largest sampled training size; null if no candidate or its RMSE is zero.',
            'training_samples': counts, 'mean_validation_rmse': mean.tolist(),
            'sd_validation_rmse': sd.tolist(), 'isotonic_validation_rmse': smooth.tolist(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_curve(pd.read_csv(args.sweep))
    summary['sweep_path'] = str(args.sweep.resolve())
    summary['sweep_sha256'] = hashlib.sha256(args.sweep.read_bytes()).hexdigest()
    summary['diagnostics_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for target, item in summary['targets'].items():
        print(f"{target}: {item['status']}; candidate={item['knee']}; marker={item['display_knee']}")


if __name__ == '__main__':
    main()
