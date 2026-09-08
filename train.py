"""Train the notebook MLP on one seven-input MAD-X dataset, without plotting.

Example (from the repository root):
    python train.py --data results/pilot_20260907/data.csv \
        --output-dir results/pilot_20260907/training
    python train.py --data results/pilot_20260907/data.csv \
        --output-dir results/pilot_20260907/training_sweep --sample-efficiency 500,1000,2000,4000

The legacy notebook is retained as a historical experiment. This entry point fits
preprocessing on training rows only, groups repeated settings into a single split,
and saves the transforms needed to convert predictions back to physical units.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml


def load_dataset(path, config):
    """Select the configured observation from eight-element simulation lists."""
    frame = pd.read_csv(path)
    columns = config['input_cols'] + config['target_cols']
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError('Dataset is missing required columns: ' + ', '.join(missing))
    selected = frame[columns].copy()
    segment = int(config['segment'])
    if not 0 <= segment < 8:
        raise ValueError('segment must be between 0 and 7')
    for col in config['target_cols']:
        values = []
        for row, value in enumerate(frame[col]):
            try:
                sequence = ast.literal_eval(value) if isinstance(value, str) else value
                if not isinstance(sequence, (list, tuple)) or len(sequence) != 8:
                    raise ValueError('expected a list of eight observation values')
                values.append(float(sequence[segment]))
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError('{} row {}: {}'.format(col, row, exc)) from exc
        selected[col] = values
    selected = selected.apply(pd.to_numeric, errors='raise')
    finite = np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)
    positive = (selected[config['target_cols']] > 0).all(axis=1).to_numpy()
    if not (finite & positive).all():
        bad = np.flatnonzero(~(finite & positive))
        raise ValueError('Nonfinite inputs/targets or nonpositive log targets in rows {}'.format(bad[:10].tolist()))
    if 'sigma_delta' in selected and (selected['sigma_delta'] < 0).any():
        raise ValueError('sigma_delta must be a nonnegative RMS relative momentum spread')
    return selected


def split_indices(frame, input_cols, seed, test_size=0.1, val_size=0.1):
    """Split unique full input tuples, keeping repeated simulations together."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError('test_size and val_size must be positive fractions summing to < 1')
    _, group = np.unique(frame[input_cols].to_numpy(), axis=0, return_inverse=True)
    n_groups = int(group.max()) + 1 if len(group) else 0
    n_test, n_val = int(n_groups * test_size), int(n_groups * val_size)
    if min(n_test, n_val, n_groups - n_test - n_val) < 1:
        raise ValueError('Too few unique input settings for the requested train/val/test split')
    order = np.random.default_rng(seed).permutation(n_groups)
    groups = {'test': order[:n_test], 'val': order[n_test:n_test + n_val],
              'train': order[n_test + n_val:]}
    rng = np.random.default_rng(seed)
    return {name: rng.permutation(np.flatnonzero(np.isin(group, ids)))
            for name, ids in groups.items()}


def fit_preprocessing(frame, config, train_idx):
    """Persist plain JSON min/range arrays in explicit model feature order."""
    inputs, targets = config['input_cols'], config['target_cols']
    x = frame.iloc[train_idx][inputs].to_numpy(dtype=float)
    y_log = np.log(frame.iloc[train_idx][targets].to_numpy(dtype=float))
    x_range, y_range = np.ptp(x, axis=0), np.ptp(y_log, axis=0)
    return {'input_cols': inputs, 'target_cols': targets, 'segment': int(config['segment']),
            'input_min': x.min(axis=0).tolist(),
            'input_range': np.where(x_range == 0, 1., x_range).tolist(),
            'target_log_min': y_log.min(axis=0).tolist(),
            'target_log_range': np.where(y_range == 0, 1., y_range).tolist(),
            'constant_input_cols': [c for c, r in zip(inputs, x_range) if r == 0],
            'constant_target_cols': [c for c, r in zip(targets, y_range) if r == 0],
            'target_transform': 'natural_log_then_minmax', 'fit_rows': len(train_idx),
            'clipping': False, 'sigma_delta_definition': 'RMS (p-p0)/p0'}


def transform(frame, preprocessing):
    p = preprocessing
    x = (frame[p['input_cols']].to_numpy(dtype=float) - p['input_min']) / p['input_range']
    y = (np.log(frame[p['target_cols']].to_numpy(dtype=float)) - p['target_log_min']) / p['target_log_range']
    return x.astype('float32'), y.astype('float32')


def inverse_targets(y_scaled, preprocessing):
    return np.exp(np.asarray(y_scaled, dtype=float) * preprocessing['target_log_range']
                  + preprocessing['target_log_min'])


def regression_metrics(y_true, y_pred):
    """Each vector is one target; errors retain the units of its input space."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {'MAE': float(np.mean(np.abs(error))),
            'RMSE': float(np.sqrt(np.mean(error ** 2))),
            'R2': float(1 - np.sum(error ** 2) / total) if total > 0 else None}


def score_targets(y_true_scaled, y_pred_scaled, preprocessing, true_physical=None):
    true_phys = (inverse_targets(y_true_scaled, preprocessing) if true_physical is None
                 else np.asarray(true_physical, dtype=float))
    pred_phys = inverse_targets(y_pred_scaled, preprocessing)
    if not np.isfinite(pred_phys).all():
        raise ValueError('Nonfinite physical predictions after reversing log transform')
    return {space: {target: regression_metrics(truth[:, j], pred[:, j])
                    for j, target in enumerate(preprocessing['target_cols'])}
            for space, truth, pred in [('scaled', y_true_scaled, y_pred_scaled),
                                       ('physical', true_phys, pred_phys)]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_model(config):
    """Notebook architecture; input width now follows seven configured columns."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(len(config['input_cols']),))
    x = inp
    for width in (256, 128, 128, 32):
        x = tf.keras.layers.Dense(width, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    out = tf.keras.layers.Dense(len(config['target_cols']))(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model


def make_dataset(x, y, rows, config, training=False):
    """Seeded, reproducible epoch reshuffling; avoids repeated NumPy-adapter batches."""
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices((x[rows], y[rows]))
    if training:
        dataset = dataset.shuffle(len(rows), seed=config['seed'], reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.deterministic = True
    return dataset.batch(config['batch_size']).with_options(options).prefetch(1)


def fit_model(frame, config, indices, preprocessing, output_dir):
    import tensorflow as tf
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    x, y = transform(frame, preprocessing)
    model = build_model(config)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                      patience=config['patience'], restore_best_weights=True)
    callbacks = [early_stopping, tf.keras.callbacks.CSVLogger(str(output_dir / 'history.csv')),
                 tf.keras.callbacks.TerminateOnNaN()]
    started = time.perf_counter()
    training_data = make_dataset(x, y, indices['train'], config, training=True)
    validation_data = make_dataset(x, y, indices['val'], config)
    history = model.fit(training_data, validation_data=validation_data,
                        epochs=config['epochs'], callbacks=callbacks, shuffle=False, verbose=2)
    fit_seconds = time.perf_counter() - started
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise ValueError('Training produced nonfinite loss/metrics')
    # Keras 2 may not restore best weights when the epoch limit is reached.
    if early_stopping.best_weights is not None:
        model.set_weights(early_stopping.best_weights)
    model.save(str(output_dir / 'best_model.keras'))
    # One-based epochs for figures; CSVLogger also leaves useful partial progress.
    pd.DataFrame({'epoch': np.arange(1, len(history.epoch) + 1), **history.history}).to_csv(
        output_dir / 'history.csv', index=False)
    prediction = model(x[indices['test']], training=False).numpy()
    metrics = score_targets(y[indices['test']], prediction, preprocessing,
                            frame.iloc[indices['test']][config['target_cols']].to_numpy())
    test_frame = frame.iloc[indices['test']][config['input_cols']].copy()
    test_frame.insert(0, 'row_index', indices['test'])
    physical = inverse_targets(prediction, preprocessing)
    for j, col in enumerate(config['target_cols']):
        test_frame['true_' + col] = frame.iloc[indices['test']][col].to_numpy()
        test_frame['pred_' + col] = physical[:, j]
        test_frame['true_scaled_' + col] = y[indices['test'], j]
        test_frame['pred_scaled_' + col] = prediction[:, j]
    test_frame.to_csv(output_dir / 'test_predictions.csv', index=False)
    write_json(output_dir / 'preprocessing.json', preprocessing)
    write_json(output_dir / 'metrics.json', metrics)
    details = {'fit_seconds': fit_seconds, 'epochs_completed': len(history.epoch),
               'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
               'parameter_count': model.count_params()}
    return model, details, prediction, metrics


def benchmark_inference(model, x_test):
    """Warm compiled CPU/GPU calls; include Python dispatch and host synchronization."""
    import tensorflow as tf
    @tf.function(input_signature=[tf.TensorSpec([None, x_test.shape[1]], tf.float32)])
    def predict(x):
        return model(x, training=False)
    result = {'description': 'Warm tf.function calls including dispatch and .numpy() synchronization; '
                             'excludes preprocessing, loading, graph tracing and cold start.'}
    for name, values, repeats in [('single', x_test[:1], 200), ('batch', x_test, 20)]:
        tensor = tf.constant(values)
        for _ in range(5):
            predict(tensor).numpy()
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            predict(tensor).numpy()
            elapsed.append(time.perf_counter() - start)
        result[name] = {'batch_size': len(values), 'repeats': repeats,
                        'median_seconds': float(np.median(elapsed)),
                        'p95_seconds': float(np.percentile(elapsed, 95)),
                        'median_seconds_per_sample': float(np.median(elapsed) / len(values))}
    return result


def sample_efficiency(frame, config, indices, reference, sizes, output_dir):
    """Fresh subset fits, fixed holdouts, comparable full-train-reference metrics.

    Internal normalization is fitted to each training subset. For the plotted
    scaled metrics only, predictions are mapped into the reference normalization
    fitted to the main training pool. No held-out row fits either transform.
    """
    output_dir = Path(output_dir)
    records = []
    _, reference_y = transform(frame, reference)
    for count in sorted(set(sizes)):
        if not 2 <= count <= len(indices['train']):
            raise ValueError('Sample-efficiency counts must be between 2 and {} training rows'.format(len(indices['train'])))
        # The train order was seeded once. Prefixes give reproducible nested subsets.
        subset = {**indices, 'train': indices['train'][:count]}
        preproc = fit_preprocessing(frame, config, subset['train'])
        destination = output_dir / 'sample_efficiency' / str(count)
        print('\nSample-efficiency fit: {} training rows'.format(count), flush=True)
        _, details, prediction, _ = fit_model(frame, config, subset, preproc, destination)
        # Do not exponentiate then log: this exact affine mapping avoids roundoff.
        predicted_log = prediction.astype(float) * preproc['target_log_range'] + preproc['target_log_min']
        reference_pred = (predicted_log - reference['target_log_min']) / reference['target_log_range']
        scores = score_targets(reference_y[indices['test']], reference_pred, reference,
                               frame.iloc[indices['test']][config['target_cols']].to_numpy())
        write_json(destination / 'reference_metrics.json', scores)
        write_json(destination / 'fit_details.json', {**details, 'train_rows': count,
                   'validation_rows': len(indices['val']), 'test_rows': len(indices['test'])})
        np.save(destination / 'train_indices.npy', subset['train'])
        for target in config['target_cols']:
            records.append({'training_samples': count, 'target': target,
                **{space + '_' + metric: value for space in ('scaled', 'physical')
                   for metric, value in scores[space][target].items()},
                'epochs': details['epochs_completed'], 'fit_seconds': details['fit_seconds']})
        pd.DataFrame(records).to_csv(output_dir / 'sample_efficiency.csv', index=False)
    write_json(output_dir / 'sample_efficiency_protocol.json', {
        'training_counts': sorted(set(sizes)), 'seed': config['seed'],
        'subsets': 'Nested prefixes of the seeded main training split; fresh model and subset-fitted preprocessing.',
        'holdouts': 'Fixed main validation/test splits for every count; repeated settings cannot cross holdouts.',
        'scaled_metrics': 'Mapped into the main full-training-pool natural-log minmax transform for comparison.',
        'interpretation': 'Single-seed pilot curve, not a confirmed minimum sample requirement.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('train_config.yaml'))
    parser.add_argument('--data', type=Path, required=True, help='One CSV with all seven input columns; legacy four-input data is rejected')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--threads', type=int, default=2, help='TensorFlow intra/inter-op thread limits')
    parser.add_argument('--sample-efficiency', default='', help='Comma-separated TRAINING row counts; optional and slower')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ('epochs', 'patience', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.threads < 1 or config['epochs'] < 1 or config['patience'] < 0:
        parser.error('threads/epochs must be positive and patience must be nonnegative')
    sizes = [int(value.strip()) for value in args.sample_efficiency.split(',') if value.strip()]
    frame = load_dataset(args.data, config)
    indices = split_indices(frame, config['input_cols'], config['seed'], config['test_size'], config['val_size'])
    if any(count < 2 or count > len(indices['train']) for count in sizes):
        parser.error('sample-efficiency counts must be within [2, {}] training rows'.format(len(indices['train'])))
    output_dir = args.output_dir.resolve()
    if (output_dir / 'best_model.keras').exists():
        parser.error('output-dir already contains a model; choose a new run directory to preserve results')
    output_dir.mkdir(parents=True, exist_ok=True)
    preproc = fit_preprocessing(frame, config, indices['train'])
    np.savez(output_dir / 'split_indices.npz', **indices)
    assignments = np.empty(len(frame), dtype=object)
    for name, rows in indices.items():
        assignments[rows] = name
    pd.DataFrame({'row_index': np.arange(len(frame)), 'split': assignments}).to_csv(output_dir / 'split.csv', index=False)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_path': str(args.data.resolve()), 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'training_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_version': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(), 'logical_cpu_count': os.cpu_count(),
        'config': config, 'rows': len(frame), 'input_dimensions': len(config['input_cols']),
        'target_dimensions': len(config['target_cols']),
        'split_rows': {key: len(value) for key, value in indices.items()},
        'unique_input_settings': len(frame[config['input_cols']].drop_duplicates()),
        'split_method': 'Seeded unique full-input-tuple group split; no repeated input setting crosses splits.',
        'preprocessing': 'Natural-log targets and minmax scaling fitted to train rows only; no clipping.',
        'model': 'Dense 256/128/128/32 with BatchNorm/ReLU, linear targets, Adam, Huber.',
        'threads': args.threads, 'sample_efficiency_counts': sizes,
        'input_pipeline': 'tf.data seeded reshuffle_each_iteration',
        'batch_order': 'Full training-row shuffle buffer; different order each epoch, reproducible for a fixed seed.'}
    write_json(output_dir / 'provenance.json', provenance)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.experimental.enable_op_determinism()
    print('Training shapes: X={}, y={}; split {}'.format(
        (len(frame), len(config['input_cols'])), (len(frame), len(config['target_cols'])), provenance['split_rows']), flush=True)
    model, details, _, metrics = fit_model(frame, config, indices, preproc, output_dir)
    x, y = transform(frame, preproc)
    baseline = np.repeat(y[indices['train']].mean(axis=0, keepdims=True), len(indices['test']), axis=0)
    write_json(output_dir / 'mean_baseline_metrics.json', score_targets(
        y[indices['test']], baseline, preproc, frame.iloc[indices['test']][config['target_cols']].to_numpy()))
    provenance.update(details)
    provenance.update({'tensorflow_version': tf.__version__, 'numpy_version': np.__version__,
                       'inference_timing': benchmark_inference(model, x[indices['test']])})
    write_json(output_dir / 'provenance.json', provenance)
    print(json.dumps(metrics, indent=2), flush=True)
    if sizes:
        sample_efficiency(frame, config, indices, preproc, sizes, output_dir)
    print('Saved training artifacts to {}'.format(output_dir), flush=True)


if __name__ == '__main__':
    main()
