"""Train the notebook MLP on one seven-input MAD-X dataset, without plotting.

Example (from the repository root):
    python train.py --data results/pilot_20260907/data.csv \
        --output-dir results/pilot_20260907/training
    python train.py --data results/pilot_20260907/data.csv \
        --output-dir results/pilot_20260907/training_sweep --sample-efficiency 500,1000,2000,4000 \
        --repeat-seeds 42,43,44 --split-seed 42

The legacy notebook is retained as a historical experiment. This entry point fits
preprocessing on training rows only, groups repeated settings into a single split,
and saves the transforms needed to convert predictions back to physical units.
Learning curves use validation-only subset fits; test metrics describe only the
predeclared full-training-size main model. Repeated seeds never change the split.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import multiprocessing
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
    # Each artifact has one writer; atomic publication keeps live readers safe.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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


def training_callbacks(config, output_dir):
    """Early stopping may warm up; the best checkpoint still covers every epoch."""
    import tensorflow as tf

    class BestValidationWeights(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_loss = float('inf')
            self.best_epoch = None
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            loss = float((logs or {}).get('val_loss', float('inf')))
            if np.isfinite(loss) and loss < self.best_loss:
                self.best_loss, self.best_epoch = loss, epoch + 1
                self.best_weights = self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    checkpoint = BestValidationWeights()
    callbacks = [checkpoint,
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'],
            start_from_epoch=config.get('warmup_epochs', 0), restore_best_weights=False),
        tf.keras.callbacks.CSVLogger(str(Path(output_dir) / 'history.csv')),
        tf.keras.callbacks.TerminateOnNaN()]
    return callbacks, checkpoint


def save_predictions(frame, config, rows, truth_scaled, prediction, preprocessing,
                     output_dir, split):
    """Keep physical values and model coordinates explicit for both holdout sets."""
    metrics = score_targets(truth_scaled, prediction, preprocessing,
                            frame.iloc[rows][config['target_cols']].to_numpy())
    output = frame.iloc[rows][config['input_cols']].copy()
    output.insert(0, 'row_index', rows)
    physical = inverse_targets(prediction, preprocessing)
    for j, col in enumerate(config['target_cols']):
        output['true_' + col] = frame.iloc[rows][col].to_numpy()
        output['pred_' + col] = physical[:, j]
        output['true_scaled_' + col] = truth_scaled[:, j]
        output['pred_scaled_' + col] = prediction[:, j]
    output.to_csv(Path(output_dir) / (split + '_predictions.csv'), index=False)
    filename = 'metrics.json' if split == 'test' else split + '_metrics.json'
    write_json(Path(output_dir) / filename, metrics)
    return metrics


def fit_model(frame, config, indices, preprocessing, output_dir, evaluate_test=True):
    import tensorflow as tf
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    x, y = transform(frame, preprocessing)
    model = build_model(config)
    callbacks, checkpoint = training_callbacks(config, output_dir)
    started = time.perf_counter()
    training_data = make_dataset(x, y, indices['train'], config, training=True)
    validation_data = make_dataset(x, y, indices['val'], config)
    history = model.fit(training_data, validation_data=validation_data,
                        epochs=config['epochs'], callbacks=callbacks, shuffle=False, verbose=2)
    fit_seconds = time.perf_counter() - started
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise ValueError('Training produced nonfinite loss/metrics')
    if checkpoint.best_weights is None:
        raise ValueError('No finite validation checkpoint was produced')
    model.save(str(output_dir / 'best_model.keras'))
    # One-based epochs for figures; CSVLogger also leaves useful partial progress.
    pd.DataFrame({'epoch': np.arange(1, len(history.epoch) + 1), **history.history}).to_csv(
        output_dir / 'history.csv', index=False)
    validation_prediction = model(x[indices['val']], training=False).numpy()
    save_predictions(frame, config, indices['val'], y[indices['val']], validation_prediction,
                     preprocessing, output_dir, 'validation')
    prediction, metrics = None, None
    if evaluate_test:
        prediction = model(x[indices['test']], training=False).numpy()
        metrics = save_predictions(frame, config, indices['test'], y[indices['test']], prediction,
                                   preprocessing, output_dir, 'test')
    write_json(output_dir / 'preprocessing.json', preprocessing)
    details = {'fit_seconds': fit_seconds, 'epochs_completed': len(history.epoch),
               'best_epoch': checkpoint.best_epoch, 'best_validation_loss': checkpoint.best_loss,
               'parameter_count': model.count_params(), 'seed': config['seed'],
               'warmup_epochs': config.get('warmup_epochs', 0)}
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


def initialize_training_worker(threads):
    """Each spawned process owns its TensorFlow state and its two RNG streams."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.experimental.enable_op_determinism()


def fit_sample_efficiency(frame, config, indices, reference, count, seed, output_dir, reused=False):
    """One private count/seed fit; return compact validation rows to the parent."""
    output_dir = Path(output_dir)
    subset = {**indices, 'train': indices['train'][:count]}
    preproc = fit_preprocessing(frame, config, subset['train'])
    destination = output_dir if reused else output_dir / 'sample_efficiency' / str(count) / ('seed_' + str(seed))
    if reused:
        saved = json.loads((output_dir / 'provenance.json').read_text())
        keys = ('fit_seconds', 'epochs_completed', 'best_epoch', 'best_validation_loss',
                'parameter_count', 'warmup_epochs')
        details = {key: saved[key] for key in keys}
        details['seed'] = seed
    else:
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / 'fit.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            _, details, _, _ = fit_model(frame, {**config, 'seed': seed}, subset,
                                         preproc, destination, evaluate_test=False)
        write_json(destination / 'fit_details.json', {**details,
            'train_rows': count, 'validation_rows': len(indices['val']),
            'test_evaluated': False, 'split_seed': config['split_seed'], 'worker_pid': os.getpid()})
        np.save(destination / 'train_indices.npy', subset['train'])
    validation = pd.read_csv(destination / 'validation_predictions.csv')
    np.testing.assert_array_equal(validation['row_index'], indices['val'])
    prediction = validation[['pred_scaled_' + t for t in config['target_cols']]].to_numpy(dtype='float32')
    predicted_log = prediction.astype(float) * preproc['target_log_range'] + preproc['target_log_min']
    reference_pred = (predicted_log - reference['target_log_min']) / reference['target_log_range']
    _, reference_y = transform(frame, reference)
    scores = score_targets(reference_y[indices['val']], reference_pred, reference,
                           frame.iloc[indices['val']][config['target_cols']].to_numpy())
    write_json(destination / 'reference_validation_metrics.json', scores)
    return [{'training_samples': count, 'target': target, 'seed': seed,
        **{'validation_' + space + '_' + metric: value
           for space in ('scaled', 'physical') for metric, value in scores[space][target].items()},
        'epochs': details['epochs_completed'], 'best_epoch': details['best_epoch'],
        'fit_seconds': details['fit_seconds'], 'reused_main_fit': reused,
        'fit_directory': str(destination.relative_to(output_dir))}
        for target in config['target_cols']]


def sample_efficiency(frame, config, indices, reference, sizes, output_dir, repeat_seeds,
                      workers=1, threads=2):
    """Validation-only curve with fixed nested data and isolated model-seed fits.

    The main model is completed and timed before any pool starts. Worker processes
    use spawn, never fork TensorFlow or share global RNG state. Only the parent
    writes progress and the final, deterministically sorted learning-curve table.
    """
    output_dir = Path(output_dir)
    sizes = sorted(set(sizes))
    if any(count < 2 or count > len(indices['train']) for count in sizes):
        raise ValueError('Sample-efficiency counts must be between 2 and {} training rows'.format(len(indices['train'])))
    records = []
    total = len(sizes) * len(repeat_seeds)
    reuse_main = len(indices['train']) in sizes and config['seed'] in repeat_seeds
    protocol = {
        'training_counts': sizes, 'repeat_seeds': repeat_seeds,
        'split_seed': config['split_seed'], 'nested_subset_seed': config['split_seed'],
        'main_model_seed': config['seed'], 'selection_split': 'validation',
        'subsets': 'Nested prefixes of the fixed training split; preprocessing fitted on each subset only.',
        'seed_variation': 'Model initialization and epoch shuffling; data splits and training subsets stay fixed.',
        'holdouts': 'Fixed validation rows at every count/seed. Subset fits do not evaluate the test set.',
        'scaled_metrics': 'Mapped into the full-training-pool natural-log minmax transform for comparison.',
        'checkpoint_selection': 'Best mean validation Huber loss across all epochs in each fit native target scaling.',
        'epochs_max': config['epochs'], 'warmup_epochs': config.get('warmup_epochs', 0),
        'patience': config['patience'], 'main_full_size_fit_reused': reuse_main,
        'sweep_workers': workers, 'worker_threads': threads,
        'multiprocessing_start_method': 'spawn' if workers > 1 else None,
        'job_schedule': 'Larger training subsets first; final rows sorted by training_samples, seed, target.',
        'uncertainty': 'Variation across optimization seeds conditional on one data split; not data-sampling confidence intervals.',
        'interpretation': 'A validation learning curve; any estimated knee is not a proven minimum or saturation threshold.',
        'completed': False, 'completed_fits': 0, 'total_fits': total}
    write_json(output_dir / 'sample_efficiency_protocol.json', protocol)

    def accept(rows):
        records.extend(rows)
        protocol['completed_fits'] += 1
        recent = {key: rows[0][key] for key in ('training_samples', 'seed', 'epochs', 'best_epoch')}
        write_json(output_dir / 'sample_efficiency_progress.json', {
            'completed_fits': protocol['completed_fits'], 'total_fits': total,
            'completed': False, 'recent_fit': recent})
        print('Sample-efficiency completed {}/{}: {} rows, seed {}, best epoch {}'.format(
            protocol['completed_fits'], total, recent['training_samples'], recent['seed'], recent['best_epoch']), flush=True)

    if reuse_main:
        accept(fit_sample_efficiency(frame, config, indices, reference,
                                     len(indices['train']), config['seed'], output_dir, reused=True))
    jobs = [(count, seed) for count in reversed(sizes) for seed in repeat_seeds
            if not (reuse_main and count == len(indices['train']) and seed == config['seed'])]
    try:
        if workers == 1:
            for count, seed in jobs:
                accept(fit_sample_efficiency(frame, config, indices, reference, count, seed, output_dir))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_training_worker, initargs=(threads,)) as pool:
                futures = [pool.submit(fit_sample_efficiency, frame, config, indices, reference,
                                       count, seed, output_dir) for count, seed in jobs]
                try:
                    for future in as_completed(futures):
                        accept(future.result())
                except BaseException:
                    for pending in futures:
                        pending.cancel()
                    raise
    except BaseException as exc:
        protocol['error'] = str(exc)
        write_json(output_dir / 'sample_efficiency_protocol.json', protocol)
        raise
    # Publish the final table only when every fit succeeds; timing is not a metric.
    records.sort(key=lambda row: (row['training_samples'], row['seed'], row['target']))
    temporary_csv = output_dir / 'sample_efficiency.csv.tmp'
    pd.DataFrame(records).to_csv(temporary_csv, index=False)
    temporary_csv.replace(output_dir / 'sample_efficiency.csv')
    protocol['completed'] = True
    write_json(output_dir / 'sample_efficiency_protocol.json', protocol)
    write_json(output_dir / 'sample_efficiency_progress.json', {
        'completed_fits': protocol['completed_fits'], 'total_fits': total, 'completed': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('train_config.yaml'))
    parser.add_argument('--data', type=Path, required=True, help='One CSV with all seven input columns; legacy four-input data is rejected')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--seed', type=int, help='Predeclared main model/epoch-shuffle seed; does not change data splits')
    parser.add_argument('--split-seed', type=int, help='Fixed data split and nested-subset order seed')
    parser.add_argument('--warmup-epochs', type=int, help='Delay early-stopping monitoring; checkpoint selection still includes all epochs')
    parser.add_argument('--repeat-seeds', help='Comma-separated initialization/shuffle seeds for validation learning curves')
    parser.add_argument('--threads', type=int, default=2, help='TensorFlow intra/inter-op thread limits per process')
    parser.add_argument('--sweep-workers', type=int, default=1, help='Spawned worker processes for validation learning-curve fits')
    parser.add_argument('--sample-efficiency', default='', help='Comma-separated TRAINING row counts; optional and slower')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ('epochs', 'patience', 'seed', 'split_seed', 'warmup_epochs'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config.setdefault('split_seed', 42)
    config.setdefault('warmup_epochs', 0)
    repeat_seeds = list(dict.fromkeys(
        [int(value.strip()) for value in args.repeat_seeds.split(',') if value.strip()]
        if args.repeat_seeds is not None else config.get('repeat_seeds', [config['seed']])))
    if not repeat_seeds or any(seed < 0 for seed in repeat_seeds + [config['seed'], config['split_seed']]):
        parser.error('Provide at least one nonnegative repeat seed; all seeds must be nonnegative')
    config['repeat_seeds'] = repeat_seeds
    if min(args.threads, args.sweep_workers, config['epochs']) < 1 or min(config['patience'], config['warmup_epochs']) < 0:
        parser.error('threads/sweep-workers/epochs must be positive and patience/warmup-epochs must be nonnegative')
    sizes = [int(value.strip()) for value in args.sample_efficiency.split(',') if value.strip()]
    frame = load_dataset(args.data, config)
    indices = split_indices(frame, config['input_cols'], config['split_seed'], config['test_size'], config['val_size'])
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
        'threads': args.threads, 'sweep_workers': args.sweep_workers,
        'sample_efficiency_counts': sizes,
        'model_seed': config['seed'], 'split_seed': config['split_seed'],
        'sample_efficiency_repeat_seeds': repeat_seeds,
        'checkpoint_selection': 'Best validation Huber across all epochs, including the early-stopping warmup.',
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
        sample_efficiency(frame, config, indices, preproc, sizes, output_dir, repeat_seeds,
                          workers=args.sweep_workers, threads=args.threads)
    print('Saved training artifacts to {}'.format(output_dir), flush=True)


if __name__ == '__main__':
    main()
