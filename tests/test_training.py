"""Checks for leakage, coordinate shape and reversible log preprocessing (no TF)."""
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Works from repository tests/ and from a same-directory staging location.
TRAIN_PATH = Path(__file__).resolve().parents[1] / 'train.py'
if not TRAIN_PATH.exists():
    TRAIN_PATH = Path(__file__).with_name('train.py')
spec = importlib.util.spec_from_file_location('training', TRAIN_PATH)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.config = {'input_cols': ['q1', 'q2', 'q3', 'q4', 'alfx', 'alfy', 'sigma_delta'],
                       'target_cols': ['emittance_x', 'emittance_y'], 'segment': 7}
        rng = np.random.default_rng(5)
        self.frame = pd.DataFrame(rng.uniform(0.1, 1, (40, 7)), columns=self.config['input_cols'])
        self.frame['emittance_x'] = np.exp(rng.uniform(-10, -8, 40))
        self.frame['emittance_y'] = np.exp(rng.uniform(-11, -7, 40))

    def test_progress_json_is_atomic_for_live_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            train.write_json(path, {'completed_fits': 0})
            done = threading.Event()

            def publish():
                try:
                    for count in range(100):
                        train.write_json(path, {'completed_fits': count})
                finally:
                    done.set()

            worker = threading.Thread(target=publish)
            worker.start()
            try:
                while not done.is_set():
                    # Truncate-then-write intermittently exposes invalid JSON.
                    self.assertIn('completed_fits', json.loads(path.read_text()))
            finally:
                worker.join()
            self.assertEqual(json.loads(path.read_text())['completed_fits'], 99)

    def test_duplicate_settings_stay_in_one_split(self):
        repeated = pd.concat([self.frame, self.frame], ignore_index=True)
        splits = train.split_indices(repeated, self.config['input_cols'], seed=42)
        seen = []
        for rows in splits.values():
            seen.append(set(map(tuple, repeated.iloc[rows][self.config['input_cols']].to_numpy())))
        for i in range(len(seen)):
            for j in range(i):
                self.assertFalse(seen[i] & seen[j])
        self.assertEqual(sum(map(len, splits.values())), len(repeated))
        again = train.split_indices(repeated, self.config['input_cols'], seed=42)
        for key in splits:
            np.testing.assert_array_equal(splits[key], again[key])

    def test_train_only_scaling_roundtrip_and_no_clipping(self):
        self.frame.loc[39, 'q1'] = 1e6  # held-out extreme must not affect train bounds
        self.frame.loc[39, 'emittance_x'] = 10
        prep = train.fit_preprocessing(self.frame, self.config, np.arange(30))
        self.assertLess(prep['input_range'][0], 1)
        x, y = train.transform(self.frame, prep)
        self.assertGreater(x[39, 0], 1)
        self.assertGreater(y[39, 0], 1)
        np.testing.assert_allclose(train.inverse_targets(y, prep),
            self.frame[self.config['target_cols']].to_numpy(), rtol=2e-6)

    def test_constant_feature_is_finite(self):
        self.frame['alfx'] = 2.6
        prep = train.fit_preprocessing(self.frame, self.config, np.arange(30))
        x, y = train.transform(self.frame, prep)
        self.assertIn('alfx', prep['constant_input_cols'])
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())

    def test_csv_requires_seven_inputs_and_eight_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            raw = self.frame.copy()
            for target in self.config['target_cols']:
                raw[target] = raw[target].map(lambda value: str([value / 2] * 7 + [value]))
            raw.to_csv(path, index=False)
            loaded = train.load_dataset(path, self.config)
            np.testing.assert_allclose(loaded, self.frame)
            raw.drop(columns=['sigma_delta']).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'sigma_delta'):
                train.load_dataset(path, self.config)
            raw.loc[0, 'emittance_x'] = '[1, 2]'
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'eight observation'):
                train.load_dataset(path, self.config)

    def test_log_target_rejects_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            raw = self.frame.copy()
            for target in self.config['target_cols']:
                raw[target] = raw[target].map(lambda value: str([value] * 8))
            raw.loc[0, 'emittance_x'] = str([0.] * 8)
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'nonpositive'):
                train.load_dataset(path, self.config)

    def test_shared_reference_scaling(self):
        subset = train.fit_preprocessing(self.frame, self.config, np.arange(10))
        reference = train.fit_preprocessing(self.frame, self.config, np.arange(30))
        _, subset_y = train.transform(self.frame, subset)
        _, reference_y = train.transform(self.frame, reference)
        physical_log = subset_y.astype(float) * subset['target_log_range'] + subset['target_log_min']
        mapped = (physical_log - reference['target_log_min']) / reference['target_log_range']
        np.testing.assert_allclose(mapped, reference_y, atol=1e-7)
        score = train.score_targets(reference_y, mapped, reference)
        self.assertGreater(score['scaled']['emittance_x']['R2'], 0.999999)


if __name__ == '__main__':
    unittest.main()
