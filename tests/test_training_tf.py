"""Regression check of actual per-epoch row delivery in the production TF pipeline."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

TRAIN_PATH = Path(__file__).resolve().parents[1] / 'train.py'
if not TRAIN_PATH.exists():
    TRAIN_PATH = Path(__file__).with_name('train.py')
spec = importlib.util.spec_from_file_location('training', TRAIN_PATH)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class RecordedModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.orders = [[], [], []]
        self.epoch = 0

    def call(self, inputs):
        return tf.zeros((tf.shape(inputs)[0], 1))

    def train_step(self, data):
        x, _ = data
        self.orders[self.epoch].extend(x[:, 0].numpy().astype(int).tolist())
        return {'loss': tf.constant(0.)}


class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.epoch = epoch


class EpochShuffleTests(unittest.TestCase):
    def test_actual_fit_reshuffles_and_reproduces(self):
        def record():
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            tf.config.experimental.enable_op_determinism()
            x = np.arange(96, dtype='float32').reshape(-1, 1)
            y = np.zeros((96, 1), dtype='float32')
            dataset = train.make_dataset(x, y, np.arange(96),
                                        {'batch_size': 16, 'seed': 42}, training=True)
            model = RecordedModel()
            model.compile(optimizer='sgd', loss='mse', run_eagerly=True)
            # One fit, three epochs: do not reset seeds or rebuild the stream per epoch.
            model.fit(dataset, epochs=3, callbacks=[EpochTracker()], shuffle=False, verbose=0)
            return model.orders

        first, second = record(), record()
        for order in first:
            self.assertEqual(sorted(order), list(range(96)))
        self.assertNotEqual(first[0], first[1])
        self.assertNotEqual(first[1], first[2])
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
