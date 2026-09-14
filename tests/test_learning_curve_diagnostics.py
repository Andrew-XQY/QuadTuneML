"""Check the predeclared marker guards and validation-only data selection."""
import json
import unittest

import numpy as np
import pandas as pd

from learning_curve_diagnostics import candidate_knee, summarize_curve

COUNTS = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7500, 9000, 9600]
ERRORS = [.4, .29, .2, .16, .15, .145, .14, .138, .136, .135]


def curve():
    return pd.DataFrame([
        {'training_samples': n, 'target': t, 'seed': s, 'validation_scaled_RMSE': e * factor}
        for t in ('emittance_x', 'emittance_y')
        for s, factor in ((42, .97), (43, 1), (44, 1.03))
        for n, e in zip(COUNTS, ERRORS)
    ])


class KneeTests(unittest.TestCase):
    def test_known_diminishing_returns_and_remaining_benefit(self):
        result = summarize_curve(curve())['targets']['emittance_x']
        self.assertTrue(result['display_knee'])
        self.assertEqual(result['knee'], 3000)
        self.assertAlmostEqual(result['remaining_relative_improvement'], .15625)
        self.assertEqual(result['supporting_seeds'], 3)

    def test_zero_error_tail_has_defined_json_without_relative_division(self):
        frame = curve()
        by_count = dict(zip(COUNTS, [1, .8, .2, 0, 0, 0, 0, 0, 0, 0]))
        frame['validation_scaled_RMSE'] = frame.training_samples.map(by_count)
        summary = summarize_curve(frame)
        result = summary['targets']['emittance_x']
        self.assertEqual(result['knee'], 3000)
        self.assertTrue(result['display_knee'])
        self.assertIsNone(result['remaining_relative_improvement'])
        json.dumps(summary, allow_nan=False)

    def test_row_order_and_common_positive_scale_do_not_move_knee(self):
        frame = curve()
        expected = summarize_curve(frame)
        self.assertEqual(summarize_curve(frame.sample(frac=1, random_state=7)), expected)
        # A shared target normalization must not change the geometric knee.
        frame['validation_scaled_RMSE'] *= 0.01
        scaled = summarize_curve(frame)
        for target in expected['targets']:
            first, second = expected['targets'][target], scaled['targets'][target]
            self.assertEqual(first['knee'], second['knee'])
            self.assertEqual(first['display_knee'], second['display_knee'])
            self.assertAlmostEqual(first['remaining_relative_improvement'], second['remaining_relative_improvement'])

    def test_flat_linear_and_increasing_have_no_marker(self):
        for errors in ([.4] * 10, [1 - n / 20000 for n in COUNTS], np.arange(10)):
            self.assertIsNone(candidate_knee(COUNTS, errors)[0])

    def test_two_later_sizes_required(self):
        counts = [1, 2, 3, 4, 5]
        self.assertIsNone(candidate_knee(counts, [1, .9, .8, .1, .099])[0])

    def test_unstable_seeds_suppress_marker(self):
        frame = curve()
        # The mean still bends, but two seeds have no identifiable knee.
        for seed in (43, 44):
            frame.loc[frame.seed == seed, 'validation_scaled_RMSE'] = .15
        result = summarize_curve(frame)['targets']['emittance_x']
        self.assertEqual(result['knee'], 3000)
        self.assertEqual(result['status'], 'unstable')
        self.assertFalse(result['display_knee'])

    def test_invalid_grids_and_values_rejected(self):
        frame = curve()
        invalid = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame[frame.seed != 44]]
        bad = frame.copy(); bad.loc[0, 'validation_scaled_RMSE'] = np.inf; invalid.append(bad)
        bad = frame.copy(); bad.loc[0, 'validation_scaled_RMSE'] = -1; invalid.append(bad)
        for data in invalid:
            with self.assertRaises(ValueError):
                summarize_curve(data)

    def test_test_results_cannot_change_selection(self):
        frame = curve()
        expected = summarize_curve(frame)
        frame['test_scaled_RMSE'] = np.random.default_rng(17).normal(size=len(frame))
        frame['test_scaled_R2'] = -1000
        frame['scaled_RMSE'] = 99999  # legacy test-metric name is also ignored
        frame['physical_RMSE'] = 0
        self.assertEqual(summarize_curve(frame), expected)


if __name__ == '__main__':
    unittest.main()
