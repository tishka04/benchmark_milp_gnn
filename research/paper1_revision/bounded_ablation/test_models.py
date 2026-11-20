"""Numerical, leakage and exact-ablation checks using synthetic small inputs."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit

from .models import objective, fit_bounded, predict, feature_columns, tune
from ..realized.models import tune as original_tune


class BoundedModelsTest(unittest.TestCase):
    def test_soft_label_gradient(self):
        rng = np.random.default_rng(14)
        matrix = np.c_[np.ones(40), rng.normal(size=(40, 3))]
        y = np.r_[0., 1., rng.uniform(size=38)]
        beta = np.array([.3, .2, .4, .1])
        value, gradient = objective(beta, matrix, y, .02)
        numeric = []
        for j in range(len(beta)):
            delta = np.zeros(len(beta)); delta[j] = 1e-6
            numeric.append((objective(beta + delta, matrix, y, .02)[0]
                            - objective(beta - delta, matrix, y, .02)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, numeric, atol=1e-8)
        self.assertTrue(np.isfinite(value))

    def test_known_fractional_model_and_far_extrapolation(self):
        rng = np.random.default_rng(7)
        x = np.c_[rng.normal(size=(180, 2)), np.full(180, 8.)]
        y = expit(-.2 + .8 * x[:, 0] + .5 * x[:, 1])
        model = fit_bounded(x, y, 0.)
        np.testing.assert_allclose(predict(model, x), y, atol=2e-5)
        self.assertTrue(np.all(np.array(model['coefficients']) >= 0))
        self.assertEqual(model['coefficients'][2], 0.)
        far = np.array([[-1e6, -1e6, -1e9], [1e6, 1e6, 1e9]])
        np.testing.assert_array_equal(predict(model, far), [0., 1.])
        changed = x.copy(); changed[:, 2] = 1e6
        np.testing.assert_array_equal(predict(model, x), predict(model, changed))

    def test_constant_fractional_and_endpoint_labels(self):
        x = np.arange(60).reshape(20, 3)
        for value in [0., .37, 1.]:
            model = fit_bounded(x, np.full(20, value), .1)
            np.testing.assert_array_equal(predict(model, -x), np.full(20, value))

    def test_invalid_training_data(self):
        x = np.ones((20, 2))
        for y in [np.full(20, 1.1), np.full(20, np.nan), np.ones(19)]:
            with self.assertRaises(ValueError):
                fit_bounded(x, y, .1)

    def test_training_preprocessing(self):
        x = np.arange(80).reshape(40, 2).astype(float)
        y = np.linspace(.1, .9, 40)
        model = fit_bounded(x, y, .1)
        np.testing.assert_array_equal(model['mean'], x.mean(axis=0))
        np.testing.assert_array_equal(model['scale'], x.std(axis=0))

    @staticmethod
    def fixture():
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(rng.normal(size=(75, 22)), columns=feature_columns('realized', 'fractional_logit'))
        frame['group'] = ['g' + str(i) for i in range(len(frame))]
        frame['target'] = np.clip(.5 + .1 * frame.real_A1 + .08 * frame.real_B9, 0, 1)
        return frame.iloc[:60].copy(), frame.iloc[60:].copy(), np.arange(60) % 3

    def test_ablation_matches_original_linear_tuning(self):
        train, test, inner = self.fixture()
        cols = feature_columns('realized', 'linear_no_b9_clip')
        expected, old = original_tune(train, test, inner, 'target', cols, 'positive', [0., .1, 1.])
        result, new = tune(train, test, inner, 'target', 'realized', 'linear_no_b9_clip', [0., .1, 1.])
        np.testing.assert_allclose(result, np.clip(expected, 0, 1), atol=1e-14)
        self.assertEqual(old['ridge'], new['ridge'])
        self.assertEqual(len(cols), 21)
        self.assertNotIn('real_B9', new['columns'])

    def test_excluded_b9_and_test_labels_do_not_affect_fits(self):
        train, test, inner = self.fixture()
        args = (inner, 'target', 'realized', 'fractional_logit_no_b9', [.01, .1])
        first, m1 = tune(train, test, *args)
        train.real_B9 *= 1e8
        test.real_B9 = -1e10
        test.target = 1 - test.target
        second, m2 = tune(train, test, *args)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(m1, m2)
        self.assertNotIn('raw_B9', feature_columns('legacy_raw', 'linear_no_b9_clip'))

    def test_scenario_leakage_is_rejected(self):
        train, test, inner = self.fixture()
        test.iloc[0, test.columns.get_loc('group')] = train.group.iloc[0]
        with self.assertRaisesRegex(ValueError, 'Outer scenario leakage'):
            tune(train, test, inner, 'target', 'realized', 'fractional_logit', [.1])
        train, test, inner = self.fixture()
        train.iloc[1, train.columns.get_loc('group')] = train.group.iloc[0]
        with self.assertRaisesRegex(ValueError, 'Inner scenario leakage'):
            tune(train, test, inner, 'target', 'realized', 'fractional_logit', [.1])


if __name__ == '__main__':
    unittest.main()
