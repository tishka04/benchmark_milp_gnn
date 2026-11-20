"""Synthetic tests only. No campaign, router, or real dispatch is executed."""
import ast
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research.paper1_weights import data as d
from research.paper1_weights.model import (arrays, fit, predict, group_folds, subset,
                                         cv_job, bootstrap_job)
from research.paper1_weights.parallel import bounded_map


def square(value):
    return value * value


def tiny_rows(groups=20):
    rng = np.random.default_rng(7)
    out = []
    for i in range(groups):
        x = rng.uniform(0, 1, len(d.COMPONENTS))
        x[13], x[16] = 24 / 312, .5
        for b in (5, 10, 30, 60, 120):
            row = dict(zip(d.COMPONENTS, x.tolist()))
            row.update(key=f'case_{i}', group=f'case_{i}', family=f'family_{i % 2}',
                       budget=b, method='highs', repetition=0, seed=i,
                       partition='historical_development', condition='historical', observed=True,
                       feasible=True, exit_s=b * .5, cost=105., reference_cost=100.,
                       reference_certified=True, ens_mwh=0., demand_mwh=1000.,
                       served_mwh=1000., co2_t_per_served_mwh=.1)
            row['targets'] = {'synthetic': {'score': .05 + b / 1200 + .2 * x[0] + .5 * x[2]}}
            out.append(row)
    return out


class Targets(unittest.TestCase):
    def setUp(self):
        self.cfg = d.config()
        self.row = tiny_rows()[0]

    def test_independent_deadline_normalization(self):
        first = d.target(self.row, self.cfg, self.cfg['policies']['service3'])
        second = d.target(dict(self.row, budget=120, exit_s=60), self.cfg, self.cfg['policies']['service3'])
        self.assertEqual(first['time_term'], .5)
        self.assertEqual(second['time_term'], .5)

    def test_negative_and_near_zero_references(self):
        out = d.target(dict(self.row, reference_cost=-100, cost=-90), self.cfg, self.cfg['policies']['service3'])
        self.assertAlmostEqual(out['cost_term'], .1 / (.1 + .05))
        out = d.target(dict(self.row, reference_cost=0, cost=.05), self.cfg, self.cfg['policies']['service3'])
        self.assertAlmostEqual(out['cost_term'], .5)

    def test_better_than_reference_is_not_penalized(self):
        out = d.target(dict(self.row, reference_cost=100, cost=90), self.cfg, self.cfg['policies']['service3'])
        self.assertEqual(out['cost_term'], 0)

    def test_failure_is_not_zero_carbon_or_zero_ens(self):
        out = d.target(dict(self.row, feasible=False, cost=None, ens_mwh=None,
                            co2_t_per_served_mwh=None), self.cfg, self.cfg['policies']['service4_co2'])
        self.assertEqual(out['score'], 1)
        self.assertIsNone(out['ens_term'])
        self.assertIsNone(out['co2_term'])

    def test_ens_priority_is_relative(self):
        policy = self.cfg['policies']['service3']
        a = d.target(self.row, self.cfg, policy)
        b = d.target(dict(self.row, ens_mwh=100), self.cfg, policy)
        self.assertGreater(b['score'], a['score'])
        self.assertTrue(0 <= b['score'] <= 1)
        self.assertGreater(policy['ens'], policy['cost'])

    def test_no_service_gets_no_carbon_reward(self):
        out = d.target(dict(self.row, ens_mwh=1000, served_mwh=0,
                            co2_t_per_served_mwh=None), self.cfg, self.cfg['policies']['service4_co2'])
        self.assertEqual(out['co2_term'], 1)

    def test_carbon_uses_dt_and_no_storage_double_count(self):
        dispatch = dict(zones=['z1', 'z2'], periods=[0, 1], values=dict(
            p_thermal=[[100., 0.], [0., 200.]], net_import=[20., 30.], net_export=[1., 2.],
            b_discharge=[[999., 999.], [888., 888.]]))
        out = d.dispatch_carbon(dispatch, .5, 200., 20., self.cfg)
        self.assertEqual(out['thermal_mwh'], 150)
        self.assertAlmostEqual(out['co2_direct_t'], 108)
        self.assertAlmostEqual(out['co2_t_per_served_mwh'], .6)
        self.assertEqual(out['imports_mwh'], 25)
        self.assertEqual(out['exports_mwh'], 1.5)
        self.assertFalse(out['import_carbon_known'])

    def test_system_imports_exports_require_one_value_per_period(self):
        base = dict(zones=['z1', 'z2'], periods=[0, 1], values=dict(
            p_thermal=[[100., 0.], [0., 200.]], net_import=[20., 30.], net_export=[1., 2.]))
        for variable in ('net_import', 'net_export'):
            for invalid in ([1.], [[1., 2.], [3., 4.]]):
                with self.subTest(variable=variable, invalid=invalid):
                    dispatch = copy.deepcopy(base)
                    dispatch['values'][variable] = invalid
                    with self.assertRaisesRegex(ValueError, 'Wrong dispatch shape: ' + variable):
                        d.dispatch_carbon(dispatch, .5, 200., 20., self.cfg)

    def test_thermal_generation_requires_zone_by_period_matrix(self):
        base = dict(zones=['z1', 'z2'], periods=[0, 1], values=dict(
            p_thermal=[[100., 0.], [0., 200.]], net_import=[20., 30.], net_export=[1., 2.]))
        for invalid in ([100., 200.], [[100., 200.]], [[100.], [200.]]):
            with self.subTest(invalid=invalid):
                dispatch = copy.deepcopy(base)
                dispatch['values']['p_thermal'] = invalid
                with self.assertRaisesRegex(ValueError, 'Wrong dispatch shape: p_thermal'):
                    d.dispatch_carbon(dispatch, .5, 200., 20., self.cfg)

    def test_missing_thermal_dispatch_rejected(self):
        with self.assertRaises(KeyError):
            d.dispatch_carbon(dict(zones=['z'], periods=[0], values={}), 1, 10, 0, self.cfg)

    def test_equal_and_each_priority_profile_are_valid(self):
        expected = {
            'equal4': {'ens': .25, 'cost': .25, 'time': .25, 'co2': .25},
            'time_priority': {'ens': 2 / 15, 'cost': 2 / 15, 'time': .6, 'co2': 2 / 15},
            'cost_priority': {'ens': 2 / 15, 'cost': .6, 'time': 2 / 15, 'co2': 2 / 15},
            'co2_priority': {'ens': 2 / 15, 'cost': 2 / 15, 'time': 2 / 15, 'co2': .6},
        }
        for name, policy in expected.items():
            with self.subTest(policy=name):
                self.assertEqual(self.cfg['policies'][name], policy)
                cfg = copy.deepcopy(self.cfg)
                cfg['policies'] = {name: policy}
                d.validate_config(cfg)

    def test_equal_normalized_terms_have_same_score_for_all_profiles(self):
        row = dict(self.row, exit_s=self.row['budget'] / 2, reference_cost=100., cost=105.,
                   ens_mwh=10.001, co2_t_per_served_mwh=.72)
        for name, policy in self.cfg['policies'].items():
            with self.subTest(policy=name):
                out = d.target(row, self.cfg, policy)
                for criterion in ('ens', 'cost', 'time', 'co2'):
                    self.assertAlmostEqual(out[criterion + '_term'], .5)
                self.assertAlmostEqual(out['score'], .5)

    def test_each_priority_controls_the_target_response(self):
        baseline = dict(self.row, exit_s=0., cost=100., reference_cost=100.,
                        ens_mwh=0., co2_t_per_served_mwh=0.)
        changes = {'ens': {'ens_mwh': 10.001}, 'cost': {'cost': 105.},
                   'time': {'exit_s': self.row['budget'] / 2},
                   'co2': {'co2_t_per_served_mwh': .72}}
        for name, preferred in (('service4_co2', 'ens'), ('time_priority', 'time'),
                                ('cost_priority', 'cost'), ('co2_priority', 'co2')):
            with self.subTest(policy=name):
                policy = self.cfg['policies'][name]
                gains = {criterion: d.target(dict(baseline, **change), self.cfg, policy)['score']
                         for criterion, change in changes.items()}
                self.assertAlmostEqual(gains[preferred], .3)
                for criterion in changes:
                    if criterion != preferred:
                        self.assertAlmostEqual(gains[criterion], 1 / 15)
                        self.assertGreater(gains[preferred], gains[criterion])
        equal = self.cfg['policies']['equal4']
        for change in changes.values():
            self.assertAlmostEqual(d.target(dict(baseline, **change), self.cfg, equal)['score'], .125)

    def test_empty_or_invalid_policy_sets_are_rejected(self):
        bad = [
            {},
            {'invalid': {'ens': -.1, 'cost': .4, 'time': .4, 'co2': .3}},
            {'invalid': {'ens': float('nan'), 'cost': .25, 'time': .25, 'co2': .25}},
            {'invalid': {'ens': .25, 'cost': .25, 'time': .25, 'co2': .5}},
        ]
        for policies in bad:
            with self.subTest(policies=policies):
                cfg = copy.deepcopy(self.cfg); cfg['policies'] = policies
                with self.assertRaises(ValueError):
                    d.validate_config(cfg)

    def test_no_new_time_budget_or_bad_policy(self):
        cfg = copy.deepcopy(self.cfg); cfg['deadlines_s'] = [5, 120]
        with self.assertRaises(ValueError):
            d.validate_config(cfg)
        cfg = copy.deepcopy(self.cfg); cfg['policies']['service3']['ens'] = .1
        with self.assertRaises(ValueError):
            d.validate_config(cfg)


class Regression(unittest.TestCase):
    def setUp(self):
        self.cfg = d.config()
        self.rows = tiny_rows(80)
        self.data = arrays(self.rows, 'synthetic')

    def test_known_shared_slopes_and_deadline_intercepts(self):
        model = fit(self.data, 'positive', 0, self.cfg)
        np.testing.assert_allclose(predict(model, self.data), self.data['y'], atol=1e-7)
        self.assertAlmostEqual(model['coefficients'][0], .2, places=6)
        self.assertAlmostEqual(model['coefficients'][2], .5, places=6)
        self.assertAlmostEqual(sum(model['index_weights']), 1)
        self.assertGreater(model['intercepts']['120'], model['intercepts']['5'])

    def test_constants_not_estimated(self):
        model = fit(self.data, 'positive', .001, self.cfg)
        for c in ('B2_horizon', 'B5_min_gen_tight'):
            j = d.COMPONENTS.index(c)
            self.assertFalse(model['active'][j])
            self.assertEqual(model['coefficients'][j], 0)

    def test_all_deadlines_of_group_stay_in_one_fold(self):
        folds = group_folds(self.data, 5, 1, 'test')
        for g in np.unique(self.data['group']):
            self.assertEqual(len(set(folds[self.data['group'] == g])), 1)
        train = subset(self.data, folds != 0)
        test = subset(self.data, folds == 0)
        self.assertFalse(set(train['group']) & set(test['group']))

    def test_nested_validation_returns_only_held_out_groups(self):
        cfg = dict(self.cfg, ridge_grid=[0, .01])
        spec = dict(name='common_positive', kind='positive', deadline=None)
        out = cv_job(('job', self.data, spec, 0, cfg))
        folds = group_folds(self.data, 5, cfg['seed'], 'outer')
        self.assertEqual({r['group'] for r in out['predictions']}, set(self.data['group'][folds == 0]))

    def test_signed_diagnostic_preserves_negative_slope(self):
        data = dict(self.data, y=.5 - .3 * self.data['x'][:, 0])
        signed = fit(data, 'signed', 0, self.cfg)
        positive = fit(data, 'positive', 0, self.cfg)
        self.assertLess(signed['coefficients'][0], -.29)
        self.assertTrue(all(c >= 0 for c in positive['coefficients']))
        self.assertLess(np.mean((predict(signed, data) - data['y']) ** 2),
                        np.mean((predict(positive, data) - data['y']) ** 2))

    def test_degenerate_index_is_not_uniform_weights(self):
        data = dict(self.data, y=np.zeros(len(self.data['y'])))
        model = fit(data, 'positive', .1, self.cfg)
        self.assertTrue(model['degenerate_index'])
        self.assertIsNone(model['index_weights'])

    def test_bootstrap_reproducible_and_grouped(self):
        spec = dict(name='common_positive', kind='positive', deadline=None)
        model = fit(self.data, 'positive', .001, self.cfg)
        task = ('job', self.data, spec, model, 0, 2, self.cfg)
        self.assertEqual(bootstrap_job(task), bootstrap_job(task))

    def test_original_index_baseline_uses_declared_weights(self):
        self.assertAlmostEqual(sum(d.ORIGINAL_WEIGHTS), 1)
        model = fit(self.data, 'original_index', 0, self.cfg)
        self.assertEqual(model['names'], ['original_index'])
        self.assertEqual(len(predict(model, self.data)), len(self.rows))


class Provenance(unittest.TestCase):
    def test_seal_detects_changed_data(self):
        obj = d.seal(dict(a=1)); d.verify(obj)
        obj['a'] = 2
        with self.assertRaises(ValueError):
            d.verify(obj)

    def test_writes_are_local_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(d, 'HOME', Path(directory)):
                path = Path(directory) / 'runs/test/result.json'
                d.put(path, {'ok': True}); d.put(path, {'ok': True})
                with self.assertRaises(ValueError):
                    d.put(path, {'ok': False})
                with self.assertRaises(ValueError):
                    d.put(Path(directory) / 'outside.json', {})

    def test_no_runtime_or_router_imports(self):
        for path in d.HOME.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [v.name for v in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                else:
                    continue
                self.assertFalse(any(n.startswith(('src.', 'research.three_papers', 'research.paper1_components')) for n in names))

    def test_grid_rejects_missing_and_duplicate_rows(self):
        rows = tiny_rows(20)
        cfg = dict(d.config(), expected_scenarios=20, expected_families=2, expected_per_family=10)
        self.assertEqual(len(d.select_rows(rows, cfg)), 100)
        with self.assertRaises(ValueError):
            d.select_rows(rows[:-1], cfg)
        with self.assertRaises(ValueError):
            d.select_rows(rows + [rows[0]], cfg)

    def test_independent_budget_identity(self):
        freeze = dict(manifest_hash='m', calibration={'reserve_s': 30.},
                      reserve_amendment={'kind': 'budget_scaled_finalization_reserve_v1', 'sha256': 'r'})
        row = tiny_rows()[0]
        a = d.observation_identity(dict(row, budget=120), freeze)
        b = d.observation_identity(dict(row, budget=5), freeze)
        self.assertNotIn('reserve_policy', a)
        self.assertEqual(b['reserve_s'], 1.25)
        self.assertNotEqual(d.digest(a), d.digest(b))

    def test_parallel_spawn_matches_serial(self):
        serial = list(bounded_map(square, range(4), 1))
        parallel = list(bounded_map(square, range(4), 2))
        self.assertEqual(sorted(serial), sorted(parallel))


if __name__ == '__main__':
    unittest.main()
