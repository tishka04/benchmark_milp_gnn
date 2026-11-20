"""Diagnose the B9 contribution in an existing interim snapshot, using frozen coefficients."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ..data import read
from ..realized.pipeline import verify_freeze
from ..realized.storage import DEST


def diagnose(snapshot, previous):
    freeze = verify_freeze()
    assert freeze == read(snapshot / 'snapshot.json')['frozen']
    features = pd.read_parquet(DEST / 'features.parquet').set_index('group', verify_integrity=True)
    training = features.loc[features.population.eq('historical')]
    np.testing.assert_allclose(training.real_B9, 1 - 1 / training.real_B1, atol=1e-12, rtol=0)
    low, high = training.real_B9.min(), training.real_B9.max()
    models = read(DEST / 'frozen_models.json')
    saved = pd.read_parquet(snapshot / 'matched_predictions.parquet')
    primary = saved[saved.kind.isin(['positive', 'logistic'])]
    keys = ['group', 'budget', 'batch', 'target', 'kind', 'y']
    paired = primary[primary.representation.eq('legacy_raw')][keys + ['p']].rename(columns={'p': 'legacy'})
    paired = paired.merge(primary[primary.representation.eq('realized')][keys + ['p']].rename(
        columns={'p': 'milp'}), on=keys, how='inner', validate='one_to_one')
    observed = pd.read_csv(snapshot / 'observations.csv')
    assert len(paired) == len(observed) * 7
    pieces = []
    for (target, kind, budget), part in paired.groupby(['target', 'kind', 'budget']):
        model = models[f'{target}/realized/{kind}'][str(budget)]
        part = part.copy()
        actual = features.loc[part.group, model['columns']]
        part['b9'] = actual.real_B9.to_numpy()
        part['outside'] = (part.b9 < low) | (part.b9 > high)
        part['below'] = part.b9 < low
        part['above'] = part.b9 > high
        if 'constant' in model:
            np.testing.assert_allclose(part.milp, model['constant'], atol=1e-12, rtol=1e-12)
            part['b9_contribution'] = 0.
            part['milp_b9_in_range'] = part.milp
        else:
            columns = model['columns']
            pos = columns.index('real_B9')
            z = (actual.to_numpy() - model['mean']) / model['scale']
            z[:, ~np.asarray(model['active'], dtype=bool)] = 0.
            terms = z * np.asarray(model['coefficients'])
            eta = model['intercept'] + terms.sum(axis=1)
            link = expit if kind == 'logistic' else np.asarray
            np.testing.assert_allclose(link(eta), part.milp, atol=1e-12, rtol=1e-12)
            part['b9_contribution'] = terms[:, pos]
            delta = (part.b9.clip(low, high).to_numpy() - part.b9.to_numpy()) / model['scale'][pos]
            part['milp_b9_in_range'] = link(eta + delta * model['coefficients'][pos])
        pieces.append(part)
    rows = pd.concat(pieces, ignore_index=True)
    for arm in ['legacy', 'milp', 'milp_b9_in_range']:
        rows[arm + '_raw_loss'] = (rows[arm] - rows.y)**2
        rows[arm + '_clipped_loss'] = (rows[arm].clip(0, 1) - rows.y)**2
    np.testing.assert_allclose(rows.loc[~rows.outside, 'milp_b9_in_range'],
                               rows.loc[~rows.outside, 'milp'], rtol=1e-12, atol=1e-12)
    rows = rows.merge(observed[['group', 'budget', 'job_id', 'reference_certified']],
                      on=['group', 'budget'], validate='many_to_one')
    old = pd.read_csv(previous / 'observations.csv')
    assert set(old.job_id) <= set(observed.job_id)
    old_matches = pd.read_parquet(previous / 'matched_predictions.parquet')
    shared = saved[saved.representation.isin(['legacy_raw', 'realized'])].merge(
        old_matches[old_matches.representation.isin(['legacy_raw', 'realized'])],
        on=['group', 'budget', 'target', 'representation', 'kind'], validate='one_to_one')
    np.testing.assert_allclose(shared[['y_x', 'p_x']], shared[['y_y', 'p_y']], atol=1e-12, rtol=0)
    scopes = {'all': rows, 'in_range': rows[~rows.outside], 'out_of_range': rows[rows.outside],
              'new_trials': rows[~rows.job_id.isin(old.job_id)],
              'new_scenarios': rows[~rows.group.isin(old.group)],
              'certified': rows[rows.reference_certified],
              'excluding_topology0043': rows[rows.group.ne('external/topology/0043')]}
    scopes.update({f'batch:{b}': q for b, q in rows.groupby('batch')})
    scopes.update({f'budget:{b}': q for b, q in rows.groupby('budget')})
    metrics = [k for k in rows.columns if k.endswith('_loss')]
    scores = []
    for scope, local in scopes.items():
        for target, part in local.groupby('target'):
            for weighting, means in [('trial', part[metrics].mean()),
                                      ('scenario', part.groupby('group')[metrics].mean().mean())]:
                scores.append(dict(scope=scope, target=target, weighting=weighting,
                                   trials=len(part), scenarios=part.group.nunique(), **means.to_dict()))
    assert verify_freeze() == freeze
    rows.to_parquet(snapshot / 'b9_paired_diagnostic.parquet', index=False)
    pd.DataFrame(scores).to_csv(snapshot / 'b9_sensitivity.csv', index=False)
    print(f'{len(observed)} trials, {observed.group.nunique()} scenarios, historical B9 [{low}, {high}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    diagnose(args.snapshot, args.previous)
