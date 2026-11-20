"""Post-hoc diagnosis of historical predictions; no fit, solve or campaign edit.

Run from the repository root with:
python -m research.paper1_revision.diagnostics.bounded_predictions_v1

Outputs are separate from the frozen comparison. Clipping is applied equally
to the legacy and realized predictors, without selecting a threshold on data.
These historical diagnostics do not constitute a new independent validation.
"""
from datetime import datetime, timezone
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import RUN, read, sha
from ..realized.pipeline import verify_freeze
from ..realized.storage import DEST, load_frames, code_hashes


def project_unit_interval(predictions):
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions must be diagnosed, not clipped')
    return np.clip(values, 0., 1.)


def run():
    before = verify_freeze()
    frozen_code = code_hashes()
    now = datetime.now(timezone.utc)
    output = RUN / 'diagnostics' / ('bounded_predictions_v1_' + now.strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    history, _, cfg = load_frames()
    sources = [DEST / 'freeze.json', DEST / 'features.parquet', RUN / 'analysis.parquet']
    rows = []
    for scheme in ['nested', 'lofo']:
        path = DEST / (scheme + '_predictions.parquet')
        sources.append(path)
        saved = pd.read_parquet(path)
        binary = saved[saved.target.isin(['E', 'ES'])]
        assert np.array_equal(binary.p.to_numpy(), project_unit_interval(binary.p))
        q = saved[saved.target.isin(cfg['policies']) & saved.representation.isin(
            ['legacy_raw', 'realized'])].copy()
        assert q.y.between(0, 1).all()
        q['bounded_p'] = project_unit_interval(q.p)
        q['raw_loss'] = (q.p - q.y)**2
        q['bounded_loss'] = (q.bounded_p - q.y)**2
        q['below_zero'] = q.p < 0
        q['above_one'] = q.p > 1
        # Projection onto the known outcome interval cannot increase squared error.
        assert (q.bounded_loss <= q.raw_loss + 1e-14).all()
        q.to_parquet(output / (scheme + '_corrected_predictions.parquet'), index=False)
        rows.append(q)
    predictions = pd.concat(rows, ignore_index=True)
    primary = predictions[predictions.kind == 'positive'].copy()
    keys = ['scheme', 'target', 'representation', 'kind']
    summary = predictions.groupby(keys).agg(
        n=('p', 'size'), raw_mse=('raw_loss', 'mean'), bounded_mse=('bounded_loss', 'mean'),
        below_zero=('below_zero', 'sum'), above_one=('above_one', 'sum'),
        p_min=('p', 'min'), p_max=('p', 'max')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    primary.groupby(['scheme', 'family', 'target', 'representation', 'budget']).agg(
        n=('p', 'size'), raw_mse=('raw_loss', 'mean'), bounded_mse=('bounded_loss', 'mean'),
        below_zero=('below_zero', 'sum'), above_one=('above_one', 'sum'),
        p_min=('p', 'min'), p_max=('p', 'max')).to_csv(output / 'family_deadline_scores.csv')
    comparisons = []
    for (scheme, target), q in summary[summary.kind == 'positive'].groupby(['scheme', 'target']):
        q = q.set_index('representation')
        a, b = q.loc['legacy_raw'], q.loc['realized']
        comparisons.append(dict(scheme=scheme, target=target,
            legacy_raw_mse=a.raw_mse, realized_raw_mse=b.raw_mse,
            legacy_bounded_mse=a.bounded_mse, realized_bounded_mse=b.bounded_mse,
            bounded_realized_gain_percent=100 * (1 - b.bounded_mse / a.bounded_mse)))
    pd.DataFrame(comparisons).to_csv(output / 'matched_comparison.csv', index=False)

    # Identify shift and attribute the low-family predictions using saved models.
    family = 'low_criticality_scenarios'
    train = history[history.family != family].drop_duplicates('group')
    test = history[history.family == family].drop_duplicates('group')
    cols = ['real_A' + str(i) for i in range(1, 13)] + ['real_B' + str(i) for i in range(1, 11)]
    stats = []
    for column in cols:
        low, high = float(train[column].min()), float(train[column].max())
        mean, std = float(train[column].mean()), float(train[column].std(ddof=0))
        z = (test[column] - mean) / std if std > 1e-12 else pd.Series(np.nan, index=test.index)
        stats.append(dict(feature=column, train_min=low, train_max=high,
            train_mean=mean, train_std=std, test_min=float(test[column].min()),
            test_max=float(test[column].max()), test_z_min=float(z.min()), test_z_max=float(z.max()),
            training_constant=std <= 1e-12,
            fraction_outside_training_range=float(((test[column] < low) | (test[column] > high)).mean())))
    pd.DataFrame(stats).to_csv(output / 'low_family_covariate_shift.csv', index=False)
    tree_error = float(abs(history.real_B9 - (1 - 1 / history.real_B1)).max())
    assert tree_error < 1e-12
    contributions, budgets = [], []
    worst = primary[(primary.scheme == 'lofo') & (primary.family == family)
                    & (primary.representation == 'realized')].nsmallest(1, 'p').iloc[0]
    reconstruction_error = 0.
    for target in cfg['policies']:
        path = DEST / 'fits' / 'lofo' / target / (family + '_realized_positive.json')
        sources.append(path)
        obj = read(path)
        assert set(obj['train_groups']) == set(train.group)
        for budget, model in obj['models'].items():
            q = history[(history.family == family) & (history.budget == int(budget))]
            cols = model['columns']
            z = (q[cols].to_numpy() - np.array(model['mean'])) / np.array(model['scale'])
            z[:, ~np.array(model['active'], bool)] = 0.
            components = z * np.array(model['coefficients'])
            p = model['intercept'] + components.sum(axis=1)
            observed = primary[(primary.scheme == 'lofo') & (primary.target == target)
                & (primary.representation == 'realized') & (primary.budget == int(budget))].set_index('group')
            error = float(abs(p - observed.loc[q.group, 'p'].to_numpy()).max())
            reconstruction_error = max(reconstruction_error, error)
            assert error < 1e-12
            b9 = cols.index('real_B9')
            negative = p < 0
            budgets.append(dict(target=target, budget=int(budget), ridge=model['ridge'],
                negative_predictions=int(negative.sum()), b9_coefficient=model['coefficients'][b9],
                b9_largest_negative_contribution=int(((components.argmin(axis=1) == b9) & negative).sum()),
                b9_contribution_min=float(components[:, b9].min()),
                b9_contribution_max=float(components[:, b9].max())))
            for i, col in enumerate(cols):
                contributions.append(dict(target=target, budget=int(budget), feature=col,
                    coefficient=model['coefficients'][i], mean_contribution=float(components[:, i].mean()),
                    min_contribution=float(components[:, i].min()), max_contribution=float(components[:, i].max())))
            if target == worst.target and int(budget) == worst.budget:
                j = np.flatnonzero(q.group.to_numpy() == worst.group)[0]
                pd.DataFrame(dict(feature=cols, value=q[cols].iloc[j].to_numpy(),
                    train_mean=model['mean'], train_scale=model['scale'], z=z[j],
                    coefficient=model['coefficients'], contribution=components[j])).sort_values(
                        'contribution').to_csv(output / 'worst_case_attribution.csv', index=False)
    pd.DataFrame(contributions).to_csv(output / 'low_family_contributions.csv', index=False)
    pd.DataFrame(budgets).to_csv(output / 'low_family_b9_diagnosis.csv', index=False)
    assert verify_freeze() == before and code_hashes() == frozen_code
    low = primary[(primary.scheme == 'lofo') & (primary.family == family)
                  & (primary.representation == 'realized')]
    receipt = dict(timestamp_utc=now.isoformat(), scope='Post-hoc historical diagnostic, not independent validation',
        correction='min(1,max(0,p)), applied to both legacy and realized predictions',
        model_refits=0, milp_solves=0, campaign_mutations=0, frozen_models_and_sources_unchanged=True,
        projection_never_increased_squared_error=True, binary_predictions_unchanged=True,
        reconstruction_max_error=reconstruction_error, tree_identity_max_error=tree_error,
        low_family_predictions=len(low), low_family_negative_predictions=int(low.below_zero.sum()),
        worst_case=dict(group=worst.group, budget=int(worst.budget), target=worst.target,
                        p=float(worst.p), y=float(worst.y), training_mean=float(worst.training_mean)),
        source_hashes={str(p.resolve()): sha(p) for p in sources}, script_sha256=sha(__file__))
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(output.resolve())
    print(pd.DataFrame(comparisons).round(6).to_string(index=False))
    print('Low-family negative predictions:', int(low.below_zero.sum()), '/', len(low))
    print('Frozen models and campaign code unchanged; no MILP solve or refit.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
