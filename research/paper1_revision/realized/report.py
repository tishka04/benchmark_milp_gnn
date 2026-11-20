"""Paired prediction and retrospective allocation reports from completed fits."""
import numpy as np
import pandas as pd
from ..data import RUN, read, put, digest, config
from ..allocation import allocate, uniform_mix
from .storage import DEST, protocol, load_frames


def scores(frame, extra=()):
    rows = []
    keys = list(extra) + ['target', 'representation', 'kind', 'budget']
    for key, q in frame.groupby(keys):
        mse = float(np.mean((q.p - q.y)**2))
        base = float(np.mean((q.training_mean - q.y)**2))
        rows.append(dict(zip(keys, key), n=len(q), mse=mse,
            baseline_mse=base, skill=1 - mse / base if base > 0 else np.nan,
            outside_fraction=float(((q.p < 0) | (q.p > 1)).mean())))
    return pd.DataFrame(rows)


def primary_contrasts(frame):
    vectors, keys = [], []
    for target in config()['policies'] + ['E', 'ES']:
        kind = 'logistic' if target in ['E', 'ES'] else 'positive'
        for budget in config()['deadlines']:
            q = frame[(frame.target == target) & (frame.kind == kind) & (frame.budget == budget)].copy()
            q['loss'] = (q.y - q.p)**2
            wide = q.pivot(index='group', columns='representation', values='loss').sort_index()
            if wide[['legacy_raw', 'realized']].isna().any().any():
                raise ValueError('Unpaired primary comparison')
            vectors.append((wide.legacy_raw - wide.realized).to_numpy())
            keys.append((target, budget))
    values = np.array(vectors).T
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Incomplete primary contrasts')
    cfg = protocol()
    rng = np.random.default_rng(cfg['seed'])
    boot = np.array([values[rng.integers(0, len(values), len(values))].mean(axis=0)
                     for _ in range(cfg['bootstrap_samples'])])
    means, se = values.mean(axis=0), boot.std(axis=0, ddof=1)
    critical = float(np.quantile(np.max(abs((boot - means) / np.maximum(se, 1e-15)), axis=1), .95))
    rows = []
    for j, (target, budget) in enumerate(keys):
        lo, hi = np.quantile(boot[:, j], [.025, .975])
        rows.append(dict(target=target, budget=budget, n=len(values),
            positive_gain_favors='realized', mean_mse_gain=means[j], lower=lo, upper=hi,
            simultaneous_lower=means[j] - critical * se[j], simultaneous_upper=means[j] + critical * se[j],
            uncertainty='Conditional on fitted predictions; 35 contrasts in this validation scheme.'))
    return pd.DataFrame(rows)


def allocation_report(predictions):
    from .pipeline import fit_signature
    sig = fit_signature()
    history, _, cfg = load_frames()
    pred = predictions[(predictions.target == 'ES') & (predictions.kind == 'logistic')]
    budgets = np.asarray(cfg['deadlines'], int)
    reps = ['size', 'model_size', 'legacy_raw', 'corrected_only', 'realized', 'realized_compact']
    all_rows = []
    for fold in range(5):
        test = history[history.outer_fold == fold]
        groups = sorted(test.group.unique(), key=lambda g: digest(['allocation-order', g]))
        values = {r: pred[(pred.fold == str(fold)) & (pred.representation == r)].pivot(
            index='group', columns='budget', values='p').loc[groups, budgets].to_numpy() for r in reps}
        y = test.pivot(index='group', columns='budget', values='ES').loc[groups, budgets].to_numpy()
        times = test.pivot(index='group', columns='budget', values='exit_s').loc[groups, budgets].to_numpy()
        values['oracle'] = y
        for average in cfg['allocation_mean_budgets']:
            path = DEST / 'allocation_checkpoints' / f'{fold}_{average}.json'
            if path.exists():
                saved = read(path)
                if saved['signature'] != sig:
                    raise ValueError('Stale comparison allocation')
                all_rows.extend(saved['rows'])
                continue
            rows = []
            n, total = len(groups), len(groups) * average
            for repetition in range(-1, protocol()['allocation_bootstrap_samples']):
                rng = np.random.default_rng(int(digest([cfg['seed'], 'portfolio', 'ES', fold, average, repetition])[:8], 16))
                indices = np.arange(n) if repetition == -1 else rng.integers(0, n, n)
                for name in ['uniform'] + list(values):
                    selected = uniform_mix(n, budgets, total) if name == 'uniform' else allocate(values[name][indices], budgets, total)
                    actual = float(times[indices, selected].sum())
                    rows.append(dict(fold=fold, mean_budget=average, repetition=repetition, representation=name,
                        successes=int(y[indices, selected].sum()), reserved_seconds=int(budgets[selected].sum()),
                        observed_seconds=actual, observed_overrun_seconds=max(0., actual - total)))
            put(path, dict(signature=sig, rows=rows))
            all_rows.extend(rows)
            print('realized allocation', fold, average, flush=True)
    a = pd.DataFrame(all_rows)
    a.to_parquet(DEST / 'allocation_resamples.parquet', index=False)
    totals = a.groupby(['mean_budget', 'repetition', 'representation'])[
        ['successes', 'reserved_seconds', 'observed_seconds', 'observed_overrun_seconds']].sum().reset_index()
    totals[totals.repetition == -1].to_csv(DEST / 'allocation_totals.csv', index=False)
    contrasts = []
    for avg, q in totals[totals.repetition >= 0].groupby('mean_budget'):
        wide = q.pivot(index='repetition', columns='representation', values='successes')
        for baseline in ['uniform', 'size', 'model_size', 'legacy_raw']:
            for arm in ['corrected_only', 'realized', 'realized_compact']:
                lo, hi = np.quantile(wide[arm] - wide[baseline], [.025, .975])
                contrasts.append(dict(mean_budget=avg, representation=arm, reference=baseline,
                                      lower=lo, upper=hi, conditional_on_fits=True))
    pd.DataFrame(contrasts).to_csv(DEST / 'allocation_intervals.csv', index=False)


def run(include_allocation=True):
    from .pipeline import collect, fit_signature
    summary = []
    for scheme in protocol()['schemes']:
        p = collect(scheme)
        p.to_parquet(DEST / (scheme + '_predictions.parquet'), index=False)
        scores(p).to_csv(DEST / (scheme + '_scores.csv'), index=False)
        primary_contrasts(p).to_csv(DEST / (scheme + '_primary_contrasts.csv'), index=False)
        calibration = p[p.target.isin(['E', 'ES'])].copy()
        calibration['probability_bin'] = np.minimum((calibration.p * 10).astype(int), 9)
        calibration.groupby(['target', 'representation', 'budget', 'probability_bin']).agg(
            n=('y', 'size'), mean_prediction=('p', 'mean'), observed_rate=('y', 'mean')).to_csv(
                DEST / (scheme + '_calibration.csv'))
        for (target, rep, kind), q in p.groupby(['target', 'representation', 'kind']):
            summary.append(dict(scheme=scheme, target=target, representation=rep, kind=kind,
                                n=len(q), mse=float(np.mean((q.p - q.y)**2))))
        if scheme == 'nested' and include_allocation:
            allocation_report(p)
    s = pd.DataFrame(summary)
    s.to_csv(DEST / 'pooled_scores.csv', index=False)
    history, features, _ = load_frames()
    one = history.drop_duplicates('group')
    diagnostics = []
    from .features import IDS
    for i in IDS:
        a, b = one['raw_' + i], one['real_' + i]
        diagnostics.append(dict(component=i, legacy_unique=a.nunique(), realized_unique=b.nunique(),
            legacy_min=a.min(), legacy_max=a.max(), realized_min=b.min(), realized_max=b.max(),
            pearson=a.corr(b) if a.std() > 1e-12 and b.std() > 1e-12 else np.nan))
    pd.DataFrame(diagnostics).to_csv(DEST / 'definition_comparison.csv', index=False)
    features.groupby('population')[['audit_load_seconds', 'audit_build_seconds', 'audit_descriptor_seconds',
        'audit_total_seconds']].agg(['count', 'median', 'max']).to_csv(DEST / 'feature_computation_times.csv')
    put(DEST / 'report_status.json', dict(signature=fit_signature(), historical_predictions_complete=True,
        independent_validation_completed=False, allocation_included=include_allocation,
        caveats=['No claim that all 22 coordinates have identical semantics across versions.',
                 'Continuous and ES decision objectives differ; allocation uses direct logistic ES predictions.',
                 'MILP construction and descriptor extraction are offline costs, excluded from historical deadlines; no net online utility claim.',
                 'Positive coefficients are an assumption; signed models are secondary controls.',
                 'Actual declared free binaries differ from post-presolve binaries; no solver statistics are used.']))


if __name__ == '__main__':
    run()
