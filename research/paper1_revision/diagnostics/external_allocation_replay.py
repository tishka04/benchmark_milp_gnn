"""Independent, retrospective portfolio allocation from frozen ES probabilities.

Run from the project root with python -m research.paper1_revision.diagnostics.external_allocation_replay.
Each 100-case batch receives its own reserve. No scheduling model is re-solved.
"""
from dataclasses import asdict
from time import perf_counter

import numpy as np
import pandas as pd

from src.analysis.criticality_index import compute_criticality
from ..allocation import allocate, uniform_mix
from ..campaign import CAMP
from ..data import COMPONENTS, RUN, digest, read, sha
from ..future_analysis import predict as predict_nominal
from ..realized.features import IDS
from ..realized.models import predict as predict_rebuilt
from ..realized.storage import DEST


BUDGETS = np.array([5, 10, 30, 60, 120])
RESERVES = [10, 20, 30, 60]
REBUILT = ['model_size', 'realized_compact', 'realized']
MODELS = ['uniform', 'size', 'nominal_index', 'nominal_three', 'nominal_22',
          'legacy_raw', *REBUILT, 'oracle']
NOMINAL_NAMES = {'nominal_index': 'original', 'nominal_three': 'compact', 'nominal_22': 'all'}
PLANNING_ALLOWANCE_S = 1.0  # Per 100 cases for the warm predictor and allocation together.
OUTPUT = RUN / 'diagnostics' / 'decision_replay_20261119'


def load_inputs():
    """Replay the saved forecasts and charge the feature implementation actually used."""
    manifest = read(CAMP / 'manifest.json')
    freeze = read(DEST / 'freeze.json')
    for key, file in [('models_sha256', 'frozen_models.json'), ('predictions_sha256', 'external_predictions.parquet')]:
        if sha(DEST / file) != freeze[key]:
            raise ValueError(f'Frozen archive mismatch: {file}')
    realized = pd.read_parquet(DEST / 'features.parquet').set_index('group')
    nominal_rows = []
    for case in manifest['entries']:
        data = read(case['file'])
        # Inputs have already been read, just as when dispatch processing starts.
        start = perf_counter()
        result = compute_criticality(data)
        raw = dict(asdict(result.stress_metrics), **asdict(result.hardness_metrics))
        elapsed = perf_counter() - start
        if data['meta']['zones'] != realized.loc[case['group'], 'real_B1']:
            raise ValueError('Nominal and actual zone counts differ')
        nominal_rows.append(dict(group=case['group'], nominal_seconds=elapsed,
                                 **{'raw_' + i: raw[c] for i, c in zip(IDS, COMPONENTS)}))
    realized = realized.join(pd.DataFrame(nominal_rows).set_index('group'), how='inner')
    outcomes = pd.read_parquet(CAMP / 'external_observations.parquet')
    archived = pd.read_parquet(DEST / 'external_predictions.parquet')
    archived = archived[archived.target.eq('ES') & archived.kind.eq('logistic')]
    nominal = read(CAMP / 'predictors_frozen.json')['models']
    fitted = read(DEST / 'frozen_models.json')
    scores = pd.read_csv(CAMP / 'external_scores.csv').query("policy == 'ES'")
    packs = {}
    for batch, cases in outcomes.groupby('batch'):
        groups = sorted(cases.group.unique(), key=lambda g: digest(['allocation-order', g]))
        n = len(groups)
        if n != 100:
            raise ValueError('Incomplete test batch')
        def matrix(table, col):
            return table.pivot(index='group', columns='budget', values=col).loc[groups, BUDGETS].to_numpy()
        y = matrix(cases, 'ES')
        exit_times = matrix(cases, 'exit_s')
        features = realized.loc[groups]
        probabilities = {'oracle': y}
        forecast_times = {}
        for model in MODELS:
            if model in ['uniform', 'oracle']:
                continue
            if model in NOMINAL_NAMES:
                name = NOMINAL_NAMES[model]
                started = perf_counter()
                pred = predict_nominal(nominal['ES/' + name], cases)
                forecast_times[model] = perf_counter() - started
                p = matrix(cases.assign(p=pred), 'p')
                reference = scores[scores.batch.eq(batch) & scores.model.eq(name)].set_index('budget')
                np.testing.assert_allclose(((p - y) ** 2).mean(axis=0), reference.loc[BUDGETS, 'mse'])
            else:
                models = fitted['ES/' + model + '/logistic']
                started = perf_counter()
                p = np.column_stack([predict_rebuilt(models[str(d)], features[models[str(d)]['columns']])
                                     for d in BUDGETS])
                forecast_times[model] = perf_counter() - started
                expected = archived[archived.batch.eq(batch) & archived.representation.eq(model)]
                np.testing.assert_allclose(p, matrix(expected, 'p'))
            if not np.isfinite(p).all() or np.min(p) < 0 or np.max(p) > 1:
                raise ValueError('Invalid frozen probabilities')
            probabilities[model] = p
        packs[batch] = (groups, y, exit_times, features, probabilities, forecast_times)
    return packs


def evaluate(pack, indices, average, models, accounting):
    """Charge archived post-build extraction under model reuse, or the full separate build."""
    groups, y, exit_times, features, probabilities, forecast_times = pack
    n = len(indices)
    total = n * average
    rows = []
    for model in models:
        charges = 0.
        if accounting != 'gross' and model not in ['uniform', 'oracle']:
            if model in REBUILT:
                column = 'audit_descriptor_seconds' if accounting == 'shared' else 'audit_total_seconds'
                charges = features[column].to_numpy()[indices].sum()
            else:
                # Common full nominal extractor provides a conservative charge also for zone count/trio.
                charges = features.nominal_seconds.to_numpy()[indices].sum()
            charges += PLANNING_ALLOWANCE_S
        reserve = int(np.floor(total - charges))
        start = perf_counter()
        choices = (uniform_mix(n, BUDGETS, reserve) if model == 'uniform'
                   else allocate(probabilities[model][indices], BUDGETS, reserve))
        elapsed = perf_counter() - start + forecast_times.get(model, 0.)
        if elapsed > PLANNING_ALLOWANCE_S and charges:
            raise RuntimeError('Warm prediction and allocation exceeded their charged allowance')
        reserved = int(BUDGETS[choices].sum())
        if reserved + charges > total or not np.isfinite(exit_times[indices, choices]).all():
            raise ValueError('Invalid reserved time or missing observed exit')
        rows.append(dict(model=model, average=average, accounting=accounting, successes=int(y[indices, choices].sum()),
                         reserved_seconds=reserved, charge_seconds=charges, portfolio_budget=total,
                         observed_exit_seconds=float(exit_times[indices, choices].sum()),
                         measured_prediction_and_choice_seconds=elapsed))
    return rows


def run(bootstrap=200):
    packs = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    points = []
    for batch, pack in packs.items():
        for average in RESERVES:
            for mode in ['gross', 'shared', 'separate']:
                rows = evaluate(pack, np.arange(100), average, MODELS, mode)
                points.extend(dict(r, batch=batch) for r in rows)
    frame = pd.DataFrame(points)
    frame.to_csv(OUTPUT / 'allocation_by_batch.csv', index=False)
    totals = frame.groupby(['average', 'accounting', 'model'])[
        ['successes', 'reserved_seconds', 'charge_seconds', 'observed_exit_seconds', 'portfolio_budget']].sum()
    totals.to_csv(OUTPUT / 'allocation_totals.csv')
    print(totals.successes.unstack().to_string(), flush=True)

    # The 30-second reserve is the historical comparison carried into the paper.
    # Resample whole scenarios by batch, rebuild allocations, keep probabilities fixed.
    draws = []
    rng = np.random.default_rng(20261119)
    for rep in range(bootstrap):
        for batch, pack in packs.items():
            indices = rng.integers(0, 100, 100)
            for mode in ['gross', 'shared']:
                rows = evaluate(pack, indices, 30, MODELS, mode)
                draws.extend(dict(r, batch=batch, repetition=rep) for r in rows)
    sample = pd.DataFrame(draws)
    sample.to_parquet(OUTPUT / 'allocation_resamples.parquet', index=False)
    aggregated = sample.groupby(['repetition', 'accounting', 'model']).successes.sum().unstack()
    contrasts = []
    for mode in ['gross', 'shared']:
        wide = aggregated.xs(mode, level='accounting')
        for model in MODELS:
            for baseline in ['uniform', 'nominal_three', 'realized_compact']:
                low, high = np.quantile(wide[model] - wide[baseline], [.025, .975])
                contrasts.append(dict(accounting=mode, average=30, model=model, baseline=baseline,
                                      draws=bootstrap, lower=low, upper=high))
    intervals = pd.DataFrame(contrasts)
    intervals.to_csv(OUTPUT / 'allocation_paired_intervals.csv', index=False)
    print(intervals.query("baseline == 'uniform'").to_string(index=False))


if __name__ == '__main__':
    run()
