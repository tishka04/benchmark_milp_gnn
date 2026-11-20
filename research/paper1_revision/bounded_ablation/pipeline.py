"""Resumable historical fitting, with separate artifacts and provenance guards."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import RUN, read, put, sha, digest, config
from ..analysis import splits
from ..realized.storage import DEST as ORIGINAL, load_frames
from ..realized.pipeline import verify_freeze
from .models import tune

PACKAGE = Path(__file__).resolve().parent
DEST = RUN / 'bounded_ablation_v1'


def protocol():
    return read(PACKAGE / 'protocol.json')


def identity():
    frozen = verify_freeze()
    paths = [PACKAGE / name for name in ['models.py', 'pipeline.py', '__main__.py', 'protocol.json']]
    paths += [ORIGINAL / (s + '_predictions.parquet') for s in protocol()['schemes']]
    paths += [RUN / 'splits.json']
    return dict(frozen_comparison=frozen, sources={str(p): sha(p) for p in paths})


def specifications():
    for target in config()['policies']:
        for representation in protocol()['representations']:
            for arm in protocol()['arms']:
                yield target, representation, arm


def run(max_jobs=None):
    begin = time.perf_counter()
    inputs = identity()
    sig = digest(inputs)
    design = dict(signature=sig, inputs=inputs, protocol=protocol(),
        targets=config()['policies'], deadlines=config()['deadlines'], historical_scenarios=600,
        expected_checkpoints=330, expected_deadline_models=1650, independent_outcomes_used=False)
    manifest = DEST / 'design.json'
    if manifest.exists() and read(manifest) != design:
        raise ValueError('Changed follow-up design or inputs; use another version')
    if not manifest.exists():
        put(manifest, design)
    history, _, cfg = load_frames()
    completed = 0
    for scheme in protocol()['schemes']:
        for fold, train, test, inner in splits(history, scheme):
            if set(train.group) & set(test.group):
                raise ValueError('Outer scenario overlap')
            for target, representation, arm in specifications():
                path = DEST / 'fits' / scheme / target / (fold + '_' + representation + '_' + arm + '.json')
                if path.exists():
                    if read(path)['signature'] != sig:
                        raise ValueError('Stale checkpoint: ' + str(path))
                    continue
                if max_jobs is not None and completed >= max_jobs:
                    return
                start = time.perf_counter()
                models, records = {}, []
                for budget in cfg['deadlines']:
                    a, b = train.budget.to_numpy() == budget, test.budget.to_numpy() == budget
                    predictions, models[str(budget)] = tune(train[a], test[b], inner[a], target,
                        representation, arm, protocol()['ridge_grid'])
                    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
                        raise ValueError('Invalid delivered predictions')
                    records.extend(dict(group=r.group, family=r.family, budget=int(budget),
                        y=float(getattr(r, target)), p=float(p),
                        reference_certified=bool(r.reference_certified))
                        for r, p in zip(test[b].itertuples(), predictions))
                put(path, dict(signature=sig, scheme=scheme, fold=fold, target=target,
                    representation=representation, arm=arm, train_groups=sorted(train.group.unique()),
                    models=models, predictions=records, elapsed_seconds=time.perf_counter() - start))
                completed += 1
                put(DEST / 'status.json', dict(status='fitting', latest_checkpoint=str(path),
                    completed_checkpoints=sum(1 for _ in (DEST / 'fits').glob('*/*/*.json')),
                    current_invocation_seconds=time.perf_counter() - begin))
                print(scheme, fold, target, representation, arm, round(time.perf_counter() - start, 2), flush=True)
    if identity() != inputs:
        raise ValueError('Sources or frozen inputs changed during fitting')
    report()


def collect(scheme, sig):
    rows, count = [], 0
    for path in sorted((DEST / 'fits' / scheme).glob('*/*.json')):
        obj = read(path)
        if obj['signature'] != sig:
            raise ValueError('Stale checkpoint')
        if set(obj['train_groups']) & {r['group'] for r in obj['predictions']}:
            raise ValueError('Checkpoint leakage')
        rows.extend(dict(r, scheme=scheme, fold=obj['fold'], target=obj['target'],
                         representation=obj['representation'], arm=obj['arm']) for r in obj['predictions'])
        count += 1
    expected = (5 if scheme == 'nested' else 6) * len(list(specifications()))
    if count != expected:
        raise ValueError(f'Incomplete scheme: {count}/{expected}')
    baseline = pd.read_parquet(ORIGINAL / (scheme + '_predictions.parquet'))
    baseline = baseline[(baseline.kind == 'positive') & baseline.target.isin(config()['policies'])
                        & baseline.representation.isin(protocol()['representations'])].copy()
    baseline['arm'] = 'current_clip'
    baseline['p'] = baseline.p.clip(0, 1)
    frame = pd.concat([pd.DataFrame(rows), baseline], ignore_index=True)
    keys = ['target', 'representation', 'arm', 'group', 'budget']
    if frame.duplicated(keys).any() or len(frame) != 120000:
        raise ValueError('Duplicate or missing predictions')
    sizes = frame.groupby(['target', 'representation', 'arm']).size()
    if not (sizes == 3000).all() or len(sizes) != 40:
        raise ValueError('Incomplete specifications')
    labels = frame.groupby(['group', 'budget', 'target']).y.agg(['min', 'max'])
    if not np.array_equal(labels['min'], labels['max']):
        raise ValueError('Unmatched outcomes')
    if not frame.p.between(0, 1).all():
        raise ValueError('Out-of-range predictions')
    frame['loss'] = (frame.p - frame.y)**2
    return frame


def contrasts(frame):
    # Average deadlines before resampling so each scenario is the sampling unit.
    vectors, keys = [], []
    groups = sorted(frame.group.unique())
    comparisons = [(a, 'current_clip') for a in protocol()['arms']]
    comparisons.append(('fractional_logit_no_b9', 'fractional_logit'))
    for (target, rep), q in frame.groupby(['target', 'representation']):
        wide = q.groupby(['group', 'arm']).loss.mean().unstack().loc[groups]
        for arm, reference in comparisons:
            vectors.append((wide[reference] - wide[arm]).to_numpy())
            keys.append((target, rep, arm, reference))
    values = np.array(vectors).T
    if values.shape != (600, 40) or not np.isfinite(values).all():
        raise ValueError('Incomplete paired contrasts')
    rng = np.random.default_rng(protocol()['seed'])
    boot = np.array([values[rng.integers(0, len(values), len(values))].mean(axis=0)
                     for _ in range(protocol()['bootstrap_samples'])])
    means, se = values.mean(axis=0), boot.std(axis=0, ddof=1)
    critical = np.quantile(np.max(abs((boot - means) / np.maximum(se, 1e-15)), axis=1), .95)
    rows = []
    for j, (target, rep, arm, reference) in enumerate(keys):
        lo, hi = np.quantile(boot[:, j], [.025, .975])
        rows.append(dict(target=target, representation=rep, arm=arm, reference=reference,
            mse_gain=float(means[j]), lower=float(lo), upper=float(hi),
            simultaneous_lower=float(means[j] - critical * se[j]),
            simultaneous_upper=float(means[j] + critical * se[j]), scenarios=600,
            note='Exploratory; conditional on fits, not adjusted for post-hoc design selection.'))
    return pd.DataFrame(rows)


def report():
    inputs = identity()
    sig = digest(inputs)
    if read(DEST / 'design.json')['signature'] != sig:
        raise ValueError('Design changed')
    summaries, conditions = [], []
    for scheme in protocol()['schemes']:
        q = collect(scheme, sig)
        q.to_parquet(DEST / (scheme + '_predictions.parquet'), index=False)
        summary = q.groupby(['target', 'representation', 'arm']).agg(
            n=('p', 'size'), mse=('loss', 'mean'), p_min=('p', 'min'), p_max=('p', 'max'),
            boundary_fraction=('p', lambda x: ((x <= 1e-12) | (x >= 1 - 1e-12)).mean())).reset_index()
        summary['scheme'] = scheme
        summaries.append(summary)
        q.groupby(['family', 'target', 'representation', 'arm']).loss.mean().rename('mse').to_csv(
            DEST / (scheme + '_family_scores.csv'))
        q.groupby(['target', 'representation', 'arm', 'budget']).loss.mean().rename('mse').to_csv(
            DEST / (scheme + '_deadline_scores.csv'))
        c = contrasts(q)
        c['scheme'] = scheme
        conditions.append(c)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(DEST / 'pooled_scores.csv', index=False)
    comparison = summary.pivot(index=['scheme', 'target', 'representation'], columns='arm', values='mse')
    for arm in protocol()['arms']:
        comparison[arm + '_gain_percent'] = 100 * (1 - comparison[arm] / comparison.current_clip)
    comparison.to_csv(DEST / 'comparison.csv')
    pd.concat(conditions, ignore_index=True).to_csv(DEST / 'paired_contrasts.csv', index=False)
    if identity() != inputs:
        raise ValueError('Inputs changed during report')
    put(DEST / 'status.json', dict(status='complete', signature=sig, completed_checkpoints=330,
        fitted_deadline_models=1650, evaluated_predictions=240000, historical_scenarios=600,
        frozen_comparison_unchanged=True, campaign_mutations=0, milp_solves=0,
        external_outcomes_used=False, interpretation=protocol()['scope']))
    print('Complete:', DEST, flush=True)
