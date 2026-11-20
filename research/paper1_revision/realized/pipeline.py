"""Matched nested evaluations and a freeze made before independent executions."""
import time
import numpy as np
import pandas as pd
from ..data import RUN, read, put, sha, digest, config
from ..analysis import splits
from ..campaign import CAMP
from research.paper1_weights.model import group_folds
from .storage import DEST, protocol, signature, immutable, no_external_outcomes, load_frames, columns, external_frame, verify_historical_inputs
from .models import tune, predict


def jobs():
    p = protocol()
    for target in config()['policies'] + ['E', 'ES']:
        for rep in p['representations']:
            yield target, rep, 'logistic' if target in ['E', 'ES'] else 'positive'
        if target not in ['E', 'ES']:
            for rep in p['secondary_representations']:
                yield target, rep, 'signed'


def fit_signature():
    return digest([signature(), sha(DEST / 'feature_manifest.json')])


def fit_job(train, test, inner, target, rep, kind):
    p = protocol()
    grid = p['logistic_grid'] if kind == 'logistic' else p['ridge_grid']
    cols = columns(rep)
    output = np.zeros(len(test))
    baseline = np.zeros(len(test))
    models = {}
    for b in sorted(train.budget.unique()):
        a = train.budget.to_numpy() == b
        q = test.budget.to_numpy() == b
        output[q], models[str(int(b))] = tune(train[a], test[q], inner[a], target, cols, kind, grid)
        baseline[q] = train.loc[a, target].mean()
    return output, models, baseline


def fit_all():
    history, _, cfg = load_frames()
    sig = fit_signature()
    if not (DEST / 'design.json').exists():
        no_external_outcomes()
    immutable(DEST / 'design.json', dict(signature=sig, protocol=protocol(),
        training_groups=sorted(history.group.unique()), independent_outcomes_used=False))
    for scheme in protocol()['schemes']:
        for fold, train, test, inner in splits(history, scheme):
            if set(train.group) & set(test.group):
                raise ValueError('Outer scenario leakage')
            for target, rep, kind in jobs():
                path = DEST / 'fits' / scheme / target / (fold + '_' + rep + '_' + kind + '.json')
                if path.exists():
                    if read(path)['signature'] != sig:
                        raise ValueError('Stale comparison checkpoint: ' + str(path))
                    continue
                begin = time.perf_counter()
                pred, models, baseline = fit_job(train, test, inner, target, rep, kind)
                records = [dict(group=r.group, budget=int(r.budget), family=r.family,
                    reference_certified=bool(r.reference_certified), y=float(getattr(r, target)),
                    p=float(p), training_mean=float(base)) for r, p, base in zip(test.itertuples(), pred, baseline)]
                put(path, dict(signature=sig, target=target, representation=rep, kind=kind,
                    scheme=scheme, fold=fold, train_groups=sorted(train.group.unique()),
                    models=models, predictions=records, elapsed_seconds=time.perf_counter() - begin))
                print('realized', scheme, fold, target, rep, kind, flush=True)


def collect(scheme):
    sig = fit_signature()
    rows, count = [], 0
    for path in sorted((DEST / 'fits' / scheme).glob('*/*.json')):
        obj = read(path)
        if obj['signature'] != sig:
            raise ValueError('Changed comparison source')
        if set(obj['train_groups']) & {r['group'] for r in obj['predictions']}:
            raise ValueError('Scenario leakage in saved predictions')
        rows.extend(dict(r, target=obj['target'], representation=obj['representation'], kind=obj['kind'],
                         fold=obj['fold'], scheme=scheme) for r in obj['predictions'])
        count += 1
    expected = (5 if scheme == 'nested' else 6) * len(list(jobs()))
    if count != expected:
        raise ValueError(f'Comparison incomplete: {scheme}, {count}/{expected} checkpoints')
    frame = pd.DataFrame(rows)
    if frame.duplicated(['target', 'representation', 'kind', 'group', 'budget']).any():
        raise ValueError('Duplicate comparison observation')
    if len(frame) != len(list(jobs())) * 3000:
        raise ValueError('Incomplete comparison predictions')
    return frame


def verify_freeze():
    if not (DEST / 'freeze.json').exists():
        raise ValueError('Run python -m research.paper1_revision.complete_campaign prepare before physical experiments')
    verify_historical_inputs()
    obj = read(DEST / 'freeze.json')
    if obj['signature'] != fit_signature():
        raise ValueError('Comparison changed after freeze; new independent version required')
    if sha(DEST / 'frozen_models.json') != obj['models_sha256']:
        raise ValueError('Frozen models changed')
    if sha(DEST / 'external_predictions.parquet') != obj['predictions_sha256']:
        raise ValueError('Frozen independent predictions changed')
    # Also verify the physical feature content, not only its manifest.
    if sha(DEST / 'features.parquet') != read(DEST / 'feature_manifest.json')['features_sha256']:
        raise ValueError('Frozen descriptor table changed')
    return obj


def freeze():
    if (DEST / 'freeze.json').exists():
        verify_freeze()
        return
    no_external_outcomes()
    history, features, cfg = load_frames()
    collect('nested')
    collect('lofo')
    sig = fit_signature()
    if read(DEST / 'design.json')['signature'] != sig:
        raise ValueError('Design changed')
    inner = group_folds({'group': history.group.to_numpy(), 'family': history.family.to_numpy()},
                        3, cfg['historical_fold_seed'], 'final-inner')
    external = external_frame(features)
    all_models, records = {}, []
    for target, rep, kind in jobs():
        path = DEST / 'final_fits' / (target + '_' + rep + '_' + kind + '.json')
        if path.exists():
            entry = read(path)
            if entry['signature'] != sig:
                raise ValueError('Stale final comparison fit')
        else:
            pred, models, base = fit_job(history, external, inner, target, rep, kind)
            entry = dict(signature=sig, target=target, representation=rep, kind=kind, models=models,
                         predictions=pred.tolist(), baselines=base.tolist())
            put(path, entry)
        all_models[target + '/' + rep + '/' + kind] = entry['models']
        records.extend(dict(group=r.group, batch=r.batch, budget=int(r.budget), target=target,
                            representation=rep, kind=kind, p=float(p), training_mean=float(base))
                       for r, p, base in zip(external.itertuples(), entry['predictions'], entry['baselines']))
        print('frozen realized', target, rep, kind, flush=True)
    no_external_outcomes()  # Catch an independent campaign started during fitting.
    if fit_signature() != sig:
        raise ValueError('Source changed during freeze')
    pd.DataFrame(records).to_parquet(DEST / 'external_predictions.parquet', index=False)
    immutable(DEST / 'frozen_models.json', all_models)
    immutable(DEST / 'freeze.json', dict(signature=sig, models_sha256=sha(DEST / 'frozen_models.json'),
        predictions_sha256=sha(DEST / 'external_predictions.parquet'), independent_outcomes_used=False,
        historical_scenarios=600, external_scenarios=400, fitted_specifications=len(all_models),
        evaluation='Frozen before any external reference or trial attempt; preprocessing uses historical training only.'))


def evaluate():
    verify_freeze()
    changed_references = CAMP / 'reference_target_changes.csv'
    if changed_references.exists():
        from ..future_reports import fixed_prediction_sensitivity
        changes = pd.read_csv(changed_references)
        if len(changes):
            historical_predictions = collect('nested')
            historical_predictions['model'] = historical_predictions.representation + '/' + historical_predictions.kind
            fixed_prediction_sensitivity(changes, historical_predictions).to_csv(
                DEST / 'reference_prediction_sensitivity.csv', index=False)
    path = CAMP / 'external_observations.parquet'
    if not path.exists():
        put(DEST / 'external_status.json', dict(status='not_executed_or_no_valid_reference', evaluated_rows=0))
        return
    outcomes = pd.read_parquet(path)
    labels = outcomes.melt(id_vars=['group', 'budget'], value_vars=config()['policies'] + ['E', 'ES'],
                           var_name='target', value_name='y')
    pred = pd.read_parquet(DEST / 'external_predictions.parquet')
    q = pred.merge(labels, on=['group', 'budget', 'target'], validate='many_to_one')
    q.to_parquet(DEST / 'external_evaluation.parquet', index=False)
    from .report import scores, primary_contrasts
    scores(q, ['batch']).to_csv(DEST / 'external_scores.csv', index=False)
    for batch, batch_rows in q.groupby('batch'):
        if all(n == 100 for n in batch_rows.groupby(['target', 'representation', 'kind', 'budget']).size()) and len(batch_rows) == len(list(jobs())) * 500:
            primary_contrasts(batch_rows).to_csv(DEST / ('external_contrasts_' + batch + '.csv'), index=False)
    put(DEST / 'external_status.json', dict(status='complete' if len(outcomes) == 2000 else 'partial',
        evaluated_scenario_deadlines=len(outcomes), expected_scenario_deadlines=2000,
        note='Missing references/trials remain unobserved; contrasts are generated only for complete batches.'))
