"""Resumable estimation and reporting; no historical output is written."""
from __future__ import annotations
import csv
import io
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from .data import (COMPONENTS, digest, seal, verify, read, put, publish, load_prepared)
from .model import arrays, specifications, cv_job, final_job, bootstrap_job
from .parallel import bounded_map


def checkpoint(run, stage, job):
    return Path(run) / 'checkpoints' / stage / (job + '.json')


def stored(run, stage, job, manifest):
    path = checkpoint(run, stage, job)
    if not path.exists():
        return None
    obj = verify(read(path))
    if obj['manifest'] != manifest['sha256'] or obj['result']['job'] != job:
        raise ValueError('Checkpoint provenance mismatch')
    return obj


def execute(run, stage, jobs, function, manifest, workers):
    expected = [job[0] for job in jobs]
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate job identifiers')
    pending = [job for job in jobs if stored(run, stage, job[0], manifest) is None]
    print(f'{stage}: {len(expected) - len(pending)}/{len(expected)} checkpoints reused', flush=True)
    for completed, value in enumerate(bounded_map(function, pending, workers), 1):
        put(checkpoint(run, stage, value['job']), seal(dict(manifest=manifest['sha256'], result=value)))
        print(f'{stage}: saved {completed}/{len(pending)} new checkpoints', flush=True)
    hashes = {job: stored(run, stage, job, manifest)['sha256'] for job in expected}
    put(Path(run) / (stage + '_receipt.json'), seal(dict(manifest=manifest['sha256'], checkpoints=hashes)))


def receipt_results(run, stage, manifest):
    receipt = verify(read(Path(run) / (stage + '_receipt.json')))
    if receipt['manifest'] != manifest['sha256']:
        raise ValueError('Receipt provenance mismatch')
    result = {}
    for job, expected in receipt['checkpoints'].items():
        obj = stored(run, stage, job, manifest)
        if obj is None or obj['sha256'] != expected:
            raise ValueError('Missing/changed checkpoint')
        result[job] = obj['result']
    return result


def fit_all(run, workers):
    manifest, prepared = load_prepared(run)
    cfg, rows = manifest['config'], prepared['rows']
    cv_tasks, final_tasks = [], []
    for policy in cfg['policies']:
        data = arrays(rows, policy)
        for spec in specifications(cfg):
            prefix = policy + '/' + spec['name']
            for fold in range(cfg['outer_folds']):
                cv_tasks.append((prefix + f'/outer_{fold}', data, spec, fold, cfg))
            final_tasks.append((prefix, data, spec, cfg))
    execute(run, 'cv', cv_tasks, cv_job, manifest, workers)
    execute(run, 'final', final_tasks, final_job, manifest, workers)


def bootstrap_all(run, workers):
    manifest, prepared = load_prepared(run)
    cfg = manifest['config']
    final = receipt_results(run, 'final', manifest)
    tasks = []
    for policy in cfg['policies']:
        data = arrays(prepared['rows'], policy)
        for spec in specifications(cfg):
            if spec['kind'] != 'positive':
                continue
            prefix = policy + '/' + spec['name']
            model = final[prefix]['model']
            for start in range(0, cfg['bootstrap_samples'], 20):
                stop = min(cfg['bootstrap_samples'], start + 20)
                tasks.append((prefix + f'/batch_{start:05d}', data, spec, model, start, stop, cfg))
    execute(run, 'bootstrap', tasks, bootstrap_job, manifest, workers)


def csv_output(path, rows):
    if not rows:
        publish(path, b'')
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    publish(path, stream.getvalue().encode('utf-8-sig'))


def scores(predictions):
    y = np.array([r['observed'] for r in predictions])
    p = np.array([r['predicted'] for r in predictions])
    errors = p - y
    variance = float(np.sum((y - y.mean()) ** 2))
    rho = None
    if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0:
        rho = float(spearmanr(y, p).statistic)
    return dict(observations=len(y), groups=len({r['group'] for r in predictions}),
                mse=float(np.mean(errors ** 2)), rmse=float(np.sqrt(np.mean(errors ** 2))),
                mae=float(np.mean(np.abs(errors))),
                r2=1 - float(errors @ errors) / variance if variance > 0 else None,
                spearman=rho, prediction_outside_unit_interval=float(np.mean((p < 0) | (p > 1))))


def coefficient_rows(final, boot, cfg):
    out = []
    for prefix, result in sorted(final.items()):
        policy, name = prefix.split('/')
        model = result['model']
        if model['kind'] != 'positive':
            continue
        samples = sorted([s for key, batch in boot.items() if key.startswith(prefix + '/')
                          for s in batch['samples']], key=lambda s: s['repetition'])
        if len(samples) != cfg['bootstrap_samples'] or len({s['repetition'] for s in samples}) != len(samples):
            raise ValueError('Incomplete coefficient bootstrap')
        valid = [s for s in samples if s['index_weights'] is not None]
        ww = np.array([s['index_weights'] for s in valid]) if valid else np.empty((0, len(COMPONENTS)))
        bb = np.array([s['coefficients'] for s in samples]) if samples else np.empty((0, len(COMPONENTS)))
        active = np.array(model['active'], dtype=bool)
        ranks = np.full((len(valid), len(COMPONENTS)), np.nan)
        if len(valid) and active.any():
            ranks[:, active] = np.array([rankdata(-w[active], method='average') for w in ww])
        point_ranks = np.full(len(COMPONENTS), np.nan)
        if model['index_weights'] is not None:
            point_ranks[active] = rankdata(-np.array(model['index_weights'])[active], method='average')
        for j, component in enumerate(COMPONENTS):
            estimable = bool(active[j])
            weight = model['index_weights'][j] if model['index_weights'] is not None else None
            ci = np.quantile(ww[:, j], [.025, .975]).tolist() if len(valid) and estimable else [None, None]
            bci = np.quantile(bb[:, j], [.025, .975]).tolist() if len(samples) and estimable else [None, None]
            rank_ci = np.quantile(ranks[:, j], [.025, .975]).tolist() if len(valid) and estimable else [None, None]
            out.append(dict(policy=policy, model=name, deadline=result['spec']['deadline'], component=component,
                status='estimated_under_positive_ridge' if estimable else 'constant_not_identifiable_zero_imposed',
                coefficient=model['coefficients'][j], coefficient_lower=bci[0], coefficient_upper=bci[1],
                applied_index_weight=weight,
                estimated_index_weight=weight if estimable else None,
                weight_lower=ci[0], weight_upper=ci[1],
                rank=float(point_ranks[j]) if estimable and np.isfinite(point_ranks[j]) else None,
                rank_lower=rank_ci[0], rank_upper=rank_ci[1],
                bootstrap_positive_fraction=float(np.mean(ww[:, j] > 1e-10)) if len(valid) and estimable else None,
                bootstrap_samples=len(samples), bootstrap_valid_indices=len(valid),
                final_ridge=model['ridge'], index_scale=model['index_scale'],
                uncertainty='pointwise scenario refit bootstrap; ridge fixed; no multiplicity adjustment'))
    return out


def paired_losses(predictions, cfg):
    indexed = {(r['policy'], r['model'], r['group'], r['budget']): r for r in predictions}
    out = []
    pairs = [('original_index', 'common_positive'), ('common_positive', 'common_signed'),
             ('common_positive', 'separate_deadlines')]
    for policy in cfg['policies']:
        for budget in [None] + cfg['deadlines_s']:
            for baseline, candidate in pairs:
                per_group = {}
                for (p, method, group, b), rr in indexed.items():
                    if p != policy or method != baseline or (budget is not None and b != budget):
                        continue
                    cc = indexed.get((policy, candidate, group, b))
                    if cc is None or not np.isclose(rr['observed'], cc['observed']):
                        raise ValueError('Unpaired held-out regression comparison')
                    difference = (rr['predicted'] - rr['observed']) ** 2 - (cc['predicted'] - cc['observed']) ** 2
                    per_group.setdefault(group, []).append(difference)
                values = np.array([np.mean(per_group[g]) for g in sorted(per_group)])
                rng = np.random.default_rng(int(digest([cfg['seed'], policy, budget, baseline, candidate])[:8], 16))
                estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(2000)]
                lo, hi = np.quantile(estimates, [.025, .975])
                out.append(dict(policy=policy, deadline=budget, baseline=baseline, candidate=candidate,
                                gain_mse=float(values.mean()), lower=float(lo), upper=float(hi), groups=len(values),
                                interpretation='positive = lower held-out loss for candidate',
                                uncertainty='pointwise paired OOF-loss bootstrap; no refitting'))
    return out


def policy_rows(cfg):
    """Export the declared target preferences, distinct from learned index weights."""
    return [dict(policy=name, **{criterion + '_weight': policy[criterion]
                                for criterion in ('ens', 'cost', 'time', 'co2')})
            for name, policy in cfg['policies'].items()]


def report(run):
    manifest, prepared = load_prepared(run)
    cfg, rows = manifest['config'], prepared['rows']
    cv = receipt_results(run, 'cv', manifest)
    final = receipt_results(run, 'final', manifest)
    boot = receipt_results(run, 'bootstrap', manifest)
    expected = len(cfg['policies']) * len(specifications(cfg))
    if len(final) != expected or len(cv) != expected * cfg['outer_folds']:
        raise ValueError('Incomplete declared regression grid')
    row_map = {(r['group'], r['budget']): r for r in rows}
    predictions, fold_coefficients = [], []
    for job, result in sorted(cv.items()):
        policy, method, _ = job.split('/')
        for pred in result['predictions']:
            rr = row_map[(pred['group'], pred['budget'])]
            predictions.append(dict(policy=policy, model=method, fold=result['fold'],
                                    reference_certified=rr['reference_certified'], **pred))
            if method.startswith('deadline_'):
                predictions.append(dict(policy=policy, model='separate_deadlines', fold=result['fold'],
                                        reference_certified=rr['reference_certified'], **pred))
        m = result['model']
        for j, component in enumerate(m['names']):
            fold_coefficients.append(dict(policy=policy, model=method, fold=result['fold'],
                component=component, coefficient=m['coefficients'][j], active=m['active'][j],
                index_weight=m['index_weights'][j] if m['index_weights'] is not None else None,
                ridge=m['ridge']))
    identities = [(p['policy'], p['model'], p['group'], p['budget']) for p in predictions]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate held-out prediction')
    for policy in cfg['policies']:
        for name in ('common_positive', 'common_signed', 'original_index', 'intercept_only', 'separate_deadlines'):
            if sum(p['policy'] == policy and p['model'] == name for p in predictions) != len(rows):
                raise ValueError('Missing held-out coverage')
    metrics = []
    combinations = sorted({(r['policy'], r['model']) for r in predictions})
    for policy, model in combinations:
        candidates = [r for r in predictions if r['policy'] == policy and r['model'] == model]
        for budget in [None] + cfg['deadlines_s']:
            for reference_subset in ('all', 'certified_only'):
                selected = [r for r in candidates if (budget is None or r['budget'] == budget)
                            and (reference_subset == 'all' or r['reference_certified'])]
                if selected:
                    metrics.append(dict(policy=policy, model=model, deadline=budget,
                                        reference_subset=reference_subset, **scores(selected)))
    weights = coefficient_rows(final, boot, cfg)
    support = []
    for b in cfg['deadlines_s']:
        selected = [r for r in rows if r['budget'] == b]
        valid = [r for r in selected if r['feasible']]
        support.append(dict(deadline=b, scenarios=len(selected), delivered=len(valid),
                            failed_delivery=len(selected) - len(valid),
                            known_dispatch_co2=len(valid),
                            reference_certified=sum(r['reference_certified'] for r in selected),
                            nonzero_ens=sum(r['ens_mwh'] > cfg['zero_ens_tolerance_mwh'] for r in valid),
                            positive_imports=sum(r['imports_mwh'] > 1e-6 for r in valid)))
    targets = [dict(group=r['group'], budget=r['budget'], policy=name,
                    reference_certified=r['reference_certified'], **r['targets'][name])
               for r in rows for name in cfg['policies']]
    carbon_columns = ('group', 'budget', 'feasible', 'demand_mwh', 'ens_mwh', 'served_mwh',
                      'thermal_mwh', 'co2_direct_t', 'co2_t_per_served_mwh', 'imports_mwh',
                      'exports_mwh', 'reference_co2_direct_t', 'reference_ens_mwh',
                      'carbon_scope', 'import_carbon_known', 'co2_price_eur_per_t',
                      'cost_ens_penalty_eur', 'cost_co2_charge_eur')
    outputs = {
        'policy_weights.csv': policy_rows(cfg),
        'weights_common.csv': [r for r in weights if r['model'] == 'common_positive'],
        'weights_by_deadline.csv': [r for r in weights if r['model'] != 'common_positive'],
        'fold_coefficients.csv': fold_coefficients,
        'prediction_scores.csv': metrics,
        'predictions.csv': predictions,
        'paired_predictive_gains.csv': paired_losses(predictions, cfg),
        'support.csv': support,
        'targets.csv': targets,
        'carbon_dispatch.csv': [{k: r[k] for k in carbon_columns} for r in rows],
    }
    report_dir = Path(run) / 'report'
    for name, table in outputs.items():
        csv_output(report_dir / name, table)
    put(report_dir / 'models.json', seal(dict(manifest=manifest['sha256'], config=cfg,
        equation='predicted_score = deadline_intercept + index_scale * sum_j(index_weight_j * component_j)',
        models={k: v['model'] for k, v in final.items()},
        note='Constant components receive imposed computational zero, not an identified scientific weight.')))
    lines = ['# Paper 1: estimated operational-index weights', '',
             'Exploratory development results on existing HiGHS dispatches. No MILP solve or router modification.', '',
             'Target policies are reported separately. MSE values across different policies are not comparable measures of superiority.',
             'Each common model has one intercept per deadline and common nonnegative component slopes.',
             'Deadlines are independent executions; no resolution times are summed. The 3000 records contain 600 independent groups.', '',
             '## Declared target preferences', '',
             'These are chosen criterion weights, not learned criticality-component weights.', '',
             '| Policy | ENS | Cost | Time | CO2 |', '|---|---:|---:|---:|---:|']
    for policy in outputs['policy_weights.csv']:
        lines.append(f"| {policy['policy']} | {policy['ens_weight']:.6g} | {policy['cost_weight']:.6g} | {policy['time_weight']:.6g} | {policy['co2_weight']:.6g} |")
    lines += ['', '## Support', '', '| Deadline | Delivered | Failed delivery | CO2 known |', '|---|---:|---:|---:|']
    lines += [f"| {r['deadline']} | {r['delivered']} | {r['failed_delivery']} | {r['known_dispatch_co2']} |" for r in support]
    lines += ['', '## Main weights', '',
              'A zero for a constant feature is imposed; its importance is not identified.',
              'Bootstrap intervals are pointwise, conditional on the final selected ridge penalty; 500 is the default, see config for the actual count.',
              'Ranks and allocations among correlated features depend on regularization and normalization.', '',
              '| Policy | Component | Weight | 95% interval | Status |', '|---|---|---:|---|---|']
    for r in outputs['weights_common.csv']:
        point = 'NA' if r['estimated_index_weight'] is None else f"{r['estimated_index_weight']:.5f}"
        interval = 'NA' if r['weight_lower'] is None else f"[{r['weight_lower']:.5f}, {r['weight_upper']:.5f}]"
        lines.append(f"| {r['policy']} | {r['component']} | {point} | {interval} | {r['status']} |")
    lines += ['', '## Interpretation limits', '',
              '- Cost is compared with the stored feasible reference; only certified references can be called optimal.',
              '- Negative cost gaps are preserved in the prepared data but contribute zero economic deterioration to the target.',
              '- Non-delivery gets score 1; unavailable ENS and CO2 stay unknown, never zero.',
              '- Time is actual program exit divided by its own deadline, not time to certified optimality.',
              '- CO2 is direct internal thermal production at the historical factor; it excludes imports, lifecycle and separate startup emissions.',
              '- Territorial emissions per served MWh are a declared proxy, not a consumption footprint: exports are included in thermal generation.',
              '- Total cost already contains ENS and carbon-price penalties; explicit target terms deliberately add emphasis.',
              '- Criterion weights are explicit preference scenarios; equal weights and each priority profile use the same normalizations and paired validation groups.',
              '- No interaction coefficient is folded into the 22 additive weights; the signed ridge diagnostic tests the nonnegative restriction.',
              '- Per-deadline fits and paired held-out gains assess whether common weights lose predictive accuracy.',
              '- These are observational predictive weights on an explored designed cohort, not causal effects or universal weights.', '']
    publish(report_dir / 'REPORT.md', '\n'.join(lines).encode('utf-8'))
    from .data import sha
    names = sorted(list(outputs) + ['models.json', 'REPORT.md'])
    put(Path(run) / 'report_receipt.json', seal(dict(manifest=manifest['sha256'],
        files={name: sha(report_dir / name) for name in names})))
    print('Reports written to ' + str(report_dir), flush=True)
