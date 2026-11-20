"""Read-only interim comparison of an explicitly closed external cohort.

Never refits predictors, solves a MILP or writes campaign/frozen artifacts.
Clipping is a secondary post-freeze sensitivity, not a new frozen predictor.
No inferential intervals are produced for the incomplete external batches.
"""
import argparse
from datetime import datetime, timezone
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd

from ..data import RUN, ROOT, config, read, sha
from ..campaign import CAMP, load_physics
from ..future_analysis import economic_threshold, bounded_economic_status
from ..realized.pipeline import verify_freeze
from ..realized.storage import DEST
from research.paper1_weights.data import dispatch_carbon, target


def run(cutoff):
    end = datetime.fromisoformat(cutoff)
    if end.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    frozen_before = verify_freeze()
    manifest_path = CAMP / 'manifest.json'
    manifest = read(manifest_path)
    manifest_hash = sha(manifest_path)
    jobs = {j['job_id']: j for j in manifest['jobs']}
    cases = {c['group']: c for c in manifest['entries']}
    paths = sorted(p for p in (CAMP / 'results').glob('*.json')
                   if p.stat().st_mtime <= end.timestamp())
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    results = [read(p) for p in paths]
    for d in results:
        assert d['job'] == jobs[d['job']['job_id']]
        assert d['result']['identity']['manifest_sha256'] == manifest_hash
    refs = {d['job']['key']: d for d in results
            if d['job']['kind'] == 'external_reference' and d['result']['success']}
    trials = [d for d in results if d['job']['kind'] == 'external_trial'
              and d['job']['key'] in refs]
    cfg = read(CAMP / 'predictors_frozen.json')['target_definition']
    policies = config()['policies']
    demand_cache, rows = {}, []
    for d in trials:
        j, outcome = d['job'], d['result']
        key = j['key']
        case = cases[key]
        assert sha(case['file']) == case['scenario_sha256']
        if key not in demand_cache:
            physical = load_physics(case['file'])
            demand_cache[key] = (sum(physical.demand.values()) * physical.dt_hours,
                                 physical.dt_hours)
        demand, dt = demand_cache[key]
        r = outcome.get('result') or {}
        check = r.get('validation', {})
        reference = refs[key]['result']['result']
        refcost = reference['validation']['cost']
        valid = bool(outcome['success'])
        if valid:
            assert r['valid'] and check['valid'] and outcome['all_processes_exited']
        row = dict(group=key, batch=case['batch'], budget=j['budget'],
                   job_id=j['job_id'], reference_job_id=refs[key]['job']['job_id'],
                   status=outcome['status'], feasible=valid,
                   reference_cost=refcost, reference_lower_bound=reference.get('bound'),
                   reference_certified=bool(reference.get('proven_optimal')),
                   exit_s=outcome['exit_s'], demand_mwh=demand,
                   cost=check.get('cost'), ens_mwh=check.get('ens_mwh'))
        if valid:
            row.update(dispatch_carbon(r['dispatch'], dt, demand, check['ens_mwh'], cfg))
        row['E'] = int(valid and row['cost'] <= economic_threshold(refcost))
        row['ES'] = int(row['E'] and row['ens_mwh'] <= .001)
        row['optimum_status'] = ('delivery_failure' if not valid else
            bounded_economic_status(row['cost'], reference.get('bound'), refcost))
        for policy in policies:
            row[policy] = target(row, cfg, cfg['policies'][policy])['score']
        rows.append(row)
    observations = pd.DataFrame(rows).sort_values(['group', 'budget'])
    assert not observations.duplicated(['group', 'budget']).any()
    labels = observations.melt(id_vars=['group', 'budget'], value_vars=policies + ['E', 'ES'],
                               var_name='target', value_name='y')
    predictions = pd.read_parquet(DEST / 'external_predictions.parquet')
    matched = predictions.merge(labels, on=['group', 'budget', 'target'], validate='many_to_one')
    assert matched.groupby(['target', 'representation', 'kind']).size().eq(len(observations)).all()
    assert np.isfinite(matched[['p', 'y']].to_numpy()).all()
    assert matched.y.between(0, 1).all()
    matched['clipped_p'] = matched.p.clip(0, 1)
    matched['raw_loss'] = (matched.p - matched.y) ** 2
    matched['clipped_loss'] = (matched.clipped_p - matched.y) ** 2
    assert (matched.clipped_loss <= matched.raw_loss + 1e-14).all()
    primary = matched[matched.representation.isin(['legacy_raw', 'realized']) &
                      matched.kind.isin(['positive', 'logistic'])].copy()
    comparisons = []
    for scope, frame in [('all', primary)] + [('batch:' + str(b), q) for b, q in primary.groupby('batch')] + [
            ('budget:' + str(b), q) for b, q in primary.groupby('budget')] + [
            ('certified_only', primary[primary.group.isin(observations.loc[observations.reference_certified, 'group'])])]:
        for label, q in frame.groupby('target'):
            for weighting in ['trial', 'scenario']:
                scores = q.groupby(['group', 'representation'])[['raw_loss', 'clipped_loss']].mean() if weighting == 'scenario' else q
                scores = scores.groupby('representation')[['raw_loss', 'clipped_loss']].mean()
                a, b = scores.loc['legacy_raw'], scores.loc['realized']
                comparisons.append(dict(scope=scope, target=label, weighting=weighting,
                    n_trials=len(q)//2, n_scenarios=q.group.nunique(),
                    legacy_raw_mse=a.raw_loss, realized_raw_mse=b.raw_loss,
                    raw_delta=b.raw_loss-a.raw_loss,
                    raw_gain_pct=100*(1-b.raw_loss/a.raw_loss) if a.raw_loss else None,
                    legacy_clipped_mse=a.clipped_loss, realized_clipped_mse=b.clipped_loss,
                    clipped_delta=b.clipped_loss-a.clipped_loss,
                    clipped_gain_pct=100*(1-b.clipped_loss/a.clipped_loss) if a.clipped_loss else None))
    bounds = primary.groupby(['target', 'representation']).agg(n=('p', 'size'),
        below_zero=('p', lambda x: int((x < 0).sum())), above_one=('p', lambda x: int((x > 1).sum())),
        p_min=('p', 'min'), p_max=('p', 'max'), y_mean=('y', 'mean'))
    assert verify_freeze() == frozen_before
    assert sha(manifest_path) == manifest_hash
    for p in paths:
        assert sha(p) == source_hashes[str(p.relative_to(ROOT))]
    output = RUN / 'diagnostics' / ('interim_external_v1_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    observations.to_csv(output / 'observations.csv', index=False)
    matched.to_parquet(output / 'matched_predictions.parquet', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'paired_comparison.csv', index=False)
    bounds.to_csv(output / 'prediction_ranges.csv')
    receipt = dict(cutoff_utc=end.astimezone(timezone.utc).isoformat(),
        recorded_jobs=len(results), paired_trials=len(observations), scenarios=observations.group.nunique(),
        manifest_sha256=manifest_hash, result_hashes=source_hashes, frozen=frozen_before,
        script_sha256=sha(__file__), selected_job_ids=observations.job_id.tolist(),
        model_refit=False, milp_solve=False, frozen_unchanged=True,
        note='Incomplete available-case cohort; descriptive only. Clipping is a post-freeze sensitivity, not a newly frozen model. No fractional-logit or B9-ablation fits applied.')
    (output / 'snapshot.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), paired_trials=len(observations), scenarios=observations.group.nunique())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', required=True)
    run(parser.parse_args().cutoff)
