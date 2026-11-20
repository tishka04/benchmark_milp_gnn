"""Summarize the completed external test using saved predictions only."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import RUN, ROOT, read, sha
from ..campaign import CAMP
from ..realized.pipeline import verify_freeze
from ..realized.storage import DEST
from .interim_b9_attribution import diagnose


def intervals(frame, columns, keys):
    """Paired scenario bootstrap, stratified by the four experimental batches."""
    vectors, labels = [], []
    for label, part in frame.groupby(keys):
        values = part.groupby('group')[columns].mean()
        vectors.append(values)
        labels.append(label if isinstance(label, tuple) else (label,))
    groups = sorted(frame.group.unique())
    values = np.column_stack([v.loc[groups].to_numpy() for v in vectors])
    batches = frame.drop_duplicates('group').set_index('group').loc[groups, 'batch']
    strata = [np.flatnonzero(batches.to_numpy() == b) for b in sorted(batches.unique())]
    rng = np.random.default_rng(20261119)
    boot = np.array([values[np.concatenate([rng.choice(s, len(s)) for s in strata])].mean(axis=0)
                     for _ in range(2000)])
    means, se = values.mean(axis=0), boot.std(axis=0, ddof=1)
    critical = np.quantile(np.max(np.abs((boot-means)/np.maximum(se, 1e-15)), axis=1), .95)
    rows = []
    for i, label in enumerate(labels):
        for k, name in enumerate(columns):
            j = i * len(columns) + k
            lo, hi = np.quantile(boot[:, j], [.025, .975])
            rows.append(dict(zip(keys, label), contrast=name, mean=means[j], lower=lo, upper=hi,
                             simultaneous_lower=means[j]-critical*se[j],
                             simultaneous_upper=means[j]+critical*se[j], scenarios=len(groups),
                             simultaneous_comparisons=values.shape[1]))
    return pd.DataFrame(rows)


def main():
    frozen = verify_freeze()
    manifest = read(CAMP/'manifest.json')
    jobs = {j['job_id']: j for j in manifest['jobs']}
    paths = sorted((CAMP/'results').glob('*.json'))
    assert {p.stem for p in paths} == set(jobs) and len(paths) == 3067
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    manifest_hash = sha(CAMP/'manifest.json')
    for p in paths:
        outcome = read(p)
        assert outcome['job'] == jobs[p.stem]
        assert outcome['result']['identity']['manifest_sha256'] == manifest_hash
    observed = pd.read_parquet(CAMP/'external_observations.parquet')
    assert len(observed) == 2000 and observed.group.nunique() == 400
    assert observed.groupby('batch').group.nunique().eq(100).all()
    assert not observed.duplicated(['group', 'budget']).any()
    assert observed.groupby('group').budget.apply(lambda x: set(x) == {5, 10, 30, 60, 120}).all()
    ids = {(j['key'], j['budget']): j['job_id'] for j in jobs.values() if j['kind'] == 'external_trial'}
    observed['job_id'] = [ids[g, b] for g, b in zip(observed.group, observed.budget)]
    targets = ['equal4', 'cost_priority', 'co2_priority', 'time_priority', 'service4_co2', 'E', 'ES']
    labels = observed.melt(id_vars=['group', 'budget'], value_vars=targets, var_name='target', value_name='y')
    matched = pd.read_parquet(DEST/'external_predictions.parquet').merge(
        labels, on=['group', 'budget', 'target'], validate='many_to_one')
    saved = pd.read_parquet(DEST/'external_evaluation.parquet')
    pd.testing.assert_frame_equal(matched, saved)
    assert matched.groupby(['target', 'representation', 'kind']).size().eq(2000).all()
    assert np.isfinite(matched[['p', 'y']]).all().all() and matched.y.between(0, 1).all()
    matched['raw_loss'] = (matched.p-matched.y)**2
    matched['clipped_loss'] = (matched.p.clip(0, 1)-matched.y)**2
    matched['baseline_loss'] = (matched.training_mean-matched.y)**2
    output = RUN/'diagnostics'/('final_external_v1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    observed.to_csv(output/'observations.csv', index=False)
    matched.to_parquet(output/'matched_predictions.parquet', index=False)
    receipt = dict(frozen=frozen, recorded_jobs=3067, paired_trials=2000, scenarios=400,
                   manifest_sha256=manifest_hash, result_hashes=hashes,
                   model_refit=False, milp_solve=False, script_sha256=sha(__file__),
                   inference='2000 stratified scenario bootstrap draws, conditional on frozen fits; pooled summaries added after test inspection')
    (output/'snapshot.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    previous = RUN/'diagnostics/interim_external_v1_20261116T121000Z'
    diagnose(output, previous)
    primary = matched[matched.kind.isin(['positive', 'logistic'])]
    summary = []
    scopes = [('all', primary)] + [('batch:'+b, q) for b, q in primary.groupby('batch')]
    scopes += [('budget:'+str(b), q) for b, q in primary.groupby('budget')]
    certified = observed.loc[observed.reference_certified, 'group'].unique()
    scopes += [('certified', primary[primary.group.isin(certified)])]
    for scope, frame in scopes:
        for (target, rep), q in frame.groupby(['target', 'representation']):
            raw, clip, base = q[['raw_loss', 'clipped_loss', 'baseline_loss']].mean()
            summary.append(dict(scope=scope, target=target, representation=rep, n=len(q),
                                raw_mse=raw, clipped_mse=clip, baseline_mse=base,
                                raw_skill=1-raw/base, clipped_skill=1-clip/base,
                                below_zero=int((q.p<0).sum()), above_one=int((q.p>1).sum()),
                                p_min=q.p.min(), p_max=q.p.max()))
    pd.DataFrame(summary).to_csv(output/'scores.csv', index=False)
    paired = pd.read_parquet(output/'b9_paired_diagnostic.parquet')
    for mode in ['raw', 'clipped']:
        paired[mode+'_milp_minus_legacy'] = paired['milp_'+mode+'_loss']-paired['legacy_'+mode+'_loss']
    intervals(paired, ['raw_milp_minus_legacy'], ['target']).to_csv(output/'primary_pooled_intervals.csv', index=False)
    intervals(paired, ['raw_milp_minus_legacy'], ['target', 'budget']).to_csv(output/'primary_deadline_intervals.csv', index=False)
    intervals(paired, ['clipped_milp_minus_legacy'], ['target']).to_csv(output/'clipped_pooled_intervals.csv', index=False)
    paired['clipped_b9_gain'] = paired.milp_clipped_loss-paired.milp_b9_in_range_clipped_loss
    intervals(paired[paired.kind.eq('positive')], ['clipped_b9_gain'], ['target']).to_csv(output/'b9_diagnostic_intervals.csv', index=False)
    wide = primary.pivot(index=['group', 'batch', 'budget', 'target'],
                         columns='representation', values='clipped_loss').reset_index()
    wide['compact_minus_legacy'] = wide.realized_compact-wide.legacy_raw
    wide['compact_minus_realized'] = wide.realized_compact-wide.realized
    intervals(wide, ['compact_minus_legacy', 'compact_minus_realized'], ['target']).to_csv(
        output/'compact_pooled_intervals.csv', index=False)
    wide['model_size_minus_compact'] = wide.model_size-wide.realized_compact
    intervals(wide, ['model_size_minus_compact'], ['target']).to_csv(
        output/'model_size_pooled_intervals.csv', index=False)
    assert verify_freeze() == frozen
    assert all(sha(ROOT/p) == h for p, h in hashes.items())
    print(output)
    print(pd.DataFrame(summary).query("scope == 'all' and representation in ['legacy_raw','realized','realized_compact']")
          [['target','representation','raw_mse','clipped_mse','clipped_skill','below_zero','above_one']].to_string(index=False))


if __name__ == '__main__':
    main()
