"""Check earlier service failures against archived feasible dispatches; no new solves."""
import pandas as pd

from ..campaign import CAMP
from ..data import RUN, read
from ..future_analysis import economic_threshold


def run():
    out = RUN / 'diagnostics' / 'decision_replay_20261119'
    out.mkdir(exist_ok=True)
    history = pd.read_parquet(RUN / 'analysis.parquet')
    external = pd.read_parquet(CAMP / 'external_observations.parquet')
    refs = pd.read_parquet(RUN / 'references.parquet').set_index('group')
    jobs = read(CAMP / 'manifest.json')['jobs']
    lookup = {(j['kind'], j['key']): CAMP / 'results' / (j['job_id'] + '.json')
              for j in jobs if j['kind'] in ['reference_upgrade', 'external_reference']}
    threshold = 0.001
    rows = []
    for cohort, frame in [('development', history), ('independent', external)]:
        for group, cases in frame.groupby('group'):
            last = cases.loc[cases.budget.eq(120)].iloc[0]
            if last.feasible and last.ens_mwh <= threshold:
                continue
            sources = []
            if cohort == 'development':
                sources.append(('historical_reference', refs.loc[group, 'source']))
            for kind in ('reference_upgrade', 'external_reference'):
                if (kind, group) in lookup:
                    sources.append((kind, lookup[kind, group]))
            witnesses = []
            for kind, path in sources:
                obj = read(path)['result']
                payload = obj.get('result') if kind != 'historical_reference' else obj
                if payload and payload.get('valid') and payload['validation']['ens_mwh'] <= threshold:
                    witnesses.append(kind)
            deadline_witness = (cases.feasible & cases.ens_mwh.le(threshold)).any()
            rows.append(dict(cohort=cohort, group=group, delivery_120=last.feasible,
                             ens_120_mwh=last.ens_mwh, other_deadline_witness=deadline_witness,
                             reference_witness=bool(witnesses), witness_sources='|'.join(witnesses)))
    service = pd.DataFrame(rows)
    service.to_csv(out / 'service_witnesses.csv', index=False)

    strengthened = pd.read_csv(CAMP / 'reference_target_changes.csv')
    costs = refs.cost.copy()
    costs.update(strengthened.groupby('group').best_reference.first())
    repeat_rows = []
    for job in jobs:
        if job['kind'] != 'timing_repeat':
            continue
        result = read(CAMP / 'results' / (job['job_id'] + '.json'))['result']
        payload = result.get('result') or {}
        delivered = bool(result['success'] and payload.get('valid'))
        metrics = payload.get('validation') or {}
        economic = delivered and metrics['cost'] <= economic_threshold(costs.loc[job['key']])
        joint = economic and metrics['ens_mwh'] <= threshold
        repeat_rows.append(dict(group=job['key'], budget=job['budget'], delivered=delivered,
                                E=economic, ES=joint, repetition=job['repetition']))
    repeats = pd.DataFrame(repeat_rows)
    if len(repeats) != 450 or not repeats.groupby(['group', 'budget']).size().eq(3).all():
        raise ValueError('Incomplete instrumented repeat triples')
    repeats.to_csv(out / 'instrumented_repeats.csv', index=False)
    print(service.groupby(['cohort', 'delivery_120', 'reference_witness', 'other_deadline_witness']).size())
    print(repeats.groupby(['budget', 'group'])[['delivered', 'E', 'ES']].nunique().gt(1).groupby('budget').sum())


if __name__ == '__main__':
    run()
