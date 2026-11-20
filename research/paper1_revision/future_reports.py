"""Post-process future reference upgrades and timing traces; never run a solver."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .data import RUN, read, put, sha
from .campaign import CAMP
from .future_analysis import analyze, economic_threshold, bounded_economic_status
from research.paper1_weights.data import target


def reassess_references(historical, references, upgrades, cfg):
    """Re-evaluate the same dispatches; keep old incumbents when upgrades fail."""
    rows = []
    for original in historical.to_dict('records'):
        key = original['group']
        if key not in upgrades:
            continue
        outcome = upgrades[key]
        result = outcome.get('result') or {}
        accepted = bool(outcome.get('success') and result.get('valid'))
        old = float(references.loc[key, 'cost'])
        new = min(old, float(result['validation']['cost'])) if accepted else old
        # A completed run may return a valid dual bound without an incumbent.
        lower = result.get('bound')
        if lower is not None and (not np.isfinite(lower) or lower > new + 1e-6):
            lower = None
        revised = dict(original, reference_cost=new)
        feasible = bool(original['feasible'])
        before_e = int(feasible and original['cost'] <= economic_threshold(old))
        after_e = int(feasible and original['cost'] <= economic_threshold(new))
        service = bool(feasible and original['ens_mwh'] <= cfg['zero_ens_tolerance_mwh'])
        base = dict(group=key, budget=int(original['budget']), family=original['family'],
                    old_reference=old, best_reference=new, upgrade_accepted=accepted,
                    reference_improved=new < old - 1e-6,
                    lower_bound=lower, optimum_status='delivery_failure' if not feasible
                    else bounded_economic_status(original['cost'], lower, new))
        for name, before, after in [('E', before_e, after_e),
                                    ('ES', before_e * service, after_e * service)]:
            rows.append(dict(base, target=name, before=float(before), after=float(after)))
        for name, weights in cfg['policies'].items():
            if name == 'service3':
                continue
            rows.append(dict(base, target=name,
                             before=target(original, cfg, weights)['score'],
                             after=target(revised, cfg, weights)['score']))
    return pd.DataFrame(rows)


def fixed_prediction_sensitivity(changes, predictions):
    """Conditional label sensitivity with untouched, historical out-of-fold fits."""
    q = predictions.merge(changes[['group', 'budget', 'target', 'before', 'after']],
                          on=['group', 'budget', 'target'], validate='many_to_one')
    q['old_loss'] = (q.p - q.before)**2
    q['new_loss'] = (q.p - q.after)**2
    return q.groupby(['target', 'model', 'budget']).agg(
        n=('group', 'size'), old_mse=('old_loss', 'mean'),
        new_mse=('new_loss', 'mean')).reset_index()


def timing_rows(outcomes, attempts):
    """Retain partial traces as partial: an interrupted stage has no known duration."""
    rows = []
    for obj in outcomes:
        job, outcome = obj['job'], obj['result']
        if job['kind'] != 'timing_repeat':
            continue
        directories = sorted(attempts.glob(job['job_id'] + '_*'))
        traces = [p / 'stages.jsonl' for p in directories if (p / 'stages.jsonl').exists()]
        # Ambiguous multiple attempts must not be silently combined.
        trace = traces[0] if len(traces) == 1 else None
        stages = {}
        if trace is not None:
            for line in trace.read_text(encoding='utf-8').splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    break  # A forced kill may leave an incomplete final line.
                stages[event['stage']] = event['seconds']
        payload = outcome.get('result') or {}
        if not stages and len(traces) <= 1:
            stages = payload.get('stage_seconds', {})
        rows.append(dict(group=job['key'], budget=job['budget'], repetition=job['repetition'],
                         job_id=job['job_id'], delivered=bool(outcome.get('success')),
                         status=outcome.get('status'), exit_s=outcome.get('exit_s'),
                         preload_s=outcome.get('preload_s'),
                         trace_ambiguous=len(traces) > 1,
                         completed_stage_count=len(stages),
                         **{'stage_' + k: v for k, v in stages.items()}))
    return pd.DataFrame(rows)


def run():
    analyze()
    manifest_hash = sha(CAMP / 'manifest.json')
    outcomes = [read(p) for p in (CAMP / 'results').glob('*.json')]
    for obj in outcomes:
        if obj['result']['identity']['manifest_sha256'] != manifest_hash:
            raise ValueError('Outcome from a different physical campaign')
    if not outcomes:
        put(CAMP / 'additional_reports_status.json', dict(status='not_executed', observed_jobs=0))
        return
    upgrades = {o['job']['key']: o['result'] for o in outcomes if o['job']['kind'] == 'reference_upgrade'}
    if upgrades:
        history = pd.read_parquet(RUN / 'analysis.parquet')
        refs = pd.read_parquet(RUN / 'references.parquet').set_index('group')
        cfg = read(CAMP / 'predictors_frozen.json')['target_definition']
        changes = reassess_references(history, refs, upgrades, cfg)
        changes.to_csv(CAMP / 'reference_target_changes.csv', index=False)
        binary = changes[changes.target.isin(['E', 'ES'])]
        binary.groupby(['target', 'budget', 'before', 'after']).size().rename('n').to_csv(
            CAMP / 'reference_label_transition_matrix.csv')
        predictions = pd.concat([pd.read_parquet(RUN / (name + '.parquet')) for name in
                                 ['continuous_nested_predictions', 'probability_nested_predictions']],
                                ignore_index=True)
        fixed_prediction_sensitivity(changes, predictions).to_csv(
            CAMP / 'reference_prediction_sensitivity.csv', index=False)
    timings = timing_rows(outcomes, CAMP / 'attempts')
    if len(timings):
        timings.to_csv(CAMP / 'instrumented_timing_rows.csv', index=False)
        stages = [c for c in timings if c.startswith('stage_')]
        if stages:
            timings.groupby(['budget', 'delivered'])[stages].agg(['count', 'mean', 'median']).to_csv(
                CAMP / 'instrumented_timing_summary.csv')
        timings.groupby(['group', 'budget']).agg(
            observed_repetitions=('repetition', 'size'), delivery_rate=('delivered', 'mean'),
            exit_mean=('exit_s', 'mean'), exit_std=('exit_s', 'std')).to_csv(
                CAMP / 'instrumented_repetition_frequencies.csv')
    put(CAMP / 'additional_reports_status.json', dict(status='available_for_observed_jobs',
        observed_jobs=len(outcomes), reference_upgrades_observed=len(upgrades), timing_jobs_observed=len(timings),
        note='Same dispatches and fixed held-out predictions; no retuning or reoptimization. Missing or interrupted stage durations remain unknown. Large reference changes require separately versioned refits.'))


if __name__ == '__main__':
    run()
