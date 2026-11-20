"""Automatically derive secondary tables and reproducibility metadata."""
import json
import time
import numpy as np
import pandas as pd
from .data import RUN,ROOT,HOME,COMPONENTS,put,sha
from .report import LABEL

def run():
    s=pd.read_parquet(RUN/'scenarios.parquet');records=[]
    for row in s.itertuples():
        raw=json.loads(row.parameters_json);record=dict(group=row.group,family=row.family,zones=row.n_zones,steps=row.n_steps,lines=row.n_lines)
        for block in ['tech','econ_policy','operation_costs','flexibility_metrics','exogenous','graph']:
            record.update({block+'.'+k:v for k,v in raw.get(block,{}).items() if isinstance(v,(int,float,str))})
        record.update({'assets.'+k:v for k,v in raw['meta']['assets'].items()});records.append(record)
    params=pd.DataFrame(records);params.to_csv(RUN/'scenario_parameters.csv',index=False)
    numeric=params.select_dtypes('number').columns
    params.groupby('family')[numeric].agg(['min','max','mean']).to_csv(RUN/'scenario_ranges_by_family.csv')
    from src.analysis.criticality_index import compute_criticality,compute_hardness_metrics,normalize_metric,NORM_BOUNDS
    samples=[]
    for raw in [json.loads(x) for x in s.parameters_json]:
        for mode in ['size','compact','six','all']:
            start=time.perf_counter()
            for _ in range(10):
                if mode=='size':normalize_metric(raw['meta']['zones'],NORM_BOUNDS['B1_n_zones'])
                elif mode=='compact':
                    h=compute_hardness_metrics(raw)
                    for c in ['B1_n_zones','B3_binary_est','B4_thermal_density']:normalize_metric(getattr(h,c),NORM_BOUNDS[c])
                else:compute_criticality(raw)
            samples.append(dict(representation=mode,microseconds=(time.perf_counter()-start)*1e5))
    timings=pd.DataFrame(samples).groupby('representation').microseconds.agg(['median',lambda x:np.quantile(x,.95)]).rename(columns={'<lambda_0>':'p95'})
    timings.to_csv(RUN/'descriptor_timings.csv')
    put(RUN/'descriptor_timing_protocol.json',dict(scope='Warm computation from already-loaded JSON, 10 repetitions per case, current machine; excludes file I/O and physical profile generation',
        caveat='The six-feature implementation shares the full 22-feature extractor. These measurements are not historical execution overheads.'))
    lines=[r'\section{Secondary sensitivities and execution variability}',r'\label{app:robustness_results}',
        r'All 48 declared preference/target specifications were fitted with the same outer folds. Across their anchor, priority and disjoint-cost variants, the pooled gain from a common direction to separate directions remains positive. This does not imply that individual coefficients or ranks are stable.',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{lrr}\toprule Preference & Minimum gain (\%) & Maximum gain (\%)\\\midrule']
    a=pd.read_csv(RUN/'sensitivity_scores_by_fold.csv').groupby(['policy','variant','model']).mse.mean().unstack()
    ranges=(100*(1-a.all_separate/a.all_rank1)).groupby('policy').agg(['min','max'])
    for policy,row in ranges.iterrows():lines.append(LABEL[policy].replace('CO2',r'CO$_2$')+f' & {row["min"]:.1f} & {row["max"]:.1f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Descriptive sensitivity of the pooled MSE gain from changing direction beyond deadline-specific scaling. No variant is selected using test performance.}\end{table}',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule Preference & Linear & Projected & Two-part\\\midrule']
    bounded=pd.read_csv(RUN/'bounded_model_comparison.csv').groupby('target')[['linear_mse','clipped_mse','two_part_mse']].mean()
    for policy,row in bounded.iterrows():lines.append(LABEL[policy].replace('CO2',r'CO$_2$')+' & '+' & '.join(f'{x:.5f}' for x in row)+r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Pooled held-out MSE for the separate linear, projected linear and bounded two-part models using all 22 descriptors.}\end{table}',
        r'\begin{table}[htbp]\centering\small\begin{tabular}{rrrr}\toprule Deadline (s) & Repeated cases & Variable delivery & Variable ES\\\midrule']
    repeated=pd.read_csv(RUN/'repetition_frequencies.csv')
    for budget,q in repeated.groupby('budget'):
        lines.append(f'{budget} & {len(q)} & {sum((q.delivery_rate>0)&(q.delivery_rate<1))} & {sum((q.joint_rate>0)&(q.joint_rate<1))}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\caption{Cases with differing outcomes across the primary run and two archived repetitions. The 30 selected cases do not establish deployment reliability.}\end{table}',
        r'Archived logs separate aggregate computation, finalization, preload and process exit, but do not identify model construction and solver search individually. Therefore the fraction of short-deadline failures caused specifically by model construction cannot be estimated from these logs.',
        r'The 383 certified and 217 uncertified reference cases differ in family and size. Subgroup losses and case-mix tables are released without treating the certified group as representative. No numerical dual bounds were retained in the 600 archived reference records. Stronger-reference label changes remain unobserved pending the prepared solves.',
        r'For the ES allocation, the 22-component gain over uniform allocation at 30 seconds per scenario is 74 successes; the conditional portfolio-bootstrap 95\% interval is approximately [59,90]. Relative to the size-only allocation, the point gain is 20 with interval approximately [10,31]. At 60 seconds, the gain over size is only 8 and its interval includes zero. These pointwise intervals condition on predictive fits and do not provide a simultaneous guarantee across budgets.',
        r'The full code, normalized and raw descriptors, complete held-out predictions, selected penalties, convergence diagnostics, grouped weights, scenario parameters and decision allocations are retained under \path{research/paper1_revision}. The reproduction check covers 480 historical MSE cells and 400 Brier cells, and refitted baseline predictions agree with their archived counterparts.']
    out=ROOT/'research/three_papers/manuscripts/paper1/results/revision_robustness.tex';out.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    # Derived Brier skill scores use training prevalence, never test prevalence.
    from .analysis import collect
    df=pd.read_parquet(RUN/'analysis.parquet');skill=[]
    for scheme in ['nested','lofo']:
        p=collect('probability_'+scheme)
        for (target,model,budget),q in p[p.target.isin(['E','ES'])].groupby(['target','model','budget']):
            baseline=[]
            for r in q.itertuples():
                train=df[(df.budget==budget)&(df.outer_fold!=int(r.fold) if scheme=='nested' else df.family!=r.fold)]
                baseline.append((r.y-train[target].mean())**2)
            brier=float(np.mean((q.y-q.p)**2));base=float(np.mean(baseline))
            skill.append(dict(scheme=scheme,target=target,model=model,budget=budget,brier=brier,training_prevalence_brier=base,brier_skill=1-brier/base if base>0 else None))
    pd.DataFrame(skill).to_csv(RUN/'brier_skill.csv',index=False)

if __name__=='__main__':run()
