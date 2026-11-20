"""Component audit, score conventions, references and observed timing variation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import RUN,HOME,WEIGHTS,COMPONENTS,load,read,put,sha,digest
from .analysis import splits,tuned,signature,collect


def diagnostic():
    df,cfg=load(); one=df.drop_duplicates('group')
    records=[]
    for c in COMPONENTS:
        for family,q in [('all',one)]+list(one.groupby('family')):
            x=q[c]; raw=q['raw_'+c]
            records.append(dict(component=c,family=family,n=len(q),unique=x.nunique(),raw_min=raw.min(),raw_max=raw.max(),
                normalized_min=x.min(),normalized_max=x.max(),saturated_low=float((x<=0).mean()),saturated_high=float((x>=1).mean())))
    pd.DataFrame(records).to_csv(RUN/'component_diagnostics.csv',index=False)
    one[list(COMPONENTS)].corr().to_csv(RUN/'component_correlations.csv')
    audit=dict(A3_formula='round(1.8 * demand_scale_factor, 2)',A3_is_realized_peak_valley=False,
        A3_reconstruction_max_error=float(np.max(np.abs(one.raw_A3_peak_to_valley-np.round(1.8*one.raw_A5_demand_scale,2)))),
        A3_A5_pearson=float(one.A3_peak_to_valley.corr(one.A5_demand_scale)),
        size_rank=int(np.linalg.matrix_rank(one[['n_zones','n_steps','n_lines','zone_steps']].to_numpy()-one[['n_zones','n_steps','n_lines','zone_steps']].mean().to_numpy())),
        correction='Rename A3 to demand-scale-derived proxy; preserve historical numerical definition; true ratio requires a versioned feature.')
    put(RUN/'definition_audit.json',audit)
    # Test feature coverage using only each training population's range.
    shifts=[]
    for family in sorted(one.family.unique()):
        a=one[one.family!=family]; b=one[one.family==family]
        for c in COMPONENTS:
            shifts.append(dict(family=family,component=c,training_min=a[c].min(),training_max=a[c].max(),
                outside_training_range=float(((b[c]<a[c].min()-1e-12)|(b[c]>a[c].max()+1e-12)).mean())))
    pd.DataFrame(shifts).to_csv(RUN/'family_covariate_shift.csv',index=False)
    stages=[]
    for budget,q in df.groupby('budget'):
        stages.append(dict(budget=int(budget),n=len(q),not_delivered=int((q.F==0).sum()),
                          delivered_quality_failure=int(((q.F==1)&(q.E==0)).sum()),
                          economic_service_failure=int(((q.E==1)&(q.ES==0)).sum()),success_ES=int(q.ES.sum())))
    pd.DataFrame(stages).to_csv(RUN/'failure_decomposition.csv',index=False)
    r=pd.read_parquet(RUN/'runs.parquet'); r=r[(r.method=='highs')&r.budget.isin(cfg['deadlines'])]
    repeated=set(r.loc[r.repetition>0,'group']); q=r[r.group.isin(repeated)].copy()
    q['E']=q.economic_5.astype(float); q['ES']=q.E*((q.ens_mwh<=.001)&q.feasible)
    freq=q.groupby(['group','family','budget']).agg(repetitions=('repetition','size'),delivery_rate=('feasible','mean'),
        economic_rate=('E','mean'),joint_rate=('ES','mean'),exit_mean=('exit_s','mean'),exit_std=('exit_s','std'),
        compute_mean=('compute_s','mean'),finalization_mean=('finalization_s','mean'),n_zones=('n_zones','first')).reset_index()
    freq.to_csv(RUN/'repetition_frequencies.csv',index=False)
    r.groupby(['budget','repetition']).agg(n=('group','size'),exit_mean=('exit_s','mean'),compute_mean=('compute_s','mean'),
        finalization_mean=('finalization_s','mean'),preload_mean=('preload_s','mean')).to_csv(RUN/'historical_timing.csv')
    one.groupby(['reference_certified','family']).agg(n=('group','size'),zones_mean=('n_zones','mean'),zones_min=('n_zones','min'),zones_max=('n_zones','max')).to_csv(RUN/'reference_case_mix.csv')
    refs=pd.read_parquet(RUN/'references.parquet')
    put(RUN/'reference_bound_audit.json',dict(references=len(refs),certified=int(refs.certified.sum()),
        recorded_dual_bounds=int(refs.lower_bound.notna().sum()),
        unknown_optimum=int((~refs.certified).sum()),
        note='No dual bounds available in archived reference records; certified status retained, unknown bounds not imputed.'))


def score_variants(df,cfg):
    legacy=read(WEIGHTS/'data.json')['cfg']; records=df.to_dict('records')
    refs=pd.read_parquet(RUN/'references.parquet').set_index('group')
    for policy in cfg['policies']:
        original=legacy['policies'][policy]
        variants=[('main',{},original)]
        for field,values in [('ens_fraction_scale',[.005,.02]),('cost_gap_scale',[.025,.1]),('co2_intensity_scale_t_per_mwh',[.36,1.44])]:
            for value in values: variants.append((field+'_'+str(value),{field:value},original))
        if policy!='equal4':
            priority=max(original,key=original.get)
            for value in [.4,.8]:
                weights={k:value if k==priority else (1-value)/3 for k in original}
                variants.append(('priority_'+str(value),{},weights))
        variants.append(('disjoint_cost',{},original))
        for name,changes,weights in variants:
            local=dict(legacy,**changes)
            from research.paper1_weights.data import target
            values=[]
            for r in records:
                r=dict(r)
                if name=='disjoint_cost' and r['feasible']:
                    costs=json.loads(refs.loc[r['group'],'cost_components_json'])
                    if 'unserved_energy' not in costs or not np.isfinite(r['cost_co2_charge_eur']):
                        raise ValueError('Cost decomposition unavailable; no imputation')
                    refcarbon=r['reference_co2_direct_t']*r['co2_price_eur_per_t']
                    r['cost']-=r['cost_ens_penalty_eur']+r['cost_co2_charge_eur']
                    r['reference_cost']-=costs['unserved_energy']+refcarbon
                values.append(target(r,local,weights)['score'])
            yield policy,name,np.array(values)


def sensitivity():
    df,cfg=load(); sig=digest([signature(),sha(HOME/'robustness.py')]); stability=[]; scores=[]
    for policy,name,values in score_variants(df,cfg):
        work=df.copy(); work['response']=values
        for b,q in work.groupby('budget'):
            stability.append(dict(policy=policy,variant=name,budget=int(b),
                rank_correlation=float(spearmanr(q[policy],q.response).statistic),
                mean_abs_target_change=float(np.mean(np.abs(q[policy]-q.response)))))
        for fold,train,test,inner in splits(work,'nested'):
            for model,rep,mode in [('all_common','all','common'),('all_rank1','all','rank1'),('all_separate','all','separate')]:
                path=RUN/'checkpoints/sensitivity'/policy/name/(fold+'_'+model+'.json')
                if path.exists():
                    obj=read(path)
                    if obj['signature']!=sig: raise ValueError('Stale sensitivity checkpoint')
                else:
                    pred,meta=tuned(train,test,inner,'response',rep,mode,cfg)
                    obj=dict(signature=sig,meta=meta,predictions=[dict(group=r.group,budget=int(r.budget),y=float(r.response),p=float(p)) for r,p in zip(test.itertuples(),pred)])
                    put(path,obj)
                for b in cfg['deadlines']:
                    q=[r for r in obj['predictions'] if r['budget']==b]
                    scores.append(dict(policy=policy,variant=name,fold=fold,model=model,budget=b,mse=float(np.mean([(r['y']-r['p'])**2 for r in q]))))
        print('sensitivity',policy,name,flush=True)
    pd.DataFrame(stability).to_csv(RUN/'target_sensitivity.csv',index=False)
    pd.DataFrame(scores).to_csv(RUN/'sensitivity_scores_by_fold.csv',index=False)
    # Binary thresholds are descriptive label sensitivity, never tuned on held-out labels.
    label=[]
    for threshold in [.01,.05,.1]:
        near=np.abs(df.reference_cost)<=1
        success=(df.F==1)&np.where(near,df.cost-df.reference_cost<=1,
            (df.cost-df.reference_cost)/np.maximum(abs(df.reference_cost),1)<=threshold)
        for b in cfg['deadlines']:
            mask=df.budget==b
            label.append(dict(threshold=threshold,budget=b,success_E=int(success[mask].sum()),
                changed_from_5pct=int((success[mask].astype(int)!=df.E[mask]).sum())))
    pd.DataFrame(label).to_csv(RUN/'binary_threshold_sensitivity.csv',index=False)


def run():
    diagnostic(); sensitivity()
