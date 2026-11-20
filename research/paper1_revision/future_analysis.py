"""Freeze historical predictors and evaluate future outputs without test tuning."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from .data import RUN,HOME,ROOT,WEIGHTS,COMPONENTS,load,read,put,sha,digest
from .campaign import CAMP
from .analysis import tuned
from .models import design,linear_predict,fractional_fit_predict
from research.paper1_weights.model import group_folds


def freeze():
    from threadpoolctl import threadpool_limits
    df,cfg=load();inner=group_folds({'group':df.group.to_numpy(),'family':df.family.to_numpy()},3,cfg['historical_fold_seed'],'final-inner')
    result={}
    with threadpool_limits(limits=1):
        for target in cfg['policies']:
            for name,rep,mode in [('size','size','separate'),('original','original','separate'),('compact','compact','separate'),('scaled','all','rank1'),('separate','all','separate')]:
                _,meta=tuned(df,df,inner,target,rep,mode,cfg)
                result[target+'/'+name]=dict(representation=rep,mode=mode,fit=meta)
        from scipy.optimize import minimize
        from scipy.special import expit
        for target in ['E','ES']:
            for rep in ['size','original','compact','all']:
                fitted={}
                for budget in cfg['deadlines']:
                    mask=df.budget.to_numpy()==budget;q=df[mask];folds=inner[mask];tuning=[]
                    for ridge in [.0001,.001,.01,.1,1]:
                        losses=[]
                        for f in np.unique(folds):
                            a=q.iloc[np.flatnonzero(folds!=f)];b=q.iloc[np.flatnonzero(folds==f)]
                            x,state=design(a,rep);xt,_=design(b,rep,state)
                            pred=fractional_fit_predict(x,a[target].to_numpy(float),xt,ridge)
                            losses.extend((pred-b[target].to_numpy())**2)
                        tuning.append((float(np.mean(losses)),ridge))
                    best=min(v[0] for v in tuning);ridge=max(v[1] for v in tuning if v[0]<=best+1e-12)
                    x,state=design(q,rep);y=q[target].to_numpy(float);mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1
                    X=np.c_[np.ones(len(x)),(x-mean)/scale]
                    def objective(beta):
                        eta=X@beta;p=expit(eta);gradient=X.T@(p-y)/len(y);gradient[1:]+=ridge*beta[1:]
                        return float(np.mean(np.logaddexp(0,eta)-y*eta)+ridge/2*(beta[1:]@beta[1:])),gradient
                    solution=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-10})
                    if not solution.success:raise RuntimeError(solution.message)
                    assert np.allclose(expit(X@solution.x),fractional_fit_predict(x,y,x,ridge),atol=1e-10)
                    fitted[str(budget)]=dict(mean=mean.tolist(),scale=scale.tolist(),coefficients=solution.x.tolist(),ridge=ridge,state=state,tuning=tuning)
                result[target+'/'+rep]=dict(representation=rep,mode='fractional',fit=fitted)
    payload=dict(training_manifest=sha(RUN/'manifest.json'),source_hashes={p.name:sha(p) for p in [HOME/'models.py',HOME/'analysis.py',HOME/'future_analysis.py']},
        target_definition=read(WEIGHTS/'data.json')['cfg'],models=result,
        scope='Five continuous preferences and direct E/ES models; immutable historical-only fits. No external outcomes used.',
        frozen_before_test=True)
    dest=CAMP/'predictors_frozen.json'
    if dest.exists() and read(dest)!=payload: raise ValueError('Predictors changed after freeze; use a new version')
    put(dest,payload)


def predict(entry,frame):
    meta=entry['fit'];output=np.zeros(len(frame));rep=entry['representation']
    if entry['mode']=='fractional':
        from scipy.special import expit
        for b,m in meta.items():
            mask=frame.budget.to_numpy()==int(b)
            if not mask.any():continue
            x,_=design(frame[mask],rep,m['state']);X=np.c_[np.ones(mask.sum()),(x-np.array(m['mean']))/np.array(m['scale'])]
            output[mask]=expit(X@np.array(m['coefficients']))
        return output
    items=meta['by_deadline'].items() if 'by_deadline' in meta else [('all',meta)]
    for b,item in items:
        mask=np.ones(len(frame),bool) if b=='all' else frame.budget.to_numpy()==int(b)
        if not mask.any():continue
        m=item['model'];x,_=design(frame[mask],rep,m['state'])
        numerical={k:np.asarray(v) for k,v in m.items() if k in ('levels','coefficients','intercepts')}
        output[mask]=linear_predict(numerical,x,frame.loc[mask,'budget'].to_numpy())
    return output


def economic_threshold(reference,tolerance=.05):
    return reference+1 if abs(reference)<=1 else reference+tolerance*abs(reference)


def bounded_economic_status(cost,lower,upper,tolerance=.05):
    """Account for negative/near-zero optimal costs and threshold discontinuities."""
    if upper is None:return 'indeterminate'
    candidates=[upper]
    if lower is not None:
        if lower>upper+1e-6:raise ValueError('Reversed objective bounds')
        candidates.append(lower)
    for edge in [-1.,1.]:
        for x in [edge,np.nextafter(edge,-np.inf),np.nextafter(edge,np.inf)]:
            if (lower is None or x>=lower) and x<=upper:candidates.append(x)
    thresholds=[economic_threshold(x,tolerance) for x in candidates]
    if cost>max(thresholds):return 'certain_failure'
    if lower is not None and cost<=min(thresholds):return 'certain_success'
    return 'indeterminate'


def analyze():
    from research.paper1_weights.data import dispatch_carbon,target
    from .campaign import load_physics
    manifest=read(CAMP/'manifest.json');rows=[];results={}
    for path in (CAMP/'results').glob('*.json'):
        o=read(path);j=o['job'];result=o['result']
        if result['identity']['manifest_sha256']!=sha(CAMP/'manifest.json'):raise ValueError('Wrong physical campaign manifest')
        results[(j['kind'],j['key'],j['budget'],j['repetition'])]=o
    if not results:
        put(CAMP/'evaluation_status.json',dict(status='not_executed',observed_jobs=0,expected_jobs=len(manifest['jobs'])))
        return
    old=pd.read_parquet(RUN/'references.parquet').set_index('group');historical=pd.read_parquet(RUN/'analysis.parquet')
    upgrades=[]
    for key,row in old.iterrows():
        found=[v for (kind,k,_,_),v in results.items() if kind=='reference_upgrade' and k==key]
        if not found:continue
        outcome=found[0]['result'];r=outcome.get('result') or {};valid=bool(outcome.get('success') and r.get('valid'))
        new=min(float(row['cost']),r['validation']['cost']) if valid else float(row['cost'])
        lower=r.get('bound') if valid else None
        for obs in historical[historical.group==key].itertuples():
            before=bool(obs.F and obs.cost<=economic_threshold(row['cost']))
            after=bool(obs.F and obs.cost<=economic_threshold(new))
            upgrades.append(dict(group=key,budget=obs.budget,old_reference=row['cost'],best_reference=new,new_reference_valid=valid,
                before_E=before,after_E=after,changed=before!=after,lower_bound=lower,
                optimum_status='delivery_failure' if not obs.F else bounded_economic_status(obs.cost,lower,new)))
    pd.DataFrame(upgrades).to_csv(CAMP/'reference_upgrade_sensitivity.csv',index=False)
    predictors=read(CAMP/'predictors_frozen.json');legacy=predictors['target_definition']
    if any(sha(HOME/name)!=expected for name,expected in predictors['source_hashes'].items()):
        raise ValueError('Predictive code changed after freeze; refuse unversioned external evaluation')
    cases={c['group']:c for c in manifest['entries']}
    for (kind,key,budget,rep),o in results.items():
        if kind!='external_trial':continue
        outcome=o['result'];c=cases[key]
        reference=[v for (k,g,_,_),v in results.items() if k=='external_reference' and g==key]
        if not reference or not reference[0]['result'].get('success'):continue
        rr=reference[0]['result']['result'];refcost=rr['validation']['cost']
        data=load_physics(c['file']);demand=sum(data.demand.values())*data.dt_hours
        valid=bool(outcome.get('success'));r=outcome.get('result') or {};check=r.get('validation',{})
        record=dict(group=key,batch=c['batch'],budget=budget,feasible=valid,reference_cost=refcost,
            reference_certified=bool(rr.get('proven_optimal')),exit_s=outcome['exit_s'],demand_mwh=demand,
            cost=check.get('cost'),ens_mwh=check.get('ens_mwh'),**c['components'])
        if valid:record.update(dispatch_carbon(r['dispatch'],data.dt_hours,demand,check['ens_mwh'],legacy))
        record['E']=int(valid and record['cost']<=economic_threshold(refcost))
        record['ES']=int(record['E'] and record['ens_mwh']<=.001)
        for policy in legacy['policies']:
            if policy!='service3':record[policy]=target(record,legacy,legacy['policies'][policy])['score']
        rows.append(record)
    if rows:
        frame=pd.DataFrame(rows);frame.to_parquet(CAMP/'external_observations.parquet',index=False);scores=[]
        for name,entry in predictors['models'].items():
            policy,model=name.split('/');p=predict(entry,frame)
            for (batch,b),q in frame.assign(predicted=p).groupby(['batch','budget']):
                scores.append(dict(policy=policy,model=model,batch=batch,budget=b,n=len(q),
                    mse=float(np.mean((q.predicted-q[policy])**2)),complete=len(q)==manifest['config']['external_cases_per_batch']))
        pd.DataFrame(scores).to_csv(CAMP/'external_scores.csv',index=False)
    put(CAMP/'evaluation_status.json',dict(status='partial' if len(results)<len(manifest['jobs']) else 'complete',
        observed_jobs=len(results),expected_jobs=len(manifest['jobs']),external_rows_with_valid_reference=len(rows),
        note='Unexecuted trials and unavailable references are not failures and receive no fabricated outcome.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','analyze']);a=p.parse_args()
    freeze() if a.action=='freeze' else analyze()
