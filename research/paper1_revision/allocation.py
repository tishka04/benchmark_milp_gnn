"""Exact multiple-choice budget allocation separately inside each held-out fold."""
import numpy as np
import pandas as pd
from .data import RUN,load,put,digest
from .analysis import collect


def allocate(values,budgets,total):
    """Maximize predicted successes; integral 5-second units, at most total."""
    values=np.asarray(values,float); budgets=np.asarray(budgets,int)
    n,k=values.shape
    if k!=len(budgets) or not np.isfinite(values).all(): raise ValueError('Invalid allocation matrix')
    minimum=int(budgets.min()); step=int(np.gcd.reduce(budgets))
    if total<n*minimum: raise ValueError('Cannot reserve the minimum deadline for each case')
    cap=min((int(total)-n*minimum)//step,n*(int(budgets.max())-minimum)//step)
    costs=(budgets-minimum)//step
    dp=np.full(cap+1,-np.inf); dp[0]=0.
    history=np.zeros((n,cap+1),np.int8)
    for i in range(n):
        nxt=np.full(cap+1,-np.inf); choice=np.zeros(cap+1,np.int8)
        for j,cost in enumerate(costs):
            if cost>cap: continue
            candidate=dp[:cap+1-cost]+values[i,j]
            improve=candidate>nxt[cost:]+1e-12
            nxt[cost:][improve]=candidate[improve];choice[cost:][improve]=j
        dp=nxt; history[i]=choice
    remaining=int(np.argmax(dp)); selected=np.zeros(n,int)
    for i in range(n-1,-1,-1):
        selected[i]=history[i,remaining];remaining-=int(costs[selected[i]])
    assert budgets[selected].sum()<=total
    return selected


def uniform_mix(n,budgets,total):
    average=total/n
    low=max(i for i,b in enumerate(budgets) if b<=average)
    selected=np.full(n,low,int)
    if low+1<len(budgets):
        extra=int((total-n*budgets[low])//(budgets[low+1]-budgets[low]))
        selected[:extra]=low+1
    return selected


def run():
    df,cfg=load(); pred=collect('probability_nested')
    expected=5*(4*2*2+5)
    files=list((RUN/'checkpoints/probability_nested').glob('*.json'))
    if len(files)!=expected: raise ValueError(f'Probability stage incomplete: {len(files)}/{expected}')
    budgets=np.array(cfg['deadlines']); results=[]; draws=[]; selections=[]
    models=['uniform','size_direct','original_direct','compact_direct','all_direct','oracle']
    for endpoint in ['ES','E']:
        for fold in range(5):
            test=df[df.outer_fold==fold]; groups=sorted(test.group.unique(),key=lambda g:digest(['allocation-order',g]))
            y=test.pivot(index='group',columns='budget',values=endpoint).loc[groups,budgets].to_numpy()
            times=test.pivot(index='group',columns='budget',values='exit_s').loc[groups,budgets].to_numpy()
            values={m:pred[(pred.target==endpoint)&(pred.model==m)&(pred.fold==str(fold))].pivot(index='group',columns='budget',values='p').loc[groups,budgets].to_numpy() for m in models if m not in ('uniform','oracle')}
            values['oracle']=y
            n=len(groups)
            for average in cfg['allocation_mean_budgets']:
                total=n*average
                for repetition in range(-1,cfg['allocation_bootstrap_samples']):
                    rng=np.random.default_rng(int(digest([cfg['seed'],'portfolio',endpoint,fold,average,repetition])[:8],16))
                    indices=np.arange(n) if repetition==-1 else rng.integers(0,n,n)
                    for model in models:
                        selected=uniform_mix(n,budgets,total) if model=='uniform' else allocate(values[model][indices],budgets,total)
                        actual=float(times[indices,selected].sum()); reserved=int(budgets[selected].sum())
                        row=dict(endpoint=endpoint,fold=fold,mean_budget=average,model=model,successes=int(y[indices,selected].sum()),
                            reserved_seconds=reserved,observed_exit_seconds=actual,portfolio_budget=total,
                            observed_overrun_seconds=max(0.,actual-total),repetition=repetition)
                        if repetition==-1:
                            results.append(row)
                            selections.extend(dict(endpoint=endpoint,fold=fold,mean_budget=average,model=model,group=g,
                                selected_budget=int(budgets[j])) for g,j in zip(groups,selected))
                        else: draws.append(row)
                print('allocation',endpoint,fold,average,flush=True)
    pd.DataFrame(results).to_csv(RUN/'allocation_results.csv',index=False)
    pd.DataFrame(selections).to_csv(RUN/'allocation_decisions.csv',index=False)
    pd.DataFrame(draws).to_parquet(RUN/'allocation_bootstrap.parquet',index=False)
    put(RUN/'allocation_protocol.json',dict(objective='ES primary, E secondary',partition='optimization separately per outer test fold',
        uncertainty='Resample cases within each held-out portfolio and solve allocation again; predictive models fixed.',
        oracle='Retrospective observed-outcome oracle, not a deployable policy',budget='reserved deadlines; observed exit separately, no strict wall-time guarantee'))
