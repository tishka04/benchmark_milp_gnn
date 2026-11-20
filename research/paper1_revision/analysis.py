"""Nested analysis jobs; each checkpoint records input and model fingerprints."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .data import RUN,HOME,WEIGHTS,load,read,put,sha,digest,COMPONENTS
from .models import fit_predict,design,hurdle_predict,fractional_fit_predict


def signature():
    return digest({str(p):sha(p) for p in [RUN/'manifest.json',HOME/'protocol.json',HOME/'models.py',HOME/'analysis.py']})


def specifications(transfer=False):
    result=[('constant','constant','separate')]
    for rep in ['original','size','compact','six','all']:
        for mode in (['common','rank1','separate'] if rep not in ['original'] else ['common','separate']):
            result.append((rep+'_'+mode,rep,mode))
    if transfer:
        return [s for s in result if s[0] in ['size_separate','original_separate','compact_separate','all_rank1','all_separate']]
    result += [('boosting','all','boosting')]
    result += [(rep+'_separate',rep,'separate') for rep in ['reduced','grouped','nonsaturating']]
    return result


def splits(df,scheme):
    info=read(RUN/'splits.json')
    if scheme=='nested':
        for f in range(5):
            train=df[df.outer_fold!=f]; test=df[df.outer_fold==f]
            inner=train.group.map(info['inner'][str(f)]).to_numpy()
            yield str(f),train,test,inner
    elif scheme=='lofo':
        for f in sorted(df.family.unique()):
            train=df[df.family!=f]; test=df[df.family==f]
            mapping={name:i for i,name in enumerate(sorted(train.family.unique()))}
            yield f,train,test,train.family.map(mapping).to_numpy()
    else: raise ValueError(scheme)


def tuned(train,test,inner,target,rep,mode,cfg):
    if mode=='separate' and train.budget.nunique()>1:
        result=np.zeros(len(test)); models={}
        for b in sorted(train.budget.unique()):
            mask=train.budget.to_numpy()==b; testmask=test.budget.to_numpy()==b
            result[testmask],models[str(int(b))]=tuned(train[mask],test[testmask],inner[mask],target,rep,mode,cfg)
        return result,{'by_deadline':models}
    grid=([0] if rep in ('constant','original') else [7,15,31] if mode=='boosting' else cfg['ridge_grid'])
    tuning=[]
    for ridge in grid:
        total=0.; count=0
        for f in np.unique(inner):
            a,b=train.iloc[np.flatnonzero(inner!=f)],train.iloc[np.flatnonzero(inner==f)]
            assert not set(a.group)&set(b.group)
            pred,_=fit_predict(a,b,target,rep,mode,ridge,cfg)
            total+=float(np.sum((pred-b[target].to_numpy())**2)); count+=len(b)
        tuning.append({'ridge':ridge,'mse':total/count})
    best=min(t['mse'] for t in tuning)
    ridge=max(t['ridge'] for t in tuning if t['mse']<=best+1e-12)
    pred,model=fit_predict(train,test,target,rep,mode,ridge,cfg)
    return pred,dict(model=model,tuning=tuning,selected_ridge=ridge)


def fit(scheme='nested'):
    df,cfg=load(); sig=signature(); jobs=specifications(scheme=='lofo')
    for target in cfg['policies']:
        for fold,train,test,inner in splits(df,scheme):
            for name,rep,mode in jobs:
                path=RUN/'checkpoints'/scheme/target/(fold+'_'+name+'.json')
                if path.exists():
                    if read(path)['signature']!=sig: raise ValueError('Changed checkpoint inputs: '+str(path))
                    continue
                start=time.perf_counter()
                pred,meta=tuned(train,test,inner,target,rep,mode,cfg)
                records=[dict(group=r.group,budget=int(r.budget),family=r.family,fold=fold,
                    y=float(getattr(r,target)),p=float(p),reference_certified=bool(r.reference_certified))
                    for r,p in zip(test.itertuples(),pred)]
                put(path,dict(signature=sig,target=target,name=name,scheme=scheme,fold=fold,
                    train_groups=sorted(train.group.unique()),meta=meta,predictions=records,
                    elapsed_seconds=time.perf_counter()-start))
                print(scheme,target,fold,name,round(time.perf_counter()-start,2),flush=True)


def binary_and_hurdle():
    df,cfg=load(); sig=signature()
    for scheme in ['nested','lofo']:
        for fold,train_all,test_all,inner_all in splits(df,scheme):
            for rep in ['size','original','compact','all']:
                targets=['E','ES']+(cfg['policies'] if scheme=='nested' and rep=='all' else [])
                for target in targets:
                    for kind in (['direct','hurdle'] if target in ('E','ES') else ['hurdle']):
                        path=RUN/'checkpoints'/('probability_'+scheme)/(fold+'_'+rep+'_'+target+'_'+kind+'.json')
                        if path.exists():
                            if read(path)['signature']!=sig: raise ValueError('Stale binary checkpoint')
                            continue
                        records=[]; selection=[]
                        for budget in cfg['deadlines']:
                            mask=train_all.budget.to_numpy()==budget
                            train=train_all[mask]; test=test_all[test_all.budget==budget]; inner=inner_all[mask]
                            def predictor(a,b,ridge):
                                if kind=='hurdle': return hurdle_predict(a,b,target,rep,ridge)
                                x,state=design(a,rep); xt,_=design(b,rep,state)
                                return fractional_fit_predict(x,a[target].to_numpy(float),xt,ridge)
                            scores=[]
                            for ridge in [.0001,.001,.01,.1,1]:
                                losses=[]
                                for f in np.unique(inner):
                                    a=train.iloc[np.flatnonzero(inner!=f)]; b=train.iloc[np.flatnonzero(inner==f)]
                                    p=predictor(a,b,ridge); losses.extend((p-b[target].to_numpy())**2)
                                scores.append((float(np.mean(losses)),ridge))
                            best=min(v[0] for v in scores); ridge=max(v[1] for v in scores if v[0]<=best+1e-12)
                            pred=predictor(train,test,ridge)
                            records.extend(dict(group=r.group,budget=int(budget),family=r.family,fold=fold,
                                y=float(getattr(r,target)),p=float(p)) for r,p in zip(test.itertuples(),pred))
                            selection.append(dict(budget=budget,ridge=ridge,scores=scores))
                        put(path,dict(signature=sig,scheme=scheme,fold=fold,representation=rep,target=target,kind=kind,
                                      train_groups=sorted(train_all.group.unique()),selection=selection,predictions=records))
                        print('probability',scheme,fold,rep,target,kind,flush=True)


def collect(stage):
    rows=[]
    for path in sorted((RUN/'checkpoints'/stage).rglob('*.json')):
        obj=read(path)
        for row in obj['predictions']:
            rows.append(dict(row,target=obj['target'],model=obj.get('name',obj.get('representation','')+'_'+obj.get('kind',''))))
    return pd.DataFrame(rows)
