"""Coherent nested conditional probabilities with shared E/ES tuning."""
import numpy as np
from .data import RUN,HOME,load,read,put,sha,digest
from .analysis import splits,signature
from .models import design,hurdle_predict,fractional_fit_predict

def run():
    df,cfg=load(); sig=digest([signature(),sha(HOME/'probability.py')])
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
                            def predictor(a,b,ridge,outcome=target):
                                if kind=='hurdle': return hurdle_predict(a,b,outcome,rep,ridge)
                                x,state=design(a,rep); xt,_=design(b,rep,state)
                                return fractional_fit_predict(x,a[outcome].to_numpy(float),xt,ridge)
                            scores=[]
                            # Identical tuning criterion for E and ES enforces shared f,q,s.
                            endpoints=['E','ES'] if kind=='hurdle' and target in ('E','ES') else [target]
                            for ridge in [.0001,.001,.01,.1,1]:
                                losses=[]
                                for f in np.unique(inner):
                                    a=train.iloc[np.flatnonzero(inner!=f)]; b=train.iloc[np.flatnonzero(inner==f)]
                                    for outcome in endpoints:
                                        p=predictor(a,b,ridge,outcome); losses.extend((p-b[outcome].to_numpy())**2)
                                scores.append((float(np.mean(losses)),ridge))
                            best=min(v[0] for v in scores); ridge=max(v[1] for v in scores if v[0]<=best+1e-12)
                            pred=predictor(train,test,ridge)
                            records.extend(dict(group=r.group,budget=int(budget),family=r.family,fold=fold,
                                y=float(getattr(r,target)),p=float(p),reference_certified=bool(r.reference_certified)) for r,p in zip(test.itertuples(),pred))
                            selection.append(dict(budget=budget,ridge=ridge,scores=scores))
                        put(path,dict(signature=sig,scheme=scheme,fold=fold,representation=rep,target=target,kind=kind,
                                      train_groups=sorted(train_all.group.unique()),selection=selection,predictions=records))
                        print('probability',scheme,fold,rep,target,kind,flush=True)
