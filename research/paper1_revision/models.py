"""Comparable nonnegative regressions, including an effective-ridge rank-one fit."""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from research.paper1_weights.data import COMPONENTS, ORIGINAL_WEIGHTS

SIX = [1,5,7,9,14,18]
PHYSICAL_GROUPS = [[1,3],[2,4],[6,7]]


def design(frame, representation, state=None):
    x=frame[list(COMPONENTS)].to_numpy(float,copy=True)
    if representation=='constant': return np.empty((len(x),0)),{}
    if representation=='original': return (x@np.array(ORIGINAL_WEIGHTS))[:,None],{}
    if representation=='size': return x[:,[12]],{}
    if representation=='compact': return x[:,[12,14,15]],{}
    if representation=='six': return x[:,SIX],{}
    if representation=='nonsaturating':
        from src.analysis.criticality_index import NORM_BOUNDS
        for j in [5,19]:
            lo,hi=NORM_BOUNDS[COMPONENTS[j]]
            z=(frame['raw_'+COMPONENTS[j]].to_numpy(float)-lo)/(hi-lo)
            x[:,j]=.5+np.arctan(z-.5)/np.pi
    if representation=='grouped':
        used=set(sum(PHYSICAL_GROUPS,[]))
        return np.column_stack([x[:,g].mean(axis=1) for g in PHYSICAL_GROUPS]+[x[:,i] for i in range(22) if i not in used]),{}
    if representation=='reduced':
        if state is None:
            keep=[]
            for i in range(x.shape[1]):
                if np.ptp(x[:,i])<=1e-12: continue
                if any(abs(np.corrcoef(x[:,i],x[:,j])[0,1])>.95 for j in keep): continue
                keep.append(i)
            state={'keep':keep}
        return x[:,state['keep']],state
    return x,{}


def positive_fit(x,y,ridge):
    active=np.ptp(x,axis=0)>1e-12
    coef=np.zeros(x.shape[1]); n=len(y)
    if active.any():
        a=x[:,active]/np.sqrt(n); rhs=y/np.sqrt(n)
        if ridge:
            a=np.vstack([a,np.sqrt(ridge)*np.eye(active.sum())]); rhs=np.r_[rhs,np.zeros(active.sum())]
        coef[active]=nnls(a,rhs,maxiter=10000)[0]
    return coef


def linear_fit(x,y,budgets,mode,ridge,cfg):
    levels=np.unique(budgets); B=len(levels)
    means={int(b):(x[budgets==b].mean(axis=0),float(y[budgets==b].mean())) for b in levels}
    xc=np.vstack([x[i]-means[int(b)][0] for i,b in enumerate(budgets)])
    yc=np.array([y[i]-means[int(b)][1] for i,b in enumerate(budgets)])
    diagnostics=[]
    if mode=='common':
        coef=positive_fit(xc,yc,ridge)
        coefficients=np.tile(coef,(B,1))
    elif mode=='separate':
        coefficients=np.array([positive_fit(xc[budgets==b],yc[budgets==b],ridge) for b in levels])
    elif mode=='rank1':
        if len({sum(budgets==b) for b in levels})!=1:
            raise ValueError('Rank-one protocol requires balanced scenario/deadline rows')
        if not x.shape[1]: return linear_fit(x,y,budgets,'common',ridge,cfg)
        common=positive_fit(xc,yc,ridge)
        active=np.ptp(xc,axis=0)>1e-12
        # B1 in the full/size/compact designs; first declared coordinate otherwise.
        starts=[common,active.astype(float),np.eye(x.shape[1])[12 if x.shape[1]==22 else 0]*active]
        candidates=[]
        gram=np.array([xc[budgets==b].T@xc[budgets==b]/sum(budgets==b) for b in levels])
        cross=np.array([xc[budgets==b].T@yc[budgets==b]/sum(budgets==b) for b in levels])
        variance=float(yc@yc/len(yc))
        def amplitudes(w):
            den=np.einsum('i,bij,j->b',w,gram,w)+ridge*(w@w)
            return np.maximum(0,np.divide(cross@w,den,out=np.zeros(B),where=den>1e-20))
        for label,start in zip(cfg['rank1_starts'],starts):
            w=start.copy()
            if w.sum()<1e-14: w=active.astype(float)
            if w.sum()<1e-14: return linear_fit(x,y,budgets,'common',ridge,cfg)
            w/=w.sum(); prev=np.inf; converged=False
            for iteration in range(cfg['rank1_max_iter']):
                alpha=amplitudes(w)
                hessian=np.einsum('b,bij->ij',alpha**2,gram)/B+ridge*np.mean(alpha**2)*np.eye(x.shape[1])
                rhs=np.mean(alpha[:,None]*cross,axis=0)
                eigen,vectors=np.linalg.eigh(hessian)
                supported=eigen>max(float(eigen.max())*1e-13,1e-20)
                new=np.zeros_like(w)
                if supported.any():
                    a=np.sqrt(eigen[supported])[:,None]*vectors[:,supported].T
                    z=(vectors[:,supported].T@rhs)/np.sqrt(eigen[supported])
                    new[active]=nnls(a[:,active],z,maxiter=10000)[0]
                if new.sum()>1e-14:
                    alpha*=new.sum(); w=new/new.sum()
                else:
                    alpha[:]=0
                # Refit amplitudes after each normalized direction update.
                alpha=amplitudes(w)
                effective=alpha[:,None]*w[None,:]
                obj=float(variance-2*np.mean(np.sum(effective*cross,axis=1))+
                          np.mean(np.einsum('bi,bij,bj->b',effective,gram,effective))+
                          ridge*np.mean(np.sum(effective**2,axis=1)))
                if np.isfinite(prev) and abs(prev-obj)<=cfg['rank1_tolerance']*(1+abs(prev)):
                    converged=True; break
                if obj>prev+1e-9: raise RuntimeError('Alternating objective increased')
                prev=obj
            diagnostics.append(dict(start=label,objective=obj,iterations=iteration+1,converged=converged))
            candidates.append((obj,effective))
        coefficients=min(candidates,key=lambda v:v[0])[1]
    else: raise ValueError(mode)
    intercepts=np.array([means[int(b)][1]-means[int(b)][0]@coef for b,coef in zip(levels,coefficients)])
    return dict(levels=levels,coefficients=coefficients,intercepts=intercepts,diagnostics=diagnostics)


def linear_predict(model,x,budgets):
    positions=np.searchsorted(model['levels'],budgets)
    return model['intercepts'][positions]+np.einsum('ij,ij->i',x,model['coefficients'][positions])


def fit_predict(train,test,target,representation,mode,ridge,cfg):
    x,state=design(train,representation); xt,_=design(test,representation,state)
    y=train[target].to_numpy(float)
    if mode=='boosting':
        outputs=np.zeros(len(test)); models=[]
        for b in sorted(train.budget.unique()):
            mask=train.budget.to_numpy()==b; mt=test.budget.to_numpy()==b
            model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=int(ridge),learning_rate=.05,
                        min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=cfg['seed'])
            model.fit(x[mask],y[mask]); outputs[mt]=model.predict(xt[mt])
        return outputs,{'diagnostics':[], 'features':x.shape[1], 'state':state}
    model=linear_fit(x,y,train.budget.to_numpy(),mode,ridge,cfg)
    pred=linear_predict(model,xt,test.budget.to_numpy())
    return pred,{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}|{'features':x.shape[1],'state':state}


def fractional_fit_predict(x,y,xt,ridge):
    """Fractional Bernoulli quasi-likelihood includes outcomes exactly 0 or 1."""
    if not len(y): return np.full(len(xt),.5)
    if np.all(y==y[0]): return np.full(len(xt),y[0])
    if len(y)<10: return np.full(len(xt),(sum(y)+1)/(len(y)+2))
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-12]=1
    X=np.c_[np.ones(len(x)),(x-mean)/scale]; T=np.c_[np.ones(len(xt)),(xt-mean)/scale]
    def fun(beta):
        eta=X@beta; p=expit(eta)
        val=np.mean(np.logaddexp(0,eta)-y*eta)+ridge/2*(beta[1:]@beta[1:])
        grad=X.T@(p-y)/len(y); grad[1:]+=ridge*beta[1:]
        return val,grad
    fitted=minimize(fun,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-10})
    if not fitted.success: raise RuntimeError('Fractional fit: '+fitted.message)
    return expit(T@fitted.x)


def hurdle_predict(train,test,target,representation,ridge):
    x,state=design(train,representation); xt,_=design(test,representation,state)
    delivery=train.F.to_numpy(bool)
    f=fractional_fit_predict(x,delivery.astype(float),xt,ridge)
    if target in ('E','ES'):
        q=fractional_fit_predict(x[delivery],train.E.to_numpy(float)[delivery],xt,ridge)
        good=train.E.to_numpy(bool)
        service=fractional_fit_predict(x[good],train.ES.to_numpy(float)[good],xt,ridge)
        return f*q if target=='E' else f*q*service
    loss=fractional_fit_predict(x[delivery],train[target].to_numpy(float)[delivery],xt,ridge)
    return (1-f)+f*loss
