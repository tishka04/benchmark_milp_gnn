"""Grouped prediction, calibration and paired coverage analyses, executed by user."""
from __future__ import annotations
from pathlib import Path
import itertools
import json
import math
import uuid
import numpy as np
import pandas as pd
from .core import config, seed, read, sha, put, sealed, frozen, writable

SIZE=["n_zones","n_steps","n_lines","zone_steps"]
COMPONENTS=["A1_vre_penetration","A2_residual_volatility","A3_peak_to_valley","A4_short_term_variability",
 "A5_demand_scale","A6_peak_to_firm","A7_inv_storage_power","A8_inv_storage_energy","A9_inv_thermal_flex",
 "A10_inv_dr_headroom","A11_trade_reliance","A12_inv_congestion","B1_n_zones","B2_horizon","B3_binary_est",
 "B4_thermal_density","B5_min_gen_tight","B6_startup_intensity","B7_ramp_tightness","B8_soc_tightness",
 "B9_interconn_density","B10_network_hetero"]
INTERPRETABLE=["A2_residual_volatility","A6_peak_to_firm","A8_inv_storage_energy",
               "A10_inv_dr_headroom","B3_binary_est","B7_ramp_tightness"]
DESCRIPTIONS={"size":SIZE,"index":SIZE+["criticality"],"axes":SIZE+["stress","hardness"],
              "components":SIZE+COMPONENTS,"interpretable":SIZE+INTERPRETABLE}

def frame(path):
    from .core import verify
    obj=verify(read(path))
    return pd.DataFrame(obj["rows"])

def folds(df):
    return df.group.map(lambda k:seed("outer-fold",k)%config()["folds"]).to_numpy()

def interval(values,groups):
    """Bootstrap independent base groups; repeated rows receive equal group weight."""
    tmp=pd.DataFrame({"value":values,"group":groups}).dropna().groupby("group").value.mean()
    if len(tmp)==0: return (None,None,None,0)
    arr=tmp.to_numpy(); rng=np.random.default_rng(seed("bootstrap"))
    samples=np.array([rng.choice(arr,len(arr),replace=True).mean() for _ in range(config()["bootstrap_samples"])])
    return float(arr.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975)),len(arr)

def fit_probability(df,y,features,kind="logistic"):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    numeric=[k for k in features if k!="family"]
    tr=[("numeric",Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
                               ("scale",StandardScaler())]),numeric)]
    if "family" in features: tr.append(("family",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["family"]))
    estimator=(LogisticRegression(C=1.,max_iter=2000) if kind=="logistic" else
               DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,random_state=42) if kind=="tree" else
               HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,random_state=42))
    if len(set(y))<2: estimator=DummyClassifier(strategy="prior")
    model=Pipeline([("prep",ColumnTransformer(tr)),("model",estimator)])
    model.fit(df[features],np.asarray(y,dtype=int))
    return model

def probability(model,df):
    classes=list(model.named_steps["model"].classes_)
    return model.predict_proba(df)[:,classes.index(1)] if 1 in classes else np.zeros(len(df))

def metrics(y,p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    if not len(y): return dict(n=0,auc=None,average_precision=None,brier=None,log_loss=None)
    return dict(n=len(y),prevalence=float(np.mean(y)),
                auc=float(roc_auc_score(y,p)) if len(set(y))>1 else None,
                average_precision=float(average_precision_score(y,p)) if np.sum(y)>0 else None,
                brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))

def coverage(df):
    rows=[]
    for keys,part in df.groupby(["partition","condition","family","method","budget","repetition"],dropna=False):
        base=dict(zip(["partition","condition","family","method","budget","repetition"],keys))
        for outcome in ("feasible","economic_1","economic_5","economic_10"):
            mean,lo,hi,n=interval(pd.to_numeric(part[outcome],errors="coerce"),part.group)
            rows.append(dict(**base,outcome=outcome,estimate=mean,lower=lo,upper=hi,groups=n,
                expected=len(part),observed=int(part.observed.sum()),known=int(part[outcome].notna().sum()),
                unexecuted=int((~part.observed).sum()),unknown_reference=int(part.reference_cost.isna().sum())))
    return pd.DataFrame(rows)

def pairs(df):
    rows=[]
    for (partition,condition,budget),part in df[df.repetition==0].groupby(["partition","condition","budget"]):
        for outcome in ("feasible","economic_5"):
            wide=part.pivot(index=["group","key"],columns="method",values=outcome)
            for a,b in itertools.combinations(wide.columns,2):
                common=wide[[a,b]].dropna()
                value,lo,hi,n=interval(common[a].astype(float)-common[b].astype(float),common.index.get_level_values("group"))
                rows.append(dict(partition=partition,condition=condition,budget=budget,outcome=outcome,
                    first=a,second=b,difference=value,lower=lo,upper=hi,groups=n,paired_cells=len(common)))
    return pd.DataFrame(rows)

def paired_conditions(df):
    rows=[]
    primary=df[df.repetition==0]
    for (method,budget),part in primary.groupby(["method","budget"]):
        for outcome in ("feasible","economic_5"):
            wide=part.pivot(index="group",columns="condition",values=outcome)
            if "base" not in wide: continue
            for condition in wide.columns:
                if condition=="base": continue
                q=wide[["base",condition]].dropna()
                v,lo,hi,n=interval(q[condition].astype(float)-q["base"].astype(float),q.index)
                rows.append(dict(method=method,budget=budget,condition=condition,outcome=outcome,
                                 difference_from_base=v,lower=lo,upper=hi,groups=n))
    return pd.DataFrame(rows)

def predictive_gains(pred):
    rows=[]
    for (control,validation,budget),part in pred.groupby(["family_control","validation","budget"]):
        wide=part.pivot(index=["group","key","y"],columns="description",values="p")
        if "size" not in wide: continue
        for description in wide.columns:
            if description=="size": continue
            q=wide[["size",description]].dropna(); y=q.index.get_level_values("y").to_numpy()
            for metric in ("brier","log_loss"):
                def loss(p):
                    p=np.clip(p,1e-7,1-1e-7)
                    return (y-p)**2 if metric=="brier" else -(y*np.log(p)+(1-y)*np.log(1-p))
                gain=loss(q["size"].to_numpy())-loss(q[description].to_numpy())
                v,lo,hi,n=interval(gain,q.index.get_level_values("group"))
                rows.append(dict(family_control=control,validation=validation,budget=budget,
                    description=description,metric=metric,gain=v,lower=lo,upper=hi,groups=n))
    return pd.DataFrame(rows)

def prediction(df):
    """All preprocessing fits only on the training groups of each fold."""
    data=df[(df.method=="highs")&(df.repetition==0)&df.economic_5.notna()].copy()
    out=[]; coefficients=[]
    for description,base in DESCRIPTIONS.items():
        for control_family in (False,True):
            features=base+["budget"]+(["family"] if control_family else [])
            for fold in range(config()["folds"]):
                split=folds(data); train=data[split!=fold]; test=data[split==fold]
                if train.empty or test.empty: continue
                model=fit_probability(train,train.economic_5,features)
                p=probability(model,test[features])
                for idx,prob in zip(test.index,p):
                    r=test.loc[idx]
                    out.append(dict(key=r.key,group=r.group,family=r.family,budget=r.budget,
                        y=int(r.economic_5),p=float(prob),description=description,
                        family_control=control_family,fold=fold,validation="grouped"))
                estimator=model.named_steps["model"]
                if hasattr(estimator,"coef_"):
                    for name,v in zip(model.named_steps["prep"].get_feature_names_out(),estimator.coef_[0]):
                        coefficients.append(dict(description=description,family_control=control_family,
                                                 fold=fold,feature=name,coefficient=float(v)))
        # Family transfer is evaluated separately from family-controlled interpolation.
        for family in FAMILIES_LOCAL(data):
            train=data[data.family!=family]; test=data[data.family==family]
            features=base+["budget"]
            if train.empty or test.empty: continue
            model=fit_probability(train,train.economic_5,features)
            for idx,prob in zip(test.index,probability(model,test[features])):
                r=test.loc[idx]
                out.append(dict(key=r.key,group=r.group,family=r.family,budget=r.budget,
                    y=int(r.economic_5),p=float(prob),description=description,family_control=False,
                    fold=-1,validation="leave_family_out"))
    return pd.DataFrame(out),pd.DataFrame(coefficients,columns=["description","family_control","fold","feature","coefficient"])

def FAMILIES_LOCAL(data):
    return sorted(data.family.unique())

def calibration(pred):
    rows=[]
    for keys,part in pred.groupby(["description","family_control","validation","budget"]):
        for b in range(10):
            q=part[(part.p>=b/10)&(part.p<((b+1)/10) if b<9 else part.p<=1)]
            if len(q):
                mean,lo,hi,n=interval(q.y,q.group)
                rows.append(dict(description=keys[0],family_control=keys[1],validation=keys[2],budget=keys[3],
                    bin=b,predicted=float(q.p.mean()),observed=mean,lower=lo,upper=hi,n=len(q),groups=n))
    return pd.DataFrame(rows)

def output_directory(run,name,input_path):
    final=Path(run)/"artifacts"/name
    if (final/"receipt.json").exists():
        receipt=read(final/"receipt.json")
        if receipt["input_sha256"]!=sha(input_path): raise ValueError("Analysis input changed")
        for rel,h in receipt["outputs"].items():
            if sha(final/rel)!=h: raise ValueError("Analysis artifact changed")
        return final,None
    if final.exists(): raise ValueError("Unexpected unpublished artifact directory")
    tmp=writable(Path(run)/"artifact_attempts"/(name+"_"+uuid.uuid4().hex[:12]))
    tmp.mkdir(parents=True); (tmp/"figures").mkdir(); (tmp/"tables").mkdir()
    return final,tmp

def publish(final,tmp,input_path):
    outputs={p.relative_to(tmp).as_posix():sha(p) for p in tmp.rglob("*") if p.is_file()}
    put(tmp/"receipt.json",sealed(dict(input_sha256=sha(input_path),outputs=outputs)))
    final.parent.mkdir(parents=True,exist_ok=True); tmp.rename(final)

def analyze(run,input_path=None,name="main",paper1=True):
    run=Path(run); frozen(run); input_path=Path(input_path or run/"main_rows.json")
    final,tmp=output_directory(run,name,input_path)
    if tmp is None: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df=frame(input_path)
    from .joint_analysis import write_tables
    write_tables(df,tmp)
    if "profile_id" in df and df.profile_id.notna().any():
        # Request-specific executions cannot be pooled in legacy pivots.
        df.to_csv(tmp/"tables"/"timed_request_rows.csv",index=False)
        publish(final,tmp,input_path)
        return
    cov=coverage(df); paired=pairs(df)
    cov.to_csv(tmp/"tables"/"coverage.csv",index=False)
    paired.to_csv(tmp/"tables"/"paired.csv",index=False)
    paired_conditions(df).to_csv(tmp/"tables"/"paired_conditions.csv",index=False)
    df.groupby(["method","budget","status"],dropna=False).size().rename("n").to_csv(tmp/"tables"/"status.csv")
    df.groupby(["method","budget","selected_method"],dropna=False).size().rename("n").to_csv(tmp/"tables"/"selections.csv")
    # Quality/ENS among successes is explicitly secondary; no invented cost for failure.
    df[df.feasible==True].groupby(["method","budget"])[["relative_gap","cost","ens_mwh","exit_s"]].agg(
        ["count","median","mean"]).to_csv(tmp/"tables"/"successful_quality.csv")
    timecols=[c for c in df if c.startswith("stage_")]+["compute_s","finalization_s","preload_s"]
    df.groupby(["method","budget"])[timecols].agg(["count","median"]).to_csv(tmp/"tables"/"timing.csv")
    df.groupby(["method","budget"])[["repair_changed_fraction"]].agg(["count","median"]).to_csv(tmp/"tables"/"repair.csv")
    saturation=df.drop_duplicates("key").groupby("family")[["sat_"+k for k in COMPONENTS]].mean()
    saturation.to_csv(tmp/"tables"/"saturation.csv")
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=True)
    for ax,family in zip(axes.flat,sorted(df.family.unique())):
        for method,part in df[(df.family==family)&(df.repetition==0)].groupby("method"):
            points=[]
            for budget,q in part.groupby("budget"):
                mean,lo,hi,n=interval(q.economic_5,q.group)
                if mean is not None: points.append((budget,mean,lo,hi))
            if points:
                a=np.array(points); ax.plot(a[:,0],a[:,1],"o-",label=method); ax.fill_between(a[:,0],a[:,2],a[:,3],alpha=.12)
        ax.set(title=family.replace("_scenarios",""),xscale="log",xlabel="Independent budget (s)",ylim=(0,1))
    axes[0,0].set_ylabel("Valid exit and gap <= 5%"); axes[-1,-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(tmp/"figures"/"coverage.pdf",metadata={"CreationDate":None}); plt.close(fig)
    # Repetitions are retained separately in coverage.csv and never inflate primary n.
    if paper1:
        pred,coef=prediction(df)
        if pred.empty:
            put(tmp/"tables"/"prediction_unavailable.json",dict(reason="No known primary HiGHS economic outcomes"))
            publish(final,tmp,input_path)
            return
        pred.to_csv(tmp/"tables"/"predictions.csv",index=False); coef.to_csv(tmp/"tables"/"coefficients.csv",index=False)
        predictive_gains(pred).to_csv(tmp/"tables"/"predictive_gain.csv",index=False)
        scores=[]
        for keys,part in pred.groupby(["description","family_control","validation","budget"]):
            scores.append(dict(description=keys[0],family_control=keys[1],validation=keys[2],budget=keys[3],**metrics(part.y,part.p)))
        scores=pd.DataFrame(scores); scores.to_csv(tmp/"tables"/"prediction_scores.csv",index=False)
        cal=calibration(pred); cal.to_csv(tmp/"tables"/"calibration.csv",index=False)
        fig,ax=plt.subplots(figsize=(7,4))
        q=scores[(scores.validation=="grouped")&(~scores.family_control)]
        for label,part in q.groupby("description"): ax.plot(part.budget,part.brier,"o-",label=label)
        ax.set(xscale="log",xlabel="Independent budget (s)",ylabel="Out-of-fold Brier score"); ax.legend(); fig.tight_layout()
        fig.savefig(tmp/"figures"/"descriptions.pdf",metadata={"CreationDate":None}); plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],"k--")
        for label,part in cal[(cal.validation=="grouped")&(~cal.family_control)&(cal.budget==30)].groupby("description"):
            ax.plot(part.predicted,part.observed,"o-",label=label)
        ax.set(xlabel="Predicted economic success",ylabel="Observed economic success",xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(tmp/"figures"/"calibration.pdf",metadata={"CreationDate":None}); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,7))
        c=coef[(coef.description=="components")&(~coef.family_control)&coef.feature.str.contains("numeric__")]
        names=sorted(c.feature.unique()); values=[c[c.feature==k].coefficient.to_numpy() for k in names]
        if values: ax.boxplot(values,vert=False,labels=names)
        ax.set_xlabel("Standardized logistic coefficient across grouped folds (association)")
        fig.tight_layout(); fig.savefig(tmp/"figures"/"components.pdf",metadata={"CreationDate":None}); plt.close(fig)
    publish(final,tmp,input_path)

def transfer(run,input_path,name):
    run=Path(run); frozen(run)
    final,tmp=output_directory(run,name,input_path)
    if tmp is None: return
    train=frame(run/"main_rows.json"); train=train[(train.method=="highs")&(train.repetition==0)&train.economic_5.notna()]
    test=frame(input_path); test=test[test.method.isin(["highs","gurobi"])&test.economic_5.notna()]
    rows=[]
    for desc,features in DESCRIPTIONS.items():
        features=features+["budget"]
        model=fit_probability(train,train.economic_5,features)
        for (method,condition,budget),part in test.groupby(["method","condition","budget"]):
            p=probability(model,part[features])
            rows.append(dict(description=desc,method=method,condition=condition,budget=budget,**metrics(part.economic_5,p)))
    table=pd.DataFrame(rows); table.to_csv(tmp/"tables"/"transfer.csv",index=False)
    if table.empty:
        put(tmp/"tables"/"transfer_unavailable.json",dict(reason="No known transfer outcomes"))
        publish(final,tmp,input_path)
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for keys,q in table.groupby(["description","method","condition"]):
        ax.plot(q.budget,q.brier,"o-",label=" / ".join(keys))
    ax.set(xscale="log",xlabel="Budget (s)",ylabel="Frozen HiGHS predictor: transfer Brier score")
    ax.legend(fontsize=5); fig.tight_layout(); fig.savefig(tmp/"figures"/"transfer.pdf",metadata={"CreationDate":None}); plt.close(fig)
    publish(final,tmp,input_path)
