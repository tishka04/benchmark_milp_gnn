"""Failure-aware paired analysis. No solver or training imports."""
from pathlib import Path
import itertools
import math
import json
import numpy as np
import pandas as pd
from .core import ROOT, config, read, verify, sha, put, sealed, digest, seed

METHOD_NAMES={"highs":"HiGHS","rh":"RH","jepa":"JEPA","pipeline":"Pipeline",
              "perturb16":"No energy gradient","gurobi":"Gurobi","mh_jepa":"MH JEPA",
              "mh_pipeline":"MH pipeline","mh_perturb16":"MH no gradient"}
COLORS={"highs":"#3269a8","rh":"#d68b23","jepa":"#348f70","pipeline":"#9457a5",
        "perturb16":"#888888","gurobi":"#bb4e52","mh_jepa":"#2c9fbd",
        "mh_pipeline":"#695bc4","mh_perturb16":"#8a8a8a"}

def finite(v):
    return v is not None and isinstance(v,(int,float,np.number)) and math.isfinite(v)

def endpoint(row, threshold=None, ens_cap=None):
    """Three-state conjunction. An observed failure is known without a reference."""
    feasible=row.get("feasible")
    if feasible is False or feasible==0: return 0.
    if feasible is not True and feasible!=1: return np.nan
    checks=[]
    if threshold is not None:
        c,ref=row.get("cost"),row.get("reference_cost")
        if finite(c) and finite(ref):
            delta=c-ref
            checks.append(delta/abs(ref)<=threshold if abs(ref)>config()["reference_zero_eur"]
                          else delta<=config()["near_zero_absolute_tolerance_eur"])
        else: checks.append(None)
    if ens_cap is not None:
        e=row.get("ens_mwh"); checks.append(e<=ens_cap if finite(e) else None)
    if any(v is False or v==False for v in checks if v is not None): return 0.
    return np.nan if None in checks else 1.

def frame(rows):
    df=pd.DataFrame(rows)
    if df.empty: return df
    for name,default in (("suite","four"),("training_seed",0),("condition","historical"),
                         ("partition","historical_development"),("repetition",0)):
        if name not in df: df[name]=default
        df[name]=df[name].fillna(default)
    for name in ("cost","reference_cost","reference_bound","ens_mwh","exit_s","preload_s",
                 "compute_s","finalization_s","demand_mwh","repair_changed_fraction"):
        if name not in df: df[name]=np.nan
    for name,threshold,cap in (("valid",None,None),("continuity",None,.001),
        ("economic_1",.01,None),("economic_5",.05,None),("economic_10",.1,None),
        ("joint_1",.01,.001),("joint_5",.05,.001),("joint_10",.1,.001)):
        df[name]=[endpoint(r,threshold,cap) for r in df.to_dict("records")]
    for cap in config()["ens_caps_mwh"]:
        for threshold in (None,.05):
            name=f"ens_{cap}_cost_{threshold}"
            df[name]=[endpoint(r,threshold,cap) for r in df.to_dict("records")]
    df["ens_share"]=df.ens_mwh/df.demand_mwh.where(df.demand_mwh>0)
    df["relative_gap"]=(df.cost-df.reference_cost)/df.reference_cost.abs().where(df.reference_cost.abs()>1)
    df["absolute_gap_eur"]=df.cost-df.reference_cost
    return df

def interval(values, groups, families, *, samples=None, key="interval", statistic="mean"):
    """Resample entire base groups within each family; retain group row counts."""
    values=np.asarray(values,dtype=float); groups=np.asarray(groups); families=np.asarray(families)
    good=np.isfinite(values); values=values[good]; groups=groups[good]; families=families[good]
    if not len(values): return dict(estimate=None,lower=None,upper=None,n=0,groups=0)
    rng=np.random.default_rng(seed(key)); draws=samples or config()["bootstrap_samples"]
    tab=pd.DataFrame(dict(y=values,group=groups,family=families))
    if tab.groupby("group").family.nunique().max()>1: raise ValueError("A base belongs to multiple families")
    buckets=[part for _,part in tab.groupby("family",sort=True)]
    if statistic=="mean":
        sums=np.zeros(draws); counts=np.zeros(draws)
        for part in buckets:
            grouped=part.groupby("group",sort=True).y.agg(["sum","count"])
            indices=rng.integers(0,len(grouped),size=(draws,len(grouped)))
            sums+=grouped["sum"].to_numpy()[indices].sum(axis=1)
            counts+=grouped["count"].to_numpy()[indices].sum(axis=1)
        boot=sums/counts; estimate=float(values.mean())
    else:
        raise ValueError("Only a mean of explicitly defined paired quantities is supported")
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(estimate=estimate,lower=float(lo),upper=float(hi),n=len(values),groups=tab.group.nunique())

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float); adjusted=np.ones(len(p)); order=np.argsort(p)
    running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]); adjusted[index]=min(1.,running)
    return adjusted

def paired(a,b,metric):
    keys=["key","repetition"]
    if a.duplicated(keys).any() or b.duplicated(keys).any(): raise ValueError("Duplicate paired observation; separate suites/seeds")
    return a[[*keys,"group","family",metric]].merge(b[[*keys,metric]],on=keys,suffixes=("_a","_b"),validate="one_to_one")

GROUPS=["partition","suite","training_seed","repetition","condition","method","budget"]
ENDPOINTS=["valid","continuity","economic_1","economic_5","economic_10","joint_1","joint_5","joint_10"]

def coverage_table(df):
    rows=[]
    for family_view in (False,True):
        keys=GROUPS+(["family"] if family_view else [])
        for group,q in df.groupby(keys,dropna=False,sort=True):
            attrs=dict(zip(keys,group)); attrs.setdefault("family","all")
            for endpoint_name in ENDPOINTS:
                values=q[endpoint_name]
                ci=interval(values,q.group,q.family,key=str((attrs,endpoint_name)))
                rows.append(dict(**attrs,endpoint=endpoint_name,total=len(q),successes=int((values==1).sum()),
                    failures=int((values==0).sum()),unknown=int(values.isna().sum()),**ci))
    return pd.DataFrame(rows)

def paired_table(df):
    rows=[]; keys=["partition","suite","training_seed","repetition","condition","budget"]
    for group,q in df.groupby(keys,dropna=False,sort=True):
        attrs=dict(zip(keys,group))
        for a,b in itertools.combinations(sorted(q.method.unique()),2):
            qa=q[q.method==a]; qb=q[q.method==b]
            for ep in ("valid","continuity","economic_5","joint_5"):
                p=paired(qa,qb,ep)
                ci=interval(p[ep+"_a"]-p[ep+"_b"],p.group,p.family,key=str((attrs,a,b,ep)))
                rows.append(dict(**attrs,method_a=a,method_b=b,metric=ep,support="all known paired trials",**ci))
            acceptable=set(qa.loc[qa.joint_5==1,"key"])&set(qb.loc[qb.joint_5==1,"key"])
            for metric in ("cost","exit_s","ens_mwh","ens_share"):
                p=paired(qa[qa.key.isin(acceptable)],qb[qb.key.isin(acceptable)],metric)
                ci=interval(p[metric+"_a"]-p[metric+"_b"],p.group,p.family,key=str((attrs,a,b,metric)))
                rows.append(dict(**attrs,method_a=a,method_b=b,metric=metric,support="both joint-acceptable",**ci))
    return pd.DataFrame(rows)

def primary_inference(df):
    """25 predeclared tests; unavailable comparisons remain unavailable, not significant."""
    from scipy.stats import binomtest
    cfg=config(); rows=[]; q=df[(df.partition=="final")&(df.repetition==0)&(df.training_seed==0)]
    for a,b,suite in cfg["primary_inference"]["contrasts"]:
        for budget in cfg["extension_budgets"]:
            pa=q[(q.method==a)&(q.budget==budget)&(q.suite==("gurobi" if a=="gurobi" else suite))]
            pb=q[(q.method==b)&(q.budget==budget)&(q.suite==("four" if suite=="gurobi_four" else suite))]
            p=paired(pa,pb,"joint_5").dropna(subset=["joint_5_a","joint_5_b"])
            expected=cfg["new_cases_per_family"]*6; complete=len(p)==expected
            differences=p.joint_5_a-p.joint_5_b; wins=int((differences==1).sum()); losses=int((differences==-1).sum())
            raw=binomtest(wins,wins+losses,p=.5).pvalue if complete and wins+losses else 1. if complete else None
            ci=interval(differences,p.group,p.family,key=f"primary/{a}/{b}/{budget}")
            rows.append(dict(method_a=a,method_b=b,budget=budget,suite=suite,complete=complete,
                             wins=wins,losses=losses,p_raw=raw,**ci))
    adjusted=holm([r["p_raw"] if r["p_raw"] is not None else 1 for r in rows])
    for row,p in zip(rows,adjusted):
        row["p_holm"]=float(p) if row["complete"] else None
        row["significant"]=bool(row["complete"] and p<.05)
    return pd.DataFrame(rows)

def structural_table(df):
    rows=[]; keys=["partition","suite","training_seed","repetition","method","budget"]
    for group,q in df[df.partition.str.startswith("controlled")].groupby(keys,dropna=False):
        base=q[q.condition=="base"]
        for condition,p in q[q.condition!="base"].groupby("condition"):
            joined=p[["group","family","joint_5"]].merge(base[["group","joint_5"]],on="group",suffixes=("_variant","_base"),validate="one_to_one")
            ci=interval(joined.joint_5_variant-joined.joint_5_base,joined.group,joined.family,key=str((group,condition)))
            rows.append(dict(**dict(zip(keys,group)),condition=condition,**ci))
    return pd.DataFrame(rows)

def historical_rows():
    path=ROOT/config()["historical_rows"]; source=verify(read(path)); frozen_source=verify(read(path.parent/"freeze.json"))
    if source["freeze"]!=frozen_source["sha256"]: raise ValueError("Historical extraction provenance mismatch")
    if len(source["rows"])!=18480 or len(frozen_source["records"])!=18480:
        raise ValueError("Historical cohort incomplete")
    fields={"key","group","family","budget","repetition","feasible","observed","status","cost","ens_mwh",
            "reference_cost","reference_bound","reference_certified","exit_s","preload_s","compute_s","finalization_s",
            "demand_mwh","n_zones","n_steps","n_lines","repair_changed_fraction","repair_stage"}
    rows=[{k:v for k,v in r.items() if k in fields or k.startswith("stage_")} for r in source["rows"]]
    identity=[(r["key"],r.get("method"),r["budget"],r["repetition"]) for r in source["rows"]]
    if len(set(identity))!=18480: raise ValueError("Duplicate historical trials")
    for out,original in zip(rows,source["rows"]): out["method"]=original["method"]
    return rows,dict(file=str(path),sha256=sha(path),source_freeze=frozen_source["sha256"],
                     validation="Reused sealed physical/process-validated extraction; no solver rerun")

def write_analysis(run,rows,name,source):
    run=Path(run); df=frame(rows)
    fingerprint=digest(dict(rows=rows,source=source,analysis_sha=sha(__file__),config=config()))
    folder=run/"analyses"/(name+"_"+fingerprint[:16]); receipt=folder/"receipt.json"
    if receipt.exists():
        r=verify(read(receipt))
        for rel,h in r["outputs"].items():
            if sha(folder/rel)!=h: raise ValueError("Published analysis changed")
        return folder
    folder.mkdir(parents=True,exist_ok=True)
    tables=folder/"tables"; tables.mkdir(exist_ok=True)
    df.to_csv(tables/"observations.csv",index=False)
    primary=df[df.budget.isin(config()["extension_budgets"])]
    coverage=coverage_table(primary); coverage.to_csv(tables/"coverage.csv",index=False)
    paired_results=paired_table(primary); paired_results.to_csv(tables/"paired.csv",index=False)
    primary_inference(primary).to_csv(tables/"primary_inference.csv",index=False)
    structural_table(primary).to_csv(tables/"structural.csv",index=False)
    ci=primary[primary.reference_certified.fillna(False).astype(bool)]
    if not ci.empty: coverage_table(ci).to_csv(tables/"certified_coverage.csv",index=False)
    sensitivity=[]
    for group,q in primary.groupby(GROUPS,dropna=False):
        for column in (c for c in q if c.startswith("ens_") and "_cost_" in c):
            sensitivity.append(dict(**dict(zip(GROUPS,group)),endpoint=column,total=len(q),successes=int((q[column]==1).sum()),unknown=int(q[column].isna().sum())))
    pd.DataFrame(sensitivity).to_csv(tables/"ens_sensitivity.csv",index=False)
    timecols=[c for c in primary if c.startswith("stage_")]+["compute_s","finalization_s","preload_s","exit_s"]
    primary.groupby(GROUPS,dropna=False)[timecols].agg(["count","mean","median"]).to_csv(tables/"timing.csv")
    primary.groupby(GROUPS+["status"],dropna=False).size().rename("n").to_csv(tables/"failures.csv")
    primary[primary.feasible==True].groupby(GROUPS,dropna=False)[["cost","ens_mwh","ens_share","relative_gap","absolute_gap_eur"]].agg(["count","mean","median"]).to_csv(tables/"conditional_quality.csv")
    repaircols=[c for c in primary if c.startswith("repair_") and c!="repair_stage"]
    if repaircols: primary.groupby(GROUPS,dropna=False)[repaircols].agg(["count","mean","median"]).to_csv(tables/"repair.csv")
    render_figures(primary,coverage,folder)
    outputs={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob("*") if p.is_file()}
    put(receipt,sealed(dict(name=name,fingerprint=fingerprint,source=source,outputs=outputs,
                           interpretation="Historical analyses exploratory; confirmation only on the locked final cohort.")))
    return folder

def render_figures(df,coverage,folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    figures=Path(folder)/"figures"; figures.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout(); fig.savefig(figures/(name+".pdf"),bbox_inches="tight",metadata={"CreationDate":None})
        fig.savefig(figures/(name+".png"),dpi=160,bbox_inches="tight"); plt.close(fig)
    hist=coverage[(coverage.partition=="historical_development")&(coverage.repetition==0)&(coverage.family=="all")]
    if not hist.empty:
        fig,axes=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
        for ax,ep,title in zip(axes,["continuity","economic_5","joint_5"],["Service continuity","Economic quality (5%)","Joint requirement"]):
            for method in ("highs","rh","jepa","pipeline"):
                p=hist[(hist.endpoint==ep)&(hist.method==method)].sort_values("budget")
                ax.plot(p.budget,p.estimate,marker="o",label=METHOD_NAMES[method],color=COLORS[method])
                ax.fill_between(p.budget,p.lower,p.upper,alpha=.13,color=COLORS[method])
            ax.set(xscale="log",xticks=config()["extension_budgets"],xticklabels=config()["extension_budgets"],ylim=(0,1.02),title=title,xlabel="Independent deadline (s)")
        axes[0].set_ylabel("Coverage of all 600 scenarios"); axes[-1].legend(fontsize=8)
        save(fig,"coverage")
        q=df[(df.partition=="historical_development")&(df.repetition==0)&(df.budget==120)]
        fig,axes=plt.subplots(1,3,figsize=(11,3.4))
        for ax,method in zip(axes,("rh","jepa","pipeline")):
            a=q[(q.method==method)&(q.joint_5==1)]; b=q[(q.method=="highs")&(q.joint_5==1)]
            p=a[["key","exit_s","cost","reference_cost"]].merge(b[["key","exit_s","cost"]],on="key",suffixes=("_a","_b"))
            ax.scatter(p.exit_s_a-p.exit_s_b,(p.cost_a-p.cost_b)/p.reference_cost.abs().clip(lower=1)*100,s=12,alpha=.6,color=COLORS[method])
            ax.axvline(0,color="grey",lw=.7); ax.axhline(0,color="grey",lw=.7)
            ax.set(title=f"{METHOD_NAMES[method]} vs HiGHS; n={len(p)}",xlabel="Exit-time difference (s)")
        axes[0].set_ylabel("Cost difference / |reference| (%)"); save(fig,"paired_tradeoffs")
        fig,axes=plt.subplots(1,2,figsize=(9,3.3))
        names=["highs","rh","jepa","pipeline"]
        for ax,cols,title in zip(axes,[("compute_s","finalization_s"),("preload_s",)],
                                ["Measured execution (all terminal records)","Static preload, excluded from deadline"]):
            x=np.arange(4); width=.32
            for j,col in enumerate(cols):
                values=[q[q.method==m][col].median() for m in names]
                ax.bar(x+(j-(len(cols)-1)/2)*width,values,width,label=col.replace("_s","").replace("_"," "))
            ax.set(xticks=x,xticklabels=[METHOD_NAMES[n] for n in names],ylabel="Median elapsed seconds",title=title); ax.legend(fontsize=8)
        save(fig,"execution_time")
    ab=coverage[(coverage.suite=="ablation")&(coverage.repetition==0)&(coverage.family=="all")&(coverage.endpoint=="joint_5")]
    if not ab.empty and ab.estimate.notna().any():
        fig,axes=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
        for ax,partition in zip(axes,["development","final"]):
            for method in ("pipeline","perturb16"):
                p=ab[(ab.partition==partition)&(ab.method==method)].sort_values("budget")
                ax.plot(p.budget,p.estimate,"o-",label=METHOD_NAMES[method],color=COLORS[method]); ax.fill_between(p.budget,p.lower,p.upper,alpha=.15,color=COLORS[method])
            ax.set(title=partition,xlabel="Deadline (s)",ylim=(0,1)); ax.legend(fontsize=7)
        axes[0].set_ylabel("Joint coverage"); save(fig,"energy_ablation")
    gen=coverage[(coverage.partition=="controlled_test")&(coverage.repetition==0)&(coverage.family=="all")&(coverage.endpoint=="joint_5")&coverage.training_seed.isin([0,config()["training_seeds"][0]])]
    if not gen.empty and gen.estimate.notna().any():
        conditions=["base","size","lines","fleet","permutation","horizon48","horizon72"]
        methods=["highs","rh","jepa","pipeline","mh_jepa","mh_pipeline","mh_perturb16"]
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,budget in zip(axes,[30,120]):
            p=gen[gen.budget==budget].pivot(index="method",columns="condition",values="estimate").reindex(index=methods,columns=conditions)
            im=ax.imshow(p,aspect="auto",vmin=0,vmax=1,cmap="viridis"); ax.set(title=f"{budget} s",xticks=range(7),xticklabels=["Base","Size","Lines","Fleet","Perm.","48 h","72 h"],yticks=range(7),yticklabels=[METHOD_NAMES[m] for m in methods]); ax.tick_params(axis="x",rotation=35)
            for i in range(7):
                for j in range(7):
                    value=p.iloc[i,j]; ax.text(j,i,"N/A" if pd.isna(value) else f"{value:.0%}",ha="center",va="center",fontsize=7,color="black" if pd.isna(value) or value>.6 else "white")
        save(fig,"generalization")

def analyze_existing(run):
    rows,source=historical_rows()
    return write_analysis(run,rows,"historical",source)

def analyze_all(run):
    from .extract import extract_extension
    rows,source=historical_rows(); sources=[source]
    for partition in ("development","final","controlled_development","controlled_test"):
        for suite in (("four","ablation") if not partition.startswith("controlled") else ("four","multi")):
            rows.extend(extract_extension(run,partition,suite))
    for suite in ("four","ablation"): rows.extend(extract_extension(run,"final",suite,extra_repeats=True))
    for training_seed in config()["training_seeds"][1:]:
        rows.extend(extract_extension(run,"controlled_test","multi",training_seed=training_seed,sensitivity=True))
    if (Path(run)/"gurobi_environment.json").exists():
        from .cases import select
        for partition in ("gurobi120","final"):
            external=extract_extension(run,partition,"gurobi")
            rows.extend(external)
            keys={c["key"] for c in select(run,partition)}
            comparison_partition="historical_development" if partition=="gurobi120" else "final"
            matched=[dict(r,partition=partition,suite="external_comparison") for r in rows
                     if r.get("partition","historical_development")==comparison_partition
                     and r.get("suite","four")=="four" and r["key"] in keys
                     and r["repetition"]==0 and r["budget"] in config()["extension_budgets"]]
            rows.extend(matched)
            rows.extend(dict(r,suite="external_comparison") for r in external)
    return write_analysis(run,rows,"complete",dict(historical=source,freeze=sha(Path(run)/"freeze.json")))
