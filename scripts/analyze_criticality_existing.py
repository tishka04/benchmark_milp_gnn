"""Statistical audit of existing results only: no solver or neural model imports.

One row per scenario. Regressions here are statistical diagnostics, not training
of the HTE / JEPA / EBM. All source experiment files remain read-only.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

from src.analysis.criticality_index import compute_criticality, NORM_BOUNDS

FAMILIES = ["low_criticality_scenarios", "medium_criticality_scenarios",
            "high_criticality_scenarios", "persona_vre_battery_scenarios",
            "persona_network_operator_scenarios", "persona_mathematician_scenarios"]
LABELS = dict(zip(FAMILIES, ["Low", "Medium", "High", "Persona VRE", "Persona réseau", "Persona math."]))
INDEX = ["criticality", "stress", "hardness"]
# Actual zone/line counts and generator estimates are deliberately distinguished.
SIZES = ["n_zones", "n_lines", "n_regions", "n_assets", "binary_est", "variables_est", "constraints_est", "n_timesteps"]
ENDPOINTS = {
    "unserved_log": "Énergie non servie / demande (référence)",
    "overgen_log": "Surproduction / demande (référence)",
    "runtime_log": "Temps MILP de référence, plafonné à 600 s",
    "ref_timeout": "Référence arrêtée à la limite de temps",
    "milp_unavailability": "Absence de solution HiGHS sur la grille de budgets",
    "pipeline_fail120": "Échec pipeline à 120 s",
    "jepa_fail120": "Échec JEPA à 120 s",
    "pipeline_bad10": "Pipeline sans solution à moins de 10 % de la référence",
    "jepa_bad10": "JEPA sans solution à moins de 10 % de la référence",
    "pipeline_gap_asinh": "Écart de coût pipeline à 120 s, si faisable",
    "jepa_gap_asinh": "Écart de coût JEPA à 120 s, si faisable",
}
BOUNDED = {"ref_timeout", "milp_unavailability", "pipeline_fail120", "jepa_fail120", "pipeline_bad10", "jepa_bad10"}
MODEL_ADDS = {"size": [], "size_crit": ["criticality"], "size_stress": ["stress"],
              "size_hard": ["hardness"], "size_both": ["stress", "hardness"],
              "size_hard_nosize": ["hardness_without_size"],
              "size_both_nosize": ["stress", "hardness_without_size"]}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def total(series):
    return float(sum(sum(v) for v in series.values()))


def persona_definitions():
    """Read literal configurations from AST; never import/run the generator."""
    path = Path("scripts/generate_persona_scenarios.py")
    def literal(node):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="_norm_weights":
            weights=literal(node.args[0]); den=sum(weights.values())
            return {k:v/den for k,v in weights.items()}
        if isinstance(node,ast.Dict):
            return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
        return ast.literal_eval(node)
    result={}
    for node in ast.parse(path.read_text("utf-8")).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript):
            target=node.targets[0]
            if isinstance(target.value,ast.Name) and target.value.id=="PERSONAS":
                value=literal(node.value)
                result[value["dir_name"]]=value
    assert len(result)==3
    return result


def extract(root, source, out):
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    raw = (source / "per_scenario.json").read_bytes()
    runs = json.loads(raw)
    assert len(runs) == 10800 and len(manifest["cohort"]) == 600
    assert all(r["completed"] for r in runs)
    lookup = {(r["family"], r["scenario_id"], r["method"], r["budget_seconds"]): r for r in runs}
    assert len(lookup) == 10800
    expected = manifest["inputs_sha256"]
    source_hashes = {str(source / "manifest.json"): sha(manifest_bytes), str(source / "per_scenario.json"): sha(raw)}
    rows = []
    personas=persona_definitions()
    for item in manifest["cohort"]:
        f, sid = item["family"], item["scenario_id"]
        sp = root / f / (sid + ".json")
        rp = root / f / "reports" / (sid + ".json")
        sraw, rraw = sp.read_bytes(), rp.read_bytes()
        for p, content in ((sp, sraw), (rp, rraw)):
            digest = sha(content)
            assert digest == expected[str(p.relative_to(root))], str(p)
            source_hashes[str(p)] = digest
        s, r = json.loads(sraw), json.loads(rraw)
        crit = compute_criticality(s)
        config=personas.get(f)
        original=compute_criticality(s,alpha=config["alpha"],stress_weights=config["stress_weights"],
                                    hardness_weights=config["hardness_weights"]) if config else crit
        d, mip = r["detail"], r["mip"]
        assert r["dr_formulation"] == "load_shifting_v1"
        assert r["solver_resources"]["highs_threads"] == 16
        provenance = r["batch_checkpoint"]["provenance"]
        assert provenance["scenario_sha256"] == sha(sraw)
        cap = provenance["time_limit_seconds"]
        dt = d["dt_hours"]
        demand, ens, spill = (total(d[k]) * dt for k in ("demand", "unserved", "overgen_spill"))
        assert demand > 0 and ens >= -.001 and spill >= -.001
        ens, spill = (max(v, 0) if v > .001 else 0. for v in (ens, spill))
        terms = list(crit.stress_normalized) + list(crit.hardness_normalized)
        row = dict(family=f, family_label=LABELS[f], scenario_id=sid, scenario_uuid=s["id"],
                   stored_criticality=s.get("criticality_index"), criticality=crit.criticality_index,
                   stress=crit.stress_index, hardness=crit.hardness_index,
                   stored_definition_recomputed=original.criticality_index,
                   n_zones=len(r["zones"]), n_lines=len(d["lines"]),
                   n_regions=s["meta"]["regions"], n_assets=sum(s["meta"]["assets"].values()),
                   n_timesteps=len(r["periods"]), binary_est=s["difficulty_indicators"]["n_binary_variables"],
                   variables_est=s["estimates"]["vars_total"], constraints_est=s["estimates"]["cons_total"],
                   reference_objective=mip["objective"], reference_status=mip["termination"],
                   ref_optimal=mip["termination"] == "optimal", ref_timeout=mip["termination"] == "maxTimeLimit",
                   ref_cap=cap, ref_seconds=mip["solve_seconds"],
                   runtime_log=np.log1p(min(mip["solve_seconds"], cap)),
                   demand_mwh=demand, unserved_mwh=ens, overgen_mwh=spill,
                   unserved_pct=100 * ens / demand, overgen_pct=100 * spill / demand,
                   unserved_log=np.log1p(100 * ens / demand), overgen_log=np.log1p(100 * spill / demand),
                   ref_zero_slack=ens+spill <= .001,
                   thermal_ramp_pct=s["tech"]["thermal_ramp_pct"],
                   n_thermal=s["meta"]["assets"]["thermal"],
                   thermal_mwh=total(d["thermal"])*dt,
                   thermal_max_mw=max(max(v) for v in d["thermal"].values()),
                   thermal_startup_blocked=(s["tech"]["thermal_ramp_pct"]<.45 and s["meta"]["assets"]["thermal"]>0))
        row.update(crit.stress_normalized)
        row.update(crit.hardness_normalized)
        row["hardness_without_size"] = np.mean([v for k,v in crit.hardness_normalized.items() if k not in ["B1_n_zones", "B2_horizon", "B3_binary_est"]])
        row.update({k + "_raw": v for k, v in (asdict(crit.stress_metrics) | asdict(crit.hardness_metrics)).items()})
        row["milp_unavailability"] = np.mean([not lookup[f, sid, "milp", b]["feasible"] for b in manifest["budgets"]])
        for method in ["pipeline", "jepa"]:
            a = lookup[f, sid, method, 120.0]
            assert a["reference_objective"] == mip["objective"]
            row[f"{method}_fail120"] = not a["feasible"]
            row[f"{method}_bad10"] = not (a["feasible"] and a["gap_reference_pct"] <= 10)
            row[f"{method}_gap_pct"] = a["gap_reference_pct"] if a["feasible"] else np.nan
            row[f"{method}_gap_asinh"] = np.arcsinh(a["gap_reference_pct"] / 100) if a["feasible"] else np.nan
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"Extraction et empreintes vérifiées : {len(rows)}/600", flush=True)
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["family", "scenario_id"]).any()
    assert frame["scenario_uuid"].nunique() == 600
    assert set(frame.ref_cap) == {600}
    frame.to_csv(out / "scenario_data.csv", index=False, encoding="utf-8-sig")
    index_audit = []
    for term in terms:
        v = frame[term]
        index_audit.append(dict(component=term, unique=v.nunique(), std=v.std(),
            minimum=v.min(), maximum=v.max(), n_zero=(v==0).sum(), n_one=(v==1).sum(),
            bounds=NORM_BOUNDS[term]))
    pd.DataFrame(index_audit).to_csv(out / "component_audit.csv", index=False, encoding="utf-8-sig")
    mismatch = frame.assign(index_delta=frame.criticality - frame.stored_criticality)
    mismatches = mismatch.groupby("family").agg(n=("scenario_id", "size"),
        n_mismatch=("index_delta", lambda v: (v.abs() > .00011).sum()),
        mean_abs_delta=("index_delta", lambda v: v.abs().mean()),
        max_abs_delta=("index_delta", lambda v: v.abs().max()))
    mismatches.to_csv(out / "stored_index_audit.csv", encoding="utf-8-sig")
    source_hashes[str(Path(__file__))] = sha(Path(__file__).read_bytes())
    index_path = Path("src/analysis/criticality_index.py")
    source_hashes[str(index_path)] = sha(index_path.read_bytes())
    for source_path in [Path("scripts/generate_persona_scenarios.py"),Path("src/milp/model.py"),Path("src/milp/scenario_loader.py"),Path("src/milp/defaults.py")]:
        source_hashes[str(source_path)]=sha(source_path.read_bytes())
    audit = dict(n_scenarios=len(frame), n_experiments=len(runs), n_sources_checked=1200,
                 reference_cap_seconds=600, benchmark_max_budget_seconds=120,
                 n_reference_optimal=int(frame.ref_optimal.sum()), n_reference_timeout=int(frame.ref_timeout.sum()),
                 stored_index_mismatches=mismatches.reset_index().to_dict("records"),
                 n_original_definitions_match=int((abs(frame.stored_definition_recomputed-frame.stored_criticality)<.00011).sum()),
                 persona_alphas={f:v["alpha"] for f,v in personas.items()},
                 stored_index_note="Les personas utilisent des poids spécifiques; les différences au score uniforme ne sont pas des corruptions.",
                 components=index_audit, input_sha256=source_hashes,
                 note="Indices recalculés avec le même code et alpha=0.5 pour les 600 scénarios. Aucun solveur ni modèle neuronal chargé.")
    save_json(out / "audit.json", audit)
    return frame, terms, audit


def size_basis(train, test=None, family=False, linear=False, core=False):
    """Fit all transformations only on train; flexible nonlinear size baseline."""
    test = train if test is None else test
    cols = ["n_zones", "binary_est"] if core else SIZES
    x = np.log1p(train[cols].to_numpy(float))
    z = np.log1p(test[cols].to_numpy(float))
    variable = x.std(axis=0) > 1e-10
    x, z = x[:, variable], z[:, variable]
    if not linear:
        spline = SplineTransformer(n_knots=3, degree=2, knots="quantile", include_bias=False,
                                   extrapolation="linear")
        x, z = spline.fit_transform(x), spline.transform(z)
    # Include the index's exact size-only encodings as well. Their contribution
    # cannot then be mistaken for information beyond size in aggregate indices.
    size_terms = ["B1_n_zones", "B2_horizon", "B3_binary_est"]
    x, z = np.column_stack([x, train[size_terms].to_numpy(float)]), np.column_stack([z, test[size_terms].to_numpy(float)])
    if family:
        xf = np.array([[int(f == v) for v in FAMILIES[1:]] for f in train.family])
        zf = np.array([[int(f == v) for v in FAMILIES[1:]] for f in test.family])
        x, z = np.column_stack([x, xf]), np.column_stack([z, zf])
    return x, z


def fit_predict(x, y, z, ridge=1.0):
    """Small fixed regularization, same for every diagnostic model; no tuning."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    keep = sd > 1e-9
    x, z = (x[:,keep]-mu[keep])/sd[keep], (z[:,keep]-mu[keep])/sd[keep]
    x, z = np.column_stack([np.ones(len(x)),x]), np.column_stack([np.ones(len(z)),z])
    penalty = np.eye(x.shape[1])*ridge
    penalty[0,0] = 0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return z @ beta


def residual(y, x):
    design = np.column_stack([np.ones(len(x)), x])
    return y - design @ np.linalg.lstsq(design, y, rcond=1e-10)[0]


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.clip(q,0,1)


def endpoint_support(data, endpoint, minimum_events=0):
    """Diagnose an uninformative response without inventing a zero association."""
    if endpoint not in data:
        return "missing response"
    y = data[endpoint].to_numpy(float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return "fewer than 10 finite observations"
    if np.std(y) < 1e-9:
        return "constant response"
    if minimum_events:
        if endpoint in {"unserved_log", "overgen_log", "ref_timeout"}:
            if np.count_nonzero(y > 0) < minimum_events:
                return f"fewer than {minimum_events} positive events"
        if endpoint == "ref_timeout" and np.count_nonzero(y == 0) < minimum_events:
            return f"fewer than {minimum_events} non-events"
    return None


def associations(frame, terms, out, *, endpoints=None, minimum_events=0, skipped=None):
    historical = endpoints is None
    endpoints = ENDPOINTS if historical else endpoints
    records = []
    for scope, data in [("all600", frame), ("reference_optimal", frame[frame.ref_optimal]),
                        ("stratified300", frame[frame.family.isin(FAMILIES[:3])]),
                        ("personas300", frame[frame.family.isin(FAMILIES[3:])])]:
        for endpoint in endpoints:
            d = data[np.isfinite(data[endpoint].astype(float))]
            if historical and endpoint == "overgen_log":
                continue  # Only one positive spill event; no meaningful inference.
            reason = endpoint_support(d, endpoint, minimum_events)
            if reason:
                if skipped is not None:
                    skipped.append(dict(stage="associations", scope=scope, endpoint=endpoint, reason=reason, n=len(d)))
                continue
            y = stats.rankdata(d[endpoint].astype(float))
            if np.std(y) < 1e-9:
                continue
            for adjustment in ["raw", "size", "size_family"]:
                design = None if adjustment == "raw" else size_basis(d, family=adjustment=="size_family")[0]
                if design is not None:
                    # Rank encodings remove pure size ranks exactly, preventing
                    # their nonlinear rescaling from looking like added value.
                    ranks = np.column_stack([stats.rankdata(d[v]) for v in SIZES + ["B1_n_zones", "B2_horizon", "B3_binary_est"]])
                    design = np.column_stack([design,ranks])
                ry = y if design is None else residual(y, design)
                df_res = len(d)-2 if design is None else len(d)-np.linalg.matrix_rank(np.column_stack([np.ones(len(d)),design]))-1
                if df_res <= 0 or np.std(ry) < 1e-8:
                    continue
                for feature in INDEX + ["hardness_without_size"] + terms:
                    v = d[feature].to_numpy(float)
                    if np.std(v) < 1e-9:
                        continue
                    rx = stats.rankdata(v)
                    if design is not None:
                        rx = residual(rx, design)
                    if np.std(rx) < 1e-8:
                        continue
                    rho = float(np.corrcoef(rx,ry)[0,1])
                    t = rho*np.sqrt(df_res/max(1-rho*rho,1e-12))
                    p = float(2*stats.t.sf(abs(t),df_res))
                    records.append(dict(scope=scope, endpoint=endpoint, adjustment=adjustment,
                                        feature=feature,n=len(d),rho=rho,p_approx=p))
    result = pd.DataFrame(records, columns=["scope", "endpoint", "adjustment", "feature", "n", "rho", "p_approx"])
    result["q_bh"] = np.nan
    # Explicit multiplicity family: all nonconstant features x all endpoints, per scope/adjustment.
    for _, ix in result.groupby(["scope","adjustment"]).groups.items():
        result.loc[ix,"q_bh"] = bh(result.loc[ix,"p_approx"])
    result.to_csv(out / "associations.csv", index=False, encoding="utf-8-sig")
    return result


def validation(frame, terms, out, repeats=10, *, endpoints=None, minimum_events=0, skipped=None):
    historical = endpoints is None
    requested_endpoints = ENDPOINTS if historical else endpoints
    records, predictions, increments = [], [], []
    variable_terms = [v for v in terms if frame[v].std() > 1e-9 and v not in ["B1_n_zones", "B2_horizon", "B3_binary_est"]]
    models = MODEL_ADDS | {"size_components": variable_terms}
    schemes = ["random_family_balanced", "random_with_family", "leave_one_family_out", "optimal_references", "size_core", "startup_allowed", "nonlinear_scores"]
    rng = np.random.default_rng(20261027)
    for scheme in schemes:
        scheme_endpoints = list(requested_endpoints) if scheme in schemes[:3] or scheme=="nonlinear_scores" else [e for e in ["unserved_log", "overgen_log", "runtime_log", "ref_timeout"] if e in requested_endpoints]
        pool = frame[frame.ref_optimal] if scheme=="optimal_references" else frame
        if scheme=="startup_allowed":
            pool=frame[~frame.thermal_startup_blocked]
        for endpoint in scheme_endpoints:
            if historical and endpoint == "overgen_log":
                continue
            d = pool[np.isfinite(pool[endpoint].astype(float))].reset_index(drop=True)
            reason = endpoint_support(d, endpoint, minimum_events)
            if not reason and (d.family.nunique() < 2 or d.family.value_counts().min() < 2):
                reason = "insufficient family support for held-out validation"
            if reason:
                if skipped is not None:
                    skipped.append(dict(stage="validation", scope=scheme, endpoint=endpoint, reason=reason, n=len(d)))
                continue
            y = d[endpoint].to_numpy(float)
            if y.std()<1e-9:
                continue
            n = len(d)
            # Average losses across repetitions, not apparent in-sample fit.
            n_rep = 1 if scheme=="leave_one_family_out" else repeats
            pred = {m:np.zeros((n_rep,n)) for m in models}
            null_pred = np.zeros((n_rep,n))
            for rep in range(n_rep):
                if scheme=="leave_one_family_out":
                    splits = [(np.where(d.family!=f)[0],np.where(d.family==f)[0]) for f in FAMILIES if (d.family==f).any()]
                else:
                    kfold=min(5,int(d.family.value_counts().min()))
                    assert kfold>=2
                    splitter = StratifiedKFold(n_splits=kfold,shuffle=True,random_state=3100+rep)
                    splits = list(splitter.split(d,d.family))
                tested = np.zeros(n,int)
                for ti, vi in splits:
                    tested[vi] += 1
                    tr, va = d.iloc[ti], d.iloc[vi]
                    xb, zb = size_basis(tr,va,family=scheme=="random_with_family",core=scheme=="size_core")
                    null_pred[rep,vi] = y[ti].mean()
                    for model, adds in models.items():
                        xa,za=tr[adds].to_numpy(float),va[adds].to_numpy(float)
                        if adds and scheme=="nonlinear_scores" and model!="size_components":
                            spl=SplineTransformer(n_knots=3,degree=2,knots="quantile",include_bias=False,extrapolation="linear")
                            xa,za=spl.fit_transform(xa),spl.transform(za)
                        x = np.column_stack([xb,xa]) if adds else xb
                        z = np.column_stack([zb,za]) if adds else zb
                        yp = fit_predict(x,y[ti],z)
                        if endpoint in BOUNDED:
                            yp = np.clip(yp,0,1)
                        elif endpoint in {"unserved_log", "overgen_log"}:
                            yp=np.maximum(yp,0)
                        elif endpoint=="runtime_log":
                            yp=np.clip(yp,0,np.log1p(600))
                        pred[model][rep,vi] = yp
                assert np.all(tested==1)
            variance = np.mean((y-y.mean())**2)
            losses = {m:np.mean((v-y[None,:])**2,axis=0) for m,v in pred.items()}
            r2 = {m:1-float(v.mean())/variance for m,v in losses.items()}
            null_mse = float(np.mean((null_pred-y[None,:])**2))
            # Paired scenario bootstrap, stratified by family, fixed OOF fits.
            fam_ix = [np.where(d.family==f)[0] for f in FAMILIES if (d.family==f).any()]
            boots = np.column_stack([rng.choice(ix,size=(1500,len(ix)),replace=True) for ix in fam_ix])
            boot_var = np.var(y[boots],axis=1)
            for model in models:
                gain = (losses["size"]-losses[model])
                interval = np.quantile(gain[boots].mean(axis=1)/np.maximum(boot_var,1e-12),[.025,.975])
                records.append(dict(scheme=scheme,endpoint=endpoint,model=model,n=n,
                                    r2_cv=r2[model],mse_cv=float(losses[model].mean()),
                                    null_mse_cv=null_mse,delta_r2=r2[model]-r2["size"],
                                    delta_low=interval[0],delta_high=interval[1],repeats=n_rep))
                for j in range(n):
                    predictions.append(dict(scheme=scheme,endpoint=endpoint,model=model,family=d.iloc[j].family,
                        scenario_id=d.iloc[j].scenario_id,observed=y[j],prediction=pred[model][:,j].mean(),
                        mse=float(losses[model][j])))
                if scheme=="leave_one_family_out":
                    for f in FAMILIES:
                        ix = np.where(d.family==f)[0]
                        if not len(ix):continue
                        vf = np.var(y[ix])
                        increments.append(dict(family=f,endpoint=endpoint,model=model,n=len(ix),
                            mse=float(losses[model][ix].mean()),size_mse=float(losses['size'][ix].mean()),
                            mse_reduction=float((losses['size'][ix]-losses[model][ix]).mean()),
                            within_family_r2=1-float(losses[model][ix].mean())/vf if vf>1e-12 else None))
            print(f"Validation statistique : {scheme} / {endpoint} / n={n}",flush=True)
    metrics = pd.DataFrame(records, columns=["scheme", "endpoint", "model", "n", "r2_cv", "mse_cv", "null_mse_cv", "delta_r2", "delta_low", "delta_high", "repeats"])
    metrics.to_csv(out / "validation.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(predictions, columns=["scheme", "endpoint", "model", "family", "scenario_id", "observed", "prediction", "mse"]).to_csv(out / "held_out_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(increments, columns=["family", "endpoint", "model", "n", "mse", "size_mse", "mse_reduction", "within_family_r2"]).to_csv(out / "leave_family_out.csv",index=False,encoding="utf-8-sig")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/load_shifting_v1/evaluation/budgets_v1/criticality_audit_20261027"))
    parser.add_argument("--reuse-extraction",action="store_true")
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    root=Path("outputs/load_shifting_v1")
    source=root/"evaluation/budgets_v1"
    if args.reuse_extraction:
        df=pd.read_csv(out/"scenario_data.csv")
        audit=json.loads((out/"audit.json").read_text("utf-8"))
        terms=[x["component"] for x in audit["components"]]
        audit["input_sha256"][str(Path(__file__))]=sha(Path(__file__).read_bytes())
        save_json(out/"audit.json",audit)
    else:
        df,terms,audit=extract(root,source,out)
    assoc=associations(df,terms,out)
    metrics=validation(df,terms,out)
    # No experimental file was changed; fingerprint the two input aggregates again.
    for p in [source/"manifest.json",source/"per_scenario.json"]:
        assert sha(p.read_bytes())==audit["input_sha256"][str(p)]
    summary=df.groupby("family").agg(n=("scenario_id","size"), zones_median=("n_zones","median"),
        stress_mean=("stress","mean"),hardness_mean=("hardness","mean"),criticality_mean=("criticality","mean"),
        optimal=("ref_optimal","sum"),timeouts=("ref_timeout","sum"),
        ens_median_pct=("unserved_pct","median"),ens_positive=("unserved_mwh",lambda x:(x>.001).sum()),
        spill_positive=("overgen_mwh",lambda x:(x>.001).sum()),runtime_median=("ref_seconds","median"),
        pipeline_failed=("pipeline_fail120","sum"),jepa_failed=("jepa_fail120","sum"))
    summary.to_csv(out/"family_summary.csv",encoding="utf-8-sig")
    ramp=df.groupby("thermal_startup_blocked").agg(n=("scenario_id","size"),
        positive_thermal=("thermal_max_mw",lambda v:(v>.001).sum()),ens_positive=("unserved_mwh",lambda v:(v>.001).sum()),
        ens_median_pct=("unserved_pct","median"),optimal=("ref_optimal","sum"),runtime_median=("ref_seconds","median"))
    ramp.to_csv(out/"thermal_startup_diagnostic.csv",encoding="utf-8-sig")
    save_json(out/"results.json",dict(audit={k:v for k,v in audit.items() if k!="input_sha256"},
        family_summary=summary.reset_index().to_dict("records"),validation=metrics.to_dict("records"),
        thermal_startup_diagnostic=ramp.reset_index().to_dict("records"),
        associations=assoc.to_dict("records"),methodology=dict(
            unit="scenario",n=600,alpha=.5,size_features=SIZES,
            size_basis="log1p, quadratic B splines, 3 training-fold quantile knots; exact B1/B2/B3 encodings also included; constant columns omitted",
            regression="ridge=1 fixed, standardized training-fold columns; intercept unpenalized; no hyperparameter tuning",
            response_bounds="all regressions use known output domains: binary/fraction [0,1]; ENS log >=0; runtime log [0,log(601)]",
            nonlinear_scores="separate sensitivity: quadratic splines for aggregate scores, learned within each fold; component regression unchanged",
            validation="5 folds stratified by family x 10 repeats; separate leave-one-family-out; all outcomes split by scenario",
            interval="1500 paired scenario bootstraps stratified by family, conditional on fixed held-out fits; not repeated campaigns",
            associations="Spearman ranks residualized on nonlinear size controls plus exact size ranks, with optional family fixed effects; t p-values approximate; BH per scope/adjustment",
            component_model="18 nonconstant non-size A/B features, fitted jointly; B1/B2/B3 already in baseline, B5 constant; exploratory regression distinct from the published equal-weight index",
            physical="Observed reference dispatch, not an unavoidable shortage certificate; optimal-only sensitivity",
            runtime="log1p(min(observed solve_seconds,600)), restricted computation duration, not uncensored time to optimality",
            costs="asinh(gap_reference_pct/100), only method-feasible cases; all-case 10% quality coverage separately",
            limits="Synthetic selected cohorts; ex-post exploratory assessment, no causal, real-grid, probability-of-failure or cross-solver claim")))
    print("Analyse terminée :",out,flush=True)


if __name__=="__main__":
    main()
