"""Requirement-aware, single-action routing. User-invoked fitting and evaluation only."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import math
import numpy as np
import pandas as pd
from .core import METHODS, config, seed, put, sealed, sha, frozen, writable, read, verify
from .requirements import (SCHEMA, profiles, profile, request_for, check_request,
                           request_features, profile_signature, finite, deadlines, request_key)
from .statistics import (frame, interval, folds, fit_probability, probability,
                         COMPONENTS, SIZE, output_directory, publish, metrics)
from .joint_analysis import expanded, CELL

SCENARIO = SIZE + COMPONENTS
PREFERENCES = ["budget", "cost_active", "threshold", "ceiling_active",
               "cost_ceiling", "ens_active", "ens_cap"]
FEATURES = SCENARIO + PREFERENCES
KINDS = ("highs", "highs_rh", "fixed", "family", "index", "logistic", "tree")

@lru_cache(maxsize=1)
def settings():
    return config()["service_routing"]

def training_table(df):
    x = expanded(df[(df.repetition == 0) & df.method.isin(METHODS)])
    if x.empty:
        raise ValueError("No development observations")
    for p in profiles():
        mask = x.profile_id == p["id"]
        for k, v in request_features(p, 1.).items():
            if k != "budget":
                x.loc[mask, k] = v
    for col in SCENARIO:
        if col not in x:
            x[col] = np.nan
    return x

def decision_cells(data):
    index = CELL
    base = data.drop_duplicates(index).set_index(index).reindex(columns=SCENARIO + ["criticality"])
    for m in METHODS:
        q = data[data.method == m].set_index(index)[["joint_success", "cost", "exit_s"]]
        q = q.rename(columns={"joint_success": m+"_y", "cost": m+"_c", "exit_s": m+"_t"})
        base = base.join(q)
    return base.reset_index()

def complete(cells):
    return cells.dropna(subset=[m+"_y" for m in METHODS]).copy()

def _stats(q):
    good = q[q.joint_success == 1]
    costs = good[np.isfinite(pd.to_numeric(good.cost, errors="coerce"))]
    times = good[np.isfinite(pd.to_numeric(good.exit_s, errors="coerce"))]
    return dict(p=float(q.joint_success.mean()) if len(q) else 0.,
                cost=float(costs.cost.mean()) if len(costs) else None,
                time=float(times.exit_s.mean()) if len(times) else None,
                groups=int(q.group.nunique()), cost_groups=int(costs.group.nunique()),
                time_groups=int(times.group.nunique()))

def _regression(q, target, kind):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    q = q[(q.joint_success == 1) & pd.to_numeric(q[target], errors="coerce").notna()]
    if q.empty:
        return None
    estimator = (Ridge(alpha=1.) if kind == "logistic" else
                 DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
                 if kind == "tree" else
                 HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
                                               l2_regularization=1., random_state=42))
    model = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                      ("scale", StandardScaler()), ("model", estimator)])
    model.fit(q[FEATURES], q[target].astype(float))
    return model

def _calibrated(q, kind):
    """Grouped OOF Platt calibration inside the outer training fold only."""
    from sklearn.linear_model import LogisticRegression
    if q.empty:
        return None, None, "no_known_labels"
    calibration = None
    state = "insufficient_grouped_calibration"
    split = q.group.map(lambda g: seed("router-calibration", g) % settings()["calibration_folds"])
    out = np.full(len(q), np.nan)
    for fold in range(settings()["calibration_folds"]):
        train, test = q[split != fold], q[split == fold]
        if train.empty or test.empty:
            continue
        fitted = fit_probability(train, train.joint_success, FEATURES, kind)
        out[np.asarray(split == fold)] = probability(fitted, test[FEATURES])
    known = np.isfinite(out)
    y = q.joint_success.to_numpy(dtype=int)
    if known.sum() >= 20 and len(set(y[known])) == 2 and np.std(out[known]) > 1e-8:
        logits = np.log(np.clip(out[known], 1e-6, 1-1e-6) /
                        (1-np.clip(out[known], 1e-6, 1-1e-6))).reshape(-1, 1)
        calibration = LogisticRegression(C=1., max_iter=1000).fit(logits, y[known])
        state = "grouped_oof_platt"
    elif len(set(y)) == 1:
        state = "constant_label_no_calibration_guarantee"
    return fit_probability(q, q.joint_success, FEATURES, kind), calibration, state

def fit_selector(data, kind):
    if kind not in KINDS + ("flexible",):
        raise ValueError(kind)
    model = dict(schema=SCHEMA, kind=kind, profile_signature=profile_signature(),
                 models={}, tables={}, support={},
                 feature_bounds={k: (float(data[k].min()), float(data[k].max()))
                                 for k in SCENARIO if data[k].notna().any()})
    for m in METHODS:
        q = data[(data.method == m) & data.joint_success.notna()].copy()
        for (pid, budget), part in q.groupby(["profile_id", "budget"]):
            model["tables"][(m, pid, float(budget), None)] = _stats(part)
            if kind == "family":
                for family, z in part.groupby("family"):
                    model["tables"][(m, pid, float(budget), str(family))] = _stats(z)
            if kind == "index":
                bins = (part.criticality * 5).clip(0, 4).astype(int)
                for group, z in part.groupby(bins):
                    model["tables"][(m, pid, float(budget), int(group))] = _stats(z)
        for pid, part in q.groupby("profile_id"):
            model["support"][(m, pid)] = dict(_stats(part),
                budgets=sorted(float(v) for v in part.budget.unique()))
        if kind in ("logistic", "tree", "flexible"):
            estimator, calibrator, status = _calibrated(q, kind)
            model["models"][m] = dict(probability=estimator, calibration=calibrator,
                calibration_status=status, cost=_regression(q, "cost", kind),
                time=_regression(q, "exit_s", kind))
    return model

def _support(model, method, pid, allowance, priority):
    sup = model["support"].get((method, pid), {})
    grid = sup.get("budgets", [])
    in_range = bool(grid and grid[0] <= allowance <= grid[-1])
    enough = (sup.get("groups", 0) >= settings()["minimum_training_groups"]
              and sup.get(priority+"_groups", 0) >= settings()["minimum_training_groups"])
    return dict(in_budget_range=in_range, supported=in_range and enough,
                interpolated_budget=bool(in_range and allowance not in grid),
                groups=sup.get("groups", 0), conditional_groups=sup.get(priority+"_groups", 0))

def _predict_many_raw(model, cells):
    """Vectorized numeric predictions; scenarios never expose references or labels."""
    if model.get("schema") != SCHEMA or model["profile_signature"] != profile_signature():
        raise ValueError("Router schema/profile mismatch; legacy models are not accepted")
    rows = cells.to_dict("records")
    requests = [request_for(r["profile_id"], float(r.get("deadline_s", r["budget"]))) for r in rows]
    features = pd.DataFrame([dict({k: r.get(k, np.nan) for k in SCENARIO},
                                  **request_features(p, float(r["budget"])))
                             for r, p in zip(rows, requests)]).reindex(columns=FEATURES)
    result = [{ } for _ in rows]
    for m in METHODS:
        pred = model["models"].get(m)
        if pred:
            probs = (probability(pred["probability"], features) if pred["probability"] is not None
                     else np.zeros(len(rows)))
            if pred["calibration"] is not None:
                logits = np.log(np.clip(probs, 1e-6, 1-1e-6) /
                                (1-np.clip(probs, 1e-6, 1-1e-6))).reshape(-1, 1)
                probs = pred["calibration"].predict_proba(logits)[:, 1]
            costs = pred["cost"].predict(features) if pred["cost"] is not None else [None]*len(rows)
            times = pred["time"].predict(features) if pred["time"] is not None else [None]*len(rows)
        for i, (r, p) in enumerate(zip(rows, requests)):
            allowance = float(r["budget"])
            sup = _support(model, m, p["id"], allowance, p["priority"])
            if pred:
                values = dict(p=float(probs[i]), cost=None if costs[i] is None else float(costs[i]),
                              time=None if times[i] is None else max(0., min(allowance, float(times[i]))),
                              calibration_status=pred["calibration_status"])
            else:
                grid = model["support"].get((m, p["id"]), {}).get("budgets", [])
                available = [b for b in grid if b <= allowance]
                budget = max(available) if available else None
                group = (str(r.get("family")) if model["kind"] == "family" else
                         min(4, max(0, int(float(r.get("criticality", 0))*5)))
                         if model["kind"] == "index" else None)
                default = model["tables"].get((m, p["id"], budget, None), {})
                values = dict(model["tables"].get((m, p["id"], budget, group), default))
                values = dict(p=values.get("p", 0.), cost=values.get("cost"), time=values.get("time"),
                              groups=values.get("groups", 0),
                              conditional_groups=values.get(p["priority"]+"_groups", 0),
                              calibration_status="empirical_development_rate")
                sup["supported"] = bool(sup["in_budget_range"] and
                    values["groups"] >= settings()["minimum_training_groups"] and
                    values["conditional_groups"] >= settings()["minimum_training_groups"])
            outside = [k for k, (lo, hi) in model["feature_bounds"].items()
                       if not finite(r.get(k)) or not lo <= float(r[k]) <= hi]
            sup["outside_feature_support"] = bool(outside)
            sup["feature_support_warnings"] = outside
            sup["supported"] = sup["supported"] and not outside
            result[i][m] = dict(sup, **values)
    return result

def _nested(stricter, broader):
    """Acceptance inclusion at one allocation; no monotonicity imposed across budgets."""
    def less_cap(a, b):
        return b is None or (a is not None and a <= b)
    if not less_cap(stricter["ens_cap_mwh"], broader["ens_cap_mwh"]):
        return False
    if broader["cost_tolerance"] is None and broader.get("cost_ceiling_eur") is None:
        return True
    if broader["cost_tolerance"] is not None:
        return less_cap(stricter["cost_tolerance"], broader["cost_tolerance"])
    return less_cap(stricter.get("cost_ceiling_eur"), broader.get("cost_ceiling_eur"))

def predict_many(model, cells):
    # Conservative lower envelope over the frozen requests enforces nested-event
    # probabilities after calibration. Its held-out calibration is reported.
    if cells.empty:
        return []
    definitions=profiles()
    original=cells.to_dict("records")
    expanded_cells=[]
    for row in original:
        for p in definitions:
            expanded_cells.append(dict(row, profile_id=p["id"]))
    raw=_predict_many_raw(model,pd.DataFrame(expanded_cells))
    answer=[]
    for i,row in enumerate(original):
        offset=i*len(definitions)
        index=next(j for j,p in enumerate(definitions) if p["id"]==row["profile_id"])
        requested=definitions[index]
        item={m:dict(raw[offset+index][m]) for m in METHODS}
        broader=[j for j,p in enumerate(definitions) if _nested(requested,p)]
        for m in METHODS:
            item[m]["p"]=min(raw[offset+j][m]["p"] for j in broader)
        answer.append(item)
    return answer

def select_prediction(predictions, request, forced=None, allowed=METHODS):
    priority = request["priority"]
    eligible = [m for m in allowed if predictions[m]["supported"] and
                predictions[m]["p"] >= 1-request["alpha"] and finite(predictions[m][priority])]
    fallback = not eligible
    if forced is not None:
        chosen = forced
        fallback = chosen not in eligible
    elif eligible:
        chosen = min(eligible, key=lambda m: (predictions[m][priority], -predictions[m]["p"], METHODS.index(m)))
    else:
        chosen = max(allowed, key=lambda m: (predictions[m]["p"], -METHODS.index(m)))
    p = predictions[chosen]
    return dict(method=chosen, risk_supported=bool(chosen in eligible),
                fallback=bool(fallback), predicted_success=p["p"],
                predicted_cost=p["cost"], predicted_time=p["time"],
                outside_budget_support=not p["in_budget_range"],
                outside_feature_support=p.get("outside_feature_support", False),
                feature_support_warnings=p.get("feature_support_warnings", []),
                interpolated_budget=p["interpolated_budget"],
                calibration_status=p["calibration_status"],
                eligible_methods=eligible, predictions=predictions)

def choose(model, features, allowance, request):
    check_request(request)
    if model.get("schema") != SCHEMA or model["profile_signature"] != profile_signature():
        raise ValueError("Unrecognized or stale router artifact")
    key = request_key(request["id"], request["deadline_s"])
    enabled = {request_key(r["profile_id"], r["deadline_s"]) for r in model["enabled_requests"]}
    if key not in enabled:
        raise ValueError("This profile/deadline pair did not pass the continuation gate")
    kind = model["selected"][key] if model["kind"] == "selected" else model["kind"]
    fitted = model["candidates"][kind]
    row = dict({k: features.get(k, np.nan) for k in SCENARIO},
               family=features.get("family"), criticality=features.get("criticality", 0.),
               profile_id=request["id"], budget=max(0., float(allowance)),
               deadline_s=request["deadline_s"])
    # Nonpositive allowance is flagged, not silently replaced by the full deadline.
    result = select_prediction(predict_many(fitted, pd.DataFrame([row]))[0], request,
                               forced="highs" if kind == "highs" else None,
                               allowed=("highs", "rh") if kind == "highs_rh" else METHODS)
    return dict(result, selector_kind=kind, prediction_allowance_s=max(0., float(allowance)))

def _evaluation(cells, model, kind):
    if cells.empty:
        return []
    predictions = predict_many(model, cells)
    records = []
    for row, pred in zip(cells.to_dict("records"), predictions):
        req = request_for(row["profile_id"], float(row["budget"]))
        decision = select_prediction(pred, req, forced="highs" if kind == "highs" else None,
                               allowed=("highs", "rh") if kind == "highs_rh" else METHODS)
        m = decision["method"]
        records.append(dict(group=row["group"], key=row["key"], profile_id=req["id"],
            family=row["family"], budget=row["budget"], selector=kind,
            choice=m, joint_success=float(row[m+"_y"]),
            cost=row[m+"_c"], exit_s=row[m+"_t"],
            predicted_success=decision["predicted_success"],
            risk_supported=decision["risk_supported"], fallback=decision["fallback"]))
    return records

def _paired_gain(q, comparator="fixed"):
    if q.empty:
        return []
    if q.profile_id.nunique() != 1 or q.budget.nunique() != 1:
        raise ValueError("Compare one profile and one deadline at a time")
    index = ["group", "key", "profile_id", "budget"]
    base = q[q.selector == comparator].set_index(index)
    out = []
    for kind, part in q.groupby("selector"):
        z = part.set_index(index).join(base[["joint_success", "cost", "exit_s"]],
                                      rsuffix="_fixed", how="inner")
        diff = z.joint_success - z.joint_success_fixed
        v, lo, hi, n = interval(diff.to_numpy(), z.index.get_level_values("group"))
        success, slo, shi, _ = interval(z.joint_success.to_numpy(), z.index.get_level_values("group"))
        pid = str(part.profile_id.iloc[0]); req = profile(pid)
        common = z[(z.joint_success == 1) & (z.joint_success_fixed == 1)].copy()
        if req["priority"] == "cost":
            gains = (common.cost_fixed-common.cost)/np.maximum(common.cost_fixed.abs(), 1.)
        else:
            gains = common.exit_s_fixed-common.exit_s
        gain, gl, gh, gn = interval(gains.to_numpy(), common.index.get_level_values("group"))
        out.append(dict(profile_id=pid, budget=float(part.budget.iloc[0]),
            selector=kind, success=success, success_lower=slo,
            success_upper=shi, coverage_gain=v, coverage_lower=lo, coverage_upper=hi,
            secondary_gain=gain, secondary_lower=gl, secondary_upper=gh,
            secondary_metric="relative_cost_reduction" if req["priority"] == "cost" else "seconds_saved",
            groups=n, paired_success_groups=gn, common_successes=len(common),
            fallback_count=int(part.fallback.sum()), **metrics(part.joint_success, part.predicted_success)))
    return out

def _continuation(req, expected, known, comparison):
    r = comparison
    enough = expected > 0 and known >= settings()["known_portfolio_fraction"]*expected
    enough = enough and r.get("groups", 0) >= settings()["minimum_paired_groups"]
    coverage_gate = bool(enough and r.get("coverage_gain") is not None and
        r["coverage_gain"] >= config()["oracle_minimum_gain"] and r["coverage_lower"] > 0)
    secondary_gate = bool(enough and r.get("secondary_gain") is not None and
        r["paired_success_groups"] >= settings()["minimum_paired_groups"] and
        r["coverage_lower"] >= -settings()["coverage_noninferiority_margin"] and
        r["success"] >= 1-req["alpha"] and r["secondary_lower"] > 0 and
        r["secondary_gain"] >= settings()["minimum_secondary_gain"][req["priority"]])
    return dict(expected=expected, complete_portfolios=known, excluded_missing=expected-known,
                coverage_gate=coverage_gate, secondary_gate=secondary_gate,
                proceed=coverage_gate or secondary_gate, comparison=r)

def _request_diagnostics(cells, known, evaluations):
    receipts, comparisons = [], []
    for req in profiles():
        pid = req["id"]
        for budget in deadlines():
            all_p = cells[(cells.profile_id == pid) & (cells.budget == budget)]
            known_p = known[(known.profile_id == pid) & (known.budget == budget)]
            part = evaluations[(evaluations.profile_id == pid) &
                               (evaluations.budget == budget)] if not evaluations.empty else evaluations
            reports = _paired_gain(part)
            r = next((v for v in reports if v["selector"] == "oracle"), {})
            if r:
                comparisons.append(r)
            receipts.append(dict(profile_id=pid, deadline_s=budget,
                request_key=request_key(pid, budget),
                **_continuation(req, len(all_p), len(known_p), r)))
    return receipts, comparisons

def _enabled_cells(cells, enabled_requests):
    allowed = {(r["profile_id"], float(r["deadline_s"])) for r in enabled_requests}
    mask = [(pid, float(b)) in allowed for pid, b in zip(cells.profile_id, cells.budget)]
    return cells.loc[mask].copy()

def _select_models(scores):
    selected = {}
    for (pid, budget), part in scores.groupby(["profile_id", "budget"]):
        req = profile(pid)
        candidates = part[(part.success >= 1-req["alpha"]) &
                          (part.coverage_lower >= -settings()["coverage_noninferiority_margin"]) &
                          part.secondary_gain.notna()]
        if len(candidates):
            winner = candidates.sort_values(["secondary_gain", "success", "selector"],
                                             ascending=[False, False, True]).iloc[0]
        else:
            winner = part.sort_values(["success", "selector"], ascending=[False, True]).iloc[0]
        selected[request_key(pid, budget)] = str(winner.selector)
    return selected

def _calibration_rows(evaluations):
    rows = []
    for (pid, budget, kind), q in evaluations.groupby(["profile_id", "budget", "selector"]):
        for b in range(10):
            z = q[(q.predicted_success >= b/10) &
                  (q.predicted_success < (b+1)/10 if b < 9 else q.predicted_success <= 1)]
            if len(z):
                v, lo, hi, n = interval(z.joint_success, z.group)
                rows.append(dict(profile_id=pid, budget=float(budget), selector=kind, bin=b,
                    predicted=float(z.predicted_success.mean()), observed=v, lower=lo, upper=hi, groups=n))
    return rows

def diagnostic(run):
    run = Path(run); snap = frozen(run)
    data = training_table(frame(run/"main_rows.json"))
    cells = decision_cells(data); known = complete(cells)
    evaluations = []
    split = folds(known)
    for fold in range(config()["folds"]):
        test = known[split == fold]
        if test.empty:
            continue
        groups = set(test.group)
        train_data = data[~data.group.isin(groups)]
        if train_data.empty:
            continue
        fixed = fit_selector(train_data, "fixed")
        ev = _evaluation(test, fixed, "fixed")
        evaluations.extend(ev)
        for row in test.to_dict("records"):
            req = profile(row["profile_id"])
            eligible = [m for m in METHODS if row[m+"_y"] == 1]
            objective = "_c" if req["priority"] == "cost" else "_t"
            ranked = [m for m in eligible if finite(row[m+objective])]
            chosen = min(ranked, key=lambda m: (row[m+objective], METHODS.index(m))) if ranked else METHODS[0]
            evaluations.append(dict(group=row["group"], key=row["key"], family=row["family"],
                profile_id=row["profile_id"], budget=row["budget"], selector="oracle", choice=chosen,
                joint_success=float(bool(eligible)), cost=row[chosen+"_c"] if ranked else None,
                exit_s=row[chosen+"_t"] if ranked else None, predicted_success=float(bool(eligible)),
                risk_supported=False, fallback=False))
    ev = pd.DataFrame(evaluations)
    receipts, by_budget = _request_diagnostics(cells, known, ev)
    enabled = [dict(profile_id=r["profile_id"], deadline_s=r["deadline_s"])
               for r in receipts if r["proceed"]]
    profile_summaries = [dict(profile=p,
        enabled_deadlines_s=[r["deadline_s"] for r in enabled if r["profile_id"] == p["id"]])
        for p in profiles()]
    receipt = sealed(dict(schema=SCHEMA, freeze=snap["sha256"], profile_signature=profile_signature(),
        input_sha256=sha(run/"main_rows.json"), deadlines_s=deadlines(),
        decision_unit="profile_id,deadline_s", profiles=profile_summaries,
        requests=receipts, by_budget=by_budget, enabled_requests=enabled,
        enabled_profiles=sorted({r["profile_id"] for r in enabled}), proceed=bool(enabled),
        decision="fit selectors for enabled profile/deadline pairs" if enabled else
                 "stop at profile/deadline complementarity diagnosis",
        warning="Retrospective oracle ignores routing overhead; no probabilistic guarantee. "
                "Pointwise intervals across declared requests; thresholds are design choices."))
    put(run/"routing"/"diagnostic.json", receipt)
    final, tmp = output_directory(run, "routing_diagnostic", run/"routing"/"diagnostic.json")
    if tmp is not None:
        ev.to_csv(tmp/"tables"/"oracle_cells.csv", index=False)
        pd.DataFrame(by_budget).to_csv(tmp/"tables"/"oracle_gain.csv", index=False)
        pd.json_normalize(receipts, sep="_").to_csv(tmp/"tables"/"continuation_by_request.csv", index=False)
        put(tmp/"tables"/"profiles.json", receipt)
        publish(final, tmp, run/"routing"/"diagnostic.json")
    return receipt

def train(run, allow_flexible=False):
    import joblib
    run = Path(run); snap = frozen(run)
    gate = verify(read(run/"routing"/"diagnostic.json"))
    if gate.get("schema") != SCHEMA or gate["profile_signature"] != profile_signature():
        raise ValueError("Recompute the service-aware diagnostic in this run")
    if gate["input_sha256"] != sha(run/"main_rows.json") or gate["freeze"] != snap["sha256"]:
        raise ValueError("Diagnostic provenance changed")
    if (run/"design_lock.json").exists():
        raise ValueError("Final design already locked")
    if not gate["proceed"]:
        print("No profile/deadline pair passed the continuation gate; no router trained.")
        return
    dest = run/"routing"/"selector.joblib"
    receipt_path = run/"routing"/"model_receipt.json"
    if receipt_path.exists():
        r = verify(read(receipt_path))
        if (r["allow_flexible"] != bool(allow_flexible) or r["profile_signature"] != profile_signature()
                or r["diagnostic_sha256"] != gate["sha256"]):
            raise ValueError("Training choices changed; use a new version")
        for p, h in r["artifacts"].items():
            if sha(p) != h:
                raise ValueError("Trained model artifact changed")
        return
    # Publish atomically after every model/receipt is ready; preserve interrupted attempts.
    attempt = writable(run/"routing_attempts"/__import__("uuid").uuid4().hex)
    attempt.mkdir(parents=True)
    data = training_table(frame(run/"main_rows.json"))
    # Fit all frozen acceptance profiles for coherent counterfactual probabilities.
    # Only gate-enabled profile/deadline pairs are scored or deployed.
    cells = _enabled_cells(complete(decision_cells(data)), gate["enabled_requests"])
    split = folds(cells)
    kinds = list(KINDS) + (["flexible"] if allow_flexible else [])
    evaluations = []
    for fold in range(config()["folds"]):
        test = cells[split == fold]
        if test.empty:
            continue
        train_data = data[~data.group.isin(set(test.group))]
        if train_data.empty:
            continue
        for kind in kinds:
            fitted = fit_selector(train_data, kind)
            evaluations.extend(dict(r, fold=fold) for r in _evaluation(test, fitted, kind))
    ev = pd.DataFrame(evaluations)
    if ev.empty:
        raise ValueError("Insufficient independent groups for router development")
    scores = pd.DataFrame([r for _, q in ev.groupby(["profile_id", "budget"]) for r in _paired_gain(q)])
    selected = _select_models(scores)
    expected_keys = {request_key(r["profile_id"], r["deadline_s"]) for r in gate["enabled_requests"]}
    if set(selected) != expected_keys:
        raise ValueError("Missing development scores for an enabled profile/deadline pair")
    fitted = {kind: fit_selector(data, kind) for kind in kinds}
    common = dict(schema=SCHEMA, profile_signature=profile_signature(),
                  deadlines_s=deadlines(), enabled_requests=gate["enabled_requests"],
                  enabled_profiles=gate["enabled_profiles"], selected=selected)
    joblib.dump(dict(common, kind="selected", candidates=fitted), attempt/"selector.joblib")
    for kind in KINDS:
        joblib.dump(dict(common, kind=kind, candidates={kind: fitted[kind]}), attempt/(kind+".joblib"))
    ev.to_csv(attempt/"development_predictions.csv", index=False)
    scores.to_csv(attempt/"development_scores.csv", index=False)
    pd.DataFrame(_calibration_rows(ev)).to_csv(attempt/"development_calibration.csv", index=False)
    output = run/"routing"; output.mkdir(parents=True, exist_ok=True)
    # A receipt is the publication marker. Unreceipted files are never silently overwritten.
    files = list(attempt.iterdir())
    if any((output/p.name).exists() for p in files):
        raise ValueError("Unreceipted router artifacts exist; preserve them and use a new run")
    for p in files:
        p.rename(output/p.name)
    artifacts = {str((output/p.name).resolve()): sha(output/p.name) for p in files}
    put(receipt_path, sealed(dict(schema=SCHEMA, freeze=snap["sha256"],
        input_sha256=sha(run/"main_rows.json"), profile_signature=profile_signature(),
        diagnostic_sha256=gate["sha256"], deadlines_s=deadlines(),
        enabled_requests=gate["enabled_requests"], enabled_profiles=gate["enabled_profiles"],
        selected=selected, allow_flexible=bool(allow_flexible),
        model_sha256=sha(dest), baseline_models={k: sha(output/(k+".joblib")) for k in KINDS},
        artifacts=artifacts, probability_calibration="inner grouped OOF Platt when support permits",
        warning="Development selection excludes overhead; probability threshold is not a risk guarantee.")))
