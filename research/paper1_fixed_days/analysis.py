"""Score fixed-grid days and compare shortlists at a common inspection quota."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research.paper1_revision.future_analysis import economic_threshold
from research.paper1_weights.data import config as loss_config, target
from .protocol import PROFILE_NAMES

BOOTSTRAP_REPLICATES = 1000


def top_k_weights(values, k):
    """Inclusion probabilities under uniform tie-breaking at the kth largest value."""
    values = np.asarray(values, dtype=float)
    n = values.shape[-1]
    if not 1 <= k <= n or not np.isfinite(values).all():
        raise ValueError("Ranking needs finite scores for all days and a feasible quota")
    cutoff = np.partition(values, n - k, axis=-1)[..., n - k:n - k + 1]
    higher, equal = values > cutoff, values == cutoff
    remaining = k - higher.sum(axis=-1, keepdims=True)
    return higher + equal * (remaining / equal.sum(axis=-1, keepdims=True))


def _collect(run_dir, manifest, cfg):
    from .runner import read_job
    from .design import RULE_DIRECTIONS

    observations, scores, predictions = [], [], []
    missing = []
    for day in manifest["days"]:
        day_id = day["day_id"]
        ref = read_job(run_dir, manifest, day_id, "reference", manifest["reference_seconds"])
        if ref is None:
            missing.append(f"{day_id}/reference")
        for pred in day["predictions"]:
            predictions.append(dict(day_id=day_id, **pred))
        for budget in manifest["deadlines"]:
            trial = read_job(run_dir, manifest, day_id, "trial", budget)
            if trial is None:
                missing.append(f"{day_id}/{budget}s")
            if trial is None or ref is None:
                continue
            obs = {key: trial[key] for key in (
                "day_id", "budget", "status", "feasible", "exit_s", "cost", "ens_mwh",
                "demand_mwh", "served_mwh", "thermal_mwh", "co2_direct_t",
                "co2_t_per_served_mwh", "imports_mwh", "exports_mwh",
            )}
            # Demand and screening measures are known before solving, even on a timeout.
            obs["demand_mwh"] = day["features"]["demand_mwh"]
            obs.update({key: day["features"][key] for key in RULE_DIRECTIONS})
            obs.update(day_seed=day.get("seed"), **day.get("regime", {}))
            ref_cost = ref["cost"] if ref["feasible"] else None
            obs.update(reference_cost=ref_cost, reference_status=ref["status"],
                       reference_certified=bool(ref["proven_optimal"]),
                       reference_bound=ref["bound"], reference_gap=ref["gap"],
                       cost_gap_pct=None, cost_excess_eur=None, E=None, ES=None)
            if not obs["feasible"]:
                obs.update(E=0, ES=0)
            elif ref_cost is None and obs["ens_mwh"] > cfg["zero_ens_tolerance_mwh"]:
                obs["ES"] = 0
            elif ref_cost is not None:
                diff = obs["cost"] - ref_cost
                e = int(obs["cost"] <= economic_threshold(ref_cost))
                obs.update(cost_gap_pct=100 * diff / max(abs(ref_cost), cfg["reference_floor_eur"]),
                           cost_excess_eur=max(0., diff),
                           E=e, ES=int(e and obs["ens_mwh"] <= cfg["zero_ens_tolerance_mwh"]))
            observations.append(obs)
            for policy in PROFILE_NAMES:
                score = {"score": None}
                if not obs["feasible"] or ref_cost is not None:
                    score = target(obs, cfg, cfg["policies"][policy])
                scores.append(dict(day_id=day_id, budget=budget, target=policy, **score))
    if missing:
        raise ValueError(f"Pilot unfinished: {len(missing)} missing job(s), including "
                         + ", ".join(missing[:5]) + ". Resume run before analyzing.")
    return pd.DataFrame(observations), pd.DataFrame(scores), pd.DataFrame(predictions)


def _prediction_errors(predictions, scores, obs):
    binary = obs.melt(id_vars=["day_id", "budget"], value_vars=["E", "ES"],
                      var_name="target", value_name="score")
    keys = ["day_id", "budget", "target"]
    actual = pd.concat([scores[keys + ["score"]], binary], ignore_index=True)
    if predictions.duplicated(keys + ["representation"]).any():
        raise ValueError("Duplicate frozen prediction")
    pairs = predictions.merge(actual, on=keys, how="left", validate="many_to_one")
    pairs["squared_error"] = (pairs["prediction"] - pd.to_numeric(pairs["score"])) ** 2
    return (pairs.groupby(["budget", "target", "representation"], sort=False)
            .agg(days=("day_id", "size"), known=("score", "count"),
                 mse_or_brier=("squared_error", "mean")).reset_index())


def _shortlists(manifest, obs, scores, predictions):
    from .design import RULE_DIRECTIONS

    days = manifest["days"]
    day_ids = [day["day_id"] for day in days]
    n, k = len(day_ids), manifest["quota"]
    rng = np.random.default_rng(20261119)
    draws = rng.integers(n, size=(BOOTSTRAP_REPLICATES, n))
    rules = {"uniform_random": np.zeros(n)}
    for key, orientation in RULE_DIRECTIONS.items():
        rules[key] = np.array([day["features"][key] for day in days]) * orientation
    shortlist, membership, dispersion = [], [], []
    for (budget, policy), group in scores.groupby(["budget", "target"], sort=False):
        y = group.set_index("day_id").loc[day_ids, "score"].to_numpy(float)
        known = y[np.isfinite(y)]
        row = dict(budget=budget, target=policy, days=n, known=len(known),
                   distinct_scores=len(np.unique(known)))
        if len(known):
            row.update(minimum=known.min(), median=np.median(known), maximum=known.max(),
                       std_population=known.std())
        dispersion.append(row)
        if len(known) != n:
            # Every draw is part of the sealed portfolio. Never rank a convenient subset.
            continue

        observed = obs[obs.budget == budget].set_index("day_id").loc[day_ids]
        relevant = top_k_weights(y, k)
        oracle_mean = np.dot(y, relevant) / k
        model_rows = predictions[(predictions.budget == budget) & (predictions.target == policy)]
        strategies = rules.copy()
        for rep, pred in model_rows.groupby("representation", sort=False):
            strategies[rep] = pred.set_index("day_id").loc[day_ids, "prediction"].to_numpy()
        for strategy, priority in strategies.items():
            chosen = top_k_weights(priority, k)
            missed = relevant * (1 - chosen)
            hits = np.dot(relevant, chosen)
            selected_mean = np.dot(y, chosen) / k
            # Paired day bootstrap, rebuilding the quota and all ties on every resample.
            draw_y = y[draws]
            draw_weights = top_k_weights(priority[draws], k)
            advantages = ((draw_weights - k / n) * draw_y).sum(axis=1) / k
            lo, hi = np.quantile(advantages, [0.025, 0.975])
            result = dict(budget=budget, target=policy, strategy=strategy, days=n, quota=k,
                          distinct_priority_values=len(np.unique(priority)),
                          expected_hits=hits, recall=hits / k, oracle_mean_loss=oracle_mean,
                          selected_mean_loss=selected_mean, mean_regret=oracle_mean - selected_mean,
                          gain_vs_random=selected_mean - y.mean(),
                          gain_bootstrap_lo=lo, gain_bootstrap_hi=hi,
                          expected_missed_nondeliveries=np.dot(missed, ~observed.feasible))
            for metric in ("ens_mwh", "cost_excess_eur", "co2_direct_t", "exit_s"):
                values = observed[metric].to_numpy(float)
                available = np.isfinite(values)
                coverage = missed[available].sum()
                result["missed_known_days_" + metric] = coverage
                result["missed_known_sum_" + metric] = (
                    np.dot(missed[available], values[available]) if coverage > 0 else None)
            shortlist.append(result)
            for i, day_id in enumerate(day_ids):
                membership.append(dict(day_id=day_id, budget=budget, target=policy,
                                       strategy=strategy, score=y[i], priority=priority[i],
                                       observed_top_weight=relevant[i],
                                       selected_weight=chosen[i], missed_weight=missed[i]))
    shortlist_table, membership_table = pd.DataFrame(shortlist), pd.DataFrame(membership)
    if shortlist_table.empty:
        shortlist_table = pd.DataFrame(columns=["budget", "target", "strategy", "days", "quota",
                                                "expected_hits", "recall", "mean_regret", "gain_vs_random"])
        membership_table = pd.DataFrame(columns=["day_id", "budget", "target", "strategy", "score",
                                                "priority", "observed_top_weight", "selected_weight", "missed_weight"])
    return pd.DataFrame(dispersion), shortlist_table, membership_table


def analyze(run_dir: Path, manifest: dict) -> dict:
    """Rebuild the decision report without fitting models or calling the solver."""
    run_dir = Path(run_dir)
    cfg = loss_config()
    obs, scores, pred = _collect(run_dir, manifest, cfg)
    errors = _prediction_errors(pred, scores, obs)
    dispersion, shortlist, membership = _shortlists(manifest, obs, scores, pred)
    n, k = len(manifest["days"]), manifest["quota"]
    refs = obs.drop_duplicates("day_id")
    incomplete = int((dispersion.known < n).sum())
    report = [
        "# Sélection de journées à système fixe", "",
        f"Protocole `{manifest['sha256']}`. Le pilote regroupe {n} journées synthétiques. "
        f"Le quota est de {k} journées par mandat et par échéance. "
        f"Les cinq profils évaluent les mêmes {len(obs)} essais.",
        f"Les références exploitables couvrent {refs.reference_cost.count()}/{n} journées, "
        f"dont {int(refs.reference_certified.sum())} certifiées optimales.", "",
        "| Échéance | Livraisons | Succès E / statuts connus | Succès ES / statuts connus |",
        "|---:|---:|---:|---:|",
    ]
    for budget, group in obs.groupby("budget"):
        report.append(f"| {budget} s | {group.feasible.sum()}/{n} | "
                      f"{int(group.E.sum())}/{group.E.count()} | "
                      f"{int(group.ES.sum())}/{group.ES.count()} |")
    report.extend([
        "", "Le succès E exige un programme livré à temps et respectant la tolérance de "
        "coût du papier. ES exige aussi au plus 0,001 MWh de demande non servie.",
        "", f"Le tirage uniforme retrouve en moyenne {k*k/n:.2f} des {k} journées prioritaires. "
        "Les ex æquo au seuil reçoivent des probabilités fractionnaires d’inclusion ; "
        "les départages observé et prédit sont indépendants. Une prévision identique "
        "pour toutes les journées donne les mêmes résultats moyens que le hasard.",
        "", "Le regret est la perte moyenne des journées prioritaires moins celle des "
        "journées sélectionnées. Un regret proche de zéro indique que la sélection "
        "capte une gravité proche de celle des meilleures sélections rétrospectives.",
        "", "## Gain de perte détectée par rapport au tirage uniforme", "",
        "Un gain positif correspond à des journées plus préoccupantes pour le mandat. "
        "Les écarts sont descriptifs et conditionnels au réseau, aux modèles et aux "
        "exécutions observés.", "",
    ])
    if not shortlist.empty:
        policies = {"equal4": "Équilibré", "service4_co2": "Service", "cost_priority": "Coût",
                    "time_priority": "Temps", "co2_priority": "Thermique"}
        strategies = {"uniform_random": "Tirage uniforme", "residual_peak_mw": "Pointe de charge résiduelle",
                      "upward_ramp_mw_per_hour": "Rampe résiduelle ascendante",
                      "capacity_margin_mw": "Faible marge à la pointe", "initial_storage_mwh": "Faible stock initial",
                      "realized_compact": "Trio structurel", "model_size": "Six dimensions",
                      "realized": "22 descripteurs réalisés"}
        for budget, group in shortlist.groupby("budget"):
            report.extend([f"### {budget} secondes", "", "| Règle | " + " | ".join(policies.values()) + " |",
                           "|---|" + "---:|" * len(policies)])
            for name, results in group.groupby("strategy", sort=False):
                by_target = dict(zip(results.target, results.gain_vs_random))
                vals = [f"{by_target[p]:.4f}" if p in by_target else "—" for p in policies]
                report.append("| " + strategies.get(name, name) + " | " + " | ".join(vals) + " |")
            report.append("")
    if incomplete:
        report.append(f"{incomplete} comparaison(s) échéance–mandat comportent une cible "
                      "inconnue. Le classement de ces portefeuilles reste indéterminé.")
    report.extend([
        "Les fichiers CSV donnent les observations en unités physiques, les scores par "
        "mandat, les erreurs de prévision, la dispersion des pertes et les probabilités "
        "d’inclusion par journée. Dans `shortlist.csv`, les sommes sur journées prioritaires "
        "manquées sont pondérées par la probabilité d’omission et accompagnées du nombre "
        "pondéré de journées avec une mesure connue. Aucune quantité physique n’est "
        "imputée à une absence de livraison. Les bornes bootstrap du gain sont des "
        f"quantiles descriptifs de {BOOTSTRAP_REPLICATES} rééchantillonnages de journées "
        "appariés entre stratégies, sous une hypothèse d’échangeabilité des journées "
        "synthétiques, à modèles fixés et sans correction de multiplicité.", "",
    ])
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    # Stage the complete export before replacing any previously successful report.
    with TemporaryDirectory(dir=output) as directory:
        staging = Path(directory)
        for name, frame in (("observations", obs), ("profile_scores", scores),
                            ("prediction_errors", errors), ("score_dispersion", dispersion),
                            ("shortlist", shortlist), ("shortlist_days", membership)):
            frame.to_csv(staging / (name + ".csv"), index=False)
        (staging / "summary.md").write_text("\n".join(report), encoding="utf-8")
        for path in sorted(staging.glob("*.csv")):
            os.replace(path, output / path.name)
        os.replace(staging / "summary.md", output / "summary.md")
    return dict(days=n, trials=len(obs), valid_references=int(refs.reference_cost.count()),
                unknown_portfolios=incomplete, report=str(output / "summary.md"))
