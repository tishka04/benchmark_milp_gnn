"""Forecast from terminal timings; never solve, acquire the writer lock or save files.

The scientific runtime hashes top-level Python files and tests. Keep this
observer separate so upgrading a display does not invalidate an ongoing run.
Only timing metadata is used: success/failure does not determine completion.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import math
from pathlib import Path
from statistics import mean

from ..core import ROOT, FAMILIES, config, digest, read, verify, sha
from ..design import CONDITIONS, schedule


REFERENCE_STAGES = {
    "ReferencesDevelopment": ("development", "controlled_development"),
    "TrainingLabels": ("mh_training",),
    "ReferencesFinal": ("final", "controlled_test"),
}
STAGE_ORDER = ("ReferencesDevelopment", "Development", "TrainingLabels",
               "TrainMultiHorizon", "ValidateMultiHorizon", "ReferencesFinal", "Final", "Gurobi")


@dataclass
class Job:
    stage: str
    kind: str
    case: dict
    item: dict
    budget: float

    @property
    def key(self):
        if self.kind == "reference":
            return ("reference", self.case["key"])
        return ("trial", self.case["key"], self.item["method"], self.item["budget"],
                self.item["repetition"], self.item["suite"], self.item.get("training_seed"))

    @property
    def context(self):
        # Keep horizons, transformations, families, methods and deadlines apart.
        # The two independent cohorts share timing predictors, never observations.
        condition = self.case["condition"]
        condition = "base" if condition == "historical" else condition
        return (self.kind, self.case["family"], condition, self.case["expected_horizon"],
                self.item.get("method", "highs"), self.budget)


def planned_cases(cfg):
    """Countable placeholders before Prepare, with the same grouping as cases.py."""
    entries = []
    for partition, count in (("development", cfg["new_cases_per_family"]),
                             ("final", cfg["new_cases_per_family"]),
                             ("controlled_development", cfg["controlled_cases_per_family"]),
                             ("controlled_test", cfg["controlled_cases_per_family"]),
                             ("mh_training", cfg["training_bases_per_family"]), ("gurobi120", 20)):
        conditions = CONDITIONS if partition.startswith("controlled") else (
            ("base", "horizon12", "horizon48") if partition == "mh_training" else ("base",))
        for family in FAMILIES:
            for i in range(count):
                group = f"{partition}/{family}/{i:04d}"
                for condition in conditions:
                    entries.append(dict(key=f"{group}/{condition}", group=group, family=family,
                        partition=partition, condition=condition,
                        expected_horizon=int(condition[7:]) if condition.startswith("horizon") else 24))
    return entries


def plan(entries, cfg):
    groups = defaultdict(list)
    for case in entries:
        groups[case["partition"]].append(case)
    jobs = []
    for stage, partitions in REFERENCE_STAGES.items():
        for partition in partitions:
            jobs.extend(Job(stage, "reference", c, {}, cfg["reference_wall_cap_seconds"])
                        for c in groups[partition])

    def add(stage, partition, suite, **options):
        jobs.extend(Job(stage, "trial", c, item, item["budget"])
                    for c, item in schedule(groups[partition], suite, **options))

    for stage, independent, controlled in (("Development", "development", "controlled_development"),
                                           ("Final", "final", "controlled_test")):
        for suite in ("four", "ablation"):
            add(stage, independent, suite)
        add(stage, controlled, "four")
    add("ValidateMultiHorizon", "controlled_development", "multi")
    add("Final", "controlled_test", "multi")
    for suite in ("four", "ablation"):
        add("Final", "final", suite, extra_repeats=True)
    for training_seed in cfg["training_seeds"][1:]:
        add("Final", "controlled_test", "multi", training_seed=training_seed, sensitivity=True)
    for partition in ("gurobi120", "final"):
        add("Gurobi", partition, "gurobi")
    return {job.key: job for job in jobs}


def timing(obj):
    values = (obj.get("preload_s"), obj.get("exit_s"))
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
           for x in values) and sum(values) > 0:
        return values
    return None


class Timings:
    def __init__(self):
        self.current = defaultdict(list)
        self.historical = defaultdict(list)
        self.preloads = defaultdict(list)
        self.forecasts = {}

    def add(self, context, values, historical=False):
        self.forecasts.clear()
        target = self.historical if historical else self.current
        target[context].append(sum(values))
        self.preloads[(context[0], context[4])].append(values[0])

    def forecast(self, job):
        if job.context not in self.forecasts:
            self.forecasts[job.context] = self._forecast(job)
        return self.forecasts[job.context]

    def _forecast(self, job):
        for source, pools in (("current", self.current), ("historical", self.historical)):
            samples = pools.get(job.context, [])
            if len(samples) >= 3:
                return mean(samples), source
        # No unmeasured speed-up is assumed for a new horizon, transformation,
        # solver or neural architecture. Provision the deadline plus loading.
        method = job.item.get("method", "highs")
        candidates = [method]
        aliases = {"perturb16": "pipeline", "mh_jepa": "jepa",
                   "mh_pipeline": "pipeline", "mh_perturb16": "pipeline", "gurobi": "highs"}
        if method in aliases:
            candidates.append(aliases[method])
        for candidate in candidates:
            samples = self.preloads.get((job.kind, candidate), [])
            if samples:
                return job.budget + mean(samples), "provision"
        # The supervisor allows 300 s for reference and 600 s for trial preload.
        # These are explicitly provisions, not observed wall-time estimates.
        return job.budget + (300 if job.kind == "reference" else 600), "provision_no_preload"


def history_rows(run, path, historical_freeze):
    """Prefer the small, already published table, checking its full provenance.

    This avoids deserializing an 86 MB extraction while a campaign is running.
    The existing table is an input, not a cache created by this observer.
    """
    source_hash = None
    for receipt_path in sorted((run / "analyses").glob("historical_*/receipt.json")):
        try:
            receipt = verify(read(receipt_path))
            table = receipt_path.parent / "tables" / "observations.csv"
            if receipt["name"] != "historical" or receipt["source"]["source_freeze"] != historical_freeze["sha256"]:
                continue
            if Path(receipt["source"]["file"]).resolve() != path.resolve():
                continue
            if source_hash is None:
                source_hash = sha(path)
            if receipt["source"]["sha256"] != source_hash or sha(table) != receipt["outputs"]["tables/observations.csv"]:
                continue
            with table.open(encoding="utf-8-sig", newline="") as stream:
                rows = []
                for row in csv.DictReader(stream):
                    rows.append(dict(family=row["family"], method=row["method"],
                        observed=row["observed"].lower() in ("true", "1"),
                        n_steps=float(row["n_steps"] or "nan"), budget=float(row["budget"]),
                        preload_s=float(row["preload_s"] or "nan"), exit_s=float(row["exit_s"] or "nan")))
            return rows
        except (OSError, ValueError, KeyError, TypeError):
            continue
    obj = verify(read(path))
    if obj["freeze"] != historical_freeze["sha256"]:
        raise ValueError("Historical extraction provenance mismatch")
    return obj["rows"]


def load_history(run, cfg, snap, pools, warnings):
    path = ROOT / cfg["historical_rows"]
    if not path.exists():
        warnings.append("Historique de temps absent : provisions utilisees pour les essais non mesures.")
        return 0
    try:
        historical_freeze = verify(read(path.parent / "freeze.json"))
        if snap and historical_freeze["manifest_hash"] != snap["manifest_hash"]:
            raise ValueError("Historical timing provenance mismatch")
        n = 0
        for row in history_rows(run, path, historical_freeze):
            values = timing(row)
            if not row.get("observed") or not values or row.get("n_steps") != 24:
                continue
            if row.get("budget") not in cfg["extension_budgets"]:
                continue
            context = ("trial", row["family"], "base", 24, row["method"], row["budget"])
            pools.add(context, values, historical=True)
            n += 1
        return n
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"Historique de temps inutilisable ({type(exc).__name__}); provisions utilisees.")
        return 0


def terminal_records(run, jobs, snap, pools, warnings):
    done, measured, issues = {}, {}, Counter()
    for folder, kind in (("references", "reference"), ("experiments", "trial")):
        for path in (run / folder).glob("*.json"):
            try:
                obj = verify(read(path))
                identity = obj["identity"]
                if not snap or identity.get("freeze") != snap["sha256"]:
                    raise ValueError("different freeze")
                if kind == "reference":
                    key = (kind, identity["case"]["key"])
                else:
                    key = (kind, identity["key"], identity["method"], identity["budget"],
                           identity["repetition"], identity["suite"], identity.get("training_seed"))
                job = jobs[key]
                if kind == "reference":
                    if identity["case"] != job.case or path.stem != digest(job.case):
                        raise ValueError("different reference case")
                elif (identity.get("case_sha256") != job.case["scenario_sha256"] or
                      identity.get("protocol") != config()["protocol"] or path.stem != digest(identity)):
                    raise ValueError("different trial identity")
                if not obj.get("complete"):
                    issues["incomplete"] += 1
                    continue
                if key in done:
                    issues["duplicate"] += 1
                    continue
                done[key] = path.stat().st_mtime
                values = timing(obj.get("supervision", {}) if kind == "reference" else obj)
                if values:
                    pools.add(job.context, values)
                    measured[key] = sum(values)
                else:
                    issues["timing_missing"] += 1
            except (OSError, ValueError, KeyError, TypeError):
                issues["unreadable_or_foreign"] += 1
    for issue, count in issues.items():
        warnings.append(f"Enregistrements {issue}: {count} (aucun fichier modifie).")
    return done, measured


def training_row(run, cfg, snap, warnings):
    row = empty_row("TrainMultiHorizon", "models")
    pending = []
    durations = defaultdict(list)
    row["checkpoints_present"] = 0
    for training_seed in cfg["training_seeds"]:
        for part in cfg["multi_horizon"]["epochs"]:
            row["total"] += 1
            folder = run / "multi_horizon" / str(training_seed)
            receipt = folder / f"{part}_receipt.json"
            complete = False
            if receipt.exists():
                try:
                    obj = verify(read(receipt))
                    if not snap or obj["freeze"] != snap["sha256"] or not (folder / f"{part}_best.pt").exists():
                        raise ValueError("unmatched model receipt")
                    row["completed"] += 1
                    complete = True
                    value = obj.get("wall_seconds")
                    if isinstance(value, (float, int)) and math.isfinite(value) and value > 0:
                        durations[part].append(value)
                        row["measured_seconds"] += value
                except (OSError, ValueError, KeyError, TypeError):
                    warnings.append(f"Recu d'entrainement inutilisable : {training_seed}/{part}.")
            if not complete:
                # Checkpoints can be large, in-flight and contain pickle objects.
                # Never load them just to display an ETA; receipts supply timings.
                if (folder / f"{part}_last.pt").exists():
                    row["checkpoints_present"] += 1
                    row["unknown_remaining"] += 1
                else:
                    pending.append(part)
    for part in pending:
        if durations[part]:
            row["estimated_seconds"] += mean(durations[part])
            row["forecast_sources"]["training_other_seed"] += 1
        else:
            row["unknown_remaining"] += 1
    return row


def empty_row(stage, unit):
    return dict(stage=stage, unit=unit, total=0, completed=0, measured_seconds=0.,
                estimated_seconds=0., provision_seconds=0., remaining_allocation_seconds=0.,
                unknown_remaining=0, forecast_sources=Counter())


def build_report(run, *, hours_per_day=24, use_history=True):
    if not 0 < hours_per_day <= 24:
        raise ValueError("hours_per_day must be greater than zero and at most 24")
    run = Path(run)
    warnings = []
    snap = verify(read(run / "freeze.json")) if (run / "freeze.json").exists() else None
    cfg = snap["config"] if snap else config()
    if cfg != config():
        raise ValueError("Frozen configuration differs from current schedule; restore it before estimating.")
    if (run / "cases.json").exists():
        manifest = verify(read(run / "cases.json"))
        if not snap or manifest["freeze"] != snap["sha256"]:
            raise ValueError("Case manifest belongs to another freeze")
        entries = manifest["entries"]
    else:
        entries = planned_cases(cfg)
        warnings.append("Cohortes non preparees : effectifs issus du plan.")
    jobs = plan(entries, cfg)
    pools = Timings()
    done, measured = terminal_records(run, jobs, snap, pools, warnings)
    historical_samples = load_history(run, cfg, snap, pools, warnings) if use_history and len(done) < len(jobs) else 0
    rows = {stage: empty_row(stage, "references" if stage in REFERENCE_STAGES else "trials")
            for stage in STAGE_ORDER if stage != "TrainMultiHorizon"}
    for key, job in jobs.items():
        row = rows[job.stage]
        row["total"] += 1
        if key in done:
            row["completed"] += 1
            row["measured_seconds"] += measured.get(key, 0.)
        else:
            value, source = pools.forecast(job)
            row["remaining_allocation_seconds"] += job.budget
            row["forecast_sources"][source] += 1
            row["provision_seconds" if source.startswith("provision") else "estimated_seconds"] += value
    rows["TrainMultiHorizon"] = training_row(run, cfg, snap, warnings)
    ordered = [rows[name] for name in STAGE_ORDER]
    for row in ordered:
        row["remaining"] = row["total"] - row["completed"]
        row["remaining_seconds"] = row["estimated_seconds"] + row["provision_seconds"]
    unknown = []
    if not (run / "multi_horizon" / "dataset.json").exists():
        unknown.append("Preparation des embeddings apres les labels")
    if rows["TrainMultiHorizon"]["unknown_remaining"]:
        unknown.append(f"{rows['TrainMultiHorizon']['unknown_remaining']} entrainements sans duree restante mesurable")
    unknown.append("Controles entre essais, analyses, compilation et pauses")
    next_stage = next((r["stage"] for r in ordered if r["remaining"]), "Analyze / Manuscript")
    if not (run / "input_audit.json").exists():
        next_stage = "Prepare"
    elif next_stage == "TrainMultiHorizon" and not (run / "multi_horizon" / "dataset.json").exists():
        next_stage = "TrainingLabels (preparation des embeddings)"
    elif next_stage in ("ReferencesFinal", "Final", "Gurobi") and not (run / "design_lock.json").exists():
        next_stage = "Lock"
    next_block = None
    stage_jobs = [j for j in jobs.values() if j.stage == next_stage]
    first_missing = next((j for j in stage_jobs if j.key not in done), None)
    if first_missing:
        def block_key(job):
            return (job.case["partition"], job.item.get("suite", "references"),
                    bool(job.item.get("repetition", 0)), job.item.get("training_seed"))
        matching = [j for j in stage_jobs if block_key(j) == block_key(first_missing)]
        left = [j for j in matching if j.key not in done]
        forecasts = [pools.forecast(j) for j in left]
        next_block = dict(partition=first_missing.case["partition"],
            suite=first_missing.item.get("suite", "references"),
            completed=len(matching)-len(left), total=len(matching), remaining=len(left),
            remaining_seconds=sum(v for v, _ in forecasts),
            provision_seconds=sum(v for v, source in forecasts if source.startswith("provision")))
    remaining_seconds = sum(r["remaining_seconds"] for r in ordered)
    return dict(generated_at=datetime.now().astimezone().isoformat(timespec="seconds"), run=str(run),
        hours_per_day=hours_per_day, next_stage_to_complete=next_stage, next_block=next_block,
        final_test_locked=(run / "design_lock.json").exists(), stages=ordered,
        measured_seconds=sum(r["measured_seconds"] for r in ordered),
        estimated_seconds=sum(r["estimated_seconds"] for r in ordered),
        provision_seconds=sum(r["provision_seconds"] for r in ordered),
        remaining_seconds=remaining_seconds, equivalent_days=remaining_seconds / (3600 * hours_per_day),
        completed_reference_files=sum(k[0] == "reference" for k in done),
        completed_trial_files=sum(k[0] == "trial" for k in done),
        references=sum(j.kind == "reference" for j in jobs.values()),
        trials=sum(j.kind == "trial" for j in jobs.values()),
        historical_timing_samples=historical_samples, current_timing_samples=len(measured),
        last_result_at=datetime.fromtimestamp(max(done.values())).astimezone().isoformat(timespec="seconds") if done else None,
        not_estimated=unknown, warnings=warnings)


def duration(seconds):
    if seconds == 0:
        return "0 h"
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d}"


def render(report):
    lines = ["PAPIER 2 - AVANCEMENT ET TEMPS RESTANT", f"Instantane : {report['generated_at']}",
             f"Dossier : {report['run']}", "",
             f"References : {report['completed_reference_files']:,} / {report['references']:,} | "
             f"Essais : {report['completed_trial_files']:,} / {report['trials']:,}",
             f"Temps deja mesure : {duration(report['measured_seconds'])} (chargement + execution)",
             f"Prochaine etape a completer : {report['next_stage_to_complete']}"]
    if report["next_block"]:
        block = report["next_block"]
        lines += [f"Prochain sous-bloc incomplet : {block['partition']} / {block['suite']}",
                  f"  {block['completed']} / {block['total']} termines, {block['remaining']} restants : "
                  f"{duration(block['remaining_seconds'])} (dont {duration(block['provision_seconds'])} de provision)."]
    lines += ["",
             f"{'Etape':25} {'Termines / prevus':>18} {'%':>5} {'Reste*':>17} {'Dont provision':>17}",
             "-" * 87]
    for row in report["stages"]:
        count = f"{row['completed']:,} / {row['total']:,}"
        pct = 100 * row["completed"] / row["total"] if row["total"] else 100
        remaining = duration(row["remaining_seconds"])
        if row["unknown_remaining"]:
            remaining = (remaining + " + ?") if row["remaining_seconds"] else "non estime"
        lines.append(f"{row['stage']:25} {count:>18} {pct:4.0f}% {remaining:>17} {duration(row['provision_seconds']):>17}")
    lines += ["", f"* Reste a planifier : {duration(report['remaining_seconds'])}, PLUS les postes non chiffres ci-dessous.",
              f"  Equivaut a {report['equivalent_days']:.1f} jours a {report['hours_per_day']:g} h de calcul/jour, hors ces postes.",
              f"  Estimation par mesures comparables : {duration(report['estimated_seconds'])}.",
              f"  Provisions faute de mesures comparables : {duration(report['provision_seconds'])}.",
              "", "Mode de calcul :",
              f"  {report['current_timing_samples']:,} durees nouvelles et {report['historical_timing_samples']:,} durees historiques disponibles.",
              "  Moyenne par famille, transformation, horizon, methode et delai (au moins 3 mesures).",
              "  Priorite aux nouvelles mesures; historique utilise pour les cas de base a 24 h.",
              "  Sinon : delai alloue + chargement observe (ou plafond de chargement si inconnu).",
              "  Une provision n'est pas une prevision ni un plafond du temps total de la campagne.",
              "  Les echecs termines comptent comme faits; les essais interrompus restent a faire.",
              "", "Temps encore non chiffres :"]
    lines.extend(f"  - {item}" for item in report["not_estimated"])
    if report["last_result_at"]:
        lines += ["", f"Dernier resultat enregistre : {report['last_result_at']}"]
    lines.append("Lecture seule : utilisable pendant un calcul. Aucune date de fin garantie.")
    if report["warnings"]:
        lines += ["", "Points a verifier :"] + [f"  - {x}" for x in report["warnings"]]
    return "\n".join(lines)
