"""Explicit stages: no default scientific computation and no implicit full campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .core import HOME, config, freeze, frozen, lock, put, sealed, read, verify, environment
from .design import STAGES, planned_blocks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=HOME/"runs"/"methods_v1")
    parser.add_argument("--stage",choices=STAGES,required=True)
    args=parser.parse_args(); run=args.run.resolve(); stage=args.stage
    if stage=="Check":
        result=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s",str(HOME/"tests"),"-p","test_*.py","-v"],cwd=HOME.parents[1])
        if result.returncode: raise SystemExit(result.returncode)
        print(json.dumps(dict(environment=environment(),note="Synthetic CPU tests; GPU and Gurobi readiness checked when requested."),indent=2))
        return
    if stage=="Estimate":
        obj=planned_blocks()
        obj["completed_reference_files"]=len(list((run/"references").glob("*.json")))
        obj["completed_trial_files"]=len(list((run/"experiments").glob("*.json")))
        obj["remaining_references"]=obj["references"]-obj["completed_reference_files"]
        obj["remaining_trials"]=obj["trials"]-obj["completed_trial_files"]
        print(json.dumps(obj,indent=2)); return
    with lock(run):
        if stage=="Prepare":
            from .cases import make_cases
            from .audit import input_audit
            if (run/"freeze.json").exists(): frozen(run)
            else: freeze(run)
            make_cases(run); input_audit(run)
            put(run/"planned_counts.json",sealed(planned_blocks()))
        elif stage=="AnalyzeExisting":
            from .analysis import analyze_existing
            print(analyze_existing(run))
        elif stage in ("ReferencesDevelopment","ReferencesFinal","TrainingLabels"):
            from .experiments import references
            partitions={"ReferencesDevelopment":["development","controlled_development"],
                        "ReferencesFinal":["final","controlled_test"],"TrainingLabels":["mh_training"]}[stage]
            for partition in partitions: references(run,partition)
            if stage=="TrainingLabels":
                from .multihorizon import prepare_dataset
                prepare_dataset(run)
        elif stage in ("Development","ValidateMultiHorizon","Final","Gurobi"):
            from .experiments import execute, gurobi_preflight
            if stage=="Development":
                execute(run,"development","four"); execute(run,"development","ablation")
                execute(run,"controlled_development","four")
            elif stage=="ValidateMultiHorizon":
                execute(run,"controlled_development","multi")
            elif stage=="Final":
                execute(run,"final","four"); execute(run,"final","ablation")
                execute(run,"controlled_test","four"); execute(run,"controlled_test","multi")
                for suite in ("four","ablation"): execute(run,"final",suite,extra_repeats=True)
                for training_seed in config()["training_seeds"][1:]:
                    execute(run,"controlled_test","multi",training_seed=training_seed,sensitivity=True)
            else:
                gurobi_preflight(run)
                execute(run,"gurobi120","gurobi"); execute(run,"final","gurobi")
        elif stage=="TrainMultiHorizon":
            from .multihorizon import train
            for training_seed in config()["training_seeds"]:
                for part in ("ae","jepa","ebm"): train(run,part,training_seed)
        elif stage=="Lock":
            from .experiments import lock_design
            lock_design(run)
        elif stage=="Analyze":
            from .analysis import analyze_all
            print(analyze_all(run))
        elif stage=="Manuscript":
            from .manuscript import update
            print(update(run))
    print(f"Completed stage {stage}",flush=True)

if __name__=="__main__": main()
