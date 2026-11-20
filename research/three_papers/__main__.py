"""CLI. Nothing is run during import; see GUIDE_COMMANDES.md before invocation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .core import HOME, config, environment, inventory, freeze, put, sealed, lock, read, sha, frozen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=HOME/"runs"/"v2_service_by_deadline")
    sub=p.add_subparsers(dest="command",required=True)
    pre=sub.add_parser("prerequisites"); pre.add_argument("--gurobi-license",action="store_true")
    for name in ("inventory","freeze","audit","extract","analyze","cases","oracle","lock-design","dataset"):
        sub.add_parser(name)
    for name in ("references","experiments","extract-extension","analyze-extension"):
        s=sub.add_parser(name); s.add_argument("--partition",required=True)
        if name!="references": s.add_argument("--suite",choices=["four","ablation","gurobi","multi","router","routing_baselines"],required=True)
    s=sub.add_parser("transfer"); s.add_argument("--partition",required=True); s.add_argument("--suite",required=True)
    s=sub.add_parser("train-router"); s.add_argument("--allow-flexible",action="store_true")
    s=sub.add_parser("train"); s.add_argument("--stage",choices=["ae","jepa","ebm"],required=True)
    s=sub.add_parser("combine"); s.add_argument("--partition",required=True); s.add_argument("--suites",nargs="+",required=True)
    sub.add_parser("match-gurobi")
    sub.add_parser("profiles")
    s=sub.add_parser("analyze-joint")
    s.add_argument("--partition")
    s.add_argument("--suite",choices=["four","ablation","gurobi","multi","router","routing_baselines","combined"])
    args=p.parse_args(); run=args.run.resolve(); cmd=args.command
    if cmd=="profiles":
        from .requirements import profiles, deadlines
        print(json.dumps(dict(deadlines_s=deadlines(), profiles=profiles()),indent=2)); return
    if cmd=="analyze-joint" and bool(args.partition)!=bool(args.suite):
        p.error("--partition and --suite must be supplied together")
    if cmd=="prerequisites":
        env=environment()
        import pandas
        import torch
        env.update(cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        if args.gurobi_license:
            import gurobipy as gp
            with gp.Env(params={"OutputFlag":0}): env["gurobi_license_initialization"]="passed"
        print(json.dumps(env,indent=2)); return
    with lock(run):
        if cmd=="inventory":
            result=inventory(); name=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            dest=run/"inventories"/(name+".json"); put(dest,sealed(result))
            print(json.dumps({k:v for k,v in result.items() if k not in ("records","state_file","issues")},indent=2))
            print(dest); return
        if cmd=="freeze":
            result=freeze(run); print("Frozen:",result["sha256"]); return
        if cmd=="cases":
            from .cases import make_cases
            make_cases(run)
        elif cmd=="audit":
            from .extract import audit_dimensions
            audit_dimensions(run)
        elif cmd=="extract":
            from .extract import extract_main
            extract_main(run)
        elif cmd=="analyze":
            from .statistics import analyze
            analyze(run)
        elif cmd=="analyze-joint":
            from .joint_analysis import analyze_joint
            source=run/f"rows_{args.partition}_{args.suite}.json" if args.partition else None
            name=f"{args.partition}_{args.suite}_joint" if args.partition else "main_joint"
            analyze_joint(run,source,name)
        elif cmd=="oracle":
            from .routing import diagnostic
            print(json.dumps(diagnostic(run),indent=2))
        elif cmd=="train-router":
            from .routing import train
            train(run,args.allow_flexible)
        elif cmd=="lock-design":
            from .experiments import lock_design
            lock_design(run)
        elif cmd=="references":
            from .experiments import references
            references(run,args.partition)
        elif cmd=="experiments":
            from .experiments import execute
            execute(run,args.partition,args.suite)
        elif cmd=="extract-extension":
            from .extract import extract_extension
            extract_extension(run,args.partition,args.suite)
        elif cmd=="analyze-extension":
            from .statistics import analyze
            analyze(run,run/f"rows_{args.partition}_{args.suite}.json",f"{args.partition}_{args.suite}",paper1=False)
        elif cmd=="transfer":
            from .statistics import transfer
            transfer(run,run/f"rows_{args.partition}_{args.suite}.json",f"transfer_{args.partition}_{args.suite}")
        elif cmd=="combine":
            from .core import verify, digest
            frozen(run); rows=[]; keys=set(); sources={}
            for suite in args.suites:
                path=run/f"rows_{args.partition}_{suite}.json"; obj=verify(read(path)); sources[str(path)]=sha(path)
                for row in obj["rows"]:
                    key=(row["key"],row["method"],row["budget"],row["repetition"],row.get("profile_id"))
                    if key in keys: raise ValueError("Duplicate observation across suites; keep ablation pipeline separate")
                    keys.add(key); rows.append(row)
            dest=run/f"rows_{args.partition}_combined.json"; put(dest,sealed(dict(rows=rows,sources=sources)))
            from .statistics import analyze
            analyze(run,dest,f"{args.partition}_combined",paper1=False)
        elif cmd=="match-gurobi":
            from .cases import select
            from .core import verify
            frozen(run)
            keys={c["key"] for c in select(run,"gurobi120")}
            data=verify(read(run/"main_rows.json"))
            rows=[dict(r,partition="gurobi120") for r in data["rows"] if r["key"] in keys
                  and r["budget"] in config()["extension_budgets"] and r["repetition"]==0]
            put(run/"rows_gurobi120_four.json",sealed(dict(rows=rows,source_sha256=sha(run/"main_rows.json"))))
        elif cmd=="dataset":
            from .multihorizon import prepare_dataset
            prepare_dataset(run)
        elif cmd=="train":
            from .multihorizon import train
            train(run,args.stage)

if __name__=="__main__": main()
