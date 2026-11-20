"""Input provenance and support checks, without solving or training."""
from dataclasses import asdict
from pathlib import Path
from .core import config, campaign, read, verify, sha, sealed, put, digest, frozen, ROOT
from .cases import load_data

def canonical(value):
    if isinstance(value,dict):
        return [[repr(k),canonical(v)] for k,v in sorted(value.items(),key=lambda kv:repr(kv[0]))]
    if isinstance(value,(list,tuple)): return [canonical(v) for v in value]
    if hasattr(value,"item"): return value.item()
    return value

def physical_digest(data):
    d=asdict(data)
    for key in ("scenario_id","scenario_name","source_path"): d.pop(key,None)
    return digest(canonical(d))

def input_audit(run):
    import numpy as np
    run=Path(run); snap=frozen(run)
    m=verify(read(campaign()/"manifest.json")); entries=verify(read(run/"cases.json"))["entries"]
    historical_ids={c["uuid"] for c in m["train"]+m["cohort"]}
    owners={}; physical={}; support=[]
    for case in entries:
        if case["partition"]=="gurobi120": continue
        if case["uuid"] in historical_ids: raise ValueError("Historical/new UUID overlap")
        if case["uuid"] in owners and owners[case["uuid"]]!=case["group"]:
            raise ValueError("Base UUID shared across partitions")
        owners[case["uuid"]]=case["group"]
        data=load_data(case["file"])
        if len(data.periods)!=case["expected_horizon"]: raise ValueError("Realized horizon differs")
        if case["condition"]=="base":
            signature=physical_digest(data)
            if signature in physical and physical[signature]!=case["group"]:
                raise ValueError("Duplicate physical base across partitions")
            physical[signature]=case["group"]
        support.append(dict(key=case["key"],group=case["group"],partition=case["partition"],
                            condition=case["condition"],zones=len(data.zones),steps=len(data.periods),lines=len(data.lines)))
    # Compare realized base inputs against ALL historical supervised/evaluation cases.
    for case in m["train"]+m["cohort"]:
        if physical_digest(load_data(case["file"])) in physical:
            raise ValueError("New input duplicates a historical physical scenario")
    dataset=verify(read(campaign()/"dataset.json")); shapes=[]
    for entry in dataset["entries"]:
        if sha(entry["file"])!=entry["sha256"]: raise ValueError("Historical label input changed")
        with np.load(entry["file"],allow_pickle=False) as arr:
            shapes.append(dict(key=entry["key"],partition=entry["partition"],shape=list(arr["u_star"].shape)))
    # Encoder audit records what is actually recoverable; missing pretraining membership
    # must never be rephrased as a proven historical no-leakage result.
    import pandas
    import torch
    hte=torch.load(snap["model_paths"]["hte"],map_location="cpu",weights_only=False)
    normalization={}
    for name in ("train_mean","train_std"):
        arr=hte[name].detach().cpu().numpy()
        if not np.isfinite(arr).all() or (name=="train_std" and (arr<=0).any()):
            raise ValueError("Invalid frozen normalization")
        normalization[name]=arr.tolist()
    membership={k:repr(v)[:5000] for k,v in hte.items() if any(s in k.lower() for s in ("split","file","dataset","train_path"))}
    contract=verify(read(snap["model_paths"]["hte_features"]))
    obj=sealed(dict(freeze=snap["sha256"],cases_sha256=sha(run/"cases.json"),support=support,
        historical_label_shapes=shapes,encoder_checkpoint_keys=sorted(hte),encoder_membership_metadata=membership,
        encoder_membership_status="metadata_recorded_not_complete_membership_proof",
        normalization=normalization,feature_contract=contract,
        limitations=["The new base groups are disjoint from historical supervised/evaluation cases by UUID and realized inputs.",
                     "Complete HTE pretraining membership is not certified by checkpoint metadata alone; historical transfer remains qualified."] ))
    put(run/"input_audit.json",obj)
    return obj
