"""Fresh variable-length AE/JEPA/EBM stack; never loads old UC-stack weights.

The frozen historical HTE and normalization are reused with a mathematically
extended sinusoidal time buffer. New UC models use content queries, not zone IDs.
Batch size one plus gradient accumulation handles variable Z and T without padding.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid
import pandas  # Initialize Arrow before Torch DLLs on Windows.
import numpy as np
import torch
from torch import nn
from .core import config, frozen, read, put, sealed, sha, digest, seed, writable, record, verify
from .cases import select, load_data

def position(n,d,device):
    p=torch.arange(n,device=device,dtype=torch.float32)[:,None]
    scale=torch.exp(torch.arange(0,d,2,device=device)*(-math.log(10000.)/d))
    result=torch.zeros(n,d,device=device)
    result[:,0::2]=torch.sin(p*scale); result[:,1::2]=torch.cos(p*scale)
    return result

def encoder(settings):
    from src.operational_exit.embedding import OnlineHTE
    hte=OnlineHTE(settings["hte"],feature_contract=settings["hte_features"],manifest_hash=settings["manifest_hash"])
    d=hte.model.pos_encoding.pe.shape[-1]
    hte.model.pos_encoding.pe=position(72,d,hte.device)[None]
    return hte

class Pool(nn.Module):
    def __init__(self,features,d,tokens,heads):
        super().__init__()
        self.gru=nn.GRU(features,d//2,batch_first=True,bidirectional=True)
        self.query=nn.Parameter(torch.randn(1,tokens,d)*.02)
        self.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        # x: [B,Z,T,F], here B=1 during training; no zone-index embedding.
        b,z,t,f=x.shape
        h,_=self.gru(x.reshape(b*z,t,f))
        zones=h.mean(dim=1).reshape(b,z,-1)
        out,_=self.attn(self.query.expand(b,-1,-1),zones,zones,need_weights=False)
        return self.norm(out)

class VariableAE(nn.Module):
    def __init__(self,h_dim,cfg):
        super().__init__(); d=cfg["latent_dim"]
        self.encode=Pool(8,d,cfg["tokens"],cfg["heads"])
        self.condition=nn.Linear(h_dim,d)
        self.attn=nn.MultiheadAttention(d,cfg["heads"],batch_first=True)
        self.norm=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,8))
    def decode(self,z,h):
        b,n,t,_=h.shape
        q=self.condition(h)+position(t,z.shape[-1],h.device)[None,None]
        q=q.flatten(1,2); a,_=self.attn(q,z,z,need_weights=False)
        return self.head(self.norm(q+a)).reshape(b,n,t,8)
    def forward(self,u,h):
        return self.decode(self.encode(u),h)

class Energy(nn.Module):
    def __init__(self,h_dim,cfg):
        super().__init__(); d=cfg["latent_dim"]
        self.net=nn.Sequential(nn.Linear(2*d+h_dim,d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,1))
    def forward(self,z,anchor,h):
        context=h.mean(dim=(1,2))[:,None].expand(-1,z.shape[1],-1)
        return self.net(torch.cat([z,anchor,context],dim=-1)).mean(dim=(1,2))

def prepare_dataset(run):
    from .worker import uc_from_dispatch
    run=Path(run); snap=frozen(run)
    settings=dict(snap["model_paths"],manifest_hash=snap["manifest_hash"])
    hte=encoder(settings); entries=[]; exclusions=[]
    for case in select(run,"mh_training"):
        ref_path=run/"references"/(digest(case)+".json")
        ref=record(ref_path)
        if not ref.get("complete"): raise ValueError("Training reference stage incomplete")
        if ref["identity"]["case"]!=case or ref["identity"]["freeze"]!=snap["sha256"]:
            raise ValueError("Training reference identity mismatch")
        result=ref["result"]
        if not result.get("valid"):
            exclusions.append(dict(key=case["key"],status=result.get("status"))); continue
        data=load_data(case["file"])
        from src.operational_exit.physics import validate_dispatch
        if not validate_dispatch(data,result["dispatch"])["valid"]: raise ValueError("Invalid label")
        if len(data.periods) not in config()["training_horizons"]: raise ValueError("Forbidden training horizon")
        file=writable(run/"multi_horizon"/"dataset"/(digest(case)[:20]+".npz"))
        file.parent.mkdir(parents=True,exist_ok=True)
        if not file.exists():
            h=hte.encode(data)[0].cpu().numpy(); u=uc_from_dispatch(result["dispatch"]).astype(np.float32)
            temp=file.with_name(file.name+"."+uuid.uuid4().hex[:8]+".tmp")
            with temp.open("xb") as stream:
                np.savez_compressed(stream,u=u,h=h,reference_sha=ref["sha256"],case_sha=case["scenario_sha256"])
                stream.flush(); os.fsync(stream.fileno())
            temp.rename(file)
        with np.load(file,allow_pickle=False) as arr:
            if str(arr["reference_sha"])!=ref["sha256"] or str(arr["case_sha"])!=case["scenario_sha256"]:
                raise ValueError("Dataset provenance mismatch")
        entries.append(dict(key=case["key"],group=case["group"],horizon=len(data.periods),
            split="validation" if seed("mh-split",case["group"])%5==0 else "train",
            file=str(file),sha256=sha(file),label_kind="certified" if result.get("proven_optimal") else "incumbent"))
    for split in ("train","validation"):
        if {e["horizon"] for e in entries if e["split"]==split}!={12,24,48}:
            raise ValueError("Each development split must contain 12, 24 and 48 hours")
    put(run/"multi_horizon"/"dataset.json",sealed(dict(freeze=snap["sha256"],entries=entries,exclusions=exclusions,
        label_policy="All independently valid reference incumbents; certification recorded separately",
        architecture=config()["multi_horizon"]["architecture"])))

def checkpoint(path,state):
    """Two-copy rolling training state, only under the new directory."""
    p=writable(path); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+"."+uuid.uuid4().hex[:8]+".tmp")
    with temp.open("xb") as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    if p.exists(): shutil.copy2(p,p.with_suffix(p.suffix+".bak"))
    os.replace(temp,p)

def load_models(run,h_dim,cfg,device):
    ae=VariableAE(h_dim,cfg).to(device)
    jepa=Pool(h_dim,cfg["latent_dim"],cfg["tokens"],cfg["heads"]).to(device)
    ebm=Energy(h_dim,cfg).to(device)
    return dict(ae=ae,jepa=jepa,ebm=ebm)

def train(run,stage):
    from src.operational_exit.embedding import require_gpu
    run=Path(run); snap=frozen(run); cfg=config()["multi_horizon"]
    if (run/"design_lock.json").exists(): raise ValueError("Final design is locked")
    dest=run/"multi_horizon"; dataset=verify(read(dest/"dataset.json"))
    if dataset["freeze"]!=snap["sha256"]: raise ValueError("Dataset freeze mismatch")
    if stage not in ("ae","jepa","ebm"): raise ValueError(stage)
    if (dest/(stage+"_receipt.json")).exists():
        receipt=verify(read(dest/(stage+"_receipt.json")))
        if receipt["model_sha256"]!=sha(dest/(stage+"_best.pt")): raise ValueError("Model changed")
        return
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]); torch.set_num_threads(1)
    device=require_gpu()
    train_entries=[e for e in dataset["entries"] if e["split"]=="train"]
    val_entries=[e for e in dataset["entries"] if e["split"]=="validation"]
    positives=np.zeros(8); total=0
    for e in dataset["entries"]:
        if sha(e["file"])!=e["sha256"]: raise ValueError("Dataset changed")
        with np.load(e["file"],allow_pickle=False) as arr:
            h_dim=arr["h"].shape[-1]
            if e["split"]=="train": positives+=arr["u"].sum(axis=(0,1)); total+=np.prod(arr["u"].shape[:2])
    weights=torch.tensor(np.clip((total-positives)/np.maximum(positives,1),1,20),dtype=torch.float32,device=device)
    models=load_models(run,h_dim,cfg,device); dependency_hashes={}
    for name in ("ae","jepa"):
        if (stage=="jepa" and name=="ae") or stage=="ebm":
            p=dest/(name+"_best.pt"); ck=torch.load(p,map_location=device,weights_only=False)
            if ck["freeze"]!=snap["sha256"] or ck["dataset"]!=dataset["sha256"]: raise ValueError("Dependency changed")
            models[name].load_state_dict(ck["state"]); models[name].eval()
            for v in models[name].parameters(): v.requires_grad_(False)
            dependency_hashes[name]=sha(p)
    model=models[stage]; optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    identity=digest(dict(freeze=snap["sha256"],dataset=dataset["sha256"],stage=stage,cfg=cfg,dependencies=dependency_hashes))
    epoch=cursor=steps=0; best=float("inf"); best_state=None; history=[]
    state_path=dest/(stage+"_last.pt")
    if state_path.exists():
        try:
            st=torch.load(state_path,map_location=device,weights_only=False)
        except Exception as exc:
            raise ValueError(f"Unreadable training state {state_path}; inspect its .bak before explicit recovery") from exc
        if st["identity"]!=identity: raise ValueError("Training resume identity mismatch")
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        epoch,cursor,steps=st["epoch"],st["cursor"],st["steps"]
        best,best_state,history=st["best"],st["best_state"],st["history"]
        torch.set_rng_state(st["torch_rng"].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in st["cuda_rng"]])
        np.random.set_state(st["numpy_rng"]); random.setstate(st["python_rng"])
    def save():
        checkpoint(state_path,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,cursor=cursor,steps=steps,best=best,best_state=best_state,history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate()))
    def loss(entry,validation=False):
        with np.load(entry["file"],allow_pickle=False) as arr:
            u=torch.tensor(arr["u"],device=device)[None]; h=torch.tensor(arr["h"],device=device)[None]
        bce=lambda logits:nn.functional.binary_cross_entropy_with_logits(logits,u,pos_weight=weights)
        if stage=="ae": return bce(models["ae"](u,h))
        with torch.no_grad(): target=models["ae"].encode(u)
        anchor=models["jepa"](h)
        if stage=="jepa":
            return nn.functional.mse_loss(anchor,target)+bce(models["ae"].decode(anchor,h))
        anchor=anchor.detach()
        gen=torch.Generator(device=device).manual_seed(seed("validation-negative",entry["key"])) if validation else None
        noise=torch.randn(anchor.shape,device=device,generator=gen)
        negative=anchor+cfg["initial_noise"]*noise
        # Contrastive energy learns to rank valid strategy latents against local proposals.
        ep=models["ebm"](target,anchor,h); en=models["ebm"](negative.detach(),anchor,h)
        return nn.functional.softplus(ep-en).mean()+1e-3*(ep.square()+en.square()).mean()
    while epoch<cfg["epochs"][stage]:
        model.train(); order=np.random.default_rng(seed(stage,epoch)).permutation(len(train_entries))
        while cursor<len(order):
            stop=min(len(order),cursor+cfg["accumulation"]); optimizer.zero_grad(set_to_none=True)
            for index in order[cursor:stop]: (loss(train_entries[index])/(stop-cursor)).backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"]); optimizer.step()
            cursor=stop; steps+=1; save()
        model.eval()
        with torch.no_grad(): validation=float(np.mean([loss(e,True).item() for e in val_entries]))
        if not math.isfinite(validation): raise ValueError("Non-finite validation loss")
        if validation<best:
            best=validation; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch,validation_loss=validation)); epoch+=1; cursor=0; save()
        print(stage,epoch,validation,flush=True)
    output=dest/(stage+"_best.pt")
    checkpoint(output,dict(state=best_state,freeze=snap["sha256"],dataset=dataset["sha256"],stage=stage,
                          config=cfg,h_dim=h_dim,architecture=cfg["architecture"],thresholds=[.5]*8))
    put(dest/(stage+"_receipt.json"),sealed(dict(freeze=snap["sha256"],dataset=dataset["sha256"],
        model_sha256=sha(output),history=history,dependencies=dependency_hashes,
        old_uc_weights_loaded=False,training_horizons=[12,24,48],test_horizon=72)))

class MultiGenerator:
    def __init__(self,settings):
        from src.operational_exit.embedding import require_gpu
        self.device=require_gpu(); self.hte=encoder(settings); self.method=settings["method"]
        ck=torch.load(settings["mh_ae"],map_location=self.device,weights_only=False)
        self.cfg=ck["config"]; self.models=load_models(None,ck["h_dim"],self.cfg,self.device)
        for name in ("ae","jepa","ebm"):
            state=torch.load(settings["mh_"+name],map_location=self.device,weights_only=False)
            if state["architecture"]!=self.cfg["architecture"] or state["dataset"]!=ck["dataset"]:
                raise ValueError("Multi-horizon stack mismatch")
            self.models[name].load_state_dict(state["state"]); self.models[name].eval()
            for p in self.models[name].parameters(): p.requires_grad_(False)
        torch.cuda.synchronize()
    def candidates(self,data,request,deadline):
        from src.operational_exit.common import seed_for
        start=time.perf_counter(); h=self.hte.encode(data); embedded=time.perf_counter()
        self.stage_seconds={"embedding":embedded-start}
        with torch.no_grad(): anchor=self.models["jepa"](h)
        k=16 if self.method=="mh_pipeline" else 1
        seeds=[seed_for(request["key"],request["repetition"],i) for i in range(k)]
        streams=[torch.Generator(device=self.device).manual_seed(s) for s in seeds]
        noise=lambda:torch.stack([torch.randn(anchor.shape[1:],generator=g,device=self.device) for g in streams])
        a=anchor.expand(k,-1,-1); hb=h.expand(k,-1,-1,-1); z=a.clone()
        if k>1:
            z=z+self.cfg["initial_noise"]*noise()
            for _ in range(self.cfg["langevin_steps"]):
                if time.perf_counter()>=deadline: raise TimeoutError("Multi-horizon generation deadline")
                z=z.detach().requires_grad_(True)
                energy=self.models["ebm"](z,a,hb)
                gradient=torch.autograd.grad(energy.sum(),z)[0].clamp(-1,1)
                z=(z-self.cfg["langevin_step"]*gradient+self.cfg["langevin_noise"]*noise()).detach().clamp(-10,10)
        with torch.no_grad():
            logits=self.models["ae"].decode(z,hb)
            if not torch.isfinite(logits).all(): raise ValueError("Nonfinite multi-horizon candidates")
            candidates=(torch.sigmoid(logits)>=.5).float().cpu().numpy()
        torch.cuda.synchronize(); self.stage_seconds["generation"]=time.perf_counter()-embedded
        return candidates,seeds
