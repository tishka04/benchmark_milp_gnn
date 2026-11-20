"""Prepare all future physical jobs; execution is an explicit separate command."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
from .data import ROOT,HOME,RUN,CAMPAIGN,config,read,put,sha,digest

CAMP=RUN/'physical_campaign'


def source_hashes():
    paths=[HOME/'campaign.py',HOME/'timing_worker.py',ROOT/'src/milp/scenario_loader.py',
           ROOT/'src/milp/model.py',ROOT/'src/operational_exit/physics.py',ROOT/'src/generator/generator_v1.py',
           ROOT/'src/milp/profiles.py',ROOT/'src/milp/defaults.py',ROOT/'src/operational_exit/supervisor.py']
    paths+=list((ROOT/'src/milp').glob('*.py'))+[ROOT/'src/thermal.py',ROOT/'src/demand_response.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def prepare():
    from src.generator import generator_v1 as g
    from src.analysis.criticality_index import compute_criticality
    if any((CAMP/'results').glob('*.json')):
        raise ValueError('This campaign already has physical outcomes; do not overwrite its frozen preparation')
    cfg=config(); manifest=read(CAMPAIGN/'manifest.json'); space=g.load_space(str(ROOT/'config/scenario_space.yaml'))
    historical={c['uuid'] for c in manifest['cohort']+manifest['train']}
    entries=[]; jobs=[]
    for batch in ['unfiltered','time_structure','topology','decoupled']:
        for index in range(cfg['external_cases_per_batch']):
            seed=int(digest([cfg['seed'],'external',batch,index])[:8],16)%2147483647
            g.set_seed(seed);graph=g.sample_graph(space)
            horizon,dt=24,60
            if batch=='time_structure': horizon,dt=[(12,60),(48,60),(72,60),(24,30),(24,120)][index%5]
            obj=g.ScenarioConfig(id=str(uuid.uuid5(uuid.NAMESPACE_URL,f'paper1-revision/{cfg["seed"]}/{batch}/{index}')),
                horizon_hours=horizon,dt_minutes=dt,graph=graph,assets=g.sample_assets(space,graph),
                econ_policy=g.sample_econ(space),tech=g.sample_tech(space),costs=g.sample_operation_costs(space),
                exogenous=g.sample_exogenous(space,graph),unit_capacities=g.sample_unit_capacities(space),
                transport_capacities=g.sample_transport_capacities(space),mip_gap_target_pct=g.sample_mip_gap(space))
            payload=asdict(obj);payload['operation_costs']=payload.pop('costs')
            # Preserve the historical loader's default capacity convention for three batches.
            payload['_sampling_audit']=dict(unused_unit_capacities=payload.pop('unit_capacities'),
                                            unused_transport_capacities=payload.pop('transport_capacities'))
            nv,nc,_=g.estimate_milp_size(obj);meta=g.build_meta(obj,nv,nc,0.)
            payload.update(meta={k:v for k,v in meta.items() if k!='estimates'},estimates=meta['estimates'],
                flexibility_metrics=g.compute_flexibility_metrics(obj),difficulty_indicators=g.compute_difficulty_indicators(obj,nv,nc,0.))
            rng=np.random.default_rng(seed)
            transform={'batch':batch,'seed':seed}
            if batch=='topology': transform['edge_multiplier']=[1,1.5,2,3][index%4]
            if batch=='decoupled':
                transform.update(storage_power_factor=float(rng.uniform(.5,2)),storage_energy_factor=float(rng.uniform(.5,2)),
                                 demand_shape_amplitude=float(rng.uniform(-.35,.35)))
            payload['_paper1_revision']=transform
            if obj.id in historical: raise ValueError('External UUID collision')
            path=CAMP/'cases'/batch/f'{index:04d}.json'
            if path.exists() and digest(read(path))!=digest(payload): raise ValueError('Frozen external case changed: '+str(path))
            put(path,payload)
            crit=compute_criticality(payload)
            entries.append(dict(group=f'external/{batch}/{index:04d}',batch=batch,file=str(path),scenario_sha256=sha(path),
                uuid=obj.id,horizon_hours=horizon,dt_minutes=dt,seed=seed,
                descriptor_version='historical_v1_generator_proxy; transform deltas recorded separately',
                components={**crit.stress_normalized,**crit.hardness_normalized},transform=transform))
    # Reference jobs preserve old incumbents; no current result is overwritten.
    refs=pd.read_parquet(RUN/'references.parquet').set_index('group')
    for c in manifest['cohort']:
        if not refs.loc[c['key'],'certified']:
            jobs.append(dict(kind='reference_upgrade',key=c['key'],file=c['file'],scenario_sha256=c['scenario_sha256'],
                seed=int(digest(['reference-upgrade',c['key'],cfg['seed']])[:8],16)%2147483647,
                budget=cfg['reference_wall_cap_seconds'],solver_seconds=cfg['reference_solver_seconds'],repetition=0))
    for c in entries:
        common={k:c[k] for k in ['file','scenario_sha256','seed']};common['key']=c['group']
        jobs.append(dict(common,kind='external_reference',budget=2100,solver_seconds=1800,repetition=0))
        for b in cfg['deadlines']:
            jobs.append(dict(common,kind='external_trial',budget=b,solver_seconds=b,repetition=0))
    historic=pd.read_parquet(RUN/'analysis.parquet').drop_duplicates('group')
    selected=[]
    for family,q in historic.groupby('family'):
        q=q.sort_values('n_zones');positions=np.linspace(0,len(q)-1,5,dtype=int)
        selected.extend(q.iloc[positions].group.tolist())
    lookup={c['key']:c for c in manifest['cohort']}
    for key in selected:
        c=lookup[key]
        for repetition in range(cfg['timing_repetitions']):
            for b in cfg['deadlines']:
                jobs.append(dict(kind='timing_repeat',key=key,file=c['file'],scenario_sha256=c['scenario_sha256'],budget=b,
                    solver_seconds=b,repetition=repetition,
                    seed=int(digest(['timing-repeat',key,repetition,cfg['seed']])[:8],16)%2147483647))
    # Random order is committed before any new outcome exists.
    jobs.sort(key=lambda j:digest([cfg['seed'],'physical-order',j]))
    for j in jobs: j['job_id']=digest(j)[:24]
    receipt=dict(config=cfg,entries=entries,jobs=jobs,source_hashes=source_hashes(),
        scenario_space_sha256=sha(ROOT/'config/scenario_space.yaml'),analysis_manifest_sha256=sha(RUN/'manifest.json'),
        status='prepared_not_executed',user_instruction='Prepare complete campaign without long solves',
        maximum_reserved_hours=sum(j['budget'] for j in jobs)/3600,
        counts=pd.Series([j['kind'] for j in jobs]).value_counts().to_dict(),
        precision='100 independent cases per batch; maximum normal 95% proportion half-width about 0.098, not a power guarantee',
        limitations=['same generator lineage, independent draws and selected physical shifts, not an external system',
                    'B2 not learned on historical training; topology and decoupled shifts may not be represented by historical proxies',
                    'no second solver or PGLib-UC adapter included in the primary protocol'])
    put(CAMP/'manifest.json',receipt)
    put(CAMP/'status.json',dict(status='prepared_not_executed',executed_jobs=0,counts=receipt['counts']))
    print('Prepared physical campaign',receipt['counts'],'reserved hours',receipt['maximum_reserved_hours'],flush=True)


def load_physics(path):
    from src.milp.scenario_loader import load_scenario_data,NetworkLine
    data=load_scenario_data(Path(path));raw=read(path);t=raw.get('_paper1_revision',{})
    if t.get('batch')=='topology':
        rng=np.random.default_rng(t['seed']);order=rng.permutation(data.zones).tolist()
        pairs={tuple(sorted((order[i],order[i+1]))) for i in range(len(order)-1)}
        extra=[(a,b) for i,a in enumerate(sorted(data.zones)) for b in sorted(data.zones)[i+1:] if (a,b) not in pairs]
        rng.shuffle(extra);target=int(round(t['edge_multiplier']*(len(order)-1)))
        pairs.update(extra[:max(0,target-len(pairs))]);old=list(data.lines.values())
        data.lines={f'R{i}':NetworkLine(f'R{i}',a,b,old[i%len(old)].capacity_mw,old[i%len(old)].distance_km) for i,(a,b) in enumerate(sorted(pairs))}
        data.lines_from_index={z:tuple(k for k,l in data.lines.items() if l.from_zone==z) for z in data.zones}
        data.lines_to_index={z:tuple(k for k,l in data.lines.items() if l.to_zone==z) for z in data.zones}
    if t.get('batch')=='decoupled':
        for name in ['battery_power','pumped_power']:
            setattr(data,name,{k:v*t['storage_power_factor'] for k,v in getattr(data,name).items()})
        for name in ['battery_energy','battery_initial','battery_final_min','battery_final_max','pumped_energy','pumped_initial','pumped_final_min','pumped_final_max']:
            setattr(data,name,{k:v*t['storage_energy_factor'] for k,v in getattr(data,name).items()})
        for z in data.zones:
            old=np.array([data.demand[z,p] for p in data.periods]);shape=1+t['demand_shape_amplitude']*np.sin(2*np.pi*np.arange(len(old))/len(old))
            new=old*shape;new*=old.sum()/new.sum()
            for p,v in zip(data.periods,new): data.demand[z,p]=float(v)
            data.peak_demand[z]=float(new.max())
    return data


def execute(kind,max_hours):
    """Only invoked by an explicit launch command, never by `all` or prepare."""
    from src.operational_exit.supervisor import run_experiment
    manifest=read(CAMP/'manifest.json')
    if manifest['source_hashes']!=source_hashes(): raise ValueError('Campaign code changed; prepare a new version before solving')
    freeze=CAMP/'predictors_frozen.json'
    if kind in ('all','external_trial') and not freeze.exists():
        raise ValueError('Freeze the complete predictive design before opening independent test trials')
    if kind in ('all','external_trial'):
        frozen=read(freeze)
        if frozen['training_manifest']!=sha(RUN/'manifest.json') or any(sha(HOME/name)!=h for name,h in frozen['source_hashes'].items()):
            raise ValueError('Predictive design changed after the independent-test freeze')
    start=time.perf_counter(); completed=0
    for job in manifest['jobs']:
        if kind!='all' and job['kind']!=kind: continue
        result_path=CAMP/'results'/(job['job_id']+'.json')
        if result_path.exists(): continue
        if (time.perf_counter()-start+job['budget'])>max_hours*3600: break
        attempt=CAMP/'attempts'/(job['job_id']+'_'+uuid.uuid4().hex[:8])
        request=dict(job,identity=dict(job=job,manifest_sha256=sha(CAMP/'manifest.json')))
        settings=dict(threads=manifest['config']['solver_threads'],solver_seconds=job['solver_seconds'],
            reserve_s=30 if 'reference' in job['kind'] else min(job['budget']*.2,2.),source_hashes=manifest['source_hashes'])
        result=run_experiment(settings,request,attempt,worker_module='research.paper1_revision.timing_worker')
        put(result_path,dict(job=job,result=result));completed+=1
        put(CAMP/'status.json',dict(status='running',completed_results=len(list((CAMP/'results').glob('*.json'))),last_job=job['job_id']))
    put(CAMP/'status.json',dict(status='stopped_at_requested_limit_or_finished',completed_this_invocation=completed,
        completed_results=len(list((CAMP/'results').glob('*.json'))),elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--kind',choices=['all','reference_upgrade','external_reference','external_trial','timing_repeat'],default='all')
    p.add_argument('--max-hours',type=float,default=2)
    a=p.parse_args()
    if a.action=='prepare': prepare()
    else: execute(a.kind,a.max_hours)
