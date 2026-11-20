"""Instrumented HiGHS worker with the existing process-exit supervisor contract."""
import json
import math
import sys
import time
from pathlib import Path


def main():
    control=sys.stdout
    def event(name,**values):
        control.write(json.dumps(dict(event=name,**values))+'\n');control.flush()
    event('BOOT')
    from src.operational_exit.supervisor import enroll_current_process
    enroll_current_process(json.loads(sys.stdin.readline())['job_name']);event('ENROLLED')
    settings=json.loads(sys.stdin.readline())
    import highspy
    from pyomo.environ import Var,Constraint
    from pyomo.contrib.appsi.solvers.highs import Highs
    from src.milp.model import build_uc_model
    from src.operational_exit.physics import apply_cost_units,extract_dispatch,validate_dispatch
    from src.operational_exit.common import atomic_json,sealed,PHYSICS
    from .campaign import load_physics,source_hashes
    from .data import sha
    if settings['source_hashes']!=source_hashes(): raise ValueError('Worker source mismatch')
    event('READY',metadata=dict(highs=highspy.Highs().version(),threads=settings['threads']))
    request=json.loads(sys.stdin.readline());stages={};last=time.perf_counter()
    trace=Path(request['result_file']).with_name('stages.jsonl')
    def mark(stage):
        nonlocal last
        now=time.perf_counter();stages[stage]=now-last;last=now
        with trace.open('a',encoding='utf-8') as f: f.write(json.dumps(dict(stage=stage,seconds=stages[stage],elapsed=now-request['start']))+'\n')
    result=dict(status='no_incumbent',valid=False,physics=PHYSICS)
    try:
        if sha(request['file'])!=request['scenario_sha256']: raise ValueError('Scenario hash mismatch')
        data=load_physics(request['file']);mark('scenario_processing')
        model=build_uc_model(data);apply_cost_units(model);mark('model_construction')
        variables=list(model.component_data_objects(Var))
        dims=dict(variables=len(variables),binaries=sum(v.is_binary() for v in variables),constraints=sum(1 for _ in model.component_data_objects(Constraint)))
        adapter=Highs();adapter.set_instance(model);native=adapter._solver_model;mark('solver_transfer')
        deadline=request['deadline']-settings['reserve_s'];remaining=min(settings['solver_seconds'],deadline-time.perf_counter())
        if remaining<=0: raise TimeoutError('Preparation exhausted compute allowance')
        options=dict(threads=settings['threads'],parallel='on' if settings['threads']>1 else 'off',time_limit=remaining,
                     random_seed=request['seed'],output_flag=False)
        for k,v in options.items():
            if native.setOptionValue(k,v)==highspy.HighsStatus.kError: raise RuntimeError('Rejected option '+k)
        def stop(e):
            if time.perf_counter()>=deadline:e.interrupt()
        for cb in [native.cbMipInterrupt,native.cbSimplexInterrupt,native.cbIpmInterrupt]:cb.subscribe(stop)
        status=native.run();mark('solver_execution')
        if status==highspy.HighsStatus.kError:raise RuntimeError('Native HiGHS failure')
        info=native.getInfo();solution=native.getSolution()
        clean=lambda x:float(x) if math.isfinite(float(x)) else None
        result.update(bound=clean(info.mip_dual_bound),gap=clean(info.mip_gap),objective=clean(info.objective_function_value),
            proven_optimal=native.getModelStatus()==highspy.HighsModelStatus.kOptimal,solver_status=str(native.getModelStatus()),
            options=options,dimensions=dims)
        if solution.value_valid:
            values=solution.col_value;mapping=adapter._pyomo_var_to_solver_var_map
            for var in variables:
                if id(var) in mapping:var.set_value(values[mapping[id(var)]],skip_validation=True)
            dispatch=extract_dispatch(model);mark('solution_extraction')
            check=validate_dispatch(data,dispatch);mark('physical_validation')
            if check['valid'] and not math.isclose(check['cost'],result['objective'],rel_tol=1e-9,abs_tol=1e-3):raise ValueError('Objective mismatch')
            result.update(dispatch=dispatch,validation=check,valid=check['valid'],status='solution' if check['valid'] else 'invalid_dispatch')
    except TimeoutError as e:result.update(status='cooperative_deadline',error=str(e))
    result.update(identity=request['identity'],compute_end=time.perf_counter(),stage_seconds=stages)
    atomic_json(request['result_file'],sealed(result),deadline=request['deadline']);mark('serialization_and_delivery')

if __name__=='__main__':main()
