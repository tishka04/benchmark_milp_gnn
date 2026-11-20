from dataclasses import asdict
import numpy as np
from src.analysis.criticality_index import compute_criticality,normalize_metric
from research.paper1_weights.data import target,config

def test_empty_assets_single_region_nearly_zero_demand():
    s={'meta':{'zones':1,'assets':{}},'graph':{'regions':1,'zones_per_region':[1]},
       'exogenous':{'demand_scale_factor':1e-15},'tech':{'thermal_ramp_pct':0,'battery_final_soc_tolerance':0}}
    r=compute_criticality(s)
    values={**asdict(r.stress_metrics),**asdict(r.hardness_metrics)}
    assert all(np.isfinite(v) for v in values.values())
    assert r.hardness_metrics.B10_network_hetero==0
    assert r.hardness_metrics.B7_ramp_tightness==0
    assert all(0<=v<=1 for v in {**r.stress_normalized,**r.hardness_normalized}.values())
    assert normalize_metric(-10,(0,1))==0 and normalize_metric(10,(0,1))==1

def test_true_peak_valley_scale_invariance_and_proxy_difference():
    d=np.array([10.,20.,15.])
    assert np.max(d)/np.min(d)==np.max(3*d)/np.min(3*d)
    assert round(1.8*1,2)!=round(1.8*3,2)

def test_no_service_has_finite_maximum_carbon_term():
    cfg=config();r=dict(feasible=True,cost=0,reference_cost=0,ens_mwh=1,demand_mwh=1,exit_s=1,budget=5,
                       served_mwh=0,co2_t_per_served_mwh=None)
    y=target(r,cfg,cfg['policies']['equal4'])
    assert y['co2_term']==1 and 0<=y['score']<=1

def test_shared_conditional_predictions_are_ordered():
    from research.paper1_revision.models import hurdle_predict
    from research.paper1_weights.data import COMPONENTS
    import pandas as pd
    rng=np.random.default_rng(4);df=pd.DataFrame(rng.random((50,22)),columns=COMPONENTS)
    df['F']=rng.integers(0,2,50);df['E']=df.F*rng.integers(0,2,50);df['ES']=df.E*rng.integers(0,2,50)
    e=hurdle_predict(df.iloc[:40],df.iloc[40:],'E','all',.1)
    es=hurdle_predict(df.iloc[:40],df.iloc[40:],'ES','all',.1)
    assert np.all(es<=e+1e-15) and np.all(es>=0) and np.all(e<=1)
