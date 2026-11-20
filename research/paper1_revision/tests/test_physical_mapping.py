import copy
import json
from pathlib import Path
import numpy as np
from research.paper1_revision.data import RUN,ROOT,read

def test_battery_metadata_is_not_installed_duration(tmp_path):
    from src.milp.scenario_loader import load_scenario_data
    scenarios=read(ROOT/'outputs/thermal_transitions_v1/operational_exit_v1/manifest.json')['cohort']
    raw=read(scenarios[0]['file']);a=copy.deepcopy(raw);b=copy.deepcopy(raw)
    a['tech']['battery_e_to_p_hours']=1.;b['tech']['battery_e_to_p_hours']=9.
    pa=tmp_path/'a.json';pb=tmp_path/'b.json';pa.write_text(json.dumps(a));pb.write_text(json.dumps(b))
    x=load_scenario_data(pa);y=load_scenario_data(pb)
    assert x.battery_energy==y.battery_energy
    for z in x.zones:
        if x.battery_power[z]>0:assert x.battery_energy[z]/x.battery_power[z]==4

def test_prepared_cases_have_independent_identities_and_all_conditions():
    from research.paper1_revision.campaign import CAMP,load_physics
    m=read(CAMP/'manifest.json');c=m['entries']
    assert len(c)==400 and len({x['uuid'] for x in c})==400
    assert {x['dt_minutes'] for x in c}=={30,60,120}
    assert {x['horizon_hours'] for x in c}=={12,24,48,72}
    for batch in ['unfiltered','time_structure','topology','decoupled']:
        case=next(x for x in c if x['batch']==batch);data=load_physics(case['file'])
        assert len(data.periods)*data.dt_hours==case['horizon_hours']
        assert all(np.isfinite(v) and v>=0 for v in data.demand.values())
    assert not list((CAMP/'results').glob('*.json'))
