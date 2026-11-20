"""Executable dictionary of the historical proxy definitions (not new features)."""
from .data import RUN,COMPONENTS,put

FORMULAS=[
('v','ratio','difficulty_indicators.vre_penetration_pct / 100; generator nominal capacity share, rounded to 0.01 percentage point'),
('round(sqrt(weather_volatility^2 + demand_profile_volatility^2),3)','ratio','weather: calm_winter .12, stormy_winter .25, sunny_summer .22, overcast_summer .10, mixed .18; demand: wkday_peak .15, wkend_flat .08, cold_snap .20, heatwave .22, shoulder .10; fallback .15'),
('round(1.8*demand_scale_factor,2)','proxy','Not a realized max(demand)/min(demand) ratio; historical numerical values preserved'),
('weather_map[weather_profile]','ratio','calm_winter .08, stormy_winter .22, sunny_summer .18, overcast_summer .06, mixed .14; fallback .12'),
('demand_scale_factor','ratio','exogenous.demand_scale_factor'),
('650*N*d/max(300*n_thermal+1000*n_nuclear,1)','ratio','capacities MW; denominator floor 1 MW'),
('clip(1-P_storage/max(R,1),-1,1)','ratio','R=650*N*d*(1-.3*v) MW; P_storage from flexibility_metrics'),
('clip(1-E_storage/max(.4*horizon_hours*R,1),-1,1)','ratio','E_storage MWh; denominator floor 1 MWh'),
('clip(1-300*n_thermal*ramp_fraction/max(R,1),-1,1)','ratio','nominal capacity proxy; not realized dispatch ramps'),
('clip(1-P_DR*horizon_hours/max(500*N*d*horizon_hours,1),0,1)','ratio','demand approximation in MWh; denominator floor 1 MWh'),
('trade_factor*min(neighbor_nations/4,1)','ratio','allow .25, cap .15, block .02; fallback .15'),
('1-intertie_density','ratio','inverse intertie-density proxy, not measured congestion'),
('N','zones','meta.zones; fallback sum(graph.zones_per_region)'),
('int(horizon_hours*60/dt_minutes)','steps','time dimension, not physical horizon alone'),
('(2*n_thermal+2*n_nuclear+3*N+1)*N_steps','estimated binaries','generator_v1 difficulty estimate; differs from actual free binary variables, including aggregation and shutdown expressions; fallback .1*estimates.vars_total'),
('n_thermal/max(N,1)','units/zone','equipment counts, not number of active thermal zones'),
('0.35','ratio','fixed proxy, not observed minimum generation ratio'),
('min((n_thermal*c_start_thermal+n_nuclear*c_start_nuclear)/max(.5*horizon_hours*(300*n_thermal+1000*n_nuclear)*c_fuel,1),.5)','ratio','denominator floor EUR 1'),
('300*n_thermal/max(300*n_thermal*ramp_fraction,1)','ratio','denominator floor 1 MW per transition; zero without thermal'),
('battery_e_to_p_hours/max(battery_final_soc_tolerance,.01)','hours','technology parameter ratio; may remain nonzero with no battery installed'),
('intertie_density*regions*(regions-1)/(2*max(N,1))','potential links/zone','regional-connectivity proxy, not realized line count'),
('std_population(zones_per_region)/max(mean(zones_per_region),1)','ratio','zero with a single region')]


def write():
    from src.analysis.criticality_index import NORM_BOUNDS
    put(RUN/'component_dictionary.json',[
        dict(component=c,formula=formula,units=unit,inputs_and_conventions=note,bounds=NORM_BOUNDS[c],
             normalization='clip((raw-lower)/(upper-lower),0,1)',
             source='src/analysis/criticality_index.py; generator-derived quantities in src/generator/generator_v1.py')
        for c,(formula,unit,note) in zip(COMPONENTS,FORMULAS)])
