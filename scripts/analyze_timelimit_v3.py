"""Analyze what makes scenarios_v3 hit MaxTimeLimit vs solve optimally."""
import json
import os
import glob
import numpy as np
from collections import defaultdict, Counter

reports_dir = 'outputs/scenarios_v3/reports'
scenarios_dir = 'outputs/scenarios_v3'

timelimit = []
optimal = []

for rpath in sorted(glob.glob(os.path.join(reports_dir, 'scenario_*.json'))):
    sid = os.path.basename(rpath).replace('.json', '')
    with open(rpath) as f:
        rep = json.load(f)
    mip = rep.get('mip', {})
    term = mip.get('termination', 'unknown')
    solve_s = mip.get('solve_seconds', 0)

    spath = os.path.join(scenarios_dir, f'{sid}.json')
    if not os.path.exists(spath):
        continue
    with open(spath) as f:
        sc = json.load(f)

    graph = sc.get('graph', {})
    assets = sc.get('assets', {})
    exo = sc.get('exogenous', {})
    tech = sc.get('techno_params_scalers', {})
    costs = sc.get('operation_costs', {})
    econ = sc.get('economics_policy', {})
    diff = sc.get('difficulty_indicators', {})
    flex = sc.get('flexibility_metrics', {})

    zpr = graph.get('zones_per_region', [1])
    n_zones = sum(zpr)
    n_regions = graph.get('regions', 1)

    n_thermal = sum(assets.get('thermal_per_zone', [0]))
    n_nuclear = sum(assets.get('nuclear_per_region', [0]))
    n_solar = sum(assets.get('solar_per_zone', [0]))
    n_wind = sum(assets.get('wind_per_zone', [0]))
    n_batt = sum(assets.get('battery_per_zone', [0]))
    n_dr = sum(assets.get('dr_per_zone', [0]))
    n_pump = sum(assets.get('hydro_pumped_per_region', [0]))
    n_reservoir = sum(assets.get('hydro_reservoir_per_region', [0]))

    rec = {
        'id': sid,
        'solve_s': solve_s,
        'obj': mip.get('objective', 0),
        'termination': term,
        # Structure
        'horizon': sc.get('horizon_hours', 24),
        'regions': n_regions,
        'n_zones': n_zones,
        'intertie_density': graph.get('intertie_density', 0),
        'neighbor_nations': graph.get('neighbor_nations', 0),
        'zone_hetero': float(np.std(zpr) / max(np.mean(zpr), 1)) if len(zpr) > 1 else 0,
        # Assets
        'n_thermal': n_thermal,
        'n_nuclear': n_nuclear,
        'n_solar': n_solar,
        'n_wind': n_wind,
        'n_batt': n_batt,
        'n_dr': n_dr,
        'n_pump': n_pump,
        'n_reservoir': n_reservoir,
        'total_assets': n_thermal + n_nuclear + n_solar + n_wind + n_batt + n_dr + n_pump + n_reservoir,
        'thermal_density': n_thermal / max(n_zones, 1),
        'vre_total': n_solar + n_wind,
        'storage_total': n_batt + n_pump + n_reservoir,
        'n_binary_est': diff.get('n_binary_variables', n_thermal * 24 * 3),
        # Exogenous
        'demand_scale': exo.get('demand_scale_factor', 1.0),
        'weather': exo.get('weather_profiles', ['unknown']),
        'demand_profile': exo.get('demand_profiles', ['unknown']),
        # Tech
        'thermal_ramp_pct': tech.get('thermal_ramp_pct', 0.5),
        'battery_soc_tol': tech.get('battery_final_soc_tolerance', 0.1),
        'battery_e2p': tech.get('battery_e_to_p_hours', 4.0),
        # Costs
        'thermal_fuel': costs.get('thermal_fuel_eur_per_mwh', 60),
        'thermal_startup': costs.get('thermal_startup_cost_eur', 5000),
        'nuclear_startup': costs.get('nuclear_startup_cost_eur', 30000),
        'voll': costs.get('value_of_lost_load_eur_per_mwh', 3000),
        'co2_price': econ.get('co2_price_eur_per_t', 50),
        'cross_border': econ.get('cross_border_policy', 'allow'),
        # Difficulty indicators (pre-computed)
        'vre_pen_pct': diff.get('vre_penetration_pct', 0),
        'net_demand_vol': diff.get('net_demand_volatility', 0),
        'peak_valley': diff.get('peak_to_valley_ratio', 1),
        'complexity': diff.get('complexity_score', 'unknown'),
    }

    if term == 'maxTimeLimit':
        timelimit.append(rec)
    elif term == 'optimal':
        optimal.append(rec)

print(f"Total analyzed: {len(timelimit) + len(optimal)}")
print(f"  MaxTimeLimit: {len(timelimit)}")
print(f"  Optimal:      {len(optimal)}")
print()


def compare(key, label=None):
    if label is None:
        label = key
    tl_vals = [r[key] for r in timelimit if isinstance(r[key], (int, float))]
    op_vals = [r[key] for r in optimal if isinstance(r[key], (int, float))]
    if not tl_vals or not op_vals:
        return
    tl_m, tl_s = np.mean(tl_vals), np.std(tl_vals)
    op_m, op_s = np.mean(op_vals), np.std(op_vals)
    ratio = tl_m / op_m if op_m != 0 else float('inf')
    sep = abs(tl_m - op_m) / max(np.sqrt((tl_s**2 + op_s**2) / 2), 0.001)  # Cohen's d-like
    flag = " ***" if sep > 0.8 else " **" if sep > 0.5 else " *" if sep > 0.3 else ""
    print(f"  {label:30s}  TL: {tl_m:10.2f} +/- {tl_s:8.2f}  |  Opt: {op_m:10.2f} +/- {op_s:8.2f}  |  ratio={ratio:.2f}  d={sep:.2f}{flag}")


print("=" * 130)
print("STRUCTURAL FEATURES")
print("=" * 130)
compare('n_zones', 'Total zones')
compare('regions', 'Regions')
compare('zone_hetero', 'Zone heterogeneity')
compare('intertie_density', 'Intertie density')
compare('neighbor_nations', 'Neighbor nations')

print()
print("=" * 130)
print("ASSET COUNTS")
print("=" * 130)
compare('n_thermal', 'Thermal units')
compare('n_nuclear', 'Nuclear units')
compare('thermal_density', 'Thermal per zone')
compare('vre_total', 'VRE units (solar+wind)')
compare('n_solar', 'Solar units')
compare('n_wind', 'Wind units')
compare('storage_total', 'Storage (batt+pump+res)')
compare('n_batt', 'Battery units')
compare('n_dr', 'DR units')
compare('total_assets', 'Total assets')
compare('n_binary_est', 'Binary vars (est)')

print()
print("=" * 130)
print("EXOGENOUS / DEMAND")
print("=" * 130)
compare('demand_scale', 'Demand scale factor')
compare('vre_pen_pct', 'VRE penetration %')
compare('net_demand_vol', 'Net demand volatility')
compare('peak_valley', 'Peak-to-valley ratio')

print()
print("=" * 130)
print("TECHNICAL PARAMETERS")
print("=" * 130)
compare('thermal_ramp_pct', 'Thermal ramp %')
compare('battery_soc_tol', 'Battery SOC tolerance')
compare('battery_e2p', 'Battery E/P hours')

print()
print("=" * 130)
print("COST PARAMETERS")
print("=" * 130)
compare('thermal_fuel', 'Thermal fuel EUR/MWh')
compare('thermal_startup', 'Thermal startup EUR')
compare('nuclear_startup', 'Nuclear startup EUR')
compare('voll', 'VOLL EUR/MWh')
compare('co2_price', 'CO2 price EUR/t')

print()
print("=" * 130)
print("OBJECTIVE VALUE")
print("=" * 130)
compare('obj', 'Objective (EUR)')

# Cross-border policy distribution
print()
print("=" * 130)
print("CATEGORICAL FEATURES")
print("=" * 130)
tl_cb = Counter(r['cross_border'] for r in timelimit)
op_cb = Counter(r['cross_border'] for r in optimal)
print(f"  Cross-border policy:  TL: {dict(tl_cb)}  |  Opt: {dict(op_cb)}")

tl_cx = Counter(r.get('complexity', 'unk') for r in timelimit)
op_cx = Counter(r.get('complexity', 'unk') for r in optimal)
print(f"  Complexity score:     TL: {dict(tl_cx)}  |  Opt: {dict(op_cx)}")

# Weather profiles
tl_wp = Counter()
op_wp = Counter()
for r in timelimit:
    for w in (r['weather'] if isinstance(r['weather'], list) else [r['weather']]):
        tl_wp[w] += 1
for r in optimal:
    for w in (r['weather'] if isinstance(r['weather'], list) else [r['weather']]):
        op_wp[w] += 1
print(f"  Weather profiles:     TL: {dict(tl_wp)}")
print(f"                        Opt: {dict(op_wp)}")

# Demand profiles
tl_dp = Counter()
op_dp = Counter()
for r in timelimit:
    for d in (r['demand_profile'] if isinstance(r['demand_profile'], list) else [r['demand_profile']]):
        tl_dp[d] += 1
for r in optimal:
    for d in (r['demand_profile'] if isinstance(r['demand_profile'], list) else [r['demand_profile']]):
        op_dp[d] += 1
print(f"  Demand profiles:      TL: {dict(tl_dp)}")
print(f"                        Opt: {dict(op_dp)}")

# Solve time distribution for optimal
print()
print("=" * 130)
print("OPTIMAL SOLVE TIME DISTRIBUTION")
print("=" * 130)
op_times = sorted([r['solve_s'] for r in optimal])
pcts = [10, 25, 50, 75, 90, 95, 99]
for p in pcts:
    idx = int(len(op_times) * p / 100)
    print(f"  P{p:02d}: {op_times[min(idx, len(op_times)-1)]:.1f}s")
print(f"  Max: {op_times[-1]:.1f}s")

# Top discriminating features (sorted by Cohen's d)
print()
print("=" * 130)
print("TOP DISCRIMINATING FEATURES (sorted by effect size)")
print("=" * 130)
features = [
    'n_zones', 'regions', 'zone_hetero', 'intertie_density', 'neighbor_nations',
    'n_thermal', 'n_nuclear', 'thermal_density', 'vre_total', 'storage_total',
    'n_batt', 'n_dr', 'total_assets', 'n_binary_est',
    'demand_scale', 'vre_pen_pct', 'net_demand_vol', 'peak_valley',
    'thermal_ramp_pct', 'battery_soc_tol', 'battery_e2p',
    'thermal_fuel', 'thermal_startup', 'nuclear_startup', 'voll', 'co2_price',
]

results = []
for key in features:
    tl_vals = [r[key] for r in timelimit if isinstance(r[key], (int, float))]
    op_vals = [r[key] for r in optimal if isinstance(r[key], (int, float))]
    if not tl_vals or not op_vals:
        continue
    tl_m, tl_s = np.mean(tl_vals), np.std(tl_vals)
    op_m, op_s = np.mean(op_vals), np.std(op_vals)
    pooled_std = max(np.sqrt((tl_s**2 + op_s**2) / 2), 0.001)
    d = (tl_m - op_m) / pooled_std
    results.append((key, d, tl_m, op_m))

results.sort(key=lambda x: abs(x[1]), reverse=True)
for key, d, tl_m, op_m in results:
    direction = "TL > Opt" if d > 0 else "TL < Opt"
    print(f"  {key:30s}  d={d:+.3f}  ({direction})  TL_mean={tl_m:.2f}  Opt_mean={op_m:.2f}")
