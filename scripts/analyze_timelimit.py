"""Analyze timelimit scenarios in eval_stress dataset."""
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

BENCHMARK_ROOT = Path(r'C:\Users\Dell\projects\multilayer_milp_gnn\benchmark')
sys.path.insert(0, str(BENCHMARK_ROOT))

# Define stub classes for unpickling
class SolveStage(Enum):
    hard_fix = "hard_fix"
    repair_20 = "repair_20"
    repair_100 = "repair_100"
    full_soft = "full_soft"

@dataclass
class FeasiblePlan:
    scenario_id: str = ""
    commitment: Optional[dict] = None

@dataclass  
class LPResult:
    scenario_id: str = ""
    status: str = ""
    stage_used: SolveStage = SolveStage.hard_fix
    objective_value: float = 0.0
    solve_time: float = 0.0
    slack_used: float = 0.0
    n_flips: int = 0

# Monkey-patch the classes into the expected module path
class FakeModule:
    FeasiblePlan = FeasiblePlan
    LPResult = LPResult
    SolveStage = SolveStage

sys.modules['src.pipeline.lp_solver'] = FakeModule()

reports_dir = BENCHMARK_ROOT / 'outputs' / 'scenarios_eval_stress' / 'reports'
scenarios_dir = BENCHMARK_ROOT / 'outputs' / 'scenarios_eval_stress'
pipeline_results_path = BENCHMARK_ROOT / 'outputs' / 'pipeline_eval_stress' / 'pipeline_eval_stress_results.pkl'

# ============ Load MILP results ============
milp_data = []
for f in reports_dir.glob('scenario_*.json'):
    with open(f, 'r') as fp:
        data = json.load(fp)
    mip = data.get('mip', {})
    milp_data.append({
        'scenario_file': f.stem,
        'milp_status': mip.get('termination', 'unknown'),
        'milp_solve_time': mip.get('solve_seconds', 0),
        'milp_objective': mip.get('objective', None),
        'milp_unserved': data.get('cost_components', {}).get('unserved_energy', 0),
    })

df_milp = pd.DataFrame(milp_data)
print(f"MILP results: {len(df_milp)}")
print(f"Status distribution:\n{df_milp['milp_status'].value_counts()}")

# ============ Load scenario metadata ============
scenario_meta = []
for sc_file in scenarios_dir.glob('scenario_*.json'):
    with open(sc_file, 'r') as fp:
        data = json.load(fp)
    
    graph = data.get('graph', {})
    diff = data.get('difficulty_indicators', {})
    flex = data.get('flexibility_metrics', {})
    exo = data.get('exogenous', {})
    
    n_zones = sum(graph.get('zones_per_region', []))
    
    scenario_meta.append({
        'scenario_file': sc_file.stem,
        'n_regions': graph.get('regions', 0),
        'n_zones': n_zones,
        'n_binary_variables': diff.get('n_binary_variables', 0),
        'complexity_score': diff.get('complexity_score', 'medium'),
        'vre_penetration_pct': diff.get('vre_penetration_pct', 0),
        'net_demand_volatility': diff.get('net_demand_volatility', 0),
        'peak_to_valley_ratio': diff.get('peak_to_valley_ratio', 1),
        'total_storage_mw': flex.get('total_storage_power_mw', 0),
        'total_dr_mw': flex.get('total_dr_capacity_mw', 0),
        'thermal_flex_ratio': flex.get('thermal_flex_ratio', 0),
        'weather_profile': exo.get('weather_profile', 'unknown'),
        'demand_scale_factor': exo.get('demand_scale_factor', 1.0),
    })

df_meta = pd.DataFrame(scenario_meta)
print(f"Scenario metadata: {len(df_meta)}")

# ============ Load pipeline results ============
with open(pipeline_results_path, 'rb') as f:
    pipeline_results = pickle.load(f)

print(f"Pipeline results: {len(pipeline_results)}")

# Build pipeline dataframe
pipeline_data = []
for item in pipeline_results:
    sc_id = item.get('scenario_id', '')
    # Extract scenario number from ID
    lp_results = item.get('lp_results', [])
    best_idx = item.get('best_sample_idx', 0)
    
    if lp_results and best_idx >= 0 and best_idx < len(lp_results):
        lp_res = lp_results[best_idx]
        
        if hasattr(lp_res, 'scenario_id'):
            row = {
                'scenario_id': sc_id,
                'pipeline_status': lp_res.status,
                'pipeline_stage': lp_res.stage_used.value if hasattr(lp_res.stage_used, 'value') else str(lp_res.stage_used),
                'pipeline_objective': lp_res.objective_value,
                'pipeline_solve_time': lp_res.solve_time,
                'pipeline_slack': getattr(lp_res, 'slack_used', 0.0),
                'pipeline_n_flips': getattr(lp_res, 'n_flips', 0),
            }
        else:
            row = {
                'scenario_id': sc_id,
                'pipeline_status': lp_res.get('status', 'unknown'),
                'pipeline_stage': lp_res.get('stage_used', 'unknown'),
                'pipeline_objective': lp_res.get('objective_value', np.nan),
                'pipeline_solve_time': lp_res.get('solve_time', 0.0),
                'pipeline_slack': lp_res.get('slack_used', 0.0),
                'pipeline_n_flips': lp_res.get('n_flips', 0),
            }
        pipeline_data.append(row)

df_pipeline = pd.DataFrame(pipeline_data)
print(f"Pipeline DataFrame: {df_pipeline.shape}")

# The scenario_id in pipeline results IS the scenario_file (e.g., 'scenario_00001')
df_pipeline['scenario_file'] = df_pipeline['scenario_id']
print(f"\nSample pipeline scenario_files: {df_pipeline['scenario_file'].head().tolist()}")
print(f"Sample MILP scenario_files: {df_milp['scenario_file'].head().tolist()}")

# ============ Merge all data ============
df = df_milp.merge(df_meta, on='scenario_file', how='inner')
df = df.merge(df_pipeline, on='scenario_file', how='inner')
print(f"\nFinal merged DataFrame: {df.shape}")

# ============ Analysis: Timelimit vs Optimal ============
df['is_timelimit'] = df['milp_status'] == 'maxTimeLimit'
print(f"\n{'='*60}")
print("TIMELIMIT vs OPTIMAL COMPARISON")
print(f"{'='*60}")

for col in ['n_zones', 'n_binary_variables', 'vre_penetration_pct', 'net_demand_volatility', 
            'peak_to_valley_ratio', 'total_storage_mw', 'thermal_flex_ratio']:
    if col in df.columns:
        tl_mean = df.loc[df['is_timelimit'], col].mean()
        opt_mean = df.loc[~df['is_timelimit'], col].mean()
        print(f"{col:30s}: timelimit={tl_mean:8.2f}, optimal={opt_mean:8.2f}, ratio={tl_mean/opt_mean if opt_mean else 0:.2f}")

# ============ Pipeline performance on timelimit scenarios ============
print(f"\n{'='*60}")
print("PIPELINE PERFORMANCE ON TIMELIMIT SCENARIOS")
print(f"{'='*60}")

timelimit_df = df[df['is_timelimit']]
print(f"\nTimelimit scenarios: {len(timelimit_df)}")
for _, row in timelimit_df.iterrows():
    print(f"\n{row['scenario_file']}:")
    print(f"  MILP: {row['milp_solve_time']:.1f}s (timelimit), obj={row['milp_objective']/1e6:.2f}M")
    print(f"  Pipeline: {row['pipeline_solve_time']:.1f}s ({row['pipeline_stage']}), obj={row['pipeline_objective']/1e6:.2f}M")
    if row['milp_objective'] and row['milp_objective'] != 0:
        gap = (row['pipeline_objective'] - row['milp_objective']) / abs(row['milp_objective']) * 100
        speedup = row['milp_solve_time'] / row['pipeline_solve_time'] if row['pipeline_solve_time'] > 0 else float('inf')
        print(f"  -> Speedup: {speedup:.0f}x, Gap: {gap:.1f}%")

# Save for notebook
df.to_pickle(BENCHMARK_ROOT / 'outputs' / 'pipeline_eval_stress' / 'analysis_merged.pkl')
print(f"\nSaved merged analysis to analysis_merged.pkl")
