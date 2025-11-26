"""
Analysis tools for experimental results.

Computes metrics and generates reports according to the protocol:
- Per-instance metrics (gap, runtime, feasibility)
- Aggregated metrics by scale
- Runtime-quality curves
- Statistical summaries
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class ResultsAnalyzer:
    """
    Analyzes experimental results and generates comprehensive reports.
    """
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.ref_dir = self.results_dir / "results_reference"
        self.std_dir = self.results_dir / "results_standard"
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
    
    def analyze_all(self):
        """Run all analysis steps."""
        print("\nAnalyzing results...")
        
        # Load all results
        results = self._load_all_results()
        
        # Compute per-instance metrics
        metrics = self._compute_per_instance_metrics(results)
        
        # Aggregate by scale
        aggregated = self._aggregate_by_scale(metrics)
        
        # Save metrics
        with open(self.analysis_dir / "per_instance_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(self.analysis_dir / "aggregated_metrics.json", 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"  Saved metrics to {self.analysis_dir}")
        
        return metrics, aggregated
    
    def _load_all_results(self) -> Dict:
        """Load all experimental results."""
        results = defaultdict(lambda: defaultdict(dict))
        
        # Load reference results
        if self.ref_dir.exists():
            for scale_dir in self.ref_dir.iterdir():
                if not scale_dir.is_dir():
                    continue
                
                scale_name = scale_dir.name
                
                for result_file in scale_dir.glob("*.json"):
                    instance_id = int(result_file.stem.split('_')[1])
                    
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    results[scale_name][instance_id]['reference'] = data
        
        # Load standard results
        if self.std_dir.exists():
            for scale_dir in self.std_dir.iterdir():
                if not scale_dir.is_dir():
                    continue
                
                scale_name = scale_dir.name
                
                for result_file in scale_dir.glob("*.json"):
                    parts = result_file.stem.split('_')
                    instance_id = int(parts[1])
                    method = parts[2]  # "milp" or "hybrid"
                    
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    results[scale_name][instance_id][method] = data
        
        return results
    
    def _compute_per_instance_metrics(self, results: Dict) -> Dict:
        """Compute metrics for each instance."""
        metrics = {}
        
        for scale_name, instances in results.items():
            metrics[scale_name] = {}
            
            for instance_id, methods in instances.items():
                instance_metrics = {}
                
                # Get reference cost (J_ref)
                j_ref = methods.get('reference', {}).get('total_cost', None)
                if j_ref is None or j_ref == float('inf'):
                    # Use best cost from either method as reference
                    costs = []
                    if 'milp' in methods and methods['milp'].get('feasible', False):
                        costs.append(methods['milp']['total_cost'])
                    if 'hybrid' in methods and methods['hybrid'].get('feasible', False):
                        costs.append(methods['hybrid']['total_cost'])
                    j_ref = min(costs) if costs else None
                
                instance_metrics['j_ref'] = j_ref
                
                # MILP metrics
                if 'milp' in methods:
                    milp = methods['milp']
                    instance_metrics['milp'] = {
                        'cost': milp.get('total_cost', float('inf')),
                        'time': milp.get('solve_time', 0),
                        'feasible': milp.get('feasible', False),
                        'optimal': milp.get('optimal', False),
                        'mip_gap': milp.get('mip_gap', None),
                        'gap_vs_ref': self._compute_gap(milp.get('total_cost'), j_ref) if j_ref else None
                    }
                
                # Hybrid metrics
                if 'hybrid' in methods:
                    hybrid = methods['hybrid']
                    instance_metrics['hybrid'] = {
                        'cost': hybrid.get('total_cost', float('inf')),
                        'time': hybrid.get('solve_time', 0),
                        'feasible': hybrid.get('feasible', False),
                        'iterations': hybrid.get('iterations', 0),
                        'time_to_first_feasible': hybrid.get('time_to_first_feasible', None),
                        'gap_vs_ref': self._compute_gap(hybrid.get('total_cost'), j_ref) if j_ref else None
                    }
                
                # Comparison
                if 'milp' in instance_metrics and 'hybrid' in instance_metrics:
                    milp_cost = instance_metrics['milp']['cost']
                    hybrid_cost = instance_metrics['hybrid']['cost']
                    milp_time = instance_metrics['milp']['time']
                    hybrid_time = instance_metrics['hybrid']['time']
                    
                    instance_metrics['comparison'] = {
                        'cost_ratio': hybrid_cost / milp_cost if milp_cost > 0 and milp_cost != float('inf') else None,
                        'time_ratio': milp_time / hybrid_time if hybrid_time > 0 else None,
                        'hybrid_wins_cost': hybrid_cost < milp_cost if both_finite(hybrid_cost, milp_cost) else False,
                        'hybrid_wins_time': hybrid_time < milp_time,
                    }
                
                metrics[scale_name][str(instance_id)] = instance_metrics
        
        return metrics
    
    def _compute_gap(self, cost: float, ref_cost: float) -> float:
        """Compute relative gap: (cost - ref) / |ref| * 100%."""
        if ref_cost is None or ref_cost == 0 or ref_cost == float('inf'):
            return None
        if cost == float('inf'):
            return float('inf')
        return (cost - ref_cost) / abs(ref_cost) * 100.0
    
    def _aggregate_by_scale(self, metrics: Dict) -> Dict:
        """Aggregate metrics by scale."""
        aggregated = {}
        
        for scale_name, instances in metrics.items():
            scale_metrics = {
                'n_instances': len(instances),
                'milp': {
                    'costs': [],
                    'times': [],
                    'gaps': [],
                    'optimal_count': 0,
                    'feasible_count': 0,
                    'infeasible_count': 0
                },
                'hybrid': {
                    'costs': [],
                    'times': [],
                    'gaps': [],
                    'times_to_first': [],
                    'feasible_count': 0,
                    'infeasible_count': 0
                },
                'comparison': {
                    'cost_ratios': [],
                    'time_ratios': [],
                    'hybrid_wins_cost_count': 0,
                    'hybrid_wins_time_count': 0
                }
            }
            
            for instance_metrics in instances.values():
                # MILP
                if 'milp' in instance_metrics:
                    milp = instance_metrics['milp']
                    if milp['feasible']:
                        scale_metrics['milp']['costs'].append(milp['cost'])
                        scale_metrics['milp']['times'].append(milp['time'])
                        if milp['gap_vs_ref'] is not None:
                            scale_metrics['milp']['gaps'].append(milp['gap_vs_ref'])
                        if milp['optimal']:
                            scale_metrics['milp']['optimal_count'] += 1
                        scale_metrics['milp']['feasible_count'] += 1
                    else:
                        scale_metrics['milp']['infeasible_count'] += 1
                
                # Hybrid
                if 'hybrid' in instance_metrics:
                    hybrid = instance_metrics['hybrid']
                    if hybrid['feasible']:
                        scale_metrics['hybrid']['costs'].append(hybrid['cost'])
                        scale_metrics['hybrid']['times'].append(hybrid['time'])
                        if hybrid['gap_vs_ref'] is not None:
                            scale_metrics['hybrid']['gaps'].append(hybrid['gap_vs_ref'])
                        if hybrid['time_to_first_feasible']:
                            scale_metrics['hybrid']['times_to_first'].append(hybrid['time_to_first_feasible'])
                        scale_metrics['hybrid']['feasible_count'] += 1
                    else:
                        scale_metrics['hybrid']['infeasible_count'] += 1
                
                # Comparison
                if 'comparison' in instance_metrics:
                    comp = instance_metrics['comparison']
                    if comp.get('cost_ratio'):
                        scale_metrics['comparison']['cost_ratios'].append(comp['cost_ratio'])
                    if comp.get('time_ratio'):
                        scale_metrics['comparison']['time_ratios'].append(comp['time_ratio'])
                    if comp.get('hybrid_wins_cost'):
                        scale_metrics['comparison']['hybrid_wins_cost_count'] += 1
                    if comp.get('hybrid_wins_time'):
                        scale_metrics['comparison']['hybrid_wins_time_count'] += 1
            
            # Compute statistics
            aggregated[scale_name] = {
                'n_instances': scale_metrics['n_instances'],
                'milp': self._compute_statistics(scale_metrics['milp']),
                'hybrid': self._compute_statistics(scale_metrics['hybrid']),
                'comparison': self._compute_statistics(scale_metrics['comparison'])
            }
        
        return aggregated
    
    def _compute_statistics(self, data: Dict) -> Dict:
        """Compute statistical summaries."""
        stats = {}
        
        for key, values in data.items():
            if isinstance(values, list) and len(values) > 0:
                values_array = np.array(values)
                stats[key] = {
                    'mean': float(np.mean(values_array)),
                    'median': float(np.median(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': len(values)
                }
            elif not isinstance(values, list):
                stats[key] = values
        
        return stats
    
    def generate_report(self):
        """Generate human-readable report."""
        # Load aggregated metrics
        with open(self.analysis_dir / "aggregated_metrics.json", 'r') as f:
            aggregated = json.load(f)
        
        report_path = self.analysis_dir / "REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Experimental Results: Hybrid vs MILP\n\n")
            f.write("## Summary by Scale\n\n")
            
            for scale_name in sorted(aggregated.keys()):
                metrics = aggregated[scale_name]
                f.write(f"### {scale_name}\n\n")
                f.write(f"**Instances:** {metrics['n_instances']}\n\n")
                
                # MILP
                f.write("#### MILP\n\n")
                milp = metrics['milp']
                if 'gaps' in milp and 'mean' in milp['gaps']:
                    f.write(f"- Gap vs Ref: {milp['gaps']['mean']:.2f}% ± {milp['gaps']['std']:.2f}%\n")
                if 'times' in milp and 'mean' in milp['times']:
                    f.write(f"- Solve Time: {milp['times']['mean']:.1f}s ± {milp['times']['std']:.1f}s\n")
                if 'feasible_count' in milp:
                    f.write(f"- Feasible: {milp['feasible_count']}/{metrics['n_instances']}\n")
                if 'optimal_count' in milp:
                    f.write(f"- Optimal: {milp['optimal_count']}/{metrics['n_instances']}\n")
                f.write("\n")
                
                # Hybrid
                f.write("#### Hybrid\n\n")
                hybrid = metrics['hybrid']
                if 'gaps' in hybrid and 'mean' in hybrid['gaps']:
                    f.write(f"- Gap vs Ref: {hybrid['gaps']['mean']:.2f}% ± {hybrid['gaps']['std']:.2f}%\n")
                if 'times' in hybrid and 'mean' in hybrid['times']:
                    f.write(f"- Solve Time: {hybrid['times']['mean']:.1f}s ± {hybrid['times']['std']:.1f}s\n")
                if 'feasible_count' in hybrid:
                    f.write(f"- Feasible: {hybrid['feasible_count']}/{metrics['n_instances']}\n")
                f.write("\n")
                
                # Comparison
                f.write("#### Comparison\n\n")
                comp = metrics['comparison']
                if 'time_ratios' in comp and 'mean' in comp['time_ratios']:
                    ratio = comp['time_ratios']['mean']
                    if ratio > 1:
                        f.write(f"- **Hybrid is {ratio:.2f}x faster**\n")
                    else:
                        f.write(f"- **MILP is {1/ratio:.2f}x faster**\n")
                if 'cost_ratios' in comp and 'mean' in comp['cost_ratios']:
                    f.write(f"- Cost Ratio (Hybrid/MILP): {comp['cost_ratios']['mean']:.3f}\n")
                if 'hybrid_wins_cost_count' in comp:
                    f.write(f"- Hybrid wins on cost: {comp['hybrid_wins_cost_count']}/{metrics['n_instances']}\n")
                if 'hybrid_wins_time_count' in comp:
                    f.write(f"- Hybrid wins on time: {comp['hybrid_wins_time_count']}/{metrics['n_instances']}\n")
                f.write("\n")
                f.write("---\n\n")
        
        print(f"\n  Report generated: {report_path}")


def both_finite(a, b):
    """Check if both values are finite."""
    return a != float('inf') and b != float('inf') and a is not None and b is not None
