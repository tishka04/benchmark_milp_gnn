"""
Economic Advantage Analyzer for Pipeline vs MILP.

Implements the economic indicator from the eval spec:
- Monetize solve time: cost_of_time = lambda * solve_time_seconds
- Total cost = solution_cost + cost_of_time
- Compare: Pipeline total vs MILP total
- Sensitivity analysis on lambda (EUR/second)

Lambda represents the operational cost of waiting (e.g., cost of outage risk,
operational standby costs, market opportunity cost per second of delay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class EconomicAdvantageAnalyzer:
    """
    Compute economic advantage of pipeline vs MILP considering time value.
    
    The economic indicator:
        Total_cost(method) = C_solution(method) + lambda * T_solve(method)
    
    Pipeline advantage:
        Advantage = Total_cost(MILP) - Total_cost(Pipeline)
              = (C_milp - C_pipeline) + lambda * (T_milp - T_pipeline)
              = -cost_gap + lambda * time_saved
    
    Positive advantage = pipeline is better (cheaper overall).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Comparison DataFrame from metrics.build_comparison_dataframe()
                Must have columns: pipeline_objective, milp_objective,
                pipeline_solve_time, milp_solve_time, family, criticality_index
        """
        self.df = df.copy()
        self._precompute()
    
    def _precompute(self):
        """Precompute derived columns."""
        self.df['cost_gap'] = self.df['pipeline_objective'] - self.df['milp_objective']
        self.df['time_saved_s'] = self.df['milp_solve_time'] - self.df['pipeline_solve_time']
    
    def compute_advantage(self, lambda_eur_per_sec: float) -> pd.DataFrame:
        """
        Compute economic advantage for a given lambda.
        
        Args:
            lambda_eur_per_sec: Cost of time in EUR/second
            
        Returns:
            DataFrame with advantage columns added
        """
        df = self.df.copy()
        df['lambda'] = lambda_eur_per_sec
        
        # Total cost = solution cost + time cost
        df['total_cost_milp'] = df['milp_objective'] + lambda_eur_per_sec * df['milp_solve_time']
        df['total_cost_pipeline'] = df['pipeline_objective'] + lambda_eur_per_sec * df['pipeline_solve_time']
        
        # Advantage = MILP total - Pipeline total (positive = pipeline better)
        df['advantage'] = df['total_cost_milp'] - df['total_cost_pipeline']
        df['advantage_pct'] = df['advantage'] / df['total_cost_milp'].abs() * 100
        
        # Decomposition
        df['advantage_from_time'] = lambda_eur_per_sec * df['time_saved_s']
        df['advantage_from_cost'] = -df['cost_gap']  # Negative gap = pipeline cheaper
        
        # Is pipeline profitable?
        df['pipeline_profitable'] = df['advantage'] > 0
        
        return df
    
    def sensitivity_analysis(
        self,
        lambda_values: Optional[List[float]] = None,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on lambda.
        
        Args:
            lambda_values: Specific lambda values to test (EUR/second)
            n_points: Number of points if lambda_values not provided
            
        Returns:
            DataFrame with one row per lambda value, aggregated statistics
        """
        if lambda_values is None:
            # Range from 0 to 100 EUR/second
            # Typical values: 1-10 EUR/s for moderate urgency, 10-100 for critical
            lambda_values = np.concatenate([
                np.linspace(0, 1, 10),
                np.linspace(1, 10, 15),
                np.linspace(10, 50, 15),
                np.linspace(50, 100, 10),
            ])
            lambda_values = sorted(set(lambda_values))
        
        rows = []
        for lam in lambda_values:
            df_adv = self.compute_advantage(lam)
            valid = df_adv.dropna(subset=['advantage'])
            
            row = {
                'lambda': lam,
                'n_scenarios': len(valid),
                'n_profitable': int(valid['pipeline_profitable'].sum()),
                'pct_profitable': float(valid['pipeline_profitable'].mean() * 100),
                'mean_advantage': float(valid['advantage'].mean()),
                'median_advantage': float(valid['advantage'].median()),
                'total_advantage': float(valid['advantage'].sum()),
                'mean_advantage_pct': float(valid['advantage_pct'].mean()),
                'mean_time_saved_s': float(valid['time_saved_s'].mean()),
                'mean_cost_gap': float(valid['cost_gap'].mean()),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def sensitivity_by_family(
        self,
        lambda_values: Optional[List[float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run sensitivity analysis per family."""
        if lambda_values is None:
            lambda_values = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
        
        families = self.df['family'].unique()
        results = {}
        
        for fam in families:
            if not fam:
                continue
            sub = EconomicAdvantageAnalyzer(self.df[self.df['family'] == fam])
            results[fam] = sub.sensitivity_analysis(lambda_values=lambda_values)
        
        return results
    
    def find_breakeven_lambda(self, target_pct_profitable: float = 50.0) -> float:
        """
        Find the lambda value where target_pct_profitable% of scenarios 
        become profitable for the pipeline.
        
        Uses binary search.
        """
        lo, hi = 0.0, 1000.0
        
        for _ in range(50):
            mid = (lo + hi) / 2
            df_adv = self.compute_advantage(mid)
            valid = df_adv.dropna(subset=['advantage'])
            pct = valid['pipeline_profitable'].mean() * 100
            
            if pct < target_pct_profitable:
                lo = mid
            else:
                hi = mid
        
        return (lo + hi) / 2
    
    def find_breakeven_lambda_by_family(
        self,
        target_pct: float = 50.0,
    ) -> Dict[str, float]:
        """Find breakeven lambda for each family."""
        families = self.df['family'].unique()
        breakevens = {}
        
        for fam in families:
            if not fam:
                continue
            sub = EconomicAdvantageAnalyzer(self.df[self.df['family'] == fam])
            breakevens[fam] = sub.find_breakeven_lambda(target_pct)
        
        return breakevens
    
    def compute_summary(
        self,
        lambda_values: List[float] = [1, 5, 10, 50],
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive summary for reporting.
        
        Args:
            lambda_values: Lambda values to highlight in summary
            
        Returns:
            Dict with global and per-family summaries
        """
        summary = {
            'breakeven_lambda_50pct': self.find_breakeven_lambda(50.0),
            'breakeven_lambda_80pct': self.find_breakeven_lambda(80.0),
            'breakeven_lambda_95pct': self.find_breakeven_lambda(95.0),
        }
        
        # Per-lambda summaries
        for lam in lambda_values:
            df_adv = self.compute_advantage(lam)
            valid = df_adv.dropna(subset=['advantage'])
            summary[f'lambda_{lam}'] = {
                'pct_profitable': float(valid['pipeline_profitable'].mean() * 100),
                'mean_advantage_eur': float(valid['advantage'].mean()),
                'total_advantage_eur': float(valid['advantage'].sum()),
            }
        
        # Per-family breakevens
        summary['family_breakevens'] = self.find_breakeven_lambda_by_family(50.0)
        
        return summary
