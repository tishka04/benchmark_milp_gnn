"""
Evaluation metrics for Energy-Based Models.

Metrics include:
- Energy gap (positive vs negative)
- Sample quality (feasibility rate)
- Constraint satisfaction
- Distribution similarity (KL divergence, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import entropy


class EBMMetrics:
    """
    Evaluation metrics for EBM training and sampling.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'energy_gap': [],
            'sample_diversity': [],
            'feasibility_rate': [],
            'constraint_violations': [],
        }
    
    def compute_energy_gap(
        self,
        model: nn.Module,
        u_pos: torch.Tensor,
        u_neg: torch.Tensor,
        h: torch.Tensor,
    ) -> float:
        """
        Compute energy gap between positive and negative samples.
        
        A larger gap indicates better separation.
        
        Args:
            model: Energy model
            u_pos: Positive samples [batch_size, dim_u]
            u_neg: Negative samples [batch_size, dim_u]
            h: Graph embeddings [batch_size, dim_h]
            
        Returns:
            Energy gap (E_neg - E_pos)
        """
        with torch.no_grad():
            E_pos = model(u_pos, h).mean().item()
            E_neg = model(u_neg, h).mean().item()
            gap = E_neg - E_pos
        
        self.metrics['energy_gap'].append(gap)
        return gap
    
    def compute_sample_diversity(
        self,
        samples: torch.Tensor,
    ) -> float:
        """
        Compute diversity of generated samples.
        
        Uses pairwise Hamming distance as a measure of diversity.
        
        Args:
            samples: Generated samples [num_samples, dim_u]
            
        Returns:
            Average pairwise Hamming distance
        """
        num_samples = samples.shape[0]
        
        if num_samples < 2:
            return 0.0
        
        # Compute pairwise Hamming distances
        distances = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                hamming_dist = (samples[i] != samples[j]).float().mean().item()
                distances.append(hamming_dist)
        
        diversity = np.mean(distances)
        self.metrics['sample_diversity'].append(diversity)
        
        return diversity
    
    def compute_feasibility_rate(
        self,
        samples: torch.Tensor,
        constraint_checker: Optional[callable] = None,
    ) -> float:
        """
        Compute fraction of samples that satisfy constraints.
        
        Args:
            samples: Generated samples [num_samples, dim_u] or [num_samples, T, dim_u_per_t]
            constraint_checker: Function that checks if sample is feasible
                               Returns True/False for each sample
            
        Returns:
            Feasibility rate (fraction of feasible samples)
        """
        if constraint_checker is None:
            # Default: check basic binary constraints
            constraint_checker = self._default_constraint_checker
        
        num_samples = samples.shape[0]
        feasible_count = 0
        
        for i in range(num_samples):
            if constraint_checker(samples[i]):
                feasible_count += 1
        
        feasibility_rate = feasible_count / num_samples
        self.metrics['feasibility_rate'].append(feasibility_rate)
        
        return feasibility_rate
    
    def _default_constraint_checker(self, u: torch.Tensor) -> bool:
        """
        Default constraint checker for UC/DR/Storage.
        
        Checks basic constraints:
        - Battery cannot charge and discharge simultaneously
        - Pumped storage cannot charge and discharge simultaneously
        - Values are binary {0, 1}
        
        Args:
            u: Binary configuration [dim_u] or [T, dim_u_per_t]
            
        Returns:
            True if feasible, False otherwise
        """
        # Check if values are binary
        if not torch.all((u == 0) | (u == 1)):
            return False
        
        # If temporal structure exists
        if u.dim() > 1:
            T = u.shape[0]
            
            for t in range(T):
                u_t = u[t]
                
                # Assuming structure: [battery_charge, battery_discharge, 
                #                      pumped_charge, pumped_discharge, ...]
                if len(u_t) >= 2:
                    # Battery constraint
                    if u_t[0] == 1 and u_t[1] == 1:
                        return False
                
                if len(u_t) >= 4:
                    # Pumped storage constraint
                    if u_t[2] == 1 and u_t[3] == 1:
                        return False
        
        return True
    
    def compute_constraint_violations(
        self,
        samples: torch.Tensor,
        constraint_functions: Optional[List[callable]] = None,
    ) -> Dict[str, float]:
        """
        Compute violation rates for specific constraints.
        
        Args:
            samples: Generated samples [num_samples, dim_u] or [num_samples, T, dim_u_per_t]
            constraint_functions: List of functions that check individual constraints
                                 Each returns violation count (0 = satisfied, >0 = violated)
            
        Returns:
            Dictionary of constraint violation rates
        """
        if constraint_functions is None:
            constraint_functions = self._default_constraint_functions()
        
        violations = {}
        num_samples = samples.shape[0]
        
        for name, func in constraint_functions:
            violation_count = 0
            for i in range(num_samples):
                violation_count += func(samples[i])
            
            violations[name] = violation_count / num_samples
        
        self.metrics['constraint_violations'].append(violations)
        
        return violations
    
    def _default_constraint_functions(self) -> List[tuple]:
        """
        Default constraint functions for UC/DR/Storage.
        
        Returns:
            List of (constraint_name, constraint_function) tuples
        """
        constraints = []
        
        # Battery simultaneous charge/discharge
        def battery_constraint(u):
            if u.dim() > 1:  # Temporal
                violations = 0
                for t in range(u.shape[0]):
                    if len(u[t]) >= 2 and u[t][0] == 1 and u[t][1] == 1:
                        violations += 1
                return violations
            else:
                if len(u) >= 2 and u[0] == 1 and u[1] == 1:
                    return 1
                return 0
        
        constraints.append(('battery_simultaneous', battery_constraint))
        
        # Pumped storage simultaneous charge/discharge
        def pumped_constraint(u):
            if u.dim() > 1:
                violations = 0
                for t in range(u.shape[0]):
                    if len(u[t]) >= 4 and u[t][2] == 1 and u[t][3] == 1:
                        violations += 1
                return violations
            else:
                if len(u) >= 4 and u[2] == 1 and u[3] == 1:
                    return 1
                return 0
        
        constraints.append(('pumped_simultaneous', pumped_constraint))
        
        return constraints
    
    def compute_kl_divergence(
        self,
        samples_p: torch.Tensor,
        samples_q: torch.Tensor,
        num_bins: int = 10,
    ) -> float:
        """
        Compute KL divergence between two sets of samples.
        
        Useful for comparing generated distribution to true distribution.
        
        Args:
            samples_p: Samples from distribution P [num_samples, dim_u]
            samples_q: Samples from distribution Q [num_samples, dim_u]
            num_bins: Number of bins for histogram
            
        Returns:
            KL(P || Q)
        """
        # Flatten samples
        p_flat = samples_p.flatten().cpu().numpy()
        q_flat = samples_q.flatten().cpu().numpy()
        
        # Compute histograms
        p_hist, _ = np.histogram(p_flat, bins=num_bins, range=(0, 1), density=True)
        q_hist, _ = np.histogram(q_flat, bins=num_bins, range=(0, 1), density=True)
        
        # Add small epsilon to avoid log(0)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # Compute KL divergence
        kl = entropy(p_hist, q_hist)
        
        return kl
    
    def compute_classification_accuracy(
        self,
        model: nn.Module,
        u_pos: torch.Tensor,
        u_neg: torch.Tensor,
        h: torch.Tensor,
    ) -> float:
        """
        Compute classification accuracy for positive vs negative samples.
        
        Treats energy as a binary classifier: lower energy = positive class.
        
        Args:
            model: Energy model
            u_pos: Positive samples
            u_neg: Negative samples
            h: Graph embeddings
            
        Returns:
            Classification accuracy (0 to 1)
        """
        with torch.no_grad():
            E_pos = model(u_pos, h)
            E_neg = model(u_neg, h)
            
            # Classify based on energy: pos should have lower energy
            correct_pos = (E_pos < E_neg).float().mean().item()
            
            # Overall accuracy
            accuracy = correct_pos
        
        return accuracy
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all tracked metrics.
        
        Returns:
            Dictionary of metric averages
        """
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                if isinstance(values[0], dict):
                    # For nested metrics like constraint violations
                    summary[metric_name] = values[-1]  # Most recent
                else:
                    summary[f'{metric_name}_mean'] = np.mean(values)
                    summary[f'{metric_name}_std'] = np.std(values)
        
        return summary


class TemporalMetrics:
    """
    Metrics specific to temporal binary sequences.
    """
    
    @staticmethod
    def compute_temporal_consistency(
        samples: torch.Tensor,
        max_flips: int = 5,
    ) -> float:
        """
        Compute temporal consistency of binary sequences.
        
        Measures how many times variables flip between timesteps.
        
        Args:
            samples: Temporal samples [num_samples, T, dim_u_per_t]
            max_flips: Maximum expected flips (for normalization)
            
        Returns:
            Temporal consistency score (higher = more consistent)
        """
        if samples.dim() < 3:
            return 1.0  # No temporal structure
        
        num_samples, T, dim_u = samples.shape
        
        # Count flips per sample
        flips = []
        for i in range(num_samples):
            sample_flips = 0
            for t in range(T - 1):
                # Count differences between consecutive timesteps
                diff = (samples[i, t] != samples[i, t + 1]).float().sum().item()
                sample_flips += diff
            
            flips.append(sample_flips)
        
        # Normalize by max expected flips
        avg_flips = np.mean(flips)
        consistency = 1.0 - min(avg_flips / (max_flips * T), 1.0)
        
        return consistency
    
    @staticmethod
    def compute_ramping_violations(
        samples: torch.Tensor,
        max_ramp_rate: int = 2,
    ) -> float:
        """
        Compute ramping constraint violations.
        
        In UC/storage, units can't change state too quickly.
        
        Args:
            samples: Temporal samples [num_samples, T, dim_u_per_t]
            max_ramp_rate: Maximum allowed state changes per timestep
            
        Returns:
            Fraction of samples with ramping violations
        """
        if samples.dim() < 3:
            return 0.0
        
        num_samples, T, dim_u = samples.shape
        violations = 0
        
        for i in range(num_samples):
            for t in range(T - 1):
                # Count state changes
                changes = (samples[i, t] != samples[i, t + 1]).float().sum().item()
                
                if changes > max_ramp_rate:
                    violations += 1
                    break  # Count sample as violated
        
        violation_rate = violations / num_samples
        return violation_rate
