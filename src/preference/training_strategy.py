# ==============================================================================
# PRODUCTION-GRADE TRAINING STRATEGY (v2)
# ==============================================================================
# Two-phase training with proper h_raw conditioning:
# - Phase 1: Contrastive (InfoNCE) without LP - stable energy geometry
# - Phase 2: Ranked (WeightedMarginRankingLoss) with LP - cost-aware
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .conditioning import (
    HConditioner,
    DecisionFeatureExtractor,
    FeatureBasedCostProxy,
    ConditionedEBMWrapper,
)
from .decoder import HierarchicalDecoder


class TrainingPhase(Enum):
    """Training phase identifier."""
    PRETRAIN = "pretrain"      # No LP, fast
    FINETUNE = "finetune"      # With LP, selective


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    epochs: int
    use_lp: bool = False
    lp_batch_ratio: float = 0.2      # Fraction of batches using LP
    num_candidates_min: int = 2
    num_candidates_max: int = 8
    use_uncertainty_filter: bool = True
    uncertainty_threshold: float = 0.5  # Top X% uncertain scenarios
    use_cost_proxy_filter: bool = True
    cost_proxy_top_k: int = 2        # Keep top-K candidates after proxy
    learning_rate: float = 1e-4
    langevin_steps: int = 20


@dataclass
class TwoPhaseConfig:
    """Full two-phase training configuration."""
    # Phase 1: Pre-training
    pretrain_epochs: int = 15
    pretrain_lr: float = 1e-3
    pretrain_langevin_steps: int = 10
    pretrain_num_candidates: int = 3
    
    # Phase 2: Fine-tuning
    finetune_epochs: int = 10
    finetune_lr: float = 1e-4
    finetune_langevin_steps: int = 20
    finetune_lp_ratio: float = 0.2
    finetune_num_candidates_min: int = 2
    finetune_num_candidates_max: int = 8
    finetune_uncertainty_top: float = 0.3  # Only LP on top 30% uncertain
    finetune_cost_proxy_top_k: int = 2
    
    # General
    batch_size: int = 4
    margin: float = 1.0
    device: str = "cuda"
    
    def get_phase_config(self, phase: TrainingPhase) -> PhaseConfig:
        """Get configuration for specific phase."""
        if phase == TrainingPhase.PRETRAIN:
            return PhaseConfig(
                name="pretrain",
                epochs=self.pretrain_epochs,
                use_lp=False,
                num_candidates_min=self.pretrain_num_candidates,
                num_candidates_max=self.pretrain_num_candidates,
                use_uncertainty_filter=False,  # No filtering in pretrain
                use_cost_proxy_filter=False,   # No cost proxy in pretrain
                learning_rate=self.pretrain_lr,
                langevin_steps=self.pretrain_langevin_steps,
            )
        else:
            return PhaseConfig(
                name="finetune",
                epochs=self.finetune_epochs,
                use_lp=True,
                lp_batch_ratio=self.finetune_lp_ratio,
                num_candidates_min=self.finetune_num_candidates_min,
                num_candidates_max=self.finetune_num_candidates_max,
                use_uncertainty_filter=True,
                uncertainty_threshold=self.finetune_uncertainty_top,
                use_cost_proxy_filter=True,
                cost_proxy_top_k=self.finetune_cost_proxy_top_k,
                learning_rate=self.finetune_lr,
                langevin_steps=self.finetune_langevin_steps,
            )


class CostProxy(nn.Module):
    """
    Lightweight cost proxy for filtering candidates before LP.
    
    Predicts approximate cost from decisions without solving LP.
    Trained alongside EBM using LP results as supervision.
    """
    
    def __init__(
        self,
        decision_dim: int,
        embedding_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(decision_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Running statistics for normalization
        self.register_buffer('cost_mean', torch.tensor(0.0))
        self.register_buffer('cost_std', torch.tensor(1.0))
        self.n_updates = 0
    
    def forward(self, u_flat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized cost.
        
        Args:
            u_flat: [B, D] flattened decisions
            h: [B, E] embeddings
        
        Returns:
            [B] predicted costs
        """
        x = torch.cat([u_flat, h], dim=-1)
        return self.net(x).squeeze(-1)
    
    def update_statistics(self, costs: torch.Tensor):
        """Update running mean/std from observed LP costs."""
        alpha = 0.1
        batch_mean = costs.mean().item()
        batch_std = costs.std().item() + 1e-6
        
        self.cost_mean = (1 - alpha) * self.cost_mean + alpha * batch_mean
        self.cost_std = (1 - alpha) * self.cost_std + alpha * batch_std
        self.n_updates += 1
    
    def normalize(self, costs: torch.Tensor) -> torch.Tensor:
        """Normalize costs using running statistics."""
        return (costs - self.cost_mean) / self.cost_std


class UncertaintyEstimator:
    """
    Estimates scenario uncertainty for selective LP evaluation.
    
    Uses EBM energy variance across candidates as uncertainty proxy.
    """
    
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self._scenario_uncertainties: Dict[str, float] = {}
    
    def update(self, scenario_id: str, energies: torch.Tensor):
        """
        Update uncertainty estimate for scenario.
        
        Args:
            scenario_id: Scenario identifier
            energies: [K] energies of K candidates
        """
        variance = energies.var().item()
        
        if scenario_id in self._scenario_uncertainties:
            old = self._scenario_uncertainties[scenario_id]
            self._scenario_uncertainties[scenario_id] = (
                (1 - self.ema_alpha) * old + self.ema_alpha * variance
            )
        else:
            self._scenario_uncertainties[scenario_id] = variance
    
    def get_uncertainty(self, scenario_id: str) -> float:
        """Get uncertainty for scenario (higher = more uncertain)."""
        return self._scenario_uncertainties.get(scenario_id, float('inf'))
    
    def should_use_lp(
        self,
        scenario_id: str,
        threshold_quantile: float = 0.3,
    ) -> bool:
        """
        Decide if LP should be used for this scenario.
        
        Args:
            scenario_id: Scenario identifier
            threshold_quantile: Use LP for top X% most uncertain
        
        Returns:
            True if LP should be used
        """
        if not self._scenario_uncertainties:
            return True  # Always use LP initially
        
        uncertainties = list(self._scenario_uncertainties.values())
        threshold = np.percentile(uncertainties, 100 * (1 - threshold_quantile))
        
        return self.get_uncertainty(scenario_id) >= threshold
    
    def get_statistics(self) -> Dict[str, float]:
        """Get uncertainty statistics."""
        if not self._scenario_uncertainties:
            return {}
        
        vals = list(self._scenario_uncertainties.values())
        return {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'n_scenarios': len(vals),
        }


class AdaptiveCandidateScheduler:
    """
    Adapts number of candidates K based on training progress.
    
    Strategy:
    - Start with few candidates (fast iteration)
    - Increase as model improves (better exploration)
    - Reduce for easy scenarios
    """
    
    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 8,
        warmup_batches: int = 100,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        
        # Track per-scenario difficulty
        self._scenario_difficulty: Dict[str, float] = {}
    
    def get_k(
        self,
        scenario_id: Optional[str] = None,
        epoch: int = 0,
        max_epochs: int = 25,
    ) -> int:
        """
        Get number of candidates for current context.
        
        Args:
            scenario_id: Optional scenario for difficulty-based adjustment
            epoch: Current epoch
            max_epochs: Total epochs
        
        Returns:
            Number of candidates K
        """
        self.batch_count += 1
        
        # Linear warmup from k_min to k_max
        progress = min(1.0, self.batch_count / self.warmup_batches)
        base_k = self.k_min + progress * (self.k_max - self.k_min)
        
        # Epoch-based adjustment (more exploration later)
        epoch_factor = 1 + 0.5 * (epoch / max(1, max_epochs))
        
        # Scenario difficulty adjustment
        if scenario_id and scenario_id in self._scenario_difficulty:
            difficulty = self._scenario_difficulty[scenario_id]
            # More candidates for harder scenarios
            diff_factor = 1 + 0.5 * difficulty
        else:
            diff_factor = 1.0
        
        k = int(base_k * epoch_factor * diff_factor)
        return max(self.k_min, min(self.k_max, k))
    
    def update_difficulty(self, scenario_id: str, loss: float):
        """Update difficulty estimate for scenario."""
        alpha = 0.1
        if scenario_id in self._scenario_difficulty:
            old = self._scenario_difficulty[scenario_id]
            self._scenario_difficulty[scenario_id] = (1 - alpha) * old + alpha * loss
        else:
            self._scenario_difficulty[scenario_id] = loss


class TwoPhaseTrainer:
    """
    Production-grade two-phase trainer (v2).
    
    Phase 1 (Pretrain): Contrastive learning
    - No LP Oracle
    - InfoNCE loss for stable energy geometry
    - Conditioned on h_raw via HConditioner
    
    Phase 2 (Finetune): Ranked learning
    - LP Oracle on 20% of batches (selective)
    - WeightedMarginRankingLoss with cost gaps
    - Feature-based cost proxy filtering
    """
    
    def __init__(
        self,
        ebm: nn.Module,
        sampler: Any,
        config: TwoPhaseConfig,
        conditioner: Optional[HConditioner] = None,
        lp_oracle: Optional[Any] = None,
        cost_proxy: Optional[FeatureBasedCostProxy] = None,
        decoder: Optional[HierarchicalDecoder] = None,
        margin: float = 1.0,
    ):
        self.ebm = ebm
        self.sampler = sampler
        self.config = config
        self.lp_oracle = lp_oracle
        self.margin = margin
        
        # Conditioner for h_raw -> h_global
        self.conditioner = conditioner
        
        # Feasibility decoder for mutual exclusion & capacity constraints
        self.decoder = decoder
        
        # Feature-based cost proxy for filtering
        self.cost_proxy = cost_proxy
        self.cost_proxy_optimizer = None
        if cost_proxy is not None:
            self.cost_proxy_optimizer = torch.optim.Adam(
                cost_proxy.parameters(), lr=1e-3
            )
        
        # Feature extractor for cost proxy
        self.feature_extractor = DecisionFeatureExtractor()
        
        # Adaptive components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.candidate_scheduler = AdaptiveCandidateScheduler(
            k_min=config.finetune_num_candidates_min,
            k_max=config.finetune_num_candidates_max,
        )
        
        # Statistics
        self.stats = {
            'lp_calls': 0,
            'lp_skipped': 0,
            'proxy_filtered': 0,
            'total_batches': 0,
        }
    
    def _should_use_lp(
        self,
        phase_config: PhaseConfig,
        batch_idx: int,
        scenario_ids: List[str],
    ) -> List[bool]:
        """Decide which samples in batch should use LP."""
        if not phase_config.use_lp:
            return [False] * len(scenario_ids)
        
        results = []
        for scenario_id in scenario_ids:
            # Random sampling based on lp_batch_ratio
            use_random = np.random.random() < phase_config.lp_batch_ratio
            
            # Uncertainty-based filtering
            if phase_config.use_uncertainty_filter:
                use_uncertain = self.uncertainty_estimator.should_use_lp(
                    scenario_id, phase_config.uncertainty_threshold
                )
                use_lp = use_random and use_uncertain
            else:
                use_lp = use_random
            
            results.append(use_lp)
        
        return results
    
    def _filter_candidates_with_proxy(
        self,
        candidates: torch.Tensor,
        h: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter candidates using cost proxy.
        
        Args:
            candidates: [K, Z, T, F] candidates
            h: [E] embedding
            top_k: Keep top-K candidates
        
        Returns:
            filtered_candidates: [top_k, Z, T, F]
            indices: [top_k] original indices
        """
        if self.cost_proxy is None or top_k >= candidates.shape[0]:
            indices = torch.arange(candidates.shape[0], device=candidates.device)
            return candidates, indices
        
        K = candidates.shape[0]
        
        # Flatten candidates
        candidates_flat = candidates.view(K, -1)
        
        # Pad to expected dimension if needed
        expected_dim = self.cost_proxy.net[0].in_features - h.shape[-1]
        if candidates_flat.shape[-1] < expected_dim:
            pad_size = expected_dim - candidates_flat.shape[-1]
            candidates_flat = torch.nn.functional.pad(candidates_flat, (0, pad_size))
        elif candidates_flat.shape[-1] > expected_dim:
            candidates_flat = candidates_flat[..., :expected_dim]
        
        # Expand embedding
        h_expanded = h.unsqueeze(0).expand(K, -1)
        
        # Predict costs
        with torch.no_grad():
            predicted_costs = self.cost_proxy(candidates_flat, h_expanded)
        
        # Select top-K (lowest predicted cost)
        _, indices = torch.topk(predicted_costs, top_k, largest=False)
        
        self.stats['proxy_filtered'] += K - top_k
        
        return candidates[indices], indices
    
    def _filter_candidates_with_feature_proxy(
        self,
        candidates: torch.Tensor,
        h: torch.Tensor,
        zone_mask: Optional[torch.Tensor],
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter candidates using feature-based cost proxy.
        
        Args:
            candidates: [K, Z, T, F] candidates
            h: [D] global embedding
            zone_mask: [Z] zone mask
            top_k: Keep top-K candidates
        
        Returns:
            filtered_candidates: [top_k, Z, T, F]
            indices: [top_k] original indices
        """
        if self.cost_proxy is None or top_k >= candidates.shape[0]:
            indices = torch.arange(candidates.shape[0], device=candidates.device)
            return candidates, indices
        
        K = candidates.shape[0]
        
        # Expand zone_mask for batch of K candidates
        if zone_mask is not None:
            zone_mask_k = zone_mask.unsqueeze(0).expand(K, -1)
        else:
            zone_mask_k = None
        
        # Expand h for K candidates
        h_expanded = h.unsqueeze(0).expand(K, -1)
        
        # Predict costs using feature-based proxy
        with torch.no_grad():
            predicted_costs = self.cost_proxy(candidates, h_expanded, zone_mask_k)
        
        # Select top-K (lowest predicted cost)
        _, indices = torch.topk(predicted_costs, top_k, largest=False)
        
        self.stats['proxy_filtered'] += K - top_k
        
        return candidates[indices], indices
    
    def _train_feature_cost_proxy(
        self,
        candidates: torch.Tensor,
        h: torch.Tensor,
        zone_mask: Optional[torch.Tensor],
        true_costs: torch.Tensor,
    ):
        """Train feature-based cost proxy on LP results (detached from main graph)."""
        if self.cost_proxy is None or self.cost_proxy_optimizer is None:
            return
        
        K = candidates.shape[0]
        
        # CRITICAL: Detach to avoid backward through main graph
        candidates_detached = candidates.detach()
        h_detached = h.detach()
        
        # Expand zone_mask and h
        if zone_mask is not None:
            zone_mask_k = zone_mask.unsqueeze(0).expand(K, -1)
        else:
            zone_mask_k = None
        h_expanded = h_detached.unsqueeze(0).expand(K, -1)
        
        # Ensure costs are on same device as candidates
        device = candidates_detached.device
        true_costs = true_costs.to(device)
        
        # Normalize costs
        self.cost_proxy.update_statistics(true_costs)
        normalized_costs = self.cost_proxy.normalize(true_costs).to(device)
        
        # Predict and compute loss
        predicted = self.cost_proxy(candidates_detached, h_expanded, zone_mask_k)
        loss = nn.functional.mse_loss(predicted, normalized_costs)
        
        # Update
        self.cost_proxy_optimizer.zero_grad()
        loss.backward()
        self.cost_proxy_optimizer.step()
    
    def _train_cost_proxy(
        self,
        candidates: torch.Tensor,
        h: torch.Tensor,
        true_costs: torch.Tensor,
    ):
        """Train cost proxy on LP results."""
        if self.cost_proxy is None or self.cost_proxy_optimizer is None:
            return
        
        K = candidates.shape[0]
        candidates_flat = candidates.view(K, -1)
        
        # Pad to expected dimension if needed
        expected_dim = self.cost_proxy.net[0].in_features - h.shape[-1]
        if candidates_flat.shape[-1] < expected_dim:
            pad_size = expected_dim - candidates_flat.shape[-1]
            candidates_flat = torch.nn.functional.pad(candidates_flat, (0, pad_size))
        elif candidates_flat.shape[-1] > expected_dim:
            candidates_flat = candidates_flat[..., :expected_dim]
        
        h_expanded = h.unsqueeze(0).expand(K, -1)
        
        # Normalize costs
        self.cost_proxy.update_statistics(true_costs)
        normalized_costs = self.cost_proxy.normalize(true_costs)
        
        # Predict and compute loss
        predicted = self.cost_proxy(candidates_flat, h_expanded)
        loss = nn.functional.mse_loss(predicted, normalized_costs)
        
        # Update
        self.cost_proxy_optimizer.zero_grad()
        loss.backward()
        self.cost_proxy_optimizer.step()
    
    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        phase_config: PhaseConfig,
        epoch: int,
        batch_idx: int,
    ) -> Dict[str, float]:
        """
        Single training step with phase-aware LP usage.
        
        Returns:
            Dictionary of metrics
        """
        self.stats['total_batches'] += 1
        
        device = self.config.device
        
        # Unpack batch
        scenario_ids = batch['scenario_ids']
        h_raw = batch['embeddings'].to(device)  # [B, Z_max, T, D]
        u_milp = batch['milp_decisions'].to(device)  # [B, Z, T, F]
        zone_mask = batch.get('zone_mask')
        if zone_mask is not None:
            zone_mask = zone_mask.to(device)  # [B, Z_max]
        
        B = h_raw.shape[0]
        n_timesteps = batch.get('n_timesteps', 24)
        n_features = batch.get('n_features', 8)
        max_zones = batch.get('max_zones', u_milp.shape[1])
        
        # Use HConditioner for proper h_raw aggregation (attention-based)
        if self.conditioner is not None:
            h_batch = self.conditioner(h_raw, zone_mask)  # [B, D]
        else:
            # Fallback to mean pooling
            if h_raw.dim() == 4:
                if zone_mask is not None:
                    zone_mask_exp = zone_mask.unsqueeze(-1).unsqueeze(-1)
                    h_masked = h_raw * zone_mask_exp
                    h_sum = h_masked.sum(dim=(1, 2))
                    h_count = zone_mask.sum(dim=1, keepdim=True) * n_timesteps
                    h_batch = h_sum / h_count.clamp(min=1)
                else:
                    h_batch = h_raw.mean(dim=(1, 2))
            else:
                h_batch = h_raw
        
        metrics = {}
        
        # Decide LP usage per sample
        use_lp_flags = self._should_use_lp(phase_config, batch_idx, scenario_ids)
        
        # Get adaptive K
        K = self.candidate_scheduler.get_k(
            scenario_id=scenario_ids[0] if scenario_ids else None,
            epoch=epoch,
            max_epochs=phase_config.epochs,
        )
        
        # Update sampler steps
        self.sampler.config.num_steps = phase_config.langevin_steps
        
        total_loss = 0.0
        n_lp_used = 0
        total_energy_pos = 0.0
        total_energy_neg = 0.0
        
        # Phase 2 specific accumulators
        total_loss_rank = 0.0
        total_mean_weight = 0.0
        total_cost_gap = 0.0
        total_cost_gap_p90 = 0.0
        
        for b in range(B):
            scenario_id = scenario_ids[b]
            h = h_batch[b]  # [E]
            u_pos = u_milp[b]  # [Z, T, F]
            
            # Sample candidates - sampler already has EBM internally
            candidates, _ = self.sampler.sample(
                h=h.unsqueeze(0),
                n_samples=K,
                n_zones=max_zones,
                n_timesteps=n_timesteps,
                n_features=n_features,
            )
            candidates = candidates[0]  # [K, Z, T, F]
            
            # Compute energies for uncertainty
            with torch.no_grad():
                cand_flat = candidates.view(K, -1)
                h_exp = h.unsqueeze(0).expand(K, -1)
                energies = self.ebm(cand_flat, h_exp)
            
            # Update uncertainty estimate
            self.uncertainty_estimator.update(scenario_id, energies)
            
            # Filter with feature-based cost proxy if enabled (Phase 2 only)
            zone_mask_b = zone_mask[b] if zone_mask is not None else None
            if phase_config.use_cost_proxy_filter and self.cost_proxy is not None:
                candidates, _ = self._filter_candidates_with_feature_proxy(
                    candidates, h, zone_mask_b, phase_config.cost_proxy_top_k
                )
            
            # Apply feasibility decoder (Phase 2 only) - enforces mutual exclusion
            if phase_config.use_lp and self.decoder is not None:
                K_curr = candidates.shape[0]
                decoded_candidates = []
                for k in range(K_curr):
                    decoded_plan = self.decoder(candidates[k], scenario=None)
                    decoded_candidates.append(decoded_plan.to_tensor())
                candidates = torch.stack(decoded_candidates, dim=0)  # [K, Z, T, F]
            
            # Get costs
            if use_lp_flags[b] and self.lp_oracle is not None:
                self.stats['lp_calls'] += candidates.shape[0]
                n_lp_used += 1
                
                # LP evaluation
                costs, results = self.lp_oracle.evaluate_candidates(
                    scenario_id, candidates
                )
                costs = costs.to(device)  # Ensure costs are on correct device
                
                # Train feature-based cost proxy
                self._train_feature_cost_proxy(candidates, h, zone_mask_b, costs)
            else:
                self.stats['lp_skipped'] += candidates.shape[0]
                # Dummy costs (thermal usage heuristic)
                costs = self._dummy_costs(candidates)
            
            # Compute loss
            # Energy of positive (MILP solution)
            u_pos_flat = u_pos.view(1, -1)
            h_single = h.unsqueeze(0)
            energy_pos = self.ebm(u_pos_flat, h_single)
            
            # Energy of negatives (candidates)
            K_curr = candidates.shape[0]
            cand_flat = candidates.view(K_curr, -1)
            h_exp = h.unsqueeze(0).expand(K_curr, -1)
            energy_neg = self.ebm(cand_flat, h_exp)
            
            # Phase-aware loss computation
            if not phase_config.use_lp:
                # PHASE 1: Improved contrastive loss with hard negatives
                # Lower temperature = sharper contrast
                temperature = 0.1
                
                # Hard negative mining: use top-2 hardest (lowest energy)
                top_k_hard = min(2, K_curr)
                hard_indices = torch.argsort(energy_neg)[:top_k_hard]
                hard_neg_energy = energy_neg[hard_indices]
                
                # Contrastive: push pos down, hard negatives up
                margin_target = 2.0
                loss_contrastive = F.relu(margin_target + energy_pos.mean() - hard_neg_energy.mean())
                
                # Also add InfoNCE for probability interpretation
                all_energies = torch.cat([energy_pos, energy_neg])
                log_partition = torch.logsumexp(-all_energies / temperature, dim=0)
                log_prob_pos = -energy_pos / temperature - log_partition
                loss_nce = -log_prob_pos.mean()
                
                # Energy centering regularization (prevent drift)
                energy_reg = 0.01 * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())
                
                # Spread regularization (stronger)
                spread_loss = 0.5 * F.relu(margin_target - (energy_neg.mean() - energy_pos.mean()))
                
                loss_b = 0.5 * loss_contrastive + 0.5 * loss_nce + energy_reg + spread_loss
            else:
                # PHASE 2: WeightedMarginRankingLoss (with LP costs)
                # w_k = log(1 + (C_k - C_pos)+)
                milp_cost = batch['milp_objectives'][b].to(device) if 'milp_objectives' in batch else 0.0
                cost_gaps = F.relu(costs - milp_cost)
                weights = torch.log1p(cost_gaps)
                weights_norm = weights / (weights.sum() + 1e-6)  # Normalize
                
                # Weighted margin loss: (1 + alpha * w_k) * max(0, m + E_pos - E_neg)
                alpha = 1.0
                margins = F.relu(self.margin + energy_pos - energy_neg)
                weighted_margins = (1 + alpha * weights_norm) * margins
                loss_ranked = weighted_margins.mean()
                
                # Energy regularization
                energy_reg = 0.01 * (energy_pos.pow(2).mean() + energy_neg.pow(2).mean())
                
                loss_b = loss_ranked + energy_reg
                
                # Track Phase 2 specific metrics
                total_loss_rank += loss_ranked.detach().item()
                total_mean_weight += weights.mean().detach().item()
                total_cost_gap += cost_gaps.mean().detach().item()
                if cost_gaps.numel() > 0:
                    sorted_gaps = torch.sort(cost_gaps)[0]
                    p90_idx = int(0.9 * len(sorted_gaps))
                    total_cost_gap_p90 += sorted_gaps[p90_idx].item() if p90_idx < len(sorted_gaps) else 0
            
            total_loss += loss_b
            total_energy_pos += energy_pos.detach().mean().item()
            total_energy_neg += energy_neg.detach().mean().item()
            
            # Update difficulty
            self.candidate_scheduler.update_difficulty(
                scenario_id, loss_b.item()
            )
        
        # Average loss
        loss = total_loss / B
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        metrics['loss'] = loss.item()
        metrics['energy_pos'] = total_energy_pos / B
        metrics['energy_neg'] = total_energy_neg / B
        metrics['energy_gap'] = (total_energy_neg - total_energy_pos) / B
        metrics['lp_ratio'] = n_lp_used / B
        metrics['K'] = K
        metrics['lp_calls_total'] = self.stats['lp_calls']
        metrics['lp_skip_total'] = self.stats['lp_skipped']
        
        # Phase 2 specific metrics
        if phase_config.use_lp:
            metrics['loss_rank'] = total_loss_rank / B
            metrics['mean_weight'] = total_mean_weight / B
            metrics['mean_cost_gap'] = total_cost_gap / B
            metrics['cost_gap_p90'] = total_cost_gap_p90 / B
            metrics['proxy_filtered'] = self.stats.get('proxy_filtered', 0)
        
        return metrics
    
    def _dummy_costs(self, candidates: torch.Tensor) -> torch.Tensor:
        """Generate dummy costs based on thermal usage heuristic."""
        # candidates: [K, Z, T, F]
        # Feature 6 is thermal
        thermal_usage = candidates[..., 6].sum(dim=(1, 2))
        # Higher thermal = higher cost
        return thermal_usage * 50.0 + torch.randn_like(thermal_usage) * 10.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        total = self.stats['lp_calls'] + self.stats['lp_skipped']
        return {
            **self.stats,
            'lp_ratio': self.stats['lp_calls'] / max(1, total),
            'uncertainty_stats': self.uncertainty_estimator.get_statistics(),
        }


def create_two_phase_trainer(
    ebm: nn.Module,
    sampler: Any,
    config: Optional[TwoPhaseConfig] = None,
    lp_oracle: Optional[Any] = None,
    embedding_dim: int = 128,
    use_cost_proxy: bool = True,
    use_conditioner: bool = True,
    use_decoder: bool = True,
) -> TwoPhaseTrainer:
    """
    Factory function for two-phase trainer (v2).
    
    Args:
        ebm: Energy-based model
        sampler: Langevin sampler
        config: Training configuration
        lp_oracle: LP Oracle for cost evaluation
        embedding_dim: Embedding dimension
        use_cost_proxy: Whether to use feature-based cost proxy
        use_conditioner: Whether to use HConditioner for h_raw
        use_decoder: Whether to use HierarchicalDecoder for feasibility
    
    Returns:
        Configured TwoPhaseTrainer with proper conditioning
    """
    if config is None:
        config = TwoPhaseConfig()
    
    # Create HConditioner for attention-based h_raw aggregation
    conditioner = None
    if use_conditioner:
        conditioner = HConditioner(
            embed_dim=embedding_dim,
            hidden_dim=64,
            use_temporal=True,
        ).to(config.device)
    
    # Create feature-based cost proxy (~50 features instead of 25000)
    cost_proxy = None
    if use_cost_proxy:
        feature_extractor = DecisionFeatureExtractor()
        feature_dim = feature_extractor.n_features
        cost_proxy = FeatureBasedCostProxy(
            feature_dim=feature_dim,
            embed_dim=embedding_dim,
            hidden_dim=128,
        ).to(config.device)
    
    # Create HierarchicalDecoder for feasibility (mutual exclusion, capacity)
    decoder = None
    if use_decoder:
        decoder = HierarchicalDecoder(
            enforce_mutual_exclusion=True,
            enforce_capacity=False,  # No scenario data in training loop
            smooth_temporal=False,
        )
    
    return TwoPhaseTrainer(
        ebm=ebm,
        sampler=sampler,
        config=config,
        conditioner=conditioner,
        lp_oracle=lp_oracle,
        cost_proxy=cost_proxy,
        decoder=decoder,
        margin=config.margin,
    )
