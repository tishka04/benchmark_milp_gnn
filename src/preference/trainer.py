# ==============================================================================
# PREFERENCE-BASED EBM TRAINER
# ==============================================================================
# Main training loop integrating:
# - HTE for scenario encoding
# - EBM for energy computation
# - Langevin sampler for candidate generation
# - Decoder + LP worker for economic evaluation
# - Preference-based loss for learning
# ==============================================================================

from __future__ import annotations

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from .data_models import DecisionVector, CandidateResult, TrainingBatch
from .ebm import ConditionalEBM
from .sampler import LangevinSampler, SamplerConfig, binarize_decisions
from .decoder import HierarchicalDecoder, decision_to_decoder_output
from .losses import CombinedPreferenceLoss, WeightedMarginRankingLoss
from .dataset import PreferenceDataset, PreferenceDataLoader


@dataclass
class TrainerConfig:
    """Configuration for preference-based training."""
    # Model
    h_dim: int = 128  # HTE embedding dimension
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    
    # Sampling
    num_candidates: int = 5  # K candidates per scenario
    langevin_steps: int = 50
    langevin_step_size: float = 0.01
    langevin_noise: float = 0.01
    
    # Loss
    margin: float = 1.0
    alpha: float = 1.0  # Cost-aware weight scaling
    w_max: float = 5.0
    energy_reg: float = 0.01
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Training
    epochs: int = 50
    eval_every: int = 5
    save_every: int = 10
    log_every: int = 10
    
    # LP Worker
    lp_solver: str = "appsi_highs"
    lp_time_limit: float = 30.0
    slack_tolerance: float = 1.0
    
    # Paths
    output_dir: str = "outputs/preference_training"
    checkpoint_dir: Optional[str] = None


class PreferenceTrainer:
    """
    Trainer for preference-based EBM learning.
    
    Training loop:
    1. Load batch of scenarios
    2. Compute HTE embeddings h = HTE(x)
    3. Sample K candidates using Langevin: {u^(k)} ~ S_θ(h)
    4. Binarize and pass through decoder + LP worker
    5. Compute preference loss and update EBM
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        ebm: ConditionalEBM,
        hte_encoder: Optional[nn.Module] = None,
        lp_worker: Optional[Any] = None,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Args:
            config: Trainer configuration
            ebm: Conditional EBM model
            hte_encoder: Optional HTE encoder (if None, uses pre-computed embeddings)
            lp_worker: Optional LP worker for cost evaluation
            device: Training device
        """
        self.config = config
        self.device = device
        
        # Models
        self.ebm = ebm.to(device)
        self.hte_encoder = hte_encoder.to(device) if hte_encoder else None
        
        # Sampler
        sampler_config = SamplerConfig(
            num_steps=config.langevin_steps,
            step_size=config.langevin_step_size,
            noise_scale=config.langevin_noise,
        )
        self.sampler = LangevinSampler(self.ebm, sampler_config)
        
        # Decoder
        self.decoder = HierarchicalDecoder()
        
        # LP Worker (optional - can be set later)
        self.lp_worker = lp_worker
        
        # Loss
        self.loss_fn = CombinedPreferenceLoss(
            margin=config.margin,
            alpha=config.alpha,
            w_max=config.w_max,
            energy_reg_weight=config.energy_reg,
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.ebm.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )
        
        # Logging
        self.history = {
            "train_loss": [],
            "ranking_loss": [],
            "energy_reg": [],
            "mean_energy_pos": [],
            "mean_energy_neg": [],
            "mean_cost_gap": [],
            "hard_fix_rate": [],
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        if config.checkpoint_dir:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(
        self,
        dataloader: PreferenceDataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dict with epoch metrics
        """
        self.ebm.train()
        
        epoch_losses = []
        epoch_ranking = []
        epoch_energy_reg = []
        epoch_energy_pos = []
        epoch_energy_neg = []
        epoch_cost_gaps = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            loss, metrics = self.train_step(batch)
            
            epoch_losses.append(loss)
            epoch_ranking.append(metrics.get("ranking_loss", 0))
            epoch_energy_reg.append(metrics.get("energy_reg", 0))
            epoch_energy_pos.append(metrics.get("mean_energy_pos", 0))
            epoch_energy_neg.append(metrics.get("mean_energy_neg", 0))
            epoch_cost_gaps.append(metrics.get("mean_cost_gap", 0))
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "E+": f"{metrics.get('mean_energy_pos', 0):.2f}",
                "E-": f"{metrics.get('mean_energy_neg', 0):.2f}",
            })
        
        # Scheduler step
        self.scheduler.step()
        
        epoch_metrics = {
            "train_loss": np.mean(epoch_losses),
            "ranking_loss": np.mean(epoch_ranking),
            "energy_reg": np.mean(epoch_energy_reg),
            "mean_energy_pos": np.mean(epoch_energy_pos),
            "mean_energy_neg": np.mean(epoch_energy_neg),
            "mean_cost_gap": np.mean(epoch_cost_gaps),
        }
        
        # Update history
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return epoch_metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Single training step.
        
        Args:
            batch: Batch from dataloader
        
        Returns:
            loss: Scalar loss value
            metrics: Dict with step metrics
        """
        self.optimizer.zero_grad()
        
        B = len(batch["scenario_ids"])
        
        # Get embeddings
        if self.hte_encoder is not None:
            # Compute embeddings from raw scenarios
            # This would require building graph data from raw_scenarios
            # For now, use pre-computed embeddings
            h = batch["embeddings"].to(self.device)
        else:
            h = batch["embeddings"].to(self.device)
        
        # Get decision dimensions from first scenario
        n_zones = max(batch["n_zones"])
        n_timesteps = max(batch["n_timesteps"])
        
        # Sample candidates using Langevin
        u_candidates, _ = self.sampler.sample(
            h=h,
            n_samples=self.config.num_candidates,
            n_zones=n_zones,
            n_timesteps=n_timesteps,
            n_features=8,
        )  # [B, K, Z, T, 8]
        
        # Binarize
        u_binary = binarize_decisions(u_candidates, method="threshold")
        
        # Get positive (MILP) decisions
        u_positive = batch["milp_decisions"].to(self.device)  # [B, Z, T, 8]
        costs_positive = batch["milp_objectives"].to(self.device)  # [B]
        
        # Evaluate candidates with LP worker (if available)
        if self.lp_worker is not None:
            costs_negative = self._evaluate_candidates(
                u_binary, 
                batch["scenario_paths"],
                batch["n_zones"],
            )
        else:
            # Use dummy costs based on energy (for testing without LP worker)
            with torch.no_grad():
                costs_negative = self._dummy_costs(u_binary, h)
        
        costs_negative = costs_negative.to(self.device)  # [B, K]
        
        # Compute energies
        # Positive: [B]
        u_pos_flat = u_positive.view(B, -1)
        energy_positive = self.ebm(u_pos_flat, h)
        
        # Negatives: [B, K]
        K = self.config.num_candidates
        u_neg_flat = u_binary.view(B * K, -1)
        h_expanded = h.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        energy_negatives = self.ebm(u_neg_flat, h_expanded).view(B, K)
        
        # Compute loss
        loss, loss_components = self.loss_fn(
            energy_positive=energy_positive,
            energy_negatives=energy_negatives,
            cost_positive=costs_positive,
            costs_negative=costs_negative,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.ebm.parameters(),
                self.config.gradient_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            cost_gaps = (costs_negative - costs_positive.unsqueeze(1)).mean()
        
        metrics = {
            **loss_components,
            "mean_energy_pos": energy_positive.mean().item(),
            "mean_energy_neg": energy_negatives.mean().item(),
            "mean_cost_gap": cost_gaps.item(),
        }
        
        return loss.item(), metrics
    
    def _evaluate_candidates(
        self,
        u_binary: torch.Tensor,  # [B, K, Z, T, 7]
        scenario_paths: List[str],
        n_zones: List[int],
    ) -> torch.Tensor:
        """
        Evaluate candidates using LP worker.
        
        Args:
            u_binary: Binary decisions [B, K, Z, T, 7]
            scenario_paths: Paths to scenario JSON files
            n_zones: Number of zones per scenario
        
        Returns:
            costs: [B, K] objective values from LP worker
        """
        B, K = u_binary.shape[:2]
        costs = torch.zeros(B, K)
        
        for b in range(B):
            scenario_id = Path(scenario_paths[b]).stem
            
            for k in range(K):
                # Convert to decoder output format
                u_k = u_binary[b, k]  # [Z, T, 7]
                decoder_out = decision_to_decoder_output(u_k)
                
                # Run LP worker
                try:
                    result = self.lp_worker.solve(scenario_id, decoder_out)
                    costs[b, k] = result.objective_value
                except Exception as e:
                    # Use high cost for failures
                    costs[b, k] = 1e9
        
        return costs
    
    def _dummy_costs(
        self,
        u_binary: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate dummy costs for testing without LP worker.
        Uses a simple heuristic based on decision patterns.
        """
        B, K, Z, T, F = u_binary.shape
        
        # Simple heuristic: more thermal = higher cost
        thermal_usage = u_binary[..., 6].sum(dim=(-1, -2))  # [B, K]
        base_cost = 1e6 + thermal_usage * 1e4
        
        # Add some noise
        noise = torch.randn(B, K) * 1e4
        
        return base_cost + noise
    
    def train(
        self,
        train_loader: PreferenceDataLoader,
        val_loader: Optional[PreferenceDataLoader] = None,
    ) -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation loader
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"PREFERENCE-BASED EBM TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Candidates per scenario: {self.config.num_candidates}")
        print(f"Langevin steps: {self.config.langevin_steps}")
        print(f"{'='*60}\n")
        
        best_loss = float("inf")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            epoch_metrics = self.train_epoch(train_loader, epoch)
            
            # Log
            if epoch % self.config.log_every == 0:
                print(f"\nEpoch {epoch}/{self.config.epochs}")
                print(f"  Loss: {epoch_metrics['train_loss']:.4f}")
                print(f"  Ranking: {epoch_metrics['ranking_loss']:.4f}")
                print(f"  E+: {epoch_metrics['mean_energy_pos']:.2f}, "
                      f"E-: {epoch_metrics['mean_energy_neg']:.2f}")
                print(f"  Cost gap: {epoch_metrics['mean_cost_gap']:.0f}")
            
            # Evaluate
            if val_loader and epoch % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_loader)
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch, epoch_metrics["train_loss"])
            
            # Best model
            if epoch_metrics["train_loss"] < best_loss:
                best_loss = epoch_metrics["train_loss"]
                self.save_checkpoint(epoch, best_loss, is_best=True)
        
        # Save final history
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Best loss: {best_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: PreferenceDataLoader,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.ebm.eval()
        
        losses = []
        
        for batch in dataloader:
            h = batch["embeddings"].to(self.device)
            u_positive = batch["milp_decisions"].to(self.device)
            costs_positive = batch["milp_objectives"].to(self.device)
            
            B = h.shape[0]
            n_zones = max(batch["n_zones"])
            n_timesteps = max(batch["n_timesteps"])
            
            # Sample candidates
            u_candidates, _ = self.sampler.sample(
                h=h,
                n_samples=self.config.num_candidates,
                n_zones=n_zones,
                n_timesteps=n_timesteps,
            )
            u_binary = binarize_decisions(u_candidates)
            
            # Dummy costs for evaluation
            costs_negative = self._dummy_costs(u_binary, h).to(self.device)
            
            # Compute energies
            K = self.config.num_candidates
            energy_positive = self.ebm(u_positive.view(B, -1), h)
            
            u_neg_flat = u_binary.view(B * K, -1)
            h_exp = h.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            energy_negatives = self.ebm(u_neg_flat, h_exp).view(B, K)
            
            loss, _ = self.loss_fn(
                energy_positive, energy_negatives,
                costs_positive, costs_negative
            )
            losses.append(loss.item())
        
        return {"val_loss": np.mean(losses)}
    
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "ebm_state_dict": self.ebm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }
        
        save_dir = self.config.checkpoint_dir or self.config.output_dir
        
        if is_best:
            path = Path(save_dir) / "best_model.pt"
        else:
            path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.ebm.load_state_dict(checkpoint["ebm_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"], checkpoint["loss"]
    
    def save_history(self):
        """Save training history."""
        path = Path(self.config.output_dir) / "training_history.json"
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_trainer(
    config: TrainerConfig,
    decision_dim: int,
    hte_encoder: Optional[nn.Module] = None,
    lp_worker: Optional[Any] = None,
    device: torch.device = torch.device("cuda"),
) -> PreferenceTrainer:
    """
    Factory function to create trainer with all components.
    
    Args:
        config: Trainer configuration
        decision_dim: Dimension of flattened decision vector (Z * T * 7)
        hte_encoder: Optional HTE encoder
        lp_worker: Optional LP worker
        device: Training device
    
    Returns:
        Configured PreferenceTrainer
    """
    from .ebm import ConditionalEBM
    
    ebm = ConditionalEBM(
        h_dim=config.h_dim,
        decision_dim=decision_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    
    return PreferenceTrainer(
        config=config,
        ebm=ebm,
        hte_encoder=hte_encoder,
        lp_worker=lp_worker,
        device=device,
    )
