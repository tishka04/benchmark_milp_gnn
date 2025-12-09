"""
Training loop for Energy-Based Models using Contrastive Divergence.

Implements the training objective from Section 4 of the methodology:
L(θ) = E[E_θ(u^+ | h)] - E[E_θ(u^- | h)]

where u^+ are positive samples (MILP-feasible) and u^- are negative samples
(generated via Gibbs sampling).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import EnergyModel, StructuredEnergyModel
from .sampler import GibbsSampler, PersistentContrastiveDivergence
from .metrics import EBMMetrics


class EBMTrainer:
    """
    Trainer for Energy-Based Models using Contrastive Divergence.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sampler: GibbsSampler,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cpu',
        use_wandb: bool = False,
    ):
        """
        Args:
            model: Energy-based model
            sampler: Sampler for generating negative samples
            optimizer: Optimizer (if None, uses AdamW with default params)
            device: Device for training
            use_wandb: Whether to log to Weights & Biases
        """
        self.model = model.to(device)
        self.sampler = sampler
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Logging disabled.")
        
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer
        
        self.metrics = EBMMetrics()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        num_negative_samples: int = 1,
        use_pcd: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with positive samples
            num_negative_samples: Number of negative samples per positive
            use_pcd: Whether to use Persistent Contrastive Divergence
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'energy_pos': [],
            'energy_neg': [],
            'energy_gap': [],
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get positive samples
            u_pos = batch['u'].to(self.device)  # [batch_size, dim_u]
            h = batch['h'].to(self.device)      # [batch_size, dim_h]
            
            batch_size = u_pos.shape[0]
            
            # Compute energy for positive samples
            E_pos = self.model(u_pos, h)  # [batch_size]
            
            # Generate negative samples via Gibbs sampling
            if use_pcd and isinstance(self.sampler, PersistentContrastiveDivergence):
                u_neg = self.sampler.sample(h)
            else:
                # Initialize from random or from positive samples
                u_neg_list = []
                for _ in range(num_negative_samples):
                    # Option 1: Initialize from positive (better mixing)
                    u_init = u_pos + torch.randn_like(u_pos) * 0.1
                    u_init = torch.clamp(u_init, 0, 1)
                    u_init = (u_init > 0.5).float()
                    
                    # Option 2: Random initialization
                    # u_init = torch.randint(0, 2, u_pos.shape, dtype=torch.float32, device=self.device)
                    
                    u_neg_sample = self.sampler.sample(h, initial_u=u_init)
                    u_neg_list.append(u_neg_sample)
                
                u_neg = torch.stack(u_neg_list, dim=1)  # [batch_size, num_neg, dim_u]
                u_neg = u_neg.reshape(-1, u_neg.shape[-1])  # [batch_size * num_neg, dim_u]
                h_expanded = h.unsqueeze(1).repeat(1, num_negative_samples, 1).reshape(-1, h.shape[-1])
            
            # Compute energy for negative samples
            if num_negative_samples > 1:
                E_neg = self.model(u_neg, h_expanded)  # [batch_size * num_neg]
                E_neg = E_neg.reshape(batch_size, num_negative_samples).mean(dim=1)  # [batch_size]
            else:
                E_neg = self.model(u_neg, h)  # [batch_size]
            
            # Contrastive Divergence loss
            # L(θ) = E[E_θ(u^+|h)] - E[E_θ(u^-|h)]
            # We want to minimize energy for positive samples and maximize for negative
            loss = E_pos.mean() - E_neg.mean()
            
            # Add regularization to prevent energy from going to -inf
            # This is important for stability
            energy_reg = 0.01 * (E_pos.pow(2).mean() + E_neg.pow(2).mean())
            loss = loss + energy_reg
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['energy_pos'].append(E_pos.mean().item())
            epoch_metrics['energy_neg'].append(E_neg.mean().item())
            epoch_metrics['energy_gap'].append((E_neg.mean() - E_pos.mean()).item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'E_pos': f"{E_pos.mean().item():.4f}",
                'E_neg': f"{E_neg.mean().item():.4f}",
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/energy_pos': E_pos.mean().item(),
                    'train/energy_neg': E_neg.mean().item(),
                    'train/energy_gap': (E_neg.mean() - E_pos.mean()).item(),
                    'global_step': self.global_step,
                })
        
        # Aggregate epoch metrics
        epoch_summary = {
            k: np.mean(v) for k, v in epoch_metrics.items()
        }
        
        self.current_epoch += 1
        
        return epoch_summary
    
    def validate(
        self,
        dataloader: DataLoader,
        num_negative_samples: int = 10,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation dataloader
            num_negative_samples: Number of negative samples to generate
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {
            'loss': [],
            'energy_pos': [],
            'energy_neg': [],
            'energy_gap': [],
            'classification_acc': [],  # How well does model distinguish pos/neg?
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                u_pos = batch['u'].to(self.device)
                h = batch['h'].to(self.device)
                
                batch_size = u_pos.shape[0]
                
                # Compute energy for positive samples
                E_pos = self.model(u_pos, h)
                
                # Generate negative samples
                u_neg_list = []
                for _ in range(num_negative_samples):
                    u_neg = self.sampler.sample(h)
                    u_neg_list.append(u_neg)
                
                u_neg = torch.stack(u_neg_list, dim=1)
                u_neg = u_neg.reshape(-1, u_neg.shape[-1])
                h_expanded = h.unsqueeze(1).repeat(1, num_negative_samples, 1).reshape(-1, h.shape[-1])
                
                E_neg = self.model(u_neg, h_expanded)
                E_neg = E_neg.reshape(batch_size, num_negative_samples).mean(dim=1)
                
                # Metrics
                loss = E_pos.mean() - E_neg.mean()
                energy_gap = (E_neg.mean() - E_pos.mean()).item()
                
                # Classification accuracy: treat energy as logit for binary classification
                # Lower energy = more likely to be positive sample
                correct = (E_pos < E_neg).float().mean().item()
                
                val_metrics['loss'].append(loss.item())
                val_metrics['energy_pos'].append(E_pos.mean().item())
                val_metrics['energy_neg'].append(E_neg.mean().item())
                val_metrics['energy_gap'].append(energy_gap)
                val_metrics['classification_acc'].append(correct)
        
        # Aggregate metrics
        val_summary = {
            k: np.mean(v) for k, v in val_metrics.items()
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f'val/{k}': v for k, v in val_summary.items()
            })
        
        return val_summary
    
    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")


class ScheduledEBMTrainer(EBMTrainer):
    """
    EBM Trainer with learning rate scheduling and temperature annealing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sampler: GibbsSampler,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cpu',
        use_wandb: bool = False,
        initial_temperature: float = 2.0,
        final_temperature: float = 1.0,
        anneal_steps: int = 10000,
    ):
        """
        Args:
            model: Energy model
            sampler: Sampler
            optimizer: Optimizer
            device: Device
            use_wandb: Log to wandb
            initial_temperature: Starting temperature for sampling
            final_temperature: Final temperature
            anneal_steps: Number of steps to anneal temperature
        """
        super().__init__(model, sampler, optimizer, device, use_wandb)
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=anneal_steps,
            eta_min=1e-6,
        )
    
    def get_current_temperature(self) -> float:
        """
        Get current sampling temperature based on training progress.
        
        Returns:
            Current temperature
        """
        if self.global_step >= self.anneal_steps:
            return self.final_temperature
        
        # Linear annealing
        alpha = self.global_step / self.anneal_steps
        temp = self.initial_temperature * (1 - alpha) + self.final_temperature * alpha
        
        return temp
    
    def train_epoch(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """
        Train with temperature annealing.
        """
        # Update sampler temperature
        current_temp = self.get_current_temperature()
        self.sampler.temperature = current_temp
        
        # Train
        metrics = super().train_epoch(dataloader, **kwargs)
        
        # Step scheduler
        self.scheduler.step()
        
        # Log temperature
        metrics['temperature'] = current_temp
        metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        if self.use_wandb:
            wandb.log({
                'train/temperature': current_temp,
                'train/learning_rate': self.scheduler.get_last_lr()[0],
            })
        
        return metrics
