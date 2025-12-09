"""
Main training script for EBM on MILP UC/DR/Storage binary variables.

Example usage:
    python -m src.ebm.train_ebm --config configs/ebm_config.yaml
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import yaml

from .model import EnergyModel, StructuredEnergyModel
from .sampler import GibbsSampler, PersistentContrastiveDivergence
from .dataset import MILPBinaryDataset, collate_ebm_batch
from .trainer import EBMTrainer, ScheduledEBMTrainer
from .metrics import EBMMetrics, TemporalMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train EBM for MILP binary variables")
    
    parser.add_argument(
        '--scenarios_dir',
        type=str,
        default='outputs/scenarios_v1',
        help='Directory containing MILP solutions'
    )
    parser.add_argument(
        '--embedding_cache_dir',
        type=str,
        default=None,
        help='Directory with cached GNN embeddings as individual .npy files (optional)'
    )
    parser.add_argument(
        '--embedding_file',
        type=str,
        default=None,
        help='Path to single .pt file with all embeddings (e.g., from Google Drive)'
    )
    parser.add_argument(
        '--embedding_loading_mode',
        type=str,
        default='lazy',
        choices=['lazy', 'full'],
        help='How to load embedding_file: lazy (on-demand) or full (load all)'
    )
    parser.add_argument(
        '--temporal_aggregation',
        type=str,
        default='mean',
        choices=['mean', 'max', 'last', 'first'],
        help='How to aggregate temporal embeddings [T, 128] -> [128]'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/ebm_models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='structured',
        choices=['basic', 'structured', 'factorized'],
        help='Type of energy model'
    )
    parser.add_argument(
        '--dim_u',
        type=int,
        default=672,  # 96 timesteps * 7 binary vars per timestep
        help='Dimension of binary decision vector'
    )
    parser.add_argument(
        '--dim_h',
        type=int,
        default=128,
        help='Dimension of graph embedding'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_gibbs_steps',
        type=int,
        default=50,
        help='Number of Gibbs sampling steps'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--use_pcd',
        action='store_true',
        help='Use Persistent Contrastive Divergence'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Log to Weights & Biases'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training EBM on {args.device}")
    print(f"Model type: {args.model_type}")
    print(f"Binary dimension: {args.dim_u}")
    print(f"Embedding dimension: {args.dim_h}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.scenarios_dir}...")
    if args.embedding_file:
        print(f"Using embedding file: {args.embedding_file}")
        print(f"Loading mode: {args.embedding_loading_mode}")
    elif args.embedding_cache_dir:
        print(f"Using embedding cache: {args.embedding_cache_dir}")
    else:
        print("⚠️  Warning: No embeddings provided, using random embeddings")
    
    dataset = MILPBinaryDataset(
        scenarios_dir=args.scenarios_dir,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_file=args.embedding_file,
        embedding_loading_mode=args.embedding_loading_mode,
        temporal=False,  # Flatten temporal structure for now
        temporal_aggregation=args.temporal_aggregation,
        device=args.device,
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ebm_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ebm_batch,
    )
    
    # Create model
    print(f"\nInitializing {args.model_type} energy model...")
    if args.model_type == 'basic':
        model = EnergyModel(
            dim_u=args.dim_u,
            dim_h=args.dim_h,
            hidden_dims=[256, 256, 64],
        )
    elif args.model_type == 'structured':
        model = StructuredEnergyModel(
            dim_u=args.dim_u,
            dim_h=args.dim_h,
            hidden_dims=[256, 256, 64],
            use_quadratic=True,
            quadratic_rank=16,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sampler
    print(f"\nInitializing Gibbs sampler...")
    if args.use_pcd:
        sampler = PersistentContrastiveDivergence(
            energy_model=model,
            num_chains=args.batch_size * 10,
            num_steps=args.num_gibbs_steps,
            temperature=args.temperature,
            device=args.device,
        )
    else:
        sampler = GibbsSampler(
            energy_model=model,
            num_steps=args.num_gibbs_steps,
            temperature=args.temperature,
            device=args.device,
        )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = ScheduledEBMTrainer(
        model=model,
        sampler=sampler,
        optimizer=optimizer,
        device=args.device,
        use_wandb=args.use_wandb,
        initial_temperature=2.0,
        final_temperature=1.0,
        anneal_steps=len(train_loader) * 20,
    )
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project='milp-ebm',
            config=vars(args),
            name=f'ebm_{args.model_type}',
        )
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_gap = -float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'=' * 60}")
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            num_negative_samples=1,
            use_pcd=args.use_pcd,
        )
        
        print(f"\nTraining metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics = trainer.validate(
                val_loader,
                num_negative_samples=5,
            )
            
            print(f"\nValidation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save best model
            if val_metrics['energy_gap'] > best_val_gap:
                best_val_gap = val_metrics['energy_gap']
                checkpoint_path = output_dir / f'ebm_best_{args.model_type}.pt'
                trainer.save_checkpoint(str(checkpoint_path))
                print(f"\nNew best model saved! Energy gap: {best_val_gap:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'ebm_epoch_{epoch + 1}_{args.model_type}.pt'
            trainer.save_checkpoint(str(checkpoint_path))
    
    # Final save
    final_path = output_dir / f'ebm_final_{args.model_type}.pt'
    trainer.save_checkpoint(str(final_path))
    
    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Best validation energy gap: {best_val_gap:.4f}")
    print(f"Final model saved to {final_path}")
    print(f"{'=' * 60}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
