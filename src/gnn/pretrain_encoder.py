"""
Pre-train HGT + Temporal Transformer encoder for EBM.

This script trains the encoder to produce high-quality embeddings using
self-supervised learning (reconstruction or contrastive loss).

The trained encoder will be used to generate embeddings for EBM training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.gnn.models.temporal_hetero_gnn import (
    HGTTemporalTransformer,
    TemporalGraphDataset,
)


def reconstruction_loss(embeddings, original_features, mask=None):
    """
    Simple reconstruction loss: project embeddings back to feature space.
    
    Args:
        embeddings: [N, T, hidden_dim]
        original_features: [N*T, feature_dim]
        mask: [N*T] which nodes to compute loss on
    """
    N, T, hidden_dim = embeddings.shape
    feature_dim = original_features.shape[1]
    
    # Project embeddings to feature space
    decoder = nn.Linear(hidden_dim, feature_dim).to(embeddings.device)
    reconstructed = decoder(embeddings).view(N * T, feature_dim)
    
    if mask is not None:
        loss = nn.MSELoss()(reconstructed[mask], original_features[mask])
    else:
        loss = nn.MSELoss()(reconstructed, original_features)
    
    return loss


def contrastive_loss(embeddings, temperature=0.07):
    """
    InfoNCE contrastive loss across time steps.
    
    Positive pairs: same node at adjacent timesteps
    Negative pairs: different nodes or distant timesteps
    
    Args:
        embeddings: [N, T, hidden_dim]
    """
    N, T, D = embeddings.shape
    
    # Flatten to [N*T, D]
    flat = embeddings.view(N * T, D)
    
    # Normalize
    flat = nn.functional.normalize(flat, dim=1)
    
    # Compute similarity matrix [N*T, N*T]
    sim_matrix = torch.matmul(flat, flat.t()) / temperature
    
    # Create positive pair mask (adjacent timesteps for same node)
    positive_mask = torch.zeros(N * T, N * T, device=embeddings.device)
    for i in range(N):
        for t in range(T - 1):
            idx = i * T + t
            idx_next = i * T + t + 1
            positive_mask[idx, idx_next] = 1
            positive_mask[idx_next, idx] = 1
    
    # InfoNCE loss
    exp_sim = torch.exp(sim_matrix)
    positive_sim = exp_sim * positive_mask
    
    loss = -torch.log(
        positive_sim.sum(dim=1) / (exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix)))
    ).mean()
    
    return loss


def train_epoch(model, loader, optimizer, device, loss_type="reconstruction"):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Get N_base and T from first graph in batch (assume all same for now)
        # PyG batching concatenates graphs, so we need to handle this carefully
        # For simplicity with batch_size=1 or same-size graphs:
        if hasattr(batch, 'N_base'):
            if isinstance(batch.N_base, torch.Tensor):
                N_base = batch.N_base[0].item() if batch.N_base.dim() > 0 else batch.N_base.item()
            else:
                N_base = batch.N_base
        else:
            print("Warning: N_base not found in batch, skipping")
            continue
        
        if hasattr(batch, 'T'):
            if isinstance(batch.T, torch.Tensor):
                T = batch.T[0].item() if batch.T.dim() > 0 else batch.T.item()
            else:
                T = batch.T
        else:
            print("Warning: T not found in batch, skipping")
            continue
        
        # Get embeddings (no decoder)
        node_type = batch.node_type if hasattr(batch, 'node_type') else None
        
        embeddings = model(
            batch.x,
            batch.edge_index,
            batch.edge_type,
            node_type,
            N_base,
            T,
            batch=batch.batch if hasattr(batch, 'batch') else None,
            return_sequence=True,
        )
        
        # Compute loss
        if loss_type == "reconstruction":
            mask = batch.label_mask if hasattr(batch, 'label_mask') else None
            loss = reconstruction_loss(embeddings, batch.x, mask)
        elif loss_type == "contrastive":
            # For batched graphs, we need to handle each graph separately
            if embeddings.dim() == 4:  # [batch_size, N_base, T, D]
                total_batch_loss = 0
                for i in range(embeddings.size(0)):
                    total_batch_loss += contrastive_loss(embeddings[i])
                loss = total_batch_loss / embeddings.size(0)
            else:  # [N_base, T, D] single graph
                loss = contrastive_loss(embeddings)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def save_embeddings(model, loader, output_path, device):
    """Generate and save embeddings for all graphs."""
    model.eval()
    embeddings_list = []
    metadata_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            batch = batch.to(device)
            
            # Get N_base and T
            if hasattr(batch, 'N_base'):
                if isinstance(batch.N_base, torch.Tensor):
                    N_base = batch.N_base[0].item() if batch.N_base.dim() > 0 else batch.N_base.item()
                else:
                    N_base = batch.N_base
            else:
                continue
            
            if hasattr(batch, 'T'):
                if isinstance(batch.T, torch.Tensor):
                    T = batch.T[0].item() if batch.T.dim() > 0 else batch.T.item()
                else:
                    T = batch.T
            else:
                continue
            
            node_type = batch.node_type if hasattr(batch, 'node_type') else None
            
            embeddings = model(
                batch.x,
                batch.edge_index,
                batch.edge_type,
                node_type,
                N_base,
                T,
                batch=batch.batch if hasattr(batch, 'batch') else None,
                return_sequence=True,
            )
            
            # Handle batched or single graph
            if embeddings.dim() == 4:  # [batch_size, N_base, T, D]
                for i in range(embeddings.size(0)):
                    embeddings_list.append(embeddings[i].cpu().numpy())
                    metadata_list.append({
                        "N_base": N_base,
                        "T": T,
                        "shape": list(embeddings[i].shape),
                    })
            else:  # [N_base, T, D]
                embeddings_list.append(embeddings.cpu().numpy())
                metadata_list.append({
                    "N_base": N_base,
                    "T": T,
                    "shape": list(embeddings.shape),
                })
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save each embedding separately with index
    save_dict = {}
    for i, (emb, meta) in enumerate(zip(embeddings_list, metadata_list)):
        save_dict[f'embedding_{i:05d}'] = emb
        # Convert dict to numpy array for storage
        save_dict[f'metadata_{i:05d}'] = np.array(meta, dtype=object)
    
    # Add count for easy loading
    save_dict['count'] = np.array(len(embeddings_list))
    
    np.savez_compressed(output_path, **save_dict)
    
    print(f"Saved {len(embeddings_list)} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train HGT + Temporal Transformer encoder"
    )
    
    # Data arguments
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=float, default=0.8)
    
    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-spatial-layers", type=int, default=3)
    parser.add_argument("--num-temporal-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training arguments
    parser.add_argument("--loss-type", choices=["reconstruction", "contrastive"], default="contrastive")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output arguments
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings after training")
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    index_path = args.data_dir / "dataset_index.json"
    index_data = json.loads(index_path.read_text())
    graph_files = [Path(e["graph_file"]) for e in index_data["entries"]]
    
    print(f"Found {len(graph_files)} graphs")
    
    # Get node feature dim from first graph
    sample_data = np.load(graph_files[0], allow_pickle=True)
    node_feature_dim = sample_data["node_features"].shape[1]
    print(f"Node feature dim: {node_feature_dim}")
    
    # Create dataset (no target selection for encoder training)
    dataset = TemporalGraphDataset(graph_files, target_indices=None)
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = HGTTemporalTransformer(
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    print(f"\nTraining with {args.loss_type} loss...\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.loss_type)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, args.save_dir / "best_encoder.pt")
        
        scheduler.step()
    
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        print("\nGenerating embeddings for all graphs...")
        model.load_state_dict(torch.load(args.save_dir / "best_encoder.pt")['model_state_dict'])
        save_embeddings(model, DataLoader(dataset, batch_size=1), args.save_dir / "embeddings.npz", args.device)
    
    print(f"\nEncoder saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
