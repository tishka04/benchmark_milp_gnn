"""
Training script for Temporal Heterogeneous GNNs on grid dispatch prediction.

Example usage:
    python -m src.gnn.train_temporal_hetero \
        --data-dir outputs/temporal_graphs/supra \
        --model-type rgcn \
        --hidden-dim 128 \
        --num-layers 3 \
        --epochs 50
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from src.gnn.models.temporal_hetero_gnn import (
    create_temporal_hetero_model,
    TemporalGraphDataset,
)


def load_dataset_index(data_dir: Path) -> List[Path]:
    """Load graph file paths from dataset index."""
    index_path = data_dir / "dataset_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")
    
    index_data = json.loads(index_path.read_text())
    
    # Extract graph file paths
    graph_files = []
    for entry in index_data["entries"]:
        graph_file = Path(entry["graph_file"])
        if graph_file.exists():
            graph_files.append(graph_file)
        else:
            print(f"Warning: Graph file not found: {graph_file}")
    
    return graph_files


def get_feature_dims(sample_graph_path: Path) -> Tuple[int, int]:
    """Get node feature and label dimensions from a sample graph."""
    data = np.load(sample_graph_path, allow_pickle=True)
    
    node_feature_dim = data["node_features"].shape[1]
    
    if "node_labels" in data and data["node_labels"] is not None:
        # node_labels shape: [T, N, label_dim]
        label_dim = data["node_labels"].shape[2]
    else:
        label_dim = 0
    
    return node_feature_dim, label_dim


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        
        # Compute loss (only on nodes with labels)
        if batch.y is not None:
            # Apply mask if available (for zone-only labels)
            if hasattr(batch, 'label_mask') and batch.label_mask is not None:
                loss = criterion(pred[batch.label_mask], batch.y[batch.label_mask])
            else:
                loss = criterion(pred, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            
            # Compute loss
            if batch.y is not None:
                # Apply mask if available (for zone-only labels)
                if hasattr(batch, 'label_mask') and batch.label_mask is not None:
                    loss = criterion(pred[batch.label_mask], batch.y[batch.label_mask])
                else:
                    loss = criterion(pred, batch.y)
                total_loss += loss.item()
                num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal Heterogeneous GNN for grid dispatch prediction"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/temporal_graphs/supra"),
        help="Directory containing temporal graphs",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--target-vars",
        type=str,
        default="thermal,nuclear,solar,wind",
        help="Comma-separated list of target variables to predict",
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        choices=["rgcn", "separated"],
        default="rgcn",
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (number of graphs)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/gnn_runs/temporal_hetero"),
        help="Directory to save checkpoints and logs",
    )
    
    args = parser.parse_args()
    
    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Temporal Heterogeneous GNN Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    print(f"Device: {args.device}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    graph_files = load_dataset_index(args.data_dir)
    print(f"Found {len(graph_files)} graphs")
    
    if len(graph_files) == 0:
        print("Error: No graphs found!")
        return
    
    # Get dimensions from first graph
    node_feature_dim, full_label_dim = get_feature_dims(graph_files[0])
    print(f"Node feature dim: {node_feature_dim}")
    print(f"Full label dim: {full_label_dim}")
    
    # Map target variables to indices
    # Standard order: thermal, nuclear, solar, wind, hydro_release, hydro_ror, dr,
    #                 battery_charge, battery_discharge, pumped_charge, pumped_discharge,
    #                 net_import, unserved
    target_var_map = {
        "thermal": 0,
        "nuclear": 1,
        "solar": 2,
        "wind": 3,
        "hydro_release": 4,
        "hydro_ror": 5,
        "dr": 6,
        "battery_charge": 7,
        "battery_discharge": 8,
        "pumped_charge": 9,
        "pumped_discharge": 10,
        "net_import": 11,
        "unserved": 12,
    }
    
    target_vars = args.target_vars.split(",")
    target_indices = [target_var_map[v] for v in target_vars if v in target_var_map]
    output_dim = len(target_indices)
    
    print(f"Predicting {output_dim} variables: {target_vars}")
    print()
    
    # Create dataset
    dataset = TemporalGraphDataset(graph_files, target_indices=target_indices)
    
    # Split into train/val
    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print()
    
    # Create data loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print("Creating model...")
    model = create_temporal_hetero_model(
        model_type=args.model_type,
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, args.device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {current_lr:.6e}")
        
        # Record history
        is_best = val_loss < best_val_loss
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': float(current_lr),
            'is_best': is_best,
        })
        
        # Save training history
        history_path = args.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save best model
        if is_best:
            best_val_loss = val_loss
            checkpoint_path = args.save_dir / "best_model.pt"
            # Save args with explicit output_dim for evaluation compatibility
            args_dict = vars(args).copy()
            args_dict['output_dim'] = output_dim
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args_dict,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.save_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
