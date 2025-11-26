"""
Quickstart example: Training a Temporal Heterogeneous GNN

This script demonstrates how to:
1. Load temporal supra-graphs
2. Create a temporal R-GCN model
3. Train on dispatch prediction
4. Evaluate results
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.gnn.models.temporal_hetero_gnn import TemporalRGCN, TemporalGraphDataset


def main():
    print("=" * 60)
    print("Temporal Heterogeneous GNN - Quickstart Example")
    print("=" * 60)
    print()
    
    # Configuration
    data_dir = Path("outputs/temporal_graphs/supra")
    hidden_dim = 128
    num_layers = 3
    num_epochs = 10  # Small for demo
    batch_size = 2
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Step 1: Load graphs
    print("Step 1: Loading temporal graphs...")
    graph_files = sorted(list(data_dir.glob("scenario_*.npz")))[:10]  # Use first 10 for demo
    
    if len(graph_files) == 0:
        print(f"Error: No graphs found in {data_dir}")
        print("Please run the temporal graph builder first:")
        print("  python -m src.gnn.build_hetero_graph_dataset \\")
        print("      outputs/scenarios_v1 \\")
        print("      outputs/scenarios_v1/reports \\")
        print("      outputs/temporal_graphs/supra \\")
        print("      --temporal --temporal-mode supra")
        return
    
    print(f"Found {len(graph_files)} graphs")
    
    # Inspect first graph
    sample_data = np.load(graph_files[0], allow_pickle=True)
    meta = sample_data["meta"].item()
    node_feature_dim = sample_data["node_features"].shape[1]
    label_dim = sample_data["node_labels"].shape[2] if "node_labels" in sample_data else 0
    
    print(f"  Base nodes (N): {meta['N_base']}")
    print(f"  Timesteps (T): {meta['T']}")
    print(f"  Total nodes (N×T): {len(sample_data['node_types'])}")
    print(f"  Total edges: {sample_data['edge_index'].shape[1]}")
    print(f"  Node feature dim: {node_feature_dim}")
    print(f"  Label dim: {label_dim}")
    print(f"  Temporal edges: {meta['temporal_edges']}")
    print()
    
    # Step 2: Create dataset
    print("Step 2: Creating dataset...")
    # Predict: thermal (0), nuclear (1), solar (2), wind (3)
    target_indices = [0, 1, 2, 3]
    output_dim = len(target_indices)
    
    dataset = TemporalGraphDataset(graph_files, target_indices=target_indices)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train size: {train_size}")
    print(f"  Val size: {val_size}")
    print(f"  Predicting: {output_dim} variables (thermal, nuclear, solar, wind)")
    print()
    
    # Step 3: Create model
    print("Step 3: Creating Temporal R-GCN model...")
    model = TemporalRGCN(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_edge_types=10,  # 7 spatial + 3 temporal
        num_layers=num_layers,
        dropout=0.1,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print()
    
    # Step 4: Setup training
    print("Step 4: Setting up training...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"  Loss: MSE")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Epochs: {num_epochs}")
    print()
    
    # Step 5: Training loop
    print("Step 5: Training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            
            # Only compute loss on nodes with valid labels (zone nodes)
            if batch.label_mask is not None:
                loss = criterion(pred[batch.label_mask], batch.y[batch.label_mask])
            else:
                loss = criterion(pred, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                
                # Only compute loss on nodes with valid labels
                if batch.label_mask is not None:
                    loss = criterion(pred[batch.label_mask], batch.y[batch.label_mask])
                else:
                    loss = criterion(pred, batch.y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}  |  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")
    
    print("-" * 60)
    print()
    
    # Step 6: Evaluate
    print("Step 6: Final evaluation...")
    model.eval()
    
    # Get predictions on validation set (only on zone nodes)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            
            # Only evaluate on nodes with valid labels
            if batch.label_mask is not None:
                all_preds.append(pred[batch.label_mask].cpu().numpy())
                all_targets.append(batch.y[batch.label_mask].cpu().numpy())
            else:
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics per variable
    var_names = ["Thermal", "Nuclear", "Solar", "Wind"]
    print("\nPer-variable MAE:")
    for i, var_name in enumerate(var_names):
        mae = np.mean(np.abs(preds[:, i] - targets[:, i]))
        rmse = np.sqrt(np.mean((preds[:, i] - targets[:, i]) ** 2))
        print(f"  {var_name:8s}: MAE = {mae:7.2f} MW, RMSE = {rmse:7.2f} MW")
    
    print()
    print("=" * 60)
    print("Quickstart complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Train on full dataset:")
    print("     python -m src.gnn.train_temporal_hetero \\")
    print("         --data-dir outputs/temporal_graphs/supra \\")
    print("         --hidden-dim 128 --epochs 50")
    print()
    print("  2. Try separated spatial-temporal model:")
    print("     python -m src.gnn.train_temporal_hetero \\")
    print("         --model-type separated --epochs 50")
    print()
    print("  3. Read the guide:")
    print("     docs/TRAINING_TEMPORAL_GNNS.md")


if __name__ == "__main__":
    main()
