# HGT + Temporal Transformer Encoder for EBM

## 🎯 Overview

This document describes the **encoder-only architecture** that combines:
1. **HGT (Heterogeneous Graph Transformer)** - Spatial encoding
2. **Temporal Transformer** - Temporal reasoning
3. **No decoder** - Outputs embeddings for EBM training

---

## 🏗️ Architecture

```
Input: Temporal Heterogeneous Graph
    ↓
┌─────────────────────────────────────────┐
│ PHASE 1: SPATIAL ENCODING (HGT)        │
│                                         │
│ Per-node-type projection                │
│ Node features → hidden_dim              │
│                                         │
│ HGT Layer 1 (multi-head attention)      │
│   - Edge-type-aware                     │
│   - Node-type-aware                     │
│                                         │
│ HGT Layer 2                             │
│ HGT Layer 3                             │
│                                         │
│ Output: h [N_base*T, hidden_dim]        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ UNROLL: Reshape to [N_base, T, D]      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ PHASE 2: TEMPORAL ENCODING (Transformer)│
│                                         │
│ Add positional encoding (sinusoidal)    │
│                                         │
│ Transformer Layer 1                     │
│   - Self-attention across time          │
│   - FFN (4x expansion)                  │
│                                         │
│ Transformer Layer 2                     │
│                                         │
│ Output: embeddings [N_base, T, D]       │
└─────────────────────────────────────────┘
    ↓
Final Embeddings (NO DECODER)
[N_base, T, hidden_dim]
```

---

## 📊 Key Differences from HGT

| Feature | HGT (with decoder) | HGT + Transformer (encoder-only) |
|---------|-------------------|----------------------------------|
| **Architecture** | HGT → MLP decoder | HGT → Temporal Transformer |
| **Output** | Predictions [N, 13] | Embeddings [N_base, T, D] |
| **Temporal** | Implicit (via temporal edges) | Explicit (Transformer) |
| **Training** | Supervised (MSE loss) | Self-supervised (contrastive) |
| **Use case** | Direct prediction | EBM training |
| **Parameters** | ~1.5M | ~2.5M |

---

## 🔧 How to Use

### **1. Pre-train the Encoder**

```bash
python -m src.gnn.pretrain_encoder --data-dir outputs/graphs/hetero_temporal_v1 --hidden-dim 1 --num-spatial-layers 1 --num-temporal-layers 1 --num-heads 1 --loss-type contrastive --epochs 2 --batch-size 1 --lr 0.001 --save-dir outputs/encoders/hgt_transformer_v1 --save-embeddings
```

python -m src.gnn.pretrain_encoder `
    --data-dir outputs/graphs/hetero_temporal_v1 `
    --hidden-dim 128 `
    --num-spatial-layers 3 `
    --num-temporal-layers 2 `
    --num-heads 8 `
    --loss-type contrastive `
    --epochs 100 `
    --batch-size 1 `
    --lr 0.001 `
    --save-dir outputs/encoders/hgt_transformer_cpu `
    --save-embeddings `
    --device cpu

**Training options:**
- `--loss-type contrastive` - InfoNCE loss (recommended)
- `--loss-type reconstruction` - Autoencoder-style loss

**What it does:**
- Trains encoder without task-specific labels
- Learns rich spatio-temporal representations
- Saves encoder checkpoint to `best_encoder.pt`
- Optionally saves embeddings to `embeddings.npz`

---

### **2. Generate Embeddings for EBM**

After pre-training, generate embeddings for all scenarios:

```python
import torch
import numpy as np
from src.gnn.models.temporal_hetero_gnn import HGTTemporalTransformer

# Load pre-trained encoder
model = HGTTemporalTransformer(
    node_feature_dim=node_feature_dim,
    hidden_dim=128,
    num_spatial_layers=3,
    num_temporal_layers=2,
    num_heads=8,
)

checkpoint = torch.load("outputs/encoders/hgt_transformer_v1/best_encoder.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load graph
data = load_graph("outputs/graphs/hetero_temporal_v1/scenario_00001.npz")

# Generate embeddings
with torch.no_grad():
    embeddings = model(
        x=data.x,
        edge_index=data.edge_index,
        edge_type=data.edge_type,
        node_type=data.node_type,
        N_base=data.N_base,
        T=data.T,
        return_sequence=True,  # [N_base, T, hidden_dim]
    )

# embeddings shape: [N_base, T, 128]
# Example: [150 nodes, 96 timesteps, 128 dims] = ~1.8M floats per scenario
```

---

### **3. Train EBM on Embeddings**

```python
# Pseudocode for EBM training

class CommitmentEBM(nn.Module):
    """Energy-based model for commitment decisions."""
    
    def __init__(self, embedding_dim=128, num_assets=150):
        super().__init__()
        self.energy_network = nn.Sequential(
            nn.Linear(embedding_dim * 96, 512),  # Flatten time
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Scalar energy
        )
    
    def forward(self, embeddings, commitment):
        """
        Args:
            embeddings: [N_base, T, D] from encoder
            commitment: [N_assets, T] binary commitment decisions
        
        Returns:
            energy: Scalar energy value (lower = better)
        """
        # Combine embeddings with commitment
        h = embeddings.flatten()  # Flatten to vector
        u = commitment.flatten()
        
        # Compute energy
        energy = self.energy_network(torch.cat([h, u]))
        return energy


# Training loop
for scenario in dataset:
    # Get embeddings from pre-trained encoder
    embeddings = encoder.get_embeddings(scenario.graph)
    
    # Oracle commitment (low energy)
    u_oracle = scenario.oracle_commitment
    energy_oracle = ebm(embeddings, u_oracle)
    
    # Negative samples (high energy)
    u_negative = sample_negative_commitments(u_oracle)
    energy_negative = ebm(embeddings, u_negative)
    
    # Contrastive loss
    loss = -torch.log(
        torch.exp(-energy_oracle) / 
        (torch.exp(-energy_oracle) + torch.exp(-energy_negative))
    )
    
    loss.backward()
    optimizer.step()
```

---

## 💡 Design Rationale

### **Why HGT first?**
- Encodes spatial structure (Nation → Region → Zone → Asset)
- Handles heterogeneous node types and edge types
- Captures transmission constraints, weather influence

### **Why Temporal Transformer after?**
- Separates spatial and temporal reasoning
- Self-attention across time captures:
  - Ramping constraints
  - Storage dynamics (SOC continuity)
  - Demand patterns
  - Inter-temporal dependencies

### **Why no decoder?**
- EBM learns on embeddings, not predictions
- Embeddings are more expressive than predictions
- Allows EBM to learn custom energy landscape
- Decouples representation learning from task-specific prediction

---

## 📈 Expected Performance

### **Embedding quality metrics:**

After pre-training, embeddings should:
- ✅ Cluster similar scenarios (k-means, t-SNE)
- ✅ Preserve temporal structure (adjacent timesteps close)
- ✅ Separate feasible vs infeasible commitments
- ✅ Correlate with system cost

### **Training time:**

| Phase | Time (GPU) | Output |
|-------|-----------|--------|
| **Pre-train encoder** | 6-8 hours | Encoder checkpoint |
| **Generate embeddings** | 30-60 min | 2000 × [N_base, T, 128] |
| **Train EBM** | 2-4 hours | EBM checkpoint |

---

## 🔗 Integration with Methodology

This fits into **Phase 5 (EBM)** of your methodology:

```
Phase 1: Generate scenarios ✅
Phase 2: Solve with MILP ✅
Phase 3: Build graphs ✅
Phase 4: [NEW] Pre-train HGT + Transformer encoder 🆕
Phase 5: Train EBM on embeddings 🆕
Phase 6: Thermodynamic sampler (guided by EBM)
Phase 7: Worker LP (dispatch given commitment)
Phase 8: End-to-end pipeline
```

---

## 🎨 Alternative Configurations

### **Configuration 1: Deep Spatial, Shallow Temporal**
```bash
--num-spatial-layers 4 \
--num-temporal-layers 1
```
Good for: Complex spatial structure, simple temporal patterns

### **Configuration 2: Shallow Spatial, Deep Temporal**
```bash
--num-spatial-layers 2 \
--num-temporal-layers 3
```
Good for: Simple topology, complex temporal dynamics

### **Configuration 3: Large Hidden Dim**
```bash
--hidden-dim 256 \
--num-heads 16
```
Good for: Very expressive embeddings, large datasets

---

## 🚀 Quick Start

```bash
# 1. Pre-train encoder
python -m src.gnn.pretrain_encoder \
    --data-dir outputs/graphs/hetero_temporal_v1 \
    --hidden-dim 128 \
    --num-spatial-layers 3 \
    --num-temporal-layers 2 \
    --num-heads 8 \
    --loss-type contrastive \
    --epochs 100 \
    --batch-size 8 \
    --save-dir outputs/encoders/baseline \
    --save-embeddings

# 2. Embeddings are now ready for EBM training!
```

---

## 📝 Summary

**Input:** Temporal heterogeneous graph  
**Output:** Spatio-temporal embeddings `[N_base, T, hidden_dim]`  
**Purpose:** Generate rich representations for EBM training  
**Advantage:** Separates representation learning from task-specific prediction  

This is the **missing piece** (Phase 5) in your methodology! 🎯
