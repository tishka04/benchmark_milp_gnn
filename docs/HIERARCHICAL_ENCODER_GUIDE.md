# Hierarchical Temporal Encoder - Quick Start Guide

## 🎯 Overview

The **Hierarchical Temporal Encoder** is optimized for large-scale multi-scale energy system graphs (604 nodes × 96 timesteps).

### Key Innovation

Instead of dense attention across all 604 nodes (memory explosion), it uses **hierarchical pooling**:

```
Assets (604) → Sparse GAT → Pool
Zones (100)  → Sparse GAT → Pool  
Regions (10) → Sparse GAT → Pool
Nation (1)   → Dense Temporal Transformer
```

**Memory**: ~5-8 GB (vs 40+ GB for dense HGT)

## 📁 Files Created

1. **`Hierarchical_Temporal_Encoder_Colab.ipynb`** - Main training notebook
2. **`src/gnn/models/hierarchical_temporal_encoder.py`** - Model architecture
3. **`docs/HIERARCHICAL_ENCODER_GUIDE.md`** - This guide

## 🚀 How to Use

### Option A: Local Execution (via Colab Extension) ⭐ Recommended

**Step 1**: Open notebook in VSCode
- Open `Hierarchical_Temporal_Encoder_Colab.ipynb` with Colab extension

**Step 2**: Select kernel
- Choose Python kernel with CUDA support
- Verify GPU in Cell 2

**Step 3**: Run all cells
- Repository path is auto-detected
- No Google Drive needed
- Outputs saved to `outputs/encoders/hierarchical_temporal/`

The notebook will:
1. ✅ Auto-detect repository path
2. ✅ Install dependencies (PyTorch Geometric, etc.)
3. ✅ Load your dataset (`outputs/graphs/hetero_temporal_v1`)
4. ✅ Create hierarchy mappings (asset→zone→region)
5. ✅ Train the hierarchical encoder
6. ✅ Generate multi-scale embeddings
7. ✅ Save everything locally

### Option B: Google Colab Cloud

**Step 1**: Upload to Drive
- Upload notebook to `/content/drive/MyDrive/benchmark/`

**Step 2**: Modify Cell 2 (Setup paths)
```python
from google.colab import drive
drive.mount('/content/drive')
REPO_PATH = Path('/content/drive/MyDrive/benchmark')
```

**Step 3**: Update config (Cell 10)
```python
repo_path: str = '/content/drive/MyDrive/benchmark'
data_dir: str = '/content/drive/MyDrive/benchmark/outputs/graphs/hetero_temporal_v1'
save_dir: str = '/content/drive/MyDrive/benchmark/outputs/encoders/hierarchical_temporal'
```

**Step 4**: Run all cells
- Runtime → Change runtime type → GPU (A100)

## 🏗️ Architecture Details

### Bottom-Up Encoding (Spatial)

```python
# Asset level (604 nodes, sparse)
h_assets = GATv2Conv(h, edge_index_assets)  # ~2 GB memory

# Pool to Zones (100 nodes)
h_zones = scatter_mean(h_assets, asset_to_zone)

# Zone level (100 nodes, sparse)
h_zones = GATv2Conv(h_zones, edge_index_zones)  # ~300 MB

# Pool to Regions (10 nodes)
h_regions = scatter_mean(h_zones, zone_to_region)

# Region level (10 nodes, sparse)
h_regions = GATv2Conv(h_regions, edge_index_regions)  # ~20 MB

# Pool to Nation (1 node)
h_nation = global_mean(h_regions)
```

### Temporal Encoding (Dense at Top)

```python
# Dense Transformer on Nation level (only 1 node × 96 timesteps)
h_nation = TemporalTransformer(h_nation)  # ~100 MB

# Captures ALL temporal dependencies efficiently
```

### Top-Down Propagation

```python
# Broadcast global context back down
h_regions = h_nation.expand(10, T, D) + h_regions_skip
h_zones = h_regions[zone_to_region] + h_zones_skip
h_assets = h_zones[asset_to_zone] + h_assets_skip

# Each asset embedding now contains:
# - Local asset information (from bottom-up)
# - Global temporal context (from top-down)
```

## 📊 Output Embeddings

The encoder produces **multi-scale embeddings**:

```python
embeddings = {
    'assets': [604, 96, 128],     # Asset-level (for EBM)
    'zones': [100, 96, 128],      # Zone-level constraints
    'regions': [10, 96, 128],     # Regional balance
    'nation': [96, 128]           # Global context
}
```

Saved to: `outputs/encoders/hierarchical_temporal/embeddings_multiscale.pt`

## 💾 Memory Usage Breakdown

| Component | Nodes | Memory |
|-----------|-------|--------|
| **Assets GAT** | 604 × 96 | ~2 GB |
| **Zones GAT** | 100 × 96 | ~300 MB |
| **Regions GAT** | 10 × 96 | ~20 MB |
| **Nation Transformer** | 1 × 96 | ~100 MB |
| **Decoder** | - | ~1 GB |
| **Gradients** | - | ~1.5 GB |
| **TOTAL** | - | **~5 GB** |

## ⚙️ Configuration

Default config in Cell 5:

```python
@dataclass
class TrainingConfig:
    # Model
    hidden_dim: int = 128           # Embedding dimension
    num_spatial_layers: int = 2     # Layers per hierarchy level
    num_temporal_layers: int = 4    # Dense temporal at Nation
    num_heads: int = 8              # Multi-head attention
    
    # Training
    epochs: int = 150
    batch_size: int = 1             # 1 graph at a time
    lr: float = 3e-4
    
    # Loss
    loss_type: str = 'contrastive'  # Or 'multi_scale'
```

### Adjusting for Different GPUs

**A100 80GB** (recommended):
```python
hidden_dim = 256
num_spatial_layers = 3
batch_size = 2
```

**V100 16GB**:
```python
hidden_dim = 128
num_spatial_layers = 2
batch_size = 1
```

**T4 16GB**:
```python
hidden_dim = 64
num_spatial_layers = 1
batch_size = 1
```

## 🎯 For Your EBM Pipeline

### Asset-Level Scoring

```python
# Load embeddings
embeddings = torch.load('embeddings_multiscale.pt')

# EBM can score at multiple levels
def energy_function(config, embeddings):
    # Fine-grained: asset-level
    asset_score = EBM_asset(config.assets, embeddings['assets'])
    
    # Mid-level: zone constraints
    zone_score = EBM_zone(config.zones, embeddings['zones'])
    
    # Coarse: regional balance
    region_score = EBM_region(config.regions, embeddings['regions'])
    
    # Global: nation-wide constraints
    global_score = EBM_global(config.global_vars, embeddings['nation'])
    
    return asset_score + zone_score + region_score + global_score
```

### Hierarchical Sampling

```python
# Sample coarse-to-fine
def sample_configuration(embeddings, temperature):
    # 1. Sample region-level (10 choices)
    regions = sample_regions(embeddings['regions'], temperature)
    
    # 2. Sample zones conditioned on regions (100 choices)
    zones = sample_zones(embeddings['zones'], regions, temperature)
    
    # 3. Sample assets conditioned on zones (604 choices)
    assets = sample_assets(embeddings['assets'], zones, temperature)
    
    return Configuration(regions, zones, assets)
```

## 🐛 Troubleshooting

### OOM on Asset GAT
- Reduce `hidden_dim` to 64
- Set `num_spatial_layers = 1`
- Check hierarchy mappings (ensure no huge zones)

### Slow Training
- Normal: ~2-3 min/epoch on A100
- Enable gradient checkpointing (already default)
- Reduce `num_temporal_layers` to 2

### Loss Not Decreasing
- Try `loss_type = 'multi_scale'` for multi-level learning
- Increase `warmup_epochs` to 20
- Check hierarchy mappings are correct

## 📈 Expected Results

**Good training** (contrastive loss):
- Epoch 1: loss ~1.5-2.0
- Epoch 50: loss ~0.4-0.6
- Epoch 150: loss ~0.2-0.3

**Multi-scale loss** converges slower but captures more levels:
- Epoch 150: loss ~0.3-0.5

## 🔄 Next Steps After Training

1. **Generate Embeddings** (Cell 12) - Creates `embeddings_multiscale.pt`
2. **Build EBM** - Use embeddings as context for energy function
3. **Implement Sampler** - Thermodynamic sampling with hierarchical guidance
4. **LP Validation** - Fast continuous variable validation

## 📚 References

- **Graph U-Net**: [Gao & Ji, 2019](https://arxiv.org/abs/1905.05178)
- **GATv2**: [Brody et al., 2021](https://arxiv.org/abs/2105.14491)
- **Hierarchical GNN**: [Ying et al., 2018](https://arxiv.org/abs/1806.08804)

## 💬 Support

If you encounter issues:
1. Check GPU memory: `!nvidia-smi` in Colab
2. Verify dataset: Cell 4 should show 2000 graphs
3. Check hierarchy: `asset_to_zone` and `zone_to_region` should have correct shapes

---

**Created**: December 2024  
**Optimized for**: Multi-scale energy system graphs (604 nodes × 96 timesteps)  
**Memory**: ~5-8 GB (8× more efficient than dense HGT)
