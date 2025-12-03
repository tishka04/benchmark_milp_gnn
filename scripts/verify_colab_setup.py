"""
Verification Script for Google Colab Setup

Run this script to verify that all dependencies and data are correctly set up
before starting the full HGT_Transformer training.

Usage (in Colab):
    !python /content/drive/MyDrive/benchmark/verify_colab_setup.py
"""

import sys
from pathlib import Path
import json

def check_imports():
    """Verify all required packages are installed."""
    print("=" * 80)
    print("📦 Checking Package Imports...")
    print("=" * 80)
    
    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name:25s} - OK")
        except ImportError:
            print(f"❌ {name:25s} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Please install them in the notebook.")
        return False
    else:
        print("\n✅ All required packages are installed!\n")
        return True


def check_cuda():
    """Check CUDA availability and GPU info."""
    print("=" * 80)
    print("🖥️  Checking GPU/CUDA...")
    print("=" * 80)
    
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Please enable GPU in Colab.")
        print("   Go to: Runtime → Change runtime type → GPU")
        return False
    
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU
    try:
        x = torch.randn(100, 100).cuda()
        y = x @ x.t()
        print("✅ GPU computation test passed")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False
    
    print()
    return True


def check_repository_structure():
    """Check if repository is properly mounted and structured."""
    print("=" * 80)
    print("📁 Checking Repository Structure...")
    print("=" * 80)
    
    # Try common mount locations
    possible_paths = [
        Path('/content/drive/MyDrive/benchmark'),
        Path('/content/drive/My Drive/benchmark'),
        Path('/content/benchmark'),
    ]
    
    repo_path = None
    for path in possible_paths:
        if path.exists():
            repo_path = path
            print(f"✅ Repository found at: {path}")
            break
    
    if repo_path is None:
        print("❌ Repository not found at any expected location!")
        print("   Expected locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return False, None
    
    # Check key directories
    required_dirs = [
        'src',
        'src/gnn',
        'src/gnn/models',
        'outputs',
        'outputs/graphs',
        'config',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name:30s} - OK")
        else:
            print(f"❌ {dir_name:30s} - MISSING")
            all_exist = False
    
    print()
    return all_exist, repo_path


def check_dataset(repo_path):
    """Check if dataset exists and is valid."""
    print("=" * 80)
    print("📊 Checking Dataset...")
    print("=" * 80)
    
    data_dir = repo_path / 'outputs' / 'graphs' / 'hetero_temporal_v1'
    
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    print(f"✅ Dataset directory exists: {data_dir}")
    
    # Check index file
    index_path = data_dir / 'dataset_index.json'
    if not index_path.exists():
        print(f"❌ Dataset index not found: {index_path}")
        return False
    
    print(f"✅ Dataset index found")
    
    # Load and validate index
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        num_graphs = len(index_data['entries'])
        print(f"✅ Number of graphs in index: {num_graphs}")
        
        if num_graphs == 0:
            print("❌ No graphs in dataset!")
            return False
        
        # Check first graph file
        first_entry = index_data['entries'][0]
        first_graph_path = data_dir / first_entry['graph_file']
        
        if not first_graph_path.exists():
            print(f"❌ First graph file not found: {first_graph_path}")
            return False
        
        print(f"✅ First graph file exists: {first_graph_path.name}")
        
        # Load and inspect first graph
        import numpy as np
        sample = np.load(first_graph_path, allow_pickle=True)
        
        print(f"\n📈 Sample graph statistics:")
        print(f"   - Node features shape: {sample['node_features'].shape}")
        print(f"   - Edge index shape: {sample['edge_index'].shape}")
        print(f"   - Node types shape: {sample['node_types'].shape}")
        print(f"   - Edge types shape: {sample['edge_types'].shape}")
        
        meta = sample['meta'].item()
        print(f"   - N_base (base nodes): {meta['N_base']}")
        print(f"   - T (timesteps): {meta['T']}")
        print(f"   - Total nodes: {meta['N_base'] * meta['T']}")
        
        node_feature_dim = sample['node_features'].shape[1]
        print(f"\n✅ Node feature dimension: {node_feature_dim}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    print()
    return True


def check_model_imports(repo_path):
    """Check if model can be imported."""
    print("=" * 80)
    print("🧠 Checking Model Imports...")
    print("=" * 80)
    
    # Add repo to path
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    
    try:
        from src.gnn.models.temporal_hetero_gnn import (
            HGTTemporalTransformer,
            TemporalGraphDataset,
        )
        print("✅ HGTTemporalTransformer imported successfully")
        print("✅ TemporalGraphDataset imported successfully")
        
        # Test model creation
        import torch
        model = HGTTemporalTransformer(
            node_feature_dim=10,
            hidden_dim=64,
            num_spatial_layers=2,
            num_temporal_layers=1,
            num_heads=4,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model creation successful")
        print(f"   - Test model parameters: {total_params:,}")
        
        del model
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_data_loading(repo_path):
    """Test loading a single batch."""
    print("=" * 80)
    print("🔄 Testing Data Loading...")
    print("=" * 80)
    
    try:
        import torch
        from src.gnn.models.temporal_hetero_gnn import TemporalGraphDataset
        from torch_geometric.loader import DataLoader
        import json
        import numpy as np
        
        # Load dataset
        data_dir = repo_path / 'outputs' / 'graphs' / 'hetero_temporal_v1'
        index_path = data_dir / 'dataset_index.json'
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        graph_files = [data_dir / e["graph_file"] for e in index_data["entries"][:5]]
        
        # Create dataset
        dataset = TemporalGraphDataset(graph_files, target_indices=None)
        print(f"✅ Dataset created with {len(dataset)} graphs")
        
        # Create loader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        print(f"✅ DataLoader created")
        
        # Load one batch
        batch = next(iter(loader))
        print(f"✅ Successfully loaded batch")
        print(f"   - Batch x shape: {batch.x.shape}")
        print(f"   - Batch edge_index shape: {batch.edge_index.shape}")
        print(f"   - Batch edge_type shape: {batch.edge_type.shape}")
        
        # Move to GPU
        if torch.cuda.is_available():
            batch = batch.to('cuda')
            print(f"✅ Successfully moved batch to GPU")
        
        del batch, loader, dataset
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def estimate_memory_requirements(repo_path):
    """Estimate memory requirements for training."""
    print("=" * 80)
    print("💾 Estimating Memory Requirements...")
    print("=" * 80)
    
    try:
        import numpy as np
        import json
        
        data_dir = repo_path / 'outputs' / 'graphs' / 'hetero_temporal_v1'
        index_path = data_dir / 'dataset_index.json'
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Sample first graph
        first_graph_path = data_dir / index_data['entries'][0]['graph_file']
        sample = np.load(first_graph_path, allow_pickle=True)
        
        meta = sample['meta'].item()
        N_base = meta['N_base']
        T = meta['T']
        num_nodes = N_base * T
        num_edges = sample['edge_index'].shape[1]
        node_feature_dim = sample['node_features'].shape[1]
        
        # Estimate model size (for hidden_dim=256, 4+3 layers, 16 heads)
        hidden_dim = 256
        num_spatial = 4
        num_temporal = 3
        num_heads = 16
        
        # Rough parameter count
        input_proj = node_feature_dim * hidden_dim * 5  # 5 node types
        spatial_params = num_spatial * (hidden_dim ** 2) * 10 * num_heads  # 10 edge types
        temporal_params = num_temporal * (hidden_dim ** 2) * 4 * num_heads  # Transformer
        total_params = input_proj + spatial_params + temporal_params
        
        # Memory estimates (in GB)
        model_memory = total_params * 4 / 1e9  # fp32
        batch_size = 4
        batch_memory = batch_size * num_nodes * hidden_dim * 4 / 1e9
        optimizer_memory = total_params * 8 / 1e9  # Adam states
        
        total_memory = model_memory + batch_memory + optimizer_memory + 2  # +2GB buffer
        
        print(f"Graph statistics (per sample):")
        print(f"  - Nodes: {num_nodes:,} ({N_base} base × {T} timesteps)")
        print(f"  - Edges: {num_edges:,}")
        print(f"  - Node features: {node_feature_dim}")
        
        print(f"\nEstimated model (hidden_dim={hidden_dim}):")
        print(f"  - Parameters: ~{total_params/1e6:.1f}M")
        print(f"  - Model memory: ~{model_memory:.2f} GB")
        print(f"  - Batch memory (size={batch_size}): ~{batch_memory:.2f} GB")
        print(f"  - Optimizer memory: ~{optimizer_memory:.2f} GB")
        print(f"  - Total estimate: ~{total_memory:.2f} GB")
        
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n{'✅' if total_memory < gpu_memory else '⚠️ '} Available GPU memory: {gpu_memory:.2f} GB")
            
            if total_memory > gpu_memory:
                print(f"\n⚠️  WARNING: Estimated memory exceeds GPU capacity!")
                print(f"   Consider reducing:")
                print(f"   - hidden_dim (256 → 128)")
                print(f"   - batch_size (4 → 2)")
                print(f"   - num_layers (4+3 → 3+2)")
        
    except Exception as e:
        print(f"❌ Memory estimation failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("🔍 GOOGLE COLAB SETUP VERIFICATION")
    print("=" * 80 + "\n")
    
    checks = []
    
    # 1. Check imports
    checks.append(("Package Imports", check_imports()))
    
    # 2. Check CUDA
    checks.append(("GPU/CUDA", check_cuda()))
    
    # 3. Check repository
    repo_ok, repo_path = check_repository_structure()
    checks.append(("Repository Structure", repo_ok))
    
    if repo_path and repo_ok:
        # 4. Check dataset
        checks.append(("Dataset", check_dataset(repo_path)))
        
        # 5. Check model imports
        checks.append(("Model Imports", check_model_imports(repo_path)))
        
        # 6. Test data loading
        checks.append(("Data Loading", test_data_loading(repo_path)))
        
        # 7. Estimate memory
        estimate_memory_requirements(repo_path)
    
    # Summary
    print("=" * 80)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:25s} {status}")
    
    all_passed = all(result for _, result in checks)
    
    print("=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED! Ready for training.")
        print("\nYou can now run the training notebook:")
        print("  HGT_Transformer_Training_Colab.ipynb")
    else:
        print("❌ SOME CHECKS FAILED! Please fix the issues above.")
        print("\nRefer to COLAB_TRAINING_GUIDE.md for troubleshooting.")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
