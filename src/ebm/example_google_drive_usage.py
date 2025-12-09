"""
Example: Training EBM with Google Drive embeddings.

This script shows how to use the 40GB embeddings file from Google Drive
(content/drive/MyDrive/benchmark/outputs/encoders/hierchical_temporal/embeddings_multiscale_full.pt)
"""

import torch
from torch.utils.data import DataLoader

from src.ebm import (
    StructuredEnergyModel,
    GibbsSampler,
    MILPBinaryDataset,
    EBMTrainer,
    GoogleDriveEmbeddingLoader,
    collate_ebm_batch,
)


def train_with_google_drive_embeddings():
    """
    Train EBM using pre-computed embeddings from Google Drive.
    """
    
    # Configuration
    EMBEDDING_FILE = "benchmark/outputs/encoders/hierchical_temporal/embeddings_multiscale_full.pt"
    SCENARIOS_DIR = "outputs/scenarios_v1"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Training EBM with Google Drive Embeddings")
    print("=" * 60)
    
    # Step 1: Create dataset with Google Drive embeddings
    print("\n1. Loading dataset with Google Drive embeddings...")
    dataset = MILPBinaryDataset(
        scenarios_dir=SCENARIOS_DIR,
        embedding_file=EMBEDDING_FILE,  # ← Path to your 40GB .pt file
        embedding_loading_mode='lazy',   # ← Lazy loading to avoid OOM
        temporal=False,                  # Aggregate temporal embeddings
        temporal_aggregation='mean',     # How to aggregate [T, 128] -> [128]
        device=DEVICE,
    )
    
    print(f"Dataset loaded: {len(dataset)} scenarios")
    
    # Step 2: Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_ebm_batch,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with lazy loading
    )
    
    # Step 3: Test loading a sample
    print("\n2. Testing sample loading...")
    sample = dataset[0]
    print(f"   Binary variables shape: {sample['u'].shape}")
    print(f"   Embedding shape: {sample['h'].shape}")
    print(f"   Embedding is dummy: {sample['h'].mean() < 0.1 and sample['h'].std() < 1.1}")
    
    # Step 4: Create model
    print("\n3. Creating EBM model...")
    dim_u = sample['u'].shape[0]
    dim_h = sample['h'].shape[0]
    
    model = StructuredEnergyModel(
        dim_u=dim_u,
        dim_h=dim_h,
        hidden_dims=[256, 256, 64],
        use_quadratic=True,
        quadratic_rank=16,
    )
    model = model.to(DEVICE)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Create sampler
    print("\n4. Creating Gibbs sampler...")
    sampler = GibbsSampler(
        energy_model=model,
        num_steps=50,
        temperature=1.0,
        device=DEVICE,
    )
    
    # Step 6: Create trainer
    print("\n5. Creating trainer...")
    trainer = EBMTrainer(
        model=model,
        sampler=sampler,
        device=DEVICE,
        use_wandb=False,
    )
    
    # Step 7: Train
    print("\n6. Starting training...")
    print("=" * 60)
    
    for epoch in range(5):  # Just 5 epochs for demo
        metrics = trainer.train_epoch(dataloader, num_negative_samples=1)
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Energy (pos): {metrics['energy_pos']:.4f}")
        print(f"  Energy (neg): {metrics['energy_neg']:.4f}")
        print(f"  Energy gap: {metrics['energy_gap']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def inspect_embedding_file():
    """
    Inspect the structure of the Google Drive embedding file.
    """
    
    EMBEDDING_FILE = "benchmark/outputs/encoders/hierchical_temporal/embeddings_multiscale_full.pt"
    
    print("=" * 60)
    print("Inspecting Google Drive Embedding File")
    print("=" * 60)
    
    # Create loader
    loader = GoogleDriveEmbeddingLoader(
        drive_path=EMBEDDING_FILE,
        loading_mode='lazy',
        device='cpu',
    )
    
    print(f"\nTotal scenarios: {len(loader)}")
    print(f"Scenario IDs (first 10): {loader.scenario_ids[:10]}")
    
    # Load a sample embedding
    if len(loader) > 0:
        sample_id = loader.scenario_ids[0]
        print(f"\nLoading sample embedding: {sample_id}")
        
        emb = loader.get_embedding(sample_id)
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding dtype: {emb.dtype}")
        print(f"Embedding stats:")
        print(f"  Mean: {emb.mean():.4f}")
        print(f"  Std: {emb.std():.4f}")
        print(f"  Min: {emb.min():.4f}")
        print(f"  Max: {emb.max():.4f}")
        
        # Test temporal aggregation
        if emb.ndim == 2:
            print(f"\nTemporal embedding detected: {emb.shape}")
            
            for method in ['mean', 'max', 'last', 'first']:
                agg_emb = loader.aggregate_temporal(emb, method=method)
                print(f"  {method:5s} aggregation: shape={agg_emb.shape}, mean={agg_emb.mean():.4f}")


def compare_loading_modes():
    """
    Compare full vs lazy loading modes.
    """
    
    EMBEDDING_FILE = "benchmark/outputs/encoders/hierchical_temporal/embeddings_multiscale_full.pt"
    
    print("=" * 60)
    print("Comparing Loading Modes")
    print("=" * 60)
    
    # Test lazy loading
    print("\n1. Lazy Loading Mode:")
    import time
    
    start = time.time()
    loader_lazy = GoogleDriveEmbeddingLoader(
        drive_path=EMBEDDING_FILE,
        loading_mode='lazy',
    )
    init_time = time.time() - start
    
    print(f"   Initialization time: {init_time:.2f}s")
    
    start = time.time()
    emb = loader_lazy.get_embedding(loader_lazy.scenario_ids[0])
    load_time = time.time() - start
    
    print(f"   First access time: {load_time:.2f}s")
    
    # Note: Full loading would take much longer and require ~40GB RAM
    print("\n2. Full Loading Mode:")
    print("   ⚠️  Skipped - requires ~40GB RAM")
    print("   Would load entire file at initialization")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='inspect',
                       choices=['inspect', 'train', 'compare'],
                       help='What to run')
    args = parser.parse_args()
    
    if args.mode == 'inspect':
        inspect_embedding_file()
    elif args.mode == 'train':
        train_with_google_drive_embeddings()
    elif args.mode == 'compare':
        compare_loading_modes()
