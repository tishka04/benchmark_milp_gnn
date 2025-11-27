"""
Utility functions to load pre-trained embeddings.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_embeddings(embeddings_path: Path) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load embeddings saved by pretrain_encoder.py
    
    Args:
        embeddings_path: Path to embeddings.npz file
    
    Returns:
        embeddings_list: List of embeddings [N_base, T, hidden_dim]
        metadata_list: List of metadata dicts
    """
    data = np.load(embeddings_path, allow_pickle=True)
    
    # Get count
    count = int(data['count'])
    
    embeddings_list = []
    metadata_list = []
    
    for i in range(count):
        emb = data[f'embedding_{i:05d}']
        meta = data[f'metadata_{i:05d}'].item()  # Convert 0-d array to dict
        
        embeddings_list.append(emb)
        metadata_list.append(meta)
    
    return embeddings_list, metadata_list


def load_single_embedding(embeddings_path: Path, index: int) -> Tuple[np.ndarray, Dict]:
    """
    Load a single embedding by index.
    
    Args:
        embeddings_path: Path to embeddings.npz file
        index: Index of embedding to load (0-based)
    
    Returns:
        embedding: [N_base, T, hidden_dim]
        metadata: Dict with N_base, T, shape
    """
    data = np.load(embeddings_path, allow_pickle=True)
    
    emb = data[f'embedding_{index:05d}']
    meta = data[f'metadata_{index:05d}'].item()
    
    return emb, meta


def get_embedding_stats(embeddings_path: Path) -> Dict:
    """
    Get statistics about saved embeddings without loading all of them.
    
    Args:
        embeddings_path: Path to embeddings.npz file
    
    Returns:
        stats: Dict with count, shapes, size info
    """
    data = np.load(embeddings_path, allow_pickle=True)
    
    count = int(data['count'])
    
    # Load first embedding to get shape info
    first_emb = data['embedding_00000']
    first_meta = data['metadata_00000'].item()
    
    stats = {
        'count': count,
        'example_shape': first_emb.shape,
        'N_base': first_meta['N_base'],
        'T': first_meta['T'],
        'hidden_dim': first_emb.shape[-1],
        'file_size_mb': embeddings_path.stat().st_size / (1024 * 1024),
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.gnn.load_embeddings <path_to_embeddings.npz>")
        sys.exit(1)
    
    embeddings_path = Path(sys.argv[1])
    
    print(f"Loading embeddings from {embeddings_path}")
    
    # Get stats
    stats = get_embedding_stats(embeddings_path)
    print("\nEmbedding Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Shape: {stats['example_shape']}")
    print(f"  N_base: {stats['N_base']}")
    print(f"  T: {stats['T']}")
    print(f"  Hidden dim: {stats['hidden_dim']}")
    print(f"  File size: {stats['file_size_mb']:.2f} MB")
    
    # Load first embedding as example
    print("\nLoading first embedding...")
    emb, meta = load_single_embedding(embeddings_path, 0)
    print(f"  Shape: {emb.shape}")
    print(f"  Min: {emb.min():.4f}, Max: {emb.max():.4f}, Mean: {emb.mean():.4f}")
    print(f"  Metadata: {meta}")
