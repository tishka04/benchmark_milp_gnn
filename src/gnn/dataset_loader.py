from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np


def load_index(index_path: Path) -> Dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def iter_split(index: Dict, split: str | float, train_split: float = 0.7, val_split: float = 0.15) -> Iterator[Dict]:
    """
    Iterate over entries in a split.
    
    Args:
        index: Dataset index dictionary
        split: Either a split name (str) or a split fraction (float)
        train_split: Fraction for training if doing automatic splitting (default: 0.7)
        val_split: Fraction for validation if doing automatic splitting (default: 0.15)
    """
    # Check if we have predefined splits
    if "splits" in index and isinstance(split, str):
        entries = index.get("splits", {}).get(split)
        if entries is None:
            raise KeyError(f"Split '{split}' not found in index. Available splits: {list(index.get('splits', {}).keys())}")
        for entry in entries:
            yield entry
    # Otherwise, do automatic splitting based on fractions
    elif "entries" in index:
        all_entries = index["entries"]
        n_total = len(all_entries)
        
        # Determine split indices based on the split type
        if isinstance(split, str):
            # Map string to default fractions
            if split == "train":
                start_idx = 0
                end_idx = int(n_total * train_split)
            elif split == "val":
                start_idx = int(n_total * train_split)
                end_idx = start_idx + int(n_total * val_split)
            elif split == "test":
                start_idx = int(n_total * (train_split + val_split))
                end_idx = n_total
            else:
                raise KeyError(f"Unknown split name '{split}'. Use 'train', 'val', or 'test', or provide split fractions.")
        else:
            # If float, it should match one of the fractions (this is a fallback)
            raise ValueError(f"When using automatic splitting, provide split names ('train', 'val', 'test'), not fractions directly.")
        
        for entry in all_entries[start_idx:end_idx]:
            yield entry
    else:
        raise RuntimeError("Index must contain either 'splits' (predefined) or 'entries' (for automatic splitting)")


def load_graph(entry: Dict) -> Dict[str, np.ndarray]:
    graph_path = Path(entry["graph_file"])
    with np.load(graph_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_split_graphs(index: Dict, split: str) -> Iterator[Dict[str, np.ndarray]]:
    for entry in iter_split(index, split):
        yield load_graph(entry)
