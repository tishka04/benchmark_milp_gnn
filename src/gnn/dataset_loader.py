from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np


def load_index(index_path: Path) -> Dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def iter_split(index: Dict, split: str) -> Iterator[Dict]:
    entries = index.get("splits", {}).get(split)
    if entries is None:
        raise KeyError(f"Split '{split}' not found in index")
    for entry in entries:
        yield entry


def load_graph(entry: Dict) -> Dict[str, np.ndarray]:
    graph_path = Path(entry["graph_file"])
    with np.load(graph_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_split_graphs(index: Dict, split: str) -> Iterator[Dict[str, np.ndarray]]:
    for entry in iter_split(index, split):
        yield load_graph(entry)
