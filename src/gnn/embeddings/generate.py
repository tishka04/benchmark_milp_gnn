"""
Generate and save embeddings for all v3 scenarios using a trained
Hierarchical Temporal Encoder.
"""

from __future__ import annotations

import json
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from tqdm.auto import tqdm

from src.gnn.embeddings.dataset import (
    TemporalGraphDatasetV3,
    get_hierarchy_from_batch,
)


@torch.no_grad()
def generate_embeddings(
    model: torch.nn.Module,
    config,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Generate embeddings for **every** graph in the v3 dataset index.

    Saves one ``.npz`` per scenario in *output_dir* containing:
        - ``assets``  : [N_assets, T, D]
        - ``zones``   : [N_zones, T, D]
        - ``regions`` : [N_regions, T, D]
        - ``nation``  : [T, D]
        - ``scenario_id`` : str

    Also writes a JSON manifest mapping scenario_id -> embedding file.

    Returns:
        manifest dict  ``{scenario_id: embedding_path}``
    """
    device = config.device
    repo_root = Path(config.repo_path)
    data_dir = Path(config.data_dir)

    if output_dir is None:
        output_dir = Path(config.save_dir) / "embeddings_v3"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset index
    index_path = data_dir / "dataset_index.json"
    with open(index_path, "r") as f:
        index_data = json.load(f)

    entries = index_data["entries"]
    graph_files = [
        repo_root / e["graph_file"].replace("\\", "/") for e in entries
    ]

    dataset = TemporalGraphDatasetV3(graph_files)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model.eval()
    manifest = {}

    print(f"Generating embeddings for {len(dataset)} scenarios ...")

    for idx, batch in enumerate(tqdm(loader, desc="Generating embeddings")):
        batch = batch.to(device)

        # Normalise
        if hasattr(batch, "x") and batch.x is not None:
            batch.x = (batch.x - train_mean) / train_std

        N_base = batch.N_base[0].item() if batch.N_base.dim() > 0 else batch.N_base.item()
        T = batch.T[0].item() if batch.T.dim() > 0 else batch.T.item()

        hierarchy = get_hierarchy_from_batch(batch, device)
        num_nodes = batch.x.size(0)
        edge_index_sl, _ = add_self_loops(batch.edge_index, num_nodes=num_nodes)

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            embeddings = model(
                batch.x,
                edge_index_sl,
                batch.node_type if hasattr(batch, "node_type") else None,
                N_base,
                T,
                hierarchy_mapping=hierarchy,
                return_sequence=True,
            )

        # Move to CPU & convert to float32
        emb_cpu = {k: v.float().cpu().numpy() for k, v in embeddings.items()}

        scenario_id = entries[idx].get("scenario_id", f"scenario_{idx+1:05d}")
        scenario_file = entries[idx].get("scenario_file", "")
        out_name = Path(scenario_file).stem if scenario_file else f"scenario_{idx+1:05d}"
        out_path = output_dir / f"{out_name}.npz"

        np.savez_compressed(
            out_path,
            assets=emb_cpu["assets"],
            zones=emb_cpu["zones"],
            regions=emb_cpu["regions"],
            nation=emb_cpu["nation"],
            scenario_id=scenario_id,
            N_base=N_base,
            T=T,
        )

        manifest[scenario_id] = str(out_path.relative_to(output_dir.parent))

        # Periodic GPU cleanup
        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest)} embeddings to {output_dir}")
    print(f"Manifest: {manifest_path}")

    return manifest
