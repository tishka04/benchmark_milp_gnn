"""Build the UC-JEPA `*_uc.npz` dataset for the persona scenarios.

Bridges the gap between
  - `outputs/persona_<persona>_scenarios/scenario_*.json` (scenario inputs)
  - `outputs/persona_<persona>_scenarios/reports/scenario_*.json` (MILP refs from
    `scripts.solve_eval_families`)
and the input format expected by
  - `src.uc_jepa.evaluation.evaluate_latent_pipeline` (UC label `*_uc.npz`).

For each persona scenario we:
1. Build a hierarchical temporal heterogeneous graph (`build_hetero_temporal_record`).
2. Forward through the trained HTE encoder (`HierarchicalTemporalEncoder`) to
   obtain per-zone, per-timestep embeddings of shape ``[Z, T, D]``.
3. Persist the embedding as an `*.npz` with a ``zones`` key (so
   `extract_uc_labels._load_embedding` picks it up).
4. Once all 3 personas are embedded, run `extract_uc_labels.run` to fuse the
   MILP report (binary `u_star`, objective, criticality) with the embedding
   (`h_scenario`) into the canonical `*_uc.npz` files.

Outputs:
- `outputs/uc_jepa/persona_graphs/<persona>/scenario_*.npz` (cached graphs,
  reused on rerun via `--resume`)
- `outputs/uc_jepa/persona_embeddings/<persona>/scenario_*.npz`
- `outputs/uc_jepa/dataset_personas/<persona>/scenario_*_uc.npz` (final UC labels
  consumed by `evaluate_latent_pipeline`)

Example
-------
python scripts/build_persona_uc_jepa_dataset.py \
    --personas persona_vre_battery_scenarios \
               persona_network_operator_scenarios \
               persona_mathematician_scenarios \
    --encoder outputs/encoders/hierarchical_temporal_v3/best_encoder.pt \
    --device cuda --resume

After this completes, the apples-to-apples pipeline run is:

python -m src.uc_jepa.evaluation.evaluate_latent_pipeline \
    --latent-ebm outputs/uc_jepa/latent_ebm/latent_ebm_best.pt \
    --data-root  outputs/uc_jepa/dataset_personas \
    --scenario-root outputs/persona_vre_battery_scenarios \
    --scenario-root outputs/persona_network_operator_scenarios \
    --scenario-root outputs/persona_mathematician_scenarios \
    --output-dir outputs/uc_jepa/latent_time_budget_120_personas_k1 \
    --k 1 --global-time-budget-s 120 --enforce-hard-lp-timeout \
    --selection-policy energy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.utils import add_self_loops
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gnn.hetero_graph_dataset import build_hetero_temporal_record, save_graph_record  # noqa: E402
from src.gnn.models.hierarchical_temporal_encoder import HierarchicalTemporalEncoder  # noqa: E402
from src.milp.scenario_loader import load_scenario_data  # noqa: E402
from src.uc_jepa.data import extract_uc_labels  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("build_persona_uc_jepa_dataset")


# Encoder hyperparameters mirror `src.eval.pipeline_runner.PipelineConfig`
# so we load `best_encoder.pt` with the same architecture it was trained with.
ENCODER_DEFAULTS = dict(
    node_feature_dim=14,
    hidden_dim=128,
    num_spatial_layers=2,
    num_temporal_layers=4,
    num_heads=8,
    dropout=0.1,
)


@dataclass
class _Hierarchy:
    asset_to_zone: torch.Tensor
    zone_to_region: torch.Tensor


def _extract_hierarchy(graph_data: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Replicates :py:meth:`PipelineRunner._extract_hierarchy` (no behaviour drift)."""
    meta = graph_data["meta"].item()
    N_base = int(meta["N_base"])
    T = int(meta["T"])

    node_types_base = graph_data["node_types"][:N_base]
    edge_index = graph_data["edge_index"]
    edge_types = graph_data["edge_types"]

    zone_to_region = torch.from_numpy(graph_data["zone_region_index"]).long().to(device)

    spatial_mask = edge_types < 7
    spatial_edges = edge_index[:, spatial_mask]
    base_node_mask = spatial_edges[0] < (N_base * T)
    base_edges = spatial_edges[:, base_node_mask]
    base_edges_mapped = base_edges % N_base

    asset_mask_base = node_types_base == 3
    zone_mask_base = node_types_base == 2

    asset_to_zone = torch.zeros(N_base, dtype=torch.long)
    zone_indices = torch.where(torch.from_numpy(zone_mask_base))[0]

    for asset_idx in torch.where(torch.from_numpy(asset_mask_base))[0]:
        outgoing_mask = base_edges_mapped[0] == asset_idx
        if not outgoing_mask.any():
            continue
        targets = base_edges_mapped[1, outgoing_mask]
        zone_targets = targets[torch.from_numpy(zone_mask_base[targets.numpy()])]
        if len(zone_targets) > 0:
            zone_node_id = zone_targets[0].item()
            zone_list_idx = (zone_indices == zone_node_id).nonzero(as_tuple=True)[0]
            if len(zone_list_idx) > 0:
                asset_to_zone[asset_idx] = zone_list_idx[0]

    asset_to_zone[~asset_mask_base] = 0
    return {
        "asset_to_zone": asset_to_zone.to(device),
        "zone_to_region": zone_to_region,
    }


def _build_graph_if_missing(scenario_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_data = load_scenario_data(scenario_path)
    record = build_hetero_temporal_record(
        scenario_data,
        mode="supra",
        time_window=None,
        stride=1,
        temporal_edges=("soc", "ramp", "dr"),
        time_encoding="sinusoidal",
    )
    save_graph_record(record, output_path)


def _load_encoder(encoder_path: Path, device: torch.device) -> HierarchicalTemporalEncoder:
    encoder = HierarchicalTemporalEncoder(**ENCODER_DEFAULTS).to(device)
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    encoder.load_state_dict(state_dict)
    encoder.eval()
    LOGGER.info(
        "Loaded HTE encoder from %s (%s params)",
        encoder_path,
        f"{sum(p.numel() for p in encoder.parameters()):,}",
    )
    return encoder


@torch.no_grad()
def _embed_graph(
    encoder: HierarchicalTemporalEncoder,
    graph_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    graph_data = dict(np.load(graph_path, allow_pickle=True))

    x = torch.from_numpy(graph_data["node_features"]).float().to(device)
    edge_index = torch.from_numpy(graph_data["edge_index"]).long().to(device)
    node_types = torch.from_numpy(graph_data["node_types"]).long().to(device)

    meta = graph_data["meta"].item()
    N_base = int(meta["N_base"])
    T = int(meta["T"])

    edge_index_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    hierarchy = _extract_hierarchy(graph_data, device)

    embeddings = encoder(
        x,
        edge_index_loops,
        node_types,
        N_base,
        T,
        hierarchy_mapping=hierarchy,
        return_sequence=True,
    )

    n_zones = int(hierarchy["zone_to_region"].numel())

    if isinstance(embeddings, dict) and embeddings.get("zones") is not None:
        zone_emb = embeddings["zones"].detach().cpu()
    else:
        # Fallback: pool asset embeddings into zones via the asset->zone map.
        assets_emb = (
            embeddings["assets"].detach().cpu()
            if isinstance(embeddings, dict)
            else embeddings.detach().cpu()
        )
        asset_to_zone = hierarchy["asset_to_zone"].cpu()
        D = assets_emb.shape[-1]
        zone_emb = torch.zeros(n_zones, T, D)
        zone_counts = torch.zeros(n_zones, 1, 1)
        for ai in range(min(len(asset_to_zone), assets_emb.shape[0])):
            zi = asset_to_zone[ai].item()
            if 0 <= zi < n_zones:
                zone_emb[zi] += assets_emb[ai]
                zone_counts[zi] += 1
        zone_counts = zone_counts.clamp(min=1)
        zone_emb = zone_emb / zone_counts

    if zone_emb.dim() == 2:
        # Some encoders return [Z, D] for time-pooled outputs. Repeat across T.
        zone_emb = zone_emb.unsqueeze(1).expand(-1, T, -1).contiguous()

    payload: Dict[str, Any] = {
        "zones": zone_emb.numpy().astype(np.float32),
        "scenario_id": np.array(str(graph_path.stem), dtype=object),
        "N_base": np.int64(N_base),
        "T": np.int64(T),
    }
    if isinstance(embeddings, dict):
        for key in ("assets", "regions", "nation"):
            tensor = embeddings.get(key)
            if tensor is not None:
                payload[key] = tensor.detach().cpu().numpy().astype(np.float32)
    return payload


def _embed_persona(
    persona_dir: Path,
    graphs_dir: Path,
    embeddings_dir: Path,
    encoder: HierarchicalTemporalEncoder,
    device: torch.device,
    resume: bool,
) -> int:
    scenario_files = sorted(persona_dir.glob("scenario_*.json"))
    if not scenario_files:
        LOGGER.warning("No scenarios in %s", persona_dir)
        return 0
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for sc_path in tqdm(scenario_files, desc=f"embed:{persona_dir.name}", unit="scenario"):
        sid = sc_path.stem
        graph_path = graphs_dir / f"{sid}.npz"
        emb_path = embeddings_dir / f"{sid}.npz"
        if resume and emb_path.exists():
            n_done += 1
            continue
        try:
            _build_graph_if_missing(sc_path, graph_path)
            payload = _embed_graph(encoder, graph_path, device)
            np.savez_compressed(emb_path, **payload)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed embedding for %s: %s", sc_path, exc)
    return n_done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=[
            "persona_vre_battery_scenarios",
            "persona_network_operator_scenarios",
            "persona_mathematician_scenarios",
        ],
        help="Persona directories under `outputs/`.",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=REPO_ROOT / "outputs/encoders/hierarchical_temporal_v3/best_encoder.pt",
        help="Path to the trained HTE encoder checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--graphs-root",
        type=Path,
        default=REPO_ROOT / "outputs/uc_jepa/persona_graphs",
        help="Where to cache the per-scenario heterogeneous temporal graphs.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=REPO_ROOT / "outputs/uc_jepa/persona_embeddings",
        help="Where to write per-scenario zone embeddings.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs/uc_jepa/dataset_personas",
        help="Final dataset directory consumed by `evaluate_latent_pipeline`.",
    )
    parser.add_argument("--n-timesteps", type=int, default=24)
    parser.add_argument("--resume", action="store_true", help="Skip already-embedded scenarios.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Stop after embedding (do not call extract_uc_labels).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = _load_encoder(args.encoder, device)

    outputs_dir = REPO_ROOT / "outputs"
    persona_jobs: List[Dict[str, Path]] = []
    for persona in args.personas:
        persona_dir = outputs_dir / persona
        if not persona_dir.is_dir():
            LOGGER.warning("Skipping missing persona dir: %s", persona_dir)
            continue
        reports_dir = persona_dir / "reports"
        if not reports_dir.is_dir():
            LOGGER.error(
                "Reports dir missing for %s. Run `scripts.solve_eval_families` first.",
                persona,
            )
            sys.exit(2)

        graphs_dir = args.graphs_root / persona
        embeddings_dir = args.embeddings_root / persona
        n = _embed_persona(persona_dir, graphs_dir, embeddings_dir, encoder, device, args.resume)
        LOGGER.info("[%s] embedded %d / %d scenarios", persona, n, len(list(persona_dir.glob('scenario_*.json'))))
        persona_jobs.append({
            "persona": persona,
            "reports_dir": reports_dir,
            "embeddings_dir": embeddings_dir,
        })

    if args.skip_extract or not persona_jobs:
        LOGGER.info("Skipping extract_uc_labels step.")
        return

    # CRITICAL: extract_uc_labels._build_embedding_index keys lookups by
    # scenario_id only. Since each persona reuses the same scenario IDs
    # (00001-00100), running extract_uc_labels with a SHARED embeddings root
    # would scramble the h_scenario assignment (later personas would overwrite
    # earlier ones in the lookup dict). We therefore run extract_uc_labels
    # ONCE PER PERSONA with the embeddings_dir scoped to that persona.
    for job in persona_jobs:
        LOGGER.info(
            "extract_uc_labels [%s]: %s + %s -> %s",
            job["persona"], job["reports_dir"], job["embeddings_dir"], args.out,
        )
        extract_uc_labels.run(
            report_dirs=[str(job["reports_dir"])],
            embeddings_dir=str(job["embeddings_dir"]),
            out_dir=str(args.out),
            n_timesteps=args.n_timesteps,
            max_per_family=None,
            embed_dim_hint=ENCODER_DEFAULTS["hidden_dim"],
            overwrite=True,
        )
    LOGGER.info(
        "DONE. Persona UC dataset at %s. Now run evaluate_latent_pipeline with --data-root %s",
        args.out,
        args.out,
    )


if __name__ == "__main__":
    main()
