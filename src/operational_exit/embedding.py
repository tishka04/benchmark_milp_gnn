"""Scenario-only HTE; same features and hierarchy as the historical encoder."""
from __future__ import annotations

import numpy as np


def require_gpu():
    import torch
    if not torch.cuda.is_available() or "4050" not in torch.cuda.get_device_name(0):
        raise RuntimeError("RTX 4050 CUDA required; CPU fallback is forbidden")
    return torch.device("cuda:0")


def graph_data(record):
    """In-memory equivalent of TemporalGraphDatasetV3, without files or labels."""
    import torch
    from torch_geometric.data import Data
    x = torch.from_numpy(record["node_features"]).float()
    ei = torch.from_numpy(record["edge_index"]).long()
    et = torch.from_numpy(record["edge_types"]).long()
    nt = torch.from_numpy(record["node_types"]).long()
    meta = record["meta"]
    if hasattr(meta, "item"):
        meta = meta.item()
    N, T = meta["N_base"], meta["T"]
    base = nt[:N]
    zone_mask = base == 2
    zone_indices = torch.where(zone_mask)[0]
    asset_to_zone = torch.zeros(N, dtype=torch.long)
    edges = ei[:, et < 7] % N
    for asset in torch.where(base == 3)[0]:
        targets = edges[1, edges[0] == asset]
        targets = targets[zone_mask[targets]]
        if len(targets):
            asset_to_zone[asset] = (zone_indices == targets[0]).nonzero(as_tuple=True)[0][0]
    zr = torch.from_numpy(np.asarray(record["zone_region_index"])).long()
    return Data(x=x, edge_index=ei, edge_type=et, node_type=nt, N_base=N, T=T,
                asset_to_zone=asset_to_zone, zone_to_region=zr)


def align_zone_embeddings(h, zones):
    # HeteroGraphBuilder creates zone nodes in lexicographic order; MILP
    # ScenarioData uses region/numeric order (R1Z2 differs from R1Z10).
    ordered = sorted(zones)
    if len(ordered) != len(set(ordered)) or h.shape[0] != len(zones):
        raise ValueError("Ambiguous HTE zone axis")
    return h[[ordered.index(z) for z in zones]]


class OnlineHTE:
    def __init__(self, checkpoint_path):
        import torch
        from src.demand_response import require_current_formulation
        from src.gnn.models.hierarchical_temporal_encoder import HierarchicalTemporalEncoder
        self.device = require_gpu()
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        require_current_formulation(ck)
        cfg = ck["config"]
        features = ck["model_state_dict"]["input_proj.weight"].shape[1]
        self.model = HierarchicalTemporalEncoder(node_feature_dim=features, hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads, num_spatial_layers=cfg.num_spatial_layers,
            num_temporal_layers=cfg.num_temporal_layers, dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.mean, self.std = ck["train_mean"].to(self.device), ck["train_std"].to(self.device)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid frozen HTE normalization")
        torch.zeros(1, device=self.device)
        torch.cuda.synchronize()

    def encode(self, data, *, check_graph=None):
        import torch
        from torch_geometric.data import Batch
        from torch_geometric.utils import add_self_loops
        from src.gnn.hetero_graph_dataset import build_hetero_temporal_record
        from src.gnn.embeddings.dataset import get_hierarchy_from_batch
        record = build_hetero_temporal_record(data, mode="supra", temporal_edges=("soc", "ramp", "dr"), time_encoding="sinusoidal")
        if check_graph is not None:
            with np.load(check_graph, allow_pickle=True) as old:
                for key in ("node_features", "edge_index", "edge_types", "node_types", "zone_region_index"):
                    if not np.array_equal(old[key], record[key]):
                        raise ValueError(f"HTE reuse refused: {key} changed in {check_graph}")
        batch = Batch.from_data_list([graph_data(record)]).to(self.device)
        batch.x = (batch.x - self.mean) / self.std
        N, T = int(batch.N_base[0]), int(batch.T[0])
        edges, _ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = self.model(batch.x, edges, batch.node_type, N, T,
                             hierarchy_mapping=get_hierarchy_from_batch(batch, self.device), return_sequence=True)
        h = align_zone_embeddings(emb["zones"].float(), data.zones)
        if h.shape[:2] != (len(data.zones), len(data.periods)) or not torch.isfinite(h).all():
            raise ValueError("HTE axes/nonfinite embedding")
        torch.cuda.synchronize()
        return h.unsqueeze(0)
