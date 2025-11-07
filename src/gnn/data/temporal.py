from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.gnn.dataset_loader import iter_split, load_index


@dataclass
class GraphSample:
    node_static: torch.Tensor
    node_time: torch.Tensor
    node_input: torch.Tensor
    node_region: torch.LongTensor
    node_type: torch.LongTensor
    edge_index: torch.LongTensor
    edge_attr: torch.Tensor
    edge_type: torch.LongTensor
    target: torch.Tensor
    target_pre: torch.Tensor
    target_correction: torch.Tensor
    duals: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]

    def num_nodes(self) -> int:
        return int(self.node_input.size(0))

    def num_edges(self) -> int:
        return int(self.edge_attr.size(0))


@dataclass
class GraphBatch:
    node_static: torch.Tensor
    node_time: torch.Tensor
    node_input: torch.Tensor
    node_region: torch.LongTensor
    node_type: torch.LongTensor
    edge_index: torch.LongTensor
    edge_attr: torch.Tensor
    edge_type: torch.LongTensor
    target: torch.Tensor
    target_pre: torch.Tensor
    target_correction: torch.Tensor
    duals: Dict[str, torch.Tensor]
    node_batch: torch.LongTensor
    edge_batch: torch.LongTensor
    metadata: List[Dict[str, Any]]

    def to(self, device: torch.device | str) -> "GraphBatch":
        duals = {name: value.to(device) for name, value in self.duals.items()}
        return GraphBatch(
            node_static=self.node_static.to(device),
            node_time=self.node_time.to(device),
            node_input=self.node_input.to(device),
            node_region=self.node_region.to(device),
            node_type=self.node_type.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            edge_type=self.edge_type.to(device),
            target=self.target.to(device),
            target_pre=self.target_pre.to(device),
            target_correction=self.target_correction.to(device),
            duals=duals,
            node_batch=self.node_batch.to(device),
            edge_batch=self.edge_batch.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return len(self.metadata)


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _derive_node_types(node_static: torch.Tensor) -> torch.LongTensor:
    thermal_like = node_static[:, 0] + node_static[:, 5]
    renewable_like = node_static[:, 1] + node_static[:, 2] + node_static[:, 6] + node_static[:, 7]
    storage_like = node_static[:, 3] + node_static[:, 8]
    demand_response = node_static[:, 4]
    type_scores = torch.stack([thermal_like, renewable_like, storage_like, demand_response], dim=1)
    return torch.argmax(type_scores, dim=1)


class GraphTemporalDataset(Dataset[GraphSample]):
    def __init__(
        self,
        index_path: Path | str,
        split: str = "train",
        *,
        include_duals: bool = True,
        preload: bool = False,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
    ) -> None:
        self.split = split
        self.include_duals = include_duals
        self._preload = preload
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._index_path = Path(index_path).resolve()
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"Dataset index not found at {self._index_path}. "
                "Run the graph dataset builder or update the config path."
            )
        self._index = load_index(self._index_path)
        self._entries = list(iter_split(self._index, split, train_split=train_fraction, val_split=val_fraction))
        if not self._entries:
            raise RuntimeError(f"Split '{split}' is empty in {index_path}")
        self._scenario_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._scenario_meta: List[Dict[str, Any]] = []
        self._time_index: List[Tuple[int, int]] = []
        self._input_dim: Optional[int] = None
        self._target_dim: Optional[int] = None
        self._max_edge_type: int = -1
        self._prepare_index()

    def _prepare_index(self) -> None:
        for scenario_idx, entry in enumerate(self._entries):
            arrays = _read_npz(Path(entry["graph_file"]))
            time_len = int(arrays["node_time"].shape[0])
            static_dim = int(arrays["node_static"].shape[1])
            temporal_dim = int(arrays["node_time"].shape[2])
            target_dim = int(arrays["node_labels"].shape[2])
            self._input_dim = self._input_dim or (static_dim + temporal_dim)
            self._target_dim = self._target_dim or target_dim
            edge_type_arr = arrays.get("edge_type")
            if edge_type_arr is not None and edge_type_arr.size:
                self._max_edge_type = max(self._max_edge_type, int(edge_type_arr.max()))
            meta = {
                "scenario_id": entry.get("scenario_id") or Path(entry["graph_file"]).stem,
                "graph_file": entry["graph_file"],
                "objective": float(entry.get("objective", float("nan"))),
                "time_steps": arrays["time_steps"],
                "time_hours": arrays["time_hours"],
            }
            self._scenario_meta.append(meta)
            for t in range(time_len):
                self._time_index.append((scenario_idx, t))
            if self._preload:
                self._scenario_cache[scenario_idx] = self._build_scenario(arrays)

        if self._input_dim is None or self._target_dim is None:
            raise RuntimeError("Failed to infer dataset feature dimensions")

    def _build_scenario(self, arrays: Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        node_static = torch.from_numpy(arrays["node_static"]).float()
        node_time = torch.from_numpy(arrays["node_time"]).float()
        node_labels = torch.from_numpy(arrays["node_labels"]).float()
        node_labels_pre = torch.from_numpy(
            arrays.get("node_labels_pre", arrays["node_labels"])
        ).float()
        node_labels_correction = torch.from_numpy(
            arrays.get(
                "node_labels_correction",
                np.zeros_like(arrays["node_labels"], dtype=np.float32),
            )
        ).float()
        zone_region = torch.from_numpy(arrays.get("zone_region_index", np.zeros(node_static.shape[0], dtype=np.int64))).long()
        edge_index = torch.from_numpy(arrays["edge_index"].astype(np.int64).T).long()
        edge_capacity = torch.from_numpy(arrays["edge_capacity"]).float()
        edge_flows = torch.from_numpy(arrays["edge_flows"]).float()
        edge_type = torch.from_numpy(arrays.get("edge_type", np.zeros(edge_capacity.shape[0], dtype=np.int64))).long()
        time_steps = torch.from_numpy(arrays["time_steps"].astype(np.int64))
        time_hours = torch.from_numpy(arrays["time_hours"]).float()
        duals: Dict[str, torch.Tensor] = {}
        if self.include_duals:
            for key, value in arrays.items():
                if key.startswith("duals_"):
                    dual_name = key[len("duals_"):]
                    duals[dual_name] = torch.from_numpy(value.T).float()
        return {
            "node_static": node_static,
            "node_time": node_time,
            "node_labels": node_labels,
            "node_labels_pre": node_labels_pre,
            "node_labels_correction": node_labels_correction,
            "zone_region": zone_region,
            "edge_index": edge_index,
            "edge_capacity": edge_capacity,
            "edge_flows": edge_flows,
            "edge_type": edge_type,
            "time_steps": time_steps,
            "time_hours": time_hours,
            "duals": duals,
        }

    def _get_scenario(self, scenario_idx: int) -> Dict[str, torch.Tensor]:
        cached = self._scenario_cache.get(scenario_idx)
        if cached is not None:
            return cached
        arrays = _read_npz(Path(self._entries[scenario_idx]["graph_file"]))
        record = self._build_scenario(arrays)
        if self._preload:
            self._scenario_cache[scenario_idx] = record
        return record

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._time_index)

    def __getitem__(self, index: int) -> GraphSample:  # type: ignore[override]
        scenario_idx, time_idx = self._time_index[index]
        record = self._get_scenario(scenario_idx)
        node_static = record["node_static"]
        node_time_series = record["node_time"]
        node_time = node_time_series[time_idx]
        node_input = torch.cat([node_static, node_time], dim=1)
        target = record["node_labels"][time_idx]
        target_pre = record["node_labels_pre"][time_idx] if "node_labels_pre" in record else target
        target_corr = record["node_labels_correction"][time_idx] if "node_labels_correction" in record else torch.zeros_like(target)
        edge_capacity = record["edge_capacity"]
        edge_flow_t = record["edge_flows"][:, time_idx]
        edge_attr = torch.stack([edge_capacity, edge_flow_t], dim=1)
        node_type = _derive_node_types(node_static)
        duals = {name: value[time_idx] for name, value in record["duals"].items()}
        meta_src = self._scenario_meta[scenario_idx]
        metadata = {
            "scenario_id": meta_src["scenario_id"],
            "graph_file": meta_src["graph_file"],
            "objective": meta_src["objective"],
            "time_index": int(record["time_steps"][time_idx].item()) if record["time_steps"].numel() else time_idx,
            "time_hour": float(record["time_hours"][time_idx].item()),
            "split": self.split,
        }
        return GraphSample(
            node_static=node_static,
            node_time=node_time,
            node_input=node_input,
            node_region=record["zone_region"],
            node_type=node_type,
            edge_index=record["edge_index"],
            edge_attr=edge_attr,
            edge_type=record["edge_type"],
            target=target,
            target_pre=target_pre,
            target_correction=target_corr,
            duals=duals,
            metadata=metadata,
        )

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            raise RuntimeError("Dataset input dimension is not initialised")
        return self._input_dim

    @property
    def target_dim(self) -> int:
        if self._target_dim is None:
            raise RuntimeError("Dataset target dimension is not initialised")
        return self._target_dim

    @property
    def edge_type_cardinality(self) -> int:
        return self._max_edge_type + 1 if self._max_edge_type >= 0 else 0

    @property
    def node_type_cardinality(self) -> int:
        # Thermal, renewable, storage, demand-response buckets
        return 4

    @property
    def num_scenarios(self) -> int:
        return len(self._entries)


def collate_graph_samples(samples: Sequence[GraphSample]) -> GraphBatch:
    if not samples:
        raise RuntimeError("Cannot collate an empty batch")
    node_static_list: List[torch.Tensor] = []
    node_time_list: List[torch.Tensor] = []
    node_input_list: List[torch.Tensor] = []
    node_region_list: List[torch.LongTensor] = []
    node_type_list: List[torch.LongTensor] = []
    target_list: List[torch.Tensor] = []
    target_pre_list: List[torch.Tensor] = []
    target_corr_list: List[torch.Tensor] = []
    node_batch_list: List[torch.LongTensor] = []
    edge_index_list: List[torch.LongTensor] = []
    edge_attr_list: List[torch.Tensor] = []
    edge_type_list: List[torch.LongTensor] = []
    edge_batch_list: List[torch.LongTensor] = []
    metadata: List[Dict[str, Any]] = []
    dual_keys = {key for sample in samples for key in sample.duals}
    dual_values: Dict[str, List[torch.Tensor]] = {key: [] for key in dual_keys}

    node_offset = 0
    for batch_idx, sample in enumerate(samples):
        num_nodes = sample.num_nodes()
        num_edges = sample.num_edges()
        node_static_list.append(sample.node_static)
        node_time_list.append(sample.node_time)
        node_input_list.append(sample.node_input)
        node_region_list.append(sample.node_region)
        node_type_list.append(sample.node_type)
        target_list.append(sample.target)
        target_pre_list.append(sample.target_pre)
        target_corr_list.append(sample.target_correction)
        node_batch_list.append(torch.full((num_nodes,), batch_idx, dtype=torch.long))
        edge_index_list.append(sample.edge_index + node_offset)
        edge_attr_list.append(sample.edge_attr)
        edge_type_list.append(sample.edge_type)
        edge_batch_list.append(torch.full((num_edges,), batch_idx, dtype=torch.long))
        metadata.append(sample.metadata)
        for key in dual_keys:
            value = sample.duals.get(key)
            if value is None:
                value = torch.zeros(num_nodes, dtype=torch.float32, device=sample.node_static.device)
            dual_values[key].append(value)
        node_offset += num_nodes

    node_static = torch.cat(node_static_list, dim=0)
    node_time = torch.cat(node_time_list, dim=0)
    node_input = torch.cat(node_input_list, dim=0)
    node_region = torch.cat(node_region_list, dim=0)
    node_type = torch.cat(node_type_list, dim=0)
    target = torch.cat(target_list, dim=0)
    target_pre = torch.cat(target_pre_list, dim=0)
    target_correction = torch.cat(target_corr_list, dim=0)
    node_batch = torch.cat(node_batch_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    edge_type = torch.cat(edge_type_list, dim=0)
    edge_batch = torch.cat(edge_batch_list, dim=0)
    duals = {key: torch.cat(value_list, dim=0) for key, value_list in dual_values.items()}

    return GraphBatch(
        node_static=node_static,
        node_time=node_time,
        node_input=node_input,
        node_region=node_region,
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        target=target,
        target_pre=target_pre,
        target_correction=target_correction,
        duals=duals,
        node_batch=node_batch,
        edge_batch=edge_batch,
        metadata=metadata,
    )
