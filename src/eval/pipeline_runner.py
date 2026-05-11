"""
Pipeline Runner: Full evaluation pipeline for GNN+EBM approach.

Runs the complete pipeline:
1. Build Hierarchical Temporal Graphs from scenario JSONs
2. Generate HTE Embeddings using trained encoder
3. Extract Zone-level Embeddings for EBM input
4. EBM + Normalized Temporal Langevin Sampler → binary candidates
5. Pass-through feasibility decoder → LP warm-start only, no binary edits
6. LP Worker Two-Stage → solve continuous LP

Designed to be imported from Colab notebook via Google Drive.
"""
from __future__ import annotations

import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from tqdm.auto import tqdm
from typing import Literal


_BINARY_FEATURE_NAMES = [
    'battery_charge',
    'battery_discharge',
    'pumped_charge',
    'pumped_discharge',
    'dr_active',
    'thermal_startup',
    'thermal_on',
]

_LP_STAGE_ORDER = [
    'hard_fix',
    'repair_20',
    'repair_100',
    'full_soft',
    'round_refix',
]


def _display_stage_name(lp_result: Any, stage: str) -> str:
    """Map internal LP-worker stage keys to user-facing labels when available."""
    if stage == 'repair_20':
        return str(getattr(lp_result, 'stage_name_repair_20', '') or stage)
    if stage == 'repair_100':
        return str(getattr(lp_result, 'stage_name_repair_100', '') or stage)
    return str(stage or '')


def _binary_stats(u_bin: torch.Tensor) -> Dict[str, float]:
    """Summarize the activity rate of each binary channel."""
    stats: Dict[str, float] = {}
    for idx, name in enumerate(_BINARY_FEATURE_NAMES):
        stats[name] = float(u_bin[..., idx].float().mean().item())
    return stats


def _pairwise_hamming_stats(samples: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Compute pairwise Hamming diversity across sampled binary candidates.

    Distances are normalized in [0, 1] by the total number of binary entries
    in the candidate tensor, matching the paper-style interpretation
    ``||u_a - u_b||_1 / (Z*T*F)``.
    """
    if not samples:
        return {
            'distances': [],
            'matrix': [],
            'mean': float('nan'),
            'max': float('nan'),
            'unique_ratio': float('nan'),
        }

    flat_samples = [
        sample.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
        for sample in samples
    ]
    distances: List[float] = []
    n_samples = len(flat_samples)
    matrix = np.zeros((n_samples, n_samples), dtype=float)
    for idx_a in range(len(flat_samples)):
        for idx_b in range(idx_a + 1, len(flat_samples)):
            dist = float(np.mean(flat_samples[idx_a] != flat_samples[idx_b]))
            distances.append(dist)
            matrix[idx_a, idx_b] = dist
            matrix[idx_b, idx_a] = dist

    unique_ratio = len({flat.tobytes() for flat in flat_samples}) / len(flat_samples)
    return {
        'distances': distances,
        'matrix': matrix.tolist(),
        'mean': float(np.mean(distances)) if distances else 0.0,
        'max': float(np.max(distances)) if distances else 0.0,
        'unique_ratio': float(unique_ratio),
    }


def _normalize_stage_name(stage: Any) -> str:
    if hasattr(stage, 'value'):
        stage = stage.value
    return str(stage or '')


def _extract_stage_flags(lp_result: Any) -> Dict[str, bool]:
    """Infer which LP-worker stages were actually executed."""
    flags: Dict[str, bool] = {}
    final_stage = _normalize_stage_name(getattr(lp_result, 'stage_used', ''))
    for stage in _LP_STAGE_ORDER:
        time_val = float(getattr(lp_result, f'time_{stage}', 0.0) or 0.0)
        slack_val = getattr(lp_result, f'slack_{stage}', 0.0)
        flags[stage] = bool(
            time_val > 0.0
            or final_stage == stage
            or (slack_val is not None and np.isfinite(float(slack_val)) and float(slack_val) > 0.0)
        )
    return flags


def _deepest_stage_reached(lp_result: Any) -> str:
    flags = _extract_stage_flags(lp_result)
    for stage in reversed(_LP_STAGE_ORDER):
        if flags[stage]:
            return _display_stage_name(lp_result, stage)
    return _display_stage_name(lp_result, _normalize_stage_name(getattr(lp_result, 'stage_used', '')))


def _stage_rank(stage: str) -> int:
    try:
        return _LP_STAGE_ORDER.index(stage) + 1
    except ValueError:
        return 0


def _diagnostic_result_key(lp_result: Any) -> Tuple[int, int, float]:
    """
    Sort key for fallback diagnostics when no finite objective exists.

    Prefer the deepest reached stage, then any loaded solution over a pure
    failure marker, then the lowest available slack.
    """
    reached_stage = _deepest_stage_reached(lp_result)
    status = str(getattr(lp_result, 'status', ''))
    def _slack_or_inf(attr_name: str) -> float:
        raw_value = getattr(lp_result, attr_name, float('inf'))
        return float('inf') if raw_value is None else float(raw_value)

    slack_candidates = [
        _slack_or_inf('slack_used'),
        _slack_or_inf('slack_round_refix'),
        _slack_or_inf('slack_full_soft'),
        _slack_or_inf('slack_repair_100'),
        _slack_or_inf('slack_repair_20'),
        _slack_or_inf('slack_hard_fix'),
    ]
    finite_slacks = [value for value in slack_candidates if np.isfinite(value)]
    best_slack = min(finite_slacks) if finite_slacks else float('inf')
    has_solution = 0 if status in {'infeasible', 'error', 'pending'} else 1
    return (_stage_rank(reached_stage), has_solution, -best_slack if np.isfinite(best_slack) else -float('inf'))


@dataclass
class PipelineConfig:
    """Configuration for pipeline evaluation."""
    # Paths (set for Colab)
    repo_path: str = '/content/drive/MyDrive/benchmark'
    encoder_path: str = 'outputs/encoders/hierarchical_temporal_v3/best_encoder.pt'
    ebm_path: str = 'outputs/ebm_models/ebm_v3/ebm_v3_final.pt'

    # Model architecture
    node_feature_dim: int = 14
    hidden_dim: int = 128
    num_spatial_layers: int = 2
    num_temporal_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # EBM params
    embed_dim: int = 128
    n_features: int = 7
    n_timesteps: int = 24

    # Langevin sampler (matched to training config for diversity)
    langevin_steps: int = 100
    step_size: float = 0.05
    noise_scale: float = 0.50
    init_temp: float = 1.0
    final_temp: float = 0.1
    n_samples: int = 5
    # Native sampler_v3 knobs for evaluation diversity control.
    # `temp_max` / `temp_min` override `init_temp` / `final_temp` when provided.
    temp_max: Optional[float] = None
    temp_min: Optional[float] = None
    init_mode: Literal['soft', 'prior', 'bernoulli', 'oracle'] = 'bernoulli'
    init_p: float = 0.5
    prior_p: float = 0.025
    prior_strength: float = 0.0
    normalize_grad: bool = True
    infer_binarize: Literal['bernoulli', 'threshold'] = 'bernoulli'
    infer_threshold: float = 0.5
    require_train_mode_for_sampling: bool = True
    diversity_priority: bool = True
    diversity_temperature_scale_min: float = 1.15
    diversity_temperature_scale_max: float = 2.50
    diversity_noise_scale_min: float = 1.00
    diversity_noise_scale_max: float = 2.00
    diversity_similarity_lambda: float = 0.35
    diversity_schedule: Literal['linear', 'staggered'] = 'linear'

    # LP Worker
    solver_name: str = 'appsi_highs'
    slack_tol_mwh: float = 1.0
    deviation_penalty: float = 10000.0
    flip_budget_repair_20: int = 100
    flip_budget_repair_100: int = 1000
    lp_worker_max_stages: int = 5
    strict_feasibility_tol_mwh: Optional[float] = None
    enable_exact_fallback: bool = False
    fallback_solver_name: Optional[str] = None
    fallback_milp_time_limit_seconds: float = 600.0

    # Ablation flags
    skip_decoder: bool = False  # bypass decoder entirely, pass raw EBM binaries to LP
    decoder_passthrough: bool = True  # kept for compatibility; evaluation always preserves source binaries
    use_gnn_dispatch: bool = False  # replace decoder+LP with GNN dispatch predictor
    gnn_dispatch_path: str = 'outputs/gnn_dispatch/dispatch_gnn_best.pt'
    top_m_enabled: bool = False
    top_m: int = 1
    top_m_score_mode: Literal['energy', 'activation', 'energy_activation', 'first', 'learned_selector'] = 'energy'
    top_m_selector_path: Optional[str] = None
    top_m_target_activation_rate: Optional[float] = None
    top_m_energy_weight: float = 1.0
    top_m_activation_weight: float = 1.0

    device: str = 'cuda'
    seed: int = 42

    @property
    def sampler_temp_max(self) -> float:
        """Resolved high-temperature endpoint for sampler_v3."""
        return float(self.temp_max if self.temp_max is not None else self.init_temp)

    @property
    def sampler_temp_min(self) -> float:
        """Resolved low-temperature endpoint for sampler_v3."""
        return float(self.temp_min if self.temp_min is not None else self.final_temp)


@dataclass
class PipelineResult:
    """Result from running the pipeline on a single scenario."""
    scenario_id: str
    family: str = ''

    # Timing breakdown (seconds)
    time_graph_build: float = 0.0
    time_embedding: float = 0.0
    time_ebm_sampling: float = 0.0
    time_decoder: float = 0.0
    time_lp_solve: float = 0.0
    time_total: float = 0.0

    # LP results (best sample)
    lp_status: str = ''
    lp_stage_used: str = ''
    lp_stage_reached: str = ''
    lp_objective: float = float('nan')
    lp_slack: float = 0.0          # final slack of the stage that was kept
    lp_n_flips: int = 0

    # Per-stage slack progression (MWh) of the best sample.
    # A scenario can carry multiple non-zero values if the cascade reached
    # Stage 4/5 (e.g. slack_full_soft and slack_round_refix coexist).
    lp_slack_hard_fix: float = 0.0
    lp_slack_repair_20: float = 0.0
    lp_slack_repair_100: float = 0.0
    lp_slack_full_soft: float = 0.0
    lp_slack_round_refix: float = 0.0
    lp_reached_hard_fix: bool = False
    lp_reached_repair_20: bool = False
    lp_reached_repair_100: bool = False
    lp_reached_full_soft: bool = False
    lp_reached_round_refix: bool = False

    # All sample results
    n_samples: int = 0
    best_sample_idx: int = 0
    all_objectives: List[float] = field(default_factory=list)
    all_objectives_raw: List[float] = field(default_factory=list)
    all_stages: List[str] = field(default_factory=list)
    all_stages_reached: List[str] = field(default_factory=list)
    all_sample_sampling_times: List[float] = field(default_factory=list)
    all_sample_decoder_times: List[float] = field(default_factory=list)
    all_sample_lp_solve_times: List[float] = field(default_factory=list)
    all_sample_total_times: List[float] = field(default_factory=list)
    all_sample_ebm_energies: List[float] = field(default_factory=list)
    all_sample_slacks: List[float] = field(default_factory=list)
    all_sample_statuses: List[str] = field(default_factory=list)
    best_sample_sampling_time: float = float('nan')
    best_sample_decoder_time: float = float('nan')
    best_sample_lp_solve_time: float = float('nan')
    best_sample_total_time: float = float('nan')
    first_sample_total_time: float = float('nan')
    all_binary_active_fractions: List[Dict[str, float]] = field(default_factory=list)
    pairwise_hamming_distances: List[float] = field(default_factory=list)
    pairwise_hamming_matrix: List[List[float]] = field(default_factory=list)
    sample_mean_pairwise_hamming: float = float('nan')
    sample_max_pairwise_hamming: float = float('nan')
    sample_unique_ratio: float = float('nan')
    sample_energy_mean: float = float('nan')
    sample_energy_std: float = float('nan')
    best_status: str = ''
    direct_feasible_count: int = 0
    repaired_feasible_count: int = 0
    direct_feasible_rate: float = float('nan')
    repair_success_rate: float = float('nan')
    fallback_rate: float = float('nan')
    fallback_used: bool = False
    time_direct_lp: float = 0.0
    time_repair: float = 0.0
    time_fallback: float = 0.0
    fallback_warm_start_vars: int = 0
    all_repair_flips: List[int] = field(default_factory=list)
    all_repair_radii: List[float] = field(default_factory=list)
    all_repair_times: List[float] = field(default_factory=list)
    mean_repair_flips: float = float('nan')
    median_repair_flips: float = float('nan')
    min_repair_flips: float = float('nan')
    max_repair_flips: float = float('nan')
    median_repair_radius_used: float = float('nan')
    max_repair_radius_used: float = float('nan')
    mean_repair_time: float = float('nan')
    median_repair_time: float = float('nan')
    selected_repair_flips: float = float('nan')
    selected_repair_radius_used: float = float('nan')
    selected_repair_time: float = float('nan')
    top_m_enabled: bool = False
    top_m_score_mode: str = ''
    top_m_projected: int = 0
    top_m_selected_indices: List[int] = field(default_factory=list)
    top_m_skipped_indices: List[int] = field(default_factory=list)
    top_m_candidate_scores: List[float] = field(default_factory=list)
    n_candidates_generated: int = 0
    n_candidates_projected: int = 0

    # Scenario metadata
    n_zones: int = 0
    n_timesteps: int = 24
    criticality_index: float = 0.0

    # Error tracking
    success: bool = True
    error_message: str = ''


class PipelineRunner:
    """
    Runs the full GNN+EBM+LP pipeline on evaluation scenarios.

    Usage:
        runner = PipelineRunner(config)
        runner.load_models()
        results = runner.evaluate_family(scenarios_dir, reports_dir)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.repo_path = Path(config.repo_path)
        self.device = config.device
        self.encoder = None
        self.ebm = None
        self.sampler = None
        self.top_m_selector = None

    def load_models(self):
        """Load encoder, EBM, and optionally GNN dispatch models."""
        from src.gnn.models.hierarchical_temporal_encoder import HierarchicalTemporalEncoder
        from src.ebm.model_v3 import TrajectoryZonalEBM
        from src.ebm.sampler_v3 import NormalizedTemporalLangevinSampler

        cfg = self.config

        # Load encoder
        print("Loading HTE encoder...")
        self.encoder = HierarchicalTemporalEncoder(
            node_feature_dim=cfg.node_feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_spatial_layers=cfg.num_spatial_layers,
            num_temporal_layers=cfg.num_temporal_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(self.device)

        encoder_path = self.repo_path / cfg.encoder_path
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        print(f"  Encoder loaded: {sum(p.numel() for p in self.encoder.parameters()):,} params")

        # Load EBM
        print("Loading EBM v3...")
        self.ebm = TrajectoryZonalEBM(
            embed_dim=cfg.embed_dim,
            n_features=cfg.n_features,
        ).to(self.device)

        ebm_path = self.repo_path / cfg.ebm_path
        ebm_checkpoint = torch.load(ebm_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in ebm_checkpoint:
            self.ebm.load_state_dict(ebm_checkpoint['model_state_dict'])
        else:
            self.ebm.load_state_dict(ebm_checkpoint)
        self.ebm.eval()
        print(f"  EBM loaded: {sum(p.numel() for p in self.ebm.parameters()):,} params")

        # Create sampler
        self.sampler = NormalizedTemporalLangevinSampler(
            model=self.ebm,
            n_features=cfg.n_features,
            num_steps=cfg.langevin_steps,
            step_size=cfg.step_size,
            noise_scale=cfg.noise_scale,
            temp_min=cfg.sampler_temp_min,
            temp_max=cfg.sampler_temp_max,
            init_mode=cfg.init_mode,
            init_p=cfg.init_p,
            prior_p=cfg.prior_p,
            prior_strength=cfg.prior_strength,
            normalize_grad=cfg.normalize_grad,
            mode='infer',
            infer_binarize=cfg.infer_binarize,
            infer_threshold=cfg.infer_threshold,
            require_train_mode_for_sampling=cfg.require_train_mode_for_sampling,
            device=self.device,
        )
        print("  Sampler ready (infer mode)")

        # Optionally load GNN dispatch predictor
        self.gnn_predictor = None
        if cfg.use_gnn_dispatch:
            from src.gnn.dispatch_predictor import GNNDispatchPredictor
            gnn_path = self.repo_path / cfg.gnn_dispatch_path
            self.gnn_predictor = GNNDispatchPredictor(
                model_path=str(gnn_path),
                device=self.device,
            )

    def _build_graph(self, scenario_path: Path, report_path: Path, output_path: Path):
        """Build temporal graph for a scenario."""
        from src.gnn.hetero_graph_dataset import build_hetero_temporal_record, save_graph_record
        from src.milp.scenario_loader import load_scenario_data

        # Validate cached graph matches scenario zone count
        if output_path.exists():
            try:
                import numpy as np
                cached = dict(np.load(output_path, allow_pickle=True))
                with open(scenario_path, 'r') as f:
                    sc = json.load(f)
                expected_zones = sum(sc.get('graph', {}).get('zones_per_region', [1]))
                # Check node count (heuristic: if very different, rebuild)
                cached_nodes = cached.get('num_nodes', np.array(0)).item() if 'num_nodes' in cached else -1
                if cached_nodes > 0 and abs(cached_nodes - expected_zones) > expected_zones:
                    import logging
                    logging.info(f"Stale graph cache for {scenario_path.stem}: "
                                 f"cached nodes={cached_nodes}, expected zones={expected_zones}. Rebuilding.")
                else:
                    return
            except Exception:
                return  # If validation fails, trust the cache

        scenario_data = load_scenario_data(scenario_path)

        record = build_hetero_temporal_record(
            scenario_data,
            mode='supra',
            time_window=None,
            stride=1,
            temporal_edges=('soc', 'ramp', 'dr'),
            time_encoding='sinusoidal',
        )
        save_graph_record(record, output_path)

    def _extract_hierarchy(self, graph_data, device):
        """Extract hierarchy mapping from graph structure."""
        meta = graph_data['meta'].item()
        N_base = meta['N_base']
        T = meta['T']

        node_types_base = graph_data['node_types'][:N_base]
        edge_index = graph_data['edge_index']
        edge_types = graph_data['edge_types']

        zone_to_region = torch.from_numpy(graph_data['zone_region_index']).long().to(device)

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
            if outgoing_mask.any():
                targets = base_edges_mapped[1, outgoing_mask]
                zone_targets = targets[torch.from_numpy(zone_mask_base[targets.numpy()])]
                if len(zone_targets) > 0:
                    zone_node_id = zone_targets[0].item()
                    zone_list_idx = (zone_indices == zone_node_id).nonzero(as_tuple=True)[0]
                    if len(zone_list_idx) > 0:
                        asset_to_zone[asset_idx] = zone_list_idx[0]

        asset_to_zone[~asset_mask_base] = 0
        return {
            'asset_to_zone': asset_to_zone.to(device),
            'zone_to_region': zone_to_region,
        }

    def _generate_embedding(self, graph_path: str):
        """Generate zone-level embeddings from a graph file."""
        from torch_geometric.utils import add_self_loops

        graph_data = np.load(graph_path, allow_pickle=True)
        x = torch.from_numpy(graph_data['node_features']).float().to(self.device)
        edge_index = torch.from_numpy(graph_data['edge_index']).long().to(self.device)
        node_types = torch.from_numpy(graph_data['node_types']).long().to(self.device)

        meta = graph_data['meta'].item()
        N_base = meta['N_base']
        T = meta['T']

        edge_index_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        hierarchy = self._extract_hierarchy(graph_data, self.device)

        with torch.no_grad():
            embeddings = self.encoder(
                x, edge_index_loops, node_types,
                N_base, T,
                hierarchy_mapping=hierarchy,
                return_sequence=True,
            )

        n_zones = len(hierarchy['zone_to_region'])

        # Extract zone embeddings
        if isinstance(embeddings, dict) and 'zones' in embeddings and embeddings['zones'] is not None:
            zone_emb = embeddings['zones'].cpu()
        else:
            assets_emb = embeddings['assets'].cpu() if isinstance(embeddings, dict) else embeddings.cpu()
            asset_to_zone = hierarchy['asset_to_zone'].cpu()
            D = assets_emb.shape[-1]
            zone_emb = torch.zeros(n_zones, T, D)
            zone_counts = torch.zeros(n_zones, 1, 1)
            for ai in range(min(len(asset_to_zone), assets_emb.shape[0])):
                zi = asset_to_zone[ai].item()
                if zi < n_zones:
                    zone_emb[zi] += assets_emb[ai]
                    zone_counts[zi] += 1
            zone_counts = zone_counts.clamp(min=1)
            zone_emb = zone_emb / zone_counts

        return zone_emb, n_zones, T

    def _candidate_diversity_schedule(self, sample_idx: int, n_samples: int) -> Tuple[float, float]:
        """Return (temperature_scale, noise_scale) for a candidate index."""
        cfg = self.config
        if not getattr(cfg, 'diversity_priority', False) or n_samples <= 1:
            return 1.0, 1.0

        if getattr(cfg, 'diversity_schedule', 'linear') == 'staggered':
            alpha = 0.0 if sample_idx == 0 else sample_idx / max(1, n_samples - 1)
            alpha = min(1.0, max(0.0, alpha ** 0.75))
        else:
            alpha = sample_idx / max(1, n_samples - 1)

        temp_scale = float(cfg.diversity_temperature_scale_min) + alpha * (
            float(cfg.diversity_temperature_scale_max) - float(cfg.diversity_temperature_scale_min)
        )
        noise_scale = float(cfg.diversity_noise_scale_min) + alpha * (
            float(cfg.diversity_noise_scale_max) - float(cfg.diversity_noise_scale_min)
        )
        return max(1e-6, temp_scale), max(0.0, noise_scale)

    def _run_ebm_sampling(self, zone_emb: torch.Tensor, n_zones: int, T: int):
        """Run EBM + Langevin sampling to get binary candidates."""
        cfg = self.config
        Z = zone_emb.shape[0]

        # Pad to batch
        h_zt = zone_emb.unsqueeze(0).to(self.device)  # [1, Z, T, D]
        zone_mask = torch.ones(1, Z, device=self.device)

        # Generate multiple samples
        all_samples = []
        sample_times = []
        previous_samples: List[torch.Tensor] = []
        for sample_idx in range(cfg.n_samples):
            temp_scale, noise_scale = self._candidate_diversity_schedule(sample_idx, cfg.n_samples)
            t0 = time.perf_counter()
            u_bin = self.sampler.sample_binary(
                h_zt=h_zt,
                zone_mask=zone_mask,
                binarize=cfg.infer_binarize,
                threshold=cfg.infer_threshold,
                sample_temperature_scale=temp_scale,
                sample_noise_scale=noise_scale,
                diversity_refs=previous_samples if cfg.diversity_priority else None,
                diversity_lambda=(
                    float(cfg.diversity_similarity_lambda)
                    if cfg.diversity_priority and previous_samples
                    else 0.0
                ),
            )
            sample_times.append(time.perf_counter() - t0)
            previous_samples.append(u_bin.detach())
            all_samples.append(u_bin.squeeze(0).cpu())  # [Z, T, F]

        return all_samples, sample_times

    def _score_ebm_samples(self, samples: List[torch.Tensor], zone_emb: torch.Tensor) -> List[float]:
        """Evaluate EBM energies for sampled binary candidates."""
        if not samples:
            return []

        Z = zone_emb.shape[0]
        h_zt = zone_emb.unsqueeze(0).to(self.device)  # [1, Z, T, D]
        zone_mask = torch.ones(1, Z, device=self.device)

        prev_train_state = self.ebm.training
        self.ebm.eval()
        energies: List[float] = []
        with torch.no_grad():
            for sample in samples:
                u_bin = sample.unsqueeze(0).to(self.device).float()
                energy = self.ebm(u_bin, h_zt, zone_mask)
                energies.append(float(energy.reshape(-1)[0].item()))
        self.ebm.train(prev_train_state)
        return energies

    def _load_top_m_selector(self) -> Dict[str, Any]:
        if self.top_m_selector is not None:
            return self.top_m_selector
        selector_path = getattr(self.config, 'top_m_selector_path', None)
        if not selector_path:
            raise ValueError('top_m_score_mode="learned_selector" requires top_m_selector_path.')
        path = Path(selector_path)
        if not path.is_absolute():
            path = self.repo_path / path
        with path.open('rb') as f:
            selector = pickle.load(f)
        if not isinstance(selector, dict):
            raise TypeError(f'Top-M selector artifact must contain a dict, got {type(selector).__name__}.')
        required = {'feature_cols', 'model_columns', 'feature_medians', 'scaler'}
        missing = sorted(required - set(selector))
        if missing:
            raise ValueError(f'Top-M selector artifact is missing required keys: {missing}')
        self.top_m_selector = selector
        return selector

    def _learned_selector_scores(
        self,
        ebm_energies: List[float],
        active_fractions: List[Dict[str, float]],
        scenario_features: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        selector = self._load_top_m_selector()
        scenario_features = scenario_features or {}
        rows: List[Dict[str, Any]] = []
        n_candidates = max(len(ebm_energies), len(active_fractions))
        for sample_idx in range(n_candidates):
            stats = active_fractions[sample_idx] if sample_idx < len(active_fractions) else {}
            row: Dict[str, Any] = {
                'sample_idx': int(sample_idx),
                'family': str(scenario_features.get('family', '')),
                'criticality_index': float(scenario_features.get('criticality_index', float('nan'))),
                'n_zones': float(scenario_features.get('n_zones', float('nan'))),
                'n_timesteps': float(scenario_features.get('n_timesteps', float('nan'))),
                'sample_ebm_energy': (
                    float(ebm_energies[sample_idx])
                    if sample_idx < len(ebm_energies)
                    else float('nan')
                ),
                'sample_activation_rate': float('nan'),
            }
            activation_values = []
            if isinstance(stats, dict):
                for key, value in stats.items():
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        continue
                    row[f'active_{key}'] = value_f
                    if np.isfinite(value_f):
                        activation_values.append(value_f)
            if activation_values:
                row['sample_activation_rate'] = float(np.mean(activation_values))
            rows.append(row)

        import pandas as pd
        candidate_df = pd.DataFrame(rows)
        feature_cols = [col for col in selector.get('feature_cols', []) if col in candidate_df.columns]
        parts = []
        if feature_cols:
            parts.append(candidate_df[feature_cols].apply(pd.to_numeric, errors='coerce'))
        if 'family' in candidate_df.columns:
            parts.append(pd.get_dummies(candidate_df['family'].astype(str), prefix='family', dtype=float))
        X_raw = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=candidate_df.index)
        X = X_raw.reindex(columns=selector.get('model_columns', []), fill_value=0.0)
        medians = pd.Series(selector.get('feature_medians', {}), dtype=float)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        Xz = selector['scaler'].transform(X)

        feasibility_model = selector.get('feasibility_model')
        if feasibility_model is not None:
            p_feasible = feasibility_model.predict_proba(Xz)[:, 1]
        else:
            p_feasible = np.full(len(candidate_df), float(selector.get('feasibility_constant', 0.5)))

        gap_model = selector.get('gap_model')
        if gap_model is not None:
            pred_abs_gap = np.maximum(gap_model.predict(Xz), 0.0)
        else:
            pred_abs_gap = np.full(len(candidate_df), float(selector.get('gap_constant', 100.0)))

        lp_time_model = selector.get('lp_time_model')
        if lp_time_model is not None:
            pred_lp_time = np.maximum(lp_time_model.predict(Xz), 0.0)
        else:
            pred_lp_time = np.full(len(candidate_df), float(selector.get('lp_time_constant', 0.0)))

        scores = (
            pred_abs_gap
            + float(selector.get('failure_penalty', 100.0)) * (1.0 - p_feasible)
            + float(selector.get('time_penalty', 0.0)) * pred_lp_time
        )
        return [float(value) for value in scores]

    def _top_m_candidate_scores(
        self,
        samples: List[torch.Tensor],
        ebm_energies: List[float],
        active_fractions: List[Dict[str, float]],
        scenario_features: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        cfg = self.config
        mode = str(getattr(cfg, 'top_m_score_mode', 'energy') or 'energy')
        if mode not in {'energy', 'activation', 'energy_activation', 'first', 'learned_selector'}:
            raise ValueError(f'Unsupported top_m_score_mode: {mode}')
        if mode == 'learned_selector':
            return self._learned_selector_scores(ebm_energies, active_fractions, scenario_features)

        scores: List[float] = []
        for sample_idx, _sample in enumerate(samples):
            energy = (
                float(ebm_energies[sample_idx])
                if sample_idx < len(ebm_energies) and np.isfinite(ebm_energies[sample_idx])
                else float('inf')
            )
            stats = active_fractions[sample_idx] if sample_idx < len(active_fractions) else {}
            activation_values = []
            if isinstance(stats, dict):
                for value in stats.values():
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value_f):
                        activation_values.append(value_f)
            activation_rate = float(np.mean(activation_values)) if activation_values else float('nan')
            target_activation = getattr(cfg, 'top_m_target_activation_rate', None)
            if target_activation is None:
                target_activation = float(getattr(cfg, 'prior_p', 0.025) or 0.025)
            activation_penalty = (
                abs(activation_rate - float(target_activation))
                if np.isfinite(activation_rate)
                else float('inf')
            )

            if mode == 'first':
                score = float(sample_idx)
            elif mode == 'activation':
                score = activation_penalty
            elif mode == 'energy_activation':
                score = (
                    float(getattr(cfg, 'top_m_energy_weight', 1.0)) * energy
                    + float(getattr(cfg, 'top_m_activation_weight', 1.0)) * activation_penalty
                )
            else:
                score = energy
            scores.append(float(score))
        return scores

    def _top_m_project_indices(
        self,
        samples: List[torch.Tensor],
        ebm_energies: List[float],
        active_fractions: List[Dict[str, float]],
        scenario_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], List[float]]:
        if not getattr(self.config, 'top_m_enabled', False):
            scores = [
                float(ebm_energies[idx])
                if idx < len(ebm_energies) and np.isfinite(ebm_energies[idx])
                else float(idx)
                for idx in range(len(samples))
            ]
            return list(range(len(samples))), scores

        scores = self._top_m_candidate_scores(samples, ebm_energies, active_fractions, scenario_features)
        top_m = max(1, int(getattr(self.config, 'top_m', 1) or 1))
        top_m = min(top_m, len(samples))
        ranked = sorted(range(len(samples)), key=lambda idx: (scores[idx], idx))
        return ranked[:top_m], scores

    def _run_decoder(self, u_bin: torch.Tensor, scenario_path: Path):
        """Run pass-through feasibility warm-start on a binary sample."""
        return self._run_decoder_passthrough(u_bin, scenario_path)

    def _run_decoder_passthrough(self, u_bin: torch.Tensor, scenario_path: Path):
        """Run pass-through decoder: keep source binaries, build warm-start only."""
        from src.ebm.feasibility import (
            HierarchicalFeasibilityDecoder, load_physics_from_scenario,
        )

        sc_id = scenario_path.stem
        scenarios_dir = str(scenario_path.parent)
        physics = load_physics_from_scenario(sc_id, scenarios_dir)
        decoder = HierarchicalFeasibilityDecoder(physics)
        plan = decoder.decode_passthrough(u_bin)
        return plan, physics

    def _run_gnn_dispatch(self, u_bin: torch.Tensor, zone_emb: torch.Tensor, n_zones: int, scenario_id: str):
        """Run GNN dispatch predictor to get continuous dispatch directly.

        Args:
            u_bin:     [Z, T, F] binary candidate from EBM
            zone_emb:  [Z, T, D] zone-level HTE embeddings
            n_zones:   number of valid zones
            scenario_id: scenario identifier

        Returns:
            GNNDispatchResult with continuous_vars dict
        """
        zone_mask = torch.ones(n_zones)
        return self.gnn_predictor.predict(
            u_bin=u_bin[:n_zones],
            h_zt=zone_emb[:n_zones],
            zone_mask=zone_mask,
            scenario_id=scenario_id,
        )

    def _run_lp_worker(self, decoder_tensor: torch.Tensor, scenario_path: Path, scenarios_dir: Path, feasible_plan=None):
        """Run LP Worker Two-Stage on source binary tensor [Z, T, F].

        Args:
            decoder_tensor: Binary tensor [Z, T, 7] from the source method.
            scenario_path: Path to scenario JSON.
            scenarios_dir: Directory containing scenario JSONs.
            feasible_plan: Optional FeasiblePlan used only for LP warm-starting.
        """
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage

        cfg = self.config
        worker = LPWorkerTwoStage(
            scenarios_dir=str(scenarios_dir),
            solver_name=cfg.solver_name,
            slack_tol_mwh=cfg.slack_tol_mwh,
            deviation_penalty=cfg.deviation_penalty,
            flip_budget_20=cfg.flip_budget_repair_20,
            flip_budget_100=cfg.flip_budget_repair_100,
            verbose=False,
        )

        sc_id = scenario_path.stem
        result = worker.solve(
            sc_id,
            decoder_tensor,
            feasible_plan=feasible_plan,
            max_stages=int(getattr(cfg, 'lp_worker_max_stages', 5) or 5),
        )
        return result

    @staticmethod
    def _assign_lp_diagnostics(result: PipelineResult, lp_result: Any) -> None:
        """Copy final LP diagnostics, including deepest reached stage."""
        result.lp_status = str(getattr(lp_result, 'status', ''))
        stage_used = _normalize_stage_name(getattr(lp_result, 'stage_used', ''))
        result.lp_stage_used = _display_stage_name(lp_result, stage_used)
        result.lp_stage_reached = _deepest_stage_reached(lp_result)
        result.lp_objective = float(getattr(lp_result, 'objective_value', float('nan')))
        result.lp_slack = float(getattr(lp_result, 'slack_used', 0.0) or 0.0)
        result.lp_n_flips = int(getattr(lp_result, 'n_flips', 0) or 0)

        result.lp_slack_hard_fix = float(getattr(lp_result, 'slack_hard_fix', 0.0) or 0.0)
        result.lp_slack_repair_20 = float(getattr(lp_result, 'slack_repair_20', 0.0) or 0.0)
        result.lp_slack_repair_100 = float(getattr(lp_result, 'slack_repair_100', 0.0) or 0.0)
        result.lp_slack_full_soft = float(getattr(lp_result, 'slack_full_soft', 0.0) or 0.0)
        result.lp_slack_round_refix = float(getattr(lp_result, 'slack_round_refix', 0.0) or 0.0)

        flags = _extract_stage_flags(lp_result)
        result.lp_reached_hard_fix = flags['hard_fix']
        result.lp_reached_repair_20 = flags['repair_20']
        result.lp_reached_repair_100 = flags['repair_100']
        result.lp_reached_full_soft = flags['full_soft']
        result.lp_reached_round_refix = flags['round_refix']

    @staticmethod
    def _assign_selected_sample_times(result: PipelineResult, sample_idx: int) -> None:
        """Persist timing telemetry for the selected candidate."""
        if 0 <= sample_idx < len(result.all_sample_sampling_times):
            result.best_sample_sampling_time = float(result.all_sample_sampling_times[sample_idx])
        if 0 <= sample_idx < len(result.all_sample_decoder_times):
            result.best_sample_decoder_time = float(result.all_sample_decoder_times[sample_idx])
        if 0 <= sample_idx < len(result.all_sample_lp_solve_times):
            result.best_sample_lp_solve_time = float(result.all_sample_lp_solve_times[sample_idx])
        if 0 <= sample_idx < len(result.all_sample_total_times):
            result.best_sample_total_time = float(result.all_sample_total_times[sample_idx])
        if result.all_sample_total_times:
            result.first_sample_total_time = float(result.all_sample_total_times[0])

    def evaluate_scenario(
        self,
        scenario_path: Path,
        report_path: Path,
        graphs_dir: Path,
        family: str = '',
    ) -> PipelineResult:
        """Run full pipeline on a single scenario."""
        sc_id = scenario_path.stem
        result = PipelineResult(scenario_id=sc_id, family=family)

        # Load scenario metadata
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)
        result.criticality_index = scenario_data.get('criticality_index', 0.0)

        t_total_start = time.perf_counter()

        try:
            # Step 1: Build graph
            graph_path = graphs_dir / f"{sc_id}.npz"
            t0 = time.perf_counter()
            self._build_graph(scenario_path, report_path, graph_path)
            result.time_graph_build = time.perf_counter() - t0

            # Step 2: Generate embeddings
            t0 = time.perf_counter()
            zone_emb, n_zones, T = self._generate_embedding(str(graph_path))
            result.time_embedding = time.perf_counter() - t0
            result.n_zones = n_zones
            result.n_timesteps = T

            # Step 3: EBM + Langevin sampling
            t0 = time.perf_counter()
            all_samples, sample_sampling_times = self._run_ebm_sampling(zone_emb, n_zones, T)
            result.time_ebm_sampling = time.perf_counter() - t0
            result.n_samples = len(all_samples)
            result.all_sample_sampling_times = [float(x) for x in sample_sampling_times]
            result.all_binary_active_fractions = [_binary_stats(sample) for sample in all_samples]

            hamming_stats = _pairwise_hamming_stats(all_samples)
            result.pairwise_hamming_distances = hamming_stats['distances']
            result.pairwise_hamming_matrix = hamming_stats['matrix']
            result.sample_mean_pairwise_hamming = hamming_stats['mean']
            result.sample_max_pairwise_hamming = hamming_stats['max']
            result.sample_unique_ratio = hamming_stats['unique_ratio']
            result.all_sample_ebm_energies = self._score_ebm_samples(all_samples, zone_emb)
            finite_energies = [value for value in result.all_sample_ebm_energies if np.isfinite(value)]
            if finite_energies:
                result.sample_energy_mean = float(np.mean(finite_energies))
                result.sample_energy_std = float(np.std(finite_energies))

            projection_indices, candidate_scores = self._top_m_project_indices(
                all_samples,
                result.all_sample_ebm_energies,
                result.all_binary_active_fractions,
                scenario_features={
                    'family': family,
                    'criticality_index': result.criticality_index,
                    'n_zones': result.n_zones,
                    'n_timesteps': result.n_timesteps,
                },
            )
            projection_index_set = set(projection_indices)
            result.top_m_enabled = bool(getattr(self.config, 'top_m_enabled', False))
            result.top_m_score_mode = str(getattr(self.config, 'top_m_score_mode', ''))
            result.top_m_projected = int(len(projection_indices))
            result.top_m_selected_indices = [int(idx) for idx in projection_indices]
            result.top_m_skipped_indices = [
                int(idx) for idx in range(len(all_samples)) if idx not in projection_index_set
            ]
            result.top_m_candidate_scores = [float(value) for value in candidate_scores]
            result.n_candidates_generated = int(len(all_samples))
            result.n_candidates_projected = int(len(projection_indices))

            # Step 4+5: Optional pass-through warm-start + LP (or GNN dispatch)
            lp_results = []
            for sample_idx, u_bin in enumerate(all_samples):
                sample_decoder_time = 0.0
                sample_lp_time = 0.0
                if sample_idx not in projection_index_set:
                    result.all_objectives_raw.append(float('nan'))
                    result.all_objectives.append(float('inf'))
                    result.all_stages.append('top_m_skipped')
                    result.all_stages_reached.append('top_m_skipped')
                    result.all_sample_slacks.append(float('nan'))
                    result.all_sample_statuses.append('top_m_skipped')
                    sample_sampling_time = (
                        sample_sampling_times[sample_idx]
                        if sample_idx < len(sample_sampling_times)
                        else 0.0
                    )
                    result.all_sample_decoder_times.append(0.0)
                    result.all_sample_lp_solve_times.append(0.0)
                    result.all_sample_total_times.append(float(sample_sampling_time))
                    continue

                try:
                    sample_result = None
                    sample_stage = None
                    if self.config.use_gnn_dispatch and self.gnn_predictor is not None:
                        # GNN dispatch: replaces decoder + LP entirely
                        t1 = time.perf_counter()
                        gnn_result = self._run_gnn_dispatch(
                            u_bin, zone_emb, n_zones, sc_id,
                        )
                        sample_lp_time = time.perf_counter() - t1
                        sample_result = gnn_result
                        sample_stage = 'gnn_dispatch'
                    elif self.config.skip_decoder:
                        # Ablation: pass raw EBM binaries directly to LP
                        lp_tensor = (u_bin > 0.5).float()  # ensure crisp 0/1
                        t1 = time.perf_counter()
                        lp_result = self._run_lp_worker(
                            lp_tensor, scenario_path, scenario_path.parent,
                            feasible_plan=None,
                        )
                        sample_lp_time = time.perf_counter() - t1
                        sample_result = lp_result
                    else:
                        # Pass-through decoder: preserve EBM binaries, build LP warm-start only
                        t1 = time.perf_counter()
                        plan, physics = self._run_decoder_passthrough(u_bin, scenario_path)
                        sample_decoder_time = time.perf_counter() - t1

                        # LP Worker sees the original EBM binaries; the feasible
                        # plan is provided only as a continuous warm-start.
                        lp_tensor = (u_bin > 0.5).float()
                        t1 = time.perf_counter()
                        lp_result = self._run_lp_worker(
                            lp_tensor,
                            scenario_path,
                            scenario_path.parent,
                            feasible_plan=plan,
                        )
                        sample_lp_time = time.perf_counter() - t1
                        sample_result = lp_result

                    if sample_result is not None:
                        lp_results.append((sample_idx, sample_result))
                        result.all_objectives_raw.append(
                            float(getattr(sample_result, 'objective_value', float('inf')))
                        )
                        result.all_objectives.append(
                            float(getattr(sample_result, 'objective_value', float('inf')))
                        )
                        if sample_stage is None:
                            sample_stage_key = (
                                sample_result.stage_used.value
                                if hasattr(sample_result.stage_used, 'value')
                                else str(sample_result.stage_used)
                            )
                            sample_stage = _display_stage_name(sample_result, sample_stage_key)
                        result.all_stages.append(sample_stage)
                        result.all_stages_reached.append(_deepest_stage_reached(sample_result))
                        result.all_sample_slacks.append(
                            float(getattr(sample_result, 'slack_used', float('nan')) or 0.0)
                        )
                        result.all_sample_statuses.append(str(getattr(sample_result, 'status', '')))

                except Exception as e:
                    import logging
                    logging.warning(
                        f"[{sc_id}] sample {sample_idx} failed: {type(e).__name__}: {e}"
                    )
                    result.all_objectives.append(float('inf'))
                    result.all_stages.append('failed')
                    result.all_stages_reached.append('failed')
                    result.all_sample_slacks.append(float('nan'))
                    result.all_sample_statuses.append('error')
                finally:
                    result.all_sample_decoder_times.append(float(sample_decoder_time))
                    result.all_sample_lp_solve_times.append(float(sample_lp_time))
                    sample_sampling_time = (
                        sample_sampling_times[sample_idx]
                        if sample_idx < len(sample_sampling_times)
                        else 0.0
                    )
                    result.all_sample_total_times.append(
                        float(sample_sampling_time + sample_decoder_time + sample_lp_time)
                    )

            result.time_decoder = float(sum(result.all_sample_decoder_times))
            result.time_lp_solve = float(sum(result.all_sample_lp_solve_times))

            # Select best sample
            if lp_results:
                valid_results = [
                    (sample_idx, lr) for sample_idx, lr in lp_results
                    if hasattr(lr, 'objective_value') and lr.objective_value < float('inf')
                ]
                if valid_results:
                    best_idx, best_lr = min(valid_results, key=lambda x: x[1].objective_value)
                    result.best_sample_idx = best_idx
                    self._assign_lp_diagnostics(result, best_lr)
                    self._assign_selected_sample_times(result, best_idx)
                else:
                    diag_idx, diag_lr = max(
                        lp_results,
                        key=lambda x: _diagnostic_result_key(x[1]),
                    )
                    result.best_sample_idx = diag_idx
                    self._assign_lp_diagnostics(result, diag_lr)
                    self._assign_selected_sample_times(result, diag_idx)
                    result.success = False
            else:
                result.success = False

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        result.time_total = time.perf_counter() - t_total_start
        return result

    def evaluate_family(
        self,
        scenarios_dir: Path,
        reports_dir: Path,
        graphs_dir: Path,
        family_name: str = '',
        max_scenarios: Optional[int] = None,
    ) -> List[PipelineResult]:
        """Evaluate all scenarios in a family directory.
        
        IMPORTANT: When evaluating multiple families that share scenario IDs
        (e.g. scenario_00001), use a *separate* graphs_dir per family to
        avoid graph cache collisions (different families have different
        zone counts / topologies).
        """
        scenarios_dir = Path(scenarios_dir)
        reports_dir = Path(reports_dir)
        # Use per-family subdirectory to prevent graph cache collisions
        if family_name:
            graphs_dir = Path(graphs_dir) / family_name
        else:
            graphs_dir = Path(graphs_dir)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        scenario_files = sorted(scenarios_dir.glob('scenario_*.json'))
        report_files = {f.stem: f for f in reports_dir.glob('scenario_*.json')}

        # Match scenarios with reports
        pairs = []
        for sf in scenario_files:
            if sf.stem in report_files:
                pairs.append((sf, report_files[sf.stem]))

        if max_scenarios:
            pairs = pairs[:max_scenarios]

        print(f"\nEvaluating {len(pairs)} scenarios from {family_name or scenarios_dir.name}")

        results = []
        for sc_path, rp_path in tqdm(pairs, desc=f"Pipeline [{family_name}]"):
            result = self.evaluate_scenario(sc_path, rp_path, graphs_dir, family=family_name)
            results.append(result)

            if not result.success:
                print(f"  FAILED {result.scenario_id}: {result.error_message}")

        return results

    @staticmethod
    def save_results(results: List[PipelineResult], output_path: Path):
        """Save pipeline results to pickle file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump([asdict(r) for r in results], f)
        print(f"Saved {len(results)} results to {output_path}")

    @staticmethod
    def load_results(path: Path) -> List[Dict]:
        """Load pipeline results from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
