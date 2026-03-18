"""
Pipeline Runner: Full evaluation pipeline for GNN+EBM approach.

Runs the complete pipeline:
1. Build Hierarchical Temporal Graphs from scenario JSONs
2. Generate HTE Embeddings using trained encoder
3. Extract Zone-level Embeddings for EBM input
4. EBM + Normalized Temporal Langevin Sampler → binary candidates
5. Hierarchical Feasibility Decoder → enforce constraints
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

    # LP Worker
    solver_name: str = 'appsi_highs'
    slack_tol_mwh: float = 1.0
    deviation_penalty: float = 10000.0

    # Ablation flags
    skip_decoder: bool = False  # bypass decoder, pass raw EBM binaries to LP
    decoder_passthrough: bool = False  # decoder builds warm-start but keeps EBM binaries unchanged
    use_gnn_dispatch: bool = False  # replace decoder+LP with GNN dispatch predictor
    gnn_dispatch_path: str = 'outputs/gnn_dispatch/dispatch_gnn_best.pt'

    device: str = 'cuda'
    seed: int = 42


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
    lp_objective: float = float('nan')
    lp_slack: float = 0.0
    lp_n_flips: int = 0

    # All sample results
    n_samples: int = 0
    best_sample_idx: int = 0
    all_objectives: List[float] = field(default_factory=list)
    all_stages: List[str] = field(default_factory=list)

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
            temp_min=cfg.final_temp,
            temp_max=cfg.init_temp,
            init_mode='bernoulli',
            mode='infer',
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

    def _run_ebm_sampling(self, zone_emb: torch.Tensor, n_zones: int, T: int):
        """Run EBM + Langevin sampling to get binary candidates."""
        cfg = self.config
        Z = zone_emb.shape[0]

        # Pad to batch
        h_zt = zone_emb.unsqueeze(0).to(self.device)  # [1, Z, T, D]
        zone_mask = torch.ones(1, Z, device=self.device)

        # Generate multiple samples
        all_samples = []
        for _ in range(cfg.n_samples):
            u_bin = self.sampler.sample_binary(
                h_zt=h_zt,
                zone_mask=zone_mask,
                binarize='bernoulli',
                threshold=0.5,
            )
            all_samples.append(u_bin.squeeze(0).cpu())  # [Z, T, F]

        return all_samples

    def _run_decoder(self, u_bin: torch.Tensor, scenario_path: Path):
        """Run feasibility decoder on binary sample."""
        from src.ebm.feasibility import (
            HierarchicalFeasibilityDecoder, load_physics_from_scenario,
        )

        sc_id = scenario_path.stem
        scenarios_dir = str(scenario_path.parent)
        physics = load_physics_from_scenario(sc_id, scenarios_dir)
        decoder = HierarchicalFeasibilityDecoder(physics)
        plan = decoder.decode(u_bin)
        return plan, physics

    def _run_decoder_passthrough(self, u_bin: torch.Tensor, scenario_path: Path):
        """Run pass-through decoder: keep EBM binaries, build warm-start only."""
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
        """Run LP Worker Two-Stage on decoded binary tensor [Z, T, F].

        Args:
            decoder_tensor: Binary tensor [Z, T, 7] from decoder.
            scenario_path: Path to scenario JSON.
            scenarios_dir: Directory containing scenario JSONs.
            feasible_plan: Optional FeasiblePlan for LP warm-starting.
        """
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage

        cfg = self.config
        worker = LPWorkerTwoStage(
            scenarios_dir=str(scenarios_dir),
            solver_name=cfg.solver_name,
            slack_tol_mwh=cfg.slack_tol_mwh,
            deviation_penalty=cfg.deviation_penalty,
            verbose=False,
        )

        sc_id = scenario_path.stem
        result = worker.solve(sc_id, decoder_tensor, feasible_plan=feasible_plan)
        return result

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
            all_samples = self._run_ebm_sampling(zone_emb, n_zones, T)
            result.time_ebm_sampling = time.perf_counter() - t0
            result.n_samples = len(all_samples)

            # Step 4+5: Decoder + LP (or GNN dispatch) for each sample
            t0 = time.perf_counter()
            lp_results = []
            for sample_idx, u_bin in enumerate(all_samples):
                try:
                    if self.config.use_gnn_dispatch and self.gnn_predictor is not None:
                        # GNN dispatch: replaces decoder + LP entirely
                        gnn_result = self._run_gnn_dispatch(
                            u_bin, zone_emb, n_zones, sc_id,
                        )
                        lp_results.append(gnn_result)
                        result.all_objectives.append(gnn_result.objective_value)
                        result.all_stages.append('gnn_dispatch')
                        continue
                    elif self.config.skip_decoder:
                        # Ablation: pass raw EBM binaries directly to LP
                        lp_tensor = (u_bin > 0.5).float()  # ensure crisp 0/1
                        lp_result = self._run_lp_worker(
                            lp_tensor, scenario_path, scenario_path.parent,
                            feasible_plan=None,
                        )
                    elif self.config.decoder_passthrough:
                        # Ablation: decoder builds warm-start but keeps EBM binaries
                        plan, physics = self._run_decoder_passthrough(u_bin, scenario_path)
                        lp_tensor = (u_bin > 0.5).float()  # EBM binaries for LP fixing
                        lp_result = self._run_lp_worker(
                            lp_tensor, scenario_path, scenario_path.parent,
                            feasible_plan=plan,
                        )
                    else:
                        # Decoder
                        plan, physics = self._run_decoder(u_bin, scenario_path)

                        # Convert FeasiblePlan to tensor [Z, T, F] for LP worker
                        decoder_tensor = plan.to_tensor()  # [Z, T, 7]

                        # LP Worker (warm-started from decoder plan)
                        lp_result = self._run_lp_worker(decoder_tensor, scenario_path, scenario_path.parent, feasible_plan=plan)
                    lp_results.append(lp_result)

                    result.all_objectives.append(lp_result.objective_value)
                    stage_name = lp_result.stage_used.value if hasattr(lp_result.stage_used, 'value') else str(lp_result.stage_used)
                    result.all_stages.append(stage_name)

                except Exception as e:
                    import logging
                    logging.warning(
                        f"[{sc_id}] sample {sample_idx} failed: {type(e).__name__}: {e}"
                    )
                    result.all_objectives.append(float('inf'))
                    result.all_stages.append('failed')

            result.time_decoder = time.perf_counter() - t0 - sum(
                getattr(lr, 'solve_time', 0) for lr in lp_results if hasattr(lr, 'solve_time')
            )
            result.time_lp_solve = sum(
                getattr(lr, 'solve_time', 0) for lr in lp_results if hasattr(lr, 'solve_time')
            )

            # Select best sample
            if lp_results:
                valid_results = [
                    (i, lr) for i, lr in enumerate(lp_results)
                    if hasattr(lr, 'objective_value') and lr.objective_value < float('inf')
                ]
                if valid_results:
                    best_idx, best_lr = min(valid_results, key=lambda x: x[1].objective_value)
                    result.best_sample_idx = best_idx
                    result.lp_status = best_lr.status
                    result.lp_stage_used = best_lr.stage_used.value if hasattr(best_lr.stage_used, 'value') else str(best_lr.stage_used)
                    result.lp_objective = best_lr.objective_value
                    result.lp_slack = getattr(best_lr, 'slack_used', 0.0)
                    result.lp_n_flips = getattr(best_lr, 'n_flips', 0)

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
