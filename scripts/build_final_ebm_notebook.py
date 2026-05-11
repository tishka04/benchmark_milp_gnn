from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO_PATH = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_PATH / "notebooks" / "final_ebm_train.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md(
            """
            # Final EBM Train (Local RTX 4050)

            This notebook retrains the **EBM only** while reusing the frozen HTE embeddings.

            It combines:
            - the two-step gold -> silver pipeline from `Graph_EBM_v3_Colab.ipynb`
            - the ranking-driven **silver v2** fine-tuning recipe from `EBM_training_v4.ipynb`
            - **global scenario conditioning** inside the EBM with the following scalars:
              - `n_zones`
              - `criticality_index`
              - `demand_scale`
              - `peak_to_valley_ratio`
              - `vre_share_mean`
              - `vre_volatility_index`
              - `storage_adequacy_hours`
              - `dr_capacity_ratio`
              - `thermal_capacity_margin`
              - `nuclear_capacity_margin`

            Default setup:
            - HTE is frozen / reused
            - scenario scalars are normalized once on the gold+silver union
            - conditioning mode starts with **FiLM**
            - silver v2 keeps the current Stage-4-oriented LP ranking setup

            If you want the simplest ablation first, change `CONDITIONING_MODE = "concat"` in the config cell.
            """
        ),
        code(
            rf"""
            import os, sys, json, math, time, random, subprocess, importlib
            from copy import deepcopy
            from pathlib import Path
            from typing import Any, Dict, List, Tuple

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import torch
            import torch.nn as nn
            from IPython.display import display
            from torch.utils.data import Dataset, DataLoader
            from tqdm.auto import tqdm

            REPO_PATH = Path(r"{REPO_PATH}")
            assert REPO_PATH.exists(), REPO_PATH

            if str(REPO_PATH) not in sys.path:
                sys.path.insert(0, str(REPO_PATH))
            if str(REPO_PATH / 'src') not in sys.path:
                sys.path.insert(0, str(REPO_PATH / 'src'))

            torch.set_float32_matmul_precision('high')
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

            print(f"Python: {{sys.version.split()[0]}}")
            print(f"Torch: {{torch.__version__}}")
            print(f"CUDA available: {{torch.cuda.is_available()}}")
            if torch.cuda.is_available():
                print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
                print(f"CUDA memory: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}} GB")
            else:
                raise RuntimeError(
                    'This notebook is designed for local GPU training. '
                    'Install a CUDA-enabled PyTorch wheel in the active environment first.'
                )
            """
        ),
        code(
            """
            # Clear cached repo modules so reruns pick up local source edits.
            for module_name in [m for m in list(sys.modules) if m.startswith('src.')]:
                del sys.modules[module_name]

            from src.ebm.config_v3 import EBMv3Config
            from src.ebm.dataset_v3 import ScenarioReportDataset, temporal_collate_fn, load_classification_index
            from src.ebm.model_v3 import TrajectoryZonalEBM
            from src.ebm.sampler_v3 import NormalizedTemporalLangevinSampler
            from src.ebm.loss_v3 import ContrastiveDivergenceLoss, CombinedLoss
            from src.ebm.train_v3 import (
                train_epoch_gold,
                validate_gold,
                train_epoch_silver,
                validate_silver_ranking,
                _get_langevin_train_steps,
                _get_silver_langevin_steps,
                _save_checkpoint,
                _save_history,
                _validate_lp_scenario_inputs,
                _fmt_float,
                _fmt_pct,
            )
            from src.eval.advanced_analysis import extract_physical_complexity_features
            from src.analysis.criticality_index import compute_criticality
            from src.milp.scenario_loader import load_scenario_data
            from src.milp.lp_worker_two_stage import LPWorkerTwoStage

            EXPERIMENT_NAME = 'final_ebm_conditioned_film'
            CONDITIONING_MODE = 'film'   # 'film' or 'concat'
            BASE_EMBED_DIM = 128
            SCENARIO_FEATURE_NAMES = [
                'n_zones',
                'criticality_index',
                'demand_scale',
                'peak_to_valley_ratio',
                'vre_share_mean',
                'vre_volatility_index',
                'storage_adequacy_hours',
                'dr_capacity_ratio',
                'thermal_capacity_margin',
                'nuclear_capacity_margin',
            ]
            RUN_DIR = REPO_PATH / 'outputs' / 'ebm_models' / EXPERIMENT_NAME
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            FEATURE_CACHE_PATH = RUN_DIR / 'scenario_feature_cache.json'
            FEATURE_STATS_PATH = RUN_DIR / 'scenario_feature_stats.json'

            # RTX 4050 starting point. If you hit CUDA OOM, reduce:
            # - config.batch_size from 24 -> 16 or 12
            # - silver_lp_scenarios_per_batch from 2 -> 1
            # - silver_lp_candidates_per_scenario from 8 -> 6 or 4
            config = EBMv3Config(
                base_dir=str(REPO_PATH),
                n_timesteps=24,
                n_features=7,
                embed_dim=BASE_EMBED_DIM,
                hidden_dim=256,
                gru_layers=3,
                bidirectional=True,
                dropout=0.1,
                use_peak_term=True,
                peak_tau=0.5,
                peak_weight=0.3,
                energy_max=50.0,
                langevin_steps=100,
                langevin_train_steps_start=10,
                langevin_train_steps_end=50,
                langevin_step_size=0.05,
                langevin_noise=0.50,
                langevin_temp_max=1.0,
                langevin_temp_min=0.1,
                langevin_init_mode='soft',
                langevin_prior_p=0.025,
                langevin_prior_strength=0.0,
                langevin_normalize_grad=True,
                langevin_ratio_start=1.0,
                langevin_ratio_end=1.0,
                random_neg_sparsity=0.025,
                corruption_flip_rate=0.05,
                batch_size=24,
                num_workers=0,
                use_amp=True,
                seed=42,
                gold_epochs=40,
                gold_lr=2e-5,
                gold_patience=8,
                silver_epochs=20,
                silver_lr=1e-5,
                silver_patience=6,
                silver_min_delta=0.01,
                silver_langevin_start=20,
                silver_langevin_end=35,
                silver_lp_eval_every=5,
                silver_lp_scenarios_per_batch=2,
                silver_preference_margin=0.1,
                silver_lambda_cd=1.0,
                silver_lambda_pref=0.5,
            )
            config.device = 'cuda'

            SILVER_STAGE1_PREFIX = 'silver_stage1'
            SILVER_V2_PREFIX = 'silver_v2'
            SILVER_STAGE1_OVERRIDES = dict(
                silver_lambda_cd=0.1,
                silver_lambda_pref=2.0,
                silver_pref_warmup_epochs=0,
                silver_lp_eval_every=5,
                silver_lp_scenarios_per_batch=2,
                silver_lp_candidates_per_scenario=4,
                silver_lp_max_stages=4,
                silver_lp_incumbent_candidates=1,
                silver_lp_oracle_langevin_candidates=1,
                silver_lp_corrupt_candidates=1,
                silver_lp_langevin_candidates=1,
                silver_pref_all_informative_pairs=True,
                silver_pref_max_pairs_per_scenario=4,
                silver_pref_min_relative_gap=0.10,
                silver_pref_slack_weight=300.0,
                silver_pref_repair_weight=0.1,
                silver_early_stop_metric='val_pref_accuracy',
                silver_val_ranking_scenarios=24,
                silver_val_candidates_per_scenario=4,
                silver_val_max_stages=4,
                silver_log_individual_pairs=False,
                silver_pair_coverage_floor=0.20,
            )
            SILVER_V2_OVERRIDES = dict(
                silver_epochs=10,
                silver_lr=5e-6,
                silver_patience=5,
                silver_min_delta=0.005,
                silver_lambda_cd=0.1,
                silver_lambda_pref=2.0,
                silver_early_stop_metric='val_pref_accuracy',
                silver_pref_warmup_epochs=0,
                silver_lp_eval_every=5,
                silver_lp_scenarios_per_batch=2,
                silver_lp_candidates_per_scenario=4,
                silver_lp_max_stages=4,
                silver_lp_incumbent_candidates=1,
                silver_lp_oracle_langevin_candidates=1,
                silver_lp_corrupt_candidates=1,
                silver_lp_langevin_candidates=1,
                silver_pref_all_informative_pairs=True,
                silver_pref_max_pairs_per_scenario=4,
                silver_pref_min_relative_gap=0.10,
                silver_pref_slack_weight=300.0,
                silver_pref_repair_weight=0.1,
                silver_pref_margin_mode='scaled_gap',
                silver_preference_margin=0.5,
                silver_pref_margin_rel_gap_cap=2.0,
                silver_val_ranking_scenarios=24,
                silver_val_candidates_per_scenario=4,
                silver_val_max_stages=4,
                silver_log_individual_pairs=False,
                silver_pair_coverage_floor=0.20,
            )

            RUN_GOLD = True
            RUN_SILVER_STAGE1 = True
            RUN_SILVER_V2 = True

            print(f"Experiment: {EXPERIMENT_NAME}")
            print(f"Conditioning mode: {CONDITIONING_MODE}")
            print(f"Scenario features ({len(SCENARIO_FEATURE_NAMES)}): {SCENARIO_FEATURE_NAMES}")
            print(f"Run dir: {RUN_DIR}")
            print(f"Reports dir: {config.reports_dir}")
            print(f"Embeddings dir: {config.embeddings_dir}")
            print(f"Raw scenarios dir: {config.scenarios_dir}")
            print(f"Device: {config.device}")
            """
        ),
        code(
            """
            def extract_conditioning_features(scenario_path: Path) -> Dict[str, float]:
                scenario_path = Path(scenario_path)
                with scenario_path.open('r', encoding='utf-8') as f:
                    scenario_json = json.load(f)

                physical = extract_physical_complexity_features(scenario_path)
                scenario_data = load_scenario_data(scenario_path)

                criticality_value = scenario_json.get('criticality_index', None)
                try:
                    criticality_value = float(criticality_value)
                except (TypeError, ValueError):
                    criticality_value = float('nan')
                if not np.isfinite(criticality_value):
                    criticality_value = float(compute_criticality(scenario_json).criticality_index)

                peak_demand_mw = float(physical['peak_demand_mw'])
                thermal_capacity = float(sum(scenario_data.thermal_capacity.values()))
                nuclear_capacity = float(sum(scenario_data.nuclear_capacity.values()))
                demand_scale = float(
                    scenario_json.get('exogenous', {}).get(
                        'demand_scale_factor',
                        scenario_json.get('meta', {}).get('demand_scale_factor', 1.0),
                    )
                )

                return {
                    'n_zones': float(physical['n_zones']),
                    'criticality_index': criticality_value,
                    'demand_scale': demand_scale,
                    'peak_to_valley_ratio': float(physical['peak_to_valley_ratio']),
                    'vre_share_mean': float(physical['vre_share_mean']),
                    'vre_volatility_index': float(physical['vre_volatility_index']),
                    'storage_adequacy_hours': float(physical['storage_adequacy_hours']),
                    'dr_capacity_ratio': float(physical['dr_capacity_ratio']),
                    'thermal_capacity_margin': thermal_capacity / max(peak_demand_mw, 1e-6),
                    'nuclear_capacity_margin': nuclear_capacity / max(peak_demand_mw, 1e-6),
                }


            def load_feature_cache(cache_path: Path = FEATURE_CACHE_PATH) -> Dict[str, Dict[str, float]]:
                cache_path = Path(cache_path)
                if not cache_path.exists():
                    return {}
                with cache_path.open('r', encoding='utf-8') as f:
                    return json.load(f)


            def save_feature_cache(cache: Dict[str, Dict[str, float]], cache_path: Path = FEATURE_CACHE_PATH) -> None:
                cache_path = Path(cache_path)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open('w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2)


            def build_feature_cache_for_ids(
                scenario_ids: List[str],
                scenarios_dir: Path,
                cache_path: Path = FEATURE_CACHE_PATH,
                refresh: bool = False,
            ) -> Dict[str, Dict[str, float]]:
                cache = {} if refresh else load_feature_cache(cache_path)
                scenario_ids = [str(sid) for sid in scenario_ids]
                missing = [sid for sid in scenario_ids if sid not in cache]

                if missing:
                    print(f"Computing scenario scalars for {len(missing)} scenarios...")
                    for scenario_id in tqdm(missing):
                        scenario_path = Path(scenarios_dir) / f"{scenario_id}.json"
                        cache[scenario_id] = extract_conditioning_features(scenario_path)
                    save_feature_cache(cache, cache_path)

                return {sid: cache[sid] for sid in scenario_ids}


            def fit_feature_stats(
                feature_cache: Dict[str, Dict[str, float]],
                scenario_ids: List[str],
                feature_names: List[str] = SCENARIO_FEATURE_NAMES,
            ) -> Dict[str, Any]:
                arr = np.asarray(
                    [
                        [float(feature_cache[sid][name]) for name in feature_names]
                        for sid in scenario_ids
                    ],
                    dtype=np.float32,
                )
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                std = np.where(std < 1e-6, 1.0, std)
                return {
                    'feature_names': list(feature_names),
                    'mean': mean,
                    'std': std,
                }


            def feature_stats_to_frame(stats: Dict[str, Any]) -> pd.DataFrame:
                return pd.DataFrame({
                    'feature': stats['feature_names'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                })


            def save_feature_stats(stats: Dict[str, Any], stats_path: Path = FEATURE_STATS_PATH) -> None:
                payload = {
                    name: {
                        'mean': float(stats['mean'][idx]),
                        'std': float(stats['std'][idx]),
                    }
                    for idx, name in enumerate(stats['feature_names'])
                }
                with Path(stats_path).open('w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)


            def build_global_feature_bundle(
                config: EBMv3Config,
                tiers: Tuple[str, ...] = ('gold', 'silver'),
                cache_path: Path = FEATURE_CACHE_PATH,
                stats_path: Path = FEATURE_STATS_PATH,
            ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
                index = load_classification_index(config.classification_index_path)
                scenario_ids = sorted({
                    fname.replace('.json', '')
                    for tier in tiers
                    for fname in index.get(tier, [])
                })
                feature_cache = build_feature_cache_for_ids(scenario_ids, Path(config.scenarios_dir), cache_path=cache_path)
                feature_stats = fit_feature_stats(feature_cache, scenario_ids)
                save_feature_stats(feature_stats, stats_path)
                return feature_stats, feature_cache


            class ScenarioConditionedSubset(Dataset):
                def __init__(
                    self,
                    base_dataset: ScenarioReportDataset,
                    indices: List[int],
                    feature_cache: Dict[str, Dict[str, float]],
                    feature_stats: Dict[str, Any],
                    feature_names: List[str] = SCENARIO_FEATURE_NAMES,
                ):
                    self.base_dataset = base_dataset
                    self.indices = list(indices)
                    self.feature_cache = feature_cache
                    self.feature_names = list(feature_names)
                    self.mean = np.asarray(feature_stats['mean'], dtype=np.float32)
                    self.std = np.asarray(feature_stats['std'], dtype=np.float32)

                def __len__(self) -> int:
                    return len(self.indices)

                def __getitem__(self, idx: int) -> Dict[str, Any]:
                    item = dict(self.base_dataset[self.indices[idx]])
                    scenario_id = item['scenario_id']
                    raw = np.asarray(
                        [float(self.feature_cache[scenario_id][name]) for name in self.feature_names],
                        dtype=np.float32,
                    )
                    norm = (raw - self.mean) / self.std
                    scenario_features = torch.tensor(norm, dtype=torch.float32)
                    s_broadcast = scenario_features.view(1, 1, -1).expand(
                        item['n_zones'], item['n_timesteps'], len(self.feature_names)
                    )
                    item['h_zt'] = torch.cat([item['h_zt'], s_broadcast], dim=-1).float()
                    item['scenario_features'] = scenario_features
                    return item


            def conditioned_temporal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
                out = temporal_collate_fn(batch)
                if 'scenario_features' in batch[0]:
                    out['scenario_features'] = torch.stack([d['scenario_features'] for d in batch])
                return out


            def split_indices(n_items: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
                generator = torch.Generator().manual_seed(seed)
                perm = torch.randperm(n_items, generator=generator).tolist()
                val_size = max(1, int(n_items * val_split))
                val_indices = perm[:val_size]
                train_indices = perm[val_size:]
                return train_indices, val_indices


            def build_conditioned_dataloaders(
                config: EBMv3Config,
                tier: str = 'gold',
                batch_size: int = None,
                feature_stats: Dict[str, Any] = None,
                feature_cache: Dict[str, Dict[str, float]] = None,
            ):
                index = load_classification_index(config.classification_index_path)
                scenario_files = index.get(tier, [])
                print(f"Building conditioned {tier} dataset: {len(scenario_files)} scenarios in index")

                base_dataset = ScenarioReportDataset(
                    reports_dir=config.reports_dir,
                    embeddings_dir=config.embeddings_dir,
                    scenario_files=scenario_files,
                    n_timesteps=config.n_timesteps,
                    embed_dim=BASE_EMBED_DIM,
                )
                print(f"  Valid scenarios with reports + embeddings: {len(base_dataset)}")
                if len(base_dataset) == 0:
                    raise ValueError(f"No valid {tier} scenarios found")

                valid_ids = [entry['scenario_id'] for entry in base_dataset.valid_scenarios]
                if feature_cache is None:
                    feature_cache = build_feature_cache_for_ids(valid_ids, Path(config.scenarios_dir))
                if feature_stats is None:
                    feature_stats = fit_feature_stats(feature_cache, valid_ids)

                train_indices, val_indices = split_indices(len(base_dataset), config.val_split, config.seed)
                train_ds = ScenarioConditionedSubset(base_dataset, train_indices, feature_cache, feature_stats)
                val_ds = ScenarioConditionedSubset(base_dataset, val_indices, feature_cache, feature_stats)

                effective_batch_size = batch_size or config.batch_size
                pin_memory = config.device == 'cuda'
                train_loader = DataLoader(
                    train_ds,
                    batch_size=effective_batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=pin_memory,
                    collate_fn=conditioned_temporal_collate_fn,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=effective_batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=pin_memory,
                    collate_fn=conditioned_temporal_collate_fn,
                )

                print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
                print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")
                return train_loader, val_loader, base_dataset, feature_stats, feature_cache


            global_feature_stats, global_feature_cache = build_global_feature_bundle(config)
            print(feature_stats_to_frame(global_feature_stats))

            gold_train_loader, gold_val_loader, gold_base_dataset, _, _ = build_conditioned_dataloaders(
                config,
                tier='gold',
                feature_stats=global_feature_stats,
                feature_cache=global_feature_cache,
            )
            batch = next(iter(gold_train_loader))
            print(f"u_zt: {tuple(batch['u_zt'].shape)}")
            print(f"h_zt (augmented): {tuple(batch['h_zt'].shape)}")
            print(f"scenario_features: {tuple(batch['scenario_features'].shape)}")
            print(f"first normalized scenario feature vector: {batch['scenario_features'][0]}")
            """
        ),
        code(
            """
            class ScenarioConditionedEBM(nn.Module):
                def __init__(
                    self,
                    base_embed_dim: int,
                    scenario_dim: int,
                    conditioning_mode: str,
                    n_features: int,
                    hidden_dim: int = 128,
                    gru_layers: int = 2,
                    bidirectional: bool = True,
                    dropout: float = 0.1,
                    use_peak_term: bool = True,
                    peak_tau: float = 0.5,
                    peak_weight: float = 0.3,
                    energy_max: float = 50.0,
                ):
                    super().__init__()
                    self.base_embed_dim = int(base_embed_dim)
                    self.scenario_dim = int(scenario_dim)
                    self.conditioning_mode = str(conditioning_mode)

                    if self.conditioning_mode not in {'film', 'concat'}:
                        raise ValueError(f"Unsupported conditioning mode: {self.conditioning_mode}")

                    backbone_embed_dim = (
                        self.base_embed_dim + self.scenario_dim
                        if self.conditioning_mode == 'concat'
                        else self.base_embed_dim
                    )
                    self.backbone = TrajectoryZonalEBM(
                        embed_dim=backbone_embed_dim,
                        n_features=n_features,
                        hidden_dim=hidden_dim,
                        gru_layers=gru_layers,
                        bidirectional=bidirectional,
                        dropout=dropout,
                        use_peak_term=use_peak_term,
                        peak_tau=peak_tau,
                        peak_weight=peak_weight,
                        energy_max=energy_max,
                    )
                    if self.conditioning_mode == 'film':
                        self.scenario_mlp = nn.Sequential(
                            nn.Linear(self.scenario_dim, hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, 2 * self.base_embed_dim),
                        )

                def _split_augmented_embedding(self, h_zt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                    if h_zt.shape[-1] != self.base_embed_dim + self.scenario_dim:
                        raise ValueError(
                            f"Expected augmented embedding dim {self.base_embed_dim + self.scenario_dim}, "
                            f"got {h_zt.shape[-1]}"
                        )
                    h_local = h_zt[..., :self.base_embed_dim]
                    scenario_vec = h_zt[:, 0, 0, self.base_embed_dim:]
                    return h_local, scenario_vec

                def forward(self, u_zt: torch.Tensor, h_zt: torch.Tensor, zone_mask: torch.Tensor = None) -> torch.Tensor:
                    if self.scenario_dim == 0:
                        return self.backbone(u_zt, h_zt, zone_mask)
                    if self.conditioning_mode == 'concat':
                        return self.backbone(u_zt, h_zt, zone_mask)

                    h_local, scenario_vec = self._split_augmented_embedding(h_zt)
                    gamma, beta = self.scenario_mlp(scenario_vec).chunk(2, dim=-1)
                    gamma = 1.0 + 0.1 * torch.tanh(gamma)
                    beta = 0.1 * torch.tanh(beta)
                    h_cond = gamma[:, None, None, :] * h_local + beta[:, None, None, :]
                    return self.backbone(u_zt, h_cond, zone_mask)


            def build_conditioned_model(config: EBMv3Config) -> nn.Module:
                model = ScenarioConditionedEBM(
                    base_embed_dim=BASE_EMBED_DIM,
                    scenario_dim=len(SCENARIO_FEATURE_NAMES),
                    conditioning_mode=CONDITIONING_MODE,
                    n_features=config.n_features,
                    hidden_dim=config.hidden_dim,
                    gru_layers=config.gru_layers,
                    bidirectional=config.bidirectional,
                    dropout=config.dropout,
                    use_peak_term=config.use_peak_term,
                    peak_tau=config.peak_tau,
                    peak_weight=config.peak_weight,
                    energy_max=config.energy_max,
                )
                return model.to(config.device)


            model = build_conditioned_model(config)
            with torch.no_grad():
                sample_energy = model(
                    batch['u_zt'].to(config.device),
                    batch['h_zt'].to(config.device),
                    batch['zone_mask'].to(config.device),
                )
            print(model)
            print(f"Conditioned model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Energy output shape: {tuple(sample_energy.shape)}")
            """
        ),
        code(
            """
            def load_conditioned_checkpoint(checkpoint_path: Path, config: EBMv3Config):
                checkpoint_path = Path(checkpoint_path)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(checkpoint_path)
                model = build_conditioned_model(config)
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                model.load_state_dict(state_dict)
                model = model.to(config.device)
                model.eval()
                print(f"Loaded checkpoint: {checkpoint_path.name}")
                return model, ckpt


            def run_conditioned_gold_pretraining(
                config: EBMv3Config,
                run_dir: Path,
                feature_stats: Dict[str, Any],
                feature_cache: Dict[str, Dict[str, float]],
            ):
                print('=' * 80)
                print('STEP A: CONDITIONED GOLD PRE-TRAINING')
                print('=' * 80)

                train_loader, val_loader, _base_dataset, _stats, _cache = build_conditioned_dataloaders(
                    config,
                    tier='gold',
                    feature_stats=feature_stats,
                    feature_cache=feature_cache,
                )

                model = build_conditioned_model(config)
                loss_fn = ContrastiveDivergenceLoss(alpha_reg=0.01)
                sampler = NormalizedTemporalLangevinSampler(
                    model=model,
                    n_features=config.n_features,
                    num_steps=_get_langevin_train_steps(0, config.gold_epochs, config),
                    step_size=config.langevin_step_size,
                    noise_scale=config.langevin_noise,
                    temp_max=config.langevin_temp_max,
                    temp_min=config.langevin_temp_min,
                    init_mode=config.langevin_init_mode,
                    prior_p=config.langevin_prior_p,
                    prior_strength=config.langevin_prior_strength,
                    normalize_grad=config.langevin_normalize_grad,
                    device=config.device,
                    mode='train',
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.gold_lr, weight_decay=config.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.gold_epochs)
                scaler = torch.amp.GradScaler('cuda') if config.use_amp and config.device == 'cuda' else None

                history = {'train': [], 'val': []}
                best_val_gap = -float('inf')
                patience_counter = 0

                for epoch in range(config.gold_epochs):
                    t0 = time.time()
                    curr_steps = _get_langevin_train_steps(epoch, config.gold_epochs, config)
                    sampler.set_num_steps(curr_steps)

                    train_metrics = train_epoch_gold(model, sampler, train_loader, optimizer, loss_fn, config, scaler, epoch)
                    val_metrics = validate_gold(model, sampler, val_loader, loss_fn, config)
                    scheduler.step()

                    dt = time.time() - t0
                    history['train'].append(train_metrics)
                    history['val'].append(val_metrics)

                    val_gap_rand = val_metrics.get('E_gap_rand', 0.0)
                    val_gap_lang = val_metrics.get('E_gap_lang', float('nan'))
                    print(
                        f"Epoch {epoch+1}/{config.gold_epochs} ({dt:.1f}s) | "
                        f"CD={train_metrics.get('cd_loss', 0.0):.4f} | "
                        f"ValGap_R={_fmt_float(val_gap_rand, 4)} | "
                        f"ValGap_L={_fmt_float(val_gap_lang, 4)} | "
                        f"LR={scheduler.get_last_lr()[0]:.2e} | "
                        f"Lsteps={curr_steps}"
                    )

                    early_stop_gap = val_gap_lang if math.isfinite(val_gap_lang) else val_gap_rand
                    if early_stop_gap > best_val_gap:
                        best_val_gap = early_stop_gap
                        patience_counter = 0
                        _save_checkpoint(model, optimizer, epoch, val_metrics, str(run_dir / 'gold_best.pt'))
                        print(f"  >> New best gold gap: {best_val_gap:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= config.gold_patience:
                            print(f"  Early stopping at epoch {epoch+1}")
                            break

                    if (epoch + 1) % config.save_every_epoch == 0:
                        _save_checkpoint(model, optimizer, epoch, val_metrics, str(run_dir / f'gold_epoch_{epoch+1}.pt'))

                best_path = run_dir / 'gold_best.pt'
                if best_path.exists():
                    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model = model.to(config.device)
                _save_history(history, str(run_dir / 'gold_history.json'))
                return model, history


            def run_conditioned_silver_finetuning(
                model: nn.Module,
                config: EBMv3Config,
                run_dir: Path,
                prefix: str,
                feature_stats: Dict[str, Any],
                feature_cache: Dict[str, Dict[str, float]],
            ):
                print('=' * 80)
                print(f'STEP B: CONDITIONED SILVER FINE-TUNING [{prefix}]')
                print('=' * 80)

                train_loader, val_loader, base_dataset, _stats, _cache = build_conditioned_dataloaders(
                    config,
                    tier='silver',
                    feature_stats=feature_stats,
                    feature_cache=feature_cache,
                )
                _validate_lp_scenario_inputs(config, base_dataset)

                sampler = NormalizedTemporalLangevinSampler(
                    model=model,
                    n_features=config.n_features,
                    num_steps=_get_silver_langevin_steps(0, config.silver_epochs, config),
                    step_size=config.langevin_step_size,
                    noise_scale=config.langevin_noise,
                    temp_max=config.langevin_temp_max,
                    temp_min=config.langevin_temp_min,
                    init_mode=config.langevin_init_mode,
                    prior_p=config.langevin_prior_p,
                    prior_strength=config.langevin_prior_strength,
                    normalize_grad=config.langevin_normalize_grad,
                    device=config.device,
                    mode='train',
                )
                print(
                    f"Silver sampler: {config.silver_langevin_start}->{config.silver_langevin_end} steps | "
                    f"step_size={config.langevin_step_size} | noise={config.langevin_noise}"
                )
                print(
                    'Silver LP score: '
                    f"J = cost + {float(getattr(config, 'silver_pref_slack_weight', 0.0)):.3g} * slack + "
                    f"{float(getattr(config, 'silver_pref_repair_weight', 0.0)):.3g} * "
                    f"{getattr(config, 'silver_pref_repair_metric', 'decoder_deviation')}"
                )

                loss_fn = CombinedLoss(
                    lambda_cd=config.silver_lambda_cd,
                    lambda_pref=config.silver_lambda_pref,
                    margin=config.silver_preference_margin,
                    alpha_reg=0.01,
                )
                cd_loss_fn = ContrastiveDivergenceLoss(alpha_reg=0.01)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.silver_lr, weight_decay=config.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.silver_epochs)
                scaler = torch.amp.GradScaler('cuda') if config.use_amp and config.device == 'cuda' else None

                lp_worker = LPWorkerTwoStage(
                    scenarios_dir=config.scenarios_dir,
                    solver_name=config.lp_solver,
                    verbose=False,
                )
                print('LP worker cached for conditioned silver fine-tuning')

                early_stop_metric = getattr(config, 'silver_early_stop_metric', 'val_gap_lang')
                ranking_val_scenarios = int(getattr(config, 'silver_val_ranking_scenarios', 0))
                ranking_val_candidates = int(
                    getattr(config, 'silver_val_candidates_per_scenario', getattr(config, 'silver_lp_candidates_per_scenario', 2))
                )
                if early_stop_metric in {'val_pref_accuracy', 'val_spearman'} and ranking_val_scenarios <= 0:
                    raise ValueError(
                        f"silver_early_stop_metric={early_stop_metric!r} requires "
                        "silver_val_ranking_scenarios > 0."
                    )
                if ranking_val_scenarios > 0:
                    print(
                        f"Silver ranking validation: {ranking_val_scenarios} scenarios x "
                        f"{ranking_val_candidates} candidates | early_stop={early_stop_metric}"
                    )
                else:
                    print(f"Silver ranking validation disabled | early_stop={early_stop_metric}")

                history = {'train': [], 'val': []}
                best_val_metric = -float('inf')
                patience_counter = 0
                warmup_epochs = getattr(config, 'silver_pref_warmup_epochs', 5)
                min_delta = getattr(config, 'silver_min_delta', 0.01)

                for epoch in range(config.silver_epochs):
                    t0 = time.time()
                    curr_steps = _get_silver_langevin_steps(epoch, config.silver_epochs, config)
                    sampler.set_num_steps(curr_steps)

                    if warmup_epochs > 0 and epoch < warmup_epochs:
                        loss_fn.lambda_pref = config.silver_lambda_pref * (epoch / warmup_epochs)
                    else:
                        loss_fn.lambda_pref = config.silver_lambda_pref

                    train_metrics = train_epoch_silver(
                        model,
                        sampler,
                        train_loader,
                        optimizer,
                        loss_fn,
                        config,
                        scaler,
                        epoch,
                        lp_worker=lp_worker,
                    )
                    val_metrics = validate_gold(model, sampler, val_loader, cd_loss_fn, config)
                    val_metrics.update(validate_silver_ranking(model, sampler, val_loader, config, lp_worker=lp_worker))
                    scheduler.step()

                    dt = time.time() - t0
                    history['train'].append(train_metrics)
                    history['val'].append(val_metrics)

                    val_gap_rand = val_metrics.get('E_gap_rand', 0.0)
                    val_gap_lang = val_metrics.get('E_gap_lang', float('nan'))
                    val_pref_acc = val_metrics.get('val_pref_accuracy', float('nan'))
                    val_spearman = val_metrics.get('val_spearman_energy_J', val_metrics.get('val_spearman', float('nan')))
                    val_bestofk_gap = val_metrics.get('val_best_of_K_gap', val_metrics.get('val_bestofk_gap', float('nan')))
                    train_pref_loss = train_metrics.get('loss_pref_weighted', float('nan'))
                    train_pref_acc = train_metrics.get('train_pref_accuracy', float('nan'))
                    n_pairs_total = train_metrics.get('n_pairs_total', 0.0)
                    n_lp_scenarios_attempted = train_metrics.get('n_lp_scenarios_attempted', 0.0)
                    n_lp_scenarios_with_pairs = train_metrics.get('n_lp_scenarios_with_pairs', 0.0)
                    pair_coverage = train_metrics.get('pair_coverage', float('nan'))
                    pct_non_finite = train_metrics.get('pct_non_finite_candidates', float('nan'))
                    lp_stage4_share = train_metrics.get('lp_stage_full_soft_share', float('nan'))
                    lp_stage5_share = train_metrics.get('lp_stage_round_refix_share', float('nan'))
                    lp_failed_share = train_metrics.get('lp_stage_failed_share', float('nan'))
                    lp_early_share = sum(
                        train_metrics.get(metric_name, 0.0)
                        for metric_name in (
                            'lp_stage_hard_fix_share',
                            'lp_stage_repair_20_share',
                            'lp_stage_repair_100_share',
                        )
                    )
                    lp_slack_mean = train_metrics.get('lp_slack_used_mean', float('nan'))
                    lp_deviation_mean = train_metrics.get('lp_decoder_deviation_mean', float('nan'))
                    lp_flips_mean = train_metrics.get('lp_rounded_flips_mean', float('nan'))

                    print(
                        f"Epoch {epoch+1}/{config.silver_epochs} ({dt:.1f}s) | "
                        f"Total={train_metrics.get('loss_total', 0.0):.4f} "
                        f"CD={train_metrics.get('cd/cd_loss', 0.0):.4f} "
                        f"Pref={_fmt_float(train_pref_loss, 4)} | "
                        f"Pairs={int(round(n_pairs_total))} "
                        f"Attempted={int(round(n_lp_scenarios_attempted))} "
                        f"WithPairs={int(round(n_lp_scenarios_with_pairs))} "
                        f"PairCoverage={_fmt_pct(pair_coverage)} "
                        f"NonFinite={_fmt_pct(pct_non_finite)} | "
                        f"Early={_fmt_pct(lp_early_share)} "
                        f"Stage4={_fmt_pct(lp_stage4_share)} "
                        f"Stage5={_fmt_pct(lp_stage5_share)} "
                        f"Failed={_fmt_pct(lp_failed_share)} "
                        f"Slack={_fmt_float(lp_slack_mean, 2)} "
                        f"Deviation={_fmt_float(lp_deviation_mean, 2)} "
                        f"Flips={_fmt_float(lp_flips_mean, 2)} | "
                        f"TrainPrefAcc={_fmt_pct(train_pref_acc)} "
                        f"ValPrefAcc={_fmt_pct(val_pref_acc)} "
                        f"Spearman(E,J)={_fmt_float(val_spearman, 3)} "
                        f"BestOfKGap={_fmt_pct(val_bestofk_gap, 1)} | "
                        f"ValGap_R={_fmt_float(val_gap_rand, 4)} "
                        f"ValGap_L={_fmt_float(val_gap_lang, 4)} | "
                        f"LR={scheduler.get_last_lr()[0]:.2e} | "
                        f"Lsteps={curr_steps} | "
                        f"LambdaPref={loss_fn.lambda_pref:.2f}"
                    )

                    coverage_floor = float(getattr(config, 'silver_pair_coverage_floor', 0.20))
                    if n_lp_scenarios_attempted > 0 and math.isfinite(pair_coverage) and pair_coverage < coverage_floor:
                        print(
                            f"  !! PairCoverage below floor: {_fmt_pct(pair_coverage)} < {_fmt_pct(coverage_floor)}. "
                            'Preference signal is sparse.'
                        )

                    metric_name = 'ValGap_L'
                    score = val_gap_lang if math.isfinite(val_gap_lang) else val_gap_rand
                    if early_stop_metric == 'val_pref_accuracy' and math.isfinite(val_pref_acc):
                        score = val_pref_acc
                        metric_name = 'ValPref'
                    elif early_stop_metric == 'val_spearman' and math.isfinite(val_spearman):
                        score = val_spearman
                        metric_name = 'ValRho'
                    elif early_stop_metric in {'val_pref_accuracy', 'val_spearman'}:
                        raise RuntimeError(
                            f"Early-stop metric {early_stop_metric!r} is unavailable for epoch {epoch+1}. "
                            'Ranking validation returned no finite value; inspect LP candidate generation and '
                            'silver_val_* settings.'
                        )

                    if score > best_val_metric + min_delta:
                        best_val_metric = score
                        patience_counter = 0
                        _save_checkpoint(model, optimizer, epoch, val_metrics, str(run_dir / f'{prefix}_best.pt'))
                        print(f"  >> New best {metric_name}: {score:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= config.silver_patience:
                            print(f"  Early stopping at epoch {epoch+1} (best {metric_name}={best_val_metric:.4f})")
                            break

                    if (epoch + 1) % config.save_every_epoch == 0:
                        _save_checkpoint(model, optimizer, epoch, val_metrics, str(run_dir / f'{prefix}_epoch_{epoch+1}.pt'))

                best_path = run_dir / f'{prefix}_best.pt'
                if best_path.exists():
                    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model = model.to(config.device)

                _save_history(history, str(run_dir / f'{prefix}_history.json'))
                return model, history
            """
        ),
        code(
            """
            GOLD_BEST_PATH = RUN_DIR / 'gold_best.pt'

            if RUN_GOLD:
                model, gold_history = run_conditioned_gold_pretraining(
                    config,
                    RUN_DIR,
                    feature_stats=global_feature_stats,
                    feature_cache=global_feature_cache,
                )
            else:
                model, _gold_ckpt = load_conditioned_checkpoint(GOLD_BEST_PATH, config)
            """
        ),
        code(
            """
            silver_stage1_config = deepcopy(config)
            for key, value in SILVER_STAGE1_OVERRIDES.items():
                setattr(silver_stage1_config, key, value)
            silver_stage1_config.silver_output_prefix = SILVER_STAGE1_PREFIX
            GOLD_BEST_PATH = RUN_DIR / 'gold_best.pt'
            SILVER_STAGE1_BEST = RUN_DIR / f'{SILVER_STAGE1_PREFIX}_best.pt'

            if RUN_SILVER_STAGE1:
                # Allow reruns that start directly from silver fine-tuning.
                model, _gold_ckpt = load_conditioned_checkpoint(GOLD_BEST_PATH, config)
                model, silver_stage1_history = run_conditioned_silver_finetuning(
                    model,
                    silver_stage1_config,
                    RUN_DIR,
                    prefix=SILVER_STAGE1_PREFIX,
                    feature_stats=global_feature_stats,
                    feature_cache=global_feature_cache,
                )
            else:
                model, _silver_stage1_ckpt = load_conditioned_checkpoint(SILVER_STAGE1_BEST, config)
            """
        ),
        code(
            """
            silver_v2_config = deepcopy(config)
            for key, value in SILVER_V2_OVERRIDES.items():
                setattr(silver_v2_config, key, value)
            silver_v2_config.silver_output_prefix = SILVER_V2_PREFIX
            SILVER_V2_BEST = RUN_DIR / f'{SILVER_V2_PREFIX}_best.pt'

            if RUN_SILVER_V2:
                if not RUN_SILVER_STAGE1:
                    GOLD_BEST_PATH = RUN_DIR / 'gold_best.pt'
                    model, _gold_ckpt = load_conditioned_checkpoint(GOLD_BEST_PATH, config)
                model, silver_v2_history = run_conditioned_silver_finetuning(
                    model,
                    silver_v2_config,
                    RUN_DIR,
                    prefix=SILVER_V2_PREFIX,
                    feature_stats=global_feature_stats,
                    feature_cache=global_feature_cache,
                )
            else:
                model, _silver_v2_ckpt = load_conditioned_checkpoint(SILVER_V2_BEST, silver_v2_config)
            """
        ),
        code(
            """
            final_ckpt_path = RUN_DIR / 'final_conditioned_ebm.pt'
            feature_stats_payload = {
                name: {
                    'mean': float(global_feature_stats['mean'][idx]),
                    'std': float(global_feature_stats['std'][idx]),
                }
                for idx, name in enumerate(global_feature_stats['feature_names'])
            }

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'model_kind': 'ScenarioConditionedEBM',
                    'conditioning_mode': CONDITIONING_MODE,
                    'scenario_feature_names': SCENARIO_FEATURE_NAMES,
                    'base_embed_dim': BASE_EMBED_DIM,
                    'scenario_feature_dim': len(SCENARIO_FEATURE_NAMES),
                    'scenario_feature_stats': feature_stats_payload,
                    'run_dir': str(RUN_DIR),
                    'config': {
                        'n_features': config.n_features,
                        'hidden_dim': config.hidden_dim,
                        'gru_layers': config.gru_layers,
                        'bidirectional': config.bidirectional,
                        'dropout': config.dropout,
                        'use_peak_term': config.use_peak_term,
                        'peak_tau': config.peak_tau,
                        'peak_weight': config.peak_weight,
                        'energy_max': config.energy_max,
                        'gold_epochs': config.gold_epochs,
                        'gold_lr': config.gold_lr,
                        'silver_stage1_prefix': SILVER_STAGE1_PREFIX,
                        'silver_stage1_overrides': SILVER_STAGE1_OVERRIDES,
                        'silver_v2_prefix': SILVER_V2_PREFIX,
                        'silver_v2_overrides': SILVER_V2_OVERRIDES,
                    },
                },
                final_ckpt_path,
            )

            print(f"Final conditioned checkpoint saved to: {final_ckpt_path}")
            print('\\nRun directory contents:')
            for path in sorted(RUN_DIR.iterdir()):
                size_mb = path.stat().st_size / 1024**2
                print(f"  {path.name:40s} {size_mb:8.2f} MB")

            print('\\nNotes:')
            print('  - This notebook retrains only the EBM; HTE embeddings remain frozen.')
            print('  - Scenario scalars are normalized once on the gold+silver union and reused for all stages.')
            print('  - The current compare_silver_checkpoints_smoketest.py script assumes the vanilla EBM.')
            print('    Use the saved scenario_feature_stats above when you later build the conditioned comparison runner.')
            """
        ),
        md(
            """
            ## Training And Evaluation Metrics

            These cells reload the saved history files and build compact tables plus figures for:
            - train curves
            - validation curves
            - silver ranking diagnostics
            - LP pair coverage / non-finite behavior

            There is no separate held-out test set in this notebook yet, so the evaluation section below focuses on the saved validation metrics and run artifacts.
            """
        ),
        code(
            """
            plt.style.use('seaborn-v0_8-whitegrid')

            HISTORY_PATHS = {
                'gold': RUN_DIR / 'gold_history.json',
                'silver_stage1': RUN_DIR / f'{SILVER_STAGE1_PREFIX}_history.json',
                'silver_v2': RUN_DIR / f'{SILVER_V2_PREFIX}_history.json',
            }

            ARTIFACT_PATHS = {
                'gold_best_ckpt': RUN_DIR / 'gold_best.pt',
                'silver_stage1_best_ckpt': RUN_DIR / f'{SILVER_STAGE1_PREFIX}_best.pt',
                'silver_v2_best_ckpt': RUN_DIR / f'{SILVER_V2_PREFIX}_best.pt',
                'final_conditioned_ckpt': RUN_DIR / 'final_conditioned_ebm.pt',
                'feature_cache': FEATURE_CACHE_PATH,
                'feature_stats': FEATURE_STATS_PATH,
                **{f'{stage}_history': path for stage, path in HISTORY_PATHS.items()},
            }


            def load_json_if_exists(path: Path):
                path = Path(path)
                if not path.exists():
                    return None
                with path.open('r', encoding='utf-8') as f:
                    return json.load(f)


            def history_to_frame(history: Dict[str, Any], split: str, stage: str) -> pd.DataFrame:
                rows = []
                for epoch_idx, metrics in enumerate(history.get(split, []), start=1):
                    row = {'stage': stage, 'split': split, 'epoch': epoch_idx}
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                row[key] = float(value)
                    rows.append(row)
                return pd.DataFrame(rows)


            def first_existing_column(df: pd.DataFrame, candidates: List[str]):
                for candidate in candidates:
                    if candidate in df.columns:
                        return candidate
                return None


            def get_series(df: pd.DataFrame, candidates: List[str], scale: float = 1.0) -> pd.Series:
                col = first_existing_column(df, candidates)
                if col is None:
                    return pd.Series(dtype=float)
                return pd.to_numeric(df[col], errors='coerce') * scale


            def summarize_stage(stage: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
                row = {
                    'stage': stage,
                    'train_epochs': int(len(train_df)),
                    'val_epochs': int(len(val_df)),
                }

                best_col = None
                best_mode = 'max'
                if stage == 'gold':
                    best_col = first_existing_column(val_df, ['E_gap_lang', 'E_gap_rand'])
                else:
                    best_col = first_existing_column(
                        val_df,
                        ['val_pref_accuracy', 'val_spearman_energy_J', 'val_spearman', 'E_gap_lang', 'E_gap_rand'],
                    )
                if best_col and not val_df.empty:
                    metric_series = pd.to_numeric(val_df[best_col], errors='coerce')
                    if metric_series.notna().any():
                        best_idx = metric_series.idxmax() if best_mode == 'max' else metric_series.idxmin()
                        row['best_epoch'] = int(val_df.loc[best_idx, 'epoch'])
                        row['best_metric_name'] = best_col
                        row['best_metric_value'] = float(metric_series.loc[best_idx])
                    else:
                        row['best_epoch'] = np.nan
                        row['best_metric_name'] = best_col
                        row['best_metric_value'] = np.nan
                else:
                    row['best_epoch'] = np.nan
                    row['best_metric_name'] = None
                    row['best_metric_value'] = np.nan

                metric_specs = {
                    'last_train_total': (train_df, ['loss_total']),
                    'last_train_cd': (train_df, ['cd/cd_loss', 'cd_loss']),
                    'last_train_pref': (train_df, ['loss_pref_weighted']),
                    'last_pair_coverage_pct': (train_df, ['pair_coverage']),
                    'last_non_finite_pct': (train_df, ['pct_non_finite_candidates']),
                    'best_val_gap_lang': (val_df, ['E_gap_lang']),
                    'best_val_gap_rand': (val_df, ['E_gap_rand']),
                    'best_val_pref_accuracy_pct': (val_df, ['val_pref_accuracy']),
                    'best_val_spearman': (val_df, ['val_spearman_energy_J', 'val_spearman']),
                    'best_val_best_of_k_gap_pct': (val_df, ['val_best_of_K_gap', 'val_bestofk_gap']),
                }

                for out_name, (df, candidates) in metric_specs.items():
                    col = first_existing_column(df, candidates)
                    if col is None or df.empty:
                        row[out_name] = np.nan
                        continue
                    series = pd.to_numeric(df[col], errors='coerce')
                    if out_name.startswith('best_') and 'best_of_k_gap' in out_name:
                        row[out_name] = float(series.min()) if series.notna().any() else np.nan
                    elif out_name.startswith('best_'):
                        row[out_name] = float(series.max()) if series.notna().any() else np.nan
                    else:
                        row[out_name] = float(series.iloc[-1]) if series.notna().any() else np.nan

                row['last_pair_coverage_pct'] *= 100.0 if np.isfinite(row['last_pair_coverage_pct']) else 1.0
                row['last_non_finite_pct'] *= 100.0 if np.isfinite(row['last_non_finite_pct']) else 1.0
                row['best_val_pref_accuracy_pct'] *= 100.0 if np.isfinite(row['best_val_pref_accuracy_pct']) else 1.0
                row['best_val_best_of_k_gap_pct'] *= 100.0 if np.isfinite(row['best_val_best_of_k_gap_pct']) else 1.0
                return row


            histories = {
                stage: load_json_if_exists(path)
                for stage, path in HISTORY_PATHS.items()
            }
            frames = []
            for stage, history in histories.items():
                if history is None:
                    continue
                frames.append(history_to_frame(history, 'train', stage))
                frames.append(history_to_frame(history, 'val', stage))

            history_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            train_history_df = history_df[history_df['split'] == 'train'].copy() if not history_df.empty else pd.DataFrame()
            val_history_df = history_df[history_df['split'] == 'val'].copy() if not history_df.empty else pd.DataFrame()

            artifact_rows = []
            for name, path in ARTIFACT_PATHS.items():
                path = Path(path)
                artifact_rows.append({
                    'artifact': name,
                    'exists': path.exists(),
                    'path': str(path),
                    'size_mb': round(path.stat().st_size / 1024**2, 3) if path.exists() else np.nan,
                })
            artifacts_df = pd.DataFrame(artifact_rows).sort_values(['exists', 'artifact'], ascending=[False, True])

            stage_summaries = []
            for stage in HISTORY_PATHS:
                train_df = train_history_df[train_history_df['stage'] == stage].copy()
                val_df = val_history_df[val_history_df['stage'] == stage].copy()
                if train_df.empty and val_df.empty:
                    continue
                stage_summaries.append(summarize_stage(stage, train_df, val_df))
            stage_summary_df = pd.DataFrame(stage_summaries)

            print('Artifacts')
            display(artifacts_df)

            if not stage_summary_df.empty:
                print('Stage summary')
                display(stage_summary_df.round(4))
            else:
                print('No history files found yet. Run training cells first, then rerun this section.')
            """
        ),
        code(
            """
            if history_df.empty:
                print('No training history available yet.')
            else:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                axes = axes.ravel()

                # 1. Gold: train CD and validation gaps
                gold_train = train_history_df[train_history_df['stage'] == 'gold']
                gold_val = val_history_df[val_history_df['stage'] == 'gold']
                if not gold_train.empty:
                    cd_series = get_series(gold_train, ['cd_loss', 'cd/cd_loss'])
                    if not cd_series.empty:
                        axes[0].plot(gold_train['epoch'], cd_series, marker='o', label='Train CD')
                if not gold_val.empty:
                    gap_rand = get_series(gold_val, ['E_gap_rand'])
                    gap_lang = get_series(gold_val, ['E_gap_lang'])
                    if not gap_rand.empty:
                        axes[0].plot(gold_val['epoch'], gap_rand, marker='s', label='Val Gap Rand')
                    if not gap_lang.empty:
                        axes[0].plot(gold_val['epoch'], gap_lang, marker='^', label='Val Gap Langevin')
                axes[0].set_title('Gold Train / Val')
                axes[0].set_xlabel('Epoch')
                axes[0].legend()

                # 2. Silver losses
                for stage, color in [('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                    df = train_history_df[train_history_df['stage'] == stage]
                    if df.empty:
                        continue
                    total = get_series(df, ['loss_total'])
                    cd = get_series(df, ['cd/cd_loss', 'cd_loss'])
                    pref = get_series(df, ['loss_pref_weighted'])
                    if not total.empty:
                        axes[1].plot(df['epoch'], total, marker='o', color=color, label=f'{stage} Total')
                    if not cd.empty:
                        axes[1].plot(df['epoch'], cd, linestyle='--', color=color, alpha=0.8, label=f'{stage} CD')
                    if not pref.empty:
                        axes[1].plot(df['epoch'], pref, linestyle=':', color=color, alpha=0.9, label=f'{stage} Pref')
                axes[1].set_title('Silver Train Losses')
                axes[1].set_xlabel('Epoch')
                axes[1].legend(fontsize=9)

                # 3. Pair coverage and non-finite candidates
                for stage, color in [('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                    df = train_history_df[train_history_df['stage'] == stage]
                    if df.empty:
                        continue
                    coverage = get_series(df, ['pair_coverage'], scale=100.0)
                    non_finite = get_series(df, ['pct_non_finite_candidates'], scale=100.0)
                    if not coverage.empty:
                        axes[2].plot(df['epoch'], coverage, marker='o', color=color, label=f'{stage} PairCoverage')
                    if not non_finite.empty:
                        axes[2].plot(df['epoch'], non_finite, linestyle='--', color=color, label=f'{stage} NonFinite')
                axes[2].axhline(20.0, color='red', linestyle=':', alpha=0.7, label='20% coverage floor')
                axes[2].set_title('LP Signal Quality')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Percent')
                axes[2].legend(fontsize=9)

                # 4. Preference accuracy
                for stage, color in [('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                    train_df = train_history_df[train_history_df['stage'] == stage]
                    val_df = val_history_df[val_history_df['stage'] == stage]
                    train_acc = get_series(train_df, ['train_pref_accuracy'], scale=100.0)
                    val_acc = get_series(val_df, ['val_pref_accuracy'], scale=100.0)
                    if not train_acc.empty:
                        axes[3].plot(train_df['epoch'], train_acc, marker='o', color=color, label=f'{stage} TrainPrefAcc')
                    if not val_acc.empty:
                        axes[3].plot(val_df['epoch'], val_acc, linestyle='--', marker='s', color=color, label=f'{stage} ValPrefAcc')
                axes[3].axhline(55.0, color='gray', linestyle=':', alpha=0.7, label='55% watch line')
                axes[3].axhline(65.0, color='green', linestyle=':', alpha=0.7, label='65% strong signal')
                axes[3].set_title('Preference Accuracy')
                axes[3].set_xlabel('Epoch')
                axes[3].set_ylabel('Percent')
                axes[3].legend(fontsize=9)

                # 5. Validation ranking metrics
                for stage, color in [('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                    df = val_history_df[val_history_df['stage'] == stage]
                    if df.empty:
                        continue
                    spearman = get_series(df, ['val_spearman_energy_J', 'val_spearman'])
                    bestofk = get_series(df, ['val_best_of_K_gap', 'val_bestofk_gap'], scale=100.0)
                    if not spearman.empty:
                        axes[4].plot(df['epoch'], spearman, marker='o', color=color, label=f'{stage} Spearman(E,J)')
                    if not bestofk.empty:
                        axes[4].plot(df['epoch'], bestofk, linestyle='--', color=color, label=f'{stage} BestOfKGap %')
                axes[4].axhline(0.30, color='green', linestyle=':', alpha=0.7, label='0.30 good sign')
                axes[4].set_title('Validation Ranking Metrics')
                axes[4].set_xlabel('Epoch')
                axes[4].legend(fontsize=9)

                # 6. Validation energy gaps
                for stage, color in [('gold', 'tab:gray'), ('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                    df = val_history_df[val_history_df['stage'] == stage]
                    if df.empty:
                        continue
                    gap_rand = get_series(df, ['E_gap_rand'])
                    gap_lang = get_series(df, ['E_gap_lang'])
                    if not gap_rand.empty:
                        axes[5].plot(df['epoch'], gap_rand, linestyle='--', color=color, alpha=0.8, label=f'{stage} Gap Rand')
                    if not gap_lang.empty:
                        axes[5].plot(df['epoch'], gap_lang, marker='o', color=color, label=f'{stage} Gap Lang')
                axes[5].set_title('Validation Energy Gaps')
                axes[5].set_xlabel('Epoch')
                axes[5].legend(fontsize=9)

                plt.tight_layout()
                plt.show()
            """
        ),
        code(
            """
            if history_df.empty:
                print('No training history available yet.')
            else:
                silver_history = train_history_df[train_history_df['stage'].isin(['silver_stage1', 'silver_v2'])].copy()
                if silver_history.empty:
                    print('No silver history available yet.')
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

                    for stage, color in [('silver_stage1', 'tab:blue'), ('silver_v2', 'tab:orange')]:
                        df = silver_history[silver_history['stage'] == stage]
                        if df.empty:
                            continue

                        early = (
                            get_series(df, ['lp_stage_hard_fix_share'], scale=100.0)
                            .add(get_series(df, ['lp_stage_repair_20_share'], scale=100.0), fill_value=0.0)
                            .add(get_series(df, ['lp_stage_repair_100_share'], scale=100.0), fill_value=0.0)
                        )
                        stage4 = get_series(df, ['lp_stage_full_soft_share'], scale=100.0)
                        stage5 = get_series(df, ['lp_stage_round_refix_share'], scale=100.0)
                        failed = get_series(df, ['lp_stage_failed_share'], scale=100.0)

                        if not early.empty:
                            axes[0].plot(df['epoch'], early, color=color, linestyle=':', label=f'{stage} Early')
                        if not stage4.empty:
                            axes[0].plot(df['epoch'], stage4, color=color, linestyle='-', label=f'{stage} Stage4')
                        if not stage5.empty:
                            axes[0].plot(df['epoch'], stage5, color=color, linestyle='--', label=f'{stage} Stage5')
                        if not failed.empty:
                            axes[0].plot(df['epoch'], failed, color=color, linestyle='-.', label=f'{stage} Failed')

                        slack = get_series(df, ['lp_slack_used_mean'])
                        deviation = get_series(df, ['lp_decoder_deviation_mean'])
                        flips = get_series(df, ['lp_rounded_flips_mean'])
                        if not slack.empty:
                            axes[1].plot(df['epoch'], slack, color=color, linestyle='-', label=f'{stage} Slack')
                        if not deviation.empty:
                            axes[1].plot(df['epoch'], deviation, color=color, linestyle='--', label=f'{stage} Deviation')
                        if not flips.empty:
                            axes[1].plot(df['epoch'], flips, color=color, linestyle=':', label=f'{stage} Flips')

                        pairs = get_series(df, ['n_pairs_total'])
                        attempted = get_series(df, ['n_lp_scenarios_attempted'])
                        with_pairs = get_series(df, ['n_lp_scenarios_with_pairs'])
                        if not pairs.empty:
                            axes[2].plot(df['epoch'], pairs, color=color, linestyle='-', label=f'{stage} nPairs')
                        if not attempted.empty:
                            axes[2].plot(df['epoch'], attempted, color=color, linestyle='--', label=f'{stage} Attempted')
                        if not with_pairs.empty:
                            axes[2].plot(df['epoch'], with_pairs, color=color, linestyle=':', label=f'{stage} WithPairs')

                    axes[0].set_title('LP Stage Distribution')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Percent')
                    axes[0].legend(fontsize=9)

                    axes[1].set_title('Repair Diagnostics')
                    axes[1].set_xlabel('Epoch')
                    axes[1].legend(fontsize=9)

                    axes[2].set_title('LP Pair Counts')
                    axes[2].set_xlabel('Epoch')
                    axes[2].legend(fontsize=9)

                    plt.tight_layout()
                    plt.show()

                    latest_silver_table = (
                        silver_history
                        .sort_values(['stage', 'epoch'])
                        .groupby('stage', as_index=False)
                        .tail(1)
                        [[
                            'stage',
                            'epoch',
                            *[c for c in [
                                'loss_total',
                                'cd/cd_loss',
                                'loss_pref_weighted',
                                'pair_coverage',
                                'pct_non_finite_candidates',
                                'train_pref_accuracy',
                                'n_pairs_total',
                                'n_lp_scenarios_attempted',
                                'n_lp_scenarios_with_pairs',
                            ] if c in silver_history.columns]
                        ]]
                        .copy()
                    )
                    for pct_col in ['pair_coverage', 'pct_non_finite_candidates', 'train_pref_accuracy']:
                        if pct_col in latest_silver_table.columns:
                            latest_silver_table[pct_col] = latest_silver_table[pct_col] * 100.0

                    print('Latest silver diagnostics')
                    display(latest_silver_table.round(4))
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
