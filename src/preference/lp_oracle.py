# ==============================================================================
# LP ORACLE FOR PREFERENCE-BASED LEARNING
# ==============================================================================
# Wrapper around LPWorkerTwoStage for economic evaluation of EBM candidates.
# Provides the cost signal C^(k) for preference learning.
# 
# Features model caching for fast repeated evaluations on same scenarios.
# ==============================================================================

from __future__ import annotations

import os
import copy
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict


class OracleStage(Enum):
    """Stage at which LP solved successfully."""
    HARD_FIX = "hard_fix"
    REPAIR = "repair"
    FULL_SOFT = "full_soft"
    FAILED = "failed"
    DUMMY = "dummy"  # For testing without LP solver


@dataclass
class OracleResult:
    """Result from LP oracle evaluation."""
    scenario_id: str
    feasible: bool
    objective_value: float  # System cost (EUR)
    slack_used: float  # Unserved energy (MWh)
    solve_time: float
    stage_used: OracleStage
    n_flips: int = 0
    message: str = ""
    
    @property
    def cost(self) -> float:
        """Alias for objective_value."""
        return self.objective_value


@dataclass
class LPOracleConfig:
    """Configuration for LP Oracle."""
    scenarios_dir: str
    solver_name: str = "appsi_highs"
    slack_tolerance: float = 1.0  # MWh
    deviation_penalty: float = 10000.0
    time_limit_hard_fix: float = 20.0
    time_limit_repair: float = 2.0
    time_limit_full_soft: float = 5.0
    dt_hours: float = 1.0
    verbose: bool = False
    use_parallel: bool = False
    max_workers: int = 4
    
    # Fallback cost for failed solves
    infeasible_cost: float = 1e9
    
    # Model caching
    use_cache: bool = True
    max_cache_size: int = 50  # Max models to keep in memory


class LPOracle:
    """
    LP Oracle for economic evaluation of EBM decision candidates.
    
    This is the economic oracle W in the preference learning methodology:
        C^(k) = W(u^(k), x)
    
    Uses the LPWorkerTwoStage with a simplified interface for EBM training.
    
    Features:
    1. Converts EBM decision format [Z, T, 8] to LP worker format
    2. Handles batch evaluation of multiple candidates
    3. Returns cost and feasibility for preference ranking
    4. Optional parallel evaluation for multiple scenarios
    """
    
    def __init__(
        self,
        config: LPOracleConfig,
        lp_worker: Optional[Any] = None,
    ):
        """
        Args:
            config: Oracle configuration
            lp_worker: Optional pre-initialized LPWorkerTwoStage instance
        """
        self.config = config
        self._lp_worker = lp_worker
        self._initialized = False
        
        # Lazy initialization of LP worker
        if lp_worker is not None:
            self._initialized = True
    
    @property
    def lp_worker(self):
        """Lazy initialization of LP worker."""
        if not self._initialized:
            self._initialize_worker()
        return self._lp_worker
    
    def _initialize_worker(self):
        """Initialize the LP worker on first use."""
        try:
            from src.milp.lp_worker_two_stage import LPWorkerTwoStage
            
            self._lp_worker = LPWorkerTwoStage(
                scenarios_dir=self.config.scenarios_dir,
                solver_name=self.config.solver_name,
                slack_tol_mwh=self.config.slack_tolerance,
                deviation_penalty=self.config.deviation_penalty,
                time_limit_hard_fix=self.config.time_limit_hard_fix,
                time_limit_repair_20=self.config.time_limit_repair,
                time_limit_repair_100=self.config.time_limit_repair,
                time_limit_full_soft=self.config.time_limit_full_soft,
                dt_hours=self.config.dt_hours,
                verbose=self.config.verbose,
            )
            self._initialized = True
            print("✓ LPOracle initialized with LPWorkerTwoStage")
            
        except ImportError as e:
            print(f"⚠️ Could not import LPWorkerTwoStage: {e}")
            print("   LP Oracle will use dummy costs")
            self._lp_worker = None
            self._initialized = True
        except Exception as e:
            print(f"⚠️ Could not initialize LP worker: {e}")
            self._lp_worker = None
            self._initialized = True
    
    def convert_ebm_to_lp_format(
        self,
        u_zt: torch.Tensor,
        zone_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Convert EBM decision format to LP worker format.
        
        EBM format [Z, T, 8]:
            0: battery_charging
            1: battery_discharging
            2: pumped_charging
            3: pumped_discharging
            4: dr_active
            5: nuclear_on
            6: thermal_on
            7: import_mode
        
        LP worker expects [Z, T, F] with slightly different ordering.
        
        Args:
            u_zt: Decision tensor [Z, T, 8]
            zone_names: Optional zone names for validation
        
        Returns:
            Tensor compatible with LP worker
        """
        # The LP worker expects the same format, just ensure binarization
        u_binary = (u_zt > 0.5).float()
        return u_binary
    
    def evaluate(
        self,
        scenario_id: str,
        u_zt: torch.Tensor,
    ) -> OracleResult:
        """
        Evaluate a single decision for a scenario.
        
        Args:
            scenario_id: Scenario identifier
            u_zt: Decision tensor [Z, T, 8]
        
        Returns:
            OracleResult with cost and feasibility
        """
        start_time = time.time()
        
        # Ensure binary
        u_binary = self.convert_ebm_to_lp_format(u_zt)
        
        # Use LP worker if available
        if self.lp_worker is not None:
            try:
                result = self.lp_worker.solve(scenario_id, u_binary)
                
                # Map stage
                stage_map = {
                    "hard_fix": OracleStage.HARD_FIX,
                    "repair_20": OracleStage.REPAIR,
                    "repair_100": OracleStage.REPAIR,
                    "full_soft": OracleStage.FULL_SOFT,
                    "failed": OracleStage.FAILED,
                }
                stage = stage_map.get(result.stage_used.value, OracleStage.FAILED)
                
                return OracleResult(
                    scenario_id=scenario_id,
                    feasible=result.status in ("optimal", "feasible"),
                    objective_value=result.objective_value,
                    slack_used=result.slack_used,
                    solve_time=time.time() - start_time,
                    stage_used=stage,
                    n_flips=result.n_flips,
                    message=result.message,
                )
                
            except Exception as e:
                return OracleResult(
                    scenario_id=scenario_id,
                    feasible=False,
                    objective_value=self.config.infeasible_cost,
                    slack_used=float("inf"),
                    solve_time=time.time() - start_time,
                    stage_used=OracleStage.FAILED,
                    message=f"Error: {str(e)}",
                )
        
        # Fallback: dummy cost based on decision heuristics
        return self._dummy_evaluate(scenario_id, u_binary, start_time)
    
    def _dummy_evaluate(
        self,
        scenario_id: str,
        u_zt: torch.Tensor,
        start_time: float,
    ) -> OracleResult:
        """
        Generate dummy cost for testing without LP solver.
        
        Uses heuristics:
        - Higher thermal usage = higher cost
        - More battery cycling = moderate cost
        - DR usage = some savings
        """
        Z, T, F = u_zt.shape
        
        # Thermal cost (high marginal cost)
        thermal_on = u_zt[..., 6].sum().item()
        thermal_cost = thermal_on * 50.0  # €50/MWh equivalent
        
        # Nuclear cost (low marginal cost, high capacity)
        nuclear_on = u_zt[..., 5].sum().item()
        nuclear_cost = nuclear_on * 10.0  # €10/MWh equivalent
        
        # Storage cycling cost (degradation)
        battery_cycles = (u_zt[..., 0] + u_zt[..., 1]).sum().item()
        pumped_cycles = (u_zt[..., 2] + u_zt[..., 3]).sum().item()
        storage_cost = (battery_cycles + pumped_cycles) * 5.0
        
        # DR cost (activation cost)
        dr_active = u_zt[..., 4].sum().item()
        dr_cost = dr_active * 20.0
        
        # Base system cost
        base_cost = 1e5 + Z * T * 10  # Scale with problem size
        
        # Total cost
        total_cost = base_cost + thermal_cost + nuclear_cost + storage_cost + dr_cost
        
        # Add some noise for diversity
        noise = np.random.normal(0, total_cost * 0.01)
        total_cost += noise
        
        return OracleResult(
            scenario_id=scenario_id,
            feasible=True,
            objective_value=total_cost,
            slack_used=0.0,
            solve_time=time.time() - start_time,
            stage_used=OracleStage.DUMMY,
            message="Dummy evaluation (LP worker not available)",
        )
    
    def evaluate_batch(
        self,
        scenario_ids: List[str],
        u_batch: torch.Tensor,
    ) -> List[OracleResult]:
        """
        Evaluate a batch of decisions.
        
        Args:
            scenario_ids: List of B scenario IDs
            u_batch: Decision tensor [B, Z, T, 8]
        
        Returns:
            List of B OracleResults
        """
        B = u_batch.shape[0]
        
        if self.config.use_parallel and self.lp_worker is not None:
            return self._evaluate_batch_parallel(scenario_ids, u_batch)
        
        results = []
        for b in range(B):
            result = self.evaluate(scenario_ids[b], u_batch[b])
            results.append(result)
        
        return results
    
    def _evaluate_batch_parallel(
        self,
        scenario_ids: List[str],
        u_batch: torch.Tensor,
    ) -> List[OracleResult]:
        """Parallel batch evaluation using ThreadPoolExecutor."""
        B = u_batch.shape[0]
        results = [None] * B
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.evaluate, scenario_ids[b], u_batch[b]): b
                for b in range(B)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = OracleResult(
                        scenario_id=scenario_ids[idx],
                        feasible=False,
                        objective_value=self.config.infeasible_cost,
                        slack_used=float("inf"),
                        solve_time=0.0,
                        stage_used=OracleStage.FAILED,
                        message=f"Error: {str(e)}",
                    )
        
        return results
    
    def evaluate_candidates(
        self,
        scenario_id: str,
        candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[OracleResult]]:
        """
        Evaluate K candidates for a single scenario.
        
        Args:
            scenario_id: Scenario identifier
            candidates: [K, Z, T, 8] K candidate decisions
        
        Returns:
            costs: [K] tensor of costs
            results: List of K OracleResults
        """
        K = candidates.shape[0]
        results = []
        costs = torch.zeros(K)
        
        for k in range(K):
            result = self.evaluate(scenario_id, candidates[k])
            results.append(result)
            costs[k] = result.objective_value
        
        return costs, results
    
    def get_stage_statistics(
        self,
        results: List[OracleResult],
    ) -> Dict[str, int]:
        """Get counts of solve stages for analysis."""
        stats = {stage.value: 0 for stage in OracleStage}
        for r in results:
            stats[r.stage_used.value] += 1
        return stats


class PreferenceLPOracle(LPOracle):
    """
    Extended LP Oracle specifically for preference learning.
    
    Adds:
    1. Cost normalization for stable training
    2. Preference pair construction
    3. Hard negative mining
    """
    
    def __init__(
        self,
        config: LPOracleConfig,
        cost_scale: float = 1e-6,  # Scale costs to ~O(1)
        use_cost_normalization: bool = True,
    ):
        super().__init__(config)
        self.cost_scale = cost_scale
        self.use_cost_normalization = use_cost_normalization
        
        # Running statistics for normalization
        self._cost_mean = 0.0
        self._cost_std = 1.0
        self._n_samples = 0
    
    def normalize_cost(self, cost: float) -> float:
        """Normalize cost to ~O(1) range for stable training."""
        if self.use_cost_normalization:
            return (cost - self._cost_mean) / max(self._cost_std, 1e-6)
        return cost * self.cost_scale
    
    def update_statistics(self, costs: torch.Tensor):
        """Update running mean/std for cost normalization."""
        batch_mean = costs.mean().item()
        batch_std = costs.std().item()
        
        # Exponential moving average
        alpha = 0.1
        self._cost_mean = (1 - alpha) * self._cost_mean + alpha * batch_mean
        self._cost_std = (1 - alpha) * self._cost_std + alpha * batch_std
        self._n_samples += len(costs)
    
    def evaluate_and_rank(
        self,
        scenario_id: str,
        candidates: torch.Tensor,
        milp_cost: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[OracleResult]]:
        """
        Evaluate candidates and compute preference-based costs.
        
        Args:
            scenario_id: Scenario identifier
            candidates: [K, Z, T, 8] K candidate decisions
            milp_cost: Optional MILP reference cost for comparison
        
        Returns:
            costs: [K] raw costs
            normalized_costs: [K] normalized costs for training
            results: List of K OracleResults
        """
        costs, results = self.evaluate_candidates(scenario_id, candidates)
        
        # Update statistics
        self.update_statistics(costs)
        
        # Normalize
        normalized = torch.tensor([
            self.normalize_cost(c.item()) for c in costs
        ])
        
        return costs, normalized, results
    
    def get_hard_negatives(
        self,
        results: List[OracleResult],
        milp_cost: float,
        top_k: int = 3,
    ) -> List[int]:
        """
        Select hard negatives: candidates with high cost but still feasible.
        
        Args:
            results: List of OracleResults
            milp_cost: Reference MILP cost
            top_k: Number of hard negatives to select
        
        Returns:
            Indices of hard negative candidates
        """
        # Sort by cost gap from MILP (descending)
        gaps = []
        for i, r in enumerate(results):
            if r.feasible:
                gap = r.objective_value - milp_cost
                gaps.append((i, gap))
        
        # Sort by gap descending (larger gap = harder negative)
        gaps.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in gaps[:top_k]]


class CachedLPOracle(LPOracle):
    """
    LP Oracle with Pyomo model caching for fast repeated evaluations.
    
    Caches:
    1. Scenario data (JSON loaded once)
    2. Base Pyomo model (built once per scenario)
    
    Can also load pre-built models from disk (created by prebuild_lp_models.py).
    
    Only the binary variable fixings are modified for each evaluation,
    avoiding the expensive model building step (~30-60s → ~0.1s).
    """
    
    def __init__(
        self,
        config: LPOracleConfig,
        prebuilt_dir: Optional[str] = None,
    ):
        super().__init__(config)
        
        # LRU cache for scenario data
        self._scenario_cache: OrderedDict[str, Any] = OrderedDict()
        # LRU cache for base models (before binary fixing)
        self._model_cache: OrderedDict[str, Any] = OrderedDict()
        self._zone_names_cache: Dict[str, List[str]] = {}
        
        self.max_cache_size = config.max_cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Pre-built models directory
        self.prebuilt_dir = Path(prebuilt_dir) if prebuilt_dir else None
        self._prebuilt_available = set()
        
        if self.prebuilt_dir and self.prebuilt_dir.exists():
            self._scan_prebuilt_models()
    
    def _scan_prebuilt_models(self):
        """Scan prebuilt directory for available models."""
        if not self.prebuilt_dir:
            return
        
        pkl_files = list(self.prebuilt_dir.glob("scenario_*.pkl"))
        self._prebuilt_available = {f.stem for f in pkl_files}
        
        if self._prebuilt_available:
            print(f"✓ Found {len(self._prebuilt_available)} pre-built LP models")
    
    def _load_prebuilt_model(self, scenario_id: str) -> Optional[Tuple[Any, List[str], int]]:
        """Load a pre-built model from disk."""
        if not self.prebuilt_dir or scenario_id not in self._prebuilt_available:
            return None
        
        model_path = self.prebuilt_dir / f"{scenario_id}.pkl"
        
        try:
            try:
                import cloudpickle
                with open(model_path, 'rb') as f:
                    prebuilt = cloudpickle.load(f)
            except ImportError:
                import pickle
                with open(model_path, 'rb') as f:
                    prebuilt = pickle.load(f)
            
            # Deserialize model
            try:
                import cloudpickle
                model = cloudpickle.loads(prebuilt.model_bytes)
            except ImportError:
                import pickle
                model = pickle.loads(prebuilt.model_bytes)
            
            return model, prebuilt.zone_names, prebuilt.n_timesteps
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading prebuilt {scenario_id}: {e}")
            return None
    
    def _evict_if_needed(self, cache: OrderedDict):
        """Evict oldest entries if cache is full."""
        while len(cache) > self.max_cache_size:
            cache.popitem(last=False)
    
    def _get_scenario_data(self, scenario_id: str) -> Optional[Any]:
        """Get scenario data with caching."""
        if scenario_id in self._scenario_cache:
            # Move to end (most recently used)
            self._scenario_cache.move_to_end(scenario_id)
            self._cache_hits += 1
            return self._scenario_cache[scenario_id]
        
        self._cache_misses += 1
        
        # Load scenario
        try:
            from src.milp.scenario_loader import load_scenario_data
            
            scenario_path = self._find_scenario_path(scenario_id)
            if scenario_path is None:
                return None
            
            scenario_data = load_scenario_data(Path(scenario_path))
            
            # Cache
            self._scenario_cache[scenario_id] = scenario_data
            self._evict_if_needed(self._scenario_cache)
            
            return scenario_data
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error loading scenario {scenario_id}: {e}")
            return None
    
    def _find_scenario_path(self, scenario_id: str) -> Optional[str]:
        """Find the scenario JSON file path."""
        for path in [
            os.path.join(self.config.scenarios_dir, f"{scenario_id}.json"),
            os.path.join(self.config.scenarios_dir, 'dispatch_batch', f"{scenario_id}.json"),
        ]:
            if os.path.exists(path):
                return path
        return None
    
    def _get_base_model(self, scenario_id: str) -> Optional[Tuple[Any, List[str], int]]:
        """
        Get base Pyomo model with caching.
        
        Priority:
        1. Memory cache (fastest)
        2. Pre-built models from disk (fast)
        3. Build from scratch (slow)
        
        Returns:
            (model, zone_names, n_timesteps) or None if failed
        """
        # 1. Check memory cache first
        if scenario_id in self._model_cache:
            self._model_cache.move_to_end(scenario_id)
            self._cache_hits += 1
            model_data = self._model_cache[scenario_id]
            zone_names = self._zone_names_cache[scenario_id]
            # Deep copy the model for modification
            return copy.deepcopy(model_data['model']), zone_names, model_data['n_timesteps']
        
        self._cache_misses += 1
        
        # 2. Try loading from pre-built models
        prebuilt = self._load_prebuilt_model(scenario_id)
        if prebuilt is not None:
            model, zone_names, n_timesteps = prebuilt
            
            # Cache in memory for next time
            self._model_cache[scenario_id] = {
                'model': model,
                'n_timesteps': n_timesteps,
            }
            self._zone_names_cache[scenario_id] = zone_names
            self._evict_if_needed(self._model_cache)
            
            return copy.deepcopy(model), zone_names, n_timesteps
        
        # 3. Build from scratch
        scenario_data = self._get_scenario_data(scenario_id)
        if scenario_data is None:
            return None
        
        try:
            from src.milp.model import build_uc_model
            
            model = build_uc_model(scenario_data, enable_duals=False)
            zone_names = list(model.Z)
            n_timesteps = len(list(model.T))
            
            # Cache
            self._model_cache[scenario_id] = {
                'model': model,
                'n_timesteps': n_timesteps,
            }
            self._zone_names_cache[scenario_id] = zone_names
            self._evict_if_needed(self._model_cache)
            
            # Return a copy for modification
            return copy.deepcopy(model), zone_names, n_timesteps
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error building model for {scenario_id}: {e}")
            return None
    
    def evaluate(
        self,
        scenario_id: str,
        u_zt: torch.Tensor,
    ) -> OracleResult:
        """
        Evaluate decision with cached model (fast path).
        """
        start_time = time.time()
        
        # Get cached model
        result = self._get_base_model(scenario_id)
        if result is None:
            return self._dummy_evaluate(scenario_id, u_zt, start_time)
        
        model, zone_names, n_timesteps = result
        
        try:
            from pyomo.environ import value, TransformationFactory
            from pyomo.opt import TerminationCondition, SolverFactory
            
            # Convert decisions
            u_binary = self.convert_ebm_to_lp_format(u_zt)
            
            # Get decoder targets
            targets = self._get_decoder_targets(u_binary, zone_names, n_timesteps)
            
            # Fix binaries
            self._fix_all_binaries(model, targets)
            
            # Relax integrality
            TransformationFactory('core.relax_integer_vars').apply_to(model)
            
            # Solve with short time limit
            solver = SolverFactory(self.config.solver_name)
            solver.options['time_limit'] = self.config.time_limit_hard_fix
            
            results = solver.solve(model, tee=False, load_solutions=False)
            
            tc = results.solver.termination_condition
            feasible = tc in (TerminationCondition.optimal, TerminationCondition.feasible)
            
            if feasible and len(results.solution) > 0:
                model.solutions.load_from(results)
                obj_value = value(model.obj)
            else:
                obj_value = self.config.infeasible_cost
            
            return OracleResult(
                scenario_id=scenario_id,
                feasible=feasible,
                objective_value=obj_value,
                slack_used=0.0,
                solve_time=time.time() - start_time,
                stage_used=OracleStage.HARD_FIX if feasible else OracleStage.FAILED,
                message=f"Cache hits: {self._cache_hits}, misses: {self._cache_misses}",
            )
            
        except Exception as e:
            return OracleResult(
                scenario_id=scenario_id,
                feasible=False,
                objective_value=self.config.infeasible_cost,
                slack_used=float("inf"),
                solve_time=time.time() - start_time,
                stage_used=OracleStage.FAILED,
                message=f"Error: {str(e)}",
            )
    
    def _get_decoder_targets(
        self,
        decoder_output: torch.Tensor,
        zone_names: List[str],
        n_timesteps: int = 24,
    ) -> Dict[str, Dict[Tuple[str, int], float]]:
        """Extract target binary values from decoder output tensor [Z, T, F]."""
        Z = len(zone_names)
        decoder_np = decoder_output.cpu().numpy()
        
        if decoder_np.shape[0] > Z:
            decoder_np = decoder_np[:Z]
        elif decoder_np.shape[0] < Z:
            pad_size = Z - decoder_np.shape[0]
            decoder_np = np.pad(decoder_np, ((0, pad_size), (0, 0), (0, 0)))
        
        targets = {
            'u_thermal': {},
            'b_charge_mode': {},
            'pumped_charge_mode': {},
            'dr_active': {},
        }
        
        for z_idx, zone in enumerate(zone_names):
            for t in range(min(n_timesteps, decoder_np.shape[1])):
                targets['u_thermal'][(zone, t)] = float(decoder_np[z_idx, t, 6] > 0.5)
                
                batt_ch = decoder_np[z_idx, t, 0]
                batt_dis = decoder_np[z_idx, t, 1]
                targets['b_charge_mode'][(zone, t)] = 1.0 if batt_ch > batt_dis else 0.0
                
                pump_ch = decoder_np[z_idx, t, 2]
                pump_dis = decoder_np[z_idx, t, 3]
                targets['pumped_charge_mode'][(zone, t)] = 1.0 if pump_ch > pump_dis else 0.0
                
                targets['dr_active'][(zone, t)] = float(decoder_np[z_idx, t, 4] > 0.5)
        
        return targets
    
    def _fix_all_binaries(self, model, targets: Dict):
        """Fix binary variables to decoder targets."""
        from pyomo.environ import value
        
        eps = 1e-6
        
        for (z, t), val in targets['u_thermal'].items():
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                has_cap = hasattr(model, 'thermal_capacity') and value(model.thermal_capacity[z]) > eps
                if has_cap or val == 0.0:
                    if not model.u_thermal[z, t].is_fixed():
                        model.u_thermal[z, t].fix(val)
        
        for (z, t), val in targets['b_charge_mode'].items():
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                has_cap = hasattr(model, 'battery_power') and value(model.battery_power[z]) > eps
                if has_cap or val == 0.0:
                    if not model.b_charge_mode[z, t].is_fixed():
                        model.b_charge_mode[z, t].fix(val)
        
        for (z, t), val in targets['pumped_charge_mode'].items():
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                has_cap = hasattr(model, 'pumped_power') and value(model.pumped_power[z]) > eps
                if has_cap or val == 0.0:
                    if not model.pumped_charge_mode[z, t].is_fixed():
                        model.pumped_charge_mode[z, t].fix(val)
        
        for (z, t), val in targets['dr_active'].items():
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                has_cap = hasattr(model, 'dr_max') and value(model.dr_max[z]) > eps
                if has_cap or val == 0.0:
                    if not model.dr_active[z, t].is_fixed():
                        model.dr_active[z, t].fix(val)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'scenario_cache_size': len(self._scenario_cache),
            'model_cache_size': len(self._model_cache),
        }


def create_lp_oracle(
    scenarios_dir: str,
    solver_name: str = "appsi_highs",
    use_full_lp: bool = True,
    use_cache: bool = True,
    max_cache_size: int = 50,
    prebuilt_dir: Optional[str] = None,
    **kwargs,
) -> LPOracle:
    """
    Factory function to create LP Oracle.
    
    Args:
        scenarios_dir: Directory with scenario files
        solver_name: LP solver to use
        use_full_lp: If False, use dummy evaluation
        use_cache: If True, use CachedLPOracle for faster repeated evals
        max_cache_size: Max number of models to cache
        prebuilt_dir: Directory with pre-built Pyomo models (from prebuild_lp_models.py)
        **kwargs: Additional config options
    
    Returns:
        Configured LPOracle instance
    """
    config = LPOracleConfig(
        scenarios_dir=scenarios_dir,
        solver_name=solver_name,
        max_cache_size=max_cache_size,
        **kwargs,
    )
    
    if use_full_lp:
        if use_cache:
            print(f"✓ Creating CachedLPOracle (cache_size={max_cache_size})")
            if prebuilt_dir:
                print(f"  Pre-built models: {prebuilt_dir}")
            return CachedLPOracle(config, prebuilt_dir=prebuilt_dir)
        else:
            return PreferenceLPOracle(config)
    else:
        # Return oracle that will use dummy evaluation
        oracle = PreferenceLPOracle(config)
        oracle._initialized = True
        oracle._lp_worker = None
        return oracle
