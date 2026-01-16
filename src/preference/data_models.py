# ==============================================================================
# DATA MODELS FOR PREFERENCE-BASED LEARNING
# ==============================================================================

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class SolveStatus(Enum):
    """Status of LP/MILP solve."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ScenarioData:
    """
    Complete scenario data for preference learning.
    
    Attributes:
        scenario_id: Unique identifier
        n_zones: Number of zones
        n_timesteps: Number of timesteps
        dt_hours: Timestep duration in hours
        
        # Time series [Z, T]
        demand: Demand per zone per timestep (MW)
        solar_available: Solar availability (MW)
        wind_available: Wind availability (MW)
        hydro_ror: Run-of-river hydro (MW)
        hydro_inflow: Reservoir inflow (MWh)
        
        # Static capacities [Z]
        thermal_capacity: Thermal generation capacity (MW)
        thermal_min: Thermal minimum output (MW)
        nuclear_capacity: Nuclear capacity (MW)
        battery_power: Battery power capacity (MW)
        battery_capacity: Battery energy capacity (MWh)
        pumped_power: Pumped hydro power (MW)
        pumped_capacity: Pumped hydro energy (MWh)
        hydro_capacity: Reservoir hydro capacity (MW)
        dr_capacity: Demand response capacity (MW)
        
        # Zone metadata
        zone_names: List of zone names
        zone_to_region: Mapping from zone to region
    """
    scenario_id: str
    n_zones: int
    n_timesteps: int
    dt_hours: float = 1.0
    
    # Time series [Z, T]
    demand: Optional[torch.Tensor] = None
    solar_available: Optional[torch.Tensor] = None
    wind_available: Optional[torch.Tensor] = None
    hydro_ror: Optional[torch.Tensor] = None
    hydro_inflow: Optional[torch.Tensor] = None
    
    # Static capacities [Z]
    thermal_capacity: Optional[torch.Tensor] = None
    thermal_min: Optional[torch.Tensor] = None
    nuclear_capacity: Optional[torch.Tensor] = None
    battery_power: Optional[torch.Tensor] = None
    battery_capacity: Optional[torch.Tensor] = None
    pumped_power: Optional[torch.Tensor] = None
    pumped_capacity: Optional[torch.Tensor] = None
    hydro_capacity: Optional[torch.Tensor] = None
    dr_capacity: Optional[torch.Tensor] = None
    
    # Zone metadata
    zone_names: List[str] = field(default_factory=list)
    zone_to_region: Optional[Dict[str, int]] = None
    
    # Raw scenario dict (for LP worker)
    raw_data: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> "ScenarioData":
        """Move all tensors to device."""
        def move(t):
            return t.to(device) if t is not None else None
        
        return ScenarioData(
            scenario_id=self.scenario_id,
            n_zones=self.n_zones,
            n_timesteps=self.n_timesteps,
            dt_hours=self.dt_hours,
            demand=move(self.demand),
            solar_available=move(self.solar_available),
            wind_available=move(self.wind_available),
            hydro_ror=move(self.hydro_ror),
            hydro_inflow=move(self.hydro_inflow),
            thermal_capacity=move(self.thermal_capacity),
            thermal_min=move(self.thermal_min),
            nuclear_capacity=move(self.nuclear_capacity),
            battery_power=move(self.battery_power),
            battery_capacity=move(self.battery_capacity),
            pumped_power=move(self.pumped_power),
            pumped_capacity=move(self.pumped_capacity),
            hydro_capacity=move(self.hydro_capacity),
            dr_capacity=move(self.dr_capacity),
            zone_names=self.zone_names,
            zone_to_region=self.zone_to_region,
            raw_data=self.raw_data,
        )
    
    def get_decision_dim(self) -> int:
        """Get dimension of binary decision vector u."""
        # 8 binary decisions per (zone, timestep):
        # [battery_charge, battery_discharge, pumped_charge, pumped_discharge, 
        #  dr_active, nuclear_on, thermal_on, import_mode]
        return self.n_zones * self.n_timesteps * 8


@dataclass
class DecisionVector:
    """
    Binary decision vector u ∈ {0,1}^M for a scenario.
    
    Structure: [Z, T, 8] where 8 features are:
        0: battery_charging
        1: battery_discharging
        2: pumped_charging
        3: pumped_discharging
        4: dr_active
        5: nuclear_on
        6: thermal_on
        7: import_mode (1=importing, 0=exporting)
    """
    u: torch.Tensor  # [Z, T, 8] binary decisions
    
    # Optional: continuous relaxation for Langevin
    u_relaxed: Optional[torch.Tensor] = None  # [Z, T, 8] in (0,1)
    logits: Optional[torch.Tensor] = None  # [Z, T, 8] in R
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.u.shape
    
    @property
    def n_zones(self) -> int:
        return self.u.shape[0]
    
    @property
    def n_timesteps(self) -> int:
        return self.u.shape[1]
    
    def flatten(self) -> torch.Tensor:
        """Flatten to [Z*T*7] vector."""
        return self.u.flatten()
    
    @classmethod
    def from_flat(cls, flat: torch.Tensor, n_zones: int, n_timesteps: int) -> "DecisionVector":
        """Create from flattened vector."""
        u = flat.view(n_zones, n_timesteps, 7)
        return cls(u=u)
    
    def to(self, device: torch.device) -> "DecisionVector":
        return DecisionVector(
            u=self.u.to(device),
            u_relaxed=self.u_relaxed.to(device) if self.u_relaxed is not None else None,
            logits=self.logits.to(device) if self.logits is not None else None,
        )
    
    def binarize(self, threshold: float = 0.5) -> "DecisionVector":
        """Binarize relaxed decisions."""
        if self.u_relaxed is not None:
            u_binary = (self.u_relaxed > threshold).float()
        else:
            u_binary = (self.u > threshold).float()
        return DecisionVector(u=u_binary, u_relaxed=self.u_relaxed, logits=self.logits)


@dataclass
class CandidateResult:
    """
    Result from evaluating a candidate decision through decoder + LP worker.
    """
    decision: DecisionVector
    
    # LP worker output
    feasible: bool
    objective_value: float  # System cost (EUR)
    slack_used: float  # Unserved energy (MWh)
    solve_time: float
    
    # Stage info
    stage_used: str = "unknown"
    n_flips: int = 0
    
    # Continuous solution (optional, for analysis)
    continuous_vars: Optional[Dict[str, np.ndarray]] = None
    
    @property
    def cost(self) -> float:
        """Alias for objective_value."""
        return self.objective_value


@dataclass
class PreferencePair:
    """
    A preference pair for training: (u+, u-) where u+ is preferred over u-.
    
    The EBM should learn: E(u+ | h) < E(u- | h)
    """
    scenario_id: str
    
    # HTE embedding of scenario
    h: torch.Tensor  # [d] or [Z, T, d] depending on level
    
    # Positive example (MILP reference or best candidate)
    u_positive: DecisionVector
    cost_positive: float
    
    # Negative examples (worse candidates)
    u_negatives: List[DecisionVector]
    costs_negative: List[float]
    
    # Cost gap for weighting
    cost_gaps: Optional[List[float]] = None
    
    def to(self, device: torch.device) -> "PreferencePair":
        return PreferencePair(
            scenario_id=self.scenario_id,
            h=self.h.to(device),
            u_positive=self.u_positive.to(device),
            cost_positive=self.cost_positive,
            u_negatives=[u.to(device) for u in self.u_negatives],
            costs_negative=self.costs_negative,
            cost_gaps=self.cost_gaps,
        )


@dataclass
class TrainingBatch:
    """
    Batched training data for preference learning.
    """
    # Scenario embeddings [B, d]
    h: torch.Tensor
    
    # Positive decisions [B, Z, T, 7]
    u_positive: torch.Tensor
    costs_positive: torch.Tensor  # [B]
    
    # Negative decisions [B, K, Z, T, 7] where K is num negatives per scenario
    u_negatives: torch.Tensor
    costs_negative: torch.Tensor  # [B, K]
    
    # Cost-aware weights [B, K]
    weights: Optional[torch.Tensor] = None
    
    # Scenario IDs for reference
    scenario_ids: Optional[List[str]] = None
    
    def to(self, device: torch.device) -> "TrainingBatch":
        return TrainingBatch(
            h=self.h.to(device),
            u_positive=self.u_positive.to(device),
            costs_positive=self.costs_positive.to(device),
            u_negatives=self.u_negatives.to(device),
            costs_negative=self.costs_negative.to(device),
            weights=self.weights.to(device) if self.weights is not None else None,
            scenario_ids=self.scenario_ids,
        )
    
    @property
    def batch_size(self) -> int:
        return self.h.shape[0]
    
    @property
    def num_negatives(self) -> int:
        return self.u_negatives.shape[1]


@dataclass 
class MILPReference:
    """
    MILP oracle reference for a scenario.
    """
    scenario_id: str
    decision: DecisionVector
    objective_value: float
    solve_time: float
    status: SolveStatus
    
    # Detailed cost components
    cost_components: Optional[Dict[str, float]] = None
    
    @classmethod
    def from_report(cls, report: Dict[str, Any], scenario_id: str) -> "MILPReference":
        """Create from MILP report JSON."""
        mip_info = report.get("mip", {})
        
        # Extract binary decisions from detail section
        detail = report.get("detail", {})
        # This will be populated by loading dispatch data
        
        return cls(
            scenario_id=scenario_id,
            decision=None,  # Will be loaded separately
            objective_value=mip_info.get("objective", float("inf")),
            solve_time=mip_info.get("solve_seconds", 0.0),
            status=SolveStatus.OPTIMAL if mip_info.get("termination") == "optimal" else SolveStatus.FEASIBLE,
            cost_components=report.get("cost_components"),
        )
