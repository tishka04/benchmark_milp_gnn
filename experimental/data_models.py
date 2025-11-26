"""
Data models for canonical UC+Dispatch problem.
Single source of truth for problem representation.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
import json


@dataclass
class UCInstance:
    """
    Canonical Unit Commitment + Dispatch instance.
    
    This is the single source of truth for problem data.
    Both MILP and Hybrid solvers must use this exact structure.
    """
    # Problem dimensions (required)
    n_units: int  # N: number of thermal units
    n_periods: int  # T: number of time periods
    
    # Unit characteristics [N] (required)
    p_min: np.ndarray  # Minimum power output (MW)
    p_max: np.ndarray  # Maximum power output (MW)
    marginal_cost: np.ndarray  # Variable cost ($/MWh)
    startup_cost: np.ndarray  # Start-up cost ($)
    
    # Demand profile [T] (required)
    demand: np.ndarray  # Load demand per period (MW)
    
    # Optional: shutdown cost
    shutdown_cost: Optional[np.ndarray] = None  # Shutdown cost ($)
    
    # Optional: ramping and min up/down time constraints
    ramp_up: Optional[np.ndarray] = None  # Max ramp up (MW/period)
    ramp_down: Optional[np.ndarray] = None  # Max ramp down (MW/period)
    min_up_time: Optional[np.ndarray] = None  # Minimum up time (periods)
    min_down_time: Optional[np.ndarray] = None  # Minimum down time (periods)
    
    # Optional: initial conditions [N]
    initial_status: Optional[np.ndarray] = None  # Initial on/off status
    initial_power: Optional[np.ndarray] = None  # Initial power output
    
    # Instance metadata (optional)
    seed: int = 0  # Random seed for reproducibility
    scale_name: str = ""  # e.g., "N50_T24"
    instance_id: int = 0  # Instance number within scale
    
    def __post_init__(self):
        """Validate dimensions and convert lists to arrays."""
        self.p_min = np.asarray(self.p_min, dtype=float)
        self.p_max = np.asarray(self.p_max, dtype=float)
        self.marginal_cost = np.asarray(self.marginal_cost, dtype=float)
        self.startup_cost = np.asarray(self.startup_cost, dtype=float)
        self.demand = np.asarray(self.demand, dtype=float)
        
        if self.shutdown_cost is None:
            self.shutdown_cost = np.zeros(self.n_units)
        else:
            self.shutdown_cost = np.asarray(self.shutdown_cost, dtype=float)
        
        # Validate shapes
        assert self.p_min.shape == (self.n_units,), f"p_min shape mismatch: {self.p_min.shape} != ({self.n_units},)"
        assert self.p_max.shape == (self.n_units,), f"p_max shape mismatch: {self.p_max.shape} != ({self.n_units},)"
        assert self.marginal_cost.shape == (self.n_units,), f"marginal_cost shape mismatch"
        assert self.startup_cost.shape == (self.n_units,), f"startup_cost shape mismatch"
        assert self.demand.shape == (self.n_periods,), f"demand shape mismatch: {self.demand.shape} != ({self.n_periods},)"
        
        # Set defaults for initial conditions
        if self.initial_status is None:
            self.initial_status = np.zeros(self.n_units, dtype=int)
        if self.initial_power is None:
            self.initial_power = np.zeros(self.n_units)
        
        self.initial_status = np.asarray(self.initial_status, dtype=int)
        self.initial_power = np.asarray(self.initial_power, dtype=float)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UCInstance':
        """Load from dictionary."""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save instance to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'UCInstance':
        """Load instance from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class UCSolution:
    """
    Solution to a UC+Dispatch problem.
    Used by both MILP and Hybrid solvers.
    """
    # Commitment schedule [N, T]
    commitment: np.ndarray  # Binary: u[i,t] ∈ {0,1}
    
    # Dispatch schedule [N, T]
    power: np.ndarray  # Continuous: P[i,t] ≥ 0
    
    # Objective breakdown
    total_cost: float
    fuel_cost: float
    startup_cost: float
    shutdown_cost: float = 0.0
    
    # Solution metadata
    solve_time: float = 0.0
    method: str = ""  # "MILP", "Hybrid", etc.
    feasible: bool = True
    optimal: bool = False
    mip_gap: Optional[float] = None  # For MILP: (UB - LB) / LB
    best_bound: Optional[float] = None  # For MILP: lower bound
    
    # Convergence info
    iterations: Optional[int] = None  # For hybrid: number of samples
    time_to_first_feasible: Optional[float] = None
    
    # Checkpoint data for anytime curves
    checkpoint_times: list = field(default_factory=list)
    checkpoint_costs: list = field(default_factory=list)
    
    def __post_init__(self):
        """Convert to arrays."""
        self.commitment = np.asarray(self.commitment, dtype=int)
        self.power = np.asarray(self.power, dtype=float)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            else:
                data[key] = value
        return data
    
    def save(self, filepath: str):
        """Save solution to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs."""
    # Scale grid
    n_units_list: list = field(default_factory=lambda: [20, 50, 100, 200, 400])
    n_periods_list: list = field(default_factory=lambda: [24, 96])
    instances_per_scale: int = 10
    
    # Time budgets (seconds) - function of scale
    time_budget_small: float = 300  # N <= 50
    time_budget_medium: float = 600  # 50 < N <= 100
    time_budget_large: float = 1800  # 100 < N <= 200
    time_budget_xlarge: float = 3600  # N > 200
    
    # Reference run budget (for J_ref)
    reference_time_budget: float = 7200  # 2 hours
    
    # MILP solver settings
    milp_solver: str = "highs"  # "highs", "gurobi", "cplex"
    mip_gap_tolerance: float = 1e-3
    milp_threads: int = 1  # For fairness
    
    # Hybrid solver settings
    hybrid_n_samples_per_period: int = 10  # Thermodynamic samples per period
    hybrid_n_seeds: int = 5  # Different random seeds
    hybrid_temperature: float = 5.0  # Ising temperature
    hybrid_warmup: int = 200  # MCMC warmup steps
    hybrid_steps_per_sample: int = 3  # MCMC steps between samples
    
    # Logging
    checkpoint_times: list = field(default_factory=lambda: [1, 10, 30, 60] + list(range(120, 3600, 60)))
    
    def get_time_budget(self, n_units: int) -> float:
        """Get time budget for a given problem scale."""
        if n_units <= 50:
            return self.time_budget_small
        elif n_units <= 100:
            return self.time_budget_medium
        elif n_units <= 200:
            return self.time_budget_large
        else:
            return self.time_budget_xlarge
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
