"""
Instance generator for canonical UC+Dispatch problem.
Creates reproducible instances with fixed random seeds.
"""
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from .data_models import UCInstance
except ImportError:
    from data_models import UCInstance


class UCInstanceGenerator:
    """
    Generates Unit Commitment instances with reproducible randomness.
    
    Creates instances of varying scale (N, T) with realistic parameters:
    - Unit capacities from plausible ranges
    - Cost structures with modest spread
    - Demand profiles with daily patterns + noise
    """
    
    def __init__(
        self,
        capacity_range: tuple[float, float] = (50.0, 500.0),
        cost_range: tuple[float, float] = (20.0, 80.0),
        startup_cost_factor: tuple[float, float] = (100.0, 1000.0),
        demand_peak_factor: float = 0.85,  # Peak demand as fraction of total capacity
        demand_base_factor: float = 0.40,  # Base demand as fraction of peak
    ):
        self.capacity_range = capacity_range
        self.cost_range = cost_range
        self.startup_cost_factor = startup_cost_factor
        self.demand_peak_factor = demand_peak_factor
        self.demand_base_factor = demand_base_factor
    
    def generate(
        self,
        n_units: int,
        n_periods: int,
        seed: int,
        instance_id: int = 0
    ) -> UCInstance:
        """
        Generate a single instance with given parameters.
        
        Args:
            n_units: Number of thermal units
            n_periods: Number of time periods
            seed: Random seed for reproducibility
            instance_id: Instance ID within the scale
        
        Returns:
            UCInstance: Generated instance
        """
        rng = np.random.RandomState(seed)
        
        # Generate unit capacities (mix of large, medium, small)
        p_max = self._generate_capacities(n_units, rng)
        
        # Minimum generation: 25-40% of max capacity
        min_gen_fraction = rng.uniform(0.25, 0.40, n_units)
        p_min = p_max * min_gen_fraction
        
        # Marginal costs: inversely correlated with size (larger = cheaper)
        # with some randomness
        normalized_size = (p_max - p_max.min()) / (p_max.max() - p_max.min() + 1e-6)
        base_cost = self.cost_range[1] - (self.cost_range[1] - self.cost_range[0]) * normalized_size
        cost_noise = rng.uniform(-5, 5, n_units)
        marginal_cost = np.clip(base_cost + cost_noise, self.cost_range[0], self.cost_range[1])
        
        # Startup costs: proportional to capacity with randomness
        startup_base = rng.uniform(self.startup_cost_factor[0], self.startup_cost_factor[1], n_units)
        startup_cost = startup_base * (p_max / p_max.mean())
        
        # Generate demand profile
        demand = self._generate_demand_profile(n_periods, p_max.sum(), rng)
        
        # Ensure feasibility: adjust if needed
        demand = self._ensure_feasibility(demand, p_min, p_max)
        
        scale_name = f"N{n_units}_T{n_periods}"
        
        return UCInstance(
            n_units=n_units,
            n_periods=n_periods,
            p_min=p_min,
            p_max=p_max,
            marginal_cost=marginal_cost,
            startup_cost=startup_cost,
            demand=demand,
            seed=seed,
            scale_name=scale_name,
            instance_id=instance_id
        )
    
    def _generate_capacities(self, n_units: int, rng: np.random.RandomState) -> np.ndarray:
        """Generate unit capacities with realistic distribution."""
        # Mix of unit sizes
        n_large = max(2, n_units // 10)  # 10% large units
        n_medium = max(3, n_units // 3)  # 33% medium units
        n_small = n_units - n_large - n_medium  # Rest are small
        
        capacities = np.concatenate([
            rng.uniform(300, 500, n_large),    # Large: 300-500 MW
            rng.uniform(100, 300, n_medium),   # Medium: 100-300 MW
            rng.uniform(50, 150, n_small)      # Small: 50-150 MW
        ])
        
        rng.shuffle(capacities)
        return capacities
    
    def _generate_demand_profile(
        self,
        n_periods: int,
        total_capacity: float,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate realistic demand profile with daily patterns.
        
        The profile has:
        - Morning peak (7-9am)
        - Evening peak (18-20pm, highest)
        - Night valley (2-5am, lowest)
        - Smooth transitions
        """
        # Convert periods to hours (assuming periods span 24h or multiples)
        if n_periods == 24:
            hours_per_period = 1.0
        elif n_periods == 96:
            hours_per_period = 0.25  # 15-min intervals
        elif n_periods == 168:
            hours_per_period = 1.0  # Weekly hourly
        else:
            hours_per_period = 24.0 / n_periods
        
        hours = np.arange(n_periods) * hours_per_period
        
        # Peak demand as fraction of total capacity
        peak_demand = total_capacity * self.demand_peak_factor
        base_demand = peak_demand * self.demand_base_factor
        
        demand = np.zeros(n_periods)
        
        for t, h in enumerate(hours):
            h_mod = h % 24  # Daily pattern
            
            # Morning peak (7-9am)
            morning = 0.25 * np.exp(-((h_mod - 8)**2) / 4)
            
            # Evening peak (18-20pm) - highest
            evening = 0.45 * np.exp(-((h_mod - 19)**2) / 4)
            
            # Night valley (2-5am) - lowest
            night = -0.15 * np.exp(-((h_mod - 3)**2) / 4)
            
            demand[t] = base_demand * (1.0 + morning + evening + night)
        
        # Add small noise for realism
        noise = rng.normal(0, 0.02 * peak_demand, n_periods)
        demand = demand + noise
        
        # Ensure positive
        demand = np.maximum(demand, 0.1 * base_demand)
        
        return demand
    
    def _ensure_feasibility(
        self,
        demand: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray
    ) -> np.ndarray:
        """
        Ensure the demand profile is feasible given unit constraints.
        
        Adjusts demand if:
        1. Peak demand > total capacity
        2. Valley demand < minimum generation of smallest unit
        """
        total_capacity = p_max.sum()
        smallest_min_gen = p_min.min()
        
        demand_adjusted = demand.copy()
        
        # Check 1: Peak demand should be feasible
        peak_demand = demand.max()
        if peak_demand > total_capacity * 0.95:  # Leave 5% margin
            scale_factor = (total_capacity * 0.90) / peak_demand
            demand_adjusted *= scale_factor
            print(f"  [Adjusted] Scaled demand by {scale_factor:.3f} to ensure peak feasibility")
        
        # Check 2: Valley demand should be achievable
        valley_demand = demand.min()
        if valley_demand < smallest_min_gen:
            # Shift demand up
            shift = smallest_min_gen - valley_demand + 10  # +10 MW buffer
            demand_adjusted += shift
            print(f"  [Adjusted] Shifted demand by +{shift:.1f} MW to ensure valley feasibility")
        
        return demand_adjusted
    
    def generate_dataset(
        self,
        n_units_list: list[int],
        n_periods_list: list[int],
        instances_per_scale: int,
        output_dir: Path,
        base_seed: int = 42
    ) -> dict:
        """
        Generate a complete dataset of instances.
        
        Args:
            n_units_list: List of unit counts to generate
            n_periods_list: List of period counts to generate
            instances_per_scale: Number of instances per (N, T) pair
            output_dir: Directory to save instances
            base_seed: Base random seed
        
        Returns:
            Dictionary mapping scale names to instance file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = {}
        instance_counter = 0
        
        print("=" * 80)
        print("GENERATING INSTANCE DATASET")
        print("=" * 80)
        
        for n_units in n_units_list:
            for n_periods in n_periods_list:
                scale_name = f"N{n_units}_T{n_periods}"
                scale_dir = output_dir / scale_name
                scale_dir.mkdir(exist_ok=True)
                
                print(f"\n{scale_name}:")
                dataset[scale_name] = []
                
                for i in range(instances_per_scale):
                    seed = base_seed + instance_counter
                    instance = self.generate(n_units, n_periods, seed, i)
                    
                    # Save instance
                    filename = f"instance_{i:03d}.json"
                    filepath = scale_dir / filename
                    instance.save(str(filepath))
                    
                    dataset[scale_name].append(str(filepath))
                    instance_counter += 1
                    
                    if i % 5 == 0:
                        print(f"  Generated instance {i}/{instances_per_scale}", end="\r")
                
                print(f"  Generated {instances_per_scale} instances for {scale_name}")
        
        print("\n" + "=" * 80)
        print(f"DATASET COMPLETE: {instance_counter} instances")
        print("=" * 80)
        
        # Save dataset index
        index_file = output_dir / "dataset_index.json"
        import json
        with open(index_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"\nDataset index saved: {index_file}")
        
        return dataset


if __name__ == "__main__":
    """Generate a default dataset for testing."""
    generator = UCInstanceGenerator()
    
    output_dir = Path(__file__).parent / "instances"
    
    dataset = generator.generate_dataset(
        n_units_list=[20, 50, 100],
        n_periods_list=[24, 96],
        instances_per_scale=10,
        output_dir=output_dir,
        base_seed=42
    )
    
    print("\nDataset ready for experimentation!")
