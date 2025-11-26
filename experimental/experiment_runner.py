"""
Experiment runner: orchestrates the full experimental protocol.

Runs experiments according to the research protocol:
1. Generate instances
2. Run reference MILP (large budget for J_ref)
3. Run MILP with standard budget
4. Run Hybrid with standard budget
5. Collect metrics and save results
"""
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    from .data_models import UCInstance, UCSolution, ExperimentConfig
    from .instance_generator import UCInstanceGenerator
    from .milp_solver import MILPSolver
    from .hybrid_solver import HybridSolver
except ImportError:
    from data_models import UCInstance, UCSolution, ExperimentConfig
    from instance_generator import UCInstanceGenerator
    from milp_solver import MILPSolver
    from hybrid_solver import HybridSolver


class ExperimentRunner:
    """
    Orchestrates the complete experimental protocol.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.milp_solver = MILPSolver(config)
        self.hybrid_solver = HybridSolver(config)
        
        # Save configuration
        config.save(str(self.output_dir / "experiment_config.json"))
    
    def run_full_protocol(self, regenerate_instances: bool = False):
        """
        Run the complete experimental protocol.
        
        Steps:
        1. Generate instances (or load existing)
        2. Run reference MILP for J_ref
        3. Run standard-budget experiments
        4. Analyze results
        """
        print("\n" + "="*100)
        print("EXPERIMENTAL PROTOCOL: HYBRID VS MILP BENCHMARK")
        print("="*100)
        
        # Step 1: Generate/load instances
        instances_dir = self.output_dir / "instances"
        
        if regenerate_instances or not instances_dir.exists():
            print("\n" + "="*100)
            print("STEP 1: GENERATING INSTANCES")
            print("="*100)
            
            generator = UCInstanceGenerator()
            dataset = generator.generate_dataset(
                n_units_list=self.config.n_units_list,
                n_periods_list=self.config.n_periods_list,
                instances_per_scale=self.config.instances_per_scale,
                output_dir=instances_dir,
                base_seed=42
            )
        else:
            print("\n" + "="*100)
            print("STEP 1: LOADING EXISTING INSTANCES")
            print("="*100)
            
            with open(instances_dir / "dataset_index.json", 'r') as f:
                dataset = json.load(f)
            
            print(f"Loaded {len(dataset)} scales from {instances_dir}")
        
        # Step 2: Run reference MILP
        print("\n" + "="*100)
        print("STEP 2: REFERENCE MILP RUNS (Large Budget for J_ref)")
        print("="*100)
        
        self._run_reference_experiments(dataset)
        
        # Step 3: Run standard-budget experiments
        print("\n" + "="*100)
        print("STEP 3: STANDARD-BUDGET EXPERIMENTS")
        print("="*100)
        
        self._run_standard_experiments(dataset)
        
        # Step 4: Analyze results
        print("\n" + "="*100)
        print("STEP 4: ANALYSIS")
        print("="*100)
        
        self._analyze_results()
        
        print("\n" + "="*100)
        print("PROTOCOL COMPLETE")
        print("="*100)
        print(f"\nResults saved to: {self.output_dir}")
    
    def _run_reference_experiments(self, dataset: Dict):
        """Run MILP with large time budget to establish J_ref."""
        results_dir = self.output_dir / "results_reference"
        results_dir.mkdir(exist_ok=True)
        
        for scale_name, instance_paths in dataset.items():
            print(f"\n{scale_name}:")
            scale_results_dir = results_dir / scale_name
            scale_results_dir.mkdir(exist_ok=True)
            
            for i, instance_path in enumerate(instance_paths):
                instance = UCInstance.load(instance_path)
                
                print(f"  Instance {i+1}/{len(instance_paths)}: ", end="")
                
                # Check if already solved
                result_file = scale_results_dir / f"instance_{i:03d}_milp_ref.json"
                if result_file.exists():
                    print("(cached)")
                    continue
                
                # Solve with large budget
                solution = self.milp_solver.solve(
                    instance,
                    time_limit=self.config.reference_time_budget
                )
                
                # Save
                solution.save(str(result_file))
                print(f"Cost=${solution.total_cost:,.0f}, Time={solution.solve_time:.1f}s")
    
    def _run_standard_experiments(self, dataset: Dict):
        """Run MILP and Hybrid with standard time budgets."""
        results_dir = self.output_dir / "results_standard"
        results_dir.mkdir(exist_ok=True)
        
        for scale_name, instance_paths in dataset.items():
            print(f"\n{scale_name}:")
            scale_results_dir = results_dir / scale_name
            scale_results_dir.mkdir(exist_ok=True)
            
            # Get time budget for this scale
            n_units = int(scale_name.split('_')[0][1:])
            time_budget = self.config.get_time_budget(n_units)
            
            print(f"  Time budget: {time_budget}s ({time_budget/60:.1f} min)")
            
            for i, instance_path in enumerate(instance_paths):
                instance = UCInstance.load(instance_path)
                
                print(f"\n  Instance {i+1}/{len(instance_paths)}:")
                
                # MILP
                milp_result_file = scale_results_dir / f"instance_{i:03d}_milp.json"
                if not milp_result_file.exists():
                    print("    Running MILP...", end=" ")
                    milp_solution = self.milp_solver.solve(
                        instance,
                        time_limit=time_budget,
                        checkpoint_times=self.config.checkpoint_times
                    )
                    milp_solution.save(str(milp_result_file))
                    print(f"Done (${milp_solution.total_cost:,.0f}, {milp_solution.solve_time:.1f}s)")
                else:
                    print("    MILP: (cached)")
                
                # Hybrid
                hybrid_result_file = scale_results_dir / f"instance_{i:03d}_hybrid.json"
                if not hybrid_result_file.exists():
                    print("    Running Hybrid...", end=" ")
                    hybrid_solution = self.hybrid_solver.solve(
                        instance,
                        time_limit=time_budget,
                        checkpoint_times=self.config.checkpoint_times
                    )
                    hybrid_solution.save(str(hybrid_result_file))
                    print(f"Done (${hybrid_solution.total_cost:,.0f}, {hybrid_solution.solve_time:.1f}s)")
                else:
                    print("    Hybrid: (cached)")
    
    def _analyze_results(self):
        """Analyze and aggregate results."""
        try:
            from .analysis import ResultsAnalyzer
        except ImportError:
            from analysis import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer(self.output_dir)
        analyzer.analyze_all()
        analyzer.generate_report()


def run_simple_test():
    """Run a simple test with small instances."""
    config = ExperimentConfig(
        n_units_list=[10, 20],
        n_periods_list=[24],
        instances_per_scale=3,
        time_budget_small=60,
        reference_time_budget=300
    )
    
    output_dir = Path(__file__).parent / "test_results"
    runner = ExperimentRunner(config, output_dir)
    runner.run_full_protocol(regenerate_instances=True)


if __name__ == "__main__":
    run_simple_test()
