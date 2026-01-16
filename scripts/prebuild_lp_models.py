# ==============================================================================
# PRE-BUILD LP MODELS FOR FAST TRAINING
# ==============================================================================
# This script pre-builds Pyomo models for all scenarios and saves them to disk.
# The CachedLPOracle can then load these pre-built models for fast evaluation.
#
# Usage:
#   python scripts/prebuild_lp_models.py --scenarios_dir outputs/scenarios_v2 --output_dir outputs/lp_models
#
# Note: Pyomo models are saved using cloudpickle for complex object serialization.
# ==============================================================================

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cloudpickle
    HAS_CLOUDPICKLE = True
except ImportError:
    import pickle
    HAS_CLOUDPICKLE = False
    print("Warning: cloudpickle not available, using pickle (may fail on complex Pyomo models)")


@dataclass
class PrebuiltModel:
    """Container for pre-built model data."""
    scenario_id: str
    zone_names: List[str]
    n_timesteps: int
    n_zones: int
    model_bytes: bytes  # Serialized Pyomo model
    build_time: float
    

def build_and_save_model(
    scenario_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> Optional[str]:
    """
    Build and save a single Pyomo model.
    
    Returns scenario_id on success, None on failure.
    """
    scenario_id = scenario_path.stem
    output_path = output_dir / f"{scenario_id}.pkl"
    
    # Skip if already exists
    if output_path.exists():
        if verbose:
            print(f"  Skipping {scenario_id} (already exists)")
        return scenario_id
    
    try:
        from src.milp.scenario_loader import load_scenario_data
        from src.milp.model import build_uc_model
        
        start_time = time.time()
        
        # Load scenario data
        scenario_data = load_scenario_data(scenario_path)
        
        # Build model
        model = build_uc_model(scenario_data, enable_duals=False)
        zone_names = list(model.Z)
        n_timesteps = len(list(model.T))
        n_zones = len(zone_names)
        
        build_time = time.time() - start_time
        
        # Serialize model
        if HAS_CLOUDPICKLE:
            model_bytes = cloudpickle.dumps(model)
        else:
            import pickle
            model_bytes = pickle.dumps(model)
        
        # Create container
        prebuilt = PrebuiltModel(
            scenario_id=scenario_id,
            zone_names=zone_names,
            n_timesteps=n_timesteps,
            n_zones=n_zones,
            model_bytes=model_bytes,
            build_time=build_time,
        )
        
        # Save
        with open(output_path, 'wb') as f:
            if HAS_CLOUDPICKLE:
                cloudpickle.dump(prebuilt, f)
            else:
                import pickle
                pickle.dump(prebuilt, f)
        
        if verbose:
            print(f"  Built {scenario_id}: {n_zones} zones, {n_timesteps} timesteps, {build_time:.1f}s")
        
        return scenario_id
        
    except Exception as e:
        print(f"  Error building {scenario_id}: {e}")
        return None


def build_all_models(
    scenarios_dir: str,
    output_dir: str,
    max_scenarios: Optional[int] = None,
    n_workers: int = 1,
    verbose: bool = True,
):
    """
    Build all Pyomo models in parallel and save to disk.
    """
    scenarios_path = Path(scenarios_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all scenario files
    scenario_files = sorted(list(scenarios_path.glob("scenario_*.json")))
    
    if max_scenarios:
        scenario_files = scenario_files[:max_scenarios]
    
    print(f"=" * 60)
    print(f"PRE-BUILDING LP MODELS")
    print(f"=" * 60)
    print(f"Scenarios dir: {scenarios_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total scenarios: {len(scenario_files)}")
    print(f"Workers: {n_workers}")
    print(f"Serializer: {'cloudpickle' if HAS_CLOUDPICKLE else 'pickle'}")
    print(f"=" * 60)
    
    # Check existing
    existing = list(output_path.glob("scenario_*.pkl"))
    print(f"Already built: {len(existing)}")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    if n_workers == 1:
        # Sequential (safer for debugging)
        for scenario_path in tqdm(scenario_files, desc="Building models"):
            result = build_and_save_model(scenario_path, output_path, verbose=False)
            if result:
                successful += 1
            else:
                failed += 1
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(build_and_save_model, sp, output_path, False): sp
                for sp in scenario_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building models"):
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"COMPLETE")
    print(f"=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/max(1, successful):.1f}s per model")
    print(f"Output: {output_path}")
    print(f"=" * 60)
    
    # Create index file
    index = {
        "scenarios_dir": str(scenarios_dir),
        "n_models": successful,
        "build_time": total_time,
        "serializer": "cloudpickle" if HAS_CLOUDPICKLE else "pickle",
    }
    with open(output_path / "index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    return successful, failed


def load_prebuilt_model(model_path: Path) -> Optional[PrebuiltModel]:
    """Load a pre-built model from disk."""
    try:
        with open(model_path, 'rb') as f:
            if HAS_CLOUDPICKLE:
                return cloudpickle.load(f)
            else:
                import pickle
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pre-build LP models for fast training")
    parser.add_argument("--scenarios_dir", type=str, default="outputs/scenarios_v2",
                        help="Directory containing scenario JSON files")
    parser.add_argument("--output_dir", type=str, default="outputs/lp_models",
                        help="Directory to save pre-built models")
    parser.add_argument("--max_scenarios", type=int, default=None,
                        help="Maximum number of scenarios to build")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    build_all_models(
        scenarios_dir=args.scenarios_dir,
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        n_workers=args.workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
