"""
Quick script to run temporal GNN evaluation and generate plots.
Run this instead of the notebook if you encounter import/path issues.

Usage:
    cd notebooks
    python run_temporal_comparison.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import seaborn as sns
    sns.set_theme(style='whitegrid')
except ImportError:
    plt.style.use('ggplot')

from src.gnn.temporal_evaluation import evaluate_test_set

# Configuration
RUN_DIR = PROJECT_ROOT / 'outputs' / 'gnn_runs' / 'temporal_hetero'
TEST_SPLIT = 0.8
DEVICE = 'cpu'

print("="*60)
print("Temporal GNN Performance Comparison")
print("="*60)
print(f"\nConfiguration:")
print(f"  Model directory: {RUN_DIR}")
print(f"  Test split: {TEST_SPLIT} (using {(1-TEST_SPLIT)*100:.0f}% for testing)")
print(f"  Device: {DEVICE}")
print(f"  Project root: {PROJECT_ROOT}")

# Check if model exists
if not (RUN_DIR / 'best_model.pt').exists():
    print(f"\n❌ ERROR: No trained model found at {RUN_DIR / 'best_model.pt'}")
    print("\nTrain a model first:")
    print("  python -m src.gnn.train_temporal_hetero --epochs 50")
    sys.exit(1)

print("\n" + "="*60)
print("Running Evaluation...")
print("="*60)

# Run evaluation
try:
    results_df, summary = evaluate_test_set(
        run_dir=RUN_DIR,
        test_split=TEST_SPLIT,
        device=DEVICE,
        verbose=True
    )
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Display summary
print("\n" + "="*60)
print("Performance Summary")
print("="*60)
print(f"Scenarios evaluated:        {summary['num_scenarios']}")
print(f"Mean cost gap:              {summary['mean_cost_gap']:8.2%}")
print(f"Median cost gap:            {summary['median_cost_gap']:8.2%}")
print(f"Mean dispatch MAE:          {summary['mean_dispatch_mae']:8.2f} MW")
print(f"Mean violation rate:        {summary['mean_violation_rate']:8.2%}")
print(f"Mean speedup:               {summary['mean_speedup']:8.1f}x")
print(f"Total GNN inference time:   {summary['total_inference_time']:8.2f}s")
print(f"Mean MILP runtime:          {summary['mean_milp_runtime']:8.2f}s")
print("="*60)

# Save results
output_dir = PROJECT_ROOT / 'outputs' / 'evaluations'
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / 'temporal_gnn_results.csv'
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Generate plots
print("\nGenerating plots...")

sorted_df = results_df.sort_values('cost_gap').reset_index(drop=True)
x = range(len(sorted_df))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Cost Gap
ax = axes[0, 0]
colors = ['green' if gap < 0 else 'red' if gap > 0.1 else 'orange' for gap in sorted_df['cost_gap']]
ax.bar(x, sorted_df['cost_gap'], color=colors, alpha=0.7)
ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Scenario (sorted)')
ax.set_ylabel('Relative Cost Gap')
ax.set_title('GNN Cost vs MILP Objective')
ax.grid(True, alpha=0.3, axis='y')

# 2. Dispatch MAE
ax = axes[0, 1]
ax.bar(x, sorted_df['dispatch_mae'], color='#4c72b0', alpha=0.7)
ax.set_xlabel('Scenario')
ax.set_ylabel('MAE (MW)')
ax.set_title('Dispatch Prediction Accuracy')
ax.grid(True, alpha=0.3, axis='y')

# 3. Violation Rate
ax = axes[1, 0]
if sorted_df['violation_rate'].sum() > 0:
    ax.bar(x, sorted_df['violation_rate'], color='#dd8452', alpha=0.7)
    ax.set_ylim(0, 1)
else:
    ax.text(0.5, 0.5, 'Violation rate not available',
            ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('Scenario')
ax.set_ylabel('Violation Rate')
ax.set_title('Feasibility')
ax.grid(True, alpha=0.3, axis='y')

# 4. Speedup
ax = axes[1, 1]
ax.bar(x, sorted_df['speedup'], color='#55a868', alpha=0.7)
ax.set_xlabel('Scenario')
ax.set_ylabel('Speedup Factor')
ax.set_title('Runtime: MILP / GNN (log scale)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

plot_path = output_dir / 'temporal_gnn_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to: {plot_path}")

plt.show()

print("\n✓ Evaluation complete!")
print(f"\nView results at: {csv_path}")
print(f"View plots at: {plot_path}")
