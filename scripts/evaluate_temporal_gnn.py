"""
Evaluate temporal GNN and generate performance comparison plots.

Usage:
    python scripts/evaluate_temporal_gnn.py
    python scripts/evaluate_temporal_gnn.py --run-dir outputs/gnn_runs/temporal_hetero --save-plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn.temporal_evaluation import evaluate_test_set

sns.set_theme(style='whitegrid')


def plot_results(results_df, summary, save_dir=None):
    """Generate all comparison plots."""
    
    # Sort by cost gap
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
        ax.text(0.5, 0.5, 'Violation rate not available\n(unserved not in targets)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Violation Rate')
    ax.set_title('Feasibility (Constraint Violations)')
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
    
    if save_dir:
        save_path = Path(save_dir) / 'temporal_gnn_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    # Summary statistics plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics = {
        'Cost Gap\n(mean)': summary['mean_cost_gap'] * 100,
        'Dispatch MAE\n(MW)': summary['mean_dispatch_mae'],
        'Violation Rate\n(mean, %)': summary['mean_violation_rate'] * 100,
        'Speedup\n(mean, x)': summary['mean_speedup'],
    }
    
    colors_map = ['red' if 'Cost' in k else 'blue' if 'MAE' in k else 'orange' if 'Violation' in k else 'green' for k in metrics.keys()]
    
    ax.barh(list(metrics.keys()), list(metrics.values()), color=colors_map, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'temporal_gnn_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved summary to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal GNN vs MILP")
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/gnn_runs/temporal_hetero'))
    parser.add_argument('--test-split', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-plots', action='store_true', help="Save plots to run directory")
    parser.add_argument('--save-csv', type=Path, default=None, help="Save results to CSV")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Temporal GNN vs MILP Performance Comparison")
    print("="*60)
    print()
    
    # Run evaluation
    results_df, summary = evaluate_test_set(args.run_dir, args.test_split, args.device, verbose=True)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics")
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
    
    # Generate plots
    save_dir = args.run_dir if args.save_plots else None
    plot_results(results_df, summary, save_dir)
    
    # Save CSV
    if args.save_csv:
        results_df.to_csv(args.save_csv, index=False)
        print(f"\n✓ Results saved to {args.save_csv}")
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
