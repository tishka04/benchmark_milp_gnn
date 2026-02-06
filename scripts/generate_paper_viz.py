"""
Generate paper-quality scenario visualizations for simple vs critical scenarios.

This script creates publication-ready figures comparing scenario #3691 (simple)
and scenario #3544 (critical) for inclusion in the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milp.scenario_loader import load_scenario_data


# Paper-quality settings
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for assets
COLORS = {
    'thermal': '#d95f02',
    'nuclear': '#7570b3',
    'solar': '#FDB462',
    'wind': '#80B1D3',
    'battery': '#e7298a',
    'hydro_reservoir': '#66a61e',
    'hydro_ror': '#a6761d',
    'pumped': '#e6ab02',
    'dr': '#66c2a5',
}


def load_scenario_json(scenario_path: Path) -> dict:
    """Load raw scenario JSON."""
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def plot_scenario_overview(scenario_path: Path, out_path: Path, title_suffix: str = "") -> None:
    """
    Create a paper-quality scenario overview figure with:
    - Top: Installed capacities by zone (stacked bar)
    - Bottom: Demand profiles over time
    """
    data = load_scenario_data(scenario_path)
    raw = load_scenario_json(scenario_path)
    
    zones = data.zones
    n_zones = len(zones)
    idx = range(n_zones)
    
    # Capacity data
    thermal_cap = [data.thermal_capacity.get(z, 0.0) for z in zones]
    nuclear_cap = [data.nuclear_capacity.get(z, 0.0) for z in zones]
    solar_cap = [data.solar_capacity.get(z, 0.0) for z in zones]
    wind_cap = [data.wind_capacity.get(z, 0.0) for z in zones]
    battery_power = [data.battery_power.get(z, 0.0) for z in zones]
    hydro_cap = [data.hydro_res_capacity.get(z, 0.0) for z in zones]
    
    # Demand profiles
    periods = data.periods
    time_hours = [t * data.dt_hours for t in periods]
    demand_curves = {z: [data.demand.get((z, t), 0.0) for t in periods] for z in zones}
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1.2, 1]})
    
    # === Top: Stacked bar chart of capacities ===
    bar_width = 0.7
    
    # Stack order: thermal, nuclear, solar, wind, battery, hydro
    bottom = np.zeros(n_zones)
    
    if any(v > 0 for v in thermal_cap):
        axes[0].bar(idx, thermal_cap, bar_width, bottom=bottom, label='Thermal', color=COLORS['thermal'])
        bottom += thermal_cap
    
    if any(v > 0 for v in nuclear_cap):
        axes[0].bar(idx, nuclear_cap, bar_width, bottom=bottom, label='Nuclear', color=COLORS['nuclear'])
        bottom += nuclear_cap
    
    if any(v > 0 for v in solar_cap):
        axes[0].bar(idx, solar_cap, bar_width, bottom=bottom, label='Solar', color=COLORS['solar'])
        bottom += solar_cap
    
    if any(v > 0 for v in wind_cap):
        axes[0].bar(idx, wind_cap, bar_width, bottom=bottom, label='Wind', color=COLORS['wind'])
        bottom += wind_cap
    
    if any(v > 0 for v in battery_power):
        axes[0].bar(idx, battery_power, bar_width, bottom=bottom, label='Battery', color=COLORS['battery'])
        bottom += battery_power
    
    if any(v > 0 for v in hydro_cap):
        axes[0].bar(idx, hydro_cap, bar_width, bottom=bottom, label='Hydro Reservoir', color=COLORS['hydro_reservoir'])
    
    axes[0].set_ylabel('Installed Capacity (MW)')
    axes[0].set_title(f'Installed Capacities by Zone{title_suffix}', fontweight='bold')
    axes[0].legend(loc='upper right', ncol=3, framealpha=0.9)
    axes[0].set_xticks([])  # Too many zones to show labels
    axes[0].set_xlabel(f'Zones (n={n_zones})')
    axes[0].grid(axis='y', alpha=0.3)
    
    # === Bottom: Demand profiles ===
    # Use a colormap for zones
    cmap = plt.cm.get_cmap('tab20', min(20, n_zones))
    
    for i, zone in enumerate(zones):
        color = cmap(i % 20)
        axes[1].plot(time_hours, demand_curves[zone], color=color, alpha=0.6, linewidth=0.8)
    
    # Add aggregate demand
    total_demand = [sum(demand_curves[z][t] for z in zones) for t in range(len(periods))]
    axes[1].plot(time_hours, [d / n_zones for d in total_demand], 
                 color='black', linewidth=2, label='Mean demand')
    
    axes[1].set_ylabel('Demand (MW)')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_title(f'Demand Profiles{title_suffix}', fontweight='bold')
    axes[1].set_xlim(0, 24)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scenario_comparison(
    simple_path: Path, 
    critical_path: Path, 
    out_path: Path
) -> None:
    """
    Create a side-by-side comparison figure of simple vs critical scenarios.
    """
    simple_data = load_scenario_data(simple_path)
    critical_data = load_scenario_data(critical_path)
    simple_raw = load_scenario_json(simple_path)
    critical_raw = load_scenario_json(critical_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # === Row 1: Capacity comparison ===
    for col, (data, raw, label) in enumerate([
        (simple_data, simple_raw, 'Simple (#3691)'),
        (critical_data, critical_raw, 'Critical (#3544)')
    ]):
        ax = axes[0, col]
        zones = data.zones
        n_zones = len(zones)
        idx = range(n_zones)
        
        thermal_cap = [data.thermal_capacity.get(z, 0.0) for z in zones]
        nuclear_cap = [data.nuclear_capacity.get(z, 0.0) for z in zones]
        solar_cap = [data.solar_capacity.get(z, 0.0) for z in zones]
        wind_cap = [data.wind_capacity.get(z, 0.0) for z in zones]
        battery_power = [data.battery_power.get(z, 0.0) for z in zones]
        hydro_cap = [data.hydro_res_capacity.get(z, 0.0) for z in zones]
        
        bar_width = 0.8
        bottom = np.zeros(n_zones)
        
        if any(v > 0 for v in thermal_cap):
            ax.bar(idx, thermal_cap, bar_width, bottom=bottom, label='Thermal', color=COLORS['thermal'])
            bottom += thermal_cap
        if any(v > 0 for v in nuclear_cap):
            ax.bar(idx, nuclear_cap, bar_width, bottom=bottom, label='Nuclear', color=COLORS['nuclear'])
            bottom += nuclear_cap
        if any(v > 0 for v in solar_cap):
            ax.bar(idx, solar_cap, bar_width, bottom=bottom, label='Solar', color=COLORS['solar'])
            bottom += solar_cap
        if any(v > 0 for v in wind_cap):
            ax.bar(idx, wind_cap, bar_width, bottom=bottom, label='Wind', color=COLORS['wind'])
            bottom += wind_cap
        if any(v > 0 for v in battery_power):
            ax.bar(idx, battery_power, bar_width, bottom=bottom, label='Battery', color=COLORS['battery'])
            bottom += battery_power
        if any(v > 0 for v in hydro_cap):
            ax.bar(idx, hydro_cap, bar_width, bottom=bottom, label='Hydro', color=COLORS['hydro_reservoir'])
        
        ax.set_ylabel('Capacity (MW)')
        ax.set_title(f'{label}\n{n_zones} zones, {raw["graph"]["regions"]} regions', fontweight='bold')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
        if col == 0:
            ax.legend(loc='upper right', ncol=2, fontsize=7)
    
    # === Row 2: Demand profiles comparison ===
    for col, (data, raw, label) in enumerate([
        (simple_data, simple_raw, 'Simple (#3691)'),
        (critical_data, critical_raw, 'Critical (#3544)')
    ]):
        ax = axes[1, col]
        zones = data.zones
        periods = data.periods
        time_hours = [t * data.dt_hours for t in periods]
        
        # Plot individual zone demands with transparency
        cmap = plt.cm.get_cmap('tab20', min(20, len(zones)))
        for i, zone in enumerate(zones):
            demand = [data.demand.get((zone, t), 0.0) for t in periods]
            ax.plot(time_hours, demand, color=cmap(i % 20), alpha=0.4, linewidth=0.5)
        
        # Total demand
        total_demand = [sum(data.demand.get((z, t), 0.0) for z in zones) for t in periods]
        ax.plot(time_hours, total_demand, color='black', linewidth=2, label='Total')
        
        # Metadata annotation
        weather = raw['meta'].get('weather_profile', 'N/A')
        demand_profile = raw['difficulty_indicators'].get('demand_profile', 'N/A')
        scale = raw['meta'].get('demand_scale_factor', 1.0)
        
        ax.set_ylabel('Demand (MW)')
        ax.set_xlabel('Time (hours)')
        ax.set_title(f'Weather: {weather}, Demand: {demand_profile}\nScale: {scale:.2f}', fontsize=9)
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    
    fig.suptitle('Scenario Comparison: Simple vs Critical', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_asset_portfolio_comparison(
    simple_path: Path, 
    critical_path: Path, 
    out_path: Path
) -> None:
    """
    Create a bar chart comparing aggregate asset portfolios.
    """
    simple_raw = load_scenario_json(simple_path)
    critical_raw = load_scenario_json(critical_path)
    
    categories = ['Thermal', 'Nuclear', 'Solar', 'Wind', 'Battery', 'DR', 'Hydro\nReservoir', 'Hydro\nRoR', 'Pumped']
    
    simple_assets = simple_raw['meta']['assets']
    critical_assets = critical_raw['meta']['assets']
    
    simple_vals = [
        simple_assets.get('thermal', 0),
        simple_assets.get('nuclear', 0),
        simple_assets.get('solar', 0),
        simple_assets.get('wind', 0),
        simple_assets.get('battery', 0),
        simple_assets.get('dr', 0),
        simple_assets.get('hydro_reservoir', 0),
        simple_assets.get('hydro_ror', 0),
        simple_assets.get('hydro_pumped', 0),
    ]
    
    critical_vals = [
        critical_assets.get('thermal', 0),
        critical_assets.get('nuclear', 0),
        critical_assets.get('solar', 0),
        critical_assets.get('wind', 0),
        critical_assets.get('battery', 0),
        critical_assets.get('dr', 0),
        critical_assets.get('hydro_reservoir', 0),
        critical_assets.get('hydro_ror', 0),
        critical_assets.get('hydro_pumped', 0),
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars1 = ax.bar(x - width/2, simple_vals, width, label='Simple (#3691)', color='#66c2a5')
    bars2 = ax.bar(x + width/2, critical_vals, width, label='Critical (#3544)', color='#fc8d62')
    
    ax.set_ylabel('Number of Units')
    ax.set_title('Asset Portfolio Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_network_structure(scenario_path: Path, out_path: Path, title: str) -> None:
    """
    Visualize the hierarchical network structure (regions and zones).
    """
    raw = load_scenario_json(scenario_path)
    data = load_scenario_data(scenario_path)
    
    zones_per_region = raw['graph']['zones_per_region']
    n_regions = len(zones_per_region)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a simple hierarchical layout
    # System at top, regions in middle, zones at bottom
    
    # Region positions
    region_y = 0.6
    region_spacing = 1.0 / (n_regions + 1)
    region_positions = {}
    
    for r_idx, n_zones in enumerate(zones_per_region):
        r_x = (r_idx + 1) * region_spacing
        region_name = f'R{r_idx + 1}'
        region_positions[region_name] = (r_x, region_y)
        
        # Draw region box
        rect = mpatches.FancyBboxPatch(
            (r_x - 0.03, region_y - 0.03), 0.06, 0.06,
            boxstyle="round,pad=0.01",
            facecolor='#a6cee3',
            edgecolor='#1f78b4',
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(r_x, region_y, f'{n_zones}', ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw connection to system
        ax.plot([0.5, r_x], [0.85, region_y + 0.03], 'k-', linewidth=0.5, alpha=0.5)
    
    # System node
    system_circle = plt.Circle((0.5, 0.88), 0.04, color='#33a02c', ec='#006d2c', linewidth=2)
    ax.add_patch(system_circle)
    ax.text(0.5, 0.88, 'S', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Legend / info box
    info_text = (
        f"Regions: {n_regions}\n"
        f"Total Zones: {sum(zones_per_region)}\n"
        f"Intertie Density: {raw['graph']['intertie_density']:.2f}\n"
        f"Neighbor Nations: {raw['graph']['neighbor_nations']}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontweight='bold', fontsize=11)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#33a02c', edgecolor='#006d2c', label='System'),
        mpatches.Patch(facecolor='#a6cee3', edgecolor='#1f78b4', label='Region (# = zones)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_paper_figure(
    simple_path: Path,
    critical_path: Path,
    out_path: Path
) -> None:
    """
    Create clean 2-panel comparison figure for paper:
    - Panel (a): Installed capacities (3 colors: Thermal/VRE/Storage), zones sorted
    - Panel (b): Demand variability (median + IQR band)
    """
    simple_data = load_scenario_data(simple_path)
    critical_data = load_scenario_data(critical_path)
    
    # Colors for 3 categories
    COLOR_THERMAL = '#d95f02'  # Orange
    COLOR_VRE = '#1b9e77'      # Teal
    COLOR_STORAGE = '#7570b3'  # Purple
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    
    # ========== Panel (a) & (b): Installed Capacities ==========
    # First pass: compute max Y for both panels
    max_capacity = 0
    capacity_data = []
    
    for data in [simple_data, critical_data]:
        zones = data.zones
        thermal = []
        vre = []
        storage = []
        
        for z in zones:
            th = data.thermal_capacity.get(z, 0.0) + data.nuclear_capacity.get(z, 0.0)
            thermal.append(th)
            v = data.solar_capacity.get(z, 0.0) + data.wind_capacity.get(z, 0.0)
            vre.append(v)
            st = (data.battery_power.get(z, 0.0) + 
                  data.hydro_res_capacity.get(z, 0.0) + 
                  data.pumped_power.get(z, 0.0))
            storage.append(st)
        
        total = [thermal[i] + vre[i] + storage[i] for i in range(len(zones))]
        sorted_idx = np.argsort(total)[::-1]
        
        thermal = np.array(thermal)[sorted_idx]
        vre = np.array(vre)[sorted_idx]
        storage = np.array(storage)[sorted_idx]
        
        max_capacity = max(max_capacity, np.max(thermal + vre + storage))
        capacity_data.append((zones, thermal, vre, storage))
    
    # Second pass: plot with shared Y scale
    for col, (label, (zones, thermal, vre, storage)) in enumerate([
        ('Simple', capacity_data[0]),
        ('Critical', capacity_data[1])
    ]):
        ax = axes[0, col]
        n_zones = len(zones)
        x = np.arange(n_zones)
        bar_width = 0.85
        
        ax.bar(x, thermal, bar_width, label='Thermal', color=COLOR_THERMAL)
        ax.bar(x, vre, bar_width, bottom=thermal, label='VRE', color=COLOR_VRE)
        ax.bar(x, storage, bar_width, bottom=thermal + vre, label='Storage', color=COLOR_STORAGE)
        
        ax.set_ylabel('Capacity (MW)')
        ax.set_title(f'({chr(97 + col)}) {label} — Installed Capacity', fontweight='bold')
        ax.set_xticks([])
        ax.set_xlabel(f'Zones (n={n_zones})')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(-0.5, n_zones - 0.5)
        ax.set_ylim(0, max_capacity * 1.05)
        
        if col == 0:
            ax.legend(loc='upper right', framealpha=0.9)
    
    # ========== Panel (c) & (d): Demand Variability ==========
    # First pass: compute Y range for both panels
    demand_min, demand_max = float('inf'), 0
    demand_data = []
    
    for data in [simple_data, critical_data]:
        zones = data.zones
        periods = data.periods
        time_hours = np.array([t * data.dt_hours for t in periods])
        
        demand_matrix = np.array([
            [data.demand.get((z, t), 0.0) for t in periods]
            for z in zones
        ])
        
        median = np.median(demand_matrix, axis=0)
        q25 = np.percentile(demand_matrix, 25, axis=0)
        q75 = np.percentile(demand_matrix, 75, axis=0)
        
        demand_min = min(demand_min, np.min(q25))
        demand_max = max(demand_max, np.max(q75))
        demand_data.append((time_hours, median, q25, q75))
    
    # Second pass: plot with shared Y scale
    for col, (label, (time_hours, median, q25, q75)) in enumerate([
        ('Simple', demand_data[0]),
        ('Critical', demand_data[1])
    ]):
        ax = axes[1, col]
        
        ax.fill_between(time_hours, q25, q75, alpha=0.3, color='#2c7fb8', label='IQR')
        ax.plot(time_hours, median, color='#2c7fb8', linewidth=2, label='Median')
        
        ax.set_ylabel('Demand (MW)')
        ax.set_xlabel('Time (h)')
        ax.set_title(f'({chr(99 + col)}) {label} — Zonal Demand Variability', fontweight='bold')
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(demand_min * 0.9, demand_max * 1.05)
        ax.grid(alpha=0.3, linestyle='--')
        
        if col == 0:
            ax.legend(loc='upper right', framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def combine_dispatch_figures(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    left_label: str = "Simple",
    right_label: str = "Critical"
) -> None:
    """
    Combine two dispatch PNG images into a single two-panel figure.
    """
    from PIL import Image
    
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)
    
    # Get dimensions
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    
    # Scale factor for larger output
    scale = 1.5
    
    # Create combined image with labels on top
    label_height = 80
    gap = 40
    combined_width = w1 + w2 + gap
    combined_height = max(h1, h2) + label_height
    
    combined = Image.new('RGB', (combined_width, combined_height), 'white')
    
    # Paste images
    combined.paste(left_img, (0, label_height))
    combined.paste(right_img, (w1 + gap, label_height))
    
    # Add labels using matplotlib with larger figure
    fig, ax = plt.subplots(figsize=(combined_width * scale / 100, combined_height * scale / 100), dpi=150)
    ax.imshow(combined)
    ax.axis('off')
    
    # Add text labels (larger font)
    ax.text(w1 / 2, label_height / 2, f'(a) {left_label}', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    ax.text(w1 + gap + w2 / 2, label_height / 2, f'(b) {right_label}', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    """Generate all paper visualizations."""
    base_dir = Path(__file__).resolve().parent.parent
    scenarios_dir = base_dir / "outputs" / "scenarios_v3"
    output_dir = base_dir / "paper_viz"
    output_dir.mkdir(exist_ok=True)
    
    simple_path = scenarios_dir / "scenario_03691.json"
    critical_path = scenarios_dir / "scenario_03544.json"
    
    # Verify files exist
    if not simple_path.exists():
        print(f"Error: {simple_path} not found")
        return
    if not critical_path.exists():
        print(f"Error: {critical_path} not found")
        return
    
    print(f"Generating paper visualizations in {output_dir}")
    print("-" * 50)
    
    # Main paper figure: clean 2-panel comparison
    plot_paper_figure(
        simple_path,
        critical_path,
        output_dir / "scenario_paper_figure.png"
    )
    
    # Combined dispatch figure
    dispatch_simple = scenarios_dir / "plots" / "scenario_03691_dispatch.png"
    dispatch_critical = scenarios_dir / "plots" / "scenario_03544_dispatch.png"
    if dispatch_simple.exists() and dispatch_critical.exists():
        combine_dispatch_figures(
            dispatch_simple,
            dispatch_critical,
            output_dir / "dispatch_comparison.png"
        )
    
    # Additional figures (optional)
    plot_scenario_overview(
        simple_path, 
        output_dir / "scenario_simple_overview.png",
        title_suffix=" — Simple Scenario (#3691)"
    )
    
    plot_scenario_overview(
        critical_path, 
        output_dir / "scenario_critical_overview.png",
        title_suffix=" — Critical Scenario (#3544)"
    )
    
    plot_scenario_comparison(
        simple_path, 
        critical_path, 
        output_dir / "scenario_comparison.png"
    )
    
    plot_asset_portfolio_comparison(
        simple_path, 
        critical_path, 
        output_dir / "asset_portfolio_comparison.png"
    )
    
    plot_network_structure(
        simple_path,
        output_dir / "network_simple.png",
        title="Network Structure — Simple Scenario (#3691)"
    )
    
    plot_network_structure(
        critical_path,
        output_dir / "network_critical.png",
        title="Network Structure — Critical Scenario (#3544)"
    )
    
    print("-" * 50)
    print("Done! All visualizations saved to paper_viz/")


if __name__ == "__main__":
    main()
