"""
Visualization script for criticality clustering results.

Generates plots showing:
1. UMAP 2D projection colored by cluster
2. Criticality index distribution by cluster
3. Stress vs Hardness scatter colored by cluster
4. Box plots of metrics by cluster
"""
from __future__ import annotations

import json
import argparse
import pathlib
from typing import Dict, Any, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Color palette for clusters (supports 20+ clusters)
CLUSTER_COLORS = {
    -1: "#CCCCCC",      # Noise - light gray
    0: "#1f77b4",       # Blue
    1: "#ff7f0e",       # Orange
    2: "#2ca02c",       # Green
    3: "#d62728",       # Red
    4: "#9467bd",       # Purple
    5: "#8c564b",       # Brown
    6: "#e377c2",       # Pink
    7: "#7f7f7f",       # Gray
    8: "#bcbd22",       # Olive
    9: "#17becf",       # Cyan
    10: "#aec7e8",      # Light blue
    11: "#ffbb78",      # Light orange
    12: "#98df8a",      # Light green
    13: "#ff9896",      # Light red
    14: "#c5b0d5",      # Light purple
    15: "#c49c94",      # Light brown
    16: "#f7b6d2",      # Light pink
    17: "#c7c7c7",      # Medium gray
    18: "#dbdb8d",      # Light olive
    19: "#9edae5",      # Light cyan
    20: "#393b79",      # Dark blue
    21: "#637939",      # Dark olive
    22: "#8c6d31",      # Dark gold
    23: "#843c39",      # Dark red
    24: "#7b4173",      # Dark magenta
}

CRITICALITY_COLORS = {
    "noise": "#CCCCCC",
    "low": "#2ecc71",       # Green
    "medium": "#f39c12",    # Yellow/Orange
    "high": "#e74c3c",      # Red
    "critical": "#8e44ad",  # Purple
}


def load_data(
    criticality_path: str,
    clustering_path: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load criticality and clustering results."""
    with open(criticality_path, "r", encoding="utf-8") as f:
        criticality_data = json.load(f)
    with open(clustering_path, "r", encoding="utf-8") as f:
        clustering_data = json.load(f)
    return criticality_data, clustering_data


def extract_features_for_umap(results: List[Dict[str, Any]]) -> np.ndarray:
    """Extract normalized features for UMAP projection."""
    features = []
    for r in results:
        row = [r["stress_index"], r["hardness_index"], r["criticality_index"]]
        for key, value in r["stress_normalized"].items():
            row.append(value)
        for key, value in r["hardness_normalized"].items():
            row.append(value)
        features.append(row)
    return np.array(features)


def compute_umap_2d(X: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    """Compute 2D UMAP embedding for visualization."""
    if not HAS_UMAP:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(X_scaled)


def plot_umap_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    ax: plt.Axes,
    title: str = "UMAP Projection by Cluster",
) -> None:
    """Plot UMAP 2D embedding colored by cluster."""
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        mask = labels == label
        color = CLUSTER_COLORS.get(label, "#333333")
        name = cluster_names.get(label, f"Cluster {label}")
        alpha = 0.3 if label == -1 else 0.6
        size = 5 if label == -1 else 15
        
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=f"{name} ({mask.sum()})",
            alpha=alpha,
            s=size,
            edgecolors="none",
        )
    
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, markerscale=2)


def plot_umap_criticality(
    embedding: np.ndarray,
    criticality: np.ndarray,
    ax: plt.Axes,
    title: str = "UMAP Projection by Criticality",
) -> None:
    """Plot UMAP 2D embedding colored by criticality index."""
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=criticality,
        cmap="RdYlGn_r",
        alpha=0.6,
        s=10,
        edgecolors="none",
    )
    plt.colorbar(scatter, ax=ax, label="Criticality Index")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)


def plot_stress_hardness_scatter(
    stress: np.ndarray,
    hardness: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    ax: plt.Axes,
) -> None:
    """Scatter plot of stress vs hardness colored by cluster."""
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        mask = labels == label
        color = CLUSTER_COLORS.get(label, "#333333")
        name = cluster_names.get(label, f"Cluster {label}")
        alpha = 0.2 if label == -1 else 0.5
        size = 5 if label == -1 else 20
        
        ax.scatter(
            stress[mask],
            hardness[mask],
            c=color,
            label=name,
            alpha=alpha,
            s=size,
            edgecolors="none",
        )
    
    ax.set_xlabel("Stress Index")
    ax.set_ylabel("Hardness Index")
    ax.set_title("Stress vs Hardness by Cluster")
    ax.legend(loc="upper right", fontsize=8)


def plot_criticality_distribution(
    criticality: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    ax: plt.Axes,
) -> None:
    """Histogram/KDE of criticality index by cluster."""
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    for label in unique_labels:
        mask = labels == label
        color = CLUSTER_COLORS.get(label, "#333333")
        name = cluster_names.get(label, f"Cluster {label}")
        
        if HAS_SEABORN:
            sns.kdeplot(
                criticality[mask],
                ax=ax,
                color=color,
                label=name,
                fill=True,
                alpha=0.3,
            )
        else:
            ax.hist(
                criticality[mask],
                bins=30,
                color=color,
                label=name,
                alpha=0.5,
                density=True,
            )
    
    # Add noise separately with lower prominence
    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        ax.hist(
            criticality[noise_mask],
            bins=30,
            color=CLUSTER_COLORS[-1],
            label="Noise",
            alpha=0.2,
            density=True,
            histtype="step",
            linewidth=2,
        )
    
    ax.set_xlabel("Criticality Index")
    ax.set_ylabel("Density")
    ax.set_title("Criticality Distribution by Cluster")
    ax.legend(loc="upper right", fontsize=8)
    ax.axvline(x=0.25, color="gray", linestyle="--", alpha=0.5, label="_Low/Med")
    ax.axvline(x=0.50, color="gray", linestyle="--", alpha=0.5, label="_Med/High")
    ax.axvline(x=0.75, color="gray", linestyle="--", alpha=0.5, label="_High/Crit")


def plot_cluster_boxplots(
    criticality: np.ndarray,
    stress: np.ndarray,
    hardness: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    ax: plt.Axes,
) -> None:
    """Box plots of criticality, stress, hardness by cluster."""
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    data_crit = [criticality[labels == l] for l in unique_labels]
    data_stress = [stress[labels == l] for l in unique_labels]
    data_hard = [hardness[labels == l] for l in unique_labels]
    
    positions_crit = np.arange(len(unique_labels)) * 4
    positions_stress = positions_crit + 1
    positions_hard = positions_crit + 2
    
    bp1 = ax.boxplot(data_crit, positions=positions_crit, widths=0.8, patch_artist=True)
    bp2 = ax.boxplot(data_stress, positions=positions_stress, widths=0.8, patch_artist=True)
    bp3 = ax.boxplot(data_hard, positions=positions_hard, widths=0.8, patch_artist=True)
    
    for patch in bp1["boxes"]:
        patch.set_facecolor("#3498db")
    for patch in bp2["boxes"]:
        patch.set_facecolor("#e74c3c")
    for patch in bp3["boxes"]:
        patch.set_facecolor("#2ecc71")
    
    ax.set_xticks(positions_crit + 1)
    ax.set_xticklabels([cluster_names.get(l, f"C{l}") for l in unique_labels])
    ax.set_ylabel("Index Value")
    ax.set_title("Indices by Cluster")
    
    legend_patches = [
        mpatches.Patch(color="#3498db", label="Criticality"),
        mpatches.Patch(color="#e74c3c", label="Stress"),
        mpatches.Patch(color="#2ecc71", label="Hardness"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)


def plot_cluster_summary(
    clustering_data: Dict[str, Any],
    ax: plt.Axes,
) -> None:
    """Bar chart of cluster sizes."""
    profiles = clustering_data["cluster_profiles"]
    
    labels = []
    counts = []
    colors = []
    
    for cid, profile in sorted(profiles.items(), key=lambda x: int(x[0])):
        cid_int = int(cid)
        labels.append(f"{profile['cluster_name'].title()}\n(C{cid})")
        counts.append(profile["count"])
        colors.append(CRITICALITY_COLORS.get(profile["cluster_name"], CLUSTER_COLORS.get(cid_int, "#333")))
    
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    ax.set_ylabel("Number of Scenarios")
    ax.set_title("Cluster Sizes")
    ax.set_ylim(0, max(counts) * 1.15)


def create_visualization(
    criticality_path: str,
    clustering_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
    umap_neighbors: int = 30,
) -> None:
    """
    Create comprehensive visualization of clustering results.
    
    Args:
        criticality_path: Path to criticality_results.json
        clustering_path: Path to clustering_results.json
        output_path: Path to save figure (optional)
        show: Whether to display the figure
        umap_neighbors: n_neighbors for UMAP 2D projection
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")
    
    print("Loading data...")
    criticality_data, clustering_data = load_data(criticality_path, clustering_path)
    
    results = criticality_data["all_results"]
    assignments = clustering_data["scenario_assignments"]
    profiles = clustering_data["cluster_profiles"]
    
    # Build lookup for cluster labels
    file_to_cluster = {a["file"]: a["cluster"] for a in assignments}
    cluster_names = {int(cid): p["cluster_name"].title() for cid, p in profiles.items()}
    
    # Extract arrays
    criticality = np.array([r["criticality_index"] for r in results])
    stress = np.array([r["stress_index"] for r in results])
    hardness = np.array([r["hardness_index"] for r in results])
    labels = np.array([file_to_cluster.get(r["file"], -1) for r in results])
    
    # Compute UMAP 2D embedding
    print("Computing UMAP 2D projection...")
    X = extract_features_for_umap(results)
    embedding = compute_umap_2d(X, n_neighbors=umap_neighbors)
    
    # Create figure
    print("Creating plots...")
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Plot 1: UMAP by cluster
    plot_umap_clusters(embedding, labels, cluster_names, ax1)
    
    # Plot 2: UMAP by criticality
    plot_umap_criticality(embedding, criticality, ax2)
    
    # Plot 3: Cluster sizes
    plot_cluster_summary(clustering_data, ax3)
    
    # Plot 4: Stress vs Hardness
    plot_stress_hardness_scatter(stress, hardness, labels, cluster_names, ax4)
    
    # Plot 5: Criticality distribution
    plot_criticality_distribution(criticality, labels, cluster_names, ax5)
    
    # Plot 6: Box plots
    plot_cluster_boxplots(criticality, stress, hardness, labels, cluster_names, ax6)
    
    # Title and layout
    n_clusters = clustering_data["clustering_info"]["n_clusters"]
    n_noise = clustering_data["clustering_info"]["n_noise"]
    sil_score = clustering_data["clustering_info"].get("silhouette_score", "N/A")
    
    fig.suptitle(
        f"Criticality Clustering Analysis\n"
        f"{len(results)} scenarios | {n_clusters} clusters | {n_noise} noise | silhouette={sil_score}",
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save and/or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize criticality clustering results"
    )
    parser.add_argument(
        "criticality_path",
        help="Path to criticality_results.json"
    )
    parser.add_argument(
        "-c", "--clustering",
        default=None,
        help="Path to clustering_results.json (default: same directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for figure (e.g., clusters.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the figure (just save)"
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="n_neighbors for UMAP 2D projection (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Default clustering path
    clustering_path = args.clustering
    if clustering_path is None:
        crit_dir = pathlib.Path(args.criticality_path).parent
        clustering_path = str(crit_dir / "clustering_results.json")
    
    create_visualization(
        criticality_path=args.criticality_path,
        clustering_path=clustering_path,
        output_path=args.output,
        show=not args.no_show,
        umap_neighbors=args.umap_neighbors,
    )


if __name__ == "__main__":
    main()
