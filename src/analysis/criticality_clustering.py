"""
HDBSCAN-based clustering for scenario criticality classification.

Uses the 22 normalized criticality metrics to identify natural clusters
of scenarios based on their stress and hardness profiles.
"""
from __future__ import annotations

import json
import argparse
import pathlib
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import numpy as np

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_criticality_results(results_path: str) -> Dict[str, Any]:
    """Load criticality results from JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_feature_matrix(
    results: List[Dict[str, Any]],
    use_normalized: bool = True,
    include_indices: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract feature matrix from criticality results.
    
    Args:
        results: List of scenario criticality results
        use_normalized: Use normalized metrics (recommended)
        include_indices: Include stress/hardness/criticality indices as features
        
    Returns:
        Feature matrix (n_scenarios x n_features), feature names, scenario files
    """
    features = []
    feature_names = []
    scenario_files = []
    
    for i, r in enumerate(results):
        row = []
        
        # Add indices if requested
        if include_indices:
            row.extend([
                r["stress_index"],
                r["hardness_index"],
                r["criticality_index"],
            ])
            if i == 0:
                feature_names.extend(["stress_index", "hardness_index", "criticality_index"])
        
        # Add individual metrics
        metric_key = "stress_normalized" if use_normalized else "stress_metrics"
        for key, value in r[metric_key].items():
            row.append(value)
            if i == 0:
                feature_names.append(key)
        
        metric_key = "hardness_normalized" if use_normalized else "hardness_metrics"
        for key, value in r[metric_key].items():
            row.append(value)
            if i == 0:
                feature_names.append(key)
        
        features.append(row)
        scenario_files.append(r["file"])
    
    return np.array(features), feature_names, scenario_files


def cluster_with_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Cluster scenarios using HDBSCAN.
    
    Args:
        X: Feature matrix (n_scenarios x n_features)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        cluster_selection_epsilon: Epsilon for cluster selection
        metric: Distance metric
        
    Returns:
        Cluster labels, probabilities, clustering info
    """
    if not HAS_HDBSCAN:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(X_scaled)
    probabilities = clusterer.probabilities_
    
    # Compute cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    
    # Silhouette score (excluding noise)
    mask = labels != -1
    if n_clusters > 1 and mask.sum() > n_clusters:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))
    else:
        sil_score = None
    
    info = {
        "algorithm": "HDBSCAN",
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_ratio": round(float(n_noise / len(labels)), 4),
        "silhouette_score": round(sil_score, 4) if sil_score else None,
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "metric": metric,
        },
    }
    
    return labels, probabilities, info


def cluster_with_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster scenarios using DBSCAN.
    
    Args:
        X: Feature matrix (n_scenarios x n_features)
        eps: Maximum distance between samples in a neighborhood
        min_samples: Minimum samples for core points
        metric: Distance metric
        
    Returns:
        Cluster labels, clustering info
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit DBSCAN
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(X_scaled)
    
    # Compute cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    
    # Silhouette score (excluding noise)
    mask = labels != -1
    if n_clusters > 1 and mask.sum() > n_clusters:
        sil_score = float(silhouette_score(X_scaled[mask], labels[mask]))
    else:
        sil_score = None
    
    info = {
        "algorithm": "DBSCAN",
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_ratio": round(float(n_noise / len(labels)), 4),
        "silhouette_score": round(sil_score, 4) if sil_score else None,
        "params": {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
        },
    }
    
    return labels, info


def compute_cluster_profiles(
    results: List[Dict[str, Any]],
    labels: np.ndarray,
    feature_names: List[str],
    X: np.ndarray,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute profile statistics for each cluster.
    
    Returns dict mapping cluster_id to profile with:
    - count, mean criticality, feature means, representative scenarios
    """
    profiles = {}
    
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_results = [results[i] for i in cluster_indices]
        cluster_features = X[mask]
        
        # Criticality stats
        crit_vals = [r["criticality_index"] for r in cluster_results]
        stress_vals = [r["stress_index"] for r in cluster_results]
        hardness_vals = [r["hardness_index"] for r in cluster_results]
        
        # Feature means
        feature_means = {
            name: round(float(cluster_features[:, i].mean()), 4)
            for i, name in enumerate(feature_names)
        }
        
        # Assign cluster name based on criticality level
        mean_crit = float(np.mean(crit_vals))
        if label == -1:
            cluster_name = "noise"
        elif mean_crit < 0.25:
            cluster_name = "low"
        elif mean_crit < 0.50:
            cluster_name = "medium"
        elif mean_crit < 0.75:
            cluster_name = "high"
        else:
            cluster_name = "critical"
        
        # Top representatives (highest membership probability if available)
        representative_files = [r["file"] for r in cluster_results[:5]]
        
        profiles[int(label)] = {
            "cluster_name": cluster_name,
            "count": int(mask.sum()),
            "criticality": {
                "mean": round(mean_crit, 4),
                "std": round(float(np.std(crit_vals)), 4),
                "min": round(float(np.min(crit_vals)), 4),
                "max": round(float(np.max(crit_vals)), 4),
            },
            "stress_mean": round(float(np.mean(stress_vals)), 4),
            "hardness_mean": round(float(np.mean(hardness_vals)), 4),
            "feature_means": feature_means,
            "representative_scenarios": representative_files,
        }
    
    return profiles


def run_clustering(
    results_path: str,
    output_path: Optional[str] = None,
    algorithm: str = "hdbscan",
    min_cluster_size: int = 50,
    min_samples: int = 10,
    eps: float = 0.5,
    cluster_selection_epsilon: float = 0.0,
    dim_reduction: str = "umap",
    n_components: int = 10,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
) -> Dict[str, Any]:
    """
    Run clustering on criticality results.
    
    Args:
        results_path: Path to criticality_results.json
        output_path: Path to save clustering results
        algorithm: "hdbscan" or "dbscan"
        min_cluster_size: Min cluster size for HDBSCAN
        min_samples: Min samples for core points
        eps: Epsilon for DBSCAN
        cluster_selection_epsilon: HDBSCAN epsilon for cluster merging (higher = less noise)
        dim_reduction: "umap", "pca", or "none"
        n_components: Number of components for dimensionality reduction
        umap_n_neighbors: UMAP n_neighbors (controls local vs global structure)
        umap_min_dist: UMAP min_dist (0.0 recommended for clustering)
        
    Returns:
        Clustering results with labels and profiles
    """
    print(f"Loading criticality results from {results_path}...")
    data = load_criticality_results(results_path)
    results = data["all_results"]
    
    print(f"Extracting features from {len(results)} scenarios...")
    X, feature_names, scenario_files = extract_feature_matrix(results)
    print(f"Feature matrix: {X.shape[0]} scenarios x {X.shape[1]} features")
    
    # Standardize features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    dim_info = {"method": dim_reduction}
    if dim_reduction == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")
        print(f"Applying UMAP (n_components={n_components}, n_neighbors={umap_n_neighbors})...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=42,
        )
        X_cluster = reducer.fit_transform(X_scaled)
        dim_info["n_components"] = n_components
        dim_info["n_neighbors"] = umap_n_neighbors
        dim_info["min_dist"] = umap_min_dist
        print(f"  Reduced to {X_cluster.shape[1]} dimensions")
    elif dim_reduction == "pca":
        print(f"Applying PCA (n_components={n_components})...")
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_cluster = pca.fit_transform(X_scaled)
        explained_var = sum(pca.explained_variance_ratio_)
        dim_info["n_components"] = n_components
        dim_info["explained_variance"] = round(explained_var, 4)
        print(f"  Explained variance: {explained_var:.2%}")
    else:
        print("No dimensionality reduction (using scaled features)...")
        X_cluster = X_scaled
    
    # Run clustering
    print(f"Running {algorithm.upper()} clustering...")
    if algorithm.lower() == "hdbscan":
        labels, probabilities, cluster_info = cluster_with_hdbscan(
            X_cluster,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
    else:
        labels, cluster_info = cluster_with_dbscan(
            X_cluster,
            eps=eps,
            min_samples=min_samples,
        )
        probabilities = np.ones(len(labels))
    
    print(f"Found {cluster_info['n_clusters']} clusters ({cluster_info['n_noise']} noise points)")
    
    # Compute cluster profiles
    print("Computing cluster profiles...")
    profiles = compute_cluster_profiles(results, labels, feature_names, X)
    
    # Add dim reduction info to cluster_info
    cluster_info["dim_reduction"] = dim_info
    
    # Build output (ensure all values are JSON-serializable Python types)
    scenario_assignments = []
    for i in range(len(labels)):
        scenario_assignments.append({
            "file": scenario_files[i],
            "cluster": int(labels[i]),
            "cluster_name": profiles[int(labels[i])]["cluster_name"],
            "probability": round(float(probabilities[i]), 4),
            "criticality_index": float(results[i]["criticality_index"]),
        })
    
    output = {
        "clustering_info": cluster_info,
        "cluster_profiles": profiles,
        "cluster_distribution": {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
        "scenario_assignments": scenario_assignments,
    }
    
    # Save results
    if output_path:
        out_file = pathlib.Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"Algorithm: {cluster_info['algorithm']}")
    print(f"Clusters found: {cluster_info['n_clusters']}")
    print(f"Noise points: {cluster_info['n_noise']} ({cluster_info['noise_ratio']:.1%})")
    if cluster_info['silhouette_score']:
        print(f"Silhouette score: {cluster_info['silhouette_score']:.4f}")
    print()
    print("Cluster Breakdown:")
    for cid, profile in sorted(profiles.items()):
        name = profile["cluster_name"].upper()
        count = profile["count"]
        crit = profile["criticality"]["mean"]
        print(f"  Cluster {cid:2d} ({name:8s}): {count:5d} scenarios, mean criticality={crit:.4f}")
    print("=" * 60)
    
    return output


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster scenarios by criticality using HDBSCAN/DBSCAN"
    )
    parser.add_argument(
        "results_path",
        help="Path to criticality_results.json"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for clustering results (default: <dir>/clustering_results.json)"
    )
    parser.add_argument(
        "-a", "--algorithm",
        choices=["hdbscan", "dbscan"],
        default="hdbscan",
        help="Clustering algorithm (default: hdbscan)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Minimum cluster size for HDBSCAN (default: 50)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples for core points (default: 10)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="Epsilon for DBSCAN (default: 0.5)"
    )
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon - higher values merge more points into clusters, reducing noise (default: 0.0)"
    )
    parser.add_argument(
        "--dim-reduction",
        choices=["umap", "pca", "none"],
        default="umap",
        help="Dimensionality reduction method (default: umap)"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of components for dim reduction (default: 10)"
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter (default: 30)"
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist parameter, 0.0 best for clustering (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        results_dir = pathlib.Path(args.results_path).parent
        output_path = str(results_dir / "clustering_results.json")
    
    run_clustering(
        results_path=args.results_path,
        output_path=output_path,
        algorithm=args.algorithm,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        eps=args.eps,
        cluster_selection_epsilon=args.cluster_epsilon,
        dim_reduction=args.dim_reduction,
        n_components=args.n_components,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
    )


if __name__ == "__main__":
    main()
