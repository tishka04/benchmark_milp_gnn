"""Materialize UC labels (u_star, h_scenario, criticality, family, objective) per scenario.

Phase 1 MVP. Reuses :func:`src.ebm.dataset_v3._extract_binaries_from_report`
to build the F=7 binary tensor and adds an extra `import_mode` channel
derived from `detail.net_import[zone][t] > 0`, yielding F=8.

Usage:
    python -m src.uc_jepa.data.extract_uc_labels \
        --reports outputs/high_criticality_scenarios outputs/medium_criticality_scenarios outputs/low_criticality_scenarios \
        --embeddings outputs/embeddings_v3 \
        --out outputs/uc_jepa/dataset \
        --n-timesteps 24

Output layout:
    {out}/{family}/scenario_XXXXX_uc.npz

Each .npz contains:
    u_star          : uint8  [Z, T, 8]
    h_scenario      : float32 [Z, T, D] (zeros if embedding missing)
    zone_names      : object array of length Z
    criticality     : float (scalar)
    milp_objective  : float (scalar)
    family          : str    (scalar)
    scenario_id     : str    (scalar)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Reuse the existing F=7 extractor and embedding lookup utilities.
from src.ebm.dataset_v3 import (
    _extract_binaries_from_report,
    extract_scenario_id,
)
from src.uc_jepa import (
    FEAT_IMPORT,
    N_UC_FEATURES,
    UC_FEATURE_NAMES,
)


def _zone_names(report: dict) -> List[str]:
    zones = report.get("zones")
    if zones:
        return list(zones)
    detail = report.get("detail", {}) or report.get("pre_dispatch", {})
    for probe in ("thermal_commitment", "thermal", "demand"):
        block = detail.get(probe, {}) if isinstance(detail, dict) else {}
        if isinstance(block, dict) and block:
            return sorted(block.keys())
    return []


def _import_mode(report: dict, zone_names: Sequence[str], n_timesteps: int) -> torch.Tensor:
    """Binary import_mode[z, t] = 1 if net_import[z, t] > 0 else 0."""
    detail = report.get("detail", {}) or report.get("pre_dispatch", {})
    net = detail.get("net_import", {}) if isinstance(detail, dict) else {}
    Z = len(zone_names)
    out = torch.zeros(Z, n_timesteps)
    if not isinstance(net, dict):
        return out
    for z_idx, zone in enumerate(zone_names):
        series = net.get(zone, []) if isinstance(net, dict) else []
        for t in range(min(n_timesteps, len(series))):
            try:
                out[z_idx, t] = 1.0 if float(series[t]) > 1e-3 else 0.0
            except (TypeError, ValueError):
                out[z_idx, t] = 0.0
    return out


def extract_u_star(report: dict, n_timesteps: int = 24) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """Build [Z, T, 8] UC tensor and return zone names. Returns None on failure."""
    zone_names = _zone_names(report)
    if not zone_names:
        return None
    base = _extract_binaries_from_report(report, n_timesteps=n_timesteps)
    if base is None:
        return None
    Z = base.shape[0]
    if Z != len(zone_names):
        # _extract_binaries_from_report sorts internally — re-derive consistent names.
        zone_names = sorted(zone_names)
    u = torch.zeros(Z, n_timesteps, N_UC_FEATURES, dtype=torch.float32)
    u[..., :base.shape[-1]] = base
    u[..., FEAT_IMPORT] = _import_mode(report, zone_names, n_timesteps)
    return u, zone_names


def _build_embedding_index(embeddings_dir: Optional[str]) -> Dict[str, str]:
    """Recursively map scenario_id -> embedding file path."""
    out: Dict[str, str] = {}
    if not embeddings_dir or not os.path.isdir(embeddings_dir):
        return out
    for root, _dirs, files in os.walk(embeddings_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".npz", ".pt", ".npy"):
                continue
            sid = extract_scenario_id(os.path.splitext(fname)[0])
            if sid is None:
                continue
            out[sid] = os.path.join(root, fname)
    return out


def _load_embedding(path: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            for key in ("zones", "zone_embeddings", "embeddings", "zone", "arr_0"):
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                        return arr.astype(np.float32, copy=False)
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    return arr.astype(np.float32, copy=False)
        elif ext == ".pt":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                return obj.float().cpu().numpy()
            if isinstance(obj, dict):
                for key in ("zones", "zone_embeddings", "embeddings"):
                    if key in obj and isinstance(obj[key], torch.Tensor):
                        return obj[key].float().cpu().numpy()
        elif ext == ".npy":
            return np.load(path).astype(np.float32, copy=False)
    except Exception as exc:  # noqa: BLE001 — log and skip a corrupt file
        print(f"  WARN: failed to load embedding {path}: {exc}")
    return None


def _family_from_path(report_path: str) -> str:
    """Use the immediate parent directory of the JSON as the family tag."""
    return Path(report_path).parent.name or "unknown"


def process_one(
    report_path: str,
    n_timesteps: int,
    embedding_lookup: Dict[str, str],
    embed_dim_hint: Optional[int],
) -> Optional[Dict[str, np.ndarray]]:
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: cannot read {report_path}: {exc}")
        return None

    parsed = extract_u_star(report, n_timesteps=n_timesteps)
    if parsed is None:
        return None
    u_star, zone_names = parsed
    Z = u_star.shape[0]

    sid = extract_scenario_id(Path(report_path).stem) or Path(report_path).stem
    family = _family_from_path(report_path)
    criticality = float(report.get("criticality_index", float("nan")))
    objective = float(report.get("mip", {}).get("objective", float("nan")))

    h_scenario: Optional[np.ndarray] = None
    emb_path = embedding_lookup.get(sid)
    if emb_path is not None:
        h_scenario = _load_embedding(emb_path)
    if h_scenario is None:
        D = embed_dim_hint or 128
        h_scenario = np.zeros((Z, n_timesteps, D), dtype=np.float32)
    else:
        # Align zone / timestep dims to u_star.
        h = h_scenario
        if h.ndim == 2:
            # [T, D] — broadcast across zones.
            h = np.broadcast_to(h[None, :, :], (Z, h.shape[0], h.shape[1])).copy()
        if h.shape[0] != Z:
            if h.shape[0] > Z:
                h = h[:Z]
            else:
                pad = np.zeros((Z - h.shape[0], h.shape[1], h.shape[2]), dtype=h.dtype)
                h = np.concatenate([h, pad], axis=0)
        if h.shape[1] != n_timesteps:
            if h.shape[1] > n_timesteps:
                h = h[:, :n_timesteps]
            else:
                pad = np.zeros((h.shape[0], n_timesteps - h.shape[1], h.shape[2]), dtype=h.dtype)
                h = np.concatenate([h, pad], axis=1)
        h_scenario = h.astype(np.float32, copy=False)

    return {
        "u_star": u_star.to(torch.uint8).cpu().numpy(),
        "h_scenario": h_scenario,
        "zone_names": np.array(zone_names, dtype=object),
        "criticality": np.float32(criticality),
        "milp_objective": np.float32(objective),
        "family": np.array(family),
        "scenario_id": np.array(sid),
        "feature_names": np.array(UC_FEATURE_NAMES, dtype=object),
    }


def run(
    report_dirs: Sequence[str],
    embeddings_dir: Optional[str],
    out_dir: str,
    n_timesteps: int = 24,
    max_per_family: Optional[int] = None,
    embed_dim_hint: Optional[int] = 128,
    overwrite: bool = False,
) -> int:
    embedding_lookup = _build_embedding_index(embeddings_dir)
    print(f"Embedding index: {len(embedding_lookup)} entries from {embeddings_dir!r}")

    n_total = 0
    for rdir in report_dirs:
        if not os.path.isdir(rdir):
            print(f"  SKIP: {rdir} (not a directory)")
            continue
        report_files = sorted(
            os.path.join(rdir, f) for f in os.listdir(rdir)
            if f.startswith("scenario_") and f.endswith(".json")
        )
        if max_per_family is not None:
            report_files = report_files[:max_per_family]
        family = Path(rdir).name
        family_out = Path(out_dir) / family
        family_out.mkdir(parents=True, exist_ok=True)
        print(f"[{family}] {len(report_files)} reports -> {family_out}")

        n_ok = 0
        for rpath in report_files:
            sid = extract_scenario_id(Path(rpath).stem) or Path(rpath).stem
            out_path = family_out / f"{sid}_uc.npz"
            if out_path.exists() and not overwrite:
                n_ok += 1
                continue
            payload = process_one(rpath, n_timesteps, embedding_lookup, embed_dim_hint)
            if payload is None:
                continue
            np.savez_compressed(out_path, **payload)
            n_ok += 1
        print(f"  -> {n_ok}/{len(report_files)} written")
        n_total += n_ok

    print(f"DONE. {n_total} UC label files written under {out_dir}")
    return n_total


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract UC labels for UC-JEPA Phase 1.")
    p.add_argument("--reports", nargs="+", required=True,
                   help="One or more directories containing scenario_*.json reports. "
                        "The directory name is used as the family tag.")
    p.add_argument("--embeddings", default=None,
                   help="Directory containing HTE zone embeddings (.npz/.pt/.npy). Optional.")
    p.add_argument("--out", required=True, help="Output directory for UC label .npz files.")
    p.add_argument("--n-timesteps", type=int, default=24)
    p.add_argument("--max-per-family", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=128,
                   help="Fallback embedding dimension when no h_scenario file is available.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    return run(
        report_dirs=args.reports,
        embeddings_dir=args.embeddings,
        out_dir=args.out,
        n_timesteps=args.n_timesteps,
        max_per_family=args.max_per_family,
        embed_dim_hint=args.embed_dim,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
