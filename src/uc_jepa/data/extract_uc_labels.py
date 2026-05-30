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
    """Binary import_mode[z, t].

    The MILP report stores ``import_mode`` and ``net_import`` as a *system-level*
    time series at ``detail.<key>.values`` (length T), not per zone. We
    broadcast the system-level binary to every zone:
        out[z, t] = 1 if (import_mode.values[t] >= 0.5) else 0
    Falls back to ``net_import.values[t] > 0`` if ``import_mode`` is missing.
    Per-zone storage is also supported (in case future reports add it).
    """
    detail = report.get("detail", {}) or report.get("pre_dispatch", {})
    Z = len(zone_names)
    out = torch.zeros(Z, n_timesteps)
    if not isinstance(detail, dict):
        return out

    im = detail.get("import_mode", None)
    ni = detail.get("net_import", None)

    def _series_from(block, threshold_fn) -> Optional[List[float]]:
        if not isinstance(block, dict):
            return None
        if "values" in block and isinstance(block["values"], list):
            return [1.0 if threshold_fn(v) else 0.0 for v in block["values"]]
        return None

    # 1) Prefer the dedicated import_mode binary (system-level).
    sys_series = _series_from(im, lambda v: float(v) >= 0.5)
    if sys_series is None:
        # 2) Fallback: derive from net_import sign.
        sys_series = _series_from(ni, lambda v: float(v) > 1e-3)

    if sys_series is not None:
        for t in range(min(n_timesteps, len(sys_series))):
            out[:, t] = float(sys_series[t])
        return out

    # 3) Per-zone storage (legacy / future-proof).
    src = im if isinstance(im, dict) and any(z in im for z in zone_names) else ni
    if isinstance(src, dict):
        for z_idx, zone in enumerate(zone_names):
            series = src.get(zone, [])
            if not isinstance(series, list):
                continue
            for t in range(min(n_timesteps, len(series))):
                try:
                    val = float(series[t])
                except (TypeError, ValueError):
                    val = 0.0
                if src is im:
                    out[z_idx, t] = 1.0 if val >= 0.5 else 0.0
                else:
                    out[z_idx, t] = 1.0 if val > 1e-3 else 0.0
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


# Directory names that should NOT be used as a family label. ``gold`` and
# ``silver`` are intentionally kept as legitimate families because they
# encode the MILP-tier classification used by the EBM training pipeline.
_GENERIC_DIRS = {"reports", "report", "json", ""}


def _family_from_path(report_path: str) -> str:
    """Resolve the family tag from a report path.

    Walks up only over generic buckets like ``reports``. Concrete tiers such
    as ``gold``/``silver`` are preserved as their own family.
    """
    p = Path(report_path).parent
    while p.name and p.name in _GENERIC_DIRS:
        p = p.parent
    return p.name or "unknown"


def _load_criticality(report_path: str, report: dict) -> float:
    """Resolve scenario criticality_index.

    Priority:
        1. Field directly in the report.
        2. Sibling scenario input file at ``<family_dir>/<scenario_id>.json``
           (the input definition where ``criticality_index`` is stored).
    """
    if "criticality_index" in report:
        try:
            return float(report["criticality_index"])
        except (TypeError, ValueError):
            pass
    # Walk up only over generic buckets to find the family root, but ALSO
    # try one extra level up because some layouts nest reports under
    # ``family/reports/<tier>/scenario.json`` and the input scenario lives at
    # ``family/scenario.json`` (i.e. above the tier directory).
    family_dir = Path(report_path).parent
    while family_dir.name in _GENERIC_DIRS:
        family_dir = family_dir.parent
    sid = extract_scenario_id(Path(report_path).stem) or Path(report_path).stem
    candidates = [family_dir / f"{sid}.json", family_dir.parent / f"{sid}.json"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r") as f:
                inp = json.load(f)
            if "criticality_index" in inp:
                return float(inp["criticality_index"])
        except Exception:  # noqa: BLE001
            continue
    return float("nan")


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
    criticality = _load_criticality(report_path, report)
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
        # Resolve family by walking up generic buckets like ``reports``.
        # Tiers like ``gold``/``silver`` are kept as families.
        fam_dir = Path(rdir).resolve()
        while fam_dir.name in _GENERIC_DIRS and fam_dir.parent != fam_dir:
            fam_dir = fam_dir.parent
        family = fam_dir.name or Path(rdir).name or "unknown"
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
