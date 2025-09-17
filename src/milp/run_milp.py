from __future__ import annotations


import argparse
import json
from pathlib import Path

from .solve import solve_scenario, SolveSummary


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summary_to_dict(summary: SolveSummary):
    return summary.as_dict()


def _serialize_report(report):
    serializable = {
        "scenario_id": report["scenario_id"],
        "mip": _summary_to_dict(report["mip"]),
        "lp": _summary_to_dict(report["lp"]),
        "cost_components": report["cost_components"],
        "lp_duals": {
            name: {f"{k[0]}|{k[1]}": float(v) for k, v in duals.items()}
            for name, duals in report["lp_duals"].items()
        },
        "periods": report["periods"],
        "zones": report["zones"],
    }
    if report["detail"] is not None:
        serializable["detail"] = report["detail"]
    return serializable


def main():
    parser = argparse.ArgumentParser(description="Solve a generated scenario with the MILP + LP relaxation pipeline.")
    parser.add_argument("scenario", type=Path, help="Path to scenario JSON (e.g., outputs/scenarios_v1/scenario_00001.json)")
    parser.add_argument("--solver", default="highs", help="Pyomo solver name (default: highs)")
    parser.add_argument("--tee", action="store_true", help="Stream solver output.")
    parser.add_argument("--top-duals", type=int, default=8, help="Number of largest duals to print from the LP relaxation.")
    parser.add_argument("--save-json", type=Path, help="Optional path to save the full report as JSON.")
    parser.add_argument("--plot", action="store_true", help="Generate scenario/result plots (requires matplotlib).")
    parser.add_argument("--plot-dir", type=Path, help="Directory for plot outputs (defaults to scenario folder).")
    parser.add_argument("--export-prefix", type=Path, help="Prefix path for CSV exports (produces _zone/_system files).")
    parser.add_argument("--export-hdf", type=Path, help="Path to HDF5 file for time-series export (requires pandas/pyarrow).")
    args = parser.parse_args()

    capture_detail = bool(args.save_json or args.plot or args.export_prefix or args.export_hdf)
    report = solve_scenario(
        args.scenario,
        solver_name=args.solver,
        tee=args.tee,
        capture_detail=capture_detail,
        export_csv_prefix=args.export_prefix,
        export_hdf=args.export_hdf,
    )

    mip = report["mip"]
    lp = report["lp"]

    print(f"Scenario: {report['scenario_id']}")
    print(f"MIP objective: {mip.objective:,.2f}  (status={mip.solver_status.name}, termination={mip.termination_condition.name})")
    print(f"LP objective:  {lp.objective:,.2f}  (status={lp.solver_status.name}, termination={lp.termination_condition.name})")

    cost_components = report["cost_components"]
    if cost_components:
        print("Cost components (currency units):")
        for name, value in sorted(cost_components.items()):
            print(f"  {name}: {value:,.2f}")

    duals = report["lp_duals"].get("power_balance", {})
    if duals:
        print("Top power-balance duals (|value| sorted):")
        top = sorted(duals.items(), key=lambda kv: abs(kv[1]), reverse=True)[: args.top_duals]
        for (zone, period), dual in top:
            print(f"  lambda[{zone}, t={period}] = {dual:,.2f}")

    if args.save_json:
        if report["detail"] is None:
            raise RuntimeError("Full detail not captured; cannot save detailed JSON.")
        _ensure_parent(args.save_json)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(_serialize_report(report), f, indent=2)
        print(f"Report saved to {args.save_json}")

    if args.plot:
        if report["detail"] is None:
            raise RuntimeError("Detail data required for plotting; rerun with --plot from scratch.")
        out_dir = args.plot_dir or args.scenario.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        from .visualize import plot_scenario_and_dispatch

        plot_scenario_and_dispatch(args.scenario, report, out_dir)
        print(f"Plots saved to {out_dir}")

    if args.export_prefix:
        print(f"CSV dispatch exported with prefix {args.export_prefix}")
    if args.export_hdf:
        print(f"HDF dispatch exported to {args.export_hdf}")


if __name__ == "__main__":
    main()
