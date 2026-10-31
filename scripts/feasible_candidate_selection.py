"""Feasibility-first selection for the new benchmark; no ML imports or side effects."""
from contextlib import contextmanager
import math


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_sample_index(policy, energies, objectives):
    """Preserve energy ranking, but only after the dispatch/DR feasibility gate.

    The repair serializer sets objective=inf for invalid DR or absent solutions.
    If all candidates fail, index zero retains a diagnostic failure, not a success.
    No oracle dispatch-cost selection is substituted for the energy policy.
    """
    if len(energies) != len(objectives):
        raise ValueError("Candidate energies/objectives must have the same length")
    if policy not in {"energy", "first", "first_feasible", "oracle_lp"}:
        raise ValueError(f"Unknown selection policy: {policy}")
    eligible = [i for i, cost in enumerate(objectives) if finite(cost)]
    if not eligible:
        return 0
    if policy in {"first", "first_feasible"}:
        return eligible[0]
    if policy == "oracle_lp":
        return min(eligible, key=lambda i: float(objectives[i]))
    ranked = [i for i in eligible if finite(energies[i])]
    return min(ranked, key=lambda i: float(energies[i])) if ranked else eligible[0]


@contextmanager
def corrected_selection(evaluation_module):
    """Install only inside an isolated new worker; never edit the live module file."""
    previous = evaluation_module._select_sample_index
    evaluation_module._select_sample_index = select_sample_index
    try:
        yield
    finally:
        evaluation_module._select_sample_index = previous
