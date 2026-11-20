import itertools
import numpy as np
from research.paper1_revision.allocation import allocate,uniform_mix

def test_exact_portfolio_solution_against_exhaustive_search():
    values=np.array([[.1,.6,.4],[.3,.4,.9],[.8,.7,.9]])
    budgets=np.array([5,10,30]); total=45
    choices=allocate(values,budgets,total)
    best=max(sum(values[i,j] for i,j in enumerate(c)) for c in itertools.product(range(3),repeat=3) if sum(budgets[list(c)])<=total)
    assert abs(values[np.arange(3),choices].sum()-best)<1e-12
    assert budgets[choices].sum()<=total

def test_duplicate_bootstrap_units_and_uniform_mixture():
    values=np.tile([.1,.6,.8],(4,1));b=np.array([5,10,30])
    c=allocate(values,b,45);assert b[c].sum()<=45
    c=uniform_mix(4,b,80);assert b[c].sum()==80
