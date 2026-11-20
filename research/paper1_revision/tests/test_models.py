import numpy as np
from research.paper1_revision.models import linear_fit,linear_predict,fractional_fit_predict
from research.paper1_revision.data import config

def toy(rotating=False):
    rng=np.random.default_rng(8); x=rng.uniform(size=(120,4)); x[:,3]=.4
    X=np.tile(x,(3,1)); b=np.repeat([5,30,120],120)
    w=np.array([[.2,.6,.2,0],[.2,.6,.2,0],[.2,.6,.2,0]])
    if rotating: w=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    y=np.concatenate([.1+j*.1+(x@w[j])*(j+1)/3 for j in range(3)])
    return X,b,y

def test_rank_one_recovers_scale_and_normalization():
    x,b,y=toy(); m=linear_fit(x,y,b,'rank1',0,config())
    assert np.mean((linear_predict(m,x,b)-y)**2)<1e-9
    coef=m['coefficients']; assert np.all(coef>=0)
    assert np.allclose(coef[:,3],0,atol=1e-8)
    assert np.allclose(coef[0]/coef[0].sum(),coef[-1]/coef[-1].sum())
    assert all(d['converged'] for d in m['diagnostics'])

def test_separate_needed_for_rotating_direction():
    x,b,y=toy(True)
    rank=linear_fit(x,y,b,'rank1',0,config()); separate=linear_fit(x,y,b,'separate',0,config())
    assert np.mean((linear_predict(separate,x,b)-y)**2)<1e-15
    assert np.mean((linear_predict(rank,x,b)-y)**2)>.005

def test_projection_never_increases_bounded_squared_loss():
    rng=np.random.default_rng(1); y=rng.uniform(size=100); p=rng.normal(size=100)
    assert np.all((np.clip(p,0,1)-y)**2<=(p-y)**2+1e-15)

def test_fractional_boundary_and_empty_condition():
    x=np.ones((20,2)); xt=np.ones((3,2))
    assert np.all(fractional_fit_predict(x,np.zeros(20),xt,.1)==0)
    assert np.all(fractional_fit_predict(x[:0],np.zeros(0),xt,.1)==.5)
    p=fractional_fit_predict(x,np.linspace(0,1,20),xt,.1)
    assert np.allclose(p,.5)
