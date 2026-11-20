"""Identical model capacity and nested training-only preprocessing for both arms."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from ..models import positive_fit


def fit(x, y, kind, ridge):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Empty or nonfinite training inputs')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    active = scale > 1e-12
    scale[~active] = 1.
    X = (x - mean) / scale
    X[:, ~active] = 0.
    result = dict(kind=kind, mean=mean.tolist(), scale=scale.tolist(), active=active.tolist(), ridge=ridge)
    if kind == 'logistic':
        if np.any((y < 0) | (y > 1)):
            raise ValueError('Invalid binary label')
        if np.all(y == y[0]) or len(y) < 10:
            return dict(result, constant=float(y[0]) if np.all(y == y[0]) else float((y.sum() + 1) / (len(y) + 2)))
        T = np.c_[np.ones(len(X)), X]
        def objective(beta):
            eta = T @ beta
            g = T.T @ (expit(eta) - y) / len(y)
            g[1:] += ridge * beta[1:]
            return float(np.mean(np.logaddexp(0, eta) - y * eta) + ridge / 2 * (beta[1:] @ beta[1:])), g
        sol = minimize(objective, np.zeros(T.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10})
        if not sol.success:
            raise RuntimeError('Logistic optimizer: ' + sol.message)
        intercept, coefficients = float(sol.x[0]), sol.x[1:]
    else:
        intercept = float(y.mean())
        if kind == 'positive':
            coefficients = positive_fit(X, y - intercept, ridge)
        elif kind == 'signed':
            a = np.vstack([X / np.sqrt(len(y)), np.sqrt(ridge) * np.eye(X.shape[1])])
            rhs = np.r_[(y - intercept) / np.sqrt(len(y)), np.zeros(X.shape[1])]
            coefficients = np.linalg.lstsq(a, rhs, rcond=None)[0]
        else:
            raise ValueError(kind)
    coefficients[~active] = 0.
    return dict(result, intercept=intercept, coefficients=coefficients.tolist())


def predict(model, x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite test inputs')
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    X = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
    X[:, ~np.asarray(model['active'], bool)] = 0.
    eta = model['intercept'] + X @ np.asarray(model['coefficients'])
    return expit(eta) if model['kind'] == 'logistic' else eta


def tune(train, test, inner, target, cols, kind, grid):
    if len(inner) != len(train) or len(np.unique(inner)) < 2:
        raise ValueError('At least two internal folds required')
    scores = []
    for ridge in grid:
        losses = []
        for f in np.unique(inner):
            a, b = train.iloc[np.flatnonzero(inner != f)], train.iloc[np.flatnonzero(inner == f)]
            if set(a.group) & set(b.group):
                raise ValueError('Scenario leakage in internal folds')
            m = fit(a[cols], a[target], kind, ridge)
            losses.extend((predict(m, b[cols]) - b[target].to_numpy())**2)
        scores.append(dict(ridge=ridge, mse=float(np.mean(losses))))
    best = min(r['mse'] for r in scores)
    ridge = max(r['ridge'] for r in scores if r['mse'] <= best + 1e-12)
    model = fit(train[cols], train[target], kind, ridge)
    return predict(model, test[cols]), dict(model, columns=cols, tuning=scores)
