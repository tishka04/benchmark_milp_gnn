"""Positive fractional-logit regression and matched B9 ablation."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from ..realized.models import fit as linear_fit, predict as original_predict
from ..realized.storage import columns


def feature_columns(representation, arm):
    result = columns(representation)
    if 'no_b9' in arm:
        result = [c for c in result if c not in ('raw_B9', 'real_B9')]
    return result


def objective(beta, matrix, y, ridge):
    """Convex soft-label cross entropy; no rounding of the continuous outcomes."""
    eta = matrix @ beta
    gradient = matrix.T @ (expit(eta) - y) / len(y)
    gradient[1:] += ridge * beta[1:]
    loss = np.mean(y * np.logaddexp(0., -eta) + (1 - y) * np.logaddexp(0., eta))
    loss += ridge / 2 * (beta[1:] @ beta[1:])
    return float(loss), gradient


def projected_gradient(beta, gradient):
    return np.r_[gradient[0], beta[1:] - np.maximum(0., beta[1:] - gradient[1:])]


def fit_bounded(x, y, ridge):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError('Incompatible or empty training arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Finite inputs and labels in [0,1] required')
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('Invalid ridge')
    mean, scale = x.mean(axis=0), x.std(axis=0)
    active = scale > 1e-12
    scale[~active] = 1.
    result = dict(kind='fractional_logit_positive', mean=mean.tolist(), scale=scale.tolist(),
                  active=active.tolist(), ridge=float(ridge))
    if np.all(y == y[0]):
        return dict(result, constant=float(y[0]), optimizer_iterations=0)
    z = (x[:, active] - mean[active]) / scale[active]
    matrix = np.c_[np.ones(len(x)), z]
    beta = np.r_[logit(np.clip(y.mean(), 1e-12, 1 - 1e-12)), np.zeros(active.sum())]
    bounds = [(None, None)] + [(0., None)] * int(active.sum())
    iterations = 0
    optimizer = 'L-BFGS-B'
    for tolerance in [1e-12, 1e-15]:
        sol = minimize(objective, beta, args=(matrix, y, ridge), jac=True, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 10000, 'ftol': tolerance,
                                               'gtol': 1e-8, 'maxls': 50, 'maxcor': 30})
        iterations += sol.nit
        gradient = objective(sol.x, matrix, y, ridge)[1]
        projected = projected_gradient(sol.x, gradient)
        if sol.success and np.max(abs(projected)) <= 2e-5:
            break
        beta = sol.x
    if not sol.success or np.max(abs(projected)) > 2e-5:
        # Collinear descriptors can trigger L-BFGS-B's relative-loss stop before
        # stationarity. Solve the SAME convex objective with another algorithm.
        optimizer = 'SLSQP_fallback'
        sol = minimize(objective, sol.x, args=(matrix, y, ridge), jac=True, method='SLSQP',
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-13})
        iterations += sol.nit
        projected = projected_gradient(sol.x, objective(sol.x, matrix, y, ridge)[1])
    if not sol.success or not np.isfinite(sol.x).all() or np.max(abs(projected)) > 2e-5:
        raise RuntimeError(f'Bounded optimizer failed: {sol.message}, projected gradient {max(abs(projected))}')
    coefficients = np.zeros(x.shape[1])
    coefficients[active] = sol.x[1:]
    return dict(result, intercept=float(sol.x[0]), coefficients=coefficients.tolist(),
                optimizer=optimizer, optimizer_iterations=int(iterations),
                projected_gradient_max=float(max(abs(projected))))


def predict(model, x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(model['mean']) or not np.isfinite(x).all():
        raise ValueError('Invalid test features')
    if model['kind'] != 'fractional_logit_positive':
        return np.clip(original_predict(model, x), 0., 1.)
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    z = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
    z[:, ~np.asarray(model['active'], bool)] = 0.
    return expit(model['intercept'] + z @ np.asarray(model['coefficients']))


def tune(train, test, inner, target, representation, arm, grid):
    if set(train.group) & set(test.group):
        raise ValueError('Outer scenario leakage')
    inner = np.asarray(inner)
    if len(inner) != len(train) or len(np.unique(inner)) < 2:
        raise ValueError('At least two internal folds required')
    cols = feature_columns(representation, arm)
    bounded = arm.startswith('fractional_logit')
    fitter = fit_bounded if bounded else lambda x, y, r: linear_fit(x, y, 'positive', r)
    tuning_predict = predict if bounded else original_predict
    scores = []
    for ridge in grid:
        losses = []
        for fold in np.unique(inner):
            a, b = train.iloc[np.flatnonzero(inner != fold)], train.iloc[np.flatnonzero(inner == fold)]
            if set(a.group) & set(b.group):
                raise ValueError('Inner scenario leakage')
            model = fitter(a[cols], a[target], ridge)
            losses.extend((tuning_predict(model, b[cols]) - b[target].to_numpy())**2)
        scores.append(dict(ridge=float(ridge), mse=float(np.mean(losses))))
    best = min(r['mse'] for r in scores)
    ridge = max(r['ridge'] for r in scores if r['mse'] <= best + 1e-12)
    model = fitter(train[cols], train[target], ridge)
    return predict(model, test[cols]), dict(model, columns=cols, tuning=scores,
        tuning_metric='bounded_mse' if bounded else 'unbounded_mse_matches_original')
