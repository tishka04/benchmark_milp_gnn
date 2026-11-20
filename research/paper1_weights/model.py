"""Additive regressions with deadline intercepts and optional positive slopes."""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from .data import COMPONENTS, ORIGINAL_WEIGHTS, digest


def arrays(rows, policy):
    return dict(x=np.array([[r[c] for c in COMPONENTS] for r in rows], dtype=float),
                y=np.array([r['targets'][policy]['score'] for r in rows], dtype=float),
                budget=np.array([r['budget'] for r in rows], dtype=int),
                group=np.array([r['group'] for r in rows]),
                family=np.array([r['family'] for r in rows]))


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def group_folds(data, count, seed, salt):
    units = {}
    for group, family in zip(data['group'], data['family']):
        if group in units and units[group] != family:
            raise ValueError('Group crosses families')
        units[group] = family
    mapping = {}
    # Family is used only to balance the fixed folds, never as a predictor.
    for family in sorted(set(units.values())):
        order = sorted((g for g in units if units[g] == family),
                       key=lambda g: (digest([seed, salt, g]), g))
        for i, g in enumerate(order):
            mapping[g] = i % count
    result = np.array([mapping[g] for g in data['group']], dtype=int)
    if len(set(result.tolist())) != count:
        raise ValueError('Too few independent groups for the declared folds')
    return result


def design(x, kind):
    if kind in ('positive', 'signed'):
        return x, list(COMPONENTS)
    if kind == 'original_index':
        return (x @ np.array(ORIGINAL_WEIGHTS))[:, None], ['original_index']
    if kind == 'intercept':
        return np.empty((len(x), 0)), []
    raise ValueError('Unknown regression kind')


def fit(data, kind, ridge, cfg, sample_weight=None):
    x, names = design(data['x'], kind)
    y, budgets = data['y'], data['budget']
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if len(weights) != len(y) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Invalid observation weights')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite regression data')
    xc, yc = np.zeros_like(x), np.zeros_like(y)
    means = {}
    for b in sorted(set(budgets.tolist())):
        mask = budgets == b
        if weights[mask].sum() <= 0:
            raise ValueError('Bootstrap lost all observations for one deadline')
        mx = np.average(x[mask], axis=0, weights=weights[mask])
        my = float(np.average(y[mask], weights=weights[mask]))
        means[b] = (mx, my)
        xc[mask], yc[mask] = x[mask] - mx, y[mask] - my
    supported = weights > 0
    active = np.ptp(xc[supported], axis=0) > cfg['constant_tolerance']
    coef = np.zeros(x.shape[1])
    if active.any():
        scale = np.sqrt(weights / weights.sum())
        a, rhs = xc[:, active] * scale[:, None], yc * scale
        if ridge > 0:
            a = np.vstack([a, np.sqrt(ridge) * np.eye(int(active.sum()))])
            rhs = np.concatenate([rhs, np.zeros(int(active.sum()))])
        if kind == 'signed':
            coef[active] = np.linalg.lstsq(a, rhs, rcond=None)[0]
        else:
            solution = lsq_linear(a, rhs, bounds=(0., np.inf), method='bvls',
                                  tol=1e-10, max_iter=1000)
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError('Positive regression did not converge: ' + str(solution.message))
            coef[active] = np.maximum(0., solution.x)
    intercepts = {str(b): float(my - mx @ coef) for b, (mx, my) in means.items()}
    total = float(coef.sum())
    index_weights = None
    if kind == 'positive' and total > cfg['constant_tolerance']:
        index_weights = (coef / total).tolist()
    return dict(kind=kind, ridge=float(ridge), names=names, coefficients=coef.tolist(),
                active=active.tolist(), intercepts=intercepts, index_scale=total,
                index_weights=index_weights,
                degenerate_index=kind == 'positive' and index_weights is None)


def predict(model, data):
    x, _ = design(data['x'], model['kind'])
    offsets = np.array([model['intercepts'][str(int(b))] for b in data['budget']])
    # No clipping: preserve the exact additive equation and report extrapolation.
    return offsets + x @ np.array(model['coefficients'])


def tune(data, kind, cfg, salt):
    if kind in ('intercept', 'original_index'):
        return 0., []
    folds = group_folds(data, cfg['inner_folds'], cfg['seed'], salt)
    scores = []
    for ridge in cfg['ridge_grid']:
        loss, count = 0., 0
        for fold in range(cfg['inner_folds']):
            train, test = subset(data, folds != fold), subset(data, folds == fold)
            model = fit(train, kind, ridge, cfg)
            residual = predict(model, test) - test['y']
            loss += float(residual @ residual); count += len(residual)
        scores.append(dict(ridge=float(ridge), mse=loss / count))
    best = min(s['mse'] for s in scores)
    # Deterministic preference for stronger regularization on numerical ties.
    chosen = max(s['ridge'] for s in scores if s['mse'] <= best + 1e-12)
    return chosen, scores


def specifications(cfg):
    return ([dict(name='common_positive', kind='positive', deadline=None),
             dict(name='common_signed', kind='signed', deadline=None),
             dict(name='original_index', kind='original_index', deadline=None),
             dict(name='intercept_only', kind='intercept', deadline=None)] +
            [dict(name=f'deadline_{b}_positive', kind='positive', deadline=b) for b in cfg['deadlines_s']])


def scoped(data, spec):
    return data if spec['deadline'] is None else subset(data, data['budget'] == spec['deadline'])


def cv_job(task):
    job, data, spec, fold, cfg = task
    data = scoped(data, spec)
    folds = group_folds(data, cfg['outer_folds'], cfg['seed'], 'outer')
    train, test = subset(data, folds != fold), subset(data, folds == fold)
    ridge, tuning = tune(train, spec['kind'], cfg, f'inner-{fold}')
    model = fit(train, spec['kind'], ridge, cfg)
    return dict(job=job, spec=spec, fold=fold, model=model, tuning=tuning,
                predictions=[dict(group=str(g), budget=int(b), observed=float(y), predicted=float(p))
                             for g, b, y, p in zip(test['group'], test['budget'], test['y'], predict(model, test))])


def final_job(task):
    job, data, spec, cfg = task
    data = scoped(data, spec)
    ridge, tuning = tune(data, spec['kind'], cfg, 'final-inner')
    return dict(job=job, spec=spec, model=fit(data, spec['kind'], ridge, cfg), tuning=tuning)


def bootstrap_job(task):
    job, data, spec, model, start, stop, cfg = task
    data = scoped(data, spec)
    groups, inverse = np.unique(data['group'], return_inverse=True)
    samples = []
    for repetition in range(start, stop):
        seed = int(digest([cfg['seed'], 'coefficient-bootstrap', repetition])[:8], 16)
        rng = np.random.default_rng(seed)
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        fitted = fit(data, spec['kind'], model['ridge'], cfg, sample_weight=counts[inverse])
        samples.append(dict(repetition=repetition, coefficients=fitted['coefficients'],
                            active=fitted['active'], index_weights=fitted['index_weights'],
                            index_scale=fitted['index_scale']))
    return dict(job=job, spec=spec, samples=samples,
                uncertainty='Scenario bootstrap with full coefficient refitting; final selected ridge held fixed.')
