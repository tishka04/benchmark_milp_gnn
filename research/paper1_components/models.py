"""Nested grouped logistic models, additive bases and hierarchical interactions."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from .dataio import COMPONENTS, SIZE, INTERPRETABLE, group_folds

@dataclass(frozen=True)
class Spec:
    name: str
    components: tuple = ()
    extra: tuple = ()
    nonlinear: bool = False
    pairs: tuple = ()

def specifications(cfg):
    pairs = tuple(tuple(p) for p in cfg["interactions"])
    all_c = tuple(COMPONENTS)
    specs = [Spec("size"), Spec("index", extra=("criticality",)),
             Spec("axes", extra=("stress", "hardness")),
             Spec("interpretable", components=tuple(INTERPRETABLE)),
             Spec("linear", components=all_c),
             Spec("additive", components=all_c, nonlinear=True),
             Spec("interactions", components=all_c, nonlinear=True, pairs=pairs)]
    for c in cfg["targeted_components"]:
        specs.append(Spec("drop_component_"+c.split("_")[0],
            components=tuple(v for v in all_c if v != c), nonlinear=True,
            pairs=tuple(p for p in pairs if c not in p)))
    for i in range(len(pairs)):
        specs.append(Spec(f"drop_interaction_{i+1}", components=all_c, nonlinear=True,
                          pairs=tuple(p for j, p in enumerate(pairs) if i != j)))
    return specs

class Design:
    """Only declared pre-solve columns are read. All fitted quantities use training rows."""
    def __init__(self, spec, family_control, cfg):
        self.spec, self.family_control, self.cfg = spec, family_control, cfg
        self.columns = SIZE + list(spec.extra) + list(spec.components)

    def fit(self, train):
        raw = train[self.columns].astype(float).replace([np.inf, -np.inf], np.nan)
        self.medians = raw.median().fillna(0.)
        clean = raw.fillna(self.medians)
        self.centers = clean.mean()
        self.categories = sorted(train.family.astype(str).unique()) if self.family_control else []
        self.inactive_pairs = [p for p in self.spec.pairs
            if any(float(np.ptp(clean[c].to_numpy())) <= self.cfg["constant_tolerance"] for c in p)]
        x, names = self._raw(train)
        self.names_all = names
        self.active = np.ptp(x, axis=0) > self.cfg["constant_tolerance"]
        self.names = [n for n, keep in zip(names, self.active) if keep]
        x = x[:, self.active]
        self.means = x.mean(axis=0)
        self.scales = x.std(axis=0)
        self.scales[self.scales <= self.cfg["constant_tolerance"]] = 1.
        return self

    def _raw(self, frame):
        clean = frame[self.columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        values, names = [], []
        for c in self.columns:
            z = clean[c].to_numpy()
            values.append(z); names.append(c)
            if self.spec.nonlinear and c in self.spec.components:
                for knot in self.cfg["hinge_knots"]:
                    values.append(np.maximum(0., z-knot)); names.append(f"{c}:hinge:{knot:g}")
        for a, b in self.spec.pairs:
            if (a, b) in self.inactive_pairs:
                continue
            values.append((clean[a].to_numpy()-self.centers[a]) *
                          (clean[b].to_numpy()-self.centers[b]))
            names.append(a+"*"+b)
        for family in self.categories:
            values.append((frame.family.astype(str) == family).to_numpy(dtype=float))
            names.append("family:"+family)
        return np.column_stack(values), names

    def transform(self, frame):
        x, names = self._raw(frame)
        if names != self.names_all:
            raise ValueError("Design columns changed")
        return (x[:, self.active]-self.means)/self.scales

@dataclass
class Fitted:
    design: Design
    estimator: object
    constant: float | None
    status: str

    def predict(self, frame):
        x = self.design.transform(frame)
        if self.constant is not None:
            return np.full(len(frame), self.constant)
        return self.estimator.predict_proba(x)[:, 1]

    def metadata(self):
        coefficients = [] if self.estimator is None else [
            dict(term=n, coefficient=float(v))
            for n, v in zip(self.design.names, self.estimator.coef_[0])]
        return dict(status=self.status, active_terms=self.design.names,
                    dropped_constant_terms=[n for n in self.design.names_all if n not in self.design.names],
                    inactive_interactions=[list(p) for p in self.design.inactive_pairs],
                    coefficients=coefficients)

def fit_one(train, y, spec, family_control, C, cfg):
    design = Design(spec, family_control, cfg).fit(train)
    x = design.transform(train)
    y = np.asarray(y, dtype=int)
    if not len(y):
        raise ValueError("Empty training labels")
    if len(np.unique(y)) == 1 or x.shape[1] == 0:
        return Fitted(design, None, float(y.mean()),
                      "constant_label" if len(np.unique(y)) == 1 else "intercept_only")
    estimator = LogisticRegression(C=C, max_iter=cfg["max_iter"], solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"Unconverged logistic fit: {spec.name}, C={C}; preserve this attempt")
    return Fitted(design, estimator, None, "fitted")

def tune_fit(train, endpoint, spec, family_control, cfg):
    """Re-tune each reduced model inside the same outer-training groups."""
    if train[endpoint].isna().any():
        raise ValueError("Unknown labels must be excluded explicitly")
    if train[endpoint].nunique() == 1:
        model = fit_one(train, train[endpoint], spec, family_control, cfg["regularization_C"][0], cfg)
        return model, dict(C=None, inner_brier=None, inner_scores=[], tuning_status="constant_label")
    split = group_folds(train, cfg["inner_folds"], "paper1-inner")
    if split.nunique() != cfg["inner_folds"]:
        raise ValueError("Insufficient independent groups for inner validation")
    scores = []
    for C in sorted(cfg["regularization_C"]):
        losses = []
        for fold in range(cfg["inner_folds"]):
            fit, valid = train[split != fold], train[split == fold]
            if set(fit.group) & set(valid.group):
                raise ValueError("Inner group leakage")
            model = fit_one(fit, fit[endpoint], spec, family_control, C, cfg)
            losses.extend((valid[endpoint].to_numpy()-model.predict(valid))**2)
        scores.append(dict(C=float(C), brier=float(np.mean(losses))))
    best = min(scores, key=lambda r: (round(r["brier"], 12), r["C"]))
    model = fit_one(train, train[endpoint], spec, family_control, best["C"], cfg)
    return model, dict(C=best["C"], inner_brier=best["brier"],
                       inner_scores=scores, tuning_status="nested_grouped")
