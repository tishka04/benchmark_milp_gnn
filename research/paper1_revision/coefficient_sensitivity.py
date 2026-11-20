"""Summarize grouped coefficients across already fitted target sensitivities."""
import numpy as np
import pandas as pd
from .data import RUN, COMPONENTS, read
from .models import PHYSICAL_GROUPS


def run():
    used = set(sum(PHYSICAL_GROUPS, []))
    groups = PHYSICAL_GROUPS + [[i] for i in range(22) if i not in used]
    names = ['+'.join(COMPONENTS[i].split('_')[0] for i in g) for g in groups]
    rows = []
    for path in (RUN / 'checkpoints/sensitivity').glob('*/*/*.json'):
        policy, variant = path.parts[-3:-1]
        fold, model_name = path.stem.split('_', 1)
        meta = read(path)['meta']
        fits = meta['by_deadline'].values() if 'by_deadline' in meta else [meta]
        for fit in fits:
            model = fit['model']
            for budget, raw in zip(model['levels'], model['coefficients']):
                coef = np.asarray(raw)
                scale = float(coef.sum())
                grouped = np.array([coef[g].sum() for g in groups])
                assert np.isclose(grouped.sum(), scale)
                normalized = grouped / scale if scale > 1e-12 else np.zeros(len(groups))
                rows.append(dict(policy=policy, variant=variant, fold=int(fold), model=model_name,
                    budget=int(budget), total_scale=scale,
                    **dict(zip(names, normalized))))
    frame = pd.DataFrame(rows)
    frame.to_csv(RUN / 'sensitivity_grouped_weights.csv', index=False)
    keys = ['policy', 'model', 'fold', 'budget']
    baseline = frame[frame.variant == 'main'][keys + names]
    paired = frame.merge(baseline, on=keys, validate='many_to_one', suffixes=('', '_main'))
    a = paired[names].to_numpy()
    b = paired[[name + '_main' for name in names]].to_numpy()
    paired['group_weight_total_variation'] = abs(a - b).sum(axis=1) / 2
    paired['same_largest_group'] = (a.argmax(axis=1) == b.argmax(axis=1)) & (a.sum(axis=1) > 0) & (b.sum(axis=1) > 0)
    paired.groupby(['policy', 'variant', 'model', 'budget']).agg(
        folds=('fold', 'size'), mean_total_variation=('group_weight_total_variation', 'mean'),
        max_total_variation=('group_weight_total_variation', 'max'),
        fraction_same_largest_group=('same_largest_group', 'mean')).to_csv(
            RUN / 'sensitivity_group_weight_stability.csv')


if __name__ == '__main__':
    run()
