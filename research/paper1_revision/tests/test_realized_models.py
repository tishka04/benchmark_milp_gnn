import numpy as np
import pandas as pd
import pytest
from research.paper1_revision.realized.models import fit, predict, tune
from research.paper1_revision.realized.storage import columns


def test_scaling_uses_training_only_and_unseen_constant_stays_inactive():
    train = np.array([[1., 5.], [2., 5.], [3., 5.]])
    y = np.array([.2, .4, .6])
    model = fit(train, y, 'positive', 0.)
    assert model['mean'] == [2., 5.]
    assert model['active'] == [True, False]
    a = predict(model, [[2., 5.]])
    b = predict(model, [[2., 1e12]])
    assert a == pytest.approx(b)
    assert predict(model, train) == pytest.approx(y)


def test_probability_single_class_and_serializable_roundtrip():
    import json
    x = np.arange(30.)[:, None]
    for constant in [0., 1.]:
        m = fit(x, np.full(30, constant), 'logistic', .1)
        assert np.all(predict(m, x) == constant)
    y = (x[:, 0] > 13).astype(float)
    m = fit(x, y, 'logistic', .01)
    p = predict(json.loads(json.dumps(m)), x)
    assert np.all((p >= 0) & (p <= 1)) and p[-1] > p[0]


def test_nested_tuning_rejects_shared_scenario():
    frame = pd.DataFrame(dict(group=['same', 'same', 'other', 'other'], x=[1., 2., 3., 4.], y=[0., 0., 1., 1.]))
    with pytest.raises(ValueError, match='leakage'):
        tune(frame, frame, np.array([0, 1, 0, 1]), 'y', ['x'], 'positive', [0.])


def test_correction_arm_is_limited_and_size_control_is_preserved():
    a, b = columns('legacy_raw'), columns('corrected_only')
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert changed == [2, 14, 19, 20]
    assert columns('size') == ['real_B1']
    assert len(columns('legacy_without_disputed')) == 18
