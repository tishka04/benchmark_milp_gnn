import copy
import json
import numpy as np
import pytest
from src.milp.model import build_uc_model
from research.paper1_revision.campaign import CAMP, load_physics
from research.paper1_revision.data import read
from research.paper1_revision.realized.features import from_instance, extract


@pytest.fixture
def small_instance():
    case = read(CAMP / 'manifest.json')['entries'][0]
    return load_physics(case['file']), case


def test_realized_peak_valley_is_scale_invariant_and_shape_sensitive(small_instance):
    data, _ = small_instance
    a, _ = from_instance(data, build_uc_model(data))
    scaled = copy.deepcopy(data)
    scaled.demand = {k: v * 3 for k, v in data.demand.items()}
    b, _ = from_instance(scaled, build_uc_model(scaled))
    assert a['A3'] == pytest.approx(b['A3'])
    flat = copy.deepcopy(data)
    for z in data.zones:
        mean = np.mean([data.demand[z, t] for t in data.periods])
        for t in data.periods:
            flat.demand[z, t] = mean
    c, _ = from_instance(flat, build_uc_model(flat))
    assert c['A3'] == pytest.approx(1.) and a['A3'] > c['A3']


def test_inactive_metadata_cannot_change_realized_features(small_instance, tmp_path):
    _, case = small_instance
    raw = read(case['file'])
    raw['tech']['battery_e_to_p_hours'] = 19.
    raw['tech']['dr_duration_hours'] = 17.
    raw['operation_costs']['nuclear_startup_cost_eur'] = 1e9
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    a, _ = extract(case['file'])
    b, _ = extract(path)
    assert a == pytest.approx(b)


def test_free_binaries_and_realized_energy_respond_to_model(small_instance):
    data, _ = small_instance
    model = build_uc_model(data)
    a, diagnostics = from_instance(data, model)
    assert a['B3'] == diagnostics['free_binary_variables'] < diagnostics['binary_variables']
    variable = next(v for v in model.b_charge_mode.values() if not v.fixed)
    variable.fix(0)
    b, _ = from_instance(data, model)
    assert b['B3'] == a['B3'] - 1
    doubled = copy.deepcopy(data)
    for name in ['battery_energy', 'battery_initial', 'battery_final_min', 'battery_final_max',
                 'pumped_energy', 'pumped_initial', 'pumped_final_min', 'pumped_final_max']:
        setattr(doubled, name, {k: 2 * v for k, v in getattr(data, name).items()})
    c, _ = from_instance(doubled, build_uc_model(doubled))
    assert c['A8'] < a['A8']
    assert c['B8'] == pytest.approx(2 * a['B8'])


def test_topology_counts_edges_and_absent_storage_is_finite(small_instance):
    data, _ = small_instance
    from src.milp.scenario_loader import NetworkLine
    data.lines['extra'] = NetworkLine('extra', data.zones[0], data.zones[-1], 321., 10.)
    data.lines_from_index = {z: tuple(k for k, l in data.lines.items() if l.from_zone == z) for z in data.zones}
    data.lines_to_index = {z: tuple(k for k, l in data.lines.items() if l.to_zone == z) for z in data.zones}
    for name in ['battery_power', 'battery_energy', 'battery_initial', 'battery_final_min', 'battery_final_max',
                 'pumped_power', 'pumped_energy', 'pumped_initial', 'pumped_final_min', 'pumped_final_max']:
        setattr(data, name, {z: 0. for z in data.zones})
    data.demand = {k: 0. for k in data.demand}
    features, _ = from_instance(data, build_uc_model(data))
    assert features['B9'] == len(data.lines) / len(data.zones)
    assert features['B8'] == 0.
    assert np.isfinite(list(features.values())).all()


def test_extraction_never_invokes_highs(small_instance, monkeypatch):
    import highspy
    def forbidden(*args, **kwargs):
        raise AssertionError('A descriptor computation must never optimize a MILP')
    monkeypatch.setattr(highspy.Highs, 'run', forbidden)
    _, case = small_instance
    features, diagnostics = extract(case['file'])
    assert len(features) == 22 and diagnostics['total_seconds'] > 0


def test_fresh_and_json_checkpoint_exports_have_identical_schema(small_instance, tmp_path):
    import pandas as pd
    from research.paper1_revision.realized.storage import feature_frame
    data, _ = small_instance
    values, diagnostics = from_instance(data, build_uc_model(data))
    diagnostics.update(load_seconds=.1, build_seconds=.2, descriptor_seconds=.3, total_seconds=.6)
    row = dict(group='example', population='historical', **{'real_' + k: v for k, v in values.items()},
               **{'audit_' + k: v for k, v in diagnostics.items()})
    cached = json.loads(json.dumps(row, sort_keys=True))
    a, b = feature_frame([row]), feature_frame([cached])
    pd.testing.assert_frame_equal(a, b)
    a.to_parquet(tmp_path / 'fresh.parquet', index=False)
    b.to_parquet(tmp_path / 'cached.parquet', index=False)
    assert (tmp_path / 'fresh.parquet').read_bytes() == (tmp_path / 'cached.parquet').read_bytes()
