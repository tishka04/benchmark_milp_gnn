import json
import numpy as np
import pandas as pd
import pytest
from research.paper1_revision.realized import storage, pipeline, report


def test_freeze_refuses_an_independent_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'CAMP', tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(jobs=[dict(job_id='abc', kind='external_trial')])))
    (tmp_path / 'attempts' / 'abc_1234').mkdir(parents=True)
    with pytest.raises(ValueError, match='attempt already exists'):
        storage.no_external_outcomes()


def test_end_to_end_comparison_and_freeze_without_physical_outcomes(tmp_path, monkeypatch):
    """Exercise all pipeline stages on synthetic labels, never an energy solve."""
    from research.paper1_revision.data import sha
    from research.paper1_revision.realized.features import IDS
    rng = np.random.default_rng(34)
    base = rng.uniform(.1, 1., (600, 22))
    cases = pd.DataFrame({'group': ['g' + str(i) for i in range(600)],
                          'family': ['f' + str(i % 6) for i in range(600)],
                          'outer_fold': np.arange(600) % 5, 'reference_certified': True})
    for j, ident in enumerate(IDS):
        cases['raw_' + ident] = base[:, j]
        cases['real_' + ident] = base[:, j] + .05 * rng.normal(size=600)
    history = pd.concat([cases.assign(budget=b, equal4=.1 + .6 * base[:, 0],
        E=(base[:, 0] > .5).astype(int), ES=(base[:, 0] > .6).astype(int)) for b in [5, 10, 30, 60, 120]], ignore_index=True)
    features = pd.DataFrame(dict(group=cases.group, population='historical', audit_load_seconds=.01,
        audit_build_seconds=.01, audit_descriptor_seconds=.01, audit_total_seconds=.03))
    cfg = dict(policies=['equal4'], deadlines=[5, 10, 30, 60, 120], historical_fold_seed=20261111)
    proto = dict(representations=['legacy_raw', 'realized'], secondary_representations=[],
        schemes=['nested', 'lofo'], ridge_grid=[.01], logistic_grid=[.01], bootstrap_samples=30, seed=12)
    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')
    def divisions(frame, scheme):
        key = 'outer_fold' if scheme == 'nested' else 'family'
        for f in sorted(frame[key].unique()):
            train, test = frame[frame[key] != f], frame[frame[key] == f]
            inner = train.group.str[1:].astype(int).to_numpy() % 3
            if scheme == 'lofo':
                inner = pd.factorize(train.family, sort=True)[0]
            yield str(f), train, test, inner
    def external(_):
        return pd.concat([cases.iloc[:400].drop(columns=['family', 'outer_fold']).assign(
            group=['x' + str(i) for i in range(400)], batch='synthetic', budget=b)
            for b in cfg['deadlines']], ignore_index=True)
    for module in [storage, pipeline, report]:
        monkeypatch.setattr(module, 'DEST', tmp_path)
        monkeypatch.setattr(module, 'put', write)
    for module in [pipeline, report]:
        monkeypatch.setattr(module, 'config', lambda: cfg)
        monkeypatch.setattr(module, 'protocol', lambda: proto)
        monkeypatch.setattr(module, 'load_frames', lambda: (history, features, cfg))
    monkeypatch.setattr(pipeline, 'fit_signature', lambda: 'synthetic-fixture')
    monkeypatch.setattr(pipeline, 'no_external_outcomes', lambda: None)
    monkeypatch.setattr(pipeline, 'splits', divisions)
    monkeypatch.setattr(pipeline, 'external_frame', external)
    features.to_parquet(tmp_path / 'features.parquet', index=False)
    write(tmp_path / 'feature_manifest.json', dict(features_sha256=sha(tmp_path / 'features.parquet')))
    pipeline.fit_all()
    p = pipeline.collect('nested')
    assert len(p) == 6 * 3000
    pipeline.freeze()
    assert pipeline.verify_freeze()['independent_outcomes_used'] is False
    assert len(pd.read_parquet(tmp_path / 'external_predictions.parquet')) == 6 * 2000
    physical = tmp_path / 'physical'
    physical.mkdir()
    monkeypatch.setattr(pipeline, 'CAMP', physical)
    labels = history[history.group == 'g0'][['budget', 'equal4', 'E', 'ES']].assign(group='x0')
    labels.to_parquet(physical / 'external_observations.parquet', index=False)
    pipeline.evaluate()
    status = json.loads((tmp_path / 'external_status.json').read_text())
    assert status['status'] == 'partial' and status['evaluated_scenario_deadlines'] == 5
    assert len(pd.read_parquet(tmp_path / 'external_evaluation.parquet')) == 30
    report.run(include_allocation=False)
    contrast = pd.read_csv(tmp_path / 'nested_primary_contrasts.csv')
    assert len(contrast) == 15 and np.isfinite(contrast.mean_mse_gain).all()
    # A resumed fit must produce the identical frozen artifact, not retrain on test.
    before = sha(tmp_path / 'external_predictions.parquet')
    pipeline.fit_all()
    pipeline.freeze()
    assert sha(tmp_path / 'external_predictions.parquet') == before
    (tmp_path / 'external_predictions.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='predictions changed'):
        pipeline.verify_freeze()


def test_controller_prepare_cannot_launch_physical_campaign(monkeypatch):
    import sys
    from research.paper1_revision import complete_campaign, campaign
    from research.paper1_revision.realized import __main__ as entry
    actions = []
    monkeypatch.setattr(sys, 'argv', ['campaign', 'prepare'])
    monkeypatch.setattr(entry, 'prepare_all', lambda: actions.append('prepare'))
    def forbidden(*args):
        raise AssertionError('Physical campaign called during preparation')
    monkeypatch.setattr(campaign, 'execute', forbidden)
    complete_campaign.main()
    assert actions == ['prepare']


def test_controller_refuses_physical_execution_without_freeze(monkeypatch):
    import sys
    from research.paper1_revision import complete_campaign, campaign
    monkeypatch.setattr(sys, 'argv', ['campaign', 'run'])
    def missing():
        raise ValueError('not frozen')
    monkeypatch.setattr(pipeline, 'verify_freeze', missing)
    monkeypatch.setattr(campaign, 'execute', lambda *args: pytest.fail('Physical run started'))
    with pytest.raises(ValueError, match='not frozen'):
        complete_campaign.main()
