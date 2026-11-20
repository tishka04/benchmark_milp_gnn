"""Versioned inputs, cached feature extraction and provenance guards."""
from pathlib import Path
import gc
import pandas as pd
from ..data import HOME, ROOT, RUN, COMPONENTS, read, put, sha, digest, config
from ..campaign import CAMP, source_hashes
from .features import IDS, DEFINITIONS, extract

PACKAGE = Path(__file__).resolve().parent
DEST = RUN / 'realized_milp_v1'


def protocol():
    return read(PACKAGE / 'protocol.json')


def code_hashes():
    paths = list(PACKAGE.glob('*.py')) + [PACKAGE / 'protocol.json', HOME / 'complete_campaign.py',
        HOME / 'models.py', HOME / 'analysis.py', HOME / 'allocation.py', HOME / 'data.py', HOME / 'protocol.json',
        HOME / 'future_analysis.py', HOME / 'future_reports.py',
        ROOT / 'research/paper1_weights/data.py', ROOT / 'research/paper1_weights/model.py',
        ROOT / 'src/analysis/criticality_index.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def signature():
    from importlib.metadata import version
    import platform
    return digest(dict(code=code_hashes(), data=sha(RUN / 'manifest.json'),
                       campaign=sha(CAMP / 'manifest.json'), physics=source_hashes(),
                       python=platform.python_version(),
                       libraries={name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyomo', 'highspy']}))


def verify_historical_inputs():
    manifest = read(RUN / 'manifest.json')
    for name, expected in manifest['tables'].items():
        if sha(RUN / name) != expected:
            raise ValueError('Historical input changed: ' + name)
    if sha(RUN / 'splits.json') != manifest['splits_sha256']:
        raise ValueError('Historical validation folds changed')


def immutable(path, obj):
    if path.exists():
        if digest(read(path)) != digest(obj):
            raise ValueError(f'Frozen file changed; use a new version: {path}')
    else:
        put(path, obj)


def no_external_outcomes():
    external_ids = {j['job_id'] for j in read(CAMP / 'manifest.json')['jobs'] if j['kind'].startswith('external_')}
    for p in (CAMP / 'results').glob('*.json'):
        if p.stem in external_ids:
            raise ValueError('Independent execution already started; cannot declare a new pre-test freeze')
    for p in (CAMP / 'attempts').glob('*'):
        if p.name.split('_')[0] in external_ids:
            raise ValueError('Independent attempt already exists; freeze must precede the first attempt')


def feature_frame(records):
    # JSON checkpoints sort keys; exports must not depend on fresh/cache key order.
    audit = ['zones', 'steps', 'dt_hours', 'horizon_hours', 'lines', 'variables',
             'free_variables', 'binary_variables', 'free_binary_variables', 'active_constraints',
             'storage_power_mw', 'storage_energy_mwh', 'installed_storage_duration_hours',
             'actual_demand_peak_mw', 'actual_demand_valley_mw', 'definition',
             'load_seconds', 'build_seconds', 'descriptor_seconds', 'total_seconds']
    schema = ['group', 'population'] + ['real_' + i for i in IDS] + ['audit_' + k for k in audit]
    frame = pd.DataFrame(records)
    if set(frame.columns) != set(schema):
        raise ValueError('Unexpected descriptor export schema')
    return frame[schema]


def prepare():
    verify_historical_inputs()
    campaign = read(CAMP / 'manifest.json')
    if campaign['source_hashes'] != source_hashes():
        raise ValueError('Physical source changed since campaign preparation')
    hist = pd.read_parquet(RUN / 'scenarios.parquet')
    historical_demand = pd.read_parquet(RUN / 'analysis.parquet').drop_duplicates('group').set_index('group').demand_mwh.to_dict()
    cases = [dict(group=r.group, file=r.file, scenario_sha256=r.scenario_sha256, population='historical')
             for r in hist.itertuples()]
    cases += [dict(c, population='external') for c in campaign['entries']]
    extraction_signature = digest(dict(features=sha(PACKAGE / 'features.py'), physics=source_hashes()))
    records = []
    for i, case in enumerate(cases):
        if sha(case['file']) != case['scenario_sha256']:
            raise ValueError('Scenario changed: ' + case['group'])
        path = DEST / 'features' / (digest(case['group']) + '.json')
        identity = dict(group=case['group'], scenario_sha256=case['scenario_sha256'],
                        extraction_signature=extraction_signature)
        if path.exists():
            obj = read(path)
            if obj['identity'] != identity:
                raise ValueError('Stale descriptor checkpoint: ' + case['group'])
        else:
            features, diagnostics = extract(case['file'])
            obj = dict(identity=identity, values=features, diagnostics=diagnostics)
            put(path, obj)
            gc.collect()
        if case['population'] == 'historical':
            import math
            d = obj['diagnostics']
            realized_energy = obj['values']['A5'] * d['zones'] * 500. * d['horizon_hours']
            if not math.isclose(realized_energy, historical_demand[case['group']], rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('Realized demand differs from the archived dispatch instance: ' + case['group'])
        records.append(dict(group=case['group'], population=case['population'],
                            **{'real_' + k: v for k, v in obj['values'].items()},
                            **{'audit_' + k: v for k, v in obj['diagnostics'].items()}))
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            print('Constructed descriptors', i + 1, '/', len(cases), flush=True)
    frame = feature_frame(records)
    if frame.group.duplicated().any() or len(frame) != 1000:
        raise ValueError('Expected 600 historical and 400 independent cases')
    DEST.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DEST / 'features.parquet', index=False)
    frame.to_csv(DEST / 'feature_audit.csv', index=False)
    immutable(DEST / 'feature_manifest.json', dict(extraction_signature=extraction_signature,
        features_sha256=sha(DEST / 'features.parquet'), historical_cases=600, external_cases=400,
        analysis_manifest=sha(RUN / 'manifest.json'), campaign_manifest=sha(CAMP / 'manifest.json'),
        dictionary=DEFINITIONS, solver_called=False))


def load_frames():
    verify_historical_inputs()
    manifest = read(DEST / 'feature_manifest.json')
    if sha(DEST / 'features.parquet') != manifest['features_sha256']:
        raise ValueError('Descriptor table changed')
    if manifest['analysis_manifest'] != sha(RUN / 'manifest.json') or manifest['campaign_manifest'] != sha(CAMP / 'manifest.json'):
        raise ValueError('Descriptor provenance mismatch')
    expected = digest(dict(features=sha(PACKAGE / 'features.py'), physics=source_hashes()))
    if expected != manifest['extraction_signature']:
        raise ValueError('Descriptor extractor changed after extraction')
    history = pd.read_parquet(RUN / 'analysis.parquet')
    for ident, column in zip(IDS, COMPONENTS):
        history['norm_' + ident] = history[column]
        history['raw_' + ident] = history['raw_' + column]
    features = pd.read_parquet(DEST / 'features.parquet')
    history = history.merge(features[features.population == 'historical'], on='group', validate='many_to_one')
    if len(history) != 3000 or history.group.nunique() != 600:
        raise ValueError('Incomplete historical join')
    return history, features, config()


def columns(rep):
    p = protocol()
    if rep == 'legacy_normalized': return ['norm_' + i for i in IDS]
    if rep == 'legacy_raw': return ['raw_' + i for i in IDS]
    if rep == 'legacy_without_disputed': return ['raw_' + i for i in IDS if i not in p['removed_coordinates']]
    if rep == 'corrected_only': return [('real_' if i in p['corrected_coordinates'] else 'raw_') + i for i in IDS]
    if rep == 'realized': return ['real_' + i for i in IDS]
    if rep == 'realized_compact': return ['real_B1', 'real_B3', 'real_B4']
    if rep == 'size': return ['real_B1']
    if rep == 'model_size': return ['real_B1', 'real_B2', 'audit_lines', 'audit_variables', 'audit_free_binary_variables', 'audit_active_constraints']
    raise ValueError(rep)


def external_frame(features):
    from dataclasses import asdict
    from src.analysis.criticality_index import compute_criticality
    rows = []
    for case in read(CAMP / 'manifest.json')['entries']:
        if sha(case['file']) != case['scenario_sha256']:
            raise ValueError('External input changed')
        crit = compute_criticality(read(case['file']))
        raw = dict(asdict(crit.stress_metrics), **asdict(crit.hardness_metrics))
        normalized = dict(crit.stress_normalized, **crit.hardness_normalized)
        values = {**{'raw_' + i: raw[c] for i, c in zip(IDS, COMPONENTS)},
                  **{'norm_' + i: normalized[c] for i, c in zip(IDS, COMPONENTS)}}
        for b in config()['deadlines']:
            rows.append(dict(group=case['group'], batch=case['batch'], budget=b, **values))
    return pd.DataFrame(rows).merge(features[features.population == 'external'], on='group', validate='many_to_one')
