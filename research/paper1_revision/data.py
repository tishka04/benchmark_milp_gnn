"""Unified, provenance checked historical tables; no physical solver calls."""
from __future__ import annotations
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from research.paper1_weights.data import COMPONENTS, ORIGINAL_WEIGHTS, digest, sha, verify, target
from research.paper1_weights.model import arrays, group_folds

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
RUN = HOME / 'runs/revision_v1'
WEIGHTS = ROOT / 'research/paper1_weights/runs/weights_v2_variants'
COMP = ROOT / 'research/paper1_components/runs/components_v1'
SOURCE = ROOT / 'research/three_papers/runs/v2_service_by_deadline'
CAMPAIGN = ROOT / 'outputs/thermal_transitions_v1/operational_exit_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HOME):
        raise ValueError('Revision outputs must remain in their own directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def config():
    return read(HOME / 'protocol.json')


def load():
    return pd.read_parquet(RUN/'analysis.parquet'), config()


def prepare():
    cfg = config()
    weighted = verify(read(WEIGHTS/'data.json'))
    components = verify(read(COMP/'data.json'))
    all_runs = verify(read(SOURCE/'main_rows.json'))
    manifest = read(CAMPAIGN/'manifest.json')
    wd, cd, rd = pd.DataFrame(weighted['rows']), pd.DataFrame(components['rows']), pd.DataFrame(all_runs['rows'])
    assert len(wd) == 3000 and wd.group.nunique() == 600
    # Join by identities, never by row position.
    base = wd.drop(columns='targets').merge(cd.drop(columns=[c for c in wd if c in cd and c not in ('group','budget')]),
                                           on=['group','budget'], validate='one_to_one')
    for p in cfg['policies']:
        base[p] = [target(r, weighted['cfg'], weighted['cfg']['policies'][p])['score'] for r in base.to_dict('records')]
        expected = wd.set_index(['group','budget']).targets.map(lambda t: t[p]['score'])
        assert np.allclose(base.set_index(['group','budget'])[p].sort_index(), expected.sort_index(), atol=1e-14)
    base['F'] = base.feasible.astype(int)
    base['E'] = base.economic_5.astype(int)
    base['ES'] = base.economic_5_ens.astype(int)
    data = arrays(weighted['rows'], cfg['policies'][0])
    folds = group_folds(data, 5, cfg['historical_fold_seed'], 'outer')
    mapping = dict(zip(data['group'], folds))
    base['outer_fold_binary_legacy'] = base.outer_fold
    base['outer_fold'] = base.group.map(mapping).astype(int)
    splits = {'outer': {g:int(f) for g,f in mapping.items()}, 'inner':{}, 'lofo':{}}
    for f in range(5):
        train = {k:v[folds != f] for k,v in data.items()}
        ff = group_folds(train,3,cfg['historical_fold_seed'],f'inner-{f}')
        splits['inner'][str(f)] = {g:int(i) for g,i in zip(train['group'],ff)}
    for family in sorted(base.family.unique()):
        splits['lofo'][family] = {'test': sorted(base.loc[base.family==family,'group'].unique()),
                                  'inner_families': sorted(set(base.family)-{family})}
    RUN.mkdir(parents=True,exist_ok=True)
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in [WEIGHTS/'data.json',COMP/'data.json',SOURCE/'main_rows.json',CAMPAIGN/'manifest.json']}
    references, scenarios = [], []
    scenario_rows = base.drop_duplicates('group').set_index('group')
    for case in manifest['cohort']:
        path = Path(case['file'])
        assert sha(path)==case['scenario_sha256']
        scenario = read(path)
        row = scenario_rows.loc[case['key']]
        scenarios.append(dict(group=case['key'], parent_group=case['key'], family=case['family'],
            file=str(path), scenario_sha256=case['scenario_sha256'], parameters_json=json.dumps(scenario,sort_keys=True),
            n_zones=int(row.n_zones), n_steps=int(row.n_steps), n_lines=int(row.n_lines),
            actual_variables=None, actual_binaries=None, actual_constraints=None,
            **{c:float(row[c]) for c in COMPONENTS}, **{'raw_'+c:float(row['raw_'+c]) for c in COMPONENTS}))
        refpath = CAMPAIGN/'references'/case['family']/(case['scenario_id']+'.json')
        ref = verify(read(refpath)); result = ref['result']
        inputs[str(refpath.relative_to(ROOT))]=sha(refpath)
        check=result.get('validation',{})
        costs=result.get('cost_components',check.get('cost_components',{}))
        references.append(dict(group=case['key'],cost=float(row.reference_cost),
            lower_bound=result.get('bound'), certified=bool(result.get('proven_optimal')),
            valid=bool(result.get('valid')),solver_seconds=result.get('solver_limit'),
            source=str(refpath),source_sha256=sha(refpath),version='historical_600s',
            cost_components_json=json.dumps(costs,sort_keys=True)))
    pd.DataFrame(scenarios).to_parquet(RUN/'scenarios.parquet',index=False)
    rd.to_parquet(RUN/'runs.parquet',index=False)
    pd.DataFrame(references).to_parquet(RUN/'references.parquet',index=False)
    base.to_parquet(RUN/'analysis.parquet',index=False)
    put(RUN/'splits.json',splits)
    import scipy, sklearn
    put(RUN/'manifest.json',dict(schema=cfg['schema'],config=cfg,inputs=inputs,
        tables={name:sha(RUN/name) for name in ['scenarios.parquet','runs.parquet','references.parquet','analysis.parquet']},splits_sha256=sha(RUN/'splits.json'),
        environment=dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
            numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        missing_fields={'actual_variables':'not recorded in historical analysis', 'actual_constraints':'not recorded',
                        'stage_times':'only compute, preload, finalization and exit available in historical rows'},
        analysis_status='development',long_resolutions_authorized=False))
    reproduce()


def reproduce():
    """Recompute every existing reported loss from its saved held-out predictions."""
    wp=pd.read_csv(WEIGHTS/'report/predictions.csv')
    ws=pd.read_csv(WEIGHTS/'report/prediction_scores.csv')
    errors=[]
    for r in ws.itertuples():
        q=wp[(wp.policy==r.policy)&(wp.model==r.model)]
        if pd.notna(r.deadline): q=q[q.budget==r.deadline]
        if r.reference_subset=='certified_only': q=q[q.reference_certified]
        if r.reference_subset not in ('all','certified_only'): raise ValueError(r.reference_subset)
        errors.append(abs(float(np.mean((q.predicted-q.observed)**2))-r.mse))
    bp=pd.read_csv(COMP/'report/tables/predictions.csv')
    bs=pd.read_csv(COMP/'report/tables/prediction_scores.csv')
    for r in bs.itertuples():
        q=bp[(bp.endpoint==r.endpoint)&(bp.budget==r.budget)&(bp.family_control==r.family_control)&(bp.description==r.description)]
        errors.append(abs(float(np.mean((q.p-q.y)**2))-r.brier))
    assert max(errors)<1e-12
    put(RUN/'reproduction.json',dict(weight_score_cells=len(ws),binary_score_cells=len(bs),max_absolute_loss_error=max(errors),
        scope='Full reproduction of reported MSE/Brier from saved predictions; new fits additionally checked against historical baselines.'))


def fingerprint():
    m=read(RUN/'manifest.json')
    for name,h in m['tables'].items():
        if sha(RUN/name)!=h: raise ValueError('Analysis table changed: '+name)
    return digest({'manifest':sha(RUN/'manifest.json'),'protocol':config(),
                   'code':{p.name:sha(p) for p in HOME.glob('*.py')}})
