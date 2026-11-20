"""Read sealed historical outputs without importing any frozen project module."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
DEFAULT_SOURCE = ROOT / 'research/three_papers/runs/v2_service_by_deadline'
DEFAULT_RUN = HOME / 'runs/weights_v2_variants'
COMPONENTS = (
    'A1_vre_penetration', 'A2_residual_volatility', 'A3_peak_to_valley',
    'A4_short_term_variability', 'A5_demand_scale', 'A6_peak_to_firm',
    'A7_inv_storage_power', 'A8_inv_storage_energy', 'A9_inv_thermal_flex',
    'A10_inv_dr_headroom', 'A11_trade_reliance', 'A12_inv_congestion',
    'B1_n_zones', 'B2_horizon', 'B3_binary_est', 'B4_thermal_density',
    'B5_min_gen_tight', 'B6_startup_intensity', 'B7_ramp_tightness',
    'B8_soc_tightness', 'B9_interconn_density', 'B10_network_hetero')
ORIGINAL_WEIGHTS = (1 / 24,) * 12 + (1 / 20,) * 10


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def seal(obj):
    body = {k: v for k, v in obj.items() if k != 'sha256'}
    return dict(body, sha256=digest(body))


def verify(obj):
    if not isinstance(obj, dict) or obj.get('sha256') != seal(obj)['sha256']:
        raise ValueError('Invalid sealed JSON')
    return obj


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writable(path):
    p = Path(path).resolve()
    if not p.is_relative_to((HOME / 'runs').resolve()):
        raise ValueError('All analysis outputs must remain in paper1_weights/runs')
    return p


def publish(path, payload):
    p = writable(path)
    if p.exists():
        if p.read_bytes() != payload:
            raise ValueError(f'Immutable output differs; use a new run directory: {p}')
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.rename(p)


def put(path, obj):
    publish(path, (json.dumps(obj, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, indent=2) + '\n').encode())


@contextlib.contextmanager
def lock(run):
    p = writable(Path(run) / '.writer.lock')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if p.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def sources():
    paths = sorted(HOME.glob('*.py')) + sorted((HOME / 'tests').glob('*.py'))
    paths += [HOME / 'protocol.json']
    return {p.relative_to(HOME).as_posix(): sha(p) for p in paths}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def config():
    cfg = read(HOME / 'protocol.json')
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg['deadlines_s'] != [5, 10, 30, 60, 120]:
        raise ValueError('Five separate historical deadlines are required')
    if min(cfg['outer_folds'], cfg['inner_folds']) < 2:
        raise ValueError('At least two folds are required')
    if not cfg['ridge_grid'] or any(not finite(v) or v < 0 for v in cfg['ridge_grid']):
        raise ValueError('Invalid ridge grid')
    for key in ('cost_gap_scale', 'ens_fraction_scale', 'reference_floor_eur',
                'co2_intensity_scale_t_per_mwh', 'constant_tolerance'):
        if not finite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('Invalid positive scale: ' + key)
    if cfg['non_delivery_score'] != 1:
        raise ValueError('Non-delivery is explicitly assigned score 1')
    if cfg['zero_ens_tolerance_mwh'] < 0 or cfg['bootstrap_samples'] < 0:
        raise ValueError('Invalid tolerance or bootstrap count')
    import re
    if not isinstance(cfg['policies'], dict) or not cfg['policies']:
        raise ValueError('At least one target policy is required')
    for name, policy in cfg['policies'].items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise ValueError('Invalid policy identifier')
        if set(policy) != {'ens', 'cost', 'time', 'co2'}:
            raise ValueError('Four policy entries required: ' + name)
        if any(not finite(v) or v < 0 for v in policy.values()) or not math.isclose(sum(policy.values()), 1):
            raise ValueError('Nonnegative policy weights must sum to one')
    carbon = cfg['carbon']
    if carbon['scope'] != 'direct_internal_thermal' or carbon['import_factor_t_per_mwh'] is not None:
        raise ValueError('This version reports internal direct thermal CO2 only')
    if not finite(carbon['thermal_factor_t_per_mwh']) or carbon['thermal_factor_t_per_mwh'] <= 0:
        raise ValueError('Invalid thermal emission factor')


def select_rows(all_rows, cfg):
    rows = [dict(r) for r in all_rows if r['method'] == cfg['method']
            and r['repetition'] == cfg['repetition'] and r['budget'] in cfg['deadlines_s']]
    seen, groups, families, by_key = set(), {}, {}, {}
    for r in rows:
        pair = (r['key'], r['budget'])
        if pair in seen:
            raise ValueError('Duplicate primary observation')
        seen.add(pair)
        if r['partition'] != 'historical_development' or r['condition'] != 'historical' or r['observed'] is not True:
            raise ValueError('Only completed original historical observations are allowed')
        if r.get('feasible') not in (True, False) or not isinstance(r.get('feasible'), bool):
            raise ValueError('Unknown delivery status')
        if r['key'] in groups and groups[r['key']] != r['group']:
            raise ValueError('Unstable scenario group')
        groups[r['key']] = r['group']
        families.setdefault(r['family'], set()).add(r['key'])
        for c in COMPONENTS:
            if not finite(r.get(c)) or not -1e-9 <= r[c] <= 1 + 1e-9:
                raise ValueError('Missing/non-normalized component: ' + c)
        values = tuple(r[c] for c in COMPONENTS)
        if r['key'] in by_key and by_key[r['key']] != values:
            raise ValueError('Pre-solve components change across deadlines')
        by_key[r['key']] = values
        if not finite(r.get('exit_s')) or r['exit_s'] < 0:
            raise ValueError('Unknown actual exit time')
        if not finite(r.get('reference_cost')):
            raise ValueError('Missing common reference cost')
        if not finite(r.get('demand_mwh')) or r['demand_mwh'] <= 0:
            raise ValueError('Missing positive demand energy')
        if r['feasible']:
            if r['exit_s'] > r['budget']:
                raise ValueError('An accepted observation misses its deadline')
            if not finite(r.get('cost')) or not finite(r.get('ens_mwh')):
                raise ValueError('Accepted dispatch missing cost or ENS')
            if not -cfg['zero_ens_tolerance_mwh'] <= r['ens_mwh'] <= r['demand_mwh'] + cfg['zero_ens_tolerance_mwh']:
                raise ValueError('ENS outside supported physical range')
    n = cfg['expected_scenarios']
    if len(groups) != n or len(set(groups.values())) != n or len(rows) != n * len(cfg['deadlines_s']):
        raise ValueError('Incomplete independent scenario/deadline grid')
    if len(families) != cfg['expected_families'] or any(len(v) != cfg['expected_per_family'] for v in families.values()):
        raise ValueError('Unexpected family support')
    return sorted(rows, key=lambda r: (r['group'], r['budget']))


def observation_identity(row, freeze):
    identity = {k: row[k] for k in ('key', 'method', 'budget', 'repetition', 'seed')}
    identity['manifest_hash'] = freeze['manifest_hash']
    calibrated = freeze['calibration']['reserve_s']
    reserve = min(.5 * float(row['budget']), max(.05, calibrated * min(1., float(row['budget']) / 120.)))
    if reserve != calibrated:
        amendment = freeze['reserve_amendment']
        if amendment['kind'] != 'budget_scaled_finalization_reserve_v1':
            raise ValueError('Unsupported reserve policy')
        identity.update(reserve_policy=amendment['kind'], timing_revision=amendment['sha256'], reserve_s=reserve)
    return identity


def dispatch_carbon(dispatch, dt, demand, ens, cfg):
    """Territorial thermal CO2; imports exported as unknown-footprint diagnostics."""
    if not finite(dt) or dt <= 0:
        raise ValueError('Invalid dispatch time step')
    zones, periods = dispatch['zones'], dispatch['periods']
    if not zones or not periods:
        raise ValueError('Empty dispatch')
    def energy(name, *, zonal):
        values = dispatch['values'][name]
        if not isinstance(values, (list, tuple)):
            raise ValueError('Wrong dispatch shape: ' + name)
        if zonal:
            # Generation is stored as (zone, period).
            if len(values) != len(zones) or any(
                    not isinstance(row, (list, tuple)) or len(row) != len(periods)
                    for row in values):
                raise ValueError('Wrong dispatch shape: ' + name)
            powers = (value for row in values for value in row)
        else:
            # Imports/exports are system-wide (period,) arrays, counted once.
            if len(values) != len(periods) or any(isinstance(value, (list, tuple)) for value in values):
                raise ValueError('Wrong dispatch shape: ' + name)
            powers = iter(values)
        total = 0.
        for value in powers:
            if not finite(value) or value < -1e-3:
                raise ValueError('Invalid dispatch power: ' + name)
            total += max(0., value)
        return total * dt
    thermal = energy('p_thermal', zonal=True)
    imports, exports = energy('net_import', zonal=False), energy('net_export', zonal=False)
    direct = thermal * cfg['carbon']['thermal_factor_t_per_mwh']
    served = max(0., demand - max(0., ens))
    return dict(thermal_mwh=thermal, co2_direct_t=direct, imports_mwh=imports,
                exports_mwh=exports, served_mwh=served,
                co2_t_per_served_mwh=direct / served if served > 1e-12 else None,
                carbon_scope=cfg['carbon']['scope'], import_carbon_known=False)


def bounded_ratio(value, anchor):
    return value / (value + anchor) if value > 0 else 0.


def target(row, cfg, policy):
    if not row['feasible']:
        # No fabricated ENS/CO2 is assigned to a killed or invalid process.
        return dict(score=cfg['non_delivery_score'], delivery_failure=True,
                    time_term=None, cost_term=None, ens_term=None, co2_term=None)
    signed_gap = (row['cost'] - row['reference_cost']) / max(abs(row['reference_cost']), cfg['reference_floor_eur'])
    ens_fraction = max(0., row['ens_mwh'] - cfg['zero_ens_tolerance_mwh']) / row['demand_mwh']
    terms = dict(time=row['exit_s'] / row['budget'],
                 cost=bounded_ratio(max(0., signed_gap), cfg['cost_gap_scale']),
                 ens=bounded_ratio(ens_fraction, cfg['ens_fraction_scale']))
    intensity = row.get('co2_t_per_served_mwh')
    if intensity is None:
        if row.get('served_mwh', 0.) > 1e-12:
            raise ValueError('CO2 intensity missing for a delivered dispatch')
        terms['co2'] = 1.  # Zero service cannot obtain a low-carbon reward.
    else:
        if not finite(intensity) or intensity < 0:
            raise ValueError('Invalid CO2 intensity')
        terms['co2'] = bounded_ratio(intensity, cfg['co2_intensity_scale_t_per_mwh'])
    score = sum(policy[k] * terms[k] for k in policy)
    if not 0 <= score <= 1 + 1e-12:
        raise ValueError('Target outside [0,1]')
    return dict(score=min(1., score), delivery_failure=False,
                **{k + '_term': v for k, v in terms.items()})


def extract_case(task):
    case, rows, campaign, freeze, cfg = task
    scenario_path = Path(case['file'])
    if sha(scenario_path) != case['scenario_sha256']:
        raise ValueError('Historical scenario changed')
    scenario = read(scenario_path)
    dt = scenario['dt_minutes'] / 60.
    if not math.isclose(dt, case['dt_hours']) or len(rows) != len(cfg['deadlines_s']):
        raise ValueError('Scenario time support mismatch')
    refpath = Path(campaign) / 'references' / case['family'] / (case['scenario_id'] + '.json')
    ref = verify(read(refpath))
    if ref['sha256'] != freeze['references'][str(refpath)] or ref['identity']['key'] != case['key']:
        raise ValueError('Reference provenance mismatch')
    rr = ref['result']
    if not rr.get('valid') or rr.get('physics') != freeze['config']['physics']:
        raise ValueError('Invalid reference dispatch')
    rv = rr['validation']
    ref_carbon = dispatch_carbon(rr['dispatch'], dt, rows[0]['demand_mwh'], rv['ens_mwh'], cfg)
    sources_read = {str(scenario_path): case['scenario_sha256'], str(refpath): ref['sha256']}
    prepared = []
    for row in rows:
        ident = observation_identity(row, freeze)
        path = Path(campaign) / 'experiments' / (digest(ident) + '.json')
        obj = verify(read(path))
        if str(path) not in freeze['records'] or obj['sha256'] != freeze['records'][str(path)] or obj['identity'] != ident:
            raise ValueError('Historical result provenance mismatch')
        if obj.get('complete') is not True or bool(obj.get('success')) != row['feasible']:
            raise ValueError('Historical delivery status mismatch')
        if not math.isclose(obj['exit_s'], row['exit_s'], abs_tol=1e-9):
            raise ValueError('Historical exit time mismatch')
        sources_read[str(path)] = obj['sha256']
        if not math.isclose(rv['cost'], row['reference_cost'], rel_tol=1e-9, abs_tol=1e-3):
            raise ValueError('Reference cost mismatch')
        if bool(rr.get('proven_optimal')) != row['reference_certified']:
            raise ValueError('Reference certification mismatch')
        out = {k: row[k] for k in COMPONENTS + ('key', 'group', 'family', 'budget', 'feasible',
                'exit_s', 'cost', 'ens_mwh', 'demand_mwh', 'reference_cost', 'reference_certified')}
        out.update(reference_co2_direct_t=ref_carbon['co2_direct_t'],
                   reference_ens_mwh=rv['ens_mwh'],
                   co2_price_eur_per_t=scenario['econ_policy']['co2_price'], dt_hours=dt,
                   reference_is_carbon_optimum=False)
        if row['feasible']:
            if obj.get('forced') or not obj.get('all_processes_exited') or obj.get('returncode') != 0:
                raise ValueError('Accepted result violates process-exit contract')
            result = obj['result']
            dispatch = result['dispatch']
            val = result['validation']
            if not val.get('valid') or dispatch.get('physics') != freeze['config']['physics']:
                raise ValueError('Invalid saved dispatch')
            if len(dispatch['periods']) != case['periods']:
                raise ValueError('Historical dispatch horizon mismatch')
            for key in ('cost', 'ens_mwh'):
                if not math.isclose(val[key], row[key], rel_tol=1e-9, abs_tol=1e-3):
                    raise ValueError('Dispatch/' + key + ' mismatch')
            out.update(dispatch_carbon(dispatch, dt, row['demand_mwh'], row['ens_mwh'], cfg))
            out['cost_ens_penalty_eur'] = val['cost_components']['unserved_energy']
            out['cost_co2_charge_eur'] = out['co2_direct_t'] * out['co2_price_eur_per_t']
        else:
            out.update({k: None for k in ('thermal_mwh', 'co2_direct_t', 'imports_mwh',
                       'exports_mwh', 'served_mwh', 'co2_t_per_served_mwh',
                       'cost_ens_penalty_eur', 'cost_co2_charge_eur')})
            out.update(carbon_scope=cfg['carbon']['scope'], import_carbon_known=False)
        out['signed_relative_cost_gap'] = ((out['cost'] - out['reference_cost']) /
            max(abs(out['reference_cost']), cfg['reference_floor_eur'])) if row['feasible'] else None
        out['targets'] = {name: target(out, cfg, policy) for name, policy in cfg['policies'].items()}
        prepared.append(out)
    return dict(rows=prepared, sources_read=sources_read)


def prepare(run, source, workers):
    from .parallel import bounded_map
    run, source, cfg = writable(run), Path(source).resolve(), config()
    if (run / 'manifest.json').exists():
        manifest, _ = load_prepared(run)
        if manifest['source_run'] != str(source):
            raise ValueError('Source run changed')
        print('Existing preparation verified.', flush=True)
        return
    paths = [source / 'freeze.json', source / 'main_rows.json', source / 'routing/model_receipt.json']
    freeze, main, router = [verify(read(p)) for p in paths]
    if main['freeze'] != freeze['sha256'] or router['freeze'] != freeze['sha256'] or router['input_sha256'] != sha(paths[1]):
        raise ValueError('Extraction/router/freeze mismatch')
    campaign = (ROOT / freeze['config']['campaign']).resolve()
    manifest_path = campaign / 'manifest.json'
    historical = verify(read(manifest_path))
    if digest(historical) != freeze['manifest_hash']:
        raise ValueError('Historical cohort changed')
    # Check the factual emission-factor source, without importing model code.
    factor_source = ROOT / 'src/milp/defaults.py'
    if sha(factor_source) != freeze['sources']['src/milp/defaults.py']:
        raise ValueError('Historical thermal defaults changed')
    source_text = factor_source.read_text(encoding='utf-8-sig')
    import re
    match = re.search(r'emission_rate_t_per_mwh:\s*float\s*=\s*([0-9.]+)', source_text)
    if not match or float(match.group(1)) != cfg['carbon']['thermal_factor_t_per_mwh']:
        raise ValueError('Configured direct factor differs from the historical model')
    selected = select_rows(main['rows'], cfg)
    case_map = {c['key']: c for c in historical['cohort']}
    grouped = {}
    for row in selected:
        grouped.setdefault(row['key'], []).append(row)
    def case_task(key):
        case = case_map[key]
        refpath = campaign / 'references' / case['family'] / (case['scenario_id'] + '.json')
        record_paths = [campaign / 'experiments' / (digest(observation_identity(r, freeze)) + '.json')
                        for r in grouped[key]]
        # Send only six relevant hashes per case to a worker, not the 18,480-file inventory.
        local_freeze = dict(manifest_hash=freeze['manifest_hash'],
            config={'physics': freeze['config']['physics']},
            calibration={'reserve_s': freeze['calibration']['reserve_s']},
            reserve_amendment={k: freeze['reserve_amendment'][k] for k in ('kind', 'sha256')},
            references={str(refpath): freeze['references'][str(refpath)]},
            records={str(p): freeze['records'][str(p)] for p in record_paths})
        return case, grouped[key], str(campaign), local_freeze, cfg
    tasks = (case_task(k) for k in sorted(grouped))
    result_rows, evidence = [], {}
    for index, value in enumerate(bounded_map(extract_case, tasks, workers), 1):
        result_rows.extend(value['rows']); evidence.update(value['sources_read'])
        if index % 25 == 0:
            print(f'Dispatch extraction: {index}/{len(grouped)} scenarios', flush=True)
    result_rows.sort(key=lambda r: (r['group'], r['budget']))
    data = seal(dict(rows=result_rows, cfg=cfg, freeze=freeze['sha256']))
    evidence_obj = seal(dict(sources=evidence,
        hash_kind='scenario files: SHA256 bytes; dispatch/reference files: verified canonical sealed SHA256'))
    put(run / 'data.json', data)
    put(run / 'dispatch_provenance.json', evidence_obj)
    paths += [manifest_path, factor_source]
    import importlib.metadata
    environment = {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')}
    put(run / 'manifest.json', seal(dict(schema=cfg['schema'], source_run=str(source),
        config=cfg, sources=sources(), inputs={str(p): sha(p) for p in paths},
        data_sha256=data['sha256'], dispatch_provenance_sha256=evidence_obj['sha256'],
        environment=environment, observations=len(result_rows), groups=len(grouped),
        note='Extraction only: existing valid dispatches, no physical re-solve or router import.')))
    print(f'Prepared {len(result_rows)} observations; no regression fitted yet.', flush=True)


def load_prepared(run):
    run = writable(run)
    manifest = verify(read(run / 'manifest.json'))
    if manifest['sources'] != sources() or manifest['config'] != config():
        raise ValueError('Analysis code/config changed; choose a new run directory')
    for p, expected in manifest['inputs'].items():
        if sha(p) != expected:
            raise ValueError('Sealed metadata input changed: ' + p)
    data = verify(read(run / 'data.json'))
    evidence = verify(read(run / 'dispatch_provenance.json'))
    if data['sha256'] != manifest['data_sha256'] or evidence['sha256'] != manifest['dispatch_provenance_sha256']:
        raise ValueError('Prepared snapshot changed')
    return manifest, data
