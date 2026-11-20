import pandas as pd
from research.paper1_weights.data import config
from research.paper1_revision.future_reports import reassess_references, timing_rows


def test_improved_reference_changes_labels_and_continuous_cost():
    row = dict(group='a', family='test', budget=10, feasible=True, cost=104.,
               reference_cost=100., ens_mwh=0., demand_mwh=10., exit_s=2.,
               served_mwh=10., co2_t_per_served_mwh=.1)
    refs = pd.DataFrame([dict(group='a', cost=100.)]).set_index('group')
    upgrade = dict(success=True, result=dict(valid=True, validation=dict(cost=90.), bound=89.))
    changed = reassess_references(pd.DataFrame([row]), refs, {'a': upgrade}, config()).set_index('target')
    assert changed.loc['E', 'before'] == 1 and changed.loc['E', 'after'] == 0
    assert changed.loc['equal4', 'after'] > changed.loc['equal4', 'before']
    upgrade['success'] = False
    unchanged = reassess_references(pd.DataFrame([row]), refs, {'a': upgrade}, config())
    assert (unchanged.before == unchanged.after).all()


def test_interrupted_trace_keeps_completed_stages_only(tmp_path):
    directory = tmp_path / 'one_abc'
    directory.mkdir()
    (directory / 'stages.jsonl').write_text(
        '{"stage":"scenario_processing","seconds":0.2}\n{"stage":', encoding='utf-8')
    obj = dict(job=dict(kind='timing_repeat', job_id='one', key='a', budget=5, repetition=0),
               result=dict(success=False, status='forced_timeout', exit_s=5.01, preload_s=.1))
    frame = timing_rows([obj], tmp_path)
    assert frame.stage_scenario_processing.iloc[0] == .2
    assert 'stage_solver_execution' not in frame
    assert frame.completed_stage_count.iloc[0] == 1
