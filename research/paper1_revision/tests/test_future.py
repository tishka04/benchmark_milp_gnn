from research.paper1_revision.future_analysis import bounded_economic_status

def test_negative_and_zero_crossing_objective_bounds():
    assert bounded_economic_status(-96,-100,-98)=='certain_success'
    assert bounded_economic_status(-90,-100,-98)=='certain_failure'
    assert bounded_economic_status(1.7,0,1.5)=='indeterminate'
    assert bounded_economic_status(2.1,0,1.5)=='certain_failure'
    assert bounded_economic_status(-100,None,-98)=='indeterminate'
