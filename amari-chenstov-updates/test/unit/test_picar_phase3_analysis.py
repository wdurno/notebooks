"""Development estimates respect exposure, retention and replica identities."""

import copy

import pytest

from numerical_experiment.picar_phase3.analysis import candidate, selection, forecast


def pilot(name="control",seed=1,fd=0.,*,easy=(.8,.75,.7),hard=(0.,.1,.2)):
    evaluations=[]
    for checkpoint,count,index in (("initial",0,0),("midpoint",320,1),("final",640,2)):
        for difficulty,successes,n in (("easy",easy,20),("hard",hard,10)):
            evaluations.append({"checkpoint":checkpoint,"difficulty":difficulty,"acquired_count":count,
                "rows":[{"seed":100+i,"success":i<round(n*successes[index])} for i in range(n)]})
    return {"settings":{"name":name,"fd_strength":fd,"ridge":.1},"seed":seed,"acquired_count":640,
        "evaluations":evaluations,"unit_path":f"{name}/{seed}",
        "timings":{"active_seconds":2400.,"evaluation_seconds":600.}}


def test_retention_uses_measured_midpoint_and_endpoint_not_training_success():
    result=candidate(pilot())
    assert result["old_gap_average"]==pytest.approx(.05)
    assert result["old_gap_endpoint"]==pytest.approx(.1)
    assert result["new_gain_endpoint"]==pytest.approx(.2)
    assert result["panel_resolution"]==.05
    assert result["checkpoints"]==3
    broken=pilot()
    del broken["evaluations"][2]["acquired_count"]
    with pytest.raises(ValueError,match="midpoint"):
        candidate(broken)


def test_panel_mismatch_unknown_outcome_or_repeated_seed_fails():
    for change in ("seed","unknown","duplicate"):
        value=pilot()
        rows=value["evaluations"][-1]["rows"]
        if change=="seed":
            rows[0]["seed"]=999
        elif change=="unknown":
            rows[0]["success"]=None
        else:
            rows[1]["seed"]=rows[0]["seed"]
        with pytest.raises(ValueError):
            candidate(value)


def test_retention_matching_requires_both_average_and_endpoint():
    controls=[pilot(seed=s) for s in (1,2)]
    fd=[pilot("fd",seed=s,fd=100.,easy=(.8,.8,.7),hard=(0.,.2,.3)) for s in (1,2)]
    result=selection(controls+fd,control_name="control")
    assert result["retention_matched"] and result["ready_to_propose_confirmation"]
    assert not result["next_phase_authorized"] and not result["sample_savings_available"]
    fd[0]["evaluations"][2]["rows"]=copy.deepcopy(fd[0]["evaluations"][0]["rows"])
    for row in fd[0]["evaluations"][2]["rows"]:
        row["success"]=False
    assert not selection(controls+fd,control_name="control")["retention_matched"]


def test_one_seed_cannot_certify_two_seed_recipe_and_control_does_not_switch_after_replication():
    values=[pilot("selected",seed=1,hard=(0.,.1,.1)),pilot("selected",seed=2,hard=(0.,0.,0.)),
            pilot("lucky",seed=1,hard=(0.,.5,.8)),pilot("fd",seed=1,fd=100.,hard=(0.,.2,.3))]
    result=selection(values,control_name="selected")
    assert result["selected_control"]=="selected"
    assert not result["ready_to_propose_confirmation"]
    assert not result["control_learning_observed"]


def test_forecast_uses_measured_processing_not_fd_gradient_probe_cost():
    controls=[pilot(seed=s) for s in (1,2)]
    fd=[pilot("fd",seed=s,fd=100.) for s in (1,2)]
    for row in fd:
        row["timings"]["active_seconds"]=4200.
    decision=selection(controls+fd,control_name="control")
    result=forecast(controls+fd,decision,phase3_seconds=10000.,prior_study_seconds=20000.)
    assert result["fd_processing_multiplier"]==2
    assert result["study_charged_seconds"]==30000
    assert result["remaining_to_20h_seconds"]==42000
    assert not result["next_phase_authorized"]
