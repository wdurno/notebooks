import pytest

from numerical_experiment.picar.schemas import TransitionRecord, metric_row


def example(**kwargs):
    values = dict(transition_id="t0", episode_id="e0", arrival_index=0,
                  observation_ref="a", next_observation_ref="b", context_refs=("a",),
                  instruction="stop", action_logits=(0.,) * 7,
                  action_probabilities=(.125,) * 8, requested_control={}, executed_control={},
                  baseline=0., model_version=0, elapsed_s=.5, reward=-.02, discount=.99,
                  terminated=True, truncated=False, end_reason="deadline")
    return TransitionRecord(**(values | kwargs))


def test_transition_label_is_one_way():
    pending = example()
    assert not pending.ready
    ready = pending.labeled(-.02, 3)
    assert ready.ready
    assert TransitionRecord.from_dict(ready.to_dict()) == ready
    with pytest.raises(ValueError):
        ready.labeled(1., 4)
    with pytest.raises(ValueError):
        example(terminated=False, truncated=True).labeled(0., 1)


def test_metrics_are_finite_and_strict():
    assert metric_row(condition="stm_fd")["schema_version"] == 1
    with pytest.raises(ValueError):
        metric_row(typo=1)
    with pytest.raises(ValueError):
        metric_row(data_nll=float("nan"))
