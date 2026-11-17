"""CPU arithmetic checks for the bounded Phase 3 Fix 3 reward contrast."""

import math

import pytest
import torch

from numerical_experiment.picar_reward_repair import math as rm
from src.continual import picar_math as pm


CONDITIONS = ("width6_linear", "width6_signed_square")


def recorded(dt=(.5, .75, .125), q=(0., 10., 5.), bonus=10., dtype=torch.float64):
    return {"elapsed": torch.tensor(dt, dtype=dtype), "scores": torch.tensor(q, dtype=dtype),
            "bonuses": torch.tensor([0.] * (len(dt) - 1) + [bonus], dtype=dtype),
            "terminated": torch.tensor([False] * (len(dt) - 1) + [True]),
            "edges": torch.linspace(-6., 12., 162, dtype=dtype)}


@pytest.mark.parametrize("condition,exponent,scale", [(CONDITIONS[0], 1, 1.), (CONDITIONS[1], 2, 10.)])
@pytest.mark.parametrize("bonus", [10., -5., -1.])
def test_score_transform_precedes_rate_and_terminal_is_transformed_separately(condition, exponent, scale, bonus):
    inputs = recorded(bonus=bonus)
    output = rm.episode_labels(**inputs, condition=condition)
    expected = torch.tensor([-.02, .75 * (.01 * 10 ** exponent - .04),
                             .125 * (.01 * 5 ** exponent - .04) + bonus * abs(bonus) ** (exponent - 1)], dtype=torch.float64)
    assert rm.condition_power(condition) == exponent
    assert rm.code_scale(condition) == scale
    assert set(output) == {"raw_stage_rewards", "coded_stage_rewards", "raw_returns", "coded_returns", "bins", "discounts"}
    torch.testing.assert_close(output["raw_stage_rewards"], expected)
    torch.testing.assert_close(output["coded_stage_rewards"], expected / scale)
    torch.testing.assert_close(output["coded_returns"], output["raw_returns"] / scale)
    discounts = .99 ** (inputs["elapsed"] / .5)
    torch.testing.assert_close(output["discounts"], discounts)
    torch.testing.assert_close(output["raw_returns"], torch.stack([
        expected[0] + discounts[0] * expected[1] + discounts[0] * discounts[1] * expected[2],
        expected[1] + discounts[1] * expected[2], expected[2]]))
    assert output["bins"].dtype == torch.int64


def test_signed_transform_zero_and_negative_signs():
    values = torch.tensor([-5., -1., 0., 1., 10.])
    torch.testing.assert_close(rm.score_transform(values, CONDITIONS[1]), torch.tensor([-25., -1., 0., 1., 100.]))
    with pytest.raises(ValueError, match="condition"):
        rm.condition_power("new_reward")


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("q", [0., 3., 10.])
def test_interval_splitting_preserves_undiscounted_reward(condition, q):
    whole = rm.episode_labels(**recorded((1.,), (q,)), condition=condition, gamma=1.)
    split = rm.episode_labels(**recorded((.25, .625, .125), (q,) * 3), condition=condition, gamma=1.)
    torch.testing.assert_close(whole["raw_returns"][0], split["raw_returns"][0])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_complete_concatenated_episodes_reset_time_budget_and_discount(condition):
    inputs = recorded((20., 20.), (10., 0.))
    inputs["terminated"][:] = True
    inputs["bonuses"][:] = torch.tensor([10., -5.])
    output = rm.episode_labels(**inputs, condition=condition)
    expected = [11.2, -5.8] if condition == CONDITIONS[0] else [119.2, -25.8]
    torch.testing.assert_close(output["raw_returns"], torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("q", [0., 10.])
@pytest.mark.parametrize("bonus", [-5., -1., 10.])
@pytest.mark.parametrize("gamma", [.7, .99, 1.])
def test_support_bounds_without_clipping(condition, q, bonus, gamma):
    inputs = recorded((.5,) * 39 + (.25, .25), (q,) * 41, bonus)
    output = rm.episode_labels(**inputs, condition=condition, gamma=gamma)
    lower, upper = (-6., 12.) if condition == CONDITIONS[0] else (-26., 120.)
    assert bool(((output["raw_returns"] >= lower) & (output["raw_returns"] <= upper)).all())
    assert bool(((output["coded_returns"] >= -6) & (output["coded_returns"] <= 12)).all())
    assert bool(((output["bins"] >= 0) & (output["bins"] < 161)).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("condition", CONDITIONS)
def test_detached_labels_preserve_input_and_accumulation_dtype(dtype, condition):
    inputs = recorded(dtype=dtype)
    # Edges themselves must retain the fixed support at usable precision.
    inputs["edges"] = torch.linspace(-6., 12., 162, dtype=torch.float32)
    for key in ("elapsed", "scores", "bonuses"):
        inputs[key].requires_grad_()
    before = {k: v.clone() for k, v in inputs.items()}
    output = rm.episode_labels(**inputs, condition=condition)
    assert output["raw_returns"].dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert all(not v.requires_grad for v in output.values())
    for key in inputs:
        torch.testing.assert_close(inputs[key], before[key])


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("key,value", [
    ("elapsed", [0., .5, .5]), ("elapsed", [-.1, .5, .5]), ("elapsed", [7., 7., 7.]),
    ("elapsed", [20.0001, .5, .5]), ("elapsed", [math.nan, .5, .5]),
    ("scores", [-1e-8, 1., 2.]), ("scores", [1., 2., 10.00000001]),
    ("scores", [1., 2., math.nan]), ("scores", [1., 2., math.inf]),
    ("bonuses", [0., 0., 0.]), ("bonuses", [0., 0., 100.]),
    ("bonuses", [0., 0., -25.]), ("bonuses", [10., 0., 10.]),
    ("bonuses", [0., 0., math.inf]), ("terminated", [False, False, False]),
])
def test_invalid_observations_never_get_complete_labels(condition, key, value):
    inputs = recorded()
    inputs[key] = torch.tensor(value, dtype=inputs[key].dtype)
    with pytest.raises(ValueError):
        rm.episode_labels(**inputs, condition=condition)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_validation_before_dtype_rounding_and_bad_shapes(condition):
    inputs = recorded(dtype=torch.float32)
    inputs["scores"] = torch.tensor([0., 0., 10. + 1e-8], dtype=torch.float64)
    with pytest.raises(ValueError):
        rm.episode_labels(**inputs, condition=condition)
    for key in ("elapsed", "scores", "bonuses", "terminated"):
        inputs = recorded()
        inputs[key] = inputs[key][:1]
        with pytest.raises(ValueError):
            rm.episode_labels(**inputs, condition=condition)
    for key in ("elapsed", "scores", "bonuses", "terminated"):
        inputs = recorded()
        inputs[key] = inputs[key].long()
        with pytest.raises(ValueError):
            rm.episode_labels(**inputs, condition=condition)


@pytest.mark.parametrize("edges", [torch.linspace(-6., 12., 161), torch.linspace(-26., 120., 162),
                                   torch.linspace(-6., 12., 162).flip(0), torch.zeros(162),
                                   torch.full((162,), math.nan)])
def test_incompatible_edges_fail(edges):
    inputs = recorded()
    inputs["edges"] = edges
    with pytest.raises(ValueError, match="edges|bins"):
        rm.episode_labels(**inputs, condition=CONDITIONS[0])


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("gamma,period", [(0., .5), (1.01, .5), (math.nan, .5), (.99, 0.)])
def test_invalid_discount_parameters(condition, gamma, period):
    with pytest.raises(ValueError):
        rm.episode_labels(**recorded(), condition=condition, gamma=gamma, nominal_period=period)


def test_raw_coded_advantage_and_categorical_weights_agree_in_both_units():
    temperature = 2.8375266913896
    for condition in CONDITIONS:
        inputs = recorded()
        labels = rm.episode_labels(**inputs, condition=condition)
        scale = rm.code_scale(condition)
        baseline = torch.tensor([-.7, .2, 1.], dtype=torch.float64)
        torch.testing.assert_close((labels["raw_returns"] - baseline * scale) / (scale * temperature),
                                   (labels["coded_returns"] - baseline) / temperature)
        coded_bins = pm.return_bin_values(inputs["edges"])
        raw_design = pm.LikelihoodDesign(coded_bins * scale, baseline * scale,
                                        temperature * scale, torch.eye(7, dtype=torch.float64))
        coded_design = pm.LikelihoodDesign(coded_bins, baseline, temperature, torch.eye(7, dtype=torch.float64))
        torch.testing.assert_close(raw_design.log_weights, coded_design.log_weights)
        torch.testing.assert_close(raw_design.weights, coded_design.weights)


def test_cpu_scripted_audit_reports_loitering_and_quantization_without_hiding_ties():
    audit = rm.reward_arithmetic_audit()
    for condition in CONDITIONS:
        rows = audit[condition]["scenarios"]
        assert audit[condition]["timely_success_beats_loitering"]
        assert rows["loiter_timeout"]["raw_returns"] > rows["disappearance"]["raw_returns"]
        assert audit[condition]["approach_log_weight_difference"] > 0
        assert audit[condition]["approach_bin_tie"] == (rows["approach_low"]["bin"] == rows["approach_high"]["bin"])
    # The signed square compresses low visual scores in common coded units.
    assert audit[CONDITIONS[1]]["approach_log_weight_difference"] < audit[CONDITIONS[0]]["approach_log_weight_difference"]
