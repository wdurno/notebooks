"""Fixed dense-visual-v1 arithmetic; no evaluator, model or dataset loading."""

import math

import pytest
import torch

from src.continual import picar_math as pm


def recorded(elapsed=(.5, .75, .125), scores=(0., 10., 5.), bonus=10.,
             dtype=torch.float64, requires_grad=False):
    def tensor(values):
        return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)
    return dict(elapsed=tensor(elapsed), visual_scores=tensor(scores),
                terminal_bonus=tensor([0.] * (len(elapsed) - 1) + [bonus]),
                terminated=torch.tensor([False] * (len(elapsed) - 1) + [True]))


def returns(values, gamma0=.99):
    return pm.discounted_returns(pm.dense_visual_rewards(**values),
                                 pm.time_discounts(values["elapsed"], gamma0),
                                 values["terminated"])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_dense_visual_formula_partial_interval_and_terminal_score(dtype):
    values = recorded(dtype=dtype, requires_grad=True)
    originals = {key: value.detach().clone() for key, value in values.items()}
    result = pm.dense_visual_rewards(**values)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    torch.testing.assert_close(result, torch.tensor([-.02, .045, 10.00125], dtype=expected_dtype))
    assert not result.requires_grad
    assert pm.DENSE_VISUAL_REWARD_VERSION == "dense-visual-v1"
    for key, value in values.items():
        torch.testing.assert_close(value, originals[key])
        assert value.grad is None


@pytest.mark.parametrize("cadence", [(.5,) * 40, (1.,) * 20, (.125,) * 160,
                                     (.7,) * 28 + (.4,), (.5,) * 39 + (.2, .3)])
def test_elapsed_not_picture_count_sets_visual_and_step_cost_budgets(cadence):
    values = recorded(cadence, [10.] * len(cadence), bonus=-1.)
    rewards = pm.dense_visual_rewards(**values)
    base = values["terminal_bonus"] - .02 * values["elapsed"] / .5
    assert (rewards - base).sum().item() == pytest.approx(2.)
    assert base.sum().item() == pytest.approx(-1.8)
    assert returns(values, gamma0=1)[0].item() == pytest.approx(.2)
    # The default discounts use actual partial intervals, not nominal ticks.
    assert pm.time_discounts(values["elapsed"]).prod().item() == pytest.approx(.99 ** 40)


@pytest.mark.parametrize("score", [0., 3., 10.])
@pytest.mark.parametrize("bonus", [-5., -1., 10.])
@pytest.mark.parametrize("gamma0", [1., .99, .7])
def test_all_complete_suffix_returns_fit_derived_enclosing_bounds(score, bonus, gamma0):
    values = recorded([.5] * 39 + [.25, .25], [score] * 41, bonus)
    labels = returns(values, gamma0)
    lower, upper = pm.dense_visual_return_bounds()
    assert (lower, upper) == (-6., 12.)
    assert bool(((labels >= lower) & (labels <= upper)).all())
    pm.bin_returns(labels, torch.linspace(lower, upper, 42, dtype=labels.dtype))
    if score == 0 and bonus == -5 and gamma0 == 1:
        assert labels[0].item() == pytest.approx(-5.8)


def test_terminal_bonus_is_unscaled_once_and_multiple_episodes_reset_returns():
    values = recorded([.125, .25, .5, .125], [0., 10., 3., 10.])
    values["terminated"] = torch.tensor([False, True, False, True])
    values["terminal_bonus"] = torch.tensor([0., 10., 0., -5.], dtype=torch.float64)
    rewards = pm.dense_visual_rewards(**values)
    stage = 2 * values["elapsed"] / 20 * (values["visual_scores"] / 10) - .02 * values["elapsed"] / .5
    torch.testing.assert_close(rewards - stage, values["terminal_bonus"])
    labels = returns(values)
    discounts = pm.time_discounts(values["elapsed"])
    torch.testing.assert_close(labels[[1, 3]], rewards[[1, 3]])
    torch.testing.assert_close(labels[0], rewards[0] + discounts[0] * rewards[1])
    torch.testing.assert_close(labels[2], rewards[2] + discounts[2] * rewards[3])
    # A separate full episode does not consume the next episode's time budget.
    values = recorded([20., 20.], [10., 10.])
    values["terminated"][:] = True
    values["terminal_bonus"][:] = 10.
    torch.testing.assert_close(returns(values), torch.tensor([11.2, 11.2], dtype=torch.float64))


def test_timely_success_beats_delayed_maximum_visibility_and_deadline_loitering():
    immediate = returns(recorded([.5], [0.]))[0].item()
    for count in range(2, 41):
        delayed = returns(recorded([.5] * count, [10.] * count))[0].item()
        assert delayed < immediate
    loiter = returns(recorded([.5] * 40, [10.] * 40, bonus=-1.))[0].item()
    disappeared = returns(recorded([.5] * 40, [10.] * 10 + [0.] * 30, bonus=-1.))[0].item()
    collision = returns(recorded([.5] * 40, [10.] * 40, bonus=-5.))[0].item()
    assert immediate > loiter > disappeared > collision
    # Positive visual scores can never turn even a full-horizon collision positive.
    assert collision < 0


def test_recorded_visual_history_survives_complete_return_instead_of_telescoping():
    low = recorded([.5] * 4, [1.] * 4)
    high = recorded([.5] * 4, [1., 9., 9., 9.])
    difference = returns(high)[0] - returns(low)[0]
    expected = sum(.04 * .99 ** index for index in (1, 2, 3))
    assert difference.item() == pytest.approx(expected)


@pytest.mark.parametrize("bins", [41, 81, 161])
def test_graded_return_bins_and_existing_likelihood_weights(bins):
    labels = torch.stack([returns(recorded([.5] * 40, [score] * 40))[0] for score in (2., 8.)])
    edges = torch.linspace(*pm.dense_visual_return_bounds(), bins + 1, dtype=torch.float64)
    indices = pm.bin_returns(labels, edges)
    assert indices[1] > indices[0]
    midpoints = pm.return_bin_values(edges)
    temperature = .7
    design = pm.LikelihoodDesign(midpoints, torch.zeros(2, dtype=torch.float64), temperature,
                                 torch.eye(7, dtype=torch.float64))
    log_weights = design.log_weights.gather(1, indices[:, None]).squeeze(1)
    bound = (edges[1:] - edges[:-1]).max() / (2 * temperature)
    assert bool((abs(log_weights - labels / temperature) <= bound + 1e-12).all())
    assert design.weights[1, indices[1]] > design.weights[0, indices[0]]
    assert design.cap_hit_rate.item() == 0
    assert labels[1] - labels[0] > .9


@pytest.mark.parametrize("case", ["empty", "elapsed_shape", "elapsed_zero", "elapsed_negative",
                                  "elapsed_nan", "elapsed_inf", "elapsed_integer", "over_horizon",
                                  "summed_over_horizon", "score_negative", "score_above_ten",
                                  "score_nan_terminal", "score_inf", "score_shape", "score_bool",
                                  "bonus_shape", "bonus_nan", "bonus_inf", "nonterminal_bonus",
                                  "terminal_zero_bonus", "unknown_bonus", "terminal_shape",
                                  "terminal_dtype", "duplicate_capture"])
def test_dense_rewards_reject_invalid_temporal_score_and_bonus_data(case):
    values = recorded()
    if case == "empty":
        values = {key: value[:0] for key, value in values.items()}
    elif case == "elapsed_shape":
        values["elapsed"] = values["elapsed"][:, None]
    elif case.startswith("elapsed_"):
        if case == "elapsed_integer":
            values["elapsed"] = values["elapsed"].long()
        else:
            values["elapsed"][0] = {"elapsed_zero": 0, "elapsed_negative": -.1,
                                      "elapsed_nan": math.nan, "elapsed_inf": math.inf}[case]
    elif case == "over_horizon":
        values["elapsed"][0] = 20.1
    elif case == "summed_over_horizon":
        values["elapsed"][:] = 7.
    elif case == "score_shape":
        values["visual_scores"] = values["visual_scores"][:1]
    elif case == "score_bool":
        values["visual_scores"] = values["visual_scores"].bool()
    elif case.startswith("score_"):
        values["visual_scores"][-1] = {"score_negative": -.001, "score_above_ten": 10.001,
                                        "score_nan_terminal": math.nan, "score_inf": math.inf}[case]
    elif case == "bonus_shape":
        values["terminal_bonus"] = values["terminal_bonus"][:1]
    elif case in ("bonus_nan", "bonus_inf", "terminal_zero_bonus", "unknown_bonus"):
        values["terminal_bonus"][-1] = {"bonus_nan": math.nan, "bonus_inf": math.inf,
                                         "terminal_zero_bonus": 0., "unknown_bonus": 1.}[case]
    elif case == "nonterminal_bonus":
        values["terminal_bonus"][0] = 10.
    elif case == "terminal_shape":
        values["terminated"] = values["terminated"][:1]
    elif case == "terminal_dtype":
        values["terminated"] = values["terminated"].long()
    elif case == "duplicate_capture":
        values["elapsed"][-1] = 0.
    with pytest.raises(ValueError):
        pm.dense_visual_rewards(**values)


def test_score_validation_precedes_dtype_rounding_and_float32_time_sums_are_stable():
    values = recorded(dtype=torch.float32)
    values["visual_scores"] = torch.tensor([0., 0., 10. + 1e-8], dtype=torch.float64)
    with pytest.raises(ValueError, match="\\[0,10\\]"):
        pm.dense_visual_rewards(**values)
    values = recorded([.1] * 200, [5.] * 200, dtype=torch.float32)
    assert bool(torch.isfinite(returns(values)).all())


def test_incomplete_rewards_are_allowed_but_never_fabricate_complete_labels():
    values = recorded()
    values["terminated"][:] = False
    values["terminal_bonus"][:] = 0.
    reward = pm.dense_visual_rewards(**values)
    discounts = pm.time_discounts(values["elapsed"])
    with pytest.raises(ValueError, match="incomplete"):
        pm.discounted_returns(reward, discounts, values["terminated"])
    values["terminated"][-1] = True
    with pytest.raises(ValueError, match="truncations"):
        pm.discounted_returns(reward, discounts, values["terminated"], torch.tensor([False, False, True]))
    with pytest.raises(ValueError, match="no silent clipping"):
        pm.bin_returns(torch.tensor([-6.001, 12.001], dtype=torch.float64),
                       torch.linspace(-6, 12, 42, dtype=torch.float64))
