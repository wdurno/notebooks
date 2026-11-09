"""Recorded-potential reward contract, independent of simulator/evaluator code."""

import pytest
import torch

from src.continual import picar_math as pm


def inputs(dtype=torch.float64, requires_grad=False):
    def value(values):
        return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)
    return dict(elapsed=value([0.5, 1.0, 0.25, 0.75]),
                potential_start=value([0.2, 0.6, 0.4, 0.3]),
                potential_end=value([0.6, 0.9, 0.3, 0.7]),
                terminal_bonus=value([0., 10., 0., -5.]),
                terminated=torch.tensor([False, True, False, True]),
                gamma0=0.8, reference_duration=0.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shaped_rewards_mask_terminal_potential_and_telescope_across_episodes(dtype):
    values = inputs(dtype)
    rewards = pm.shaped_rewards(**values)
    discounts = pm.time_discounts(values["elapsed"], values["gamma0"], values["reference_duration"])
    base = values["terminal_bonus"] - 0.02 * values["elapsed"] / values["reference_duration"]
    base_returns = pm.discounted_returns(base, discounts, values["terminated"])
    shaped_returns = pm.discounted_returns(rewards, discounts, values["terminated"])
    torch.testing.assert_close(shaped_returns, base_returns - values["potential_start"])
    torch.testing.assert_close(rewards[:2], torch.tensor([0.26, 9.36], dtype=dtype))
    assert rewards.dtype == dtype
    assert values["potential_end"][-1] != 0  # Input snapshot is not modified.


def test_shaped_rewards_do_not_backpropagate_through_recorded_inputs():
    values = inputs(requires_grad=True)
    rewards = pm.shaped_rewards(**values)
    assert not rewards.requires_grad
    assert all(value.grad is None for value in values.values()
               if isinstance(value, torch.Tensor) and value.is_floating_point())


def test_zero_potential_matches_existing_smoke_reward():
    values = inputs()
    values["potential_start"] = torch.zeros(4, dtype=torch.float64)
    values["potential_end"] = torch.zeros(4, dtype=torch.float64)
    expected = values["terminal_bonus"] - .02 * values["elapsed"] / values["reference_duration"]
    torch.testing.assert_close(pm.shaped_rewards(**values), expected)


@pytest.mark.parametrize("case", ["elapsed_zero", "elapsed_shape", "start_negative", "end_above_one",
                                  "terminal_end_nan", "bonus_nan", "nonterminal_bonus",
                                  "bonus_shape", "terminal_shape", "terminal_dtype", "gamma", "duration"])
def test_shaped_rewards_reject_invalid_recorded_data(case):
    values = inputs()
    if case == "elapsed_zero":
        values["elapsed"][0] = 0
    elif case == "elapsed_shape":
        values["elapsed"] = values["elapsed"][:, None]
    elif case == "start_negative":
        values["potential_start"][0] = -0.1
    elif case == "end_above_one":
        values["potential_end"][0] = 1.1
    elif case == "terminal_end_nan":
        values["potential_end"][-1] = float("nan")
    elif case == "bonus_nan":
        values["terminal_bonus"][-1] = float("nan")
    elif case == "nonterminal_bonus":
        values["terminal_bonus"][0] = 1
    elif case == "bonus_shape":
        values["terminal_bonus"] = values["terminal_bonus"][:2]
    elif case == "terminal_shape":
        values["terminated"] = values["terminated"][:2]
    elif case == "terminal_dtype":
        values["terminated"] = values["terminated"].long()
    elif case == "gamma":
        values["gamma0"] = 1.1
    elif case == "duration":
        values["reference_duration"] = 0
    with pytest.raises(ValueError):
        pm.shaped_rewards(**values)
