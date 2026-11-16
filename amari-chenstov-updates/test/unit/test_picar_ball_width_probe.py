"""Prompt-probe outputs are inspected, never silently repaired into rewards."""

import pytest

from numerical_experiment.picar_ball_width_probe import PROMPT, parse


def test_width_and_native_reward_remain_separate_observations():
    assert parse('{"width_percent": 40, "reward": 5}') == {"width_percent": 40, "reward": 5}
    assert parse('{"width_percent": 40, "reward": 3}') == {"width_percent": 40, "reward": 3}
    assert "width_percent / 8" in PROMPT


@pytest.mark.parametrize("text", [
    '{"width_percent": true, "reward": 1}',
    '{"width_percent": -1, "reward": 1}',
    '{"width_percent": 120, "reward": 10}',
    '{"width_percent": 30, "reward": 11}',
    '{"width_percent": 30, "reward": NaN}',
    '{"reward": 2}',
])
def test_invalid_judgments_are_not_converted_to_zero(text):
    with pytest.raises(ValueError):
        parse(text)
