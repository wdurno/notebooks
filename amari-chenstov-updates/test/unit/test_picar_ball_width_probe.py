"""Prompt-probe outputs are inspected, never silently repaired into rewards."""

import pytest

from numerical_experiment.picar_ball_width_probe import PROMPT, parse, parse_edges


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


def test_width_and_reward_are_derived_from_qwen_edges():
    assert parse_edges('{"bbox": [450, 500, 550, 650]}') == {
        "bbox": [450, 500, 550, 650], "width_percent": 10., "reward": 1.25}
    assert parse_edges('{"bbox": [0, 1, 950, 1000]}')["reward"] == 10.
    assert parse_edges('{"bbox": null}') == {"bbox": None, "width_percent": 0., "reward": 0.}


@pytest.mark.parametrize("text", [
    '{"bbox": [true, 500, 550, 650]}',
    '{"bbox": [-1, 500, 550, 650]}',
    '{"bbox": [450, 500, 1001, 650]}',
    '{"bbox": [550, 500, 450, 650]}',
    '{"bbox": [450, 500, 450, 650]}',
    '{"bbox": [450, 650, 550, 500]}',
    '{"bbox": [450, 500, 550]}',
    '{"bbox": [450, 500, 550, NaN]}',
    '{"bbox": null, "reward": 0}',
])
def test_invalid_edges_are_not_converted_to_zero(text):
    with pytest.raises(ValueError):
        parse_edges(text)


def test_pixel_edges_use_processed_dimensions_not_normalized_units():
    result = parse_edges('{"bbox_2d": [140, 90, 170, 120]}', width=300, height=224, key="bbox_2d")
    assert result["width_percent"] == 10.
    assert result["reward"] == 1.25
    with pytest.raises(ValueError):
        parse_edges('{"bbox_2d": [140, 90, 450, 120]}', width=300, height=224, key="bbox_2d")


def test_horizontal_edges_do_not_require_vertical_localization():
    result = parse_edges('{"span": [140, 170]}', width=300, key="span", horizontal=True)
    assert result["width_percent"] == 10.
    assert result["reward"] == 1.25
    assert parse_edges('{"span": null}', width=300, key="span", horizontal=True)["reward"] == 0.
    for raw in ('{"span": [170, 140]}', '{"span": [140, 301]}', '{"span": [140, 170, 200]}'):
        with pytest.raises(ValueError):
            parse_edges(raw, width=300, key="span", horizontal=True)
