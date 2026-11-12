import math

import pytest

from numerical_experiment.picar.fix3_roles import compose_score, parse_components, centering_resolution_audit


def test_centering_bonus_is_bounded_and_discounts_camera_pan():
    value = {"visible": True, "proximity": 3., "centering": 1.}
    assert compose_score(value, 0.)["score"] == 3.5
    assert compose_score(value, math.radians(5))["score"] == pytest.approx(3.25)
    assert compose_score(value, -math.radians(10))["score"] == 3.
    assert compose_score({**value, "proximity": 10.}, 0.)["score"] == 10.
    assert compose_score({"visible": False, "proximity": 0., "centering": 0.}, 0.)["score"] == 0.
    with pytest.raises(ValueError):
        compose_score(value, float("nan"))


@pytest.mark.parametrize("text", [
    '{"visible":1,"proximity":3,"centering":1}',
    '{"visible":true,"proximity":-1,"centering":1}',
    '{"visible":true,"proximity":3,"centering":1.1}',
    '{"visible":false,"proximity":3,"centering":0}',
    '{"visible":true,"proximity":true,"centering":1}',
    '{"visible":true,"proximity":3,"centering":1,"reason":"extra"}',
])
def test_invalid_visual_components_are_not_silently_repaired(text):
    with pytest.raises(ValueError):
        parse_components(text)


def test_centering_return_signal_and_bin_alignment_are_measured():
    result = centering_resolution_audit()
    assert 0 < result["return_difference"] <= .1
    assert result["selected_bins"] in {81, 161}
    assert result["rows"][-1]["fraction_different_labels"] > result["rows"][0]["fraction_different_labels"]
    assert result["rows"][-1]["mean_absolute_log_weight_change"] > 0
