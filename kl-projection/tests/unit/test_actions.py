import pytest

from picar_kl.actions import (
    ACTION_NAMES,
    action_distribution_to_vector,
    action_index_to_name,
    action_name_to_distribution,
    action_name_to_index,
    action_vector_for_name,
    distribution_to_action_name,
    validate_action_distribution,
    validate_action_vector,
)


def test_action_name_index_round_trip():
    for index, action_name in enumerate(ACTION_NAMES):
        assert action_name_to_index(action_name) == index
        assert action_index_to_name(index) == action_name


def test_action_name_to_distribution_is_one_hot():
    distribution = action_name_to_distribution("look-up")

    assert distribution == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert distribution_to_action_name(distribution) == "look-up"


def test_distribution_to_vector_weighted_mix():
    distribution = (
        0.0,
        0.0,
        0.50,
        0.0,
        0.25,
        0.0,
        0.25,
        0.0,
    )

    assert action_distribution_to_vector(distribution) == {
        "pan": 0.25,
        "tilt": 0.25,
        "turn": 0.0,
        "drive": 0.5,
    }


def test_action_vectors_match_legacy_semantics():
    assert action_vector_for_name("drive-left") == {
        "pan": 0.0,
        "tilt": 0.0,
        "turn": -1.0,
        "drive": 1.0,
    }


def test_distribution_validation_rejects_bad_shape_and_sum():
    with pytest.raises(ValueError, match="Expected 8"):
        validate_action_distribution([1.0])

    with pytest.raises(ValueError, match="sum to 1"):
        validate_action_distribution([0.1] * 8)


def test_action_vector_validation_rejects_out_of_range_value():
    with pytest.raises(ValueError, match="tilt"):
        validate_action_vector({"pan": 0.0, "tilt": -0.1, "turn": 0.0, "drive": 0.0})


def test_distribution_validation_preserves_tiny_positive_mass():
    distribution = (0.9999992, 0.0000002, 0.0000002, 0.0000002, 0.0000002, 0.0, 0.0, 0.0)

    checked = validate_action_distribution(distribution)

    assert checked == distribution
    assert distribution_to_action_name(checked) == "drive-left"
