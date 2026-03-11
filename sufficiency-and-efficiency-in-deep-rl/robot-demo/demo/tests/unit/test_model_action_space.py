from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.action_space import (
    action_name_to_one_hot,
    one_hot_to_action_name,
    one_hot_to_action_vector,
    mix_action_vectors,
)
from model.backbones import _parse_action_and_text


def test_one_hot_round_trip_preserves_action_name():
    one_hot = action_name_to_one_hot("drive-forward")
    assert one_hot_to_action_name(one_hot) == "drive-forward"


def test_one_hot_maps_to_expected_vector_dictionary():
    one_hot = action_name_to_one_hot("look-left")
    vector = one_hot_to_action_vector(one_hot)
    assert vector == {"pan": 1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0}


def test_drive_left_vector_keeps_forward_drive():
    one_hot = action_name_to_one_hot("drive-left")
    vector = one_hot_to_action_vector(one_hot)
    assert vector == {"pan": 0.0, "tilt": 0.0, "turn": -1.0, "drive": 1.0}


def test_mixed_vectors_interpolate_with_t():
    mixed = mix_action_vectors(
        {"pan": 1.0, "tilt": 0.0, "turn": 0.0, "drive": 0.0},
        {"pan": 0.0, "tilt": 0.0, "turn": 0.0, "drive": 1.0},
        0.25,
    )
    assert mixed == {"pan": 0.75, "tilt": 0.0, "turn": 0.0, "drive": 0.25}


def test_parse_action_and_text_prefers_embedded_json():
    action_name, spoken_text = _parse_action_and_text(
        "prefix noise {\"action\": \"look-right\", \"say\": \"checking right\"} suffix noise"
    )
    assert action_name == "look-right"
    assert spoken_text == "checking right"


def test_parse_action_and_text_uses_last_exact_action_match():
    action_name, spoken_text = _parse_action_and_text(
        "Choose exactly one action from drive-left, drive-right, look-up. Final answer: look-up"
    )
    assert action_name == "look-up"
    assert spoken_text == ""
