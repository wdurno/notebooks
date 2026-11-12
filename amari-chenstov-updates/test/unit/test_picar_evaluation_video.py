import copy

import pytest
import torch

from numerical_experiment.picar_evaluation_video import assert_unchanged, select_panel


def test_selects_first_original_hard_layout_not_best_outcome():
    panel = {"difficulty": "hard", "seeds": [1, 2], "config": {}}
    state = {"identity": {"evaluations": [panel]}}
    rows = [{"seed": 1, "success": False}, {"seed": 2, "success": True}]
    summary = {"evaluations": [{"checkpoint": "final", "difficulty": "hard", "rows": rows}]}
    assert select_panel(state, summary, "hard", None) == (panel, rows[0])
    with pytest.raises(ValueError, match="original evaluation"):
        select_panel(state, summary, "hard", 3)


@pytest.mark.parametrize("field", ["parameters", "rng", "diagonal", "optimizer"])
def test_detects_learning_or_rng_mutation(field):
    before = {field: {"value": torch.tensor([1.])}, "version": 10, "banks": ["head"]}
    after = copy.deepcopy(before)
    assert_unchanged(before, after)
    after[field]["value"].add_(1)
    with pytest.raises(AssertionError):
        assert_unchanged(before, after)


def test_detects_update_counter_change():
    with pytest.raises(AssertionError):
        assert_unchanged({"update_count": 10}, {"update_count": 11})
