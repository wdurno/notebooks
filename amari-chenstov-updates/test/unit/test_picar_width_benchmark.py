"""Visibility masks and error accounting never enter the Qwen test context."""

import numpy as np
import pytest

from numerical_experiment.picar_width_benchmark import error_fields, summarize
from numerical_experiment.picar_width_scenes import SCENES, mask_statistics, target_mask


def test_panel_is_fixed_with_exactly_three_invisible_cases():
    assert len(SCENES) == 20
    assert sum(not scene.visible for scene in SCENES) == 3
    assert len({scene.name for scene in SCENES}) == 20
    assert {scene.barrier for scene in SCENES} >= {"none", "full", "beside", "partial_left", "partial_right"}


def test_object_id_mask_decodes_link_bits_and_ignores_background():
    segmentation = np.array([[-1, 7, 9], [7 + (2 << 24), 9 + (8 << 24), 0]])
    assert target_mask(segmentation, 7).tolist() == [[False, True, False], [True, False, False]]


def test_visible_width_is_inclusive_span_not_area_or_distance():
    mask = np.zeros((10, 100), dtype=bool)
    assert mask_statistics(mask)["width_percent"] == 0.
    mask[4, 40] = mask[5, 59] = True
    assert mask_statistics(mask)["width_percent"] == 20.
    assert mask_statistics(mask)["visible_pixels"] == 2
    mask[3, 0] = mask[3, 99] = True
    assert mask_statistics(mask)["width_percent"] == 100.


def test_arithmetic_error_is_distinct_from_visual_measurement_error():
    row = error_fields(40., 7., 20.)
    assert row["width_error_pp"] == 20.
    assert row["ideal_reward"] == 2.5
    assert row["reward_from_reported_width"] == 5.
    assert row["arithmetic_error"] == 2.
    assert row["reward_error"] == 4.5


def test_invalid_outputs_are_counted_not_replaced_by_zero():
    panels = [{"panel_id": "a"}, {"panel_id": "b"}]
    rows = []
    for method in ("direct", "four_examples"):
        rows.append({"panel_id": "a", "method": method, "true_visible": False,
            "width_percent": 0., "reward": 0., **error_fields(0., 0., 0.)})
    rows.append({"panel_id": "b", "method": "direct", "true_visible": True,
        "width_percent": 10., "reward": 1.25, **error_fields(10., 1.25, 20.)})
    rows.append({"panel_id": "b", "method": "four_examples", "true_visible": True,
                 "parse_error": "invalid JSON"})
    metrics = summarize(panels, rows)
    assert metrics["four_examples"]["invalid"] == 1
    assert metrics["four_examples"]["valid_visible"] == 0
    assert metrics["four_examples"]["width_mae_pp_visible"] is None
    assert metrics["paired_width_errors"]["valid_pairs"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        summarize(panels, rows[:-1])
    rows[0]["target_pixels_sha256"] = "first"
    rows[1]["target_pixels_sha256"] = "different"
    with pytest.raises(ValueError, match="target images differ"):
        summarize(panels, rows)
