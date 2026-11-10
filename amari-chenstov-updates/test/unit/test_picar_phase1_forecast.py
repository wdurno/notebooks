import pytest

from numerical_experiment.picar.phase1_forecast import forecast


def probe():
    return {"timings": {"inference_seconds": .1, "fit_control_seconds": .2,
        "fit_fd_seconds": .4, "fisher_seconds": .1, "evaluator_seconds": .05,
        "teacher_seconds": .2}, "recipe": {"image_width": 160, "image_height": 120},
        "checkpoint": {"learner_state_bytes": 5_000_000, "transaction_seconds": .01}}


def test_pair_cost_scales_without_recharging_shared_bootstrap():
    a, b = [forecast(probe(), pairs=p, transitions=1200) for p in (2, 4)]
    assert b["phase_hours"]["phase4"][0] == 2 * a["phase_hours"]["phase4"][0]
    assert b["phase_hours"]["phase2"] == a["phase_hours"]["phase2"]
    assert b["phase_hours"]["phase3"] == a["phase_hours"]["phase3"]


def test_evaluation_and_sleep_are_charged_and_physics_not_double_counted():
    a = forecast(probe(), pairs=1, transitions=1200, control_period_s=1)
    assert a["heldout_seconds_per_arm"] == 72 * 20
    assert a["online_seconds_per_arm"]["control"] == pytest.approx(1200 + 4 * .2)
    assert a["online_seconds_per_arm"]["fd"] >= 1200


def test_zero_or_missing_measurement_is_not_a_free_operation():
    bad = probe()
    bad["timings"]["evaluator_seconds"] = 0
    with pytest.raises(ValueError, match="evaluator"):
        forecast(bad)


def test_storage_counts_camera_capture_not_downsampled_learner_image():
    result = forecast(probe(), pairs=1, transitions=1200)
    assert result["storage"]["raw_frame_bytes"] == 448 * 336 * 3 + 128
    assert result["storage"]["retain_all_checkpoint_bytes"] > result["storage"]["raw_and_record_bytes"]
