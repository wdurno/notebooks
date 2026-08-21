import numpy as np

from src.plan4_analysis import (
    PI_MIN,
    corrected_displacement_moments,
    interpolate_paths,
    phase2_schedules,
    policy_opportunity,
)


def test_interpolate_paths_retains_complete_fit_trajectories() -> None:
    source_p = np.asarray([0.0, 0.5, 1.0])
    paths = np.asarray(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
        ]
    )

    interpolated = interpolate_paths(
        source_p, paths, np.asarray([0.0, 0.25, 0.75, 1.0])
    )

    assert interpolated.shape == (2, 4, 1)
    assert np.allclose(interpolated[0, :, 0], [0.0, 0.5, 1.5, 2.0])
    assert np.allclose(interpolated[1] - interpolated[0], 10.0)


def test_paired_fit_offsets_do_not_create_displacement_noise() -> None:
    true_path = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0]])
    offsets = np.asarray([[0.0, 0.0], [2.0, -1.0], [-3.0, 4.0], [1.5, 2.5]])
    paths = true_path[None, :, :] + offsets[:, None, :]
    counts = np.tile(np.ones(4, dtype=np.int64), (100, 1))

    result = corrected_displacement_moments(paths, counts)

    assert np.allclose(result["central_raw"], [5.0, 4.0])
    assert np.allclose(result["central_noise"], 0.0, atol=1e-12)
    assert np.allclose(result["central_corrected"], [5.0, 4.0])


def test_policy_opportunity_rolls_policy_specific_q_state() -> None:
    d2 = np.zeros((1, 12), dtype=np.float64)
    trace = np.full((1, 12), 100.0, dtype=np.float64)
    event = np.ones(12, dtype=bool)

    result = policy_opportunity(d2, trace, event)

    assert result["pi"].shape == (1, 12)
    assert np.all(result["pi"] >= PI_MIN)
    assert np.all(np.diff(result["q"][0]) >= 0.0)
    assert np.isfinite(result["event_relative_risk_reduction"]).all()


def test_phase2_schedules_share_endpoints_and_exposure() -> None:
    schedules = phase2_schedules()

    assert tuple(schedules) == (
        "linear",
        "logistic-k8",
        "logistic-k16",
        "logistic-k32",
        "logistic-k64",
        "logistic-k128",
        "logistic-k256",
    )
    for schedule in schedules.values():
        assert schedule.p_values[0] == 0.0
        assert schedule.p_values[-1] == 0.2
        assert np.isclose(sum(schedule.p_values), 10.0)
