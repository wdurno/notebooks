from mnist_experiment.continual_subgd.refresh_notebook import (
    PHASE_LABELS,
    _code_cell,
    _expected_positive_curve,
    _phase_sort_key,
)


def test_phase_sort_key_matches_plan_order() -> None:
    phases = [
        "phase6b_hundred_positive",
        "phase6a_ten_positive",
        "phase6_low_prevalence",
        "phase3b_floor",
        "phase5r",
        "phase1",
        "phase3b_trust",
        "phase0",
        "phase5",
        "phase4",
        "phase3a",
        "phase2",
        "phase3b_geometry",
    ]

    assert sorted(phases, key=_phase_sort_key) == [
        "phase0",
        "phase1",
        "phase2",
        "phase3a",
        "phase3b_trust",
        "phase3b_geometry",
        "phase3b_floor",
        "phase4",
        "phase5",
        "phase5r",
        "phase6_low_prevalence",
        "phase6a_ten_positive",
        "phase6b_hundred_positive",
    ]
    assert all(phase in PHASE_LABELS for phase in phases)


def test_plot_cell_records_its_phase() -> None:
    cell = _code_cell("# Artifact-backed figure.", [], phase="phase5")

    assert cell["metadata"]["plan13_phase"] == "phase5"


def test_expected_positive_curve_uses_pre_update_mixture() -> None:
    rows = [
        {"post_burn_in_observations": 0, "p": 0.01},
        {"post_burn_in_observations": 182, "p": 0.02},
        {"post_burn_in_observations": 364, "p": 0.02},
    ]

    observations, expected = _expected_positive_curve(rows)

    assert observations == [0, 182, 364]
    assert expected == [0.0, 1.82, 5.46]
