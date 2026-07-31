import pytest
import torch

from src.directional_ridge import DirectionalRidgeLFUState


def _state(
    *,
    half_life_steps: float = 1.0,
    amplitude_epsilon: float = 1e-9,
    coherence_threshold: float = 0.75,
) -> DirectionalRidgeLFUState:
    return DirectionalRidgeLFUState(
        half_life_steps=half_life_steps,
        amplitude_epsilon=amplitude_epsilon,
        coherence_threshold=coherence_threshold,
    )


def _matrices() -> tuple[torch.Tensor, torch.Tensor]:
    ac = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    residual = torch.diag(torch.tensor([0.5, -0.25], dtype=torch.float64))
    return ac, residual


def test_cold_start_attenuates_first_correction_without_bias_correction() -> None:
    state = _state(half_life_steps=1.0)
    direction = torch.tensor([2.0, 0.0], dtype=torch.float64)
    ac, residual = _matrices()

    update = state.update(direction, ac, residual)

    expected_scale = state.beta * 4.0 / (4.0 + state.amplitude_epsilon**2)
    torch.testing.assert_close(update.amari_chentsov, expected_scale * ac)
    torch.testing.assert_close(update.residual, expected_scale * residual)
    assert update.cold_started
    assert update.warmup_mass == pytest.approx(state.beta)
    assert update.prior_mass == pytest.approx(state.rho)
    assert update.amplitude_denominator == pytest.approx(4.0)


def test_constant_direction_warms_toward_the_raw_correction() -> None:
    state = _state(half_life_steps=1.0)
    direction = torch.tensor([2.0, 0.0], dtype=torch.float64)
    ac, residual = _matrices()

    first = state.update(direction, ac, residual)
    second = state.update(direction, ac, residual)

    torch.testing.assert_close(first.amari_chentsov, 0.5 * ac)
    torch.testing.assert_close(second.amari_chentsov, 0.75 * ac)
    torch.testing.assert_close(second.residual, 0.75 * residual)
    assert second.warmup_mass == pytest.approx(0.75)
    assert second.prior_mass == pytest.approx(0.25)


def test_no_motion_applies_zero_and_decays_without_rotating_state() -> None:
    state = _state(half_life_steps=1.0, amplitude_epsilon=1e-4)
    direction = torch.tensor([2.0, 0.0], dtype=torch.float64)
    ac, residual = _matrices()
    first = state.update(direction, ac, residual)

    update = state.update(torch.zeros_like(direction), ac, residual)

    assert update.no_motion
    torch.testing.assert_close(update.amari_chentsov, torch.zeros_like(ac))
    torch.testing.assert_close(update.residual, torch.zeros_like(residual))
    torch.testing.assert_close(
        update.reference_direction,
        first.reference_direction,
    )
    assert update.amplitude_denominator == pytest.approx(
        state.rho * first.amplitude_denominator
    )
    assert torch.isfinite(update.full).all()


def test_signed_reversal_reuses_the_same_directional_segment() -> None:
    state = _state(half_life_steps=1.0)
    direction = torch.tensor([2.0, 0.0], dtype=torch.float64)
    ac, residual = _matrices()
    state.update(direction, ac, residual)

    update = state.update(-direction, -ac, -residual)

    assert update.signed_amplitude == pytest.approx(-2.0)
    assert update.coherence_before_reset == pytest.approx(1.0)
    assert not update.direction_reset
    torch.testing.assert_close(update.amari_chentsov, -0.75 * ac)
    torch.testing.assert_close(update.residual, -0.75 * residual)


def test_incoherent_direction_starts_a_new_cold_segment() -> None:
    state = _state(half_life_steps=1.0, coherence_threshold=0.75)
    ac, residual = _matrices()
    state.update(torch.tensor([1.0, 0.0], dtype=torch.float64), ac, residual)

    update = state.update(
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        3.0 * ac,
        4.0 * residual,
    )

    assert update.direction_reset
    assert update.coherence_before_reset == pytest.approx(0.0)
    assert update.orthogonal_ratio_before_reset == pytest.approx(1.0)
    assert update.segment_eligible_steps == 1
    assert update.warmup_mass == pytest.approx(state.beta)
    torch.testing.assert_close(update.amari_chentsov, 1.5 * ac)
    torch.testing.assert_close(update.residual, 2.0 * residual)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("half_life_steps", 0.0),
        ("amplitude_epsilon", 0.0),
        ("coherence_threshold", 0.0),
        ("coherence_threshold", 1.1),
    ],
)
def test_invalid_settings_are_rejected(name: str, value: float) -> None:
    settings = {
        "half_life_steps": 1.0,
        "amplitude_epsilon": 1e-6,
        "coherence_threshold": 0.75,
    }
    settings[name] = value

    with pytest.raises(ValueError):
        DirectionalRidgeLFUState(**settings)
