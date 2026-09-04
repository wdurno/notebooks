import json
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.phase4_config import load_phase4_config
from mnist_experiment.rotated_mnist.phase5_artifacts import RotatedPhase5RunStore
from mnist_experiment.rotated_mnist.phase5_config import (
    PHASE5_CONDITIONS,
    RotatedPhase5Config,
    load_phase5_config,
)
from mnist_experiment.rotated_mnist.phase5_reconstruct import (
    edr_reconstruction_gate,
)
from mnist_experiment.rotated_mnist.phase5_sensitivity import (
    sensitivity_gate,
    summarize_sensitivity_rows,
)


REPO_ROOT = Path(__file__).parents[3]
CONFIG_ROOT = REPO_ROOT / "mnist_experiment" / "rotated_mnist" / "configs"
SMOKE_CONFIG = CONFIG_ROOT / "phase5_smoke.json"
SOURCE_CONFIG = CONFIG_ROOT / "phase4_smoke.json"


def test_phase5_config_round_trips_and_freezes_challenge() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_phase5_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == RotatedPhase5Config.from_mapping(raw)
    assert config.conditions == PHASE5_CONDITIONS
    assert config.fixed_pis == (0.01, 0.025, 0.05, 0.10)
    assert config.controller.cold_start_pi == 0.05
    assert config.controller.action_half_life_steps == 4.0
    assert config.controller.trend_half_life_degrees == 7.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditions", ["fixed_pi005"], "conditions"),
        ("fixed_pis", [0.05], "fixed pi bracket"),
        ("controller.risk_metric", "euclidean", "Fisher"),
        ("controller.pi_min", 0.025, "pi_min is frozen"),
        ("controller.action_half_life_steps", 8.0, "half_life_steps is frozen"),
    ],
)
def test_phase5_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        RotatedPhase5Config.from_mapping(raw)


def _gate_rows(actions: list[float]) -> list[dict[str, object]]:
    regions = ("first_ascent", "return", "second_ascent")
    rows = []
    for step, action in enumerate(actions):
        rows.append(
            {
                "step": step,
                "region": regions[min(step // 2, 2)],
                "predictable_fisher_source_step": None if step == 0 else step - 1,
                "cold_start_active": step == 0,
                "applied_pi": action,
                "instantaneous_old_risk": 1.0,
                "instantaneous_new_risk": 2.0,
                "old_risk_moment": 1.0,
                "new_risk_moment": 2.0,
            }
        )
    return rows


def test_phase5_edr_gate_requires_predictable_nontrivial_actions() -> None:
    passed = edr_reconstruction_gate(
        _gate_rows([0.05, 0.02, 0.03, 0.06, 0.07, 0.08]),
        pi_min=0.01,
        prospective_pi=0.05,
    )
    floor_driven = edr_reconstruction_gate(
        _gate_rows([0.05, 0.01, 0.01, 0.01, 0.01, 0.01]),
        pi_min=0.01,
        prospective_pi=0.05,
    )

    assert passed["recommendation"] == "proceed"
    assert passed["checks"]["predictable_source_only"] is True
    assert floor_driven["recommendation"] == "stop"
    assert floor_driven["checks"]["not_floor_driven"] is False


def test_phase5_completed_run_store_is_immutable(tmp_path: Path) -> None:
    config = load_phase5_config(SMOKE_CONFIG)
    source = load_phase4_config(SOURCE_CONFIG)
    store = RotatedPhase5RunStore(
        tmp_path, run_kind="phase5_edr_reconstruction"
    )
    session = store.begin(config, source, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(Exception, match="completed"):
        store.begin(config, source, REPO_ROOT)


def _sensitivity_rows(actions: list[float]) -> list[dict[str, object]]:
    rows = []
    for step, action in enumerate(actions):
        if step < 40:
            region = "first_ascent"
        elif step < 60:
            region = "return"
        else:
            region = "second_ascent"
        rows.append(
            {
                "step": step,
                "region": region,
                "action_half_life_steps": 2.0,
                "trend_half_life_degrees": 7.5,
                "applied_pi": action,
                "cold_start_active": step < 10,
                "lower_bound_active": False,
                "upper_bound_active": False,
                "instantaneous_old_risk": 1.0,
                "instantaneous_new_risk": 2.0,
                "old_risk_moment": 1.0,
                "new_risk_moment": 2.0,
            }
        )
    return rows


def test_phase5_sensitivity_response_begins_after_dynamic_knot() -> None:
    actions = [0.05] * 100
    actions[40] = 0.90
    actions[41:45] = [0.07] * 4
    actions[61:65] = [0.03] * 4

    summary = summarize_sensitivity_rows(_sensitivity_rows(actions))

    first = summary["dynamic_knot_responses"][0]
    assert first["knot_action"] == pytest.approx(0.90)
    assert first["pre_mean"] == pytest.approx(0.05)
    assert first["post_mean"] == pytest.approx(0.07)
    assert first["absolute_post_pre_shift"] == pytest.approx(0.02)
    assert summary["mean_same_speed_knot_response"] == pytest.approx(0.0)


def test_phase5_sensitivity_gate_requires_response_without_roughness_blowup() -> None:
    baseline = {
        "mean_dynamic_knot_response": 0.002,
        "mean_absolute_second_difference": 0.001,
        "dynamic_to_same_speed_response_ratio": 1.5,
        "post_cold_floor_fraction": 0.0,
        "all_finite": True,
    }
    good = {
        **baseline,
        "mean_dynamic_knot_response": 0.004,
        "mean_absolute_second_difference": 0.0015,
        "dynamic_to_same_speed_response_ratio": 2.0,
    }
    noisy = {
        **good,
        "mean_absolute_second_difference": 0.003,
    }
    summaries = {
        "hpi4__hphi7p5": baseline,
        "hpi1__hphi7p5": good,
        "hpi2__hphi7p5": noisy,
    }

    gate = sensitivity_gate(summaries)

    assert gate["recommendation"] == "reopen"
    assert gate["closed_loop_candidates"] == ["hpi1__hphi7p5"]
