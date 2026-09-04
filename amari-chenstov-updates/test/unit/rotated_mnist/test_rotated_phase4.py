import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mnist_experiment.rotated_mnist.config import RotatedConfigError
from mnist_experiment.rotated_mnist.analysis import phase4_revisit_rows
from mnist_experiment.rotated_mnist.phase4_artifacts import RotatedPhase4RunStore
from mnist_experiment.rotated_mnist.phase4_config import (
    PHASE4_CONDITIONS,
    PHASE4_KNOTS,
    RotatedPhase4Config,
    load_phase4_config,
)
from mnist_experiment.rotated_mnist.phase4_metrics import (
    evaluate_materialized_classifier,
)
from mnist_experiment.rotated_mnist.run_phase4 import (
    _audit_arrivals,
    _event_batch,
    _smoke_projection,
    _window_auc,
)
from mnist_experiment.rotated_mnist.schedule import resolve_rotation_schedule
from mnist_experiment.rotated_mnist.transform import tensor_content_hash
from src.replay import stream_events


REPO_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase4_smoke.json"
)


class FixedClassifier(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.full((inputs.shape[0], 10), -2.0)
        predicted = inputs[:, 0, 0, 0].long().remainder(10)
        logits.scatter_(1, predicted[:, None], 2.0)
        return logits


def test_phase4_config_round_trips_and_freezes_memory_screen() -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config = load_phase4_config(SMOKE_CONFIG)

    assert config.to_mapping() == raw
    assert config == RotatedPhase4Config.from_mapping(raw)
    assert config.conditions == PHASE4_CONDITIONS
    assert config.rotation.knots_degrees == PHASE4_KNOTS
    assert config.replay.bounded_capacity == 32
    assert config.fisher.rank == 8
    assert config.fisher.fixed_pi == 0.05


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conditions", ["current_only"], "conditions"),
        ("rotation.knots_degrees", [0.0, 15.0, 30.0], "complete repeated"),
        ("data.samples_per_step", 4, "m=8"),
        ("replay.bounded_capacity", 16, "capacity is frozen"),
        ("fisher.rank", 4, "rank is frozen"),
    ],
)
def test_phase4_config_rejects_treatment_drift(
    field: str, value, message: str
) -> None:
    raw = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    target = raw
    names = field.split(".")
    for name in names[:-1]:
        target = target[name]
    target[names[-1]] = value

    with pytest.raises(RotatedConfigError, match=message):
        RotatedPhase4Config.from_mapping(raw)


def test_one_pass_metrics_keep_confusion_and_digit_nine_metrics() -> None:
    targets = torch.arange(10).repeat_interleave(2)
    predicted = targets.clone()
    predicted[0] = 9
    inputs = torch.zeros(20, 1, 28, 28)
    inputs[:, 0, 0, 0] = predicted

    metrics = evaluate_materialized_classifier(
        FixedClassifier(),
        inputs,
        targets,
        batch_size=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        nine_prevalence=0.1,
    )

    assert metrics["environment_accuracy"] == pytest.approx(19 / 20)
    assert metrics["nine_recall"] == pytest.approx(1.0)
    assert metrics["nine_precision"] == pytest.approx(2 / 3)
    assert metrics["nine_ovr_accuracy"] == pytest.approx(0.1 + 0.9 * 17 / 18)
    assert metrics["confusion_matrix"][0][9] == 1
    assert metrics["per_class_recall"][0] == pytest.approx(0.5)
    assert metrics["per_class_precision"][9] == pytest.approx(2 / 3)


def test_replay_batch_uses_exact_arrival_tensor_and_angle() -> None:
    config = load_phase4_config(SMOKE_CONFIG)
    schedule = resolve_rotation_schedule(config.rotation)
    inputs = torch.arange(6 * 8 * 28 * 28, dtype=torch.float32).reshape(
        6, 8, 1, 28, 28
    )
    targets = torch.arange(6 * 8, dtype=torch.long).reshape(6, 8).remainder(10)
    events = stream_events(
        2,
        tuple(range(8)),
        tuple(targets[2].tolist()),
        samples_per_step=8,
    )
    transformed_hashes = tuple(
        tuple(tensor_content_hash(inputs[step, within]) for within in range(8))
        for step in range(6)
    )

    class StreamPlan:
        pass

    plan = StreamPlan()
    plan.transformed_hashes = transformed_hashes
    plan.schedule = schedule
    batch_inputs, batch_targets = _event_batch(
        events,
        inputs,
        targets,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    records, violations = _audit_arrivals(events, inputs, plan)

    assert torch.equal(batch_inputs, inputs[2])
    assert torch.equal(batch_targets, targets[2])
    assert violations == 0
    assert {record["arrival_angle_degrees"] for record in records} == {30.0}


def test_phase4_window_auc_uses_stored_exposure() -> None:
    rows = [
        {
            "observations_before_evaluation": step * 8,
            "current_environment_accuracy": value,
        }
        for step, value in enumerate((0.5, 0.7, 0.9, 0.8, 0.6, 0.4))
    ]

    assert _window_auc(
        rows, "current_environment_accuracy", 0, 2
    ) == pytest.approx(0.7)
    assert _window_auc(
        rows, "current_environment_accuracy", 3, 5
    ) == pytest.approx(0.6)


def test_smoke_projection_accepts_final_no_update_row() -> None:
    config = load_phase4_config(SMOKE_CONFIG)
    states = {}
    for condition in PHASE4_CONDITIONS:
        states[condition] = SimpleNamespace(
            rows=[
                {"active_block": {"presented_event_count": 8}},
                {"active_block": None},
            ],
            evaluation_wall_seconds=0.1,
            learner_wall_seconds=0.01,
            fisher_wall_seconds=0.0,
            archive=None,
            archive_wall_seconds=0.0,
        )

    projected = _smoke_projection(
        states, config, shared_panel_materialization_seconds=0.1
    )

    assert projected is not None
    assert projected > 0.0


def test_phase4_completed_run_store_is_immutable(tmp_path: Path) -> None:
    config = load_phase4_config(SMOKE_CONFIG)
    store = RotatedPhase4RunStore(tmp_path)
    session = store.begin(config, REPO_ROOT)
    path = session.complete(required=())

    assert (path / "COMPLETED").is_file()
    with pytest.raises(Exception, match="completed"):
        store.begin(config, REPO_ROOT)


def test_phase4_revisit_rows_pair_the_two_ascents_by_angle() -> None:
    config = load_phase4_config(SMOKE_CONFIG)
    angles = (0.0, 15.0, 30.0, 0.0, 15.0, 30.0)
    metrics = {}
    for condition in PHASE4_CONDITIONS:
        metrics[condition] = tuple(
            {
                "step": step,
                "angle_degrees": angle,
                "current_environment_accuracy": 0.5 + 0.01 * step,
                "current_nll": 1.0 - 0.01 * step,
                "current_worst_class_recall": 0.3 + 0.01 * step,
            }
            for step, angle in enumerate(angles)
        )
    run = SimpleNamespace(config=config, trajectory_metrics=metrics)

    rows = phase4_revisit_rows(run)

    first_condition = rows[:3]
    assert [row["angle_degrees"] for row in first_condition] == [0.0, 15.0, 30.0]
    assert all(
        row["environment_accuracy_revisit_lift"] == pytest.approx(0.03)
        for row in first_condition
    )
