"""Tiny CPU trajectory integration, using generated images and no MNIST assets."""

from __future__ import annotations

import json
import pytest
import torch

from mnist_experiment.akaike_score_gate import artifacts, trajectory
from mnist_experiment.akaike_score_gate.artifacts import UnitStore, file_hash
from mnist_experiment.akaike_score_gate.budget import WorkBudget
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.model import build_model
from mnist_experiment.rotated_mnist.artifacts import RotatedRunSession


class SimulatedCrash(Exception):
    pass


@pytest.fixture(autouse=True)
def deterministic_test_runtime(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # Concurrent implementation must not change the fixture's frozen protocol.
    sources = artifacts.source_hashes()
    monkeypatch.setattr(artifacts, "source_hashes", lambda: dict(sources))
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_case(tmp_path):
    raw = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json").mapping()
    raw.update(dtype="float64", mixture_steps=3, episode_arrivals=2,
               score_samples=2, fisher_chunk_size=2, max_iter=8, max_eval=12,
               tolerance_grad=1e-6, tolerance_change=1e-12)
    study = Study(raw)
    model, layout = build_model(54, dtype=torch.float64)
    generator = torch.Generator().manual_seed(234)
    inputs = torch.randn((3, 2, 1, 28, 28), generator=generator, dtype=torch.float64)
    targets = torch.tensor([[0, 1], [2, 9], [3, 9]])
    evaluation = torch.randn((4, 1, 28, 28), generator=generator, dtype=torch.float64)
    environment = {"name": "mixture_m2", "family": "mixture", "samples": 2, "schedule": "linear"}
    asset_path = tmp_path / "mock_assets"
    asset_path.mkdir()
    (asset_path / "config.json").write_text(json.dumps({"fixture": "generated-images-v1"}))
    initial_fisher = torch.eye(layout.total_numel, dtype=torch.float64) * 0.2
    assets = {
        "initial_model": model.state_dict(), "parameter_layout": layout.metadata(),
        "initial_fisher": initial_fisher,
        "streams": {environment["name"]: {"positions": [0., 0.1, 0.2],
                                          "inputs": inputs, "targets": targets}},
        "evaluation_inputs": evaluation, "evaluation_targets": torch.tensor([0, 1, 9, 9]),
    }
    torch.save(assets, asset_path / "assets.pt")
    policy = {"name": "depth1_diag", "depth": 1, "gate": None, "ewc": "diag",
              "rank": 1, "half_life": 8., "cadence": 1, "c": 1., "score_samples": 2}
    retention = {"omega": 19., "ridge_ratio": 0., "residual_tolerance": 1e-5}
    return {"study": study, "asset_path": asset_path, "environment": environment,
            "policy": policy, "retention": retention, "assets": assets}


def profile(allowance=100.):
    charges = {}
    for n in (2, 4):
        charges[f"prequential:w5:n{n}"] = 1.
        for ewc in ("diag", "lr"):
            charges[f"archive:w5:n{n}:e{ewc}"] = 2.
            for depth in range(1, 7):
                charges[f"fit:w5:d{depth}:n{n}:e{ewc}"] = 1.
                for gate in ("diag", "lr"):
                    charges[f"probe:w5:d{depth}:n{n}:e{ewc}:g{gate}"] = 1.
    return {"b0": {"mixture": allowance, "rotation": allowance},
            "buckets": {"mixture": [2, 4], "rotation": [2, 4]}, "charges": charges}


def run(root, case, *, work_profile=None, policy=None):
    return trajectory.ensure_trajectory(
        UnitStore(root, case["study"]), "phase0", 0, case["environment"],
        case["policy"] if policy is None else policy, case["asset_path"], width=5,
        retention=case["retention"], profile=work_profile,
        budget_tier="unlimited" if work_profile is None else "strict",
        snapshot_steps=(0, 1, 2), resume=True,
    )


def load(path, name):
    return torch.load(path / name, map_location="cpu", weights_only=False)


def assert_same_science(actual, expected):
    if isinstance(actual, torch.Tensor):
        # PR artifacts deliberately mark undefined precision at zero recall.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key == "started_unix" or (key.endswith("_seconds") and key != "charged_seconds"):
                continue
            assert_same_science(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for one, two in zip(actual, expected):
            assert_same_science(one, two)
    else:
        assert actual == expected


def assert_same_artifacts(actual, expected):
    for name in ("summary.json", "metrics.json", "gates.json"):
        assert_same_science(json.loads((actual / name).read_text()),
                            json.loads((expected / name).read_text()))
    for name in ("final_state.pt", "snapshots.pt", "checkpoint.pt"):
        assert_same_science(load(actual, name), load(expected, name))


def test_three_arrivals_weighted_episode_and_completed_resume(tiny_case, tmp_path, monkeypatch):
    root = tmp_path / "run"
    completed = run(root, tiny_case, work_profile=profile())
    assert (completed / "COMPLETED").is_file()
    metrics = json.loads((completed / "metrics.json").read_text())
    assert [r["observations"] for r in metrics] == [0, 2, 4, 6]
    assert [r["step"] for r in metrics] == [-1, 0, 1, 2]
    assert metrics[1]["n_eff"] == metrics[3]["n_eff"] == 2
    assert 3 < metrics[2]["n_eff"] < 4
    final = load(completed, "final_state.pt")
    assert final["ledger"] == {"0": [0, 1, 2, 3], "2": [4, 5]}
    budget = WorkBudget.from_state_dict(final["budget"])
    assert budget.elapsed_arrivals == 3
    assert budget.reserved == 0
    assert 0 < budget.spent <= budget.granted
    snapshots = load(completed, "snapshots.pt")
    torch.testing.assert_close(snapshots[0]["anchor"], snapshots[1]["anchor"], rtol=0, atol=0)
    assert not torch.equal(snapshots[1]["anchor"], snapshots[2]["anchor"])
    before_hashes = {path.name: file_hash(path) for path in completed.iterdir() if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("completed resume performed scientific work")

    monkeypatch.setattr(trajectory, "build_model", forbidden)
    monkeypatch.setattr(trajectory, "fit_restricted", forbidden)
    assert run(root, tiny_case, work_profile=profile()) == completed
    assert before_hashes == {path.name: file_hash(path) for path in completed.iterdir() if path.is_file()}


@pytest.mark.parametrize("step,stage", [
    (0, "fit"), (0, "evaluate"), (0, "advance"), (1, "arrival"),
    (1, "consolidate"), (1, "advance"), (2, "advance"),
])
def test_checkpoint_boundary_resume_matches_uninterrupted(
    tiny_case, tmp_path, monkeypatch, step, stage,
):
    work_profile = profile()
    uninterrupted = run(tmp_path / "uninterrupted", tiny_case, work_profile=work_profile)
    original_write = RotatedRunSession.write_torch
    interrupted = []

    def write_then_crash(session, name, value):
        result = original_write(session, name, value)
        if name == "checkpoint.pt" and value["step"] == step and value["stage"] == stage:
            interrupted.append(value)
            raise SimulatedCrash("persisted safe boundary")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(RotatedRunSession, "write_torch", write_then_crash)
        with pytest.raises(SimulatedCrash):
            run(tmp_path / "interrupted", tiny_case, work_profile=work_profile)
    assert len(interrupted) == 1
    state = interrupted[0]
    checkpoint_budget = WorkBudget.from_state_dict(state["budget"])
    assert checkpoint_budget.elapsed_arrivals <= 3
    assert checkpoint_budget.spent + checkpoint_budget.reserved <= checkpoint_budget.granted
    resumed = run(tmp_path / "interrupted", tiny_case, work_profile=work_profile)
    assert_same_artifacts(resumed, uninterrupted)


def test_crash_inside_unsealed_fit_replays_without_credit_reset(tiny_case, tmp_path, monkeypatch):
    work_profile = profile()
    reference = run(tmp_path / "reference", tiny_case, work_profile=work_profile)
    original_fit = trajectory.fit_restricted

    def crash_after_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        assert result["function_evaluations"] > 0
        raise SimulatedCrash("fit finished but its boundary was not persisted")

    root = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(trajectory, "fit_restricted", crash_after_fit)
        with pytest.raises(SimulatedCrash):
            run(root, tiny_case, work_profile=work_profile)
    partial_path = next(root.rglob("checkpoint.pt")).parent
    checkpoint = load(partial_path, "checkpoint.pt")
    assert checkpoint["stage"] == "fit"
    assert checkpoint["budget"]["elapsed_arrivals"] == 1
    resumed = run(root, tiny_case, work_profile=work_profile)
    assert_same_artifacts(resumed, reference)


def test_zero_budget_denies_learning_without_fake_fit(tiny_case, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-budget trajectory attempted a fit or Fisher sample")

    monkeypatch.setattr(trajectory, "fit_restricted", forbidden)
    monkeypatch.setattr(trajectory, "model_fisher", forbidden)
    completed = run(tmp_path / "zero", tiny_case, work_profile=profile(0.))
    final = load(completed, "final_state.pt")
    summary = json.loads((completed / "summary.json").read_text())
    rows = json.loads((completed / "metrics.json").read_text())
    assert_same_science(final["model"], tiny_case["assets"]["initial_model"])
    torch.testing.assert_close(final["archive"], tiny_case["assets"]["initial_fisher"], rtol=0, atol=0)
    assert final["ledger"] == {}
    assert summary["fit_calls"] == 0
    assert summary["unresolved_fits"] == 0
    assert summary["gate_denials"] > 0
    budget = WorkBudget.from_state_dict(final["budget"])
    assert budget.elapsed_arrivals == 3
    assert budget.spent == budget.remaining == budget.reserved == 0
    assert all(row["gate_status"].startswith("budget_denied") for row in rows[1:])


def test_budget_denied_checkpoint_restores_identical_credit_and_statuses(tiny_case, tmp_path, monkeypatch):
    work_profile = profile(3.)
    reference = run(tmp_path / "reference", tiny_case, work_profile=work_profile)
    original_write = RotatedRunSession.write_torch
    denied = []

    def crash_when_denied(session, name, value):
        result = original_write(session, name, value)
        if name == "checkpoint.pt" and value["stage"] == "evaluate" and value["gate_status"] == "budget_denied":
            denied.append(value)
            raise SimulatedCrash("denied fit boundary")
        return result

    root = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(RotatedRunSession, "write_torch", crash_when_denied)
        with pytest.raises(SimulatedCrash):
            run(root, tiny_case, work_profile=work_profile)
    assert len(denied) == 1
    resumed = run(root, tiny_case, work_profile=work_profile)
    assert_same_artifacts(resumed, reference)


def test_unfitted_arrival_does_not_reuse_previous_residual(tiny_case, tmp_path, monkeypatch):
    original = WorkBudget.spend

    def spend(self, key, units=1):
        if self.elapsed_arrivals == 2 and key.startswith("prequential:"):
            return False
        return original(self, key, units)

    monkeypatch.setattr(WorkBudget, "spend", spend)
    completed = run(tmp_path / "denied_prediction", tiny_case, work_profile=profile())
    rows = json.loads((completed / "metrics.json").read_text())
    assert rows[1]["fit_residual"] is not None
    assert rows[2]["fit_residual"] is None
    assert rows[2]["gate_status"] == "budget_denied_prediction"


def test_nested_gate_refit_checkpoint_does_not_repeat_accepted_work(tiny_case, tmp_path, monkeypatch):
    policy = {**tiny_case["policy"], "name": "asg_diag_diag", "depth": None, "gate": "diag"}

    def controlled_gate(g, r, h, precision, covariance, active, candidate, **kwargs):
        opens = active.numel() == 54
        return {"open": opens, "status": "open" if opens else "closed",
                "d_eff_active": 1., "d_eff_expanded": 2.}

    monkeypatch.setattr(trajectory, "evaluate_gate", controlled_gate)
    work_profile = profile()
    reference = run(tmp_path / "reference", tiny_case, work_profile=work_profile, policy=policy)
    gates = json.loads((reference / "gates.json").read_text())
    assert gates and any(row["open"] for row in gates)
    original_write = RotatedRunSession.write_torch

    def crash_after_opening(session, name, value):
        result = original_write(session, name, value)
        if name == "checkpoint.pt" and value["step"] == 0 and value["stage"] == "fit" and value["depth"] == 2:
            assert value["d_eff"] == 2.
            raise SimulatedCrash("completed probe before expanded refit")
        return result

    root = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(RotatedRunSession, "write_torch", crash_after_opening)
        with pytest.raises(SimulatedCrash):
            run(root, tiny_case, work_profile=work_profile, policy=policy)
    resumed = run(root, tiny_case, work_profile=work_profile, policy=policy)
    assert_same_artifacts(resumed, reference)
