"""Lifecycle-only validation tests; never load MNIST or execute a GPU smoke."""

from __future__ import annotations

import json

import pytest
import torch

from mnist_experiment.akaike_score_gate import artifacts, validation
from mnist_experiment.akaike_score_gate.artifacts import RunLimit, StopExecution, UnitStore
from mnist_experiment.akaike_score_gate.budget import WorkBudget
from mnist_experiment.akaike_score_gate.config import Study


def config(device="cpu"):
    raw = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json").mapping()
    return Study({**raw, "device": device})


@pytest.fixture
def mocked_pipeline(monkeypatch):
    sources = artifacts.source_hashes()
    monkeypatch.setattr(artifacts, "source_hashes", lambda: dict(sources))
    monkeypatch.setattr(artifacts, "runtime_metadata", lambda *args: {"mocked_for_unit_test": True})
    events = []
    asset_required = ("assets.pt", "summary.json", "metrics.json")
    trajectory_required = ("final_state.pt", "snapshots.pt", "summary.json", "metrics.json", "gates.json")

    def fake_assets(store, phase, index, family, width, *, source_only=False, resume=True):
        kind = "source" if source_only else "assets"
        unit = store.unit(phase, kind, index, environment=family, detail={"width": width})
        session = store.begin(unit, asset_required, resume=resume)
        if session is None:
            return store.completed(unit, asset_required)
        if not source_only:
            fake_assets(store, phase, index, family, width, source_only=True, resume=True)
        events.append((kind, store.study.device, family))
        session.write_torch("assets.pt", {"streams": {
            "rotation_linear": {"positions": [0., 15., 30.]},
            "mixture_m2": {"positions": [0., 0.1, 0.2]},
        }})
        session.write_json("summary.json", {"status": "completed"})
        session.write_json("metrics.json", [])
        return store.finish(session, asset_required)

    def fake_trajectory(store, phase, index, environment, policy, asset_path, **kwargs):
        assert kwargs["profile"] is None
        assert kwargs["budget_tier"] == "unlimited"
        assert policy["score_samples"] == 4
        assert kwargs["snapshot_steps"] == (0, 1, 2)
        unit = store.unit(phase, "trajectory", index, environment=environment["name"], policy=policy["name"])
        session = store.begin(unit, trajectory_required, resume=kwargs["resume"])
        if session is None:
            return store.completed(unit, trajectory_required)
        store.limit.check()
        checkpoint = session.path / "checkpoint.pt"
        if not checkpoint.exists():
            events.append(("fit", store.study.device, policy["name"]))
            budget = WorkBudget(None, {"fit": 1})
            budget.grant_arrival()
            budget.spend("fit", units=2)
            state = {"fit_calls": 1, "stage": "evaluate", "model": {"weight": torch.tensor([1., 2.])},
                     "budget": budget.state_dict()}
            session.write_torch("checkpoint.pt", state)
            store.limit.check()
        else:
            state = torch.load(checkpoint, weights_only=False)
        final = {"model": state["model"], "archive": torch.eye(2), "anchor": torch.tensor([1., 2.]),
                 "ledger": {"0": [0, 1]}, "budget": state["budget"], "history": []}
        session.write_torch("final_state.pt", final)
        session.write_torch("snapshots.pt", [{"theta": torch.tensor([1., 2.]), "precision": torch.eye(2)}])
        session.write_json("summary.json", {"nll_auc": 1.2, "charged_seconds": 2.,
                                             "online_seconds": float(len(events))})
        session.write_json("metrics.json", [{"current_nll": 1.2, "observations": 2,
                                            "online_seconds": float(len(events)), "charged_seconds": 2.}])
        session.write_json("gates.json", [] if policy["gate"] is None else
                           [{"step": 0, "depth": 1, "open": False, "status": "closed"}])
        return store.finish(session, trajectory_required)

    monkeypatch.setattr(validation, "ensure_assets", fake_assets)
    monkeypatch.setattr(validation, "ensure_trajectory", fake_trajectory)
    return events


def test_cpu_validation_seals_four_cases_and_completed_resume_does_no_work(tmp_path, mocked_pipeline):
    store = UnitStore(tmp_path, config())
    before_threads = torch.get_num_threads()
    final = validation.ensure_pipeline_validation(store, {"charges": {"large_gpu_fit": 100.}})
    assert torch.get_num_threads() == before_threads
    summary = json.loads((final / "summary.json").read_text())
    assert summary["cpu_status"] == "passed"
    assert summary["cuda_status"] == "not_requested"
    assert summary["validated_cases"] == 4
    manifest = json.loads((final / "validation_manifest.json").read_text())
    assert all(record["interruption"]["checks"] == 3 for record in manifest["cases"].values())
    assert all(record["result"]["completed_resume_units"] == 0 for record in manifest["cases"].values())
    assert all(record["result"]["asset_path"].startswith("validation/cpu/")
               for record in manifest["cases"].values())
    units, events = store.limit.units, list(mocked_pipeline)
    assert validation.ensure_pipeline_validation(store, {"charges": {"large_gpu_fit": 100.}}) == final
    assert store.limit.units == units
    assert mocked_pipeline == events
    assert (final / "COMPLETED").is_file()
    with pytest.raises(RuntimeError, match="completed"):
        validation.ensure_pipeline_validation(store, {"charges": {"large_gpu_fit": 100.}}, resume=False)


def test_cuda_parent_always_validates_cpu_then_cuda(tmp_path, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    final = validation.ensure_pipeline_validation(UnitStore(tmp_path, config("cuda")), {})
    summary = json.loads((final / "summary.json").read_text())
    assert summary["cpu_status"] == summary["cuda_status"] == "passed"
    assert summary["validated_cases"] == 8
    devices = [event[1] for event in mocked_pipeline]
    assert devices[0] == "cpu"
    assert devices[-1] == "cuda"
    assert "cpu" not in devices[devices.index("cuda"):]


def test_validation_selects_current_revision_checkpoint(tmp_path, mocked_pipeline):
    original = UnitStore(tmp_path, config())
    first = validation.ensure_pipeline_validation(original, {})
    old_manifest = json.loads((first / "validation_manifest.json").read_text())
    old_paths = [tmp_path / row["result"][key] for row in old_manifest["cases"].values()
                 for key in ("reference_path", "resumed_path")]
    old_hashes = validation._hashes([first, *old_paths])
    revised = UnitStore(tmp_path, config())
    revised.sources = {**revised.sources, "preflight_revision.py": "new-revision"}
    second = validation.ensure_pipeline_validation(revised, {})
    assert second != first
    assert json.loads((second / "summary.json").read_text())["validated_cases"] == 4
    assert validation._hashes([first, *old_paths]) == old_hashes


def test_validation_rejects_adaptive_trajectory_without_a_gate(tmp_path, mocked_pipeline, monkeypatch):
    original = validation.ensure_trajectory

    def no_gate(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[4]["gate"] is not None:
            (result / "gates.json").write_text("[]")
        return result

    monkeypatch.setattr(validation, "ensure_trajectory", no_gate)
    with pytest.raises(AssertionError, match="exercise a real gate"):
        validation.ensure_pipeline_validation(UnitStore(tmp_path, config()), {})


def test_unavailable_requested_cuda_preserves_completed_cpu_evidence(tmp_path, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    store = UnitStore(tmp_path, config("cuda"))
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        validation.ensure_pipeline_validation(store, {})
    manifest_path = next((tmp_path / "phase0" / ".incomplete").rglob("validation_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cpu_status"] == "passed"
    assert manifest["cuda_status"] == "unavailable"
    assert len(manifest["cases"]) == 4


def test_parent_limits_are_not_mistaken_for_simulated_interruptions(tmp_path, mocked_pipeline):
    limit = RunLimit(max_units=2)
    store = UnitStore(tmp_path, config(), limit=limit)
    with pytest.raises(StopExecution, match="completed-unit limit"):
        validation.ensure_pipeline_validation(store, {})
    assert limit.units == 2
    completed_events = list(mocked_pipeline)
    resumed = validation.ensure_pipeline_validation(UnitStore(tmp_path, config()), {})
    assert (resumed / "COMPLETED").is_file()
    assert mocked_pipeline[:len(completed_events)] == completed_events
    assert mocked_pipeline.count(("source", "cpu", "rotation")) == 1


def test_scientific_comparator_ignores_wall_time_and_peak_memory_but_not_charged_work():
    validation._compare({"online_seconds": 1., "peak_cuda_bytes": 100, "peak_cpu_bytes": 200,
                         "charged_seconds": 2.},
                        {"online_seconds": 9., "peak_cuda_bytes": 500, "peak_cpu_bytes": 800,
                         "charged_seconds": 2.})
    with pytest.raises(AssertionError, match="charged_seconds"):
        validation._compare({"online_seconds": 1., "charged_seconds": 2.},
                            {"online_seconds": 1., "charged_seconds": 3.})
    with pytest.raises(AssertionError, match="archive"):
        validation._compare({"archive": torch.eye(2)}, {"archive": torch.zeros((2, 2))})


def test_corrupt_validation_progress_is_rejected(tmp_path, mocked_pipeline, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        validation.ensure_pipeline_validation(UnitStore(tmp_path, config("cuda")), {})
    path = next((tmp_path / "phase0" / ".incomplete").rglob("validation_checkpoint.json"))
    value = json.loads(path.read_text())
    value["payload"]["cpu_status"] = "faked"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="corrupt"):
        validation.ensure_pipeline_validation(UnitStore(tmp_path, config("cuda")), {})
