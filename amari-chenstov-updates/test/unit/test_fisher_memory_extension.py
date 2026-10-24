"""Plan 15 optional-method extension contracts, without production computation."""

import json

from mnist_experiment.fisher_memory import extension

import nbformat
import pytest
import torch

from mnist_experiment.fisher_memory.artifacts import ArtifactError, Store
from mnist_experiment.fisher_memory.config import Config, Treatment, content_hash
from mnist_experiment.fisher_memory.extension import _job_id, initial_state_hash, reference_ledger
from mnist_experiment.fisher_memory.extension_report import build_notebook, collect
from mnist_experiment.fisher_memory.report import ArtifactError as ReportArtifactError, _extension_cell


def sealed(store, unit, metadata, name, payload):
    store.begin(unit, metadata)
    store.write_json(unit, name, payload)
    store.complete(unit)


def shared_state():
    return {"assets": {"plan_hash": "paired-plan"}, "learner": dict(
        theta=torch.tensor([.1, .2]), ltm_indices=torch.tensor([0]),
        anchor_L=torch.tensor([.1]), fisher_L=dict(kind="diagonal", diagonal=torch.tensor([.3])),
        replay=[], consolidated_ids=[], cumulative_count=0, accepted_arrivals=0)}


def source_root(tmp_path, cfg):
    root = tmp_path / "source"
    identity = {"config_hash": cfg.scientific_hash, "data_hash": "same-data", "source_hash": "old-source"}
    with Store(root, identity) as store:
        store.write_json(None, "provenance.json", {"config_hash": cfg.scientific_hash,
                                                   "source_hash": "old-source"})
        sealed(store, "scheduler-confirmation-ledger-a", {"name": "confirmation-ledger"}, "result.json",
               {"result": {"replicas": 1, "evaluation_protocol_hash": "same-evaluation"}})
        jvp = Treatment.default(cfg, "jvp", .25)
        directional = Treatment.default(cfg, "directional", .5)
        sealed(store, "scheduler-selection-a", {"name": "selection", "environment": "rotation_linear"},
               "result.json", {"result": {"shortlist": {"jvp": jvp.mapping(),
                                                          "directional": directional.mapping()}}})
        asset = "assets-confirmation-rotation-0000"
        store.begin(asset, {"plan_hash": "paired-plan"})
        store.save_checkpoint(asset, shared_state())
        store.write_json(asset, "data_plan.json", {"plan_hash": "paired-plan"})
        store.complete(asset)
        control_unit = "trajectory-confirmation-rotation_linear-0000-replay_ewc-a"
        sealed(store, control_unit,
               {"kind": "trajectory", "family": "confirmation", "environment": "rotation_linear",
                "replica": 0, "condition": "replay_ewc", "config_hash": cfg.scientific_hash,
                "config": Treatment.default(cfg, "replay_ewc").mapping()}, "summary.json",
               {"unit_id": control_unit, "current_auc": .8, "historical_auc": .9, "active_seconds": 10.,
                "nonfinite": False, "solver_failures": 0})
    return root


def test_reference_ledger_uses_sealed_paired_controls_and_pilot_settings(tmp_path):
    cfg = Config(device="cpu")
    root = source_root(tmp_path, cfg)
    ledger = reference_ledger(root, cfg, ("rotation_linear",), ("jvp", "directional"), 1, "same-data")
    assert [job["method"] for job in ledger["jobs"]] == ["jvp", "directional"]
    assert [job["treatment"]["strength"] for job in ledger["jobs"]] == [.25, .5]
    assert {job["asset_plan_hash"] for job in ledger["jobs"]} == {"paired-plan"}
    assert {job["initial_state_hash"] for job in ledger["jobs"]} == {initial_state_hash(shared_state())}
    changed = shared_state()
    changed["learner"]["theta"][0] += .001
    assert initial_state_hash(changed) != ledger["jobs"][0]["initial_state_hash"]
    assert {job["control"]["current_auc"] for job in ledger["jobs"]} == {.8}
    assert len({job["control_seal"] for job in ledger["jobs"]}) == 1
    with pytest.raises(ValueError, match="replicas"):
        reference_ledger(root, cfg, ("rotation_linear",), ("jvp",), 2, "same-data")
    with pytest.raises(ValueError, match="dataset"):
        reference_ledger(root, cfg, ("rotation_linear",), ("jvp",), 1, "different-data")


def test_reference_ledger_rejects_corrupt_sealed_control(tmp_path):
    cfg = Config(device="cpu")
    root = source_root(tmp_path, cfg)
    summary = next((root / "units").glob("trajectory-confirmation-*/summary.json"))
    summary.write_text(summary.read_text().replace('"current_auc":0.8', '"current_auc":0.7'))
    with pytest.raises(ArtifactError, match="artifact checksum mismatch"):
        reference_ledger(root, cfg, ("rotation_linear",), ("jvp",), 1, "same-data")


def test_extension_scope_is_immutable_and_partial_notebook_is_clear(tmp_path):
    cfg = Config(device="cpu")
    source = source_root(tmp_path, cfg)
    ledger = reference_ledger(source, cfg, ("rotation_linear",), ("jvp", "directional"), 1, "same-data")
    root = tmp_path / "extension"
    identity = {"extension_ledger_hash": content_hash(ledger)}
    with Store(root, identity) as store:
        sealed(store, "extension-ledger-v1", {"kind": "extension_ledger"}, "result.json", ledger)
        store.write_json(None, "status.json", {"status": "running", "active_seconds": 0,
                                               "current_job": None})
    notebook = build_notebook(root)
    nbformat.validate(notebook)
    assert "0/2 trajectories" in notebook.cells[0].source
    assert "post-selection" in notebook.cells[0].source
    assert len(notebook.cells) == 3
    with pytest.raises(Exception, match="identity changed"):
        with Store(root, {"extension_ledger_hash": "different"}, resume=True):
            pass

    job = ledger["jobs"][0]
    unit = _job_id(job)
    summary = {"unit_id": unit, "current_auc": .7, "historical_auc": .8,
               "active_seconds": 20., "nonfinite": False, "solver_failures": 0}
    with Store(root, identity, resume=True) as store:
        sealed(store, unit, {"kind": "trajectory", "environment": job["environment"],
                             "replica": job["replica"], "config": job["treatment"]},
               "summary.json", summary)
        store.write_json(None, "status.json", {"status": "paused", "active_seconds": 20,
                                               "current_job": None})
    observed = collect(root)["observed"]
    assert list(observed) == [unit]
    notebook = build_notebook(root)
    assert "1/2 trajectories" in notebook.cells[0].source
    assert "+0.1000" in notebook.cells[1].source
    assert len(notebook.cells) == 4
    assert notebook.cells[2].outputs[0].data["image/png"]
    nbformat.validate(notebook)
    cell = _extension_cell(root, source, [])
    assert "Phase 5: Alternative STM penalties" in cell.source
    assert "+0.1000 (descriptive)" in cell.source
    with pytest.raises(ReportArtifactError, match="different Plan 15 source"):
        _extension_cell(root, tmp_path / "different", [])


def test_entrypoint_resumes_only_remaining_jobs_and_rejects_scope_change(tmp_path, monkeypatch):
    cfg = Config(device="cpu", free_reserve_bytes=0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg.mapping()))
    budget = tmp_path / "budget"
    source = budget / "source"
    root = budget / "extension"
    source.mkdir(parents=True)
    notebook = tmp_path / "extension.ipynb"

    def fake_reference(source_root, config, environments, methods, replicas, data_hash):
        jobs = []
        for method in methods:
            treatment = Treatment.default(config, method, .25)
            jobs.append(dict(environment=environments[0], replica=0, method=method,
                             treatment=treatment.mapping(), asset_plan_hash="paired-plan",
                             initial_state_hash=initial_state_hash(shared_state()),
                             control={"current_auc": .8, "historical_auc": .9}))
        return dict(schema_version=1, protocol="plan15-extension-v1",
                    scope=dict(environments=list(environments), methods=list(methods), replicas=replicas),
                    jobs=jobs)

    class FakeRunner:
        def __init__(self, config, store, source_obj, *, on_progress):
            self.store, self.on_progress = store, on_progress
            self.completed_this_invocation = 0

        def _assets(self, family, environment, replica):
            return "assets", shared_state()

        def run_trajectory(self, family, phase, environment, replica, treatment):
            job = dict(environment=environment, replica=replica, treatment=treatment.mapping())
            unit = _job_id(job)
            summary = dict(unit_id=unit, current_auc=.7, historical_auc=.8,
                           active_seconds=1., nonfinite=False, solver_failures=0)
            sealed(self.store, unit, dict(kind="trajectory", environment=environment,
                                          replica=replica, config=treatment.mapping()),
                   "summary.json", summary)
            self.completed_this_invocation += 1
            self.on_progress()
            return summary

    monkeypatch.setattr(extension, "reference_ledger", fake_reference)
    monkeypatch.setattr(extension, "Runner", FakeRunner)
    monkeypatch.setattr(extension, "MNISTSource", lambda path: object())
    monkeypatch.setattr(extension, "dataset_manifest", lambda path: {"cached": True})
    monkeypatch.setattr(extension, "external_bytes", lambda args: 0)
    monkeypatch.setattr(extension, "provenance", lambda config: {"source_hash": "new-source"})
    args = ["--config", str(config_path), "--source-root", str(source), "--root", str(root),
            "--budget-root", str(budget), "--notebook", str(notebook),
            "--methods", "jvp", "directional", "--environments", "rotation_linear",
            "--replicas", "1", "--allocation-hours", "1", "--max-units", "1"]
    extension.main(args)
    assert json.loads((root / "status.json").read_text())["completed_units"] == 1
    assert json.loads((root / "status.json").read_text())["status"] == "paused"
    extension.main(args + ["--resume"])
    assert json.loads((root / "status.json").read_text())["completed_units"] == 2
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    assert notebook.exists()
    with pytest.raises(Exception, match="identity changed"):
        extension.main(args[:args.index("--methods")] + ["--methods", "directional",
                       "--environments", "rotation_linear", "--replicas", "1",
                       "--allocation-hours", "1", "--resume"])
