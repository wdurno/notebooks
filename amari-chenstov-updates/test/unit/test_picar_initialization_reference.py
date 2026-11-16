"""CPU-only independent MC reference, recovery, source and budget tests."""

import builtins
import copy
from dataclasses import replace
import hashlib
import json
import signal

import pytest
import torch

from numerical_experiment import picar_initialization_reference as ref
from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.learner import LearningBatch, MemoryLearner, predict_histories
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar.report import read_completed_unit
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_initialization.numerics import InitializationFit, _tensor_hash
from src.continual import picar_math as pm
from src.seeding import derive_component_seed


class MemoryStore:
    completed = False

    def __init__(self):
        self.state, self.rows, self.saves = None, [], 0

    def load_checkpoint(self, **kwargs):
        return copy.deepcopy(self.state)

    def checkpoint(self, state, *, metrics=(), progress=None):
        self.state = copy.deepcopy(state)
        self.rows.extend(metrics)
        self.saves += 1
        json.dumps(progress, allow_nan=False)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def tiny():
    return build_tiny_model(31, hidden_size=2, width=4, depth=1, return_bins=3, context_length=4)


def row(index, namespace="original", length=3):
    generator = torch.Generator().manual_seed(index + 100)
    return {"record_id": f"{namespace}/bootstrap:0/{index}", "episode_id": f"{namespace}/bootstrap:0",
        "arrival": index, "baseline": float(index) / 20., "bin": 1,
        "logits": torch.zeros(7), "executed_probabilities": torch.ones(1, 8) / 8.,
        "history": {"images": torch.rand(1, length, 3, 8, 8, generator=generator),
            "elapsed": torch.ones(1, length, 1) / 10., "executed_actions": torch.zeros(1, length, 4),
            "contexts": [[{"text": "Approach the ball and stop.", "issued_at": 0.}]]}}


@pytest.fixture
def prepared():
    config = PicarConfig(seed=71)
    config = replace(config, learning=replace(config.learning, temperature=2.8375266913896,
        action_variance=16., return_min=-1., return_max=1.))
    model = tiny()
    learner = MemoryLearner(model, config.learning, config.seeds(0))
    records = [row(0), row(1, length=1), row(2, length=2)]
    names = learner.banks["ltm"]
    params = {n: learner.params[n] for n in names}
    sums = {str(m): {n: torch.zeros_like(p).double() for n, p in params.items()} for m in (2, 4)}
    for record in records:
        batch = LearningBatch([record["history"]], torch.tensor([record["baseline"]]),
            torch.tensor([record["bin"]]), record["logits"].reshape(1, 7), torch.tensor([record["arrival"]]))
        for m in (2, 4):
            contribution = pm.model_fisher_diagonal(lambda p: predict_histories(model, p, batch.histories),
                params, learner.design(batch), method="sampled", draws=m,
                generator=torch.Generator().manual_seed(record["arrival"] + 10))
            for n, value in contribution.items():
                sums[str(m)][n] += value.double()
    learner.diagonal = {n: (v / len(records)).to(params[n]) for n, v in sums["4"].items()}
    learner.retired_count = len(records)
    anchor = _tensor_hash(params)
    data = {"train": InitializationFit._references(records), "validation": []}
    primary = {"stage": "done", "identity": {"config": config.to_dict(), "data": data},
        "learner": learner.state_dict(), "retired": [0, 1, 2], "eligible": [],
        "acquired_count": 9, "accepted_identity": {"parameters_sha256": anchor},
        "distill_receipt": {"stage": "test_fitted"},
        "fisher": {"count": len(records), "anchor_sha256": anchor, "nested_sums": sums,
            "subset_ids": [], "subset_sum": {}, "independent_sum": {}}}
    return model, config, records, primary


def make(prepared, store=None, *, records=None, model=None, **kwargs):
    live, config, original, primary = prepared
    return ref.FisherReference(model if model is not None else live, config,
        original if records is None else records, primary,
        store if store is not None else MemoryStore(), **kwargs)


def test_reference_equal_mean_counts_public_kernel_and_primary_preserved(prepared, monkeypatch):
    original = copy.deepcopy(prepared[3])
    reference = make(prepared, draws=16)
    model_before = reference.learner.state_dict()
    expected = {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in reference.ltm.items()}
    for record in reference.records:
        for n, value in reference._one(record).items():
            expected[n] += value.detach().double() / len(reference.records)
    public = pm.model_fisher_diagonal
    calls = []

    def sampled(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["method"] == "sampled" and kwargs["draws"] == 16
        return public(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("reference attempted fitting, retirement or an online EMA")

    monkeypatch.setattr(pm, "model_fisher_diagonal", sampled)
    monkeypatch.setattr(pm, "eviction_ema", forbidden)
    monkeypatch.setattr(MemoryLearner, "fit", forbidden)
    monkeypatch.setattr(MemoryLearner, "retire", forbidden)
    summary = reference.run()
    assert len(calls) == 3
    assert reference.s["count"] == reference.s["cursor"] == 3
    assert summary["counts"] == {"N": 3, "M": 16, "histories": 3, "model_score_draws": 48,
                                  "new_observations": 0, "validation_histories": 0}
    assert summary["normalization"]["history_weight"] == 1 / 3
    assert summary["normalization"]["summed_information_multiplier"] == 3
    assert summary["normalization"]["pi"] is None
    assert not summary["reference_is_oracle"]
    assert summary["primary_unchanged"] and summary["model_parameters_unchanged"]
    for n in expected:
        torch.testing.assert_close(reference.s["reference_diagonal"][n], expected[n], rtol=1e-15, atol=1e-15)
    equal(original, prepared[3])
    equal(model_before, reference.learner.state_dict())
    assert reference.store.rows == []
    json.dumps(summary, allow_nan=False)


def test_record_order_partition_and_committed_resume_are_identical(prepared):
    full = make(prepared, draws=16)
    full_summary = full.run()
    reversed_run = make(prepared, records=list(reversed(prepared[2])), draws=16, block_items=1)
    reversed_run.run()
    equal(full.s["reference_diagonal"], reversed_run.s["reference_diagonal"])
    store = MemoryStore()

    def stop():
        if store.state["count"] == 1:
            raise InterruptedError("saved exact next cursor")

    with pytest.raises(InterruptedError):
        make(prepared, store, draws=16, block_items=1, stopped=stop).run()
    assert store.state["cursor"] == 1
    torch.rand(12)
    resumed = make(prepared, store, draws=16, model=tiny())
    result = resumed.run()
    equal(full.s["reference_diagonal"], resumed.s["reference_diagonal"])
    equal(full_summary["comparisons"], result["comparisons"])
    assert resumed.seed == derive_component_seed(prepared[1].seed, ref.SEED_COMPONENT)
    assert not store.rows


def test_deadline_during_one_history_does_not_admit_partial_contribution(prepared, monkeypatch):
    reference = make(prepared)
    reference.save()

    def interrupted(record):
        raise ref.ReferenceLimit("interrupted within public Fisher")

    monkeypatch.setattr(reference, "_one", interrupted)
    with pytest.raises(ref.ReferenceLimit):
        reference.run()
    assert reference.s["count"] == reference.store.state["count"] == 0
    assert all(not bool(v.count_nonzero()) for v in reference.s["sums"].values())


def test_changed_conditioning_anchor_and_cursor_fail(prepared):
    records = copy.deepcopy(prepared[2])
    records[0]["history"]["images"].add_(.01)
    with pytest.raises(ValueError, match="conditioning"):
        make(prepared, records=records)
    reference = make(prepared)
    reference.save()
    reference.store.state["count"] = 1
    with pytest.raises(ValueError, match="cursor"):
        make(prepared, reference.store)
    primary = copy.deepcopy(prepared[3])
    primary["learner"]["anchor"][next(iter(reference.ltm))].add_(1)
    with pytest.raises(ValueError, match="anchor"):
        ref.FisherReference(prepared[0], prepared[1], prepared[2], primary, MemoryStore())


def seal(path, config, summary, state=None):
    with RunStore(path, config, checkpoint_policy="journal-v1") as store:
        store.checkpoint(state or {"stage": "done"})
        store.complete({"schema_version": 1, **summary})
    return read_completed_unit(path)


def source_fixture(tmp_path, prepared, *, charge=30000., seconds=.1):
    root = tmp_path / "campaign"
    primary = prepared[3]
    fit_state = {**copy.deepcopy(primary), "stage": "fit_done"}
    fit = seal(root / "units/fit-selected", {"schema_version": 1},
        {"kind": "initialization_fit"}, fit_state)
    bootstrap = seal(root / "units/bootstrap-selected", {"schema_version": 1},
        {"kind": "initialization_bootstrap", "settings": {"fit": fit["path"]},
         "active_seconds": seconds}, primary)
    main = seal(root / "orchestration", {"schema_version": 1, "version": "plan19-phase3-fix1-v1",
        "source_sha256": {}}, {"kind": "initialization_campaign", "repair_active_seconds": charge,
            "selected_fit": fit["path"], "initial_asset": bootstrap["path"]})
    return root, {"root": root, "main": main, "main_charge": charge, "fit": fit, "bootstrap": bootstrap}


def test_running_main_returns_pending_without_torch_import_or_output(tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("running main admitted Torch/GPU work")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "sidecar"
    summary = ref.run(tmp_path / "running", output)
    assert summary["status"] == "pending"
    assert not output.exists()


def test_insufficient_shared_budget_exits_before_model_or_torch(prepared, tmp_path, monkeypatch):
    root, _ = source_fixture(tmp_path, prepared, charge=35900., seconds=175.)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("insufficient budget admitted Torch/GPU work")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = ref.run(root, tmp_path / "sidecar")
    assert result["status"] == "pending"
    assert result["combinedrepaircharged_seconds"] <= 36000
    assert not (tmp_path / "sidecar").exists()


def test_cli_pipeline_cpu_substitutes_only_loader_preserves_all_source_bytes(prepared, tmp_path, monkeypatch):
    root, source = source_fixture(tmp_path, prepared)
    files = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(ref, "reconstruct_training", lambda *args: (prepared[2], {"test": True}))
    monkeypatch.setattr(ref, "_load_model", lambda *args: tiny())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    output = tmp_path / "reference"
    result = ref.run(root, output)
    assert result["status"] == "completed"
    assert result["kind"] == "initialization_mc_reference"
    assert result["combinedrepaircharged_seconds"] <= 36000
    assert result["sidecar_charged_seconds"] <= min(1200, 36000 - source["main_charge"])
    assert result["sidecar_charged_seconds"] >= result["active_seconds"] + ref.CLOSEOUT_SECONDS
    for path, content in files.items():
        assert path.read_bytes() == content
    read_completed_unit(output)
    marker = (output / "COMPLETED").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("completed sidecar resume loaded a model")

    monkeypatch.setattr(ref, "_load_model", forbidden)
    equal(result, ref.run(root, output, resume=True))
    assert marker == (output / "COMPLETED").read_bytes()
    with pytest.raises(FileExistsError):
        ref.run(root, output)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_reconstruction_uses_selected_refs_exact_offsets_and_ignores_later_chunks(prepared, tmp_path, monkeypatch):
    from numerical_experiment.picar_initialization import data
    from numerical_experiment.picar_phase2_rescue import data as old_data
    original_row = row(0)
    raw_original = {**original_row, "episode_id": "bootstrap:0"}
    raw_chunk_row = {**row(1), "episode_id": "bootstrap:0", "arrival": 0}
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "COMPLETED").write_text("teacher identity")
    original = {"records": [raw_original], "episodes": [], "acquired_count": 5}
    expected = data.namespace_data(original, "original")["records"]
    expected += data.namespace_data({"records": [raw_chunk_row], "episodes": [], "acquired_count": 7},
                                   "collection-1", arrival_offset=5)["records"]
    fitted = {"stage": "fit_done", "accepted_identity": {"parameters_sha256": "fitted"},
        "identity": {"data": {"train": InitializationFit._references(expected), "validation": []},
                     "config": prepared[1].to_dict()}}
    primary = {"identity": copy.deepcopy(fitted["identity"]), "accepted_identity": fitted["accepted_identity"]}
    root = tmp_path / "campaign"
    seal(root / "units/collection-first", {"schema_version": 1, "settings": {"index": 1}},
        {"kind": "initialization_collection"}, {"stage": "collection_done", "eligible": [raw_chunk_row],
            "episodes": [], "acquired_count": 7})
    seal(root / "units/collection-unrelated", {"schema_version": 1, "settings": {"index": 2}},
        {"kind": "initialization_collection"}, {"stage": "invalid_must_not_load"})
    source = {"root": root, "main": {"stored_config": {"teacher_source": str(teacher),
        "teacher_completion_sha256": hashlib.sha256((teacher / "COMPLETED").read_bytes()).hexdigest()}}}
    monkeypatch.setattr(old_data, "restore_demonstrations", lambda path: copy.deepcopy(original))
    actual, identity = ref.reconstruct_training(source, fitted, primary)
    equal(actual, sorted(expected, key=lambda r: r["record_id"]))
    assert sorted(r["arrival"] for r in actual) == [0, 5]
    assert set(identity["collection_completion_sha256"]) == {"1"}
    fitted["identity"]["data"]["train"][0]["history_sha256"] = "different"
    primary["identity"] = copy.deepcopy(fitted["identity"])
    with pytest.raises(ValueError, match="hash"):
        ref.reconstruct_training(source, fitted, primary)


def test_simple_gaussian_score_reference_approaches_exact_diagonal(monkeypatch):
    parameter = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
    design = pm.LikelihoodDesign(torch.tensor([-1., 1.], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), 2.8375266913896, torch.eye(7, dtype=torch.float64) * 16.)

    def predict(parameters):
        return {"return_logits": torch.zeros(1, 2, dtype=torch.float64),
                "action_mean": parameters["mean"].expand(1, 7)}

    exact = pm.model_fisher_diagonal(predict, {"mean": parameter}, design)
    expected = 7 * float(design.weights.mean()) / 16
    assert float(exact["mean"]) == pytest.approx(expected)
    sampled = pm.model_fisher_diagonal(predict, {"mean": parameter}, design, method="sampled",
        draws=2048, generator=torch.Generator().manual_seed(731))
    assert float(sampled["mean"]) == pytest.approx(expected, rel=.1)
    assert parameter.item() == .3


def test_cli_mid_reference_resume_accounts_prior_work_and_never_duplicates(prepared, tmp_path, monkeypatch):
    root, _ = source_fixture(tmp_path, prepared)
    monkeypatch.setattr(ref, "reconstruct_training", lambda *args: (prepared[2], {}))
    monkeypatch.setattr(ref, "_load_model", lambda *args: tiny())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = ref.FisherReference._one
    calls = []

    def interrupt(self, record):
        calls.append(record["record_id"])
        if len(calls) == 2:
            raise InterruptedError("test bounded stop within second history")
        return original(self, record)

    monkeypatch.setattr(ref.FisherReference, "_one", interrupt)
    output = tmp_path / "reference"
    pending = ref.run(root, output)
    assert pending["status"] == "pending" and not (output / "COMPLETED").exists()
    with RunStore(output, json.loads((output / "config.json").read_text()), resume=True,
                  checkpoint_policy="journal-v1") as store:
        saved = store.load_checkpoint(trusted=True)
    assert saved["cursor"] == 1
    assert saved["charged_seconds"] == pending["sidecar_charged_seconds"]
    monkeypatch.setattr(ref.FisherReference, "_one", original)
    result = ref.run(root, output, resume=True)
    assert result["status"] == "completed"
    assert result["active_seconds"] >= pending["active_seconds"]
    assert result["sidecar_charged_seconds"] >= pending["sidecar_charged_seconds"] + ref.CLOSEOUT_SECONDS
    with RunStore(output, json.loads((output / "config.json").read_text()), resume=True,
                  checkpoint_policy="journal-v1") as store:
        final = store.load_checkpoint(trusted=True)
    whole = make(prepared)
    whole.run()
    equal(whole.s["reference_diagonal"], final["reference_diagonal"])
    assert final["cursor"] == final["count"] == 3


def test_preparation_deadline_preserves_pending_no_scores_then_resume(prepared, tmp_path, monkeypatch):
    root, _ = source_fixture(tmp_path, prepared)
    monkeypatch.setattr(ref, "reconstruct_training", lambda *args: (prepared[2], {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_time(*args):
        raise ref.ReferenceLimit("test model-load deadline")

    monkeypatch.setattr(ref, "_load_model", no_time)
    output = tmp_path / "reference"
    pending = ref.run(root, output)
    assert pending["status"] == "pending" and pending["counts"]["histories"] == 0
    assert not (output / "COMPLETED").exists()
    monkeypatch.setattr(ref, "_load_model", lambda *args: tiny())
    result = ref.run(root, output, resume=True)
    assert result["status"] == "completed"
    assert result["active_seconds"] >= pending["active_seconds"]
    assert result["sidecar_charged_seconds"] >= pending["sidecar_charged_seconds"] + ref.CLOSEOUT_SECONDS


def test_alarm_bounds_blocking_work_and_restores_handler():
    import time
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(ref.ReferenceLimit, match="deadline"):
        with ref._deadline(.01):
            time.sleep(.05)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_primary_subset_comparison_uses_only_same_histories_no_extra_draws(prepared):
    reference = make(prepared)
    primary = reference.primary
    primary["fisher"]["subset_ids"] = [prepared[2][0]["record_id"]]
    one = reference._one(prepared[2][0])
    primary["fisher"]["subset_sum"] = {n: v.detach().double() for n, v in one.items()}
    primary["fisher"]["independent_sum"] = {n: v.detach().double() * 2 for n, v in one.items()}
    result = reference.run()
    subset = result["comparisons"]["primary_4_subset_vs_reference"]
    assert subset["relative_l2"] == 0
    assert result["comparisons"]["original_independent_4_subset_vs_reference"]["relative_l2"] == 1
    assert result["counts"]["model_score_draws"] == 3 * 16


@pytest.mark.parametrize("draws", [0, -1, True])
def test_invalid_draw_counts_fail_before_source_or_output(tmp_path, draws):
    with pytest.raises(ValueError, match="draws"):
        ref.run(tmp_path / "main", tmp_path / "output", draws=draws)
    assert not (tmp_path / "output").exists()
