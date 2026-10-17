import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mnist_experiment.akaike_score_gate import assets
from mnist_experiment.akaike_score_gate.artifacts import RunLimit, StopExecution, UnitStore, file_hash
from mnist_experiment.akaike_score_gate.config import Study
from src.parameters import ParameterLayout


def study(**overrides):
    raw = Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json").mapping()
    return Study({**raw, "source_size": 40, "source_validation_size": 30,
                  "online_pool_size": 300, "reference_pool_size": 80,
                  "development_evaluation_size": 100, "final_evaluation_size": 100,
                  "source_epochs": 2, "source_epoch_ceiling": 3,
                  "initial_fisher_samples": 6, "fisher_chunk_size": 2,
                  "source_batch_size": 16, **overrides})


def store_at(path, config=None, limit=None):
    store = UnitStore(path, config or study(), limit=limit)
    # Other agents may edit unrelated module sources while these unit tests run.
    store.sources = {}
    return store


class Dataset:
    def __init__(self, count, seed):
        self.targets = torch.arange(count) % 10
        self.data = torch.randint(0, 256, (count, 28, 28), dtype=torch.uint8,
                                  generator=torch.Generator().manual_seed(seed))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(1, 10)

    def forward(self, batch):
        return self.output(batch.mean((1, 2, 3))[:, None])


def tiny_model(seed, width=5, *, device="cpu", dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = TinyModel().to(device=device, dtype=dtype)
    return model, ParameterLayout.from_module(model)


@pytest.fixture
def synthetic(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    train, test = Dataset(1000, 10), Dataset(200, 20)
    monkeypatch.setattr(assets, "load_mnist_datasets", lambda *a, **kw: (train, test))
    monkeypatch.setattr(assets, "build_model", tiny_model)
    yield train, test
    torch.set_num_threads(old_threads)


def read(path, name="assets.pt"):
    return torch.load(path / name, weights_only=True, map_location="cpu")


def scalar_fisher(calls):
    def estimate(model, layout, inputs, weights=None, *, sample_size, seed, **kwargs):
        calls.append(seed)
        choices = torch.arange(len(inputs)) if weights is None else torch.nonzero(weights).flatten()
        rng = torch.Generator().manual_seed(seed)
        ids = choices[torch.randint(len(choices), (sample_size,), generator=rng)]
        scale = 1 + float(layout.flatten_module(model, detach=True).double().square().sum())
        return torch.eye(layout.total_numel, dtype=torch.float64) * scale, {
            "input_indices": ids, "seeds": {"inputs": seed, "labels": seed + 1},
            "min_eigenvalue_before_psd": scale, "negative_spectral_mass": 0.0,
            "psd_projection_frobenius": 0.0,
        }
    return estimate


def test_exact_disjoint_partitions_and_named_phase_pairing():
    config = Study.from_path("mnist_experiment/akaike_score_gate/configs/default.json")
    train_targets, test_targets = torch.arange(60000) % 10, torch.arange(10000) % 10
    for family in ("rotation", "mixture"):
        seeds = assets.named_seeds(config, "phase1", 2, family)
        first = assets.partition_ids(config, train_targets, test_targets, family=family, seeds=seeds)
        second = assets.partition_ids(config, train_targets, test_targets, family=family, seeds=seeds)
        assert first["hash"] == second["hash"]
        ids = first["ids"]
        assert [len(ids[name]) for name in ("source", "source_validation", "online", "reference",
                                          "development_evaluation", "final_evaluation")] == [28000, 2000, 10000, 16000, 4000, 10000]
        combined = torch.cat(list(ids.values()))
        assert combined.unique().numel() == combined.numel()
        if family == "mixture":
            assert not (train_targets[ids["source"]] == 9).any()
            assert not (train_targets[ids["source_validation"]] == 9).any()
        else:
            assert set(train_targets[ids["source"]].tolist()) == set(range(10))
        seed_sets = [set(assets.named_seeds(config, phase, 2, family).values())
                     for phase in ("phase1", "phase3", "phase4", "phase5")]
        assert len(set.union(*seed_sets)) == sum(map(len, seed_sets))


def test_mixture_nested_prefixes_bernoulli_counts_and_separate_warmup():
    config = study(mixture_steps=10, mixture_samples=[18, 182], scarcity_samples=8,
                   online_pool_size=10000)
    targets = torch.arange(12000) % 10
    seeds = assets.named_seeds(config, "phase4", 1, "mixture")
    plan = {"ids": {"online": torch.arange(10000)}}
    result = assets.stream_ids(config, targets, plan, family="mixture", seeds=seeds)
    big = result["streams"]["mixture_m182"]
    uniforms = torch.rand(10, 182, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(seeds["stream_membership"]))
    expected_membership = uniforms < torch.tensor(big["positions"], dtype=torch.float64)[:, None]
    assert torch.equal(big["targets"] == 9, expected_membership)
    assert big["positions"] == pytest.approx([i / 100 for i in range(1, 11)])
    for size, expected in ((8, 4.4), (18, 9.9), (182, 100.1)):
        stream = result["streams"][f"mixture_m{size}"]
        assert torch.equal(stream["observation_ids"], big["observation_ids"][:, :size])
        assert stream["exposure"]["expected_nines"] == pytest.approx(expected)
        assert stream["exposure"]["duplicates"] == 0
        assert not torch.isin(stream["observation_ids"], result["warmup_ids"]).any()
    assert len(result["warmup_ids"]) == 8
    assert not (targets[result["warmup_ids"]] == 9).any()


def test_rotation_schedules_share_observations_and_cover_all_transitions():
    config = study(rotation_transitions_per_arrow=40)
    targets = torch.arange(1000) % 10
    seeds = assets.named_seeds(config, "phase1", 0, "rotation")
    result = assets.stream_ids(config, targets, {"ids": {"online": torch.arange(1000)}},
                                family="rotation", seeds=seeds)
    linear, sigmoid = (result["streams"][f"rotation_{kind}"] for kind in ("linear", "sigmoid"))
    assert torch.equal(linear["observation_ids"], sigmoid["observation_ids"])
    assert torch.equal(linear["targets"], sigmoid["targets"])
    assert linear["observation_ids"].shape == (120, 4)
    assert [linear["positions"][i] for i in (39, 79, 119)] == [30, 0, 30]
    assert [sigmoid["positions"][i] for i in (39, 79, 119)] == [30, 0, 30]
    assert linear["positions"][0] != sigmoid["positions"][0]
    assert linear["exposure"]["duplicates"] == 0


def test_exhausted_pools_report_duplicates_instead_of_changing_prevalence():
    config = study(mixture_steps=10, mixture_samples=[18, 182], scarcity_samples=8)
    targets = torch.arange(40) % 10
    seeds = assets.named_seeds(config, "phase4", 0, "mixture")
    result = assets.stream_ids(config, targets, {"ids": {"online": torch.arange(40)}},
                                family="mixture", seeds=seeds)
    stream = result["streams"]["mixture_m182"]
    assert stream["exposure"]["duplicates"] > 0
    assert stream["exposure"]["presentations"] == 1820
    assert not torch.isin(stream["observation_ids"], result["warmup_ids"]).any()
    ids = assets._draw_pool(torch.arange(3), 8, 5)
    assert ids[:3].unique().numel() == ids[3:6].unique().numel() == 3


def test_source_only_is_sealed_and_skips_fisher_warmup_and_test_inference(tmp_path, synthetic, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("source-only computation crossed a forbidden dependency")

    monkeypatch.setattr(assets, "model_fisher", unexpected)
    monkeypatch.setattr(assets, "fit_restricted", unexpected)
    monkeypatch.setattr(assets, "stream_ids", unexpected)
    monkeypatch.setattr(assets, "evaluate", unexpected)
    store = store_at(tmp_path)
    path = assets.ensure_assets(store, "phase1", 0, "mixture", 5, source_only=True)
    data = read(path)
    summary = json.loads((path / "summary.json").read_text())
    assert (path / "COMPLETED").exists()
    assert "initial_fisher" not in data and "streams" not in data
    assert "evaluation_inputs" not in data
    assert summary["source_nll"] == data["source_quality"]["nll"]
    assert summary["source_accuracy"] == data["source_quality"]["accuracy"]
    assert summary["completed_epochs"] == 2
    assert summary["epoch_selection_panel"] == "source_validation"
    assert len(summary["epoch_quality"]) == 2
    checkpoint = read(path, "source_checkpoint.pt")
    assert {"optimizer", "order_rng_state", "torch_rng_state", "cuda_rng_states"} <= checkpoint.keys()
    digest = file_hash(path / "assets.pt")
    assert assets.ensure_assets(store, "phase1", 0, "mixture", 5, source_only=True) == path
    assert file_hash(path / "assets.pt") == digest
    with pytest.raises(RuntimeError, match="completed unit exists"):
        assets.ensure_assets(store, "phase1", 0, "mixture", 5, source_only=True, resume=False)


class StopAtCheckpoint(RunLimit):
    def __init__(self, root, name, field, value):
        super().__init__()
        self.root, self.name, self.field, self.value = root, name, field, value

    def check(self):
        super().check()
        for path in self.root.rglob(self.name):
            if ".incomplete" in path.parts and read(path.parent, path.name)[self.field] == self.value:
                raise StopExecution("test checkpoint interruption")


def assert_same_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_epoch_resume_restores_adam_order_and_rng(tmp_path, synthetic):
    baseline = assets.ensure_assets(store_at(tmp_path / "baseline"), "phase1", 0, "rotation", 5,
                                     source_only=True)
    root = tmp_path / "interrupted"
    with pytest.raises(StopExecution):
        assets.ensure_assets(store_at(root, limit=StopAtCheckpoint(root, "source_checkpoint.pt", "epoch", 1)),
                             "phase1", 0, "rotation", 5, source_only=True)
    resumed = assets.ensure_assets(store_at(root), "phase1", 0, "rotation", 5, source_only=True)
    assert_same_state(read(baseline)["initial_model"], read(resumed)["initial_model"])
    a, b = read(baseline, "source_checkpoint.pt"), read(resumed, "source_checkpoint.pt")
    assert_same_state(a["model"], b["model"])
    assert torch.equal(a["order_rng_state"], b["order_rng_state"])
    assert [r["validation_nll"] for r in a["metrics"]] == [r["validation_nll"] for r in b["metrics"]]
    for key in a["optimizer"]["state"]:
        assert_same_state(a["optimizer"]["state"][key], b["optimizer"]["state"][key])


def test_extended_source_uses_last_optimizer_checkpoint_not_best_state(tmp_path, synthetic):
    store = store_at(tmp_path / "extended")
    base = assets.ensure_assets(store, "phase1", 0, "rotation", 5, source_only=True)
    digest = file_hash(base / "source_checkpoint.pt")
    extended = assets.ensure_assets(store, "phase1", 0, "rotation", 5,
                                    source_only=True, extended_epochs=True)
    fresh = assets.ensure_assets(store_at(tmp_path / "fresh"), "phase1", 0, "rotation", 5,
                                 source_only=True, extended_epochs=True)
    assert_same_state(read(extended, "source_checkpoint.pt")["model"],
                      read(fresh, "source_checkpoint.pt")["model"])
    assert read(extended, "source_checkpoint.pt")["epoch"] == 3
    assert file_hash(base / "source_checkpoint.pt") == digest
    assert len(json.loads((extended / "summary.json").read_text())["parent_hashes"]) == 1


def test_equal_extended_epoch_ceiling_still_seals_checkpoint(tmp_path, synthetic):
    store = store_at(tmp_path, study(source_epochs=1, source_epoch_ceiling=1))
    base = assets.ensure_assets(store, "phase1", 0, "rotation", 5, source_only=True)
    extended = assets.ensure_assets(store, "phase1", 0, "rotation", 5,
                                    source_only=True, extended_epochs=True)
    assert extended != base
    assert (extended / "COMPLETED").is_file()
    assert_same_state(read(base)["initial_model"], read(extended)["initial_model"])


def test_source_epoch_selection_uses_validation_minimum_not_last_epoch(tmp_path, synthetic, monkeypatch):
    values = iter([{"nll": 3.0, "accuracy": 0.1}, {"nll": 1.0, "accuracy": 0.8},
                   {"nll": 2.0, "accuracy": 0.2}])
    monkeypatch.setattr(assets, "source_quality", lambda *a, **kw: next(values))
    path = assets.ensure_assets(store_at(tmp_path), "phase1", 0, "rotation", 5, source_only=True)
    summary = json.loads((path / "summary.json").read_text())
    assert summary["selected_epoch"] == 1
    assert summary["source_nll"] == 1.0
    assert summary["source_accuracy"] == 0.8
    checkpoint = read(path, "source_checkpoint.pt")
    assert_same_state(read(path)["initial_model"], checkpoint["best_model"])
    assert any(not torch.equal(checkpoint["model"][key], checkpoint["best_model"][key])
               for key in checkpoint["model"])


def test_widths_share_data_and_source_seeds_and_old_reference_chart(tmp_path, synthetic):
    store = store_at(tmp_path, study(source_epochs=1))
    paths = [assets.ensure_assets(store, "phase1", 0, "rotation", width, source_only=True)
             for width in (5, 6, 0)]
    data = [read(path) for path in paths]
    assert data[0]["seeds"] == data[1]["seeds"] == data[2]["seeds"]
    for key in data[0]["partition_ids"]:
        assert torch.equal(data[0]["partition_ids"][key], data[1]["partition_ids"][key])
        assert torch.equal(data[0]["partition_ids"][key], data[2]["partition_ids"][key])
    assert data[2]["parameter_layout"]["total_numel"] == 487


def test_full_assets_reuse_source_and_warmup_once_then_recompute_fisher(tmp_path, synthetic, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "model_fisher", scalar_fisher(calls))
    warmup_calls = []

    def fit(model, layout, inputs, targets, weights, anchor, precision, indices, **kwargs):
        warmup_calls.append(len(targets))
        assert len(indices) == layout.total_numel
        assert not (targets == 9).any()
        assert len(targets) == 8
        assert torch.equal(anchor, layout.flatten_module(model, detach=True))
        scale = 1 + float(anchor.double().square().sum())
        torch.testing.assert_close(precision, 19 * scale * torch.eye(len(anchor), dtype=torch.float64))
        with torch.no_grad():
            next(model.parameters()).add_(0.05)
        return {"converged": True, "displacement": torch.ones_like(anchor)}

    monkeypatch.setattr(assets, "fit_restricted", fit)
    store = store_at(tmp_path)
    source = assets.ensure_assets(store, "phase1", 0, "mixture", 5, source_only=True)
    source_digest = file_hash(source / "assets.pt")
    path = assets.ensure_assets(store, "phase1", 0, "mixture", 5)
    data = read(path)
    assert warmup_calls == [8]
    assert len(calls) == 6
    assert file_hash(source / "assets.pt") == source_digest
    assert_same_state(data["source_model"], read(source)["initial_model"])
    assert any(not torch.equal(data["source_model"][k], data["initial_model"][k]) for k in data["source_model"])
    assert data["parent_hashes"]
    assert data["initial_fisher"].dtype == torch.float64
    assert data["reference_inputs"].shape == (80, 1, 28, 28)
    large, small = data["streams"]["mixture_m4"], data["streams"]["mixture_m2"]
    assert torch.equal(small["inputs"], large["inputs"][:, :2])
    assert torch.equal(small["observation_ids"], large["observation_ids"][:, :2])
    assert set(data["evaluation_ids"].tolist()).isdisjoint(data["reference_ids"].tolist())
    rows = json.loads((path / "metrics.json").read_text())
    assert [r["stage"] for r in rows] == ["source_initialization", "post_warmup"]
    summary = json.loads((path / "summary.json").read_text())
    assert summary["warmup_ridge"] == 0
    assert summary["warmup_exposure"]["presentations"] == 8
    assert summary["source_nll"] == read(source)["source_quality"]["nll"]
    assert summary["evaluation_split"] == "development_train"

    def all_cpu(tree):
        if isinstance(tree, torch.Tensor):
            assert tree.device.type == "cpu"
        elif isinstance(tree, dict):
            for value in tree.values():
                all_cpu(value)
        elif isinstance(tree, (tuple, list)):
            for value in tree:
                all_cpu(value)
    all_cpu(data)


def test_fisher_chunk_resume_skips_completed_chunks(tmp_path, synthetic, monkeypatch):
    calls = []
    monkeypatch.setattr(assets, "model_fisher", scalar_fisher(calls))
    root = tmp_path / "resume"
    with pytest.raises(StopExecution):
        assets.ensure_assets(store_at(root, limit=StopAtCheckpoint(root, "source_fisher_checkpoint.pt", "count", 2)),
                             "phase1", 0, "rotation", 5)
    assert len(calls) == 1
    path = assets.ensure_assets(store_at(root), "phase1", 0, "rotation", 5)
    assert len(calls) == len(set(calls)) == 3
    baseline = assets.ensure_assets(store_at(tmp_path / "baseline"), "phase1", 0, "rotation", 5)
    assert_same_state(read(path)["initial_model"], read(baseline)["initial_model"])
    assert torch.equal(read(path)["initial_fisher"], read(baseline)["initial_fisher"])
    assert torch.equal(read(path)["initial_fisher_observation_ids"], read(baseline)["initial_fisher_observation_ids"])
    assert read(path)["streams"]["rotation_linear"]["inputs"].shape == (3, 4, 1, 28, 28)


def test_real_core_fisher_and_warmup_contract_on_tiny_cpu_model(tmp_path, synthetic):
    path = assets.ensure_assets(store_at(tmp_path, study(initial_fisher_samples=2, fisher_chunk_size=1,
                                                       max_iter=1, max_eval=3, source_epochs=1)),
                                 "phase1", 0, "mixture", 5)
    matrix = read(path)["initial_fisher"]
    assert torch.isfinite(matrix).all()
    torch.testing.assert_close(matrix, matrix.mT)
    assert torch.linalg.eigvalsh(matrix).min() >= -1e-12
    summary = json.loads((path / "summary.json").read_text())
    assert summary["initial_fisher"]["input_draw_count"] == 2
    assert summary["warmup"]["active_dimension"] == 20


def test_final_assets_use_test_ids_only_after_source_selection(tmp_path, synthetic, monkeypatch):
    monkeypatch.setattr(assets, "model_fisher", scalar_fisher([]))
    train, test = synthetic
    path = assets.ensure_assets(store_at(tmp_path), "phase5", 0, "rotation", 5)
    data = read(path)
    assert (data["evaluation_ids"] >= len(train.targets)).all()
    assert torch.equal(data["evaluation_targets"], test.targets[data["evaluation_ids"] - len(train.targets)])
    assert json.loads((path / "summary.json").read_text())["evaluation_split"] == "test"


def test_completed_corruption_is_not_repaired(tmp_path, synthetic):
    path = assets.ensure_assets(store_at(tmp_path), "phase1", 0, "rotation", 5, source_only=True)
    (path / "assets.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        assets.ensure_assets(store_at(tmp_path), "phase1", 0, "rotation", 5, source_only=True)
