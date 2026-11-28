"""Tiny synthetic CPU fixtures; never launch the Fix 5A experiment."""

import copy
from dataclasses import replace
import json

import pytest
import torch

from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_head_awac import initialization
from numerical_experiment.picar_head_awac.archive import Archive
from numerical_experiment.picar_head_awac.learner import Learner, Recipe
from numerical_experiment.picar_head_awac.model import Heads
from numerical_experiment.picar_head_rank import numerics


def _exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _exact(left, right)
    else:
        assert actual == expected


class MemoryStore:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state)
        self.cursors = []

    def load_checkpoint(self, *, trusted):
        assert trusted
        return copy.deepcopy(self.state)

    def checkpoint(self, state, *, progress):
        assert progress["n"] == state["cursor"]
        json.dumps(progress, allow_nan=False)
        self.cursors.append(state["cursor"])
        self.state = copy.deepcopy(state)


@pytest.fixture
def fixture(monkeypatch):
    def heads():
        return Heads(hidden=3, width=2, seed=991)

    monkeypatch.setattr(numerics, "Heads", heads)
    monkeypatch.setattr(initialization, "Heads", heads)
    fitted = {"selected_heads": copy.deepcopy(heads().state_dict())}
    generator = torch.Generator().manual_seed(992)
    records = [{"feature": torch.randn(3, generator=generator),
                "next_feature": torch.randn(3, generator=generator),
                "action": torch.randn(7, generator=generator),
                "reward": .2 * (i % 3 - 1), "discount": .9,
                "return": .1 * (i % 5), "arrival": i, "record_id": f"train-{i}"}
               for i in range(65)]
    recipe = Recipe(capacity=256, batch_size=64, rounds=4, steps=3, rank=64,
                    temperature=1., action_variance=16., critic_variance=16.,
                    integration_draws=4, exploration_std=2., fd_strength=numerics.FD_STRENGTH)
    return fitted, records, recipe


def test_one_shared_equal_weight_stream_matches_old_selected_geometry(fixture, monkeypatch):
    fitted, records, recipe = fixture
    records = records[:18]
    old_store = MemoryStore()
    old = initialization.initial_archive(fitted, records, replace(recipe, rank=10), old_store, lambda: None)
    assert old_store.cursors == [16, 18]
    observed = []
    update = Archive.update

    def observe(archive, factor, alpha, beta):
        observed.append((archive.rank, factor.data_ptr(), factor.clone(), alpha, beta))
        return update(archive, factor, alpha, beta)

    monkeypatch.setattr(Archive, "update", observe)
    store = MemoryStore()
    before = copy.deepcopy((fitted, records))
    global_rng = torch.get_rng_state().clone()
    rebuilt = numerics.rebuild_initial(fitted, records, recipe, store, lambda: None,
                                      reference_archive=old["learner"]["archive"])
    assert rebuilt.keys() == {"archives", "diagnostics", "cursor"}
    assert rebuilt["cursor"] == 18 and store.cursors == [16, 18]
    assert rebuilt["diagnostics"]["reference_rank10"]["matches"]
    for i in range(18):
        calls = observed[4 * i:4 * (i + 1)]
        assert [call[0] for call in calls] == list(numerics.RANKS)
        assert len({call[1] for call in calls}) == 1
        for rank, _, factor, alpha, beta in calls:
            assert factor.shape[1] == 11
            assert alpha == i / (i + 1) and beta == 1 / (i + 1)
            _exact(factor, calls[0][2])
    _exact(rebuilt["archives"]["10"], old["learner"]["archive"])
    _exact(store.state["fisher_rng"], old["learner"]["rng"]["fisher"])
    _exact(store.state["archives"], rebuilt["archives"])
    for rank in numerics.RANKS:
        _exact(rebuilt["archives"][str(rank)]["diagonal"], old["learner"]["archive"]["diagonal"])
        row = rebuilt["diagnostics"]["ranks"][str(rank)]
        assert len(row["factor_eigenvalues"]) <= rank
        assert row["diagonal_vs_rank10_max_abs_error"] == 0
    json.dumps({k: v for k, v in rebuilt.items() if k != "archives"}, allow_nan=False)
    _exact((fitted, records), before)
    _exact(torch.get_rng_state(), global_rng)


def test_interrupted_transaction_resume_replays_unsaved_record_and_rng_exactly(fixture, tmp_path, monkeypatch):
    fitted, records, recipe = fixture
    records = records[:19]
    expected_store = MemoryStore()
    expected = numerics.rebuild_initial(fitted, records, recipe, expected_store, lambda: None)
    root = tmp_path / "initial"
    identity = {"fixture": "Fix5A-shared-stream"}
    factor = numerics._initial_factor
    calls = []

    def interrupt(learner, record):
        calls.append(record["record_id"])
        if len(calls) == 18:
            raise InterruptedError("after saved record 16 and unsaved record 17")
        return factor(learner, record)

    monkeypatch.setattr(numerics, "_initial_factor", interrupt)
    with RunStore(root, identity, checkpoint_policy="journal-v1", reserve_bytes=0) as store:
        with pytest.raises(InterruptedError):
            numerics.rebuild_initial(fitted, records, recipe, store, lambda: None)
        saved = store.load_checkpoint(trusted=True)
        assert saved["cursor"] == 16
        assert set(saved["archives"]) == {"0", "10", "32", "64"}
    resumed_records = []

    def resume_factor(learner, record):
        resumed_records.append(record["record_id"])
        return factor(learner, record)

    monkeypatch.setattr(numerics, "_initial_factor", resume_factor)
    with RunStore(root, identity, resume=True, checkpoint_policy="journal-v1", reserve_bytes=0) as store:
        resumed = numerics.rebuild_initial(fitted, records, recipe, store, lambda: None)
        _exact(resumed, expected)
        _exact(store.load_checkpoint(trusted=True), expected_store.state)
        assert resumed_records == [f"train-{i}" for i in range(16, 19)]
        monkeypatch.setattr(numerics, "_initial_factor", lambda *args: pytest.fail("final cursor generated more factors"))
        _exact(numerics.rebuild_initial(fitted, records, recipe, store, lambda: None), expected)


@pytest.mark.parametrize("change", ["recipe", "records", "fitted", "cursor", "diagonal", "ranks"])
def test_resume_rejects_changed_identity_or_invalid_archive_state(fixture, change):
    fitted, records, recipe = fixture
    records = records[:2]
    store = MemoryStore()
    numerics.rebuild_initial(fitted, records, recipe, store, lambda: None)
    if change == "recipe":
        recipe = replace(recipe, temperature=2.)
    elif change == "records":
        records = list(reversed(records))
    elif change == "fitted":
        fitted["selected_heads"]["actor.ltm.bias"][0] += .1
    elif change == "cursor":
        store.state["cursor"] = 100
    elif change == "diagonal":
        store.state["archives"]["0"]["diagonal"][0] += 1
    else:
        del store.state["archives"]["64"]
    with pytest.raises(ValueError):
        numerics.rebuild_initial(fitted, records, recipe, store, lambda: None)


def test_rebuild_rejects_empty_records_and_wrong_reference(fixture):
    fitted, records, recipe = fixture
    with pytest.raises(ValueError, match="training records"):
        numerics.rebuild_initial(fitted, [], recipe, MemoryStore(), lambda: None)
    store = MemoryStore()
    rebuilt = numerics.rebuild_initial(fitted, records[:2], recipe, store, lambda: None)
    wrong = copy.deepcopy(rebuilt["archives"]["10"])
    wrong["diagonal"] += 1.
    with pytest.raises(ValueError, match="reference initialization diagonal mismatch"):
        numerics.rebuild_initial(fitted, records[:2], recipe, store, lambda: None, wrong)


@pytest.mark.parametrize("rank,method", [(0, "td_replay"), (0, "td_fd"), (64, "td_fd")])
def test_new_learner_is_exact_old_fresh_control_bootstrap(fixture, rank, method):
    fitted, records, recipe = fixture
    recipe = replace(recipe, rank=rank)
    rebuilt = numerics.rebuild_initial(fitted, records[:2], recipe, MemoryStore(), lambda: None)
    archives = rebuilt["archives"]
    before = copy.deepcopy(archives)
    seed = 199101
    heads = numerics.Heads().cpu()
    heads.load_state_dict(fitted["selected_heads"])
    old = Learner(heads, recipe, seed, method)
    if rank == 0:
        # Exact frozen campaign convention: diagonal control from the rank-ten
        # selected CPU geometry, without loading the initializer learner.
        source = archives["10"]
        old.archive = Archive(source["diagonal"].cpu(), source["factor"][:, :0].cpu(), rank=0)
    else:
        source = archives[str(rank)]
        old.archive = Archive(source["diagonal"].cpu(), source["factor"].cpu(), rank=rank)
    actual = numerics.new_learner(fitted, recipe, seed, method, archives)
    _exact(actual.state_dict(), old.state_dict())
    assert actual.evaluations == actual.version == actual.retired == 0
    assert not actual.moment.count_nonzero()
    _exact(actual.target, actual.vector)
    _exact(actual.anchor, actual.vector[actual.ltm])
    output = actual.choose(records[0]["feature"], explore=False)
    _exact(output, old.choose(records[0]["feature"], explore=False))
    assert actual.state_dict()["rng"].keys() == {"exploration", "targets", "fisher", "fd"}
    assert len({g.initial_seed() for g in actual.generators.values()}) == 4
    for name in actual.generators:
        _exact(actual.generators[name].get_state(), old.generators[name].get_state())
    actual.archive.update(torch.zeros(len(actual.ltm), 1), 0., 1.)
    _exact(archives, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dependent", [False, True])
def test_rank64_preserves_diagonal_psd_chunk_bound_solve_and_gradients(dtype, dependent, monkeypatch):
    generator = torch.Generator().manual_seed(994)
    archive = Archive(torch.zeros(83, dtype=dtype), rank=64)
    factor = torch.randn(83, 79, dtype=dtype, generator=generator) * .1
    if dependent:
        factor[:, 1:] = factor[:, :1]
        factor[0] = 0
    widths = []
    svd = torch.linalg.svd

    def observe(value, **kwargs):
        widths.append(value.shape[1])
        return svd(value, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", observe)
    archive.update(factor, 0., 1.)
    assert max(widths) <= 64 + Archive.FRESH_CHUNK_COLUMNS
    assert archive.diagnostics()["chunk_sizes"] == [32, 32, 15]
    assert archive.factor.shape == (83, 64)
    dense = archive.factor @ archive.factor.mT + torch.diag(archive.residual)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(dense.diagonal(), factor.square().sum(1), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(dense.trace(), factor.square().sum(), atol=tolerance, rtol=tolerance)
    assert archive.residual.min() >= 0
    # Inspect the represented operator in FP64, rather than eigendecomposing
    # an already rounded FP32 dense product at a rank-deficient boundary.
    spectral = archive.factor.double() @ archive.factor.double().mT + torch.diag(archive.residual.double())
    assert torch.linalg.eigvalsh((spectral + spectral.mT) / 2).min() >= -1e-12 * max(float(spectral.trace()), 1.)
    assert len(archive.diagnostics()["factor_eigenvalues"]) == 64
    if dependent:
        assert archive.diagnostics()["realized_rank"] == 1
        assert not archive.factor[0].count_nonzero()
    rhs = torch.randn(83, 2, dtype=dtype, generator=generator)
    damping = torch.linspace(.3, 1.1, 83, dtype=dtype)
    solution = archive.solve(rhs, damping, 2.)
    torch.testing.assert_close(solution, torch.linalg.solve(torch.diag(damping) + 2 * dense, rhs),
                               atol=10 * tolerance, rtol=10 * tolerance)
    vector = rhs[:, 0].clone().requires_grad_()
    gradient, = torch.autograd.grad(archive.quadratic(vector), vector)
    torch.testing.assert_close(gradient, 2 * dense @ vector, atol=10 * tolerance, rtol=10 * tolerance)
    epsilon = .05 if dtype == torch.float32 else 1e-4
    direction = rhs[:, 1]
    finite_difference = (archive.quadratic(vector.detach() + epsilon * direction)
                         - archive.quadratic(vector.detach() - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(finite_difference, gradient @ direction, atol=50 * tolerance, rtol=50 * tolerance)
    restored = Archive.from_state_dict(archive.state_dict(), dtype=dtype)
    _exact(restored.state_dict(), archive.state_dict())


def test_default_block_diagnostics_uses_actor_first455_and_remaining_critics():
    diagonal = torch.ones(2209)
    factor = torch.zeros(2209, 64)
    factor[:455, 0] = .5
    factor[455:, 1] = .25
    result = numerics.block_diagnostics(Archive(diagonal, factor, rank=64))
    assert result["actor_rows"] == 455 and result["critic_rows"] == 1754
    assert result["actor"]["trace"] == 455 and result["critic"]["trace"] == 1754
    assert result["actor"]["correlated_trace_fraction"] == .25
    assert result["critic"]["correlated_trace_fraction"] == .0625
    assert result["actor"]["factor_column_energy"][:2] == [455 * .25, 0.]
    assert result["critic"]["factor_column_energy"][:2] == [0., 1754 * .0625]
    json.dumps(result, allow_nan=False)


def test_cached_resource_audit_uses_selected_batch_isolated_repeats_and_no_state_changes(fixture, monkeypatch):
    fitted, records, recipe = fixture
    archives = numerics.rebuild_initial(fitted, records[:2], recipe, MemoryStore(), lambda: None)["archives"]
    before = copy.deepcopy((fitted, records, archives))
    global_rng = torch.get_rng_state().clone()
    fit = Learner.fit
    fits = []

    def observe(learner, selected, acquired, check):
        assert len(selected) == 65 and learner.recipe.steps == 3
        assert selected[-1]["arrival"] == acquired - 1
        assert acquired == 257
        initial = copy.deepcopy(learner.state_dict())
        metrics = fit(learner, selected, acquired, check)
        fits.append((initial, learner.state_dict(), metrics))
        return metrics

    monkeypatch.setattr(Learner, "fit", observe)
    checks = []
    result = numerics.resource_audit(fitted, records, recipe, archives, lambda: checks.append(True))
    assert result["healthy"], result
    assert result["healthy_ranks"] == [0, 10, 32, 64]
    assert len(fits) == 8 * 4 and checks
    assert result["device"] == "cpu" and result["new_observations"] == 0
    assert result["repeats"] == 3 and result["warmups"] == 1
    assert result["batch_records"] == 65 and result["steps"] == 3
    assert [(r["rank"], r["fd_strength"]) for r in result["timings"]] == [
        (rank, fd) for rank in numerics.RANKS for fd in (0., numerics.FD_STRENGTH)]
    for i, row in enumerate(result["timings"]):
        assert row["timing_available"] and row["fit_seconds"] > 0
        assert row["health"]["accepted_anchor_equal"] and row["health"]["target_update_equal"]
        assert row["health"]["finite"] and row["health"]["fd"]["finite"]
        for component, timing in row["timing_samples"].items():
            assert len(timing["samples_seconds"]) == 3 and timing["warmups"] == 1
            assert row[component + "_seconds"] == sorted(timing["samples_seconds"])[1]
        group = fits[4 * i:4 * (i + 1)]
        for initial, final, metrics in group:
            _exact(initial, group[0][0])
            _exact(final, group[0][1])
            assert metrics["fd"] == pytest.approx(row["fd_strength"] * row["health"]["fd"]["unit_penalty"], rel=1e-5)
        if row["rank"] == 0:
            assert row["archive"]["factor_eigenvalues"] == []
    audit = result["compression_audit"]
    assert audit["healthy"] and audit["records"] == 8 and audit["reference_columns"] == 88
    assert audit["probe_count"] == 4 and len(audit["ranks"]) == 4
    for row in audit["ranks"]:
        assert len(row["penalty_errors"]) == 8
        assert row["diagonal_max_abs_error"] < 1e-5
        assert row["random_probe_operator_relative_error"] >= 0
        assert row["random_probe_solve_relative_error"] >= 0
    json.dumps(result, allow_nan=False)
    _exact((fitted, records, archives), before)
    _exact(torch.get_rng_state(), global_rng)


def test_compression_reference_is_exact_without_truncation_including_actual_direction_penalty(fixture):
    fitted, records, recipe = fixture
    count = len(Learner(numerics.Heads(), recipe, 1, "td_replay").ltm)
    direction = torch.linspace(-.01, .02, count)
    result = numerics._compression_audit(fitted, records[:2], recipe, {"cached": direction}, lambda: None)
    rank64 = next(row for row in result["ranks"] if row["rank"] == 64)
    assert rank64["random_probe_operator_relative_error"] < 1e-5
    # Omega=39 amplifies the frozen FP32 compression's roundoff in the solve.
    assert rank64["random_probe_solve_relative_error"] < 1e-4
    assert rank64["penalty_errors"][0]["relative_error"] < 1e-5


def test_nonfinite_profile_returns_json_error_and_rejects_rank_not_silently_admitting(fixture, monkeypatch):
    fitted, records, recipe = fixture
    archives = numerics.rebuild_initial(fitted, records[:2], recipe, MemoryStore(), lambda: None)["archives"]
    row = numerics._resource_row

    def fail_rank64(fitted, selected, cfg, archives, acquired, method, check):
        if cfg.rank == 64:
            raise FloatingPointError("nonfinite head proposal")
        return row(fitted, selected, cfg, archives, acquired, method, check)

    monkeypatch.setattr(numerics, "_resource_row", fail_rank64)
    result = numerics.resource_audit(fitted, records, recipe, archives, lambda: None)
    assert not result["healthy"]
    assert result["healthy_ranks"] == [0, 10, 32]
    failed = [row for row in result["timings"] if row["rank"] == 64]
    assert len(failed) == 2
    assert all(not r["timing_available"] and not r["health"]["healthy"] for r in failed)
    assert all("nonfinite head proposal" in r["error"] for r in failed)
    json.dumps(result, allow_nan=False)


def test_resource_input_guards_and_interruption_are_not_swallowed(fixture):
    fitted, records, recipe = fixture
    with pytest.raises(ValueError, match="65 cached"):
        numerics.resource_audit(fitted, records[:2], recipe, {}, lambda: None)
    with pytest.raises(ValueError, match="three inner steps"):
        numerics.resource_audit(fitted, records, replace(recipe, steps=1), {}, lambda: None)
    archives = numerics.rebuild_initial(fitted, records[:2], recipe, MemoryStore(), lambda: None)["archives"]

    def stop():
        raise InterruptedError("stop cached audit")

    with pytest.raises(InterruptedError, match="stop cached audit"):
        numerics.resource_audit(fitted, records, recipe, archives, stop)
