import json
import copy
from pathlib import Path

import pytest
import torch

from mnist_experiment.akaike_score_gate import phase0
from mnist_experiment.akaike_score_gate.artifacts import RunLimit, StopExecution, UnitStore
from mnist_experiment.akaike_score_gate.config import Study
from mnist_experiment.akaike_score_gate.budget import WorkBudget
from mnist_experiment.akaike_score_gate.config import canonical_hash


def study():
    return Study.from_path("mnist_experiment/akaike_score_gate/configs/smoke.json")


def test_profile_charge_key_coverage_matches_parent_contract():
    specs = phase0.operation_specs()
    keys = {spec["key"] for spec in specs}
    assert len(keys) == len(specs) == 846
    for width in (5, 6):
        for n in (4, 8, 18, 32, 64, 128, 180, 512, 1820):
            assert f"prequential:w{width}:n{n}" in keys
            for depth in range(1, 7):
                assert f"gradient:w{width}:d{depth}:n{n}" in keys
                assert f"preview:w{width}:d{depth}:n{n}" in keys
                for ewc in ("diag", "lr"):
                    assert f"fit:w{width}:d{depth}:n{n}:e{ewc}" in keys
            for ewc in ("diag", "lr"):
                assert f"archive:w{width}:n{n}:e{ewc}" in keys
                for depth in range(1, 6):
                    for gate in ("diag", "lr"):
                        assert f"probe:w{width}:d{depth}:n{n}:e{ewc}:g{gate}" in keys
    assert {spec["depth"] for spec in specs if spec["operation"] == "probe"} == set(range(1, 6))


def test_standard_work_packages_use_equal_widths_and_family_budgets_not_quality():
    charges = {spec["key"]: float(spec["width"] + spec["n"] / 100)
               for spec in phase0.operation_specs()}
    b0, rows = phase0.standard_work_packages(charges, episode_arrivals=32)
    assert set(b0) == {"rotation", "mixture"}
    for family in b0:
        family_rows = [row for row in rows if row["family"] == family]
        assert len(family_rows) == 2 * len(phase0.BUCKETS[family]) * 4
        assert sum(row["width"] == 5 for row in family_rows) == sum(row["width"] == 6 for row in family_rows)
        expected = []
        for row in family_rows:
            unit_cost = row["width"] + row["n"] / 100
            cost = unit_cost * (50 + 1 + 50 + 1 / 32)
            expected.append(cost)
            assert row["seconds"] == pytest.approx(cost)
        assert b0[family] == pytest.approx(phase0.upper_quartile(expected))
    assert phase0.upper_quartile([1., 2., 5.]) == 3.5


def test_measurement_uses_upper_quartile_floor_and_excludes_warmup():
    calls = []
    times = iter([0., 1., 10., 12., 20., 25.])
    result = phase0.measure_operation(lambda: calls.append(1), device="cpu", repeats=3,
                                      warmups=1, clock=lambda: next(times))
    assert len(calls) == 4
    assert result["seconds"] == [1., 2., 5.]
    assert result["charge"] == 3.5
    fast = phase0.measure_operation(lambda: None, device="cpu", repeats=1,
                                    warmups=0, clock=lambda: 1.)
    assert fast["charge"] == 1e-6
    with pytest.raises(ValueError, match="positive"):
        phase0.measure_operation(lambda: None, device="cpu", repeats=0)


def test_local_math_contracts_and_quadratic_replications_are_deterministic():
    state = torch.random.get_rng_state().clone()
    result = phase0.mathematical_validation(3456)
    assert result["status"] == "passed"
    assert result["trajectory_smokes"] == "parent_managed_not_run_here"
    quadratic = phase0.quadratic_validation(128)
    assert quadratic == phase0.quadratic_validation(128)
    assert quadratic["optimism_relative_error"] < .05
    assert quadratic["covariance_relative_error"] < .06
    assert len(quadratic["expected_parameter_covariance"]) == 3
    assert torch.equal(state, torch.random.get_rng_state())


def test_tiny_cpu_operation_bodies_are_finite_and_do_not_train_or_mutate_model():
    context = phase0._OperationContext(study(), 5, 2)
    before = context.layout.flatten_module(context.model, detach=True)
    for depth, ewc in ((1, "diag"), (6, "lr")):
        loss, gradient = context.fit(depth, ewc)
        assert torch.isfinite(loss)
        assert gradient.shape == (context.indices[depth].numel(),)
        assert torch.isfinite(gradient).all()
    loss, gradient = context.gradient(3)
    assert gradient.dtype == torch.float64
    frozen = torch.ones_like(gradient, dtype=torch.bool)
    frozen[context.indices[3]] = False
    assert not torch.count_nonzero(gradient[frozen])
    assert torch.isfinite(context.preview(2))
    nll, predictions, probabilities = context.prequential()
    assert torch.isfinite(nll) and predictions.shape == probabilities.shape == (2,)
    archive = context.archive("diag")
    assert archive["archive"].shape == (501, 501)
    probe = context.probe(1, "lr", "diag")
    assert torch.isfinite(probe["displacement"]).all()
    assert probe["displacement"].shape == (501,)
    torch.testing.assert_close(context.layout.flatten_module(context.model), before)
    assert all(parameter.grad is None for parameter in context.model.parameters())


def _mock_small_profile(monkeypatch):
    """Exercise lifecycle with cheap fake timings, never the real profile grid."""
    monkeypatch.setattr(phase0, "WIDTHS", (5, 6))
    monkeypatch.setattr(phase0, "BUCKETS", {"rotation": (4,), "mixture": (8,)})
    monkeypatch.setattr(phase0, "mathematical_validation", lambda seed: {"status": "passed", "seed": seed})
    calls = []

    class FakeContext:
        def __init__(self, study, width, n):
            assert study.device == "cpu"

        def operation(self, spec):
            return lambda: calls.append(spec["key"])

    def fake_measure(operation, *, device, repeats, warmups):
        assert device == "cpu" and repeats == 1 and warmups == 0
        operation()
        return {"seconds": [.001], "upper_quartile_seconds": .001,
                "charge": .001, "warmups": warmups, "repeats": repeats}

    monkeypatch.setattr(phase0, "_OperationContext", FakeContext)
    monkeypatch.setattr(phase0, "measure_operation", fake_measure)
    return calls


def test_phase0_seals_profile_and_completed_resume_does_no_work(monkeypatch, tmp_path):
    calls = _mock_small_profile(monkeypatch)
    config = study()
    store = UnitStore(tmp_path, config)
    before_threads = torch.get_num_threads()
    path = phase0.ensure_phase0(store)
    assert isinstance(path, Path) and (path / "COMPLETED").is_file()
    assert torch.get_num_threads() == before_threads
    for name in phase0.REQUIRED_ARTIFACTS:
        assert (path / name).is_file()
    profile = json.loads((path / "profile.json").read_text())
    assert len(profile["charges"]) == len(calls)
    assert profile["widths"] == [5, 6]
    assert profile["buckets"] == {"rotation": [4], "mixture": [8]}
    assert profile["bucket_union"] == [4, 8]
    assert profile["family_buckets"] == {"rotation": [4], "mixture": [8]}
    assert profile["metadata"]["score_samples"] == 4
    assert profile["metadata"]["selection"] == "no predictive-quality or architecture selection"
    assert profile["standard_work_packages"]
    from mnist_experiment.akaike_score_gate.analysis import _rows

    rows = _rows(path / "metrics.json")
    assert len(rows) == len(calls)
    assert all(row["metric_kind"] == "operation_timing" for row in rows)
    assert json.loads((path / "math_validation.json").read_text())["status"] == "passed"
    assert "math_validation.json" in json.loads((path / "integrity.json").read_text())
    for family in profile["b0"]:
        assert profile["budgets"][family]["strict"] == profile["b0"][family]
        assert profile["budgets"][family]["principal"] == 2 * profile["b0"][family]
    saved = {name: (path / name).read_bytes() for name in phase0.REQUIRED_ARTIFACTS}
    count = len(calls)
    assert phase0.ensure_phase0(UnitStore(tmp_path, config)) == path
    assert len(calls) == count
    assert saved == {name: (path / name).read_bytes() for name in phase0.REQUIRED_ARTIFACTS}
    with pytest.raises(RuntimeError, match="completed"):
        phase0.ensure_phase0(UnitStore(tmp_path, config), resume=False)


def test_profile_lookup_matches_parent_trajectory_operation_keys():
    from mnist_experiment.akaike_score_gate.trajectory import operation_key

    profile = {"buckets": {family: list(values) for family, values in phase0.BUCKETS.items()},
               "charges": {spec["key"]: .001 for spec in phase0.operation_specs()}}
    assert operation_key(profile, "fit", 5, 19, depth=6, ewc="diag") == "fit:w5:d6:n32:ediag"
    assert operation_key(profile, "probe", 6, 1819, depth=5, ewc="lr", gate="diag") == "probe:w6:d5:n1820:elr:gdiag"
    assert operation_key(profile, "archive", 5, 1820, ewc="lr") == "archive:w5:n1820:elr"
    assert operation_key(profile, "prequential", 6, 182) == "prequential:w6:n512"
    assert operation_key(profile, "fit", 5, 5, family="rotation") == "fit:w5:d1:n32:elr"
    assert operation_key(profile, "fit", 5, 5, family="mixture") == "fit:w5:d1:n8:elr"
    with pytest.raises(ValueError, match="exceeds"):
        operation_key(profile, "archive", 5, 1821)


def test_interrupted_profile_reuses_prior_charge_and_restores_thread_count(monkeypatch, tmp_path):
    calls = _mock_small_profile(monkeypatch)
    config = study()

    class InterruptAfterFirst(RunLimit):
        def check(self):
            if calls:
                raise StopExecution("test interruption")

    previous_threads = torch.get_num_threads()
    with pytest.raises(StopExecution):
        phase0.ensure_phase0(UnitStore(tmp_path, config, limit=InterruptAfterFirst()))
    assert torch.get_num_threads() == previous_threads
    assert len(calls) == 1
    first_key = calls[0]
    final = phase0.ensure_phase0(UnitStore(tmp_path, config))
    assert (final / "COMPLETED").is_file()
    assert calls.count(first_key) == 1
    assert len(calls) == len(phase0.operation_specs())


def test_corrupt_partial_profile_is_rejected_not_repaired(monkeypatch, tmp_path):
    calls = _mock_small_profile(monkeypatch)

    class InterruptAfterFirst(RunLimit):
        def check(self):
            if calls:
                raise StopExecution("test interruption")

    config = study()
    with pytest.raises(StopExecution):
        phase0.ensure_phase0(UnitStore(tmp_path, config, limit=InterruptAfterFirst()))
    partial = next(tmp_path.rglob("timing_parts/*.json"))
    record = json.loads(partial.read_text())
    record["payload"]["charge"] = 100.
    partial.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="corrupt Phase 0 partial"):
        phase0.ensure_phase0(UnitStore(tmp_path, config))


def _repair_inputs():
    config = study()
    charges = {spec["key"]: .001 + i * 1e-7 for i, spec in enumerate(phase0.operation_specs())
               if spec["n"] in (4, 8)}
    base = {
        "schema_version": 1, "charges": charges,
        "charge_table_hash": WorkBudget(None, charges).charge_table_hash,
        "widths": [5, 6], "buckets": {"rotation": [4], "mixture": [8]},
        "b0": {"rotation": 2., "mixture": 3.},
        "budgets": {"rotation": {"strict": 2., "principal": 4.},
                    "mixture": {"strict": 3., "principal": 6.}},
        "standard_work_packages": [{"seconds": 2., "origin": "base"}],
        "metadata": {"device": "cpu", "dtype": config.dtype, "cpu_threads": config.cpu_threads,
                     "source_hashes": {"base_source": "original"}},
    }
    policy = {"name": "repair_score_and_rank", "depth": None, "ewc": "lr", "gate": "lr",
              "rank": config.rank, "score_samples": 8, "gate_rank": 16,
              "c": 1., "half_life": 8., "cadence": 1}
    return config, base, policy


def _fake_repair_timing(monkeypatch):
    contexts, calls = [], []

    class FakeContext:
        def __init__(self, study, width, n, *, gate_rank=None, probe_score_samples=None):
            assert study.device == "cpu"
            contexts.append({"width": width, "n": n, "gate_rank": gate_rank,
                             "score_samples": probe_score_samples, "archive_rank": study.rank})

        def operation(self, spec):
            assert spec["operation"] == "probe"
            return lambda: calls.append(spec.copy())

    def measure(operation, *, device, repeats, warmups):
        assert device == "cpu" and repeats == 1 and warmups == 0
        operation()
        return {"seconds": [.004], "upper_quartile_seconds": .004,
                "charge": .004, "warmups": warmups, "repeats": repeats}

    monkeypatch.setattr(phase0, "_OperationContext", FakeContext)
    monkeypatch.setattr(phase0, "measure_operation", measure)
    return contexts, calls


def test_repair_profile_only_replaces_selected_probes_without_increasing_allowance(monkeypatch, tmp_path):
    config, base, policy = _repair_inputs()
    saved = copy.deepcopy(base)
    contexts, calls = _fake_repair_timing(monkeypatch)
    final = phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, 6)
    profile = json.loads((final / "profile.json").read_text())
    rows = json.loads((final / "metrics.json").read_text())
    summary = json.loads((final / "summary.json").read_text())
    unit = json.loads((final / "config.json").read_text())
    assert unit["phase"] == "phase4" and unit["kind"] == "repair_profile"
    assert unit["detail"]["parent_profile_hash"] == canonical_hash(base)
    assert unit["detail"]["policy"] == policy
    assert len(calls) == len(rows) == 10
    assert {(row["n"], row["depth"]) for row in calls} == {(n, d) for n in (4, 8) for d in range(1, 6)}
    assert all(row["width"] == 6 and row["ewc"] == row["gate"] == "lr" for row in calls)
    changed = {key for key in base["charges"] if profile["charges"][key] != base["charges"][key]}
    assert changed == set(profile["repair_probe_keys"]) == {row["key"] for row in rows}
    for key in ("b0", "budgets", "standard_work_packages", "buckets", "widths"):
        assert profile[key] == saved[key]
    assert base == saved
    assert profile["probe_units"] == 1
    assert profile["repair_policy"] == policy
    assert profile["metadata"]["source_hashes"] == base["metadata"]["source_hashes"]
    assert profile["charge_table_hash"] == WorkBudget(None, profile["charges"]).charge_table_hash
    assert summary["non_probe_charges_preserved"] and summary["status"] == "completed"
    assert all(row["gate_rank"] == 16 and row["score_samples"] == 8 for row in rows)
    assert contexts == [{"width": 6, "n": n, "gate_rank": 16, "score_samples": 8,
                         "archive_rank": config.rank} for n in (4, 8)]


def test_repair_profile_completed_resume_is_immutable_and_does_no_timing(monkeypatch, tmp_path):
    config, base, policy = _repair_inputs()
    _, calls = _fake_repair_timing(monkeypatch)
    final = phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, 5)
    before = {name: (final / name).read_bytes() for name in phase0.REPAIR_PROFILE_ARTIFACTS}
    count = len(calls)
    assert phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, 5) == final
    assert len(calls) == count
    assert before == {name: (final / name).read_bytes() for name in phase0.REPAIR_PROFILE_ARTIFACTS}
    with pytest.raises(RuntimeError, match="completed"):
        phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, 5, resume=False)


def test_repair_profile_resumes_only_untimed_probes(monkeypatch, tmp_path):
    config, base, policy = _repair_inputs()
    _, calls = _fake_repair_timing(monkeypatch)

    class InterruptAfterFirst(RunLimit):
        def check(self):
            if calls:
                raise StopExecution("test repair timing interruption")

    before_threads = torch.get_num_threads()
    with pytest.raises(StopExecution):
        phase0.ensure_repair_profile(UnitStore(tmp_path, config, limit=InterruptAfterFirst()), base, policy, 5)
    assert torch.get_num_threads() == before_threads
    assert len(calls) == 1
    first_key = calls[0]["key"]
    final = phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, 5)
    assert (final / "COMPLETED").is_file()
    assert len(calls) == 10
    assert sum(row["key"] == first_key for row in calls) == 1


def test_repair_profiles_bind_parent_and_actual_policy_settings(monkeypatch, tmp_path):
    config, base, policy = _repair_inputs()
    _fake_repair_timing(monkeypatch)
    store = UnitStore(tmp_path, config)
    first = phase0.ensure_repair_profile(store, base, policy, 5)
    second = phase0.ensure_repair_profile(store, base, {**policy, "score_samples": 16}, 5)
    third = phase0.ensure_repair_profile(store, base, {**policy, "gate_rank": 8}, 5)
    other_base = copy.deepcopy(base)
    other_base["b0"]["rotation"] += .1
    fourth = phase0.ensure_repair_profile(store, other_base, policy, 5)
    assert len({first, second, third, fourth}) == 4


@pytest.mark.parametrize("change", ["archive_rank", "archive_samples", "score_samples", "gate_rank",
                                   "width", "gate", "missing_charge", "hash"])
def test_repair_profile_rejects_changes_beyond_probe_scope(monkeypatch, tmp_path, change):
    config, base, policy = _repair_inputs()
    _, calls = _fake_repair_timing(monkeypatch)
    width = 5
    if change == "archive_rank":
        policy["rank"] = 16
    elif change == "archive_samples":
        policy["archive_samples"] = 8
    elif change in ("score_samples", "gate_rank"):
        policy[change] = 0
    elif change == "width":
        width = 7
    elif change == "gate":
        policy["gate"] = None
    elif change == "missing_charge":
        del base["charges"]["probe:w5:d1:n4:elr:glr"]
        base["charge_table_hash"] = WorkBudget(None, base["charges"]).charge_table_hash
    else:
        base["charge_table_hash"] = "corrupt"
    with pytest.raises(ValueError):
        phase0.ensure_repair_profile(UnitStore(tmp_path, config), base, policy, width)
    assert not calls
    assert not list(tmp_path.rglob("COMPLETED"))


def test_probe_overrides_do_not_reach_archive_rank_or_sampling(monkeypatch):
    config = study()
    context = phase0._OperationContext(config, 5, 2, gate_rank=16, probe_score_samples=7)
    fisher_samples, represented_ranks, archive_ranks = [], [], []
    original_represent = phase0.represent
    original_consolidate = phase0.consolidate

    def fisher(model, layout, inputs, weights, *, sample_size, **kwargs):
        fisher_samples.append(sample_size)
        return torch.eye(layout.total_numel, device=inputs.device, dtype=torch.float64), {}

    def represented(matrix, kind, *, rank, **kwargs):
        represented_ranks.append(rank)
        return original_represent(matrix, kind, rank=rank, **kwargs)

    def consolidated(*args, **kwargs):
        archive_ranks.append(kwargs["rank"])
        return original_consolidate(*args, **kwargs)

    monkeypatch.setattr(phase0, "model_fisher", fisher)
    monkeypatch.setattr(phase0, "represent", represented)
    monkeypatch.setattr(phase0, "consolidate", consolidated)
    context.probe(1, "lr", "lr")
    context.archive("lr")
    assert fisher_samples == [7, 4]
    assert represented_ranks == [16]
    assert archive_ranks == [config.rank]
