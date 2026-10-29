"""CPU-only wrapper checks; no MNIST, production kernels, or long pilots."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from mnist_experiment import plan15_grid_subspace as wrapper
from mnist_experiment.fisher_memory.artifacts import Store, ArtifactError
from src.representations import representation_from_artifact


def test_config_versions_execution_without_changing_protocol_or_component_seeds():
    cfg, base = wrapper.Config(), wrapper.BaseConfig()
    cfg.validate()
    assert cfg.archive_rank == 64 and base.archive_rank == 8
    assert cfg.protocol == base.protocol == "plan15-grid-registration-v1"
    assert cfg.scientific_hash != base.scientific_hash
    assert wrapper.Config.from_mapping(cfg.mapping()) == cfg
    assert wrapper.Config.smoke_config().archive_rank == 64
    for family in ("development--grid", "pilot--grid", "confirmation--grid"):
        for component in ("source_sampling", "initialization_model", "factors", "initial_compression"):
            assert cfg.seed_for(family, 3, component) == base.seed_for(family, 3, component)
    with pytest.raises(FrozenInstanceError):
        cfg.power_steps = 8
    for changes in (dict(power_steps=8), dict(power_steps=4.), dict(oversample=8), dict(ritz_tol=1e-3),
                    dict(cluster_tol=1e-5), dict(numerical_policy="unversioned"), dict(archive_rank=8), dict(archive_rank=64.)):
        with pytest.raises(ValueError, match="policy"):
            replace(cfg, **changes).validate()


def test_sources_add_exact_wrapper_and_probe_keys_without_changing_frozen_sources():
    base = wrapper.base_sources()
    code = wrapper.sources()
    assert {k: code[k] for k in base} == base
    assert set(code)-set(base) == {"mnist_experiment/plan15_grid_subspace.py",
                                 "mnist_experiment/plan15_grid_compression_probe.py"}
    assert wrapper.run.content_hash(base) != wrapper.run.content_hash(code)
    assert code["mnist_experiment/plan15_grid_compression_probe.py"] == wrapper.file_digest(Path(wrapper.probe.__file__))


def test_real_tiny_source_plan_is_identical_with_new_numerical_configuration():
    source = SimpleNamespace(train_labels=torch.arange(1800) % 10, test_labels=torch.arange(400) % 10)
    old = wrapper.plan_data(source, wrapper.BaseConfig.smoke_config(), "development--grid", 0, "development")
    new = wrapper.plan_data(source, wrapper.Config.smoke_config(), "development--grid", 0, "development")
    assert old == new


def test_bindings_restore_all_symbols_options_and_nested_context_after_failure(tmp_path):
    original = (wrapper.run.Config, wrapper.run.Runner, wrapper.run.Study, wrapper.run.sources,
                wrapper.learner.compress_fisher, wrapper.acceptance.sources, wrapper.run.process_alive)
    with pytest.raises(RuntimeError, match="interrupted"):
        with wrapper.bindings(tmp_path / "old", tmp_path / "evidence"):
            assert wrapper.run.Config is wrapper.Config and wrapper.run.Runner is wrapper.Runner
            assert wrapper.run.Study is wrapper.Study and wrapper.run.sources is wrapper.sources
            assert wrapper.learner.compress_fisher is wrapper.compress_fisher
            assert wrapper.acceptance.sources is wrapper.sources
            with wrapper.bindings():
                assert wrapper.ROOTS.get() == (None, None)
            assert wrapper.ROOTS.get() == (tmp_path / "old", tmp_path / "evidence")
            raise RuntimeError("interrupted")
    assert (wrapper.run.Config, wrapper.run.Runner, wrapper.run.Study, wrapper.run.sources,
            wrapper.learner.compress_fisher, wrapper.acceptance.sources, wrapper.run.process_alive) == original
    assert wrapper.ROOTS.get() == (None, None)


def test_small_cold_compression_is_deterministic_psd_and_preserves_diagonal_trace_and_rng():
    matrix = torch.diag(torch.arange(1, 13, dtype=torch.float64)) + torch.ones(12, 12)*.1
    rng = torch.random.get_rng_state().clone()
    first, diagnostics = wrapper.compress_fisher(matrix, rank=64, seed=51, backend="subspace_cpu")
    second, _ = wrapper.compress_fisher(matrix, rank=64, seed=51, backend="subspace_cpu")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(first["factor"], second["factor"])
    represented = representation_from_artifact(first)
    torch.testing.assert_close(represented.diagonal_vector(), matrix.diagonal())
    torch.testing.assert_close(represented.to_dense(), matrix)
    assert float(torch.linalg.eigvalsh(represented.to_dense()).min()) >= 0
    assert diagnostics["compression"]["chosen_steps"] == 4
    assert diagnostics["compression"]["warm_start_columns"] == 0
    assert diagnostics["compression"]["fallback_reason"] is None


@pytest.mark.parametrize("success_at", [8, 16, None])
def test_ritz_retries_are_cold_four_eight_sixteen_then_exact_fallback(monkeypatch, success_at):
    approximate, exact = wrapper.probe.approximate_fisher, wrapper._exact
    calls, fallbacks = [], []

    def candidate(matrix, **kwargs):
        calls.append(kwargs)
        packed, diagnostics = approximate(matrix, **kwargs)
        diagnostics["compression"]["ritz_residual_relative_fro"] = 1e-6 if kwargs["steps"] == success_at else .1
        return packed, diagnostics

    def reference(*args):
        fallbacks.append(args[-1])
        return exact(*args)

    monkeypatch.setattr(wrapper.probe, "approximate_fisher", candidate)
    monkeypatch.setattr(wrapper, "_exact", reference)
    _, diagnostics = wrapper.compress_fisher(torch.eye(6, dtype=torch.float64), backend="subspace_cpu")
    assert [k["steps"] for k in calls] == ([4, 8] if success_at == 8 else [4, 8, 16])
    assert all(k["oversample"] == 16 and "warm_start" not in k for k in calls)
    assert diagnostics["compression"]["chosen_steps"] == success_at
    assert fallbacks == (["exact_cpu"] if success_at is None else [])
    assert bool(diagnostics["compression"]["fallback_reason"]) == (success_at is None)


@pytest.mark.parametrize("failure", ["orthogonality", "residual", "exception"])
def test_invalid_approximation_uses_exact_fallback_without_changing_diagonal(monkeypatch, failure):
    approximate = wrapper.probe.approximate_fisher

    def invalid(matrix, **kwargs):
        if failure == "exception":
            raise ValueError("negative Nystrom core")
        packed, diagnostics = approximate(matrix, **kwargs)
        c = diagnostics["compression"]
        if failure == "orthogonality":
            c["retained_directions_orthogonality_error_fro"] = .1
        else:
            c["residual_diagonal_minimum_before_clipping"] = -1.
        return packed, diagnostics

    monkeypatch.setattr(wrapper.probe, "approximate_fisher", invalid)
    packed, diagnostics = wrapper.compress_fisher(torch.eye(4, dtype=torch.float64), backend="subspace_cpu")
    assert diagnostics["compression"]["chosen_steps"] is None
    assert len(diagnostics["compression"]["policy_attempts"]) == 3
    assert diagnostics["compression"]["fallback_reason"]
    torch.testing.assert_close(representation_from_artifact(packed).diagonal_vector(), torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("error", [wrapper.learner.StopRequested, wrapper.run.StopRequested, wrapper.run.ResourceLimit])
def test_cancellation_and_budget_limits_never_enter_numeric_fallback(monkeypatch, error):
    def interrupted(*args, **kwargs):
        raise error("stop")

    monkeypatch.setattr(wrapper.probe, "approximate_fisher", interrupted)
    monkeypatch.setattr(wrapper, "_exact", lambda *args: pytest.fail("must not fall back after cancellation"))
    with pytest.raises(error):
        wrapper.compress_fisher(torch.eye(3), backend="subspace_cpu")


@contextmanager
def store_at(path, identity, resume=False):
    with Store(path, identity, resume=resume, free_reserve_bytes=0, finalization_reserve_bytes=0) as store:
        yield store


def seal(store, name, value):
    store.begin(name, dict(kind="test_decision"))
    store.write_json(name, "result.json", value)
    store.complete(name)
    return wrapper.read_json(store.unit_path(name)/"COMPLETED")["seal_hash"]


@pytest.fixture
def evidence(tmp_path, monkeypatch, request):
    monkeypatch.setattr(wrapper, "base_sources", lambda: {"frozen.py": "original-source"})
    cfg = wrapper.BaseConfig(device="cpu")
    code = wrapper.base_sources()
    identity = dict(protocol=cfg.protocol, allocation_family=wrapper.run.ALLOCATION_FAMILY, role="production",
        config_hash=cfg.scientific_hash, source_hash=wrapper.run.content_hash(code), data_hash="fixed-data")
    parent, assessed = tmp_path/"old", tmp_path/"assessment"
    with store_at(parent, identity) as store:
        store.write_json(None, "resolved_config.json", cfg.mapping())
        store.write_json(None, "provenance.json", dict(sources=code, source_hash=identity["source_hash"]))
        seals = {"decision-"+n: seal(store, "decision-"+n, value) for n, value in (
            ("initialization", dict(epochs=45, channels=[4, 6])), ("readiness", dict(blur_max=1.8)),
            ("archive-rank", dict(rank=64, adequate=True, rows=[dict(rank=64, snapshot=n, relative_frobenius=.005)
                                                              for n in ("initial", "n25", "n75")])))}
    parent_identity = wrapper._identity(parent)
    assessment_identity = dict(allocation_family=wrapper.run.ALLOCATION_FAMILY, role="compression_assessment",
        source_identity=parent_identity["identity_hash"], config_hash=cfg.scientific_hash,
        code_hashes={Path(wrapper.probe.__file__).name: wrapper.file_digest(Path(wrapper.probe.__file__))})
    cases = [dict(case=n, mode="cold", rank=64, steps=4, passed=True,
        matrix_audit=dict(passed=True, matrix_relative_error=.0001, source_relative_error=.0051,
            reference_source_relative_error=.005, quadratic_relative_error=.0002,
            damped_solve_relative_errors={"0.1": .0069}),
        fit_audit=dict(passed=True, prediction_difference_relative_to_reference_change=.0009),
        exact_profile=dict(timings=[dict(warmup=True, seconds=.08), dict(warmup=False, seconds=.07)]),
        candidate_profile=dict(timings=[dict(warmup=True, seconds=.02), dict(warmup=False, seconds=.01)],
            diagnostics=dict(compression=dict(ritz_residual_relative_fro=1e-5)))) for n in ("initial", "n25", "n75")]
    source_evidence = dict(source_identity=parent_identity, source_seals=seals)
    summary = dict(selected_steps=dict(cold=4), all_cases_passed=True, cases={"cold-s4": cases}, source_evidence=source_evidence)
    damage = getattr(request, "param", None)
    if damage == "probe":
        assessment_identity["code_hashes"][Path(wrapper.probe.__file__).name] = "obsolete-probe"
    elif damage == "warm_only":
        summary["selected_steps"] = dict(warm=4)
    elif damage == "failed":
        cases[1]["passed"] = False
    elif damage == "ritz":
        cases[1]["candidate_profile"]["diagnostics"]["compression"]["ritz_residual_relative_fro"] = .1
    elif damage == "fit":
        cases[1]["fit_audit"]["passed"] = False
    elif damage == "missing":
        cases.pop()
    elif damage == "source_error":
        cases[1]["matrix_audit"]["source_relative_error"] = .11
    elif damage == "old_schema":
        cases[1]["matrix_audit"].pop("source_relative_error")
    with store_at(assessed, assessment_identity) as store:
        seal(store, "source-evidence", source_evidence)
        seal(store, "summary", summary)
    return SimpleNamespace(parent=parent, assessed=assessed, cfg=wrapper.Config(device="cpu"),
                           identity=identity, summary=summary)


def lightweight_study(monkeypatch, cfg, store):
    def initialize(self, cfg, args, store, budget):
        self.cfg, self.args, self.store, self.budget = cfg, args, store, budget
        self.runner = SimpleNamespace(compression_backend="subspace_cpu", archive_rank=64)
    monkeypatch.setattr(wrapper.BaseStudy, "__init__", initialize)
    return wrapper.Study(cfg, SimpleNamespace(), store, SimpleNamespace(check=lambda: None))


def fingerprints(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_production_lineage_reuses_actual_rank_audits_and_never_runs_pilot(evidence, monkeypatch, tmp_path):
    before = fingerprints(evidence.parent), fingerprints(evidence.assessed)
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(tmp_path/"new", evidence.identity) as store:
        study = lightweight_study(monkeypatch, evidence.cfg, store)
        monkeypatch.setattr(study, "job", lambda *args, **kwargs: pytest.fail("must not repeat pilot"))
        study.resource_assessment()
        profile = study.decision("resource-assessment")
        assert profile["selected_backend"] == "subspace_cpu" and profile["power_steps"] == 4
        assert profile["backends"]["subspace_cpu"]["median_transaction_seconds"] == .01
        assert profile["attempts"][0]["rows"][1] == dict(warmup=False, transaction_seconds=.01)
        comparison = profile["numerical_comparison"]
        assert comparison["candidate_median_seconds"] == .01 and comparison["exact_median_seconds"] == .07
        assert comparison["max_matrix_relative_error"] == .0001 and comparison["max_prediction_relative_error"] == .0009
        assert comparison["max_source_relative_error"] == .0051
        assert comparison["proof_root"] == str(evidence.assessed)
        assert comparison["proof_seal"] == study.decision("lineage")["evidence_seals"]["summary"]
        study.rank_calibration()
        rank = study.decision("archive-rank")
        assert rank["rank"] == 64 and rank["adequate"]
        assert len(rank["numerical_discrepancy_rows"]) == 3
        assert rank["numerical_discrepancy_rows"][0] == dict(snapshot="initial", relative_frobenius_to_exact_same_rank=.0001,
            candidate_relative_frobenius_to_source=.0051, exact_same_rank_relative_frobenius_to_source=.005)
        assert rank["rows"][0]["relative_frobenius"] == .005
        assert rank["source_evidence"] == study.decision("lineage")
        with pytest.raises(ValueError, match="architecture"):
            study.resource_assessment(channels=(4, 8))
    assert before == (fingerprints(evidence.parent), fingerprints(evidence.assessed))


def test_smoke_uses_small_cpu_proof_without_external_evidence(monkeypatch, tmp_path):
    cfg = wrapper.Config.smoke_config()
    with wrapper.bindings(), store_at(tmp_path/"smoke", dict(role="smoke")) as store:
        study = lightweight_study(monkeypatch, cfg, store)
        study.rank_calibration()
        proof = study.decision("archive-rank")["synthetic_proof"]
        assert proof["matrix_relative_error"] < 1e-8
        assert proof["diagnostics"]["compression"]["chosen_steps"] == 4
        assert study.runner.archive_rank == 64


def test_parent_config_allows_only_the_deliberate_base_rank_difference(evidence):
    original, _ = wrapper.parent_config(evidence.cfg, evidence.parent)
    assert original.archive_rank == 8 and evidence.cfg.archive_rank == 64
    with pytest.raises(ValueError, match="configuration"):
        wrapper.parent_config(replace(evidence.cfg, init_epochs=30), evidence.parent)


def test_bad_probe_hash_incomplete_or_modified_sealed_evidence_is_rejected(evidence, monkeypatch, tmp_path):
    (evidence.assessed/"units/summary/COMPLETED").unlink()
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(tmp_path/"new", evidence.identity) as store:
        with pytest.raises(ValueError, match="incomplete source"):
            lightweight_study(monkeypatch, evidence.cfg, store)
    assert not (tmp_path/"new/units/decision-lineage").exists()


def test_lineage_is_checked_on_resume_and_source_decision_corruption_is_not_repaired(evidence, monkeypatch, tmp_path):
    root = tmp_path/"new"
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity) as store:
        lightweight_study(monkeypatch, evidence.cfg, store)
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity, resume=True) as store:
        lightweight_study(monkeypatch, evidence.cfg, store)
    path = evidence.parent/"units/decision-readiness/result.json"
    path.write_text("{}")
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity, resume=True) as store:
        with pytest.raises(ArtifactError, match="checksum|size"):
            lightweight_study(monkeypatch, evidence.cfg, store)
    assert path.read_text() == "{}"


def test_peer_ledger_includes_original_both_acceptance_seals_and_assessment(tmp_path):
    for name, role, seconds in (("old", "production", 500.), ("accept-old", "acceptance", 40.),
                                ("accept-new", "acceptance", 50.), ("assessment", "compression_assessment", 17.4),
                                ("smoke", "smoke", 99.)):
        with store_at(tmp_path/name, dict(allocation_family=wrapper.run.ALLOCATION_FAMILY, role=role)) as store:
            store.control.restore_active_seconds(seconds)
    assert wrapper.run.peer_seconds(tmp_path, tmp_path/"new") == pytest.approx(607.4, abs=1.)


def test_cli_requires_explicit_production_roots_and_accept_mode_gets_same_bindings(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper.run, "main", lambda argv: pytest.fail("must fail preflight"))
    with pytest.raises(ValueError, match="requires --reuse-root"):
        wrapper.main(["--root", str(tmp_path/"new")])
    with pytest.raises(ValueError, match="smokes must not reuse"):
        wrapper.main(["--root", str(tmp_path/"new"), "--smoke", "--reuse-root", str(tmp_path/"old")])
    with pytest.raises(ValueError, match="distinct peer roots"):
        wrapper.main(["--root", str(tmp_path/"old"), "--budget-root", str(tmp_path),
            "--reuse-root", str(tmp_path/"old"), "--evidence-root", str(tmp_path/"assessment"),
            "--acceptance-root", str(tmp_path/"accept-new")])
    original = wrapper.run.Config

    def accept(argv):
        assert argv == ["--root", "new-acceptance"]
        assert wrapper.acceptance.sources() == wrapper.sources()
        assert wrapper.run.Config is wrapper.Config
        return 19

    monkeypatch.setattr(wrapper.acceptance, "main", accept)
    assert wrapper.main(["accept", "--root", "new-acceptance"]) == 19
    assert wrapper.run.Config is original


@pytest.mark.parametrize("evidence", ["probe", "warm_only", "failed", "ritz", "fit", "missing", "source_error", "old_schema"], indirect=True)
def test_only_sealed_qualified_cold_four_current_probe_evidence_is_admitted(evidence, monkeypatch, tmp_path):
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(tmp_path/"new", evidence.identity) as store:
        with pytest.raises(ValueError, match="assessment|identity"):
            lightweight_study(monkeypatch, evidence.cfg, store)
        assert not store.is_complete("decision-lineage")


def test_resume_rejects_changed_evidence_root_even_if_its_bytes_are_identical(evidence, monkeypatch, tmp_path):
    root = tmp_path/"new"
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity) as store:
        lightweight_study(monkeypatch, evidence.cfg, store)
    alternative = tmp_path/"other-evidence"
    shutil.copytree(evidence.assessed, alternative)
    with wrapper.bindings(evidence.parent, alternative), store_at(root, evidence.identity, resume=True) as store:
        with pytest.raises(ValueError, match="sealed decision changed: lineage"):
            lightweight_study(monkeypatch, evidence.cfg, store)


@pytest.mark.parametrize("omitted,gap,should_fallback", [(1., 0., True), (1., 1e-6, True),
    (1., None, True), (1., 1.1e-6, False), (None, None, False), (0., 0., False)])
def test_tied_positive_cutoff_falls_back_without_refining_or_adding_modes(monkeypatch, omitted, gap, should_fallback):
    approximate, calls = wrapper.probe.approximate_fisher, []

    def candidate(matrix, **kwargs):
        calls.append(kwargs["steps"])
        packed, diagnostics = approximate(matrix, **kwargs)
        diagnostics["compression"].update(largest_omitted_core_eigenvalue=omitted,
            cutoff_relative_gap=gap, ritz_residual_relative_fro=0.)
        return packed, diagnostics

    monkeypatch.setattr(wrapper.probe, "approximate_fisher", candidate)
    packed, diagnostics = wrapper.compress_fisher(torch.eye(5, dtype=torch.float64), rank=2, backend="subspace_cpu")
    assert packed["factor"].shape[1] == 2
    assert calls == [4]
    assert (diagnostics["compression"]["chosen_steps"] is None) == should_fallback
    assert ("cluster" in (diagnostics["compression"]["fallback_reason"] or "")) == should_fallback


def make_assets(evidence, monkeypatch):
    def plan(source, cfg, family, replica, partition):
        return dict(family=family, replica=replica, partition=partition,
                    seed=cfg.seed_for(family, replica, "source_plan"))

    monkeypatch.setattr(wrapper, "plan_data", plan)
    with store_at(evidence.parent, evidence.identity, resume=True) as store:
        for replica in (0, 1):
            for epochs in (15, 30, 45):
                unit = f"assets-development-grid-{replica:04d}-c4x6-e{epochs:04d}-r08"
                data = plan(None, evidence.cfg, "development", replica, "development")
                theta = torch.arange(6, dtype=torch.float64) + epochs + replica
                matrix = torch.diag(torch.arange(1, 7, dtype=torch.float64))
                ratio = {15: .6, 30: .55, 45: .4}[epochs]
                state = dict(stage="ready", epoch=epochs, theta=theta, plan_hash=wrapper.run.content_hash(data),
                    training_optimizer=dict(state={0: dict(exp_avg=theta*.2, exp_avg_sq=theta*.3)}),
                    reference_fisher=matrix, initialization_validation=dict(mse=ratio, zero_predictor_mse=1.,
                        per_cell_mse=[ratio]*9, per_cell_zero_mse=[1.]*9),
                    learner=wrapper.learner.initial_state(theta, torch.arange(6), matrix, backend="exact_cpu"))
                store.begin(unit, dict(kind="grid_registration_assets", schema_version=1, family="development",
                    replica=replica, partition="development", plan_hash=state["plan_hash"], init_epochs=epochs,
                    channels=[4, 6], output_dim=54, archive_rank=8, compression_backend="exact_cpu"))
                store.write_json(unit, "plan.json", data)
                store.write_json(unit, "health.json", dict(pretrain_seconds=100., reference_seconds=10., tag=unit))
                store.save_checkpoint(unit, state)
                store.complete(unit)
    return plan


def fake_base_assets(self, family, replica, partition):
    unit = f"assets-development-grid-{replica:04d}-c4x6-e{self.init_epochs:04d}-r64"
    if not self.store.is_complete(unit):
        return "fresh-assets-would-train", None, None, None, None
    state = self.store.load_checkpoint(unit)
    assert state["plan_hash"] == wrapper.run.content_hash(wrapper.plan_data(self.source, self.cfg, "development", replica, partition))
    assert state["learner"]["compression_backend"] == self.compression_backend
    return unit, state, "base-rendered-data", None, None


def test_all_pretrained_epochs_are_reused_with_identical_theta_adam_fisher_plan_and_exact_metadata(evidence, monkeypatch, tmp_path):
    make_assets(evidence, monkeypatch)
    before = fingerprints(evidence.parent)
    monkeypatch.setattr(wrapper.BaseRunner, "assets", fake_base_assets)
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(tmp_path/"new", evidence.identity) as store:
        runner = wrapper.Runner(evidence.cfg, store, None, SimpleNamespace(check=lambda: None))
        for replica in (0, 1):
            for epochs in (15, 30, 45):
                runner.init_epochs = epochs
                unit, state, data, _, _ = runner.assets("development--grid", replica, "development")
                assert data == "base-rendered-data"
                parent = state["source_parent"]
                source = wrapper.sealed(evidence.parent, parent["unit_id"])
                original = wrapper.checkpoint(source, wrapper._identity(evidence.parent)["identity_hash"])
                for key in ("theta", "reference_fisher"):
                    assert torch.equal(state[key], original[key])
                assert torch.equal(state["training_optimizer"]["state"][0]["exp_avg"],
                                   original["training_optimizer"]["state"][0]["exp_avg"])
                expected = {**source["metadata"], "archive_rank": 64, "compression_backend": "subspace_cpu"}
                assert wrapper.read_json(store.unit_path(unit)/"manifest.json")["metadata"] == expected
                assert wrapper.read_json(store.unit_path(unit)/"health.json") == wrapper.read_json(source["path"]/"health.json")
                assert parent["seal_hash"] == source["completion"]["seal_hash"]
                assert state["numerical_policy"] == wrapper.policy(evidence.cfg)
                # Completed reuses load without recompressing or mutating either root.
                again = runner.assets("development--grid", replica, "development")[1]
                assert torch.equal(again["theta"], state["theta"])
        assert runner.assets("development--grid", 9, "development")[0] == "fresh-assets-would-train"
        study = lightweight_study(monkeypatch, evidence.cfg, store)
        study.runner, study.status = runner, dict(changes=[])
        monkeypatch.setattr(study, "progress", lambda **kwargs: None)
        study.initialize()
        assert study.decision("initialization")["epochs"] == 45
        assert study.decision("initialization")["channels"] == [4, 6]
    assert fingerprints(evidence.parent) == before


def test_interrupted_asset_copy_resumes_sealing_without_retraining_or_recompression(evidence, monkeypatch, tmp_path):
    make_assets(evidence, monkeypatch)
    monkeypatch.setattr(wrapper.BaseRunner, "assets", fake_base_assets)
    root = tmp_path/"new"
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity) as store:
        runner = wrapper.Runner(evidence.cfg, store, None, SimpleNamespace(check=lambda: None))
        write = store.write_json

        def interrupt(unit, name, value, **kwargs):
            if name == "health.json":
                raise wrapper.run.StopRequested("after imported checkpoint")
            return write(unit, name, value, **kwargs)

        monkeypatch.setattr(store, "write_json", interrupt)
        with pytest.raises(wrapper.run.StopRequested):
            runner.assets("development--grid", 0, "development")
    with wrapper.bindings(evidence.parent, evidence.assessed), store_at(root, evidence.identity, resume=True) as store:
        monkeypatch.setattr(wrapper.probe, "approximate_fisher", lambda *a, **k: pytest.fail("must reuse imported checkpoint"))
        runner = wrapper.Runner(evidence.cfg, store, None, SimpleNamespace(check=lambda: None))
        unit = runner.assets("development--grid", 0, "development")[0]
        assert store.is_complete(unit)


def test_actual_isotropic_cutoff_has_zero_ritz_but_still_requires_exact_fallback():
    packed, diagnostics = wrapper.compress_fisher(torch.eye(10, dtype=torch.float64), rank=3, backend="subspace_cpu")
    assert packed["factor"].shape[1] == 3
    assert diagnostics["compression"]["chosen_steps"] is None
    assert "cluster" in diagnostics["compression"]["fallback_reason"]
    assert len(diagnostics["compression"]["policy_attempts"]) == 1


def test_run_subcommand_dispatches_existing_cli_and_restores_bindings(monkeypatch, tmp_path):
    original = wrapper.run.Config

    def main(argv):
        assert argv == ["--smoke", "--root", str(tmp_path/"new")]
        assert wrapper.run.Config is wrapper.Config
        return 23

    monkeypatch.setattr(wrapper.run, "main", main)
    assert wrapper.main(["run", "--smoke", "--root", str(tmp_path/"new")]) == 23
    assert wrapper.run.Config is original


def test_wrapper_process_status_checks_module_pid_and_root(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "base_process_alive", lambda root, pid: False)
    monkeypatch.setattr(Path, "read_bytes", lambda self: (
        "python\0-m\0mnist_experiment.plan15_grid_subspace\0run\0--root\0"+str(tmp_path/"new")+"\0").encode())
    assert wrapper.process_alive(tmp_path/"new", 123)
    assert not wrapper.process_alive(tmp_path/"other", 123)
    assert not wrapper.process_alive(tmp_path/"new", None)
