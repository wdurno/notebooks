"""Small CPU initialization tests: no teacher, downloads, simulator or GPU."""

import copy
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.learner import MemoryLearner, predict_histories
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar.storage import RunStore
from numerical_experiment.picar_initialization import numerics as nm
from src.continual import picar_math as pm


class MemoryStore:
    completed = False

    def __init__(self):
        self.state, self.rows, self.progress, self.saves = None, [], None, 0

    def load_checkpoint(self, **kwargs):
        return copy.deepcopy(self.state)

    def checkpoint(self, state, *, metrics=(), progress=None):
        self.state = copy.deepcopy(state)
        self.rows.extend(copy.deepcopy(metrics))
        self.progress = copy.deepcopy(progress)
        self.saves += 1


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
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
def fast_policy(monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(nm, "FIT_POLICY", {**nm.FIT_POLICY,
        "kl_max_passes": 2, "nll_max_passes": 3, "min_passes": 2,
        "patience": 2, "block_items": 5, "diagnostic_histories": 2})
    yield
    torch.set_num_threads(threads)


def model():
    return build_tiny_model(31, hidden_size=2, width=4, depth=1,
                           return_bins=3, context_length=4)


def record(index, episode, length=3):
    rng = torch.Generator().manual_seed(index + 101)
    logits = torch.linspace(-.2, .3, 7)
    probabilities = torch.cat((logits, logits.new_zeros(1))).softmax(0)
    return {"record_id": f"{episode}/{index}", "arrival": index, "episode_id": episode,
        "history": {"images": torch.rand(1, length, 3, 8, 8, generator=rng),
            "executed_actions": torch.zeros(1, length, 4),
            "elapsed": torch.full((1, length, 1), .1),
            "contexts": [[{"text": "Approach the ball and stop.", "issued_at": 0.}]]},
        "baseline": 0., "bin": 2, "logits": logits, "executed_probabilities": probabilities}


@pytest.fixture
def setup():
    config = PicarConfig(seed=71)
    config = replace(config, learning=replace(config.learning,
        temperature=2.8375266913896, action_variance=16., learning_rate=.01,
        pi=.9, ridge=100., fd_strength=100., optimizer_steps=8, fisher_draws=1))
    live = model()
    pristine = MemoryLearner(live, config.learning, config.seeds(0)).state_dict()
    train = [record(0, "train-a"), record(1, "train-a", 1), record(2, "train-b", 2)]
    validation = [record(4, "validation", 2)]
    return live, config, pristine, train, validation


def runner(setup, store=None, callback=lambda: None, *, fitted=None, live=None, train=None):
    original, config, pristine, records, validation = setup
    return nm.InitializationFit(live if live is not None else original, config,
        records if train is None else train, validation, store if store is not None else MemoryStore(),
        callback, pristine=pristine, acquired_count=9, fitted_state=fitted)


def fitted(setup):
    fit = runner(setup)
    summary = fit.run_fit()
    return fit, summary


def test_fit_is_ltm_only_unpenalized_multiepoch_and_full_context(setup, monkeypatch):
    _, _, pristine, train, validation = setup
    original = copy.deepcopy(pristine)
    data_before = copy.deepcopy(train + validation)

    def forbidden(*args, **kwargs):
        raise AssertionError("online regularization/retirement entered initialization")

    monkeypatch.setattr(MemoryLearner, "fit", forbidden)
    monkeypatch.setattr(MemoryLearner, "retire", forbidden)
    for name in ("regularized_objective", "eviction_ema", "stm_fd_penalty"):
        monkeypatch.setattr(pm, name, forbidden)
    fit, summary = fitted(setup)
    assert fit.s["stage"] == "fit_done"
    assert fit.s["distilled_learner"]
    assert summary["initial_pi_used"] is summary["initial_penalties_used"] is False
    assert summary["counts"]["optimization_exposures"] == {"kl": 6, "nll": 9}
    assert [r["epoch"] for r in summary["curves"] if r["stage"] == "kl"] == [0, 1, 2]
    assert [r["epoch"] for r in summary["curves"] if r["stage"] == "nll"] == [0, 1, 2, 3]
    assert fit._batch([train[0]]).histories[0] is train[0]["history"]
    assert fit._batch([train[0]]).histories[0]["images"].shape[1] == 3
    assert fit.s["learner"]["optimizer"]["state"] == {}
    assert fit.s["distilled_learner"]["optimizer"]["state"] == {}
    assert fit.s["learner"]["retired_count"] == 0
    assert set(fit.learner.banks["ltm"]) | set(fit.learner.banks["stm"]) == set(fit.learner.params)
    for name in fit.learner.banks["stm"]:
        equal(pristine["parameters"][name], fit.s["learner"]["parameters"][name])
        equal(pristine["parameters"][name], fit.s["distilled_learner"]["parameters"][name])
        assert fit.learner.params[name].requires_grad
    assert any(not torch.equal(pristine["parameters"][n], fit.s["learner"]["parameters"][n]) for n in fit.ltm)
    equal(original, pristine)
    equal(data_before, train + validation)
    assert fit.s["eligible"] == []
    assert fit.s["metrics_cursor"] == len(fit.store.rows) == len(summary["curves"])
    json.dumps(summary, allow_nan=False)


def test_progress_flat_integer_counts_and_committed_scalar_curves(setup):
    fit, summary = fitted(setup)
    progress = fit.store.progress
    assert all(type(v) is int and v >= 0 for v in progress["counts"].values())
    assert progress["counts"]["optimization_exposures"] == 15
    assert progress["counts"]["kl_steps"] == 6
    equal(progress["diagnostic_curves"], summary["curves"])
    json.dumps(progress, allow_nan=False)


def test_fixed_dataset_means_and_stagewise_validation_selection(setup):
    fit, summary = fitted(setup)
    nll_rows = [r for r in summary["curves"] if r["stage"] == "nll"]
    for stage, objective in (("kl", "teacher_kl"), ("nll", "joint_nll")):
        rows = [r for r in summary["curves"] if r["stage"] == stage]
        selected = min(rows, key=lambda r: r["validation"][objective])
        assert summary["selection"][stage]["epoch"] == selected["epoch"]
        assert summary["selection"][stage]["validation_objective"] == selected["validation"][objective]
        for row in rows:
            for split in ("train", "validation"):
                assert row[split]["joint_nll"] == pytest.approx(row[split]["actor_nll"] + row[split]["critic_nll"])
            if row["epoch"]:
                assert row["diagnostics"]["steps"] == len(fit.train)
                assert row["diagnostics"]["gradient_norm_mean"] > 0
                assert row["diagnostics"]["update_norm_mean"] > 0
    accepted = next(r for r in nll_rows if r["epoch"] == summary["selection"]["nll"]["epoch"])
    for split, records in (("train", fit.train), ("validation", fit.validation)):
        with torch.no_grad():
            expected = fit._losses(records)
        for key, value in expected.items():
            assert accepted[split][key] == pytest.approx(float(value.mean()), abs=3e-6)
    assert summary["readiness"]["generalization_established"] is False
    assert summary["readiness"]["global_mle_claimed"] is False
    assert summary["readiness"]["final_B_required"] is False
    assert summary["readiness"]["data_adequacy"]["status"] == "unresolved"


def test_future_stm_B_gradients_are_live_after_fit(setup):
    fit, _ = fitted(setup)
    fit.model.zero_grad(set_to_none=True)
    fit._losses(fit.train)["joint_nll"].mean().backward()
    b_names = [n for n in fit.learner.banks["stm"] if n.endswith(".B")]
    assert b_names
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for n, p in fit.learner.params.items() if n in b_names)
    for n in fit.learner.banks["stm"]:
        assert bool(torch.count_nonzero(fit.learner.params[n])) == n.endswith(".A")


def test_each_rung_resets_pristine_even_after_another_fit(setup):
    fit, first = fitted(setup)
    second = runner(setup, live=fit.model)
    equal(second.learner.state_dict(), setup[2])
    result = second.run_fit()
    equal(first["curves"], result["curves"])
    equal(fit.s["learner"], second.s["learner"])


@pytest.mark.parametrize("stage,phase", [("kl", "train"), ("kl", "evaluate"), ("nll", "train"), ("nll", "evaluate")])
def test_midfit_resume_optimizer_order_global_rng_and_metrics(setup, stage, phase):
    nm.FIT_POLICY["block_items"] = 1
    full, full_summary = fitted(setup)
    store = MemoryStore()

    def interrupt():
        s = store.state
        if s["stage"] == stage and s["phase"] == phase and 0 < s["cursor"] < len(setup[3]):
            raise InterruptedError("test stop after committed block")

    with pytest.raises(InterruptedError):
        runner(setup, store, interrupt).run_fit()
    state = copy.deepcopy(store.state)
    torch.rand(17)
    np.random.random(17)
    random.random()
    resumed = runner(setup, store, live=model())
    equal(nm._global_rng(resumed.learner.device), state["rng"])
    equal(resumed.optimizer.state_dict(), state["fit_optimizer"])
    summary = resumed.run_fit()
    equal(full_summary["curves"], summary["curves"])
    equal(full.s["learner"], resumed.s["learner"])
    equal(full.s["rng"], resumed.s["rng"])
    equal(full.s["order_rng"], resumed.s["order_rng"])
    equal(full.store.rows, store.rows)
    assert len(store.rows) == store.state["metrics_cursor"]


def test_bulk_fisher_equal_weights_archive_counts_anchor_and_no_ema(setup, monkeypatch):
    fit, _ = fitted(setup)
    state = copy.deepcopy(fit.s)
    bulk = runner(setup, fitted=state)
    before = bulk.learner.state_dict()
    expected = {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in bulk.ltm.items()}
    for record in bulk.train:
        sampled = bulk._score_draws(record)["4"]
        for n, v in sampled.items():
            expected[n].add_(v / len(bulk.train))

    def forbidden(*args, **kwargs):
        raise AssertionError("initial Fisher used online EMA")

    monkeypatch.setattr(pm, "eviction_ema", forbidden)
    monkeypatch.setattr(MemoryLearner, "retire", forbidden)
    summary = bulk.run_fisher()
    assert bulk.s["stage"] == "done"
    assert bulk.s["identity"]["config"] == setup[1].to_dict()
    assert bulk.s["retired"] == [0, 1, 2]
    assert bulk.s["eligible"] == []
    assert bulk.s["learner"]["retired_count"] == 3
    assert bulk.s["acquired_count"] == 9
    assert bulk.s["selected_exploration_std"] == setup[1].learning.exploration_std
    assert bulk.s["exploration_calibration"] is None
    assert bulk.s["distill_receipt"]
    assert bulk.s["learner"]["optimizer"]["state"] == {}
    assert len(bulk.s["learner"]["optimizer"]["param_groups"][0]["params"]) == len(bulk.learner.params)
    equal(before["parameters"], bulk.s["learner"]["parameters"])
    for n in bulk.ltm:
        torch.testing.assert_close(bulk.learner.diagonal[n].double(), expected[n], rtol=6e-8, atol=1e-10)
        equal(bulk.s["learner"]["anchor"][n], state["learner"]["parameters"][n])
    equal(state, fit.s)
    assert summary["normalization"]["history_weight"] == 1 / 3
    assert summary["normalization"]["summed_information_trace"] == 3 * summary["normalization"]["mean_trace"]
    assert summary["normalization"]["pi"] is None
    assert summary["normalization"]["projection_or_floor"] is False
    assert summary["counts"]["model_score_draws"] == 12
    assert summary["counts"]["independent_diagnostic_draws"] == 8
    assert [r["draws_per_history"] for r in summary["fisher_stability"]["nested_draws"]] == [1, 2, 4]
    assert summary["fisher_stability"]["nested_draws"][-1]["relative_l2"] == 0
    assert summary["archive"]["validation_admitted"] == 0
    assert summary["local_quadratic"]["rows"]
    for row in summary["local_quadratic"]["rows"]:
        assert row["exact_predictive_kl"] >= 0
        assert row["corrected_observed_nll_change"] == pytest.approx(row["observed_nll_change"] - row["first_order"])
    json.dumps(summary, allow_nan=False)


def test_per_history_scores_use_detached_model_labels_and_nested_draws(setup, monkeypatch):
    fit, _ = fitted(setup)
    bulk = runner(setup, fitted=fit.s)
    record = bulk.train[0]
    batch = bulk._batch([record])
    ltm = bulk.ltm
    output = predict_histories(bulk.model, ltm, batch.histories)
    design = bulk.learner.design(batch)
    labels_seen = []
    sampler = pm.sample_model_labels

    def sample(prediction, fixed, *, generator):
        labels, actions = sampler(prediction, fixed, generator=generator)
        assert not labels.requires_grad and not actions.requires_grad
        labels_seen.append((labels, actions))
        return labels, actions

    monkeypatch.setattr(pm, "sample_model_labels", sample)
    nested = bulk._score_draws(record)
    assert len(labels_seen) == 4
    assert any(not torch.equal(actions[0], record["logits"]) for _, actions in labels_seen)
    expected = {n: torch.zeros(p.shape, dtype=torch.float64) for n, p in ltm.items()}
    for i, (labels, actions) in enumerate(labels_seen, 1):
        scores = pm.per_record_scores(lambda p: predict_histories(bulk.model, p, batch.histories),
                                     ltm, labels, actions, design)
        for n, score in scores.items():
            expected[n].add_(score[0].double().square())
        if i in (1, 2, 4):
            for n in expected:
                torch.testing.assert_close(nested[str(i)][n], expected[n] / i)
    independent = bulk._score_draws(record, independent=True)
    assert any(not torch.equal(nested["4"][n], independent["4"][n]) for n in nested["4"])
    labels, actions = labels_seen[0]
    gradients = torch.autograd.grad(pm.joint_nll(output, labels, actions, design).sum(), tuple(ltm.values()), allow_unused=True)
    scores = pm.per_record_scores(lambda p: predict_histories(bulk.model, p, batch.histories), ltm, labels, actions, design)
    for (n, p), g in zip(ltm.items(), gradients):
        torch.testing.assert_close(scores[n][0], -g if g is not None else torch.zeros_like(p))


@pytest.mark.parametrize("which", ["bulk", "independent", "local"])
def test_midfisher_resume_no_duplicate_draws_or_archive(setup, which):
    nm.FIT_POLICY["block_items"] = 1
    fit, _ = fitted(setup)
    full = runner(setup, fitted=fit.s)
    full_result = full.run_fisher()
    store = MemoryStore()

    def interrupt():
        s, f = store.state, store.state["fisher"]
        if ((which == "bulk" and f["count"] == 1)
                or (which == "independent" and f["independent_count"] == 1)
                or (which == "local" and s["stage"] == "local_quadratic")):
            raise InterruptedError("saved Fisher block")

    with pytest.raises(InterruptedError):
        runner(setup, store, interrupt, fitted=fit.s).run_fisher()
    resumed = runner(setup, store, fitted=fit.s, live=model())
    result = resumed.run_fisher()
    equal(full.s["learner"], resumed.s["learner"])
    equal(full.s["fisher"], resumed.s["fisher"])
    equal(full_result["local_quadratic"], result["local_quadratic"])
    equal(full_result["fisher_stability"], result["fisher_stability"])
    assert resumed.s["retired"] == [0, 1, 2]
    assert len(store.rows) == 0


def test_fisher_record_order_and_block_partition_invariance(setup):
    fit, _ = fitted(setup)
    whole = runner(setup, fitted=fit.s)
    whole.run_fisher()
    permuted = runner(setup, fitted=fit.s, train=list(reversed(setup[3])))
    permuted.policy["block_items"] = 1
    permuted.run_fisher()
    equal(whole.s["learner"]["diagonal"], permuted.s["learner"]["diagonal"])
    equal(whole.s["fisher"], permuted.s["fisher"])


def test_real_store_immutable_and_completed_no_training_path(setup, tmp_path, monkeypatch):
    path = tmp_path / "fit"
    with RunStore(path, {"test": "initialization"}, checkpoint_policy="journal-v1") as store:
        fit = runner(setup, store)
        result = fit.run_fit()
        store.complete(result)
    marker = (path / "COMPLETED").read_bytes()
    with RunStore(path, {"test": "initialization"}, resume=True, checkpoint_policy="journal-v1") as store:
        fit = runner(setup, store)

        def forbidden(*args, **kwargs):
            raise AssertionError("completed resume did numerical work")

        monkeypatch.setattr(fit, "_train_batch", forbidden)
        monkeypatch.setattr(fit, "save", forbidden)
        equal(result["curves"], fit.run_fit()["curves"])
        with pytest.raises(Exception, match="completed|immutable"):
            store.checkpoint(fit.s)
    assert (path / "COMPLETED").read_bytes() == marker


@pytest.mark.parametrize("invalid", ["overlap", "duplicate", "acquired", "temperature", "variance", "zero_A", "nonzero_B", "dirty_pristine"])
def test_invalid_data_likelihood_or_pristine_fails_early(setup, invalid):
    live, config, pristine, train, val = setup
    pristine, train, val = copy.deepcopy(pristine), copy.deepcopy(train), copy.deepcopy(val)
    acquired = 9
    if invalid == "overlap":
        val[0]["episode_id"] = train[0]["episode_id"]
    elif invalid == "duplicate":
        val[0]["arrival"] = train[0]["arrival"]
    elif invalid == "acquired":
        acquired = 3
    elif invalid == "temperature":
        config = replace(config, learning=replace(config.learning, temperature=1.))
    elif invalid == "variance":
        config = replace(config, learning=replace(config.learning, action_variance=1.))
    elif invalid in {"zero_A", "nonzero_B"}:
        name = next(n for n in pristine["banks"]["stm"] if n.endswith(".A" if invalid == "zero_A" else ".B"))
        pristine["parameters"][name].fill_(0 if invalid == "zero_A" else 1)
    else:
        pristine["update_count"] = 1
    with pytest.raises(ValueError):
        nm.InitializationFit(live, config, train, val, MemoryStore(), lambda: None,
            pristine=pristine, acquired_count=acquired)


def test_fisher_rejects_missing_fit_and_changed_accepted_identity(setup):
    fit, _ = fitted(setup)
    with pytest.raises(ValueError, match="fitted_state"):
        runner(setup).run_fisher()
    bad = copy.deepcopy(fit.s)
    bad["learner"]["parameters"][next(iter(fit.ltm))].add_(1)
    with pytest.raises(ValueError, match="parameters"):
        runner(setup, fitted=bad)
    bad = copy.deepcopy(fit.s)
    bad["stage"] = "nll"
    with pytest.raises(ValueError, match="completed"):
        runner(setup, fitted=bad)


def test_zero_diagonal_relative_diagnostics_are_honest():
    zero = {"x": torch.zeros(2, dtype=torch.float64)}
    diagnostic = nm._disagreement(zero, zero, 1e-12)
    assert diagnostic == {"absolute_l2": 0., "reference_l2": 0., "relative_l2": None, "near_zero_reference": True}


def test_model_scores_match_analytic_categorical_and_gaussian_bias_gradients(setup, monkeypatch):
    fit, _ = fitted(setup)
    bulk = runner(setup, fitted=fit.s)
    actor_name, critic_name = "action_head.ltm.bias", "return_head.ltm.bias"

    def bias_predict(model, parameters, histories):
        # Tiny exactly differentiable output chart; all other coordinates are unused.
        return {"return_logits": parameters[critic_name].reshape(1, 3),
                "action_mean": parameters[actor_name].reshape(1, 7)}

    observed = []

    def fixed_labels(output, design, *, generator):
        label = len(observed) % 3
        labels = torch.tensor([label])
        noise = torch.linspace(-1., 1., 7).reshape(1, 7)
        actions = (output["action_mean"].detach() + 4 * noise / design.weights[0, label].sqrt()).detach()
        observed.append((labels, actions, design))
        return labels, actions

    monkeypatch.setattr(nm, "predict_histories", bias_predict)
    monkeypatch.setattr(pm, "sample_model_labels", fixed_labels)
    nested = bulk._score_draws(bulk.train[0])
    categorical = torch.zeros(3, dtype=torch.float64)
    gaussian = torch.zeros(7, dtype=torch.float64)
    probabilities = bulk.ltm[critic_name].detach().softmax(-1)
    mean = bulk.ltm[actor_name].detach()
    for draw, (labels, actions, design) in enumerate(observed, 1):
        label = int(labels[0])
        one_hot = torch.nn.functional.one_hot(torch.tensor(label), 3).float()
        categorical.add_((probabilities - one_hot).double().square())
        gaussian.add_((design.weights[0, label] * (mean - actions[0]) / 16).double().square())
        if draw in (1, 2, 4):
            torch.testing.assert_close(nested[str(draw)][actor_name], gaussian / draw)
            torch.testing.assert_close(nested[str(draw)][critic_name], categorical / draw)
    for name, value in nested["4"].items():
        if name not in {actor_name, critic_name}:
            assert not bool(value.count_nonzero())


@pytest.mark.parametrize("invalid", ["negative_count", "overshoot", "early_independent", "shape", "negative_sum"])
def test_fisher_resume_rejects_inconsistent_counts_or_sums(setup, invalid):
    fit, _ = fitted(setup)
    store = MemoryStore()
    bulk = runner(setup, store, fitted=fit.s)
    bulk.save()
    fisher = store.state["fisher"]
    if invalid == "negative_count":
        fisher["count"] = -1
    elif invalid == "overshoot":
        fisher["count"] = len(setup[3]) + 1
    elif invalid == "early_independent":
        fisher["independent_count"] = 1
    else:
        name = next(iter(bulk.ltm))
        if invalid == "shape":
            fisher["nested_sums"]["4"][name] = torch.zeros(1)
        else:
            fisher["nested_sums"]["4"][name].fill_(-1)
    with pytest.raises(ValueError, match="Fisher"):
        runner(setup, store, fitted=fit.s)


def test_nonfinite_model_score_does_not_advance_fisher_cursor(setup, monkeypatch):
    fit, _ = fitted(setup)
    bulk = runner(setup, fitted=fit.s)
    bulk.save()
    original = torch.autograd.grad

    def invalid_grad(*args, **kwargs):
        gradients = list(original(*args, **kwargs))
        index = next(i for i, g in enumerate(gradients) if g is not None)
        gradients[index] = torch.full_like(gradients[index], float("inf"))
        return tuple(gradients)

    monkeypatch.setattr(torch.autograd, "grad", invalid_grad)
    with pytest.raises(FloatingPointError, match="model score"):
        bulk.run_fisher()
    assert bulk.s["fisher"]["count"] == bulk.store.state["fisher"]["count"] == 0


def test_completed_fisher_resume_returns_summary_without_scores_or_writes(setup, monkeypatch):
    fit, _ = fitted(setup)
    store = MemoryStore()
    bulk = runner(setup, store, fitted=fit.s)
    result = bulk.run_fisher()
    store.completed = True
    resumed = runner(setup, store, fitted=fit.s)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed Fisher resume did work")

    monkeypatch.setattr(resumed, "_score_draws", forbidden)
    monkeypatch.setattr(resumed, "save", forbidden)
    equal(result, resumed.run_fisher())
