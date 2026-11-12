"""Tiny CPU numerical probes; no simulator, historical checkpoints, or GPU.

Probe inputs are a current-likelihood initialization. Historical source
admission remains campaign-owned; these tests check no retirement or transfer
is performed by a scratch probe and that its supplied initialization survives.
"""

import copy
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from numerical_experiment.picar.config import PicarConfig
from numerical_experiment.picar.fix3_student import _batch
from numerical_experiment.picar.learner import MemoryLearner
from numerical_experiment.picar.models import build_tiny_model
from numerical_experiment.picar_phase2_rescue.model import NormalizedStudent
from numerical_experiment.picar_phase3 import calibration


@pytest.fixture(autouse=True)
def cpu_threads_and_rng():
    threads = torch.get_num_threads()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    torch.set_rng_state(rng[2])


def equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == "cpu" and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            equal(value, reference)
    else:
        assert actual == expected


def global_rng():
    return copy.deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state()))


@pytest.fixture
def initialized():
    base = PicarConfig()
    config = replace(base, model=replace(base.model, hidden_size=4, context_length=2, return_bins=7),
        learning=replace(base.learning, fd_strength=0., optimizer_steps=1, temperature=2.,
            replay_capacity=2, replay_batch=1, fisher_draws=1, return_min=-6., return_max=12.),
        runtime=replace(base.runtime, device="cpu"))
    model = NormalizedStudent(build_tiny_model(31, width=4, depth=1, hidden_size=4,
        context_length=2, return_bins=7, dtype=torch.float64), "feature_and_projection_rms").eval()
    records = []
    for index in range(2):
        history = {"images": torch.full((1, 1, 3, 4, 4), .2 + .5 * index, dtype=torch.float64),
            "executed_actions": torch.zeros(1, 1, 4, dtype=torch.float64),
            "elapsed": torch.full((1, 1, 1), .1, dtype=torch.float64),
            "contexts": [[{"text": "Approach the ball.", "issued_at": 0.}]],
            "start_time": torch.zeros(1, dtype=torch.float64),
            "decision_time": torch.full((1, 1), .1, dtype=torch.float64)}
        records.append({"arrival": index, "episode_id": f"fixture:{index}", "history": history,
            "baseline": 0., "bin": index + 2, "logits": torch.tensor([.1, -.2, .3, 0., .2, -.1, 0.], dtype=torch.float64)})
    learner = MemoryLearner(model, config.learning, config.seeds(0))
    assert all(not bool(d.any()) for d in learner.diagonal.values())
    batch = _batch(records[0], records)
    learner.fit(new=batch, replay=None, acquired_count=2)
    learner.retire(batch)
    learner.optimizer.zero_grad(set_to_none=True)
    assert learner.optimizer.state and any(bool(d.any()) for d in learner.diagonal.values())
    initial = {"learner": learner.state_dict(), "eligible": records, "acquired_count": 2}
    return model, config, initial


def observe_learners(monkeypatch):
    observed = []

    def construct(*args, **kwargs):
        learner = MemoryLearner(*args, **kwargs)
        assert all(not bool(d.any()) for d in learner.diagonal.values())
        observed.append(learner)
        return learner

    monkeypatch.setattr(calibration, "MemoryLearner", construct)
    return observed


@pytest.mark.parametrize("probe", [calibration.optimizer_probe, calibration.fd_probe])
def test_real_probe_restores_parameters_adam_current_fisher_anchors_counters_and_all_rng(initialized, monkeypatch, probe):
    model, config, initial = initialized
    learners = observe_learners(monkeypatch)
    before = copy.deepcopy(initial)
    model_before, rng_before = copy.deepcopy(model.state_dict()), global_rng()

    def forbidden(*args, **kwargs):
        raise AssertionError("probe attempted historical checkpoint access, retirement, or CUDA initialization")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(MemoryLearner, "retire", forbidden)
    result = probe(model, config, initial)
    assert result["scratch_state_restored"] is True and result["new_observations"] == 0
    assert len(learners) == 1
    equal(learners[0].state_dict(), before["learner"])
    equal(model.state_dict(), model_before)
    equal(global_rng(), rng_before)
    equal(initial, before)
    assert any(bool(d.any()) for d in learners[0].diagonal.values())
    if probe is calibration.optimizer_probe:
        assert len(result["optimizer_rows"]) == 6
        assert all(r["finite"] for r in result["optimizer_rows"])
    else:
        assert result["fd_resolution_groups"]
        if result["fd_convergence_pass"]:
            assert len(result["fd_strength_rows"]) == 3 and len(result["fit_cost_rows"]) == 2
            assert [r["target_gradient_ratio"] for r in result["fd_strength_rows"]] == [.01, .1, 1.]


@pytest.mark.parametrize("probe", [calibration.optimizer_probe, calibration.fd_probe])
def test_probe_exception_restores_source_and_shared_model_after_a_real_scratch_fit(initialized, monkeypatch, probe):
    model, config, initial = initialized
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, .25)
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    learners = observe_learners(monkeypatch)
    before = copy.deepcopy(initial)
    model_before, rng_before = copy.deepcopy(model.state_dict()), global_rng()
    original_fit = MemoryLearner.fit
    fits = []

    def fail_after_fit(learner, *args, **kwargs):
        result = original_fit(learner, *args, **kwargs)
        fits.append(result)
        raise RuntimeError("injected scratch-fit interruption")

    monkeypatch.setattr(MemoryLearner, "fit", fail_after_fit)
    if probe is calibration.fd_probe:
        # Force a converged, differentiable numerical gate to exercise fit cost rollback.
        monkeypatch.setattr(calibration.pm, "stm_fd_penalty", lambda predict, params, *args:
                            sum(p.square().sum() for p in params.values()) + 1.)
    with pytest.raises(RuntimeError, match="scratch-fit interruption"):
        probe(model, config, initial)
    assert fits
    equal(learners[0].state_dict(), before["learner"])
    equal(model.state_dict(), model_before)
    equal({name: p.grad for name, p in model.named_parameters()}, gradients)
    equal(global_rng(), rng_before)
    equal(initial, before)


@pytest.mark.parametrize("invalid", ["zero", "detached_zero", "detached_positive", "negative", "divergent", "nan_middle",
                                     "ltm_only", "stm_only"])
def test_invalid_fd_resolution_never_produces_a_strength_shortlist_or_fit_cost(initialized, monkeypatch, invalid):
    model, config, initial = initialized
    before = copy.deepcopy(initial)
    calls, fits = [], []

    def penalty(predict, params, design, names, probes, weights, epsilon):
        quadratic = sum(p.square().sum() for p in params.values()) + 1.
        calls.append(epsilon)
        if invalid.startswith("detached_"):
            return quadratic.detach().new_tensor(0. if invalid == "detached_zero" else 1.)
        if invalid.endswith("_only"):
            names = initial["learner"]["banks"][invalid.removesuffix("_only")]
            return sum(params[name].square().sum() for name in names) + 1.
        if invalid == "zero":
            return quadratic * 0.
        if invalid == "negative":
            return -quadratic
        if invalid == "divergent":
            return quadratic * (epsilon / .01) ** 2
        return quadratic + (float("nan") if len(calls) % 3 == 2 else 0.)

    def fit(*args, **kwargs):
        fits.append(kwargs)
        return {"update_seconds": .01}

    monkeypatch.setattr(calibration.pm, "stm_fd_penalty", penalty)
    monkeypatch.setattr(MemoryLearner, "fit", fit)
    result = calibration.fd_probe(model, config, initial)
    assert result["fd_convergence_pass"] is False
    assert result["selected_fd_step"] is None
    assert result["fd_strength_rows"] == result["fit_cost_rows"] == fits == []
    assert len(calls) == 9 and all(g["passed"] is False for g in result["fd_resolution_groups"])
    json.dumps(result, allow_nan=False)
    if invalid == "nan_middle":
        assert all(g["rows"][1]["penalty"] is None for g in result["fd_resolution_groups"])
    equal(initial, before)


@pytest.mark.parametrize("data_norm", [0., float("nan"), float("inf")])
def test_invalid_data_gradient_cannot_pass_fd_resolution(initialized, monkeypatch, data_norm):
    model, config, initial = initialized
    before = copy.deepcopy(initial)
    original_vector = calibration._vector
    calls, fits = [], []

    def vector(loss, params):
        calls.append(None)
        if len(calls) == 1:
            return next(iter(params.values())).detach().new_tensor([data_norm])
        return original_vector(loss, params)

    def fit(*args, **kwargs):
        fits.append(kwargs)
        return {"update_seconds": .01}

    monkeypatch.setattr(calibration, "_vector", vector)
    monkeypatch.setattr(calibration.pm, "stm_fd_penalty", lambda predict, params, *args:
                        sum(p.square().sum() for p in params.values()) + 1.)
    monkeypatch.setattr(MemoryLearner, "fit", fit)
    result = calibration.fd_probe(model, config, initial)
    assert result["fd_convergence_pass"] is False and result["selected_fd_step"] is None
    assert result["fd_strength_rows"] == result["fit_cost_rows"] == fits == []
    assert len(calls) == 10 and all(g["passed"] is False for g in result["fd_resolution_groups"])
    json.dumps(result, allow_nan=False)
    equal(initial, before)


@pytest.mark.parametrize("probe", [calibration.optimizer_probe, calibration.fd_probe])
@pytest.mark.parametrize("existing_gradients", [False, True])
def test_probe_restores_incoming_gradient_buffers(initialized, monkeypatch, probe, existing_gradients):
    model, config, initial = initialized
    if existing_gradients:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, .125)
    before = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    if probe is calibration.fd_probe:
        monkeypatch.setattr(calibration.pm, "stm_fd_penalty", lambda predict, params, *args:
                            sum(p.square().sum() for p in params.values()) + 1.)
    probe(model, config, initial)
    equal({name: p.grad for name, p in model.named_parameters()}, before)
