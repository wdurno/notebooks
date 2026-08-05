import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.mnist_data import ReferenceSamplePlan
from src.parameters import ParameterLayout
from src.reference import (
    ReferenceError,
    ReferenceFisherStore,
    adaptive_reference_fisher,
    central_fisher_stencil,
    chunked_lfu_estimate,
    chunked_reference_fisher,
    convergence_diagnostics,
    reference_cache_key,
)


class BernoulliDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.zeros(4, 1, dtype=torch.float64)
        self.targets = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


class BernoulliLogit(nn.Module):
    def __init__(self, probability: float = 0.25) -> None:
        super().__init__()
        logit = math.log(probability / (1 - probability))
        self.logit = nn.Parameter(torch.tensor([logit], dtype=torch.float64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(inputs.shape[0])


def bernoulli_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="sum",
    )


def bernoulli_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )


def _plan(sample_size: int = 4) -> ReferenceSamplePlan:
    indices = tuple(index % 4 for index in range(sample_size))
    labels = tuple(0 if index % 4 < 3 else 1 for index in range(sample_size))
    return ReferenceSamplePlan(
        p=0.25,
        observation_indices=indices,
        class_labels=labels,
        non_nine_sampling="empirical",
        seed=123,
        partition_hash="partition",
    )


def _reference(
    model: nn.Module,
    layout: ParameterLayout,
    plan: ReferenceSamplePlan,
    *,
    chunk_size: int,
):
    return chunked_reference_fisher(
        model,
        BernoulliDataset(),
        plan,
        bernoulli_nll,
        layout,
        chunk_size=chunk_size,
        device=torch.device("cpu"),
        derivative_dtype=torch.float64,
        strategy="vmap",
    )


def test_chunked_and_unchunked_reference_fishers_agree() -> None:
    model = BernoulliLogit()
    layout = ParameterLayout.from_module(model)
    plan = _plan()

    one_chunk = _reference(model, layout, plan, chunk_size=4)
    many_chunks = _reference(model, layout, plan, chunk_size=1)

    torch.testing.assert_close(
        one_chunk.matrix,
        torch.tensor([[0.1875]], dtype=torch.float64),
    )
    torch.testing.assert_close(many_chunks.matrix, one_chunk.matrix)
    assert many_chunks.score_gradient_count == 4
    assert many_chunks.effective_sample_size == pytest.approx(4.0)


def test_adaptive_fisher_stops_on_frobenius_radius() -> None:
    model = BernoulliLogit()
    layout = ParameterLayout.from_module(model)

    estimate = adaptive_reference_fisher(
        model,
        BernoulliDataset(),
        _plan(16),
        bernoulli_nll,
        layout,
        chunk_size=4,
        minimum_chunks=2,
        sigma=6.0,
        relative_epsilon=0.01,
        absolute_epsilon=1e-8,
        device=torch.device("cpu"),
        derivative_dtype=torch.float64,
    )

    assert estimate.sample_count == 8
    assert estimate.convergence["converged"] is True
    assert estimate.convergence["geometry"] == "frobenius"
    assert estimate.convergence["confidence_radius"] == 0.0
    torch.testing.assert_close(
        estimate.matrix,
        torch.tensor([[0.1875]], dtype=torch.float64),
    )


def test_importance_weighted_stencil_recovers_ac_measure_term() -> None:
    model = BernoulliLogit()
    layout = ParameterLayout.from_module(model)
    direction = torch.ones(1, dtype=torch.float64)
    plan = _plan(400)
    dataset = BernoulliDataset()

    lfu = chunked_lfu_estimate(
        model,
        dataset,
        plan,
        bernoulli_nll,
        layout,
        direction,
        chunk_size=40,
        device=torch.device("cpu"),
        derivative_dtype=torch.float64,
    )
    weighted = central_fisher_stencil(
        model,
        dataset,
        plan,
        bernoulli_nll,
        bernoulli_losses,
        layout,
        direction,
        1e-4,
        chunk_size=40,
        device=torch.device("cpu"),
        derivative_dtype=torch.float64,
        importance_weighted=True,
    )
    fixed_measure = central_fisher_stencil(
        model,
        dataset,
        plan,
        bernoulli_nll,
        bernoulli_losses,
        layout,
        direction,
        1e-4,
        chunk_size=40,
        device=torch.device("cpu"),
        derivative_dtype=torch.float64,
        importance_weighted=False,
    )

    expected = torch.tensor([[0.09375]], dtype=torch.float64)
    torch.testing.assert_close(lfu.estimate.amari_chentsov, expected)
    torch.testing.assert_close(
        lfu.estimate.residual,
        torch.zeros_like(expected),
        atol=1e-15,
        rtol=0,
    )
    torch.testing.assert_close(weighted.derivative, expected, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        fixed_measure.derivative,
        torch.zeros_like(expected),
        atol=1e-9,
        rtol=0,
    )
    assert weighted.plus.effective_sample_size > 399.99


def test_reference_cache_key_covers_scientific_inputs() -> None:
    model = BernoulliLogit()
    layout = ParameterLayout.from_module(model)
    plan = _plan()
    base = reference_cache_key(
        model,
        layout,
        plan,
        derivative_dtype=torch.float64,
        matrix_dtype=torch.float64,
    )
    smaller_plan = plan.prefix(3)
    smaller = reference_cache_key(
        model,
        layout,
        smaller_plan,
        derivative_dtype=torch.float64,
        matrix_dtype=torch.float64,
    )
    changed_model = BernoulliLogit(0.3)
    changed_checkpoint = reference_cache_key(
        changed_model,
        ParameterLayout.from_module(changed_model),
        plan,
        derivative_dtype=torch.float64,
        matrix_dtype=torch.float64,
    )
    weighted = reference_cache_key(
        model,
        layout,
        plan,
        derivative_dtype=torch.float64,
        matrix_dtype=torch.float64,
        sampling_model=model,
    )

    assert len(
        {base.digest, smaller.digest, changed_checkpoint.digest, weighted.digest}
    ) == 4


def test_reference_cache_round_trip_and_layout_rejection(tmp_path: Path) -> None:
    model = BernoulliLogit()
    layout = ParameterLayout.from_module(model)
    plan = _plan()
    estimate = _reference(model, layout, plan, chunk_size=2)
    key = reference_cache_key(
        model,
        layout,
        plan,
        derivative_dtype=torch.float64,
        matrix_dtype=torch.float64,
    )
    store = ReferenceFisherStore(tmp_path)

    path = store.save(key, estimate)
    loaded = store.load(key, layout)

    assert (path / "COMPLETED").is_file()
    torch.testing.assert_close(loaded.matrix, estimate.matrix)
    with pytest.raises(ReferenceError, match="already exists"):
        store.save(key, estimate)

    incompatible = nn.Linear(1, 2, dtype=torch.float64)
    incompatible_layout = ParameterLayout.from_module(incompatible)
    with pytest.raises(ReferenceError, match="layout"):
        store.load(key, incompatible_layout)


def test_nested_convergence_diagnostics_use_largest_reference() -> None:
    estimates = {
        2: torch.tensor([[1.0]], dtype=torch.float64),
        4: torch.tensor([[1.5]], dtype=torch.float64),
        8: torch.tensor([[2.0]], dtype=torch.float64),
    }

    diagnostics = convergence_diagnostics(estimates)

    assert diagnostics[0]["relative_to_previous"] is None
    assert diagnostics[0]["relative_to_largest"] == pytest.approx(0.5)
    assert diagnostics[-1]["relative_to_largest"] == 0.0
