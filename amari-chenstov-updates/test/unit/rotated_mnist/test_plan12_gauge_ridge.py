from __future__ import annotations

import torch

from mnist_experiment.rotated_mnist.plan12.gauge import (
    GAUGE_FIXED_PARAMETER_COUNT,
    build_gauge_fixed_model,
    chart_embedding,
    exact_gauge_basis,
    load_canonical_state,
    load_gauge_state,
)
from mnist_experiment.rotated_mnist.plan12.ridge import RidgeFisher, displacement_components
from src.mnist_model import build_canonical_model
from src.representations import DenseFisher


def test_gauge_chart_is_functionally_equivalent_and_has_487_coordinates() -> None:
    raw, raw_layout = build_canonical_model(123, dtype=torch.float64)
    chart, chart_layout = build_gauge_fixed_model(456, dtype=torch.float64)
    load_canonical_state(chart, raw)
    inputs = torch.randn(5, 1, 28, 28, dtype=torch.float64)
    raw_logits = raw(inputs)
    chart_logits = chart(inputs)
    difference = raw_logits - chart_logits
    assert chart_layout.total_numel == GAUGE_FIXED_PARAMETER_COUNT
    torch.testing.assert_close(
        difference,
        difference.mean(dim=1, keepdim=True).expand_as(difference),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(raw_logits.softmax(1), chart_logits.softmax(1), rtol=1e-11, atol=1e-11)

    centered_raw, _ = build_canonical_model(789, dtype=torch.float64)
    load_gauge_state(centered_raw, chart)
    torch.testing.assert_close(centered_raw(inputs), chart_logits, rtol=1e-11, atol=1e-11)
    assert raw_layout.total_numel - chart_layout.total_numel == 25


def test_chart_embedding_and_exact_gauge_form_an_orthonormal_basis() -> None:
    raw, raw_layout = build_canonical_model(1, dtype=torch.float64)
    chart, chart_layout = build_gauge_fixed_model(2, dtype=torch.float64)
    del raw, chart
    embedding = chart_embedding(raw_layout, chart_layout)
    gauge = exact_gauge_basis(raw_layout)
    full = torch.cat((embedding, gauge), dim=1)
    torch.testing.assert_close(full.mT @ full, torch.eye(512, dtype=torch.float64), rtol=0, atol=2e-13)


def test_chart_score_is_raw_score_projected_away_from_gauge() -> None:
    raw, raw_layout = build_canonical_model(4, dtype=torch.float64)
    chart, chart_layout = build_gauge_fixed_model(5, dtype=torch.float64)
    load_canonical_state(chart, raw)
    inputs = torch.randn(3, 1, 28, 28, dtype=torch.float64)
    targets = torch.tensor([1, 4, 9])
    raw_loss = torch.nn.functional.cross_entropy(raw(inputs), targets)
    chart_loss = torch.nn.functional.cross_entropy(chart(inputs), targets)
    raw_gradient = torch.cat([part.reshape(-1) for part in torch.autograd.grad(raw_loss, tuple(raw.parameters()))])
    chart_gradient = torch.cat([part.reshape(-1) for part in torch.autograd.grad(chart_loss, tuple(chart.parameters()))])
    embedding = chart_embedding(raw_layout, chart_layout)
    gauge = exact_gauge_basis(raw_layout)
    torch.testing.assert_close(chart_gradient, embedding.mT @ raw_gradient, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(gauge.mT @ raw_gradient, torch.zeros(25, dtype=torch.float64), rtol=0, atol=2e-11)


def test_isotropic_and_tail_ridge_quadratics_match_dense_forms() -> None:
    generator = torch.Generator().manual_seed(12)
    factor = torch.randn(7, 7, generator=generator, dtype=torch.float64)
    base_matrix = factor @ factor.mT
    base = DenseFisher(base_matrix)
    q, _ = torch.linalg.qr(torch.randn(7, 3, generator=generator, dtype=torch.float64))
    vector = torch.randn(7, generator=generator, dtype=torch.float64)
    kappa = 0.37
    isotropic = RidgeFisher(base, kappa, "isotropic")
    tail = RidgeFisher(base, kappa, "tail", q)
    identity = torch.eye(7, dtype=torch.float64)
    tail_projector = identity - q @ q.mT
    torch.testing.assert_close(isotropic.quadratic(vector), vector @ (base_matrix + kappa * identity) @ vector)
    torch.testing.assert_close(tail.quadratic(vector), vector @ (base_matrix + kappa * tail_projector) @ vector)
    torch.testing.assert_close(tail.matvec(vector), (base_matrix + kappa * tail_projector) @ vector)
    components = displacement_components(vector, q)
    assert abs(components["total_squared"] - components["resolved_squared"] - components["unresolved_squared"]) < 1e-11
