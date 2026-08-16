import pytest
import torch

from src.lanczos_wrapper import (
    LEGACY_LANCZOS_SHA256,
    approximate_low_rank_diagonal,
    legacy_lanczos_source_hash,
)


def _positive_definite_fixture() -> torch.Tensor:
    generator = torch.Generator().manual_seed(91)
    factor = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    return factor @ factor.mT + 0.25 * torch.eye(8, dtype=torch.float64)


def test_legacy_lanczos_source_is_pinned() -> None:
    assert legacy_lanczos_source_hash() == LEGACY_LANCZOS_SHA256


def test_rank_zero_is_the_clipped_operator_diagonal() -> None:
    diagonal = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    matrix = torch.diag(diagonal)

    result = approximate_low_rank_diagonal(
        lambda vector: matrix @ vector,
        diagonal,
        rank=0,
        seed=12,
    )

    assert result.representation.rank == 0
    torch.testing.assert_close(
        result.representation.residual_diagonal,
        diagonal.clamp_min(0),
    )
    assert result.diagnostics.requested_rank == 0
    assert result.diagnostics.realized_rank == 0


def test_lanczos_approximation_is_deterministic_and_reports_diagonal_error() -> None:
    matrix = _positive_definite_fixture()
    arguments = {
        "operator": lambda vector: matrix @ vector,
        "diagonal": torch.diagonal(matrix),
        "rank": 4,
        "seed": 712,
    }

    first = approximate_low_rank_diagonal(**arguments)
    second = approximate_low_rank_diagonal(**arguments)

    torch.testing.assert_close(
        first.representation.to_dense(),
        second.representation.to_dense(),
    )
    represented_diagonal = first.representation.diagonal_vector()
    assert bool(
        (
            represented_diagonal
            >= torch.diagonal(matrix)
            - torch.finfo(matrix.dtype).eps
        ).all()
    )
    assert first.diagnostics.realized_rank <= 4
    expected_error = torch.linalg.vector_norm(
        represented_diagonal - torch.diagonal(matrix)
    ) / torch.linalg.vector_norm(torch.diagonal(matrix))
    assert first.diagnostics.represented_diagonal_relative_error == pytest.approx(
        float(expected_error)
    )


def test_lanczos_wrapper_rejects_an_invalid_operator_contract() -> None:
    diagonal = torch.ones(4, dtype=torch.float64)

    with pytest.raises(ValueError, match="preserve shape"):
        approximate_low_rank_diagonal(
            lambda vector: vector.reshape(-1),
            diagonal,
            rank=2,
            seed=0,
        )


def test_lanczos_wrapper_retries_a_nonfinite_krylov_depth(monkeypatch) -> None:
    calls = []

    def flaky_lanczos(*, r, p, device, diag_alternate, **_):
        calls.append(r)
        if r == 3:
            return (
                torch.full((p, r), float("nan"), dtype=torch.float64),
                torch.full((p, 1), float("nan"), dtype=torch.float64),
            )
        return (
            torch.zeros((p, r), dtype=torch.float64),
            diag_alternate(),
        )

    monkeypatch.setattr("src.lanczos_wrapper.l_lanczos", flaky_lanczos)
    diagonal = torch.ones(4, dtype=torch.float64)
    result = approximate_low_rank_diagonal(
        lambda vector: vector,
        diagonal,
        rank=3,
        seed=9,
    )

    assert calls == [3, 2]
    assert result.diagnostics.requested_rank == 3
    assert result.diagnostics.executed_krylov_rank == 2
    assert result.diagnostics.numerical_retry_count == 1
