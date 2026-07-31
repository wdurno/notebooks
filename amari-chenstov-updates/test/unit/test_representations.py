import torch

from src.representations import (
    DenseFisher,
    DiagonalFisher,
    LowRankDiagonalFisher,
    project_psd_frobenius,
    representation_from_artifact,
)


def _structured_fixture() -> tuple[LowRankDiagonalFisher, torch.Tensor]:
    factor = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [-0.25, 0.75], [0.0, -0.5]],
        dtype=torch.float64,
    )
    residual = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)
    representation = LowRankDiagonalFisher(factor, residual)
    return representation, factor @ factor.mT + torch.diag(residual)


def test_fisher_representations_match_explicit_dense_operations() -> None:
    low_rank, matrix = _structured_fixture()
    dense = DenseFisher(matrix)
    diagonal = DiagonalFisher(torch.diagonal(matrix))
    vector = torch.tensor([0.5, -1.0, 0.25, 2.0], dtype=torch.float64)
    probes = torch.stack((vector, vector.flip(0)), dim=1)

    torch.testing.assert_close(low_rank.matvec(vector), matrix @ vector)
    torch.testing.assert_close(low_rank.matvec(probes), matrix @ probes)
    torch.testing.assert_close(low_rank.quadratic(vector), vector @ matrix @ vector)
    torch.testing.assert_close(low_rank.diagonal_vector(), torch.diagonal(matrix))
    torch.testing.assert_close(dense.matvec(vector), matrix @ vector)
    torch.testing.assert_close(
        diagonal.matvec(vector),
        torch.diagonal(matrix) * vector,
    )


def test_structured_damped_solve_and_inverse_trace_match_dense() -> None:
    representation, matrix = _structured_fixture()
    right_hand_side = torch.tensor(
        [[1.0, -0.5], [0.0, 2.0], [-1.0, 0.25], [0.5, 1.0]],
        dtype=torch.float64,
    )
    damping = 0.3
    damped = matrix + damping * torch.eye(4, dtype=torch.float64)

    torch.testing.assert_close(
        representation.damped_solve(right_hand_side, damping),
        torch.linalg.solve(damped, right_hand_side),
    )
    torch.testing.assert_close(
        representation.inverse_trace(damping),
        torch.trace(torch.linalg.inv(damped)),
    )


def test_rank_zero_low_rank_representation_is_diagonal() -> None:
    values = torch.tensor([0.25, 1.0, 2.0], dtype=torch.float64)
    low_rank = LowRankDiagonalFisher(
        torch.empty((3, 0), dtype=torch.float64),
        values,
    )
    diagonal = DiagonalFisher(values)
    vector = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)

    torch.testing.assert_close(low_rank.to_dense(), diagonal.to_dense())
    torch.testing.assert_close(low_rank.matvec(vector), diagonal.matvec(vector))
    torch.testing.assert_close(
        low_rank.damped_solve(vector, 0.1),
        diagonal.damped_solve(vector, 0.1),
    )


def test_representation_artifact_round_trip() -> None:
    representation, _ = _structured_fixture()

    restored = representation_from_artifact(representation.artifact_mapping())

    assert isinstance(restored, LowRankDiagonalFisher)
    torch.testing.assert_close(restored.factor, representation.factor)
    torch.testing.assert_close(
        restored.residual_diagonal,
        representation.residual_diagonal,
    )
    assert restored.storage_bytes() == (
        representation.factor.numel()
        + representation.residual_diagonal.numel()
    ) * representation.factor.element_size()


def test_frobenius_projection_is_symmetric_psd_and_reports_diagnostics() -> None:
    matrix = torch.tensor(
        [[2.0, 3.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.5]],
        dtype=torch.float64,
    )

    result = project_psd_frobenius(matrix)
    eigenvalues = torch.linalg.eigvalsh(result.projected)

    torch.testing.assert_close(result.projected, result.projected.mT)
    assert float(eigenvalues.min()) >= -1e-12
    assert result.diagnostics.symmetry_error_fro > 0
    assert result.diagnostics.minimum_eigenvalue < 0
    assert result.diagnostics.negative_eigenvalue_count == 1
    assert result.diagnostics.negative_spectral_mass > 0
    assert result.diagnostics.projection_distance_fro > 0


def test_psd_projection_does_not_increase_distance_to_psd_target() -> None:
    generator = torch.Generator().manual_seed(456)
    factor = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    target = factor @ factor.mT
    perturbation = torch.tensor(
        [
            [-8.0, 0.5, 0.0, 0.0],
            [0.5, -3.0, 0.2, 0.0],
            [0.0, 0.2, 1.0, 0.1],
            [0.0, 0.0, 0.1, -1.0],
        ],
        dtype=torch.float64,
    )
    candidate = target + perturbation

    result = project_psd_frobenius(candidate)
    before = torch.linalg.matrix_norm(result.symmetrized - target, ord="fro")
    after = torch.linalg.matrix_norm(result.projected - target, ord="fro")

    assert float(after) <= float(before) + 1e-12


def test_psd_input_is_unchanged_up_to_roundoff() -> None:
    factor = torch.tensor(
        [[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]],
        dtype=torch.float64,
    )
    matrix = factor @ factor.mT

    result = project_psd_frobenius(matrix)

    torch.testing.assert_close(result.projected, matrix, atol=1e-12, rtol=1e-12)
    assert result.diagnostics.minimum_eigenvalue >= -1e-12
    assert result.diagnostics.projection_distance_fro < 1e-12
