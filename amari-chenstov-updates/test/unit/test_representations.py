import torch

from src.representations import project_psd_frobenius


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
