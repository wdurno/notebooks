"""Conditional Akaike score gate in weighted mean-negative-log-likelihood units.

``g`` is the data-loss gradient (negative score); ``r`` is the retained
objective gradient, including the *actual* penalty gradient. Only A+B moves.
Dense float64 algebra here is an explicit small-model diagnostic realization.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _square(value: Tensor, name: str, dimension: int | None = None,
            device: torch.device | None = None) -> Tensor:
    value = torch.as_tensor(value, device=device).detach().to(torch.float64)
    if (value.ndim != 2 or value.shape[0] != value.shape[1]
            or (dimension is not None and value.shape[0] != dimension)
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite square matrix in the global chart")
    return (value + value.mT) / 2


def _coordinates(value, dimension: int, device: torch.device) -> Tensor:
    value = torch.as_tensor(value, device=device)
    if value.numel() == 0 and value.ndim == 1:
        return value.to(torch.int64)
    if (value.ndim != 1 or value.dtype not in (torch.int64, torch.int32)
            or value.min() < 0 or value.max() >= dimension
            or value.unique().numel() != value.numel()):
        raise ValueError("coordinate sets must be unique integer GLOBAL indices in range")
    return value.to(torch.int64)


def _block(value: Tensor, indices: Tensor) -> Tensor:
    return value.index_select(0, indices).index_select(1, indices)


def _response(matrix: Tensor, floor_relative: float, *, damp: bool = True) -> tuple[Tensor, dict]:
    """Supported spectral solve; a numerical floor never creates information.

    Positive supported modes alone may receive numerical damping. Trace-based
    optimism and dimension use the undamped supported response, so a solve
    floor cannot masquerade as statistical ridge. Null-space RHS components
    are diagnosed by the caller instead of turned into huge updates.
    """
    if not math.isfinite(floor_relative) or floor_relative < 0:
        raise ValueError("floor_relative must be finite and nonnegative")
    d = matrix.shape[0]
    if d == 0:
        return matrix.clone(), {"rank": 0, "dimension": 0, "min_eigenvalue": 0.,
                                "min_resolved_eigenvalue": 0., "condition_estimate": 0.,
                                "floor": 0., "damping": 0., "damped_modes": 0,
                                "indefinite": False}
    values, vectors = torch.linalg.eigh((matrix + matrix.mT) / 2)
    eps = torch.finfo(torch.float64).eps
    scale = float(values.abs().max())
    tolerance = d * eps * scale
    support = values > tolerance
    positive_diagonal = matrix.diagonal().clamp_min(0)
    floor = max(floor_relative * float(positive_diagonal.mean()), eps)
    denominators = values[support]
    used = denominators.clamp_min(floor) if damp else denominators
    inverse = (vectors[:, support] / used) @ vectors[:, support].mT
    damping = used - denominators
    diagnostics = {
        "rank": int(support.sum()), "dimension": d,
        "min_eigenvalue": float(values.min()),
        "min_resolved_eigenvalue": float(denominators.min()) if denominators.numel() else 0.,
        "condition_estimate": float(denominators.max() / denominators.min()) if denominators.numel() else 0.,
        "floor": floor, "damping": float(damping.max()) if damping.numel() else 0.,
        "damped_modes": int((damping > 0).sum()),
        "indefinite": bool(values.min() < -100 * tolerance),
    }
    return (inverse + inverse.mT) / 2, diagnostics


def _relative_residual(matrix: Tensor, displacement: Tensor, rhs: Tensor) -> float:
    numerator = torch.linalg.vector_norm(matrix @ displacement - rhs)
    denominator = torch.linalg.vector_norm(rhs)
    return float(numerator / denominator) if denominator > 0 else float(numerator)


def block_preview(
    r: Tensor, K: Tensor, active_indices, candidate_indices, *, floor_relative: float = 1e-8,
) -> dict:
    """Schur-profile a conditional expansion and its active-only correction.

    Returns full-coordinate vectors; all coordinates outside A+B are zero.
    Nonzero r_A is permitted for diagnostics, never treated as candidate gain.
    """
    K = _square(K, "K")
    r = torch.as_tensor(r, device=K.device, dtype=torch.float64).detach()
    if r.shape != (K.shape[0],) or not torch.isfinite(r).all():
        raise ValueError("r must be a finite global-coordinate vector")
    a = _coordinates(active_indices, r.numel(), K.device)
    b = _coordinates(candidate_indices, r.numel(), K.device)
    joined = torch.cat((a, b))
    if joined.unique().numel() != joined.numel():
        raise ValueError("active and candidate coordinate sets must be disjoint")
    restricted = _block(K, joined)
    na = a.numel()
    kaa, kab = restricted[:na, :na], restricted[:na, na:]
    kba, kbb = restricted[na:, :na], restricted[na:, na:]
    inv_a, diag_a = _response(kaa, floor_relative)
    schur = kbb - kba @ inv_a @ kab
    schur = (schur + schur.mT) / 2
    inv_s, diag_s = _response(schur, floor_relative)
    residual_b = r[b] - kba @ inv_a @ r[a]
    delta_b = -inv_s @ residual_b
    delta_a = -inv_a @ (r[a] + kab @ delta_b)
    displacement = torch.zeros_like(r)
    displacement[a], displacement[b] = delta_a, delta_b
    active_only = torch.zeros_like(r)
    active_only[a] = -inv_a @ r[a]
    residual = _relative_residual(restricted, displacement[joined], -r[joined])
    diagnostics = {
        "active": diag_a, "schur": diag_s, "relative_residual": residual,
        "active_relative_residual": _relative_residual(kaa, active_only[a], -r[a]),
        "restricted_residual_norm": float(torch.linalg.vector_norm(r[a])),
        "restricted_residual_max": float(r[a].abs().max()) if na else 0.,
        "active_newton_decrement_squared": float(r[a] @ inv_a @ r[a]),
        "schur_subtraction_norm": float(torch.linalg.matrix_norm(kba @ inv_a @ kab)),
    }
    status = "ok"
    if diag_a["indefinite"] or diag_s["indefinite"]:
        status = "invalid_curvature"
    elif residual > 1e-5:
        status = "solve_unresolved"
    return {"displacement": displacement, "active_only_displacement": active_only,
            "schur": schur, "Schur": schur, "conditional_residual": residual_b,
            "solve_diagnostics": diagnostics, "status": status}


def direct_block_preview(
    r: Tensor, K: Tensor, active_indices, candidate_indices, *, floor_relative: float = 1e-8,
) -> Tensor:
    """Independent direct joint solve for checking the Schur preview."""
    K = _square(K, "K")
    r = torch.as_tensor(r, device=K.device, dtype=torch.float64).detach()
    if r.shape != (K.shape[0],) or not torch.isfinite(r).all():
        raise ValueError("r must be a finite global-coordinate vector")
    a = _coordinates(active_indices, r.numel(), K.device)
    b = _coordinates(candidate_indices, r.numel(), K.device)
    joined = torch.cat((a, b))
    if joined.unique().numel() != joined.numel():
        raise ValueError("active and candidate coordinate sets must be disjoint")
    response, _ = _response(_block(K, joined), floor_relative)
    result = torch.zeros_like(r)
    result[joined] = -response @ r[joined]
    return result


def optimism(
    H: Tensor, P: Tensor, V: Tensor, indices=None, *, floor_relative: float = 1e-8,
) -> dict:
    """O_C = tr(K_CC^-1 V_CC), d_eff = tr(K_CC^-1 H_CC).

    These traces use the supported *undamped* solve. Numerical floors are
    not likelihood information or a new statistical regularizer.
    """
    H = _square(H, "H")
    P = _square(P, "P", H.shape[0], H.device)
    V = _square(V, "V", H.shape[0], H.device)
    selected = torch.arange(H.shape[0], device=H.device) if indices is None else _coordinates(indices, H.shape[0], H.device)
    h, p, v = (_block(matrix, selected) for matrix in (H, P, V))
    response, diagnostics = _response(h + p, floor_relative, damp=False)
    diagnostics["undamped_supported_trace"] = True
    return {"optimism": float(torch.sum(response * v.mT)),
            "d_eff": float(torch.sum(response * h.mT)),
            "solve_diagnostics": diagnostics}


def likelihood_gain(g: Tensor, H: Tensor, displacement: Tensor) -> float:
    """Data-term gain, not retained-objective decrease; g = -score."""
    return float(-g @ displacement - .5 * displacement @ H @ displacement)


def evaluate_gate(
    g: Tensor, r: Tensor, H: Tensor, P: Tensor, V: Tensor, active_indices,
    candidate_indices, *, c: float = 1., floor_relative: float = 1e-8,
    alpha: float = 1., restricted_tolerance: float = 1e-5,
) -> dict:
    """Forecast a conditional expansion, subtracting active-only data gain.

    Unresolved active fits remain closed but retain the corrected diagnostic
    forecast. ``alpha`` shortens both preview displacements, not the optimism
    of a subsequently refitted coordinate set. Optimizer retry and backtracking
    use real data and are the caller's responsibility, outside this math API.
    """
    if not math.isfinite(c) or c < 0:
        raise ValueError("c must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    if not math.isfinite(restricted_tolerance) or restricted_tolerance < 0:
        raise ValueError("restricted_tolerance must be finite and nonnegative")
    H = _square(H, "H")
    P = _square(P, "P", H.shape[0], H.device)
    V = _square(V, "V", H.shape[0], H.device)
    g = torch.as_tensor(g, device=H.device, dtype=torch.float64).detach()
    if g.shape != (H.shape[0],) or not torch.isfinite(g).all():
        raise ValueError("g must be a finite global-coordinate loss gradient")
    a = _coordinates(active_indices, g.numel(), H.device)
    b = _coordinates(candidate_indices, g.numel(), H.device)
    preview = block_preview(r, H + P, a, b, floor_relative=floor_relative)
    displacement = alpha * preview["displacement"]
    active_only = alpha * preview["active_only_displacement"]
    expanded_gain = likelihood_gain(g, H, displacement)
    active_gain = likelihood_gain(g, H, active_only)
    gain = expanded_gain - active_gain
    active = optimism(H, P, V, a, floor_relative=floor_relative)
    expanded = optimism(H, P, V, torch.cat((a, b)), floor_relative=floor_relative)
    delta_optimism = expanded["optimism"] - active["optimism"]
    tolerance = 1e-10 * max(abs(active["optimism"]), abs(expanded["optimism"]), 1.)
    status = preview["status"]
    if expanded["solve_diagnostics"]["indefinite"]:
        status = "invalid_curvature"
    elif delta_optimism < -tolerance:
        status = "negative_delta_optimism"
    elif status == "ok" and preview["solve_diagnostics"]["restricted_residual_max"] > restricted_tolerance:
        status = "restricted_fit_unresolved"
    elif status == "ok" and not b.numel():
        status = "no_candidate"
    elif status == "ok" and expanded["d_eff"] <= 0:
        status = "no_information"
    # Tiny negative trace roundoff is not a free reward for adding coordinates.
    threshold = c * max(delta_optimism, 0.)
    opens = status == "ok" and gain > threshold
    if status == "ok":
        status = "open" if opens else "closed"
    diagnostics = dict(preview["solve_diagnostics"])
    diagnostics["optimism_active"] = active["solve_diagnostics"]
    diagnostics["optimism_expanded"] = expanded["solve_diagnostics"]
    return {"displacement": displacement, "active_only_displacement": active_only,
            "schur": preview["schur"], "Schur": preview["schur"],
            "conditional_residual": preview["conditional_residual"],
            "gain": gain, "gain_expanded": expanded_gain, "gain_active_only": active_gain,
            "optimism_active": active["optimism"], "optimism_expanded": expanded["optimism"],
            "delta_optimism": delta_optimism, "d_eff_active": active["d_eff"],
            "d_eff_expanded": expanded["d_eff"], "risk_reduction": gain - threshold,
            "threshold": threshold, "alpha": alpha, "open": opens, "status": status,
            "solve_diagnostics": diagnostics}
