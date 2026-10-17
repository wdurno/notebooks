"""Research-only, restartable Plan 14 geometry and local-depth diagnostics.

Scores estimate the likelihood Fisher; observed-label loss gradients are
``g = -s``. Historical archives are replayed at their saved consolidation
states, never replaced by a Fisher at the audit's current parameters. Dense
realizations of structured matrices here make no scalability claim.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from mnist_experiment.rotated_mnist.artifacts import RotatedArtifactError, _read_json
from mnist_experiment.rotated_mnist.config import RotationConfig
from mnist_experiment.rotated_mnist.transform import rotate_mnist_batch
from src.seeding import derive_component_seed

from .artifacts import file_hash
from .config import canonical_hash
from .evidence import effective_sample_size, retention_precision
from .fisher import model_fisher, represent, representation_diagnostics
from .gate import evaluate_gate
from .model import active_indices, build_model, layer_indices
from .optimizer import fit_restricted


AUDIT_SCHEMA_VERSION = 1
REPRESENTATIONS = (("diagonal", "diag", 0), ("layer_block", "layer_block", 0),
                   ("lr4_diag", "lr", 4), ("lr8_diag", "lr", 8),
                   ("lr16_diag", "lr", 16), ("dense_same_samples", "dense", 0))
REQUIRED = ("audit_metrics.json", "summary.json", "diagnostics.pt")


def _square(value: Tensor) -> Tensor:
    value = torch.as_tensor(value).detach().to(dtype=torch.float64)
    if (value.ndim != 2 or not value.shape[0] or value.shape[0] != value.shape[1]
            or not torch.isfinite(value).all()):
        raise ValueError("expected a finite nonempty square matrix")
    return (value + value.mT) / 2


def symmetric_relative_difference(left: Tensor, right: Tensor) -> float:
    """Disagreement divided by the mean norm; zero/zero is exactly zero."""
    left, right = torch.as_tensor(left).double(), torch.as_tensor(right).double()
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("comparison requires finite tensors with identical shapes")
    denominator = (torch.linalg.vector_norm(left) + torch.linalg.vector_norm(right)) / 2
    return float(torch.linalg.vector_norm(left - right) / denominator) if denominator > 0 else 0.


def _relative(left: Tensor, right: Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(right)
    error = torch.linalg.vector_norm(left - right)
    return float(error / denominator) if denominator > 0 else (0. if error == 0 else None)


def matrix_metrics(estimate: Tensor, reference: Tensor, *, subspace_rank: int = 8) -> dict:
    """Matrix errors relative to a *finite* reference, without PSD repair."""
    estimate, reference = _square(estimate), _square(reference)
    if estimate.shape != reference.shape or subspace_rank < 1:
        raise ValueError("incompatible matrix shapes or subspace rank")
    values, vectors = torch.linalg.eigh(estimate)
    ref_values, ref_vectors = torch.linalg.eigh(reference)
    tolerance = estimate.shape[0] * torch.finfo(torch.float64).eps
    supported = values > tolerance * float(values.abs().max())
    ref_supported = ref_values > tolerance * float(ref_values.abs().max())
    rank = min(subspace_rank, int(supported.sum()), int(ref_supported.sum()))
    overlap = float((vectors[:, -rank:].mT @ ref_vectors[:, -rank:]).square().sum() / rank) if rank else None
    positive = values.clamp_min(0)
    return {
        "frobenius_error": float(torch.linalg.matrix_norm(estimate - reference)),
        "relative_frobenius_error": _relative(estimate, reference),
        "diagonal_error": float(torch.linalg.vector_norm(estimate.diagonal() - reference.diagonal())),
        "relative_diagonal_error": _relative(estimate.diagonal(), reference.diagonal()),
        "trace": float(estimate.trace()), "reference_trace": float(reference.trace()),
        "trace_error": float(estimate.trace() - reference.trace()),
        "eigenvalues": values.cpu().tolist(), "reference_eigenvalues": ref_values.cpu().tolist(),
        "resolved_rank": int(supported.sum()), "reference_resolved_rank": int(ref_supported.sum()),
        "minimum_eigenvalue": float(values.min()),
        "negative_spectral_mass": float(-values.clamp_max(0).sum()),
        "condition_estimate": float(values[supported].max() / values[supported].min()) if supported.any() else None,
        "participation_rank": float(positive.sum().square() / positive.square().sum()) if positive.any() else 0.,
        "leading_subspace_overlap": overlap, "subspace_rank": rank,
        "reference_is_finite": True,
    }


def reference_disagreement(left: Tensor, right: Tensor, left_gate: dict | None = None,
                           right_gate: dict | None = None, *, residual_norm: float = 0.,
                           residual_tolerance: float = 1e-12) -> dict:
    """Plan 14's .05 matrix/.10 conditional-solve/reference-gate expansion rule."""
    if not math.isfinite(residual_norm) or residual_norm < 0 or residual_tolerance < 0:
        raise ValueError("residual scales must be finite and nonnegative")
    matrix_error = symmetric_relative_difference(left, right)
    solve_error, gate_flip = None, False
    gates_resolved = False
    if (left_gate is None) != (right_gate is None):
        raise ValueError("supply both reference gate results or neither")
    if left_gate is not None:
        # conditional_solution is delta_B, not the full active readjustment.
        solve_error = symmetric_relative_difference(left_gate["conditional_solution"],
                                                     right_gate["conditional_solution"])
        gates_resolved = all(g["status"] in {"open", "closed"} for g in (left_gate, right_gate))
        gate_flip = gates_resolved and bool(left_gate["open"]) != bool(right_gate["open"])
    reasons = []
    if matrix_error > .05:
        reasons.append("matrix_disagreement")
    if solve_error is not None and residual_norm > residual_tolerance and solve_error > .10:
        reasons.append("conditional_solve_disagreement")
    if gate_flip:
        reasons.append("reference_gate_disagreement")
    return {"matrix_disagreement": matrix_error, "conditional_solve_disagreement": solve_error,
            "gate_disagreement": gate_flip, "gate_comparison_resolved": gates_resolved,
            "residual_nonnegligible": residual_norm > residual_tolerance,
            "expand": bool(reasons), "reasons": reasons,
            "reference_is_finite": True, "matrix_threshold": .05, "solve_threshold": .10}


def weighted_history(batch_positions, batch_sizes, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return environment positions and their actual per-arrival weight mass."""
    positions = torch.as_tensor(batch_positions, dtype=torch.float64).cpu()
    sizes = torch.as_tensor(batch_sizes)
    a = torch.as_tensor(weights, dtype=torch.float64).cpu()
    if (positions.ndim != 1 or not positions.numel() or not torch.isfinite(positions).all()
            or sizes.shape != positions.shape or sizes.dtype not in (torch.int32, torch.int64)
            or (sizes <= 0).any() or a.ndim != 1 or int(sizes.sum()) != a.numel()
            or not torch.isfinite(a).all() or (a < 0).any() or a.sum() <= 0):
        raise ValueError("invalid weighted environment history")
    a = a / a.sum()
    mass = torch.stack([part.sum() for part in a.split(sizes.tolist())])
    return positions, mass


def sample_reference_inputs(inputs: Tensor, targets: Tensor, *, family: str,
                            batch_positions, batch_sizes, weights: Tensor,
                            sample_size: int, seed: int,
                            instantaneous: float | None = None,
                            rotation: RotationConfig | None = None) -> tuple[Tensor, dict]:
    """Independent-pool draws matching weighted past environments, not labels.

    Mixtures use the declared weighted prevalence, preserving the pool's
    empirical non-nine conditional distribution. Rotations sample past angles
    by arrival weight mass. Model-label integration happens separately.
    """
    positions, mass = weighted_history(batch_positions, batch_sizes, weights)
    if instantaneous is not None:
        if not math.isfinite(instantaneous):
            raise ValueError("instantaneous position must be finite")
        positions, mass = torch.tensor([instantaneous], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    if (sample_size < 1 or isinstance(sample_size, bool) or inputs.shape[0] == 0
            or targets.shape != (inputs.shape[0],) or targets.dtype != torch.long):
        raise ValueError("invalid reference pool or sample size")
    inputs, targets = inputs.detach().cpu(), targets.detach().cpu()
    seeds = {name: derive_component_seed(seed, name) for name in ("environment", "pool")}
    rng = {name: torch.Generator().manual_seed(value) for name, value in seeds.items()}
    if family == "mixture":
        if ((positions < 0) | (positions > 1)).any():
            raise ValueError("mixture positions must be prevalences")
        prevalence = float(positions @ mass)
        nine = torch.rand(sample_size, generator=rng["environment"]) < prevalence
        ids = torch.empty(sample_size, dtype=torch.long)
        for flag in (False, True):
            selected = torch.nonzero(nine == flag).flatten()
            pool = torch.nonzero((targets == 9) == flag).flatten()
            if ((1 - prevalence) if not flag else prevalence) > 0 and not pool.numel():
                raise ValueError("independent reference pool lacks a required class stratum")
            if selected.numel():
                ids[selected] = pool[torch.randint(pool.numel(), (selected.numel(),), generator=rng["pool"])]
        sampled = inputs[ids].clone()
        environment_ids = nine.long()
    elif family == "rotation":
        prevalence = None
        rotation = rotation or RotationConfig((0., 30., 0., 30.), 1, "bilinear", False, 0.)
        rotation.validate()
        environment_ids = torch.multinomial(mass, sample_size, replacement=True, generator=rng["environment"])
        ids = torch.randint(len(inputs), (sample_size,), generator=rng["pool"])
        sampled = inputs[ids].clone()
        for component in environment_ids.unique():
            selected = environment_ids == component
            angle = float(positions[component])
            if angle:
                sampled[selected] = rotate_mnist_batch(sampled[selected], angle, rotation)
    else:
        raise ValueError("family must be mixture or rotation")
    return sampled, {"seed": seed, "seeds": seeds, "pool_indices": ids,
                     "environment_indices": environment_ids, "positions": positions,
                     "arrival_weight_mass": mass, "weighted_prevalence": prevalence,
                     "instantaneous": instantaneous, "unique_pool_inputs": ids.unique().numel(),
                     "sample_size": sample_size, "sampling_with_replacement": True}


def bias_variance(matrices: Tensor, reference: Tensor) -> dict:
    """Finite-replication MSE = squared bias + variance (denominator R)."""
    matrices, reference = torch.as_tensor(matrices).double(), _square(reference)
    if matrices.ndim != 3 or matrices.shape[0] < 2 or matrices.shape[1:] != reference.shape:
        raise ValueError("bias/variance needs at least two same-state matrix draws")
    if not torch.isfinite(matrices).all():
        raise ValueError("matrix draws must be finite")
    mean = matrices.mean(0)
    bias = float((mean - reference).square().sum())
    variance = float((matrices - mean).square().sum((1, 2)).mean())
    return {"draws": len(matrices), "squared_bias_to_finite_reference": bias,
            "variance_population_denominator": variance,
            "variance_unbiased": variance * len(matrices) / (len(matrices) - 1),
            "mean_squared_error_to_finite_reference": float((matrices - reference).square().sum((1, 2)).mean()),
            "reference_uncertainty_not_subtracted": True}


def orientation_control(matrix: Tensor, seed: int) -> Tensor:
    """Coordinate-sign conjugation preserves PSD, spectrum and diagonal."""
    matrix = _square(matrix)
    signs = (2 * torch.randint(2, (len(matrix),), generator=torch.Generator().manual_seed(seed)) - 1).to(matrix)
    return signs[:, None] * matrix * signs[None, :]


def replay_archive(initial_fisher: Tensor, history: list[dict], *, kind: str = "dense",
                   rank: int = 8, contributions: list[Tensor] | None = None) -> Tensor:
    """Replay EMA and compression in order, always sharing the initial asset."""
    archive = represent(initial_fisher, kind=kind, rank=rank)
    if contributions is not None and len(contributions) != len(history):
        raise ValueError("one contribution is required per historical consolidation")
    for i, entry in enumerate(history):
        eta = float(entry["eta"])
        if not math.isfinite(eta) or not 0 < eta <= 1:
            raise ValueError("consolidation eta must lie in (0, 1]")
        new = _square(entry["dense_new_fisher"] if contributions is None else contributions[i])
        if new.shape != archive.shape:
            raise ValueError("historical contribution layout mismatch")
        archive = represent((1 - eta) * archive + eta * new, kind=kind, rank=rank)
    return archive


def identical_operator_solve(diagonal: Tensor, factors: Tensor, rhs: Tensor) -> dict:
    """Dense/Woodbury numerical check of exactly D+LL', not two estimators."""
    diagonal, factors, rhs = (torch.as_tensor(x).double() for x in (diagonal, factors, rhs))
    if (diagonal.ndim != 1 or factors.ndim != 2 or factors.shape[0] != len(diagonal)
            or rhs.shape != diagonal.shape or not all(torch.isfinite(x).all() for x in (diagonal, factors, rhs))
            or (diagonal <= 0).any()):
        raise ValueError("Woodbury requires finite inputs and a strictly positive diagonal")
    matrix = torch.diag(diagonal) + factors @ factors.mT
    if matrix.is_cuda:
        torch.cuda.synchronize(matrix.device)
    started = time.perf_counter()
    dense = torch.linalg.solve(matrix, rhs)
    if matrix.is_cuda:
        torch.cuda.synchronize(matrix.device)
    dense_seconds = time.perf_counter() - started
    started = time.perf_counter()
    inverse_factors = factors / diagonal[:, None]
    middle = torch.eye(factors.shape[1], device=matrix.device, dtype=torch.float64) + factors.mT @ inverse_factors
    woodbury = rhs / diagonal - inverse_factors @ torch.linalg.solve(middle, factors.mT @ (rhs / diagonal))
    if matrix.is_cuda:
        torch.cuda.synchronize(matrix.device)
    return {"solution_relative_difference": symmetric_relative_difference(dense, woodbury),
            "dense_relative_residual": _relative(matrix @ dense, rhs),
            "woodbury_relative_residual": _relative(matrix @ woodbury, rhs),
            "dense_seconds": dense_seconds, "woodbury_seconds": time.perf_counter() - started,
            "comparison": "identical_sampled_operator_numerical_only",
            "dense_solution": dense, "woodbury_solution": woodbury}


def _cpu(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    return value


def _json(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite audit metric")
    return value


class _Checkpoints:
    """Each completed operation/chunk is atomically saved and checksum sealed."""

    def __init__(self, session, limit):
        self.session, self.limit = session, limit
        path = session.path / "checkpoint_index.json"
        self.index = _read_json(path) if path.exists() else {}

    def get(self, key: str, compute: Callable[[], Any]):
        if key in self.index:
            record = self.index[key]
            path = self.session.path / record["path"]
            if not path.is_file() or file_hash(path) != record["sha256"]:
                raise RotatedArtifactError(f"corrupt sealed audit checkpoint: {key}")
            return torch.load(path, map_location="cpu", weights_only=False)
        self.limit.check()
        value = _cpu(compute())
        name = f"chunks/{canonical_hash(key)}.pt"
        path = self.session.write_torch(name, value)
        self.index[key] = {"path": name, "sha256": file_hash(path)}
        self.session.write_json("checkpoint_index.json", self.index)
        self.limit.check()
        return value

    def required(self) -> tuple[str, ...]:
        return ("checkpoint_index.json", *(value["path"] for value in self.index.values()))


def select_snapshot(snapshots: list[dict], step: int) -> dict:
    matches = [snapshot for snapshot in snapshots if snapshot["step"] == step]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one snapshot at step {step}, found {len(matches)}")
    return matches[0]


def _loss(model, inputs, targets, weights):
    losses = nn.functional.cross_entropy(model(inputs), targets, reduction="none").double()
    return (weights.double() / weights.sum()) @ losses


def loss_gradient(model, layout, inputs, targets, weights) -> Tensor:
    """Observed weighted NLL gradient, explicitly the negative score."""
    gradients = torch.autograd.grad(_loss(model, inputs, targets, weights), tuple(model.parameters()))
    return torch.cat([part.reshape(-1) for part in gradients]).detach().double()


def observed_hvp(model, layout, inputs, targets, weights, direction: Tensor) -> Tensor:
    """One aggregate observed-Hessian product; no per-sample Hessians."""
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(_loss(model, inputs, targets, weights), parameters, create_graph=True)
    gradient = torch.cat([part.reshape(-1) for part in gradients])
    product = torch.autograd.grad(gradient @ direction.to(gradient), parameters)
    return torch.cat([part.reshape(-1) for part in product]).detach().double()


def _gate(g, r, h, p, ess, a, b, study, *, floor_scale=1., converged=True,
          residual_tolerance=None):
    result = evaluate_gate(g, r, h, p, h / ess, a, b, c=1.,
                           floor_relative=study.solve_floor_relative * floor_scale,
                           restricted_tolerance=study.tolerance_grad if residual_tolerance is None else residual_tolerance)
    result["conditional_solution"] = result["displacement"][b]
    if not converged or result["status"] == "restricted_fit_unresolved":
        result.update(status="restricted_fit_unresolved", open=False)
        for key in ("gain", "gain_expanded", "gain_active_only", "risk_reduction"):
            result[key] = None
    return result


def decision_metrics(estimate: dict, reference: dict, candidate_indices, *,
                     estimate_k: Tensor, reference_k: Tensor) -> dict:
    b = torch.as_tensor(candidate_indices, dtype=torch.long, device=estimate_k.device)
    kbb = estimate_k[b[:, None], b[None, :]]
    ref_kbb = reference_k[b[:, None], b[None, :]]
    resolved = all(g["status"] in {"open", "closed"} for g in (estimate, reference))
    gain_error = None if estimate["gain"] is None or reference["gain"] is None else estimate["gain"] - reference["gain"]
    return {"conditional_solve_relative_error": _relative(estimate["conditional_solution"], reference["conditional_solution"]),
            "conditional_solve_absolute_error": float(torch.linalg.vector_norm(estimate["conditional_solution"] - reference["conditional_solution"])),
            "schur_relative_error": _relative(estimate["schur"], reference["schur"]),
            "schur_subtraction_fraction": _relative(estimate["schur"], kbb),
            "reference_schur_subtraction_fraction": _relative(reference["schur"], ref_kbb),
            "gain_error": gain_error,
            "optimism_error": estimate["delta_optimism"] - reference["delta_optimism"],
            "gate_flip": bool(estimate["open"]) != bool(reference["open"]) if resolved else None,
            "decision_comparison_resolved": resolved,
            "risk_margin": estimate["risk_reduction"], "reference_risk_margin": reference["risk_reduction"],
            "status": estimate["status"], "reference_status": reference["status"]}


def _parent(store, path: Path, required: tuple[str, ...]) -> dict:
    unit = _read_json(path / "config.json")
    validated = store.completed(unit, required)
    if validated is None or validated.resolve() != path.resolve():
        raise RotatedArtifactError(f"audit requires a completed compatible parent: {path}")
    return unit


class _Audit:
    def __init__(self, store, index, case, assets, snapshot, driver_config, driver_summary, cache,
                 *, namespace="audit"):
        self.store, self.study, self.index, self.case, self.cache = store, store.study, index, case, cache
        self.namespace = namespace
        self.assets, self.snapshot = assets, snapshot
        self.family = case["environment"]["family"]
        self.rotation = None
        if self.family == "rotation":
            self.rotation = RotationConfig.from_mapping(assets["streams"][case["environment"]["name"]]["rotation"])
        self.device = torch.device(self.study.device)
        self.dtype = getattr(torch, self.study.dtype)
        self.width = driver_summary["width"]
        self.model, self.layout = build_model(self.seed("initialization"), self.width,
                                               device=self.device, dtype=self.dtype)
        self.layout.assert_metadata(assets["parameter_layout"])
        self.load(snapshot["model"])
        theta = self.layout.flatten_module(self.model, detach=True).cpu()
        if not torch.equal(theta, snapshot["theta"].to(theta)):
            raise ValueError("saved snapshot model and theta disagree")
        detail = driver_config["detail"]
        self.fit_policy = detail["retention"]
        self.retention = {name: self.fit_policy[name] for name in ("omega", "ridge_ratio")}
        self.policy = detail["policy"]
        self.rank = self.policy["rank"]
        self.groups = layer_indices(self.layout)
        self.precision = _square(snapshot["precision"]).to(self.device)
        self.anchor = snapshot["anchor"].to(device=self.device, dtype=self.dtype)
        expected, _ = retention_precision(snapshot["archive"], **self.retention)
        if not torch.allclose(self.precision.cpu(), expected.cpu(), rtol=1e-7, atol=1e-10):
            raise ValueError("snapshot precision disagrees with resolved retention objective")
        self.ess = effective_sample_size(snapshot["weights"])

    def load(self, state):
        self.model.load_state_dict(state, strict=True)

    def seed(self, component):
        return self.study.seed(f"{self.namespace}:{component}", self.index)

    def data(self, snapshot=None):
        snapshot = self.snapshot if snapshot is None else snapshot
        return (snapshot["inputs"].to(device=self.device, dtype=self.dtype),
                snapshot["targets"].to(device=self.device),
                snapshot["weights"].to(device=self.device, dtype=torch.float64))

    def vector(self):
        return self.layout.flatten_module(self.model, detach=True)

    def fit(self, key, state, depth, precision, *, snapshot=None):
        def compute():
            self.load(state)
            a, _ = active_indices(self.layout, depth)
            result = fit_restricted(self.model, self.layout, *self.data(snapshot), self.anchor,
                                    precision.to(self.device), a.to(self.device),
                                    max_iter=self.fit_policy.get("max_iter", self.study.max_iter),
                                    max_eval=self.fit_policy.get("max_eval", self.study.max_eval),
                                    tolerance_grad=self.study.tolerance_grad,
                                    residual_tolerance=self.fit_policy.get("residual_tolerance", self.study.tolerance_grad),
                                    tolerance_change=self.study.tolerance_change)
            return {"model": self.model.state_dict(), "fit": result,
                    "theta": self.vector(), "depth": depth}
        try:
            return self.cache.get(key, compute)
        finally:
            self.load(self.snapshot["model"])

    def estimate(self, key, state, count, *, snapshot=None, independent=False, instantaneous=None,
                 seed_key=None):
        """Resume independent chunks, including low-budget draws, without redrawing."""
        snapshot = self.snapshot if snapshot is None else snapshot
        self.load(state)
        total = torch.zeros((self.layout.total_numel,) * 2, dtype=torch.float64)
        chunks = []
        for start in range(0, count, self.study.fisher_chunk_size):
            stop = min(count, start + self.study.fisher_chunk_size)
            seed = self.seed(f"{key if seed_key is None else seed_key}:scores:{start}")

            def compute():
                sampling = None
                if independent:
                    x, sampling = sample_reference_inputs(
                        self.assets["reference_inputs"], self.assets["reference_targets"],
                        family=self.family, batch_positions=snapshot["batch_positions"],
                        batch_sizes=snapshot["batch_sizes"], weights=snapshot["weights"],
                        sample_size=stop - start, seed=derive_component_seed(seed, "reference_inputs"),
                        instantaneous=instantaneous, rotation=self.rotation)
                    x, a = x.to(device=self.device, dtype=self.dtype), None
                else:
                    x, _, a = self.data(snapshot)
                # Exact labels on independently drawn reference inputs remove
                # label MC variance, but not finite-input reference uncertainty.
                matrix, diagnostics = model_fisher(
                    self.model, self.layout, x, a, sample_size=stop - start,
                    seed=derive_component_seed(seed, "model_labels"), exact_labels=independent,
                    chunk_size=self.study.fisher_chunk_size)
                return {"matrix": matrix, "diagnostics": diagnostics,
                        "reference_sampling": sampling, "start": start, "stop": stop, "seed": seed}

            part = self.cache.get(f"{key}/chunk/{start}-{stop}", compute)
            total += (stop - start) * part["matrix"] / count
            chunks.append({"start": start, "stop": stop, "seed": seed})
        return {"matrix": (total + total.mT) / 2, "input_count": count,
                "chunks": chunks, "independent_pool": independent,
                "model_label_mode": "exact" if independent else "sampled"}

    def gate(self, g, r, h, p, a, b, *, converged, floor_scale=1.):
        return _gate(g, r, h.to(self.device), p.to(self.device), self.ess, a, b,
                     self.study, converged=converged, floor_scale=floor_scale,
                     residual_tolerance=self.fit_policy.get("residual_tolerance", self.study.tolerance_grad))

    def reference(self, key, state, g, r, precision, a, b, *, converged, instantaneous=None,
                  seed_key=None):
        initial, maximum = self.study.reference_samples, self.study.reference_max_samples
        if initial % 2 or maximum % 2 or maximum < initial:
            raise ValueError("reference budgets must be even and ordered")
        seed_key = key if seed_key is None else seed_key
        halves = [self.estimate(f"{key}/half{half}/initial", state, initial // 2,
                               independent=True, instantaneous=instantaneous,
                               seed_key=f"{seed_key}/half{half}/initial")["matrix"] for half in range(2)]

        def disagreement():
            gates = [self.gate(g, r, h, precision, a, b, converged=converged) for h in halves]
            norm = max(float(torch.linalg.vector_norm(v["conditional_residual"])) for v in gates)
            return reference_disagreement(*halves, *gates, residual_norm=norm)

        checks = [{"input_count": initial, **disagreement()}]
        count = initial
        if checks[-1]["expand"] and maximum > initial:
            for half in range(2):
                extra = self.estimate(f"{key}/half{half}/extension", state, (maximum - initial) // 2,
                                      independent=True, instantaneous=instantaneous,
                                      seed_key=f"{seed_key}/half{half}/extension")["matrix"]
                halves[half] = (initial * halves[half] + (maximum - initial) * extra) / maximum
            count = maximum
            checks.append({"input_count": count, **disagreement()})
        return {"matrix": (halves[0] + halves[1]) / 2, "halves": halves,
                "input_count": count, "checks": checks,
                "unresolved": checks[-1]["expand"], "gate_comparison_resolved": checks[-1]["gate_comparison_resolved"],
                "uncertainty_scope": "independent_input_halves_exact_model_labels_finite_pool",
                "not_population_truth": True, "instantaneous": instantaneous}

    def losses(self, state, *, position, selection=False, scoring=False):
        self.load(state)
        x, y = self.assets["evaluation_inputs"], self.assets["evaluation_targets"]
        ids = torch.arange(len(y))
        if selection or scoring:
            # Deterministic within-class splitting keeps tiny rare-class panels
            # from accidentally putting every nine in just one half.
            pieces = [torch.nonzero(y == label).flatten()[int(scoring)::2] for label in y.unique()]
            ids = torch.cat(pieces).sort().values
            x, y = x[ids], y[ids]
        if not len(y):
            raise ValueError("empty independent evaluation split")
        if self.family == "rotation" and position:
            x = rotate_mnist_batch(x.cpu(), float(position), self.rotation)
        result = []
        with torch.no_grad():
            for start in range(0, len(y), 256):
                logits = self.model(x[start:start + 256].to(device=self.device, dtype=self.dtype))
                result.append(nn.functional.cross_entropy(logits.double(), y[start:start + 256].to(self.device), reduction="none").cpu())
        return {"losses": torch.cat(result), "targets": y.cpu(), "ids": ids}

    def evaluation(self, key, state, **split):
        def compute():
            return {"current": self.losses(state, position=self.snapshot["position"], **split),
                    "retention": self.losses(state, position=0., **split)}
        try:
            return self.cache.get(key, compute)
        finally:
            self.load(self.snapshot["model"])


def paired_panel_gain(before: dict, after: dict, *, family: str, position: float) -> dict:
    """Paired finite-panel NLL gain and conditional normal-approximation SE."""
    if not torch.equal(before["ids"], after["ids"]) or not torch.equal(before["targets"], after["targets"]):
        raise ValueError("paired panel identities differ")
    if any(panel["losses"].shape != panel["targets"].shape or not torch.isfinite(panel["losses"]).all()
           for panel in (before, after)):
        raise ValueError("paired losses must be aligned and finite")
    differences = before["losses"].double() - after["losses"].double()
    groups = [(1., torch.ones(len(differences), dtype=torch.bool))]
    if family == "mixture":
        if not 0 <= position <= 1:
            raise ValueError("invalid evaluation prevalence")
        groups = [(1 - position, before["targets"] != 9), (position, before["targets"] == 9)]
    elif family != "rotation":
        raise ValueError("unknown evaluation family")
    gain, variance, variance_available = 0., 0., True
    for mass, mask in groups:
        if mass == 0:
            continue
        n = int(mask.sum())
        if n == 0:
            raise ValueError("evaluation panel lacks a required stratum")
        gain += mass * float(differences[mask].mean())
        if n < 2:
            variance_available = False
        else:
            variance += mass**2 * float(differences[mask].var(unbiased=True)) / n
    se = math.sqrt(variance) if variance_available else None
    return {"gain": gain, "standard_error": se,
            "interval95": None if se is None else [gain - 1.96 * se, gain + 1.96 * se],
            "uncertainty": "paired_finite_panel_normal_approximation_not_replica_interval"}


def _shadow_gains(audit, before, after):
    return {name: paired_panel_gain(before[name], after[name], family=audit.family,
                                   position=audit.snapshot["position"] if name == "current" else 0.)
            for name in ("current", "retention")}


def _preview(audit, fitted, gate, h, g):
    """Compare Fisher and observed-Hessian forecasts to actual preview losses."""
    if gate["gain"] is None:
        return {"status": "restricted_fit_unresolved", "actual_weighted_nll_gain": None,
                "observed_hessian_gain": None, "fisher_gain": None}
    audit.load(fitted["model"])
    theta = audit.vector()
    x, y, a = audit.data()
    delta, active = (gate[key].to(audit.device) for key in ("displacement", "active_only_displacement"))
    try:
        hv = observed_hvp(audit.model, audit.layout, x, y, a, delta)
        hv_active = observed_hvp(audit.model, audit.layout, x, y, a, active)
        observed = float(-g @ delta - .5 * delta @ hv + g @ active + .5 * active @ hv_active)
        with torch.no_grad():
            audit.layout.copy_vector_to_module(audit.model, theta + active.to(theta))
            active_loss = float(_loss(audit.model, x, y, a))
            audit.layout.copy_vector_to_module(audit.model, theta + delta.to(theta))
            expanded_loss = float(_loss(audit.model, x, y, a))
        return {"status": gate["status"], "actual_weighted_nll_gain": active_loss - expanded_loss,
                "observed_hessian_gain": observed, "fisher_gain": gate["gain"],
                "fisher_hvp_disagreement": symmetric_relative_difference(h @ delta, hv),
                "observed_hvp": hv, "displacement_norm": float(torch.linalg.vector_norm(delta))}
    finally:
        audit.load(audit.snapshot["model"])


def _history(audit):
    """Saved-state history replay, with two independent finite contribution banks."""
    history, initial = audit.snapshot["history"], audit.assets["initial_fisher"]
    same = replay_archive(initial, history)
    compressed = {kind: replay_archive(initial, history, kind=kind, rank=audit.rank) for kind in ("diag", "lr")}
    actual = replay_archive(initial, history, kind=audit.policy["ewc"], rank=audit.rank)
    if not torch.allclose(actual, audit.snapshot["archive"].double(), rtol=1e-6, atol=1e-10):
        raise ValueError("saved archive does not match its actual consolidation history")
    result = {"same_bank_dense": same, "compressed": compressed,
              "actual_archive": audit.snapshot["archive"], "initial_fisher_shared": True,
              "consolidation_count": len(history), "reference": None, "contributions": []}
    if not audit.case["historical"]:
        return result
    half_contributions = [[], []]
    for i, entry in enumerate(history):
        if "archive_before" not in entry or "model" not in entry:
            raise ValueError("historical replay needs each consolidation model and archive_before")
        before = replay_archive(initial, history[:i], kind=audit.policy["ewc"], rank=audit.rank)
        if not torch.allclose(before, entry["archive_before"].double(), rtol=1e-6, atol=1e-10):
            raise ValueError("historical archive_before does not match its saved prefix")
        initial_count, max_count = audit.study.reference_samples, audit.study.reference_max_samples
        halves = [audit.estimate(f"history/{i}/half{half}/initial", entry["model"], initial_count // 2,
                                  snapshot=entry)["matrix"] for half in range(2)]
        checks = [{"input_count": initial_count, **reference_disagreement(*halves)}]
        count = initial_count
        if checks[-1]["expand"] and max_count > initial_count:
            for half in range(2):
                extra = audit.estimate(f"history/{i}/half{half}/extension", entry["model"],
                                       (max_count - initial_count) // 2, snapshot=entry)["matrix"]
                halves[half] = (initial_count * halves[half] + (max_count - initial_count) * extra) / max_count
            count = max_count
            checks.append({"input_count": count, **reference_disagreement(*halves)})
        for half in range(2):
            half_contributions[half].append(halves[half])
        result["contributions"].append({"index": i, "eta": entry["eta"], "input_count": count,
                                         "checks": checks, "unresolved": checks[-1]["expand"],
                                         "reference_measure": "saved_consolidation_weighted_buffer"})
    halves = [replay_archive(initial, history, contributions=contributions) for contributions in half_contributions]
    result["reference"] = {"matrix": (halves[0] + halves[1]) / 2, "halves": halves,
                           "checks": [reference_disagreement(*halves)],
                           "unresolved": any(v["unresolved"] for v in result["contributions"]),
                           "initial_fisher_uncertainty_not_reestimated": True,
                           "not_population_truth": True}
    result["half_contributions"] = half_contributions
    return result


def _frontier_state(audit, key, depth, precision):
    fitted = audit.fit(f"{key}/restricted", audit.snapshot["model"], depth, precision)
    expanded = audit.fit(f"{key}/expanded", fitted["model"], depth + 1, precision)
    before = audit.evaluation(f"{key}/restricted_evaluation", fitted["model"])
    after = audit.evaluation(f"{key}/expanded_evaluation", expanded["model"])
    audit.load(fitted["model"])
    g = loss_gradient(audit.model, audit.layout, *audit.data())
    theta = audit.vector().double()
    r = g + precision.to(audit.device) @ (theta - audit.anchor.double())
    a, meta = active_indices(audit.layout, depth)
    a = a.to(audit.device)
    b = torch.tensor(meta["candidate_indices"], device=audit.device, dtype=torch.long)
    return fitted, expanded, g, r, a, b, _shadow_gains(audit, before, after)


def _numerical_solve(source, precision, rhs, rank, indices):
    values, vectors = torch.linalg.eigh(_square(source))
    keep = min(rank, len(values))
    factors = vectors[:, -keep:] * values[-keep:].clamp_min(0).sqrt() if keep else vectors[:, :0]
    diagonal = (source.diagonal() - factors.square().sum(1)).clamp_min(0)
    pv, pq = torch.linalg.eigh(_square(precision))
    factors = torch.cat((factors, pq * pv.clamp_min(0).sqrt()), dim=1)
    # This floor belongs only to this paired numerical check. It never feeds
    # the gate, optimism, effective dimension, or the retained objective.
    floor = max(float(diagonal.mean()) * 1e-8, torch.finfo(torch.float64).eps)
    diagonal = diagonal.clamp_min(floor)
    result = identical_operator_solve(diagonal[indices], factors[indices], rhs[indices])
    result.update(numerical_diagonal_floor=floor,
                  retention_factorization="dense_square_root_diagnostic_not_scalable",
                  restricted_before_solve=True)
    return result


def _comparison(audit, key, depth, repeat, name, kind, rank, source, reference,
                precision, fitted, expanded, g, r, a, b, gains, history, *, orientation=False):
    h = represent(source, kind=kind, rank=rank, layer_indices=audit.groups).to(audit.device)
    orientation_seed = None
    if orientation:
        orientation_seed = audit.seed(f"orientation:{depth}")
        h = orientation_control(h, orientation_seed)
    ref_h = reference["matrix"].to(audit.device)
    converged = fitted["fit"]["converged"]
    estimated_gate = audit.gate(g, r, h, precision, a, b, converged=converged)
    reference_gate = audit.gate(g, r, ref_h, precision, a, b, converged=converged)
    same_bank_gate = audit.gate(g, r, source.to(audit.device), precision, a, b, converged=converged)
    floor_gate = audit.gate(g, r, h, precision, a, b, converged=converged, floor_scale=10.)
    floor_flip = estimated_gate["open"] != floor_gate["open"]
    preview = _preview(audit, fitted, estimated_gate, h, g)
    realized = (expanded["theta"] - fitted["theta"]).double().to(audit.device)
    p_reference_source = history["same_bank_dense"] if history["reference"] is None else history["reference"]["matrix"]
    reference_p, _ = retention_precision(p_reference_source.to(audit.device), **audit.retention)
    delta = estimated_gate["displacement"]
    difference = precision - reference_p
    current_gain, retention_gain = gains["current"]["gain"], gains["retention"]["gain"]
    decision_resolved = converged and estimated_gate["status"] in {"open", "closed"} and not floor_flip
    shadow_resolved = expanded["fit"]["converged"] and converged
    row = {
        "metric_schema_version": AUDIT_SCHEMA_VERSION, "category": "same_state_representation",
        "step": audit.case["step"], "frontier": depth, "repeat": repeat,
        "representation": name, "ess": audit.ess, "rank": rank,
        "compression": matrix_metrics(h.cpu(), source),
        "finite_reference": matrix_metrics(h.cpu(), reference["matrix"]),
        "diagonal_preservation": representation_diagnostics(source, h.cpu(), rank=rank),
        "decision": decision_metrics(estimated_gate, reference_gate, b,
                                     estimate_k=h + precision, reference_k=ref_h + precision),
        "same_bank_decision": decision_metrics(estimated_gate, same_bank_gate, b,
                                               estimate_k=h + precision, reference_k=source.to(audit.device) + precision),
        "cross_layer_relative_error": _relative(h[a[:, None], b[None, :]], ref_h[a[:, None], b[None, :]]),
        "floor_10x_gate_flip": floor_flip,
        "floor_10x_solve_difference": symmetric_relative_difference(estimated_gate["conditional_solution"], floor_gate["conditional_solution"]),
        "gate_gain": estimated_gate["gain"], "delta_optimism": estimated_gate["delta_optimism"],
        "d_eff_active": estimated_gate["d_eff_active"], "d_eff_expanded": estimated_gate["d_eff_expanded"],
        "gate_open": bool(estimated_gate["open"]) if decision_resolved else None,
        "gate_status": "floor_sensitive_unresolved" if floor_flip else estimated_gate["status"],
        "one_frontier_selected_depth": depth + int(estimated_gate["open"]) if decision_resolved else None,
        "reference_unresolved": reference["unresolved"],
        "restricted_fit_converged": converged, "expanded_fit_converged": expanded["fit"]["converged"],
        "restricted_residual_max": fitted["fit"]["restricted_residual_max"],
        "preview": {k: v for k, v in preview.items() if k != "observed_hvp"},
        "realized_current_gain": current_gain, "realized_retention_gain": retention_gain,
        "realized_gain_uncertainty": gains,
        "shadow_result_status": "converged" if shadow_resolved else "optimization_unresolved",
        "gate_regret": (max(-current_gain, 0.) if estimated_gate["open"] else max(current_gain, 0.)) if decision_resolved and shadow_resolved else None,
        "regret_uncertainty": "gain_interval_reported; positive_part_is_a_noisy_local_benchmark",
        "ewc_proposed_directional_energy_error": float(delta @ difference @ delta),
        "ewc_realized_directional_energy_error": float(realized @ difference @ realized),
        "ewc_reference_kind": "high_sample_history" if history["reference"] is not None else "same_bank_history",
        "orientation_seed": orientation_seed, "solve_diagnostics": estimated_gate["solve_diagnostics"],
        "estimation_error_scope": "score_sampling_compression_and_finite_buffer_vs_independent_input_measure",
    }
    train_gain = fitted["fit"]["data_loss_after"] - expanded["fit"]["data_loss_after"]
    row["expanded_weighted_training_gain"] = train_gain
    row["training_minus_current_panel_gain"] = train_gain - current_gain
    row["optimism_calibration_scope"] = "drifting_buffer_vs_instantaneous_panel_not_pure_optimism"
    return {"row": row, "gate": estimated_gate, "reference_gate": reference_gate,
            "floor_gate": floor_gate, "observed_hvp": preview.get("observed_hvp")}


def _historical_uncertainty(audit, history, reference_h, fitted, g, r, a, b):
    """Inverse sensitivity of the historical reference at the actual residual.

    Holding r fixed makes this a geometry uncertainty check, not a claim to
    have refitted the alternative historical objectives.
    """
    if history["reference"] is None:
        return None
    gates = []
    for matrix in history["reference"]["halves"]:
        precision, _ = retention_precision(matrix.to(audit.device), **audit.retention)
        gates.append(audit.gate(g, r, reference_h.to(audit.device), precision, a, b,
                               converged=fitted["fit"]["converged"]))
    check = reference_disagreement(*history["reference"]["halves"], *gates,
                                   residual_norm=max(float(torch.linalg.vector_norm(gate["conditional_residual"])) for gate in gates))
    check["scope"] = "fixed_actual_residual_historical_geometry_uncertainty_not_refitted_objective"
    return check


def _resolve_history(audit, history, frontiers):
    """Apply the solve/gate expansion checks to the accumulated history too."""
    if history["reference"] is None:
        return history
    states = []
    for depth in frontiers:
        key = f"frontier/{depth}"
        fitted, _, g, r, a, b, _ = _frontier_state(audit, key, depth, audit.precision)
        reference = audit.cache.get(f"{key}/reference", lambda: audit.reference(
            f"{key}/reference", fitted["model"], g, r, audit.precision, a, b,
            converged=fitted["fit"]["converged"]))
        states.append((depth, reference["matrix"], fitted, g, r, a, b))

    def checks():
        return {depth: _historical_uncertainty(audit, history, h, fitted, g, r, a, b)
                for depth, h, fitted, g, r, a, b in states}

    initial_checks = checks()
    initial, maximum = audit.study.reference_samples, audit.study.reference_max_samples
    expanded = False
    if any(check["expand"] for check in initial_checks.values()) and maximum > initial:
        for i, contribution in enumerate(history["contributions"]):
            if contribution["input_count"] == maximum:
                continue
            entry = audit.snapshot["history"][i]
            for half in range(2):
                extra = audit.estimate(f"history/{i}/half{half}/extension", entry["model"],
                                       (maximum - initial) // 2, snapshot=entry)["matrix"]
                old = history["half_contributions"][half][i]
                history["half_contributions"][half][i] = (initial * old + (maximum - initial) * extra) / maximum
            halves = [history["half_contributions"][half][i] for half in range(2)]
            check = reference_disagreement(*halves)
            contribution.update(input_count=maximum, unresolved=check["expand"])
            contribution["checks"].append({"input_count": maximum, **check})
            expanded = True
        halves = [replay_archive(audit.assets["initial_fisher"], audit.snapshot["history"],
                                 contributions=part) for part in history["half_contributions"]]
        history["reference"].update(matrix=(halves[0] + halves[1]) / 2, halves=halves)
    final_checks = checks()
    history["reference"].update(
        initial_frontier_checks=initial_checks, final_frontier_checks=final_checks,
        expanded_for_conditional_geometry=expanded,
        unresolved=any(check["expand"] for check in final_checks.values())
        or any(part["unresolved"] for part in history["contributions"]))
    return history


def _prepare(store, index, case, asset_path, driver_path, *, resume, phase, kind):
    if case["environment"]["family"] not in {"mixture", "rotation"}:
        raise ValueError("unknown audit environment")
    if isinstance(case["step"], bool) or not isinstance(case["step"], int) or case["step"] < 0:
        raise ValueError("audit step must be a nonnegative integer")
    if kind == "fisher_audit" and (isinstance(case["frontier"], bool) or case["frontier"] not in range(1, 6)):
        raise ValueError("frontier must lie in 1 through 5")
    asset_path, driver_path = Path(asset_path), Path(driver_path)
    _parent(store, asset_path, ("assets.pt", "summary.json"))
    driver_config = _parent(store, driver_path, ("snapshots.pt", "summary.json"))
    if driver_config["environment"] != case["environment"]["name"]:
        raise ValueError("audit environment differs from its source driver")
    detail = driver_config["detail"]
    if "asset_hash" in detail and detail["asset_hash"] != file_hash(asset_path / "config.json"):
        raise ValueError("audit assets differ from the driver's source assets")
    if detail["policy"].get("depth") is not None and detail["policy"]["depth"] != case["source_depth"]:
        raise ValueError("audit source depth differs from its preassigned driver")
    if kind == "depth_audit" and driver_config["phase"] != "phase1":
        raise ValueError("depth opportunity must use a Phase 1 source driver")
    unit = store.unit(phase, kind, index, environment=case["environment"]["name"],
                      detail={"case": case, "parents": {"assets": canonical_hash(_read_json(asset_path / "config.json")),
                                                         "driver": canonical_hash(driver_config)},
                              "asset_hash": file_hash(asset_path / "assets.pt"),
                              "snapshot_hash": file_hash(driver_path / "snapshots.pt"),
                              "audit_schema_version": AUDIT_SCHEMA_VERSION})
    session = store.begin(unit, REQUIRED, resume=resume)
    if session is None:
        return store.paths(unit)[0], None, None
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    snapshots = torch.load(driver_path / "snapshots.pt", map_location="cpu", weights_only=False)
    snapshot = select_snapshot(snapshots, case["step"])
    cache = _Checkpoints(session, store.limit)
    audit = _Audit(store, index, case, assets, snapshot, driver_config,
                   _read_json(driver_path / "summary.json"), cache,
                   namespace=f"{phase}:{kind}:{canonical_hash(case)[:16]}")
    return None, session, audit


def ensure_audit(store, index, case, asset_path, driver_path, *, resume=True) -> Path:
    """Produce Plan 14's exact-step matrix, decision, and EWC-factor audit.

    All work is research-only. No audit changes its driver or production
    buffers. Default budgets come from the frozen Study (16 x 64 low draws;
    4096 -> 8192 references); smoke overrides remain visibly identified.
    """
    completed, session, audit = _prepare(store, index, case, asset_path, driver_path,
                                        resume=resume, phase="phase2", kind="fisher_audit")
    if completed is not None:
        return completed
    cache = audit.cache
    frontiers = list(range(1, 6)) if case["historical"] else [case["frontier"]]
    history = cache.get("history/replay", lambda: _history(audit))
    history = cache.get("history/resolved", lambda: _resolve_history(audit, history, frontiers))
    history["same_bank_error"] = matrix_metrics(history["actual_archive"], history["same_bank_dense"])
    if history["reference"] is not None:
        history["finite_reference_error"] = matrix_metrics(history["actual_archive"], history["reference"]["matrix"])
        history["same_bank_dense_reference_error"] = matrix_metrics(history["same_bank_dense"], history["reference"]["matrix"])
    rows, diagnostics = [], {"history": history, "frontiers": {}, "factorial": {}}
    for depth in frontiers:
        key = f"frontier/{depth}"
        fitted, expanded, g, r, a, b, gains = _frontier_state(audit, key, depth, audit.precision)
        reference = cache.get(f"{key}/reference", lambda: audit.reference(
            f"{key}/reference", fitted["model"], g, r, audit.precision, a, b,
            converged=fitted["fit"]["converged"]))
        instantaneous = cache.get(f"{key}/instantaneous", lambda: audit.reference(
            f"{key}/instantaneous", fitted["model"], g, r, audit.precision, a, b,
            converged=fitted["fit"]["converged"], instantaneous=audit.snapshot["position"]))
        low_matrices, comparisons = [], []
        for repeat in range(audit.study.audit_score_repeats):
            low = audit.estimate(f"{key}/low/{repeat}", fitted["model"], audit.study.score_samples)
            low_matrices.append(low["matrix"])
            for name, kind, rank in (*REPRESENTATIONS, ("lr8_sign_control", "lr", 8)):
                record = cache.get(f"{key}/row/{repeat}/{name}", lambda: _comparison(
                    audit, key, depth, repeat, name, kind, rank, low["matrix"], reference,
                    audit.precision, fitted, expanded, g, r, a, b, gains, history,
                    orientation=name == "lr8_sign_control"))
                rows.append(record["row"])
                comparisons.append({"repeat": repeat, "representation": name,
                                    **{k: v for k, v in record.items() if k != "row"}})
        variances = {}
        for name, kind, rank in REPRESENTATIONS:
            variances[name] = cache.get(f"{key}/variance/{name}", lambda: bias_variance(
                torch.stack([represent(matrix, kind=kind, rank=rank, layer_indices=audit.groups) for matrix in low_matrices]),
                reference["matrix"]))
        numerical = cache.get(f"{key}/identical_operator", lambda: _numerical_solve(
            low_matrices[0], audit.precision.cpu(), r.cpu(), audit.rank, torch.cat((a, b)).cpu()))
        history_check = _historical_uncertainty(audit, history, reference["matrix"], fitted, g, r, a, b)
        diagnostics["frontiers"][depth] = {
            "restricted": fitted, "expanded": expanded, "reference": reference,
            "instantaneous_reference": instantaneous, "low_budget_matrices": low_matrices,
            "comparisons": comparisons, "bias_variance": variances,
            "historical_reference_uncertainty": history_check, "identical_operator_solve": numerical,
            "temporal_mismatch": matrix_metrics(reference["matrix"], instantaneous["matrix"]),
        }

        # Changing EWC changes the stationary point. Both rows restart at the
        # saved common state, each with its own restricted and expanded fits.
        for ewc in ("diag", "lr"):
            factor_key = f"factorial/{depth}/{ewc}"
            precision, _ = retention_precision(history["compressed"][ewc].to(audit.device), **audit.retention)
            f, e, fg, fr, fa, fb, gain = _frontier_state(audit, factor_key, depth, precision)
            ref = cache.get(f"{factor_key}/reference", lambda: audit.reference(
                f"{factor_key}/reference", f["model"], fg, fr, precision, fa, fb,
                converged=f["fit"]["converged"], seed_key=f"factorial/{depth}/reference"))
            bank = audit.estimate(f"{factor_key}/bank", f["model"], audit.study.score_samples,
                                  seed_key=f"factorial/{depth}/bank")
            factor_rows = []
            for gate_kind in ("diag", "lr", "dense_reference"):
                name, kind = gate_kind, "dense" if gate_kind == "dense_reference" else gate_kind
                source = ref["matrix"] if gate_kind == "dense_reference" else bank["matrix"]
                record = cache.get(f"{factor_key}/row/{name}", lambda: _comparison(
                    audit, factor_key, depth, 0, name, kind, audit.rank, source, ref,
                    precision, f, e, fg, fr, fa, fb, gain, history))
                row = {**record["row"], "category": "ewc_gate_attribution", "ewc": ewc, "gate": gate_kind,
                       "refitted_after_precision_change": True}
                rows.append(row)
                factor_rows.append(record)
            diagnostics["factorial"][factor_key] = {"restricted": f, "expanded": e,
                                                    "reference": ref, "bank": bank, "comparisons": factor_rows}
    unresolved = sum(not row["restricted_fit_converged"] for row in rows)
    limitations = ["All references are finite; input-pool and initial-asset uncertainty remain.",
                   "Dense structured realizations and dense retention square roots do not establish scalable costs.",
                   "Fixed-state resamplings and holdout intervals are not independent trajectory replicas."]
    if unresolved:
        limitations.append("Unresolved restricted fits are retained; their gain forecasts and gate regret are unavailable.")
    if any(not row["expanded_fit_converged"] for row in rows):
        limitations.append("Some expanded shadows did not converge; their measured changes are not optimum depth gains.")
    if any(v["historical_reference_uncertainty"] and v["historical_reference_uncertainty"]["expand"]
           for v in diagnostics["frontiers"].values()):
        limitations.append("Historical inverse-sensitive reference comparisons remain unresolved; see per-frontier checks.")
    summary = {"metric_schema_version": AUDIT_SCHEMA_VERSION, "case": case, "width": audit.width,
               "frontiers": frontiers, "row_count": len(rows), "unresolved_restricted_rows": unresolved,
               "all_restricted_fits_unresolved": unresolved == len(rows),
               "score_repeats": audit.study.audit_score_repeats, "scores_per_repeat": audit.study.score_samples,
               "reference_samples": audit.study.reference_samples, "reference_max_samples": audit.study.reference_max_samples,
               "research_only": True, "smoke": audit.study.smoke, "limitations": limitations,
               "audit_initialization_seed": audit.seed("initialization"),
               "parameter_layout": audit.layout.metadata()}
    session.write_json("audit_metrics.json", _json(rows))
    session.write_json("summary.json", _json(summary))
    session.write_torch("diagnostics.pt", _cpu(diagnostics))
    return store.finish(session, (*REQUIRED, *cache.required()))


def ensure_depth_audit(store, index, case, asset_path, driver_path, *, resume=True) -> Path:
    """Six independent same-start depth fits, with disjoint selection/scoring panels."""
    completed, session, audit = _prepare(store, index, case, asset_path, driver_path,
                                        resume=resume, phase="phase1", kind="depth_audit")
    if completed is not None:
        return completed
    fits, panels, rows = {}, {}, []
    baseline = {split: audit.evaluation(f"depth/baseline/{split}", audit.snapshot["model"], **{split: True})
                for split in ("selection", "scoring")}
    for depth in range(1, 7):
        fitted = audit.fit(f"depth/{depth}/fit", audit.snapshot["model"], depth, audit.precision)
        fits[depth] = fitted
        panels[depth] = {split: audit.evaluation(f"depth/{depth}/{split}", fitted["model"], **{split: True})
                         for split in ("selection", "scoring")}
        gains = {split: _shadow_gains(audit, baseline[split], panels[depth][split]) for split in panels[depth]}
        rows.append({"metric_schema_version": AUDIT_SCHEMA_VERSION, "depth": depth,
                     "step": case["step"], "converged": fitted["fit"]["converged"],
                     "restricted_residual_max": fitted["fit"]["restricted_residual_max"],
                     "gains": gains, "common_start": True})
    valid = [row for row in rows if row["converged"]]
    selected = max(valid, key=lambda row: (row["gains"]["selection"]["current"]["gain"], -row["depth"]))["depth"] if valid else None
    for row in rows:
        row["selected_depth"] = selected
        row["scoring_regret"] = None if selected is None else paired_panel_gain(
            panels[row["depth"]]["scoring"]["current"], panels[selected]["scoring"]["current"],
            family=audit.family, position=audit.snapshot["position"])
    limitations = ["Local selection uses only its selection panel; scoring regret may be negative.",
                   "The selected finite-panel depth is not an achievable online oracle."]
    if not valid:
        limitations.append("No depth fit converged; all cases retained and no optimum depth selected.")
    session.write_json("audit_metrics.json", _json(rows))
    session.write_json("summary.json", {"metric_schema_version": AUDIT_SCHEMA_VERSION, "case": case,
                                        "width": audit.width, "selected_depth": selected,
                                        "converged_depths": len(valid), "research_only": True,
                                        "selection_scoring_split": "within_class_even_odd_disjoint",
                                        "limitations": limitations})
    session.write_torch("diagnostics.pt", _cpu({"fits": fits, "panels": panels, "baseline": baseline}))
    return store.finish(session, (*REQUIRED, *audit.cache.required()))
