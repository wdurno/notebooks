"""Local mathematical validation and frozen, synthetic operation-cost profiling.

This unit never loads MNIST, trains a source model, chooses an architecture,
or runs a trajectory. Real-data CPU/CUDA smokes are separate parent-run units.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import platform
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.derivatives import per_sample_derivatives
from src.mnist_model import mnist_nll
from src.parameters import ParameterLayout
from src.seeding import derive_component_seed

from .artifacts import UnitStore
from .budget import WorkBudget
from .config import canonical_hash
from .evidence import (
    consolidate, effective_sample_size, gradient_noise_covariance,
    retention_precision, weights,
)
from .fisher import model_fisher, represent
from .gate import block_preview, direct_block_preview, evaluate_gate, optimism
from .model import active_indices, build_model


PROFILE_SCHEMA_VERSION = 1
WIDTHS = (5, 6)
BUCKETS = {"rotation": (4, 32, 128), "mixture": (8, 18, 64, 180, 512, 1820)}
REQUIRED_ARTIFACTS = ("summary.json", "profile.json", "metrics.json", "math_validation.json")
REPAIR_PROFILE_ARTIFACTS = ("profile.json", "summary.json", "metrics.json")
MIN_CHARGE_SECONDS = 1e-6
STANDARD_FIT_EVALUATIONS = 50


def upper_quartile(values: list[float]) -> float:
    """Linear-interpolated sample Q.75, shared by operations and packages."""
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("timings must be a nonempty sequence of finite nonnegative values")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), .75))


def quadratic_validation(seed: int, samples: int = 16384) -> dict:
    """Check penalized quadratic optimism and parameter covariance by replication.

    These are estimator-response tests, not a Gaussian auxiliary-score LFU
    experiment. The historical penalty/anchor are fixed across replications.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("quadratic validation requires at least two replications")
    H = torch.tensor([[2., .4, .1], [.4, 1.3, -.2], [.1, -.2, .9]], dtype=torch.float64)
    P = torch.tensor([[.8, -.2, .1], [-.2, .7, .05], [.1, .05, .5]], dtype=torch.float64)
    V = torch.tensor([[.06, .01, -.004], [.01, .04, .003], [-.004, .003, .03]], dtype=torch.float64)
    g_population = torch.tensor([.2, -.1, .4], dtype=torch.float64)
    anchor_offset = torch.tensor([.4, -.2, .3], dtype=torch.float64)
    response = torch.linalg.solve(H + P, torch.eye(3, dtype=torch.float64))
    generator = torch.Generator().manual_seed(seed)
    xi = torch.randn(samples, 3, generator=generator, dtype=torch.float64) @ torch.linalg.cholesky(V).mT
    population_fit = -response @ (g_population + P @ anchor_offset)
    replicated_fits = population_fit - xi @ response.mT
    errors = replicated_fits - population_fit
    empirical_optimism = float(-(xi * errors).sum(1).mean())
    centered = errors - errors.mean(0)
    empirical_covariance = centered.mT @ centered / (samples - 1)
    expected_covariance = response @ V @ response.mT
    expected_optimism = optimism(H, P, V)["optimism"]
    optimism_error = abs(empirical_optimism - expected_optimism) / expected_optimism
    covariance_error = float(torch.linalg.matrix_norm(empirical_covariance - expected_covariance)
                             / torch.linalg.matrix_norm(expected_covariance))
    if optimism_error > .05 or covariance_error > .06:
        raise AssertionError("replicated quadratic moments disagree with the declared response")
    return {
        "status": "passed", "seed": seed, "replications": samples,
        "expected_optimism": expected_optimism, "empirical_optimism": empirical_optimism,
        "optimism_relative_error": optimism_error,
        "expected_parameter_covariance": expected_covariance.tolist(),
        "empirical_parameter_covariance": empirical_covariance.tolist(),
        "covariance_relative_error": covariance_error,
        "assumptions": "fixed PSD penalty, fixed nonzero anchor offset, independent gradient noise",
    }


def mathematical_validation(seed: int) -> dict:
    """Small deterministic CPU checks complement, but do not invoke, pytest."""
    checks = {}
    counts = {}
    for width in WIDTHS:
        model, layout = build_model(seed, width, dtype=torch.float64)
        counts[str(width)] = {"total": layout.total_numel,
                              "depths": [active_indices(layout, d)[0].numel() for d in range(1, 7)]}
        expected = {5: [54, 84, 114, 239, 461, 501], 6: [63, 105, 147, 297, 519, 559]}[width]
        if counts[str(width)]["depths"] != expected:
            raise AssertionError("Plan 14 coordinate chart changed")
        chart = model.output.contrast
        torch.testing.assert_close(chart.mT @ chart, torch.eye(9, dtype=torch.float64))
        torch.testing.assert_close(chart.sum(0), torch.zeros(9, dtype=torch.float64))
    checks["parameter_counts_and_helmert_chart"] = counts

    H = torch.tensor([[2., .3, .7], [.3, 1., .2], [.7, .2, 2.]], dtype=torch.float64)
    P = torch.tensor([[1., .2, .1], [.2, .8, .1], [.1, .1, .9]], dtype=torch.float64)
    r = torch.tensor([.2, -.7, .5], dtype=torch.float64)
    preview = block_preview(r, H + P, [2], [0])
    torch.testing.assert_close(preview["displacement"], direct_block_preview(r, H + P, [2], [0]))
    if preview["displacement"][1] != 0:
        raise AssertionError("a frozen coordinate moved")
    checks["global_coordinate_schur"] = preview["solve_diagnostics"]["relative_residual"]
    eye = torch.eye(2, dtype=torch.float64)
    zero = torch.zeros_like(eye)
    for score in (.9, 1., 1.1):
        g = torch.tensor([0., score], dtype=torch.float64)
        result = evaluate_gate(g, g, eye, zero, eye / 2, [0], [1])
        if result["open"] != (2 * score**2 > 2):
            raise AssertionError("classical S > 2q limit or equality rule failed")
        summed = evaluate_gate(2*g, 2*g, 2*eye, zero, 2*eye, [0], [1])
        torch.testing.assert_close(torch.tensor(summed["gain"]), torch.tensor(2*result["gain"]))
        if summed["open"] != result["open"]:
            raise AssertionError("mean/sum objective units changed the gate")
    checks["classical_limit_and_mean_sum_scaling"] = "passed"
    g = torch.tensor([10., 1.], dtype=torch.float64)
    corrected = evaluate_gate(g, g, eye, zero, eye / 100, [0], [1])
    if corrected["gain"] != .5 or corrected["status"] != "restricted_fit_unresolved":
        raise AssertionError("active residual was rewarded as candidate gain")
    diagonal_h = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    cross_p = torch.tensor([[3., 1.], [1., 2.]], dtype=torch.float64)
    cross = block_preview(torch.tensor([0., 1.]), diagonal_h + cross_p, [0], [1])
    torch.testing.assert_close(cross["schur"], torch.tensor([[3.75]], dtype=torch.float64))
    checks["residual_correction_and_cross_penalty"] = "passed"

    a = weights([2, 1], 1)
    torch.testing.assert_close(a, torch.tensor([.25, .25, .5], dtype=torch.float64))
    torch.testing.assert_close(gradient_noise_covariance(eye, a), .375 * eye)
    if abs(effective_sample_size(a) - 8 / 3) > 1e-12:
        raise AssertionError("finite-weight ESS mismatch")
    checks["finite_ess_and_squared_weights"] = "passed"
    informationless = evaluate_gate(torch.zeros(2), torch.zeros(2), zero, zero, zero, [0], [1])
    if informationless["open"] or informationless["d_eff_expanded"] != 0:
        raise AssertionError("zero information created capacity")
    weak = torch.diag(torch.tensor([1., 1e-12], dtype=torch.float64))
    dimensions = [optimism(weak, zero, weak, floor_relative=f)["d_eff"] for f in (1e-9, 1e-8, 1e-7)]
    if any(abs(value - 2.) > 1e-12 for value in dimensions):
        raise AssertionError("numerical damping masqueraded as statistical ridge")
    checks["zero_information_and_floor_dimension"] = "passed"

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_component_seed(seed, "score_validation"))
        linear = nn.Linear(2, 10, dtype=torch.float64)
    layout = ParameterLayout.from_module(linear)
    x = torch.tensor([[.2, -.7], [.5, 1.]], dtype=torch.float64)
    labels = torch.tensor([2, 7])
    direction = torch.linspace(-.1, .1, layout.total_numel, dtype=torch.float64)
    derivatives = per_sample_derivatives(linear, x, labels, mnist_nll, layout,
                                          direction=direction, strategy="vmap")
    fisher, diagnostic = model_fisher(linear, layout, x, exact_labels=True, return_score_bank=True)
    bank = diagnostic["score_bank"]
    torch.testing.assert_close(bank["scores"][[2, 17]], -derivatives.gradients)
    expected_fisher = torch.zeros_like(fisher)
    for row in x:
        p = linear(row).detach().softmax(-1)
        jacobian = torch.cat((torch.kron(torch.eye(10, dtype=torch.float64), row[None]),
                              torch.eye(10, dtype=torch.float64)), dim=1)
        expected_fisher += .5 * jacobian.mT @ (torch.diag(p) - torch.outer(p, p)) @ jacobian
    torch.testing.assert_close(fisher, expected_fisher, atol=1e-12, rtol=1e-10)
    vector = layout.flatten_module(linear, detach=True)
    displaced_gradients = []
    epsilon = 1e-5
    for sign in (-1, 1):
        layout.copy_vector_to_module(linear, vector + sign * epsilon * direction)
        displaced_gradients.append(per_sample_derivatives(linear, x, labels, mnist_nll, layout,
                                                         strategy="vmap").gradients)
    layout.copy_vector_to_module(linear, vector)
    finite_difference = (displaced_gradients[1] - displaced_gradients[0]) / (2 * epsilon)
    torch.testing.assert_close(derivatives.hvps, finite_difference, atol=1e-9, rtol=1e-7)
    for kind in ("dense", "diag", "lr", "layer_block"):
        matrix = represent(fisher, kind, rank=4, layer_indices=[list(range(20)), list(range(20, 30))])
        torch.testing.assert_close(matrix.diagonal(), fisher.diagonal(), atol=1e-12, rtol=1e-9)
        if torch.linalg.eigvalsh(matrix).min() < -1e-12:
            raise AssertionError("Fisher representation is not PSD")
    checks["score_sign_exact_fisher_hvp_and_representations"] = "passed"

    updated = consolidate(eye, torch.zeros(2), 2*eye, omega=19, arrivals=2,
                          episode_id="validation", observation_ids=["arrival0:0", "arrival1:0"], kind="dense")
    try:
        consolidate(updated["archive"], updated["anchor"], eye, omega=19, arrivals=2,
                    episode_id="validation", observation_ids=["arrival2:0"], ledger=updated["ledger"])
    except ValueError:
        pass
    else:
        raise AssertionError("episode could be consolidated twice")
    budget = WorkBudget(.3, {"fit": .1, "archive": .2})
    budget.grant_arrival()
    if not budget.reserve("archive", "archive") or not budget.spend("fit") or budget.spend("fit"):
        raise AssertionError("work accounting permits future borrowing")
    restored = WorkBudget.from_state_dict(budget.state_dict())
    restored.spend_reserved("archive")
    if abs(restored.spent - .3) > 1e-15 or restored.elapsed_arrivals != 1:
        raise AssertionError("work accounting refilled credit on resume")
    checks["single_consolidation_and_budget_restoration"] = "passed"
    quadratic = quadratic_validation(derive_component_seed(seed, "quadratic_validation"))
    checks["quadratic_optimism_and_covariance"] = quadratic
    return {"status": "passed", "seed": seed, "checks": checks,
            "trajectory_smokes": "parent_managed_not_run_here", "unit_tests": "parent_managed"}


def operation_specs() -> list[dict]:
    """Charge-key coverage is frozen, not reduced when a smoke config is used."""
    specs = []
    for width in WIDTHS:
        for n in sorted({n for values in BUCKETS.values() for n in values}):
            base = {"width": width, "n": n}
            specs.append({**base, "operation": "prequential", "key": f"prequential:w{width}:n{n}"})
            for depth in range(1, 7):
                for operation in ("gradient", "preview"):
                    specs.append({**base, "operation": operation, "depth": depth,
                                  "key": f"{operation}:w{width}:d{depth}:n{n}"})
                for ewc in ("diag", "lr"):
                    specs.append({**base, "operation": "fit", "depth": depth, "ewc": ewc,
                                  "key": f"fit:w{width}:d{depth}:n{n}:e{ewc}"})
            for ewc in ("diag", "lr"):
                specs.append({**base, "operation": "archive", "ewc": ewc,
                              "key": f"archive:w{width}:n{n}:e{ewc}"})
                for depth in range(1, 6):
                    for gate in ("diag", "lr"):
                        specs.append({**base, "operation": "probe", "depth": depth,
                                      "ewc": ewc, "gate": gate,
                                      "key": f"probe:w{width}:d{depth}:n{n}:e{ewc}:g{gate}"})
    return specs


class _OperationContext:
    """Read-only parameter state; no optimizer iteration occurs during timing."""

    def __init__(self, study, width: int, n: int, *, gate_rank: int | None = None,
                 probe_score_samples: int | None = None):
        self.study = study
        self.model, self.layout = build_model(study.seed("phase0:initialization"), width,
                                              device=study.device, dtype=getattr(torch, study.dtype))
        self.device = next(self.model.parameters()).device
        self.vector = self.layout.flatten_module(self.model, detach=True)
        self.parameters = {name: p.detach() for name, p in self.model.named_parameters()}
        self.buffers = {name: b.detach() for name, b in self.model.named_buffers()}
        # Reinitializing named generators makes different buffer sizes nested.
        self.inputs = torch.randn(n, 1, 28, 28, generator=torch.Generator().manual_seed(
            study.seed("phase0:inputs")), dtype=getattr(torch, study.dtype)).to(self.device)
        self.targets = torch.randint(10, (n,), generator=torch.Generator().manual_seed(
            study.seed("phase0:observed_labels"))).to(self.device)
        self.weights = torch.full((n,), 1 / n, device=self.device, dtype=torch.float64)
        self.indices = {d: active_indices(self.layout, d)[0].to(self.device) for d in range(1, 7)}
        d = self.layout.total_numel
        factors = torch.randn(d, study.rank, dtype=torch.float64,
                              generator=torch.Generator().manual_seed(study.seed("phase0:precision_factors"))) / math.sqrt(d)
        diagonal = .1 + torch.rand(d, dtype=torch.float64, generator=torch.Generator().manual_seed(
            study.seed("phase0:precision_diagonal"))) / 10
        source = (factors @ factors.mT + torch.diag(diagonal)).to(self.device)
        self.archives = {kind: represent(source, kind, rank=study.rank) for kind in ("diag", "lr")}
        self.precisions = {kind: retention_precision(matrix, 19)[0] for kind, matrix in self.archives.items()}
        offset = .001 * torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(
            study.seed("phase0:anchor_offset"))).to(self.device)
        self.anchor = self.vector.double() - offset
        self.score_samples = 4 if study.smoke else 64
        self.probe_score_samples = self.score_samples if probe_score_samples is None else probe_score_samples
        self.gate_rank = study.rank if gate_rank is None else gate_rank
        self.score_seed = study.seed("phase0:fisher_scores")
        self.selections = {}
        for depth, indices in self.indices.items():
            pieces = []
            for spec in self.layout.specs:
                positions = torch.nonzero((indices >= spec.start) & (indices < spec.stop)).flatten()
                if positions.numel():
                    pieces.append((spec, positions, indices[positions] - spec.start))
            self.selections[depth] = pieces

    def _parameters_at(self, coordinates: Tensor, depth: int) -> dict[str, Tensor]:
        parameters = dict(self.parameters)
        for spec, positions, local_indices in self.selections[depth]:
            parameters[spec.name] = self.parameters[spec.name].reshape(-1).index_copy(
                0, local_indices, coordinates[positions]).reshape(spec.shape)
        return parameters

    def _data_gradient(self, depth: int) -> tuple[Tensor, Tensor]:
        coordinates = self.vector[self.indices[depth]].clone().requires_grad_(True)
        logits = functional_call(self.model, (self._parameters_at(coordinates, depth), self.buffers),
                                 (self.inputs,), strict=True)
        loss = self.weights @ nn.functional.cross_entropy(logits, self.targets, reduction="none").double()
        gradient = torch.autograd.grad(loss, coordinates)[0].detach().double()
        full_gradient = torch.zeros_like(self.vector, dtype=torch.float64)
        full_gradient[self.indices[depth]] = gradient
        return loss.detach(), full_gradient

    def fit(self, depth: int, ewc: str):
        loss, gradient = self._data_gradient(depth)
        offset = self.vector.double() - self.anchor
        product = self.precisions[ewc] @ offset
        return loss + .5 * offset @ product, (gradient + product)[self.indices[depth]]

    def gradient(self, depth: int):
        return self._data_gradient(depth)

    def preview(self, depth: int):
        with torch.no_grad():
            coordinates = self.vector[self.indices[depth]] + .001
            logits = functional_call(self.model, (self._parameters_at(coordinates, depth), self.buffers),
                                     (self.inputs,), strict=True)
            loss = self.weights @ nn.functional.cross_entropy(logits, self.targets, reduction="none").double()
            offset = self.vector.index_copy(0, self.indices[depth], coordinates).double() - self.anchor
            return loss + .5 * offset @ self.precisions["lr"] @ offset

    def prequential(self):
        with torch.inference_mode():
            logits = self.model(self.inputs)
            log_probabilities = logits.double().log_softmax(-1)
            nll = -self.weights @ log_probabilities.gather(1, self.targets[:, None]).squeeze(1)
            return nll, logits.argmax(-1), log_probabilities[:, 9].exp()

    def archive(self, ewc: str):
        fisher, _ = model_fisher(self.model, self.layout, self.inputs, self.weights,
                                 sample_size=self.score_samples, seed=self.score_seed,
                                 chunk_size=self.study.fisher_chunk_size)
        return consolidate(self.archives[ewc], self.vector, fisher, omega=19,
                           arrivals=self.study.episode_arrivals, episode_id="profile",
                           observation_ids=list(range(len(self.inputs))), kind=ewc, rank=self.study.rank)

    def probe(self, depth: int, ewc: str, gate: str):
        indices = self.indices[depth + 1]
        _, gradient = self._data_gradient(depth + 1)
        fisher, _ = model_fisher(self.model, self.layout, self.inputs, self.weights,
                                 sample_size=self.probe_score_samples, seed=self.score_seed,
                                 chunk_size=self.study.fisher_chunk_size)
        H = represent(fisher, gate, rank=self.gate_rank)
        residual = gradient + self.precisions[ewc] @ (self.vector.double() - self.anchor)
        active = self.indices[depth]
        candidate = indices[~torch.isin(indices, active)]
        return evaluate_gate(gradient, residual, H, self.precisions[ewc], H * self.weights.square().sum(),
                             active, candidate, floor_relative=self.study.solve_floor_relative)

    def operation(self, spec: dict) -> Callable:
        operation = spec["operation"]
        if operation in ("gradient", "preview"):
            return lambda: getattr(self, operation)(spec["depth"])
        if operation == "fit":
            return lambda: self.fit(spec["depth"], spec["ewc"])
        if operation == "archive":
            return lambda: self.archive(spec["ewc"])
        if operation == "probe":
            return lambda: self.probe(spec["depth"], spec["ewc"], spec["gate"])
        if operation == "prequential":
            return self.prequential
        raise ValueError(f"unknown profile operation: {operation}")


def measure_operation(operation: Callable, *, device: torch.device | str, repeats: int,
                      warmups: int = 1, clock: Callable = time.perf_counter) -> dict:
    """Synchronized wall timing only; outcomes never select costs or budgets."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("timing repeats must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("timing warmups must be a nonnegative integer")
    device = torch.device(device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmups):
        operation()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        started = clock()
        operation()
        synchronize()
        samples.append(max(0., clock() - started))
    return {"seconds": samples, "upper_quartile_seconds": upper_quartile(samples),
            "charge": max(MIN_CHARGE_SECONDS, upper_quartile(samples)),
            "warmups": warmups, "repeats": repeats}


def standard_work_packages(charges: dict[str, float], episode_arrivals: int) -> tuple[dict, list[dict]]:
    """Q.75 over equal width/bucket/EWC/gate cells of the fixed work package."""
    if isinstance(episode_arrivals, bool) or not isinstance(episode_arrivals, int) or episode_arrivals < 1:
        raise ValueError("episode_arrivals must be a positive integer")
    rows = []
    for family, buckets in BUCKETS.items():
        for width in WIDTHS:
            for n in buckets:
                for ewc in ("diag", "lr"):
                    for gate in ("diag", "lr"):
                        terms = {
                            "output_fit": STANDARD_FIT_EVALUATIONS * charges[f"fit:w{width}:d1:n{n}:e{ewc}"],
                            "frontier_probe": charges[f"probe:w{width}:d1:n{n}:e{ewc}:g{gate}"],
                            "next_depth_fit": STANDARD_FIT_EVALUATIONS * charges[f"fit:w{width}:d2:n{n}:e{ewc}"],
                            "amortized_archive": charges[f"archive:w{width}:n{n}:e{ewc}"] / episode_arrivals,
                        }
                        rows.append({"family": family, "width": width, "n": n, "ewc": ewc,
                                     "gate": gate, **terms, "seconds": sum(terms.values())})
    b0 = {family: upper_quartile([row["seconds"] for row in rows if row["family"] == family])
          for family in BUCKETS}
    return b0, rows


@contextlib.contextmanager
def _cpu_threads(count: int):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield
    finally:
        torch.set_num_threads(previous)


def _read_checked(path: Path, identity: dict) -> dict:
    value = json.loads(path.read_text())
    if value.get("identity") != identity or value.get("checksum") != canonical_hash(value.get("payload")):
        raise RuntimeError(f"incompatible or corrupt Phase 0 partial: {path}")
    return value["payload"]


def _write_checked(session, name: str, identity: dict, payload: dict):
    session.write_json(name, {"identity": identity, "payload": payload, "checksum": canonical_hash(payload)})


def ensure_phase0(store: UnitStore, resume: bool = True) -> Path:
    """Seal/reuse one validation/profile unit; parent orchestration invokes it.

    Each completed operation timing is atomically checkpointed. A bounded
    interruption loses at most the current measurement and never regenerates
    an already checkpointed charge. All source/config hashes are inherited
    from UnitStore. No data download or experimental trajectory is performed.
    """
    study = store.study
    repeats, warmups = (1, 0) if study.smoke else (3, 1)
    detail = {"profile_schema_version": PROFILE_SCHEMA_VERSION,
              "widths": list(WIDTHS), "buckets": {k: list(v) for k, v in BUCKETS.items()},
              "repeats": repeats, "warmups": warmups,
              "score_samples": 4 if study.smoke else 64,
              "standard_fit_evaluations": STANDARD_FIT_EVALUATIONS,
              "episode_arrivals": study.episode_arrivals}
    unit = store.unit("phase0", "validation_profile", 0, detail=detail)
    session = store.begin(unit, REQUIRED_ARTIFACTS, resume=resume)
    if session is None:
        return store.completed(unit, REQUIRED_ARTIFACTS)
    if study.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA timing device is unavailable; no CPU substitution is allowed")
    seeds = {name: study.seed(f"phase0:{name}") for name in (
        "initialization", "inputs", "observed_labels", "precision_factors", "precision_diagonal",
        "anchor_offset", "fisher_scores", "math_validation")}
    identity = {"unit_hash": canonical_hash(unit), "schema_version": PROFILE_SCHEMA_VERSION}
    timings = []
    with _cpu_threads(study.cpu_threads):
        math_path = session.path / "math_validation.checkpoint.json"
        if math_path.exists():
            validation = _read_checked(math_path, identity)
        else:
            store.limit.check()
            validation = mathematical_validation(seeds["math_validation"])
            _write_checked(session, "math_validation.checkpoint.json", identity, validation)
        context = None
        context_key = None
        for spec in operation_specs():
            name = f'timing_parts/{canonical_hash(spec)[:24]}.json'
            path = session.path / name
            row_identity = {**identity, "spec": spec}
            if path.exists():
                row = _read_checked(path, row_identity)
            else:
                store.limit.check()
                key = (spec["width"], spec["n"])
                if context_key != key:
                    context = _OperationContext(study, *key)
                    context_key = key
                measured = measure_operation(context.operation(spec), device=study.device,
                                             repeats=repeats, warmups=warmups)
                row = {**spec, **measured}
                _write_checked(session, name, row_identity, row)
            timings.append(row)
    charges = {row["key"]: row["charge"] for row in timings}
    b0, packages = standard_work_packages(charges, study.episode_arrivals)
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION, "charges": charges,
        "charge_table_hash": WorkBudget(None, charges).charge_table_hash,
        "buckets": detail["buckets"],
        "bucket_union": sorted({n for values in BUCKETS.values() for n in values}),
        "family_buckets": detail["buckets"], "widths": list(WIDTHS), "b0": b0,
        "budgets": {family: {"strict": value, "principal": 2*value} for family, value in b0.items()},
        "standard_work_packages": packages,
        "metadata": {
            **detail, "device": study.device, "dtype": study.dtype, "matrix_dtype": "float64",
            "cpu_threads": study.cpu_threads, "python": platform.python_version(),
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name() if study.device == "cuda" else platform.processor(),
            "seeds": seeds, "data": "deterministic synthetic 28x28 tensors; no MNIST",
            "operation_count": len(charges), "units": "profiled synchronized wall seconds",
            "quantile": "linear interpolation at 0.75; operation charges floored at 1e-6 seconds",
            "standard_work_package": "50 output-fit evaluations + one depth-1 probe + 50 depth-2 fit evaluations + archive/episode_arrivals",
            "fit_cost_interpretation": "fixed evaluation workload, not observed convergence or an actual optimizer fit",
            "package_aggregation": "Q.75 of package charge sums; equal width, bucket, EWC, and gate cells within each family",
            "precision": "same named-seed PSD source within each width across every buffer shape; diag/lr realizations; omega=19",
            "preview": "one no-grad candidate objective forward, full dense-realized LR precision; depths 1..6",
            "gradient": "weighted observed-label data gradient in the named active suffix; depths 1..6",
            "probe": "mean gradient on depth+1, full-coordinate model-label Fisher and representation, retained-precision gate solve; depths 1..5",
            "archive": "full-coordinate model-label Fisher, episode consolidation/compression, and precision reconstruction",
            "score_cost_scope": "full_network even for restricted Fisher output; no frozen-prefix cache",
            "excluded_costs": ["model/data/precision fixture setup", "artifact I/O", "real-data transforms", "optimizer iteration bookkeeping"],
            "selection": "no predictive-quality or architecture selection",
            "rounding": "caller rounds actual buffer size up to the next family bucket without changing actual work; bucket_union supports family-free lookup",
            "study_hash": study.config_hash, "source_hashes": store.sources,
        },
    }
    summary = {"schema_version": 1, "phase": "phase0", "status": "completed",
               "scope": ["local_mathematical_validation", "synthetic_operation_timing"],
               "mathematical_validation": validation["status"], "profile_hash": canonical_hash(profile),
               "operation_count": len(charges), "b0": b0,
               "trajectory_smokes": "parent_managed_not_run_here", "quality_selection": False}
    metrics = [{"schema_version": 1, "phase": "phase0", "metric_kind": "operation_timing", **row}
               for row in timings]
    session.write_json("summary.json", summary)
    session.write_json("profile.json", profile)
    session.write_json("metrics.json", metrics)
    session.write_json("math_validation.json", validation)
    return store.finish(session, REQUIRED_ARTIFACTS)


def ensure_repair_profile(
    store: UnitStore, base_profile: dict, policy: dict, width: int, resume: bool = True,
) -> Path:
    """Freeze actual repaired probe costs without changing the common allowance.

    Only this width/EWC/gate's frontier probes are measured. ``score_samples``
    and ``gate_rank`` belong to the probe; archive rank, archive sampling, and
    every other charge remain the base protocol's. The returned profile is for
    this repair policy only. Its ``probe_units=1`` disables sample-ratio scaling
    by the trajectory, because each replacement charge already measures one
    complete repaired probe. No profiling runs until the parent invokes this.
    """
    study = store.study
    base = copy.deepcopy(base_profile)
    policy = copy.deepcopy(policy)
    if base.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("repair profiling requires a compatible base profile schema")
    if (isinstance(width, bool) or not isinstance(width, int) or width not in (5, 6)
            or width not in base.get("widths", ())):
        raise ValueError("repair width must be present in the base profile")
    if (not isinstance(policy.get("name"), str) or not policy["name"]
            or policy.get("depth") is not None or policy.get("ewc") not in ("diag", "lr")
            or policy.get("gate") not in ("diag", "lr")):
        raise ValueError("repair profiling requires a named adaptive diag/lr policy")
    score_samples = policy.get("score_samples")
    archive_rank = policy.get("rank", study.rank)
    gate_rank = policy.get("gate_rank", archive_rank)
    for name, value in (("score_samples", score_samples), ("rank", archive_rank), ("gate_rank", gate_rank)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"repair {name} must be a positive integer")
    if archive_rank != study.rank:
        raise ValueError("gate-only repair profiling cannot change archive rank")
    if policy.get("archive_samples", study.score_samples) != study.score_samples:
        raise ValueError("gate-only repair profiling cannot change archive sampling")
    metadata = base.get("metadata", {})
    for field in ("device", "dtype", "cpu_threads"):
        if field in metadata and metadata[field] != getattr(study, field):
            raise ValueError(f"repair timing {field} differs from the base profile")
    charge_table = WorkBudget(None, base["charges"])
    if base.get("charge_table_hash", charge_table.charge_table_hash) != charge_table.charge_table_hash:
        raise ValueError("base profile charge-table hash mismatch")
    buckets = base["buckets"]
    values = [n for family in buckets.values() for n in family] if isinstance(buckets, dict) else list(buckets)
    if not values or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in values):
        raise ValueError("repair profiling requires positive integer buffer buckets")
    buckets = sorted(set(values))
    specs = [{"operation": "probe", "width": width, "n": n, "depth": depth,
              "ewc": policy["ewc"], "gate": policy["gate"],
              "key": f'probe:w{width}:d{depth}:n{n}:e{policy["ewc"]}:g{policy["gate"]}'}
             for n in buckets for depth in range(1, 6)]
    if any(spec["key"] not in base["charges"] for spec in specs):
        raise ValueError("base profile is missing a required repair probe charge")
    repeats, warmups = (1, 0) if study.smoke else (3, 1)
    parent_hash = canonical_hash(base)
    detail = {"profile_schema_version": PROFILE_SCHEMA_VERSION, "parent_profile_hash": parent_hash,
              "policy": policy, "width": width, "buckets": buckets,
              "score_samples": score_samples, "gate_rank": gate_rank,
              "archive_rank": archive_rank, "archive_samples": study.score_samples,
              "repeats": repeats, "warmups": warmups, "probe_units": 1}
    unit = store.unit("phase4", "repair_profile", 0, policy=policy["name"], detail=detail)
    session = store.begin(unit, REPAIR_PROFILE_ARTIFACTS, resume=resume)
    if session is None:
        return store.completed(unit, REPAIR_PROFILE_ARTIFACTS)
    if study.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA timing device is unavailable; no CPU substitution is allowed")
    identity = {"unit_hash": canonical_hash(unit), "schema_version": PROFILE_SCHEMA_VERSION}
    timings = []
    with _cpu_threads(study.cpu_threads):
        context = None
        context_n = None
        for spec in specs:
            name = f'timing_parts/{canonical_hash(spec)[:24]}.json'
            row_identity = {**identity, "spec": spec}
            path = session.path / name
            if path.exists():
                row = _read_checked(path, row_identity)
            else:
                store.limit.check()
                if context_n != spec["n"]:
                    context = _OperationContext(study, width, spec["n"], gate_rank=gate_rank,
                                                probe_score_samples=score_samples)
                    context_n = spec["n"]
                measurement = measure_operation(context.operation(spec), device=study.device,
                                                repeats=repeats, warmups=warmups)
                row = {**spec, **measurement, "score_samples": score_samples, "gate_rank": gate_rank}
                _write_checked(session, name, row_identity, row)
            timings.append(row)
    profile = copy.deepcopy(base)
    profile["charges"].update({row["key"]: row["charge"] for row in timings})
    profile["charge_table_hash"] = WorkBudget(None, profile["charges"]).charge_table_hash
    profile["probe_units"] = 1
    profile["parent_profile_hash"] = parent_hash
    profile["repair_policy"] = policy
    profile["repair_probe_keys"] = [spec["key"] for spec in specs]
    profile.setdefault("metadata", {})["repair_profiling"] = {
        **detail, "phase": "phase4", "operation_count": len(timings),
        "device": study.device, "dtype": study.dtype, "cpu_threads": study.cpu_threads,
        "scope": "complete probe at actual score_samples/gate_rank; no sample-ratio multiplier",
        "allowance": "base b0 and budgets preserved; no outcome-based allowance increase",
        "archive": "base rank, sampling, objective, and charges unchanged",
        "source_hashes": store.sources,
    }
    summary = {"schema_version": 1, "phase": "phase4", "kind": "repair_profile", "status": "completed",
               "policy": policy["name"], "width": width, "parent_profile_hash": parent_hash,
               "profile_hash": canonical_hash(profile), "operation_count": len(timings),
               "score_samples": score_samples, "gate_rank": gate_rank, "archive_rank": archive_rank,
               "b0": profile["b0"], "probe_units": 1, "non_probe_charges_preserved": True}
    metrics = [{"schema_version": 1, "phase": "phase4", "metric_kind": "repair_probe_timing",
                "policy": policy["name"], **row} for row in timings]
    session.write_json("profile.json", profile)
    session.write_json("summary.json", summary)
    session.write_json("metrics.json", metrics)
    return store.finish(session, REPAIR_PROFILE_ARTIFACTS)
