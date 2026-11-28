"""CPU-only Fix 5A initialization and bounded, cached-data numerical audits.

Use the frozen Fix 5 Archive and Learner unchanged. Fisher columns come from
the conditional likelihood (scores, equivalently negative NLL gradients), not
outer products of batch-mean gradients. No LFU or new Fisher estimand is used.
"""

import copy
from dataclasses import asdict, replace
import hashlib
import math
import statistics
import time

import torch

from numerical_experiment.picar_head_awac.archive import Archive
from numerical_experiment.picar_head_awac.learner import Learner
from numerical_experiment.picar_head_awac.likelihood import fisher_factor, joint_kl
from numerical_experiment.picar_head_awac.model import Heads


RANKS = (0, 10, 32, 64)
INITIAL_SEED = 199001
AUDIT_SEED = 199811
FD_STRENGTH = 12.142951965332031
SCHEMA = 1
CHECKPOINT_RECORDS = 16
COMPRESSION_RECORDS = 8
REPEATS = 3
DIAGONAL_RTOL = 1e-5
DIAGONAL_ATOL = 1e-7


def _fresh(fitted, recipe, seed, method):
    heads = Heads().cpu()
    heads.load_state_dict(fitted["selected_heads"])
    return Learner(heads, recipe, seed, method)


def new_learner(fitted, recipe, seed, method, archives):
    """Legacy campaign bootstrap, with only the requested archive replaced.

    Never load the initializer's spent RNG, target, moments or counters. The
    archive dictionary has string rank keys, exactly as in the saved rebuild.
    """
    learner = _fresh(fitted, recipe, seed, method)
    state = archives[str(recipe.rank)]
    if state["rank"] != recipe.rank or state["diagonal"].shape != (len(learner.ltm),):
        raise ValueError("initial archive rank/coordinate dimension differs")
    # The old bootstrap constructs an archive, rather than loading an entire
    # learner (or the initializer's historical update diagnostics).
    learner.archive = Archive(state["diagonal"].cpu(), factor=state["factor"].cpu(), rank=recipe.rank)
    return learner


def _initial_factor(learner, record):
    # Both draw blocks, including the detached design, are part of the old
    # initialization.initial_archive RNG convention. Do not batch/reorder them.
    data, design, _ = learner.design([record], generator="fisher")
    cfg = learner.recipe
    draws = torch.randn((cfg.integration_draws, 7),
                        generator=learner.generators["fisher"], device="cpu")
    return fisher_factor(learner.coordinates, learner.vector, data["feature"][0],
                         design.weights[0], cfg.action_variance,
                         cfg.critic_variance, draws)[learner.ltm]


def _identity(fitted, records, recipe):
    digest = hashlib.sha256()

    def tensor(name, value):
        value = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    for name, value in sorted(fitted["selected_heads"].items()):
        tensor(name, value)
    for i, record in enumerate(records):
        for name in ("feature", "action", "next_feature", "reward", "discount", "return"):
            tensor(f"{i}:{name}", record[name])
    return {"schema": SCHEMA, "recipe": asdict(recipe), "records": len(records),
            "seed": INITIAL_SEED, "ranks": list(RANKS),
            "chunk_columns": Archive.FRESH_CHUNK_COLUMNS, "sha256": digest.hexdigest()}


def _actor_rows(learner):
    # Coordinates preserve named-parameter order: the default heads have the
    # actor's 455 LTM rows first, followed by 1,754 twin-critic rows.
    return sum(row["count"] for row in learner.coordinates.inventory()
               if row["bank"] == "ltm" and row["name"].startswith("actor."))


def block_diagnostics(archive, actor_rows=455):
    """JSON-only actor/critic allocation; default is the unchanged 2,209 LTM.

    ``actor_rows`` is only overridden for small synthetic numerical fixtures.
    The full diagonal includes both correlated and residual trace.
    """
    if not 0 <= actor_rows <= len(archive.diagonal):
        raise ValueError("actor block does not fit archive coordinates")
    result = {"actor_rows": actor_rows, "critic_rows": len(archive.diagonal) - actor_rows}
    for name, rows in (("actor", slice(None, actor_rows)), ("critic", slice(actor_rows, None))):
        trace = float(archive.diagonal[rows].sum())
        energy = archive.factor[rows].square().sum(0)
        correlated = float(energy.sum())
        result[name] = {"trace": trace, "correlated_trace": correlated,
                        "correlated_trace_fraction": correlated / trace if trace else None,
                        "residual_trace": float(archive.residual[rows].sum()),
                        "factor_column_energy": energy.tolist()}
    return result


def _initial_diagnostics(archives, learner, cursor, reference_archive):
    diagonal = archives[10].diagonal
    reference = {"checked": reference_archive is not None,
                 "rtol": DIAGONAL_RTOL, "atol": DIAGONAL_ATOL}
    if reference_archive is not None:
        expected = (reference_archive.diagonal if isinstance(reference_archive, Archive)
                    else reference_archive["diagonal"]).detach().cpu()
        if expected.shape != diagonal.shape or not torch.isfinite(expected).all():
            raise ValueError("invalid reference initialization diagonal")
        error = diagonal - expected
        reference.update(diagonal_max_abs_error=float(error.abs().max()),
                         diagonal_l2_error=float(error.norm()),
                         matches=bool(torch.allclose(diagonal, expected,
                                      rtol=DIAGONAL_RTOL, atol=DIAGONAL_ATOL)))
        if not reference["matches"]:
            raise ValueError(f"reference initialization diagonal mismatch: {reference}")
    return {"records": cursor, "initial_seed": INITIAL_SEED, "device": "cpu",
            "weighting": "equal weight per original training record",
            "chunk_columns": Archive.FRESH_CHUNK_COLUMNS,
            "reference_rank10": reference,
            "ranks": {str(rank): {**archive.diagnostics(),
                       "diagonal_vs_rank10_max_abs_error": float((archive.diagonal - diagonal).abs().max()),
                       "allocation": block_diagnostics(archive, _actor_rows(learner))}
                      for rank, archive in archives.items()}}


def rebuild_initial(fitted, records, recipe, store, check, reference_archive=None):
    """Rebuild all ranks from one old-compatible, equal-weight factor stream.

    ``store.checkpoint`` transactionally saves ``archives`` (string keys),
    ``cursor``, ``fisher_rng`` and an input identity every 16 records and at the
    final record. The return is {archives, diagnostics, cursor}; only archives
    contains tensors. The caller owns completion/sealing and JSON summaries.
    Production uses the 515 original training records; tiny fixtures may use
    fewer. No heads are fitted, no observations are acquired.
    """
    if not records:
        raise ValueError("initial Fisher requires original training records")
    identity = _identity(fitted, records, recipe)
    learner = _fresh(fitted, replace(recipe, rank=10), INITIAL_SEED, "td_replay")
    saved = store.load_checkpoint(trusted=True)
    if saved is None:
        cursor = 0
        archives = {rank: Archive(torch.zeros(len(learner.ltm)), rank=rank) for rank in RANKS}
    else:
        if saved.get("schema") != SCHEMA or saved.get("identity") != identity:
            raise ValueError("initial Fisher resume identity differs")
        cursor = saved["cursor"]
        if type(cursor) is not int or not 0 <= cursor <= len(records):
            raise ValueError("invalid initial Fisher cursor")
        if set(saved["archives"]) != {str(rank) for rank in RANKS}:
            raise ValueError("initial Fisher archive ranks differ")
        archives = {rank: Archive.from_state_dict(saved["archives"][str(rank)]) for rank in RANKS}
        for rank, archive in archives.items():
            if archive.rank != rank or archive.diagonal.shape != (len(learner.ltm),):
                raise ValueError("initial Fisher archive rank/coordinates differ")
            if not torch.equal(archive.diagonal, archives[10].diagonal):
                raise ValueError("initial Fisher archives do not share a diagonal")
        learner.generators["fisher"].set_state(saved["fisher_rng"].cpu())

    def checkpoint(n, diagnostics):
        store.checkpoint({"schema": SCHEMA, "identity": identity, "cursor": n,
                          "archives": {str(r): a.state_dict() for r, a in archives.items()},
                          "fisher_rng": learner.generators["fisher"].get_state().cpu().clone(),
                          "diagnostics": diagnostics},
                         progress={"stage": "shared equal-weight initial Fisher",
                                   "n": n, "total": len(records)})

    for i in range(cursor, len(records)):
        factor = _initial_factor(learner, records[i])
        for archive in archives.values():
            archive.update(factor, alpha=i / (i + 1), beta=1 / (i + 1))
        if (i + 1) % CHECKPOINT_RECORDS == 0 and i + 1 < len(records):
            checkpoint(i + 1, _initial_diagnostics(archives, learner, i + 1, None))
            check()
    diagnostics = _initial_diagnostics(archives, learner, len(records), reference_archive)
    checkpoint(len(records), diagnostics)
    check()
    return {"archives": {str(r): a.state_dict() for r, a in archives.items()},
            "diagnostics": diagnostics, "cursor": len(records)}


def _benchmark(factory, operation, check):
    """One discarded warmup and three repetitions, each from isolated state."""
    operation(factory())
    check()
    durations = []
    for _ in range(REPEATS):
        subject = factory()
        start = time.perf_counter()
        result = operation(subject)
        durations.append(time.perf_counter() - start)
        check()
    return subject, result, {"seconds": statistics.median(durations),
                              "samples_seconds": durations, "warmups": 1, "repeats": REPEATS}


def _relative(error, reference):
    denominator = float(reference.norm())
    return float(error.norm()) / denominator if denominator else (0. if not bool(error.count_nonzero()) else None)


def _archive_health(archive, direction, weight):
    generator = torch.Generator().manual_seed(AUDIT_SEED + 1)
    rhs = torch.randn(len(archive.diagonal), generator=generator)
    damping = torch.ones_like(rhs) * max(float(archive.diagonal.mean()), .01)
    solution = archive.solve(rhs, damping, weight)
    residual = damping * solution + weight * archive.matvec(solution) - rhs
    vector = rhs.clone().requires_grad_()
    gradient, = torch.autograd.grad(archive.quadratic(vector), vector)
    exact_gradient = 2 * archive.matvec(rhs)
    other = torch.roll(rhs, 1)
    left, right = rhs @ archive.matvec(other), other @ archive.matvec(rhs)
    symmetry = float((left - right).abs()) / max(float(left.abs() + right.abs()), 1e-12)
    gradient_error = _relative(gradient - exact_gradient, exact_gradient)
    solve_error = _relative(residual, rhs)
    diagnostics = archive.diagnostics()
    finite = all(bool(torch.isfinite(value).all()) for value in
                 (solution, gradient, direction, archive.factor, archive.residual))
    return {"finite": finite, "solve_relative_residual": solve_error,
            "quadratic_gradient_relative_error": gradient_error,
            "bilinear_symmetry_relative_error": symmetry,
            "direction_norm": float(direction.norm()),
            "direction_penalty": float(archive.quadratic(direction)),
            "healthy": bool(finite and solve_error is not None and solve_error <= 2e-3
                            and gradient_error is not None and gradient_error <= 1e-5
                            and symmetry <= 2e-4 and diagnostics["minimum_residual"] >= 0
                            and diagnostics["diagonal_max_abs_error"] <= DIAGONAL_ATOL
                            + DIAGONAL_RTOL * float(archive.diagonal.max()))}


def _fd_health(learner, records, acquired):
    data, design, _ = learner.design(records)
    cfg = learner.recipe
    vector = learner.vector.clone().requires_grad_()
    draws = torch.randn((len(records), cfg.integration_draws, 7), generator=learner.generators["fd"])
    probe = torch.zeros_like(vector)
    probe[learner.stm] = (2 * torch.randint(2, (len(learner.stm),),
                                         generator=learner.generators["fd"]) - 1).to(vector)
    shift = .5 * cfg.fd_step * probe
    kl = .5 * (joint_kl(learner.coordinates, vector + shift, vector - shift, data["feature"], design, draws)
               + joint_kl(learner.coordinates, vector - shift, vector + shift, data["feature"], design, draws))
    ages = vector.new_tensor([min(1., max(0., (acquired - r["arrival"] - 1) / cfg.capacity)) for r in records])
    penalty = (ages.square() * kl).mean() / (cfg.fd_step ** 2 * len(learner.stm))
    gradient, = torch.autograd.grad(penalty, vector)
    return {"unit_penalty": float(penalty.detach()), "unit_gradient_norm": float(gradient.norm()),
            "finite": bool(torch.isfinite(penalty) and torch.isfinite(gradient).all())}


def _thin_solve(factor, rhs, damping, weight):
    scaled = factor / damping[:, None]
    gram = factor.mT @ scaled
    middle = torch.eye(gram.shape[0], dtype=gram.dtype) + weight * (gram + gram.mT) / 2
    free = rhs / damping[:, None]
    return free - weight * scaled @ torch.linalg.solve(middle, factor.mT @ free)


def _compression_audit(fitted, records, recipe, directions, check):
    """Bounded cold-start operator comparison, not a full-history error bound."""
    learner = _fresh(fitted, replace(recipe, rank=10), INITIAL_SEED, "td_replay")
    archives = {r: Archive(torch.zeros(len(learner.ltm)), rank=r) for r in RANKS}
    factors = []
    for i, record in enumerate(records[:COMPRESSION_RECORDS]):
        factor = _initial_factor(learner, record)
        factors.append(factor.double())
        for archive in archives.values():
            archive.update(factor, alpha=i / (i + 1), beta=1 / (i + 1))
        check()
    reference = torch.cat(factors, 1) / math.sqrt(len(factors))
    generator = torch.Generator().manual_seed(AUDIT_SEED + 2)
    probes = torch.randn((len(learner.ltm), 4), generator=generator).double()
    probes /= probes.norm(dim=0)
    exact_product = reference @ (reference.mT @ probes)
    weight = (1 - recipe.pi) / recipe.pi
    damping = torch.full((len(learner.ltm),), max(float(reference.square().sum(1).mean()), .01), dtype=torch.float64)
    exact_solution = _thin_solve(reference, probes, damping, weight)
    rows = []
    for rank, archive in archives.items():
        factor, residual = archive.factor.double(), archive.residual.double()
        product = factor @ (factor.mT @ probes) + residual[:, None] * probes
        solution = _thin_solve(factor, probes, damping + weight * residual, weight)
        penalties = []
        for label, direction in directions.items():
            direction = direction.double()
            exact = (reference.mT @ direction).square().sum()
            approximate = (factor.mT @ direction).square().sum() + (residual * direction.square()).sum()
            error = float((approximate - exact).abs())
            penalties.append({"direction": label, "reference": float(exact), "approximate": float(approximate),
                              "absolute_error": error, "relative_error": error / float(exact) if exact else None})
        diagonal_error = archive.diagonal.double() - reference.square().sum(1)
        rows.append({"rank": rank, "random_probe_operator_relative_error": _relative(product - exact_product, exact_product),
                     "random_probe_solve_relative_error": _relative(solution - exact_solution, exact_solution),
                     "diagonal_max_abs_error": float(diagonal_error.abs().max()),
                     "penalty_errors": penalties, "archive": archive.diagnostics(),
                     "allocation": block_diagnostics(archive, _actor_rows(learner))})
    return {"records": len(factors), "reference_columns": reference.shape[1],
            "probe_count": probes.shape[1], "reference_trace": float(reference.square().sum()),
            "scope": "cold-start equal-weight bounded stream; uncompressed FP64 thin-operator reference",
            "operator_error": "relative L2 on four shared random probes, not spectral norm",
            "direction_source": "actual cached fit endpoint displacements from the initialized model",
            "ranks": rows}


def _finite_scalars(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("nonfinite audit diagnostic")
    if isinstance(value, dict):
        for item in value.values():
            _finite_scalars(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_scalars(item)


def _resource_row(fitted, selected, cfg, archives, acquired, method, check):
    def fresh():
        return new_learner(fitted, cfg, AUDIT_SEED, method, archives)

    initial = fresh().state_dict()
    learner, metrics, fit_timing = _benchmark(fresh, lambda l: l.fit(selected, acquired, check), check)
    direction = learner.vector[learner.ltm] - initial["vector"][learner.ltm]
    fitted_state = learner.state_dict()

    def after_fit():
        subject = fresh()
        subject.load_state_dict(fitted_state)
        return subject

    _, factor, factor_timing = _benchmark(after_fit, lambda l: _initial_factor(l, selected[0]), check)
    archive_factory = lambda: Archive.from_state_dict(fitted_state["archive"])
    _, _, compression_timing = _benchmark(archive_factory, lambda a: a.update(factor, 1 - cfg.pi, cfg.pi), check)
    _, _, retirement_timing = _benchmark(after_fit, lambda l: l.retire(selected[0]), check)
    rhs = torch.linspace(-1., 1., len(learner.ltm))
    damping = torch.ones_like(rhs) * max(float(learner.archive.diagonal.mean()), .01)
    weight = (1 - cfg.pi) / cfg.pi
    _, _, solve_timing = _benchmark(archive_factory, lambda a: a.solve(rhs, damping, weight), check)
    health = _archive_health(learner.archive, direction, weight)
    fd_health = _fd_health(fresh(), selected, acquired)
    finite_fit = all(math.isfinite(value) for row in metrics["inner"] for value in row.values())
    anchor_equal = torch.equal(learner.anchor, learner.vector[learner.ltm])
    target_equal = torch.equal(learner.target, initial["target"].clone().lerp_(learner.vector, cfg.target_rate))
    health.update(fd=fd_health, finite_fit=finite_fit,
                  accepted_anchor_equal=anchor_equal, target_update_equal=target_equal,
                  healthy=health["healthy"] and fd_health["finite"] and finite_fit and anchor_equal and target_equal)
    components = {"fit": fit_timing, "factor_generation": factor_timing,
                  "compression": compression_timing, "retirement": retirement_timing, "solve": solve_timing}
    row = {"rank": cfg.rank, "fd_strength": cfg.fd_strength, "method": method, "recipe": asdict(cfg),
           "batch_records": len(selected), "steps": cfg.steps, "timing_available": True,
           **{name + "_seconds": timing["seconds"] for name, timing in components.items()},
           "timing_samples": components, "health": health,
           "archive": learner.archive.diagnostics(),
           "allocation": block_diagnostics(learner.archive, _actor_rows(learner))}
    _finite_scalars(row)
    return row, direction


def resource_audit(fitted, records, recipe, archives, check):
    """Return JSON-only ``timings`` rows and bounded numerical diagnostics.

    Benchmarks use cached training records, never a GPU or model fitting stage.
    Each component has one isolated warmup plus three isolated repetitions.
    Whole Learner.fit timing includes batch construction and CPU transfers,
    reference/FD generation, three optimizer steps and accepted recentering.
    These are audit learners, not production RNG streams or acquisitions.
    """
    if recipe.batch_size != 64 or recipe.steps != 3:
        raise ValueError("resource audit requires selected batch 64 plus current and three inner steps")
    if len(records) < recipe.batch_size + 1:
        raise ValueError("resource audit needs 65 cached training records")
    selected = copy.deepcopy(records[:recipe.batch_size + 1])
    for i, record in enumerate(selected):
        record["arrival"] = i * recipe.capacity // recipe.batch_size
    acquired = recipe.capacity + 1
    strength = recipe.fd_strength or FD_STRENGTH
    timings, directions = [], {}
    numerical_errors = (FloatingPointError, ValueError, torch.linalg.LinAlgError)
    for rank in RANKS:
        for fd in (0., strength):
            cfg = replace(recipe, rank=rank, fd_strength=fd)
            method = "td_fd" if fd else "td_replay"
            try:
                row, direction = _resource_row(fitted, selected, cfg, archives, acquired, method, check)
                directions[f"rank{rank}:{method}"] = direction.clone()
            except numerical_errors as error:
                row = {"rank": rank, "fd_strength": fd, "method": method,
                       "fit_seconds": 0., "timing_available": False,
                       "health": {"healthy": False}, "error": f"{type(error).__name__}: {error}"}
            timings.append(row)
            check()
    try:
        compression = _compression_audit(fitted, records, recipe, directions, check)
        _finite_scalars(compression)
        compression["healthy"] = True
    except numerical_errors as error:
        compression = {"healthy": False, "error": f"{type(error).__name__}: {error}"}
    return {"timings": timings, "compression_audit": compression,
            "healthy": compression["healthy"] and all(row["health"]["healthy"] for row in timings),
            "healthy_ranks": [rank for rank in RANKS if compression["healthy"]
                              and all(row["health"]["healthy"] for row in timings if row["rank"] == rank)],
            "audit_seed": AUDIT_SEED, "device": "cpu", "warmups": 1, "repeats": REPEATS,
            "batch_records": len(selected), "steps": recipe.steps, "new_observations": 0,
            "timing_scope": "transfer-inclusive CPU fit; isolated components, cold solve setup included",
            "compression_method": "unchanged Fix 5 streaming thin SVD, not dense recompression"}
