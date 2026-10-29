"""Immutable nine-cell source plans and exact Phase 10 cellwise rendering.

Each row is one grid. IDs/digits are (N,9), forward-affine draws (N,9,6),
and labels (N,54) ordered by cell, anchor, then horizontal/vertical component.
Sampling sources uniformly without replacement gives empirical class marginals;
conditional class counts deplete, rather than imposing independent class draws.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from mnist_experiment.fisher_memory.data import MNISTSource, _choose
from mnist_experiment.fisher_memory_registration import data as cell_data
from mnist_experiment.fisher_memory_registration.data import (
    ANCHORS, BLUR_RADIUS, PARAMETERS, SUPPORT_CORNERS, affine_from_parameters,
    drift_strength, encode_affine, mixture_probability, sample_transforms, support_mask,
)

from . import PROTOCOL


SCHEMA_VERSION = 1
DOMAINS = ("base", "novel")
GRID_SHAPE = (3, 3)
CELL_SIZE = 28
CELLS = 9
OUTPUT_DIM = 54
TRANSFORM_CHUNK = 32  # Grids; inherited render/blur helpers also chunk cells.
TRAIN_ROLES = ("initialization", "validation", "reference", "diagnostic_base",
               "diagnostic_novel", "diagnostic_reserve", "stream")
CELL_OFFSETS = tuple((28 * column, 28 * row) for row in range(3) for column in range(3))


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def assemble_grid(cells: Tensor) -> Tensor:
    """Assemble (N,9,C,28,28) cells in row-major order without interpolation."""
    if cells.ndim != 5 or cells.shape[1] != 9 or tuple(cells.shape[-2:]) != (28, 28):
        raise ValueError("cells must have shape (N,9,C,28,28)")
    count, _, channels, _, _ = cells.shape
    return cells.reshape(count, 3, 3, channels, 28, 28).permute(
        0, 3, 1, 4, 2, 5).reshape(count, channels, 84, 84)


def split_grid(images: Tensor) -> Tensor:
    """Inverse assembly, retaining row-major cells and channel correspondence."""
    if images.ndim != 4 or tuple(images.shape[-2:]) != (84, 84):
        raise ValueError("grid images must have shape (N,C,84,84)")
    count, channels = images.shape[:2]
    return images.reshape(count, channels, 3, 28, 3, 28).permute(
        0, 2, 4, 1, 3, 5).reshape(count, 9, channels, 28, 28)


def target_tensor(parameters: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Encode (...,9,6) forward-affine draws as (...,54) cell-local pixels."""
    values = torch.as_tensor(parameters, dtype=torch.float64, device="cpu")
    if values.ndim < 2 or tuple(values.shape[-2:]) != (9, 6):
        raise ValueError("grid transforms must end in shape (9,6)")
    return encode_affine(*affine_from_parameters(values)).reshape(*values.shape[:-2], 54).to(dtype)


def decode_affine(targets: Any) -> tuple[Tensor, Tensor]:
    """Decode 54-output grids or six-output cells without clipping predictions."""
    values = targets if isinstance(targets, Tensor) else torch.as_tensor(targets, dtype=torch.float64)
    if values.ndim and values.shape[-1] == 54:
        values = values.reshape(*values.shape[:-1], 9, 6)
    return cell_data.decode_affine(values)


def geometry_metrics(prediction: Tensor, targets: Tensor, *, weights: Tensor | None = None,
                     digits: Tensor | None = None) -> dict[str, Any]:
    """Grid-mean square-pixel risk per landmark, plus cell/digit breakdowns.

    No class poststratification is applied: grids have equal sampling weight.
    Per-digit summaries condition on the observed cells, not replica units.
    """
    if (prediction.ndim != 2 or prediction.shape != targets.shape
            or prediction.shape[1] != 54 or not len(prediction)
            or not prediction.is_floating_point() or not targets.is_floating_point()
            or prediction.device != targets.device):
        raise ValueError("geometry requires matching nonempty (N,54) floating predictions/targets")
    residual = (prediction.detach().double() - targets.detach().double()).cpu()
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("nonfinite grid geometry")
    count = len(residual)
    weights = (torch.full((count,), 1 / count, dtype=torch.float64) if weights is None
               else torch.as_tensor(weights, dtype=torch.float64, device="cpu"))
    if (weights.shape != (count,) or not torch.isfinite(weights).all()
            or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.))):
        raise ValueError("weights must be nonnegative finite grid weights summing to one")
    coordinate = residual.square().reshape(count, 9, 3, 2)
    squared = coordinate.sum(-1)
    distances = squared.sqrt()
    cell_risk = squared.mean(-1)
    mse = float((weights * cell_risk.mean(-1)).sum())
    per_digit = {}
    if digits is not None:
        digits = torch.as_tensor(digits, device="cpu")
        if (digits.shape != (count, 9) or digits.dtype not in (torch.int32, torch.int64)
                or ((digits < 0) | (digits > 9)).any()):
            raise ValueError("digits must be an integer (N,9) matrix in [0,9]")
        cell_weights = weights[:, None].expand(-1, 9)
        for digit in range(10):
            take = digits == digit
            mass = cell_weights[take].sum()
            if mass > 0:
                per_digit[str(digit)] = float((cell_weights[take] * cell_risk[take]).sum() / mass)
    coordinate_mse = (weights[:, None] * residual.square()).sum(0)
    return dict(mse=mse, rmse=math.sqrt(mse),
                mae=float((weights * distances.mean((1, 2))).sum()),
                accuracy1=float((weights * (distances <= 1).double().mean((1, 2))).sum()),
                common_mse=float(coordinate_mse[0]), coordinate_mse=coordinate_mse.tolist(),
                per_cell_mse=(weights[:, None] * cell_risk).sum(0).tolist(),
                per_anchor_mse=(weights[:, None, None] * squared).sum(0).tolist(),
                per_digit_mse=per_digit)


def _grid_transforms(count: int, seed: int) -> tuple[list, dict]:
    parameters, diagnostics = sample_transforms(9 * count, seed)
    parameters = parameters.reshape(count, 9, 6)
    targets = target_tensor(parameters, dtype=torch.float64)
    mean = targets.mean(0) if count else torch.zeros(54, dtype=torch.float64)
    centered = targets - mean
    return parameters.tolist(), dict(diagnostics,
        accepted_grids=count, accepted_cells=9 * count,
        target_mean=mean.tolist(),
        target_covariance=(centered.T @ centered / max(count - 1, 1)).tolist(),
        covariance_ddof=1)


def plan_data(source: MNISTSource, cfg: Any, family: str, replica: int,
              partition: str) -> dict[str, Any]:
    """Freeze all disjoint roles before branch draws; never redraw to fit a pool.

    Uniform sequential source selection equals choosing classes proportional to
    remaining source counts. Unconditional marginals are the fixed empirical
    source proportions; draws are not independent in this finite population.
    Pool exhaustion returns the original incomplete trajectory, never a retry.
    """
    if partition not in ("development", "confirmation"):
        raise ValueError("partition must be development or confirmation")
    for name in ("init_size", "init_validation_size", "reference_size", "reference_max_size",
                 "max_init_epochs", "max_diagnostic_epochs", "eval_size", "novel_count",
                 "nominal_batch", "ramp_groups", "diagnostic_base_size", "diagnostic_novel_size",
                 "diagnostic_reserve_size"):
        _positive_integer(getattr(cfg, name), name)
    if cfg.reference_size > cfg.reference_max_size or cfg.nominal_batch > 4:
        raise ValueError("invalid reference reserve or nominal group size")
    mixture_probability(1, cfg.novel_probability, cfg.ramp_groups)
    family = family.split("--", 1)[0]
    labels = torch.as_tensor(source.train_labels, dtype=torch.long, device="cpu")
    test_labels = torch.as_tensor(source.test_labels, dtype=torch.long, device="cpu")
    for values in (labels, test_labels):
        if values.ndim != 1 or not len(values) or bool(((values < 0) | (values > 9)).any()):
            raise ValueError("source labels must be nonempty digit vectors in [0,9]")
    frequencies = torch.bincount(labels, minlength=10).double()
    if bool((frequencies == 0).any()):
        raise ValueError("all ten training digits are required")
    probabilities = frequencies / frequencies.sum()
    names = ("initialization_data", "validation_data", "reference", "stream_sources",
             "stream_domains", "diagnostic_base_data", "diagnostic_novel_data",
             "diagnostic_reserve_data", "evaluation_sources", "evaluation_strengths",
             "diagnostic_novel_strengths", "diagnostic_validation_strengths",
             "initialization_model", "initialization_order", "factors", "replay", "probes",
             "compression", "bootstrap",
             *(f"{role}_transforms" for role in TRAIN_ROLES),
             *(f"{role}_cell_positions" for role in TRAIN_ROLES),
             *(f"evaluation_{domain}_transforms" for domain in DOMAINS),
             *(f"evaluation_{domain}_cell_positions" for domain in DOMAINS))
    seeds = {name: cfg.seed_for(family, replica, name) for name in names}
    seeds["epochs"] = [cfg.seed_for(family, replica, f"epoch:{i}") for i in range(cfg.max_init_epochs)]
    seeds["diagnostic_epochs"] = [cfg.seed_for(family, replica, f"diagnostic_order:{i}")
                                  for i in range(cfg.max_diagnostic_epochs)]
    seeds["evaluation_partition"] = cfg.seed_for("evaluation-partition", 0, "strata")

    def generator(name):
        return torch.Generator().manual_seed(seeds[name])

    def position(ids, name):
        rows = torch.tensor(ids, dtype=torch.long).reshape(-1, 9)
        order = torch.rand(rows.shape, generator=generator(name), dtype=torch.float64).argsort(1)
        return rows.gather(1, order).tolist()

    role_sizes = dict(initialization=cfg.init_size, validation=cfg.init_validation_size,
                      reference=cfg.reference_max_size, diagnostic_base=cfg.diagnostic_base_size,
                      diagnostic_novel=cfg.diagnostic_novel_size,
                      diagnostic_reserve=cfg.diagnostic_reserve_size)
    if 9 * sum(role_sizes.values()) > len(labels):
        raise ValueError("insufficient disjoint training sources for reserved grid roles")
    available = torch.ones(len(labels), dtype=torch.bool)
    role_ids = {}
    for role, size in role_sizes.items():
        seed_name = {"initialization": "initialization_data", "validation": "validation_data",
                     "reference": "reference"}.get(role, f"{role}_data")
        ids = _choose(torch.where(available)[0], 9 * size, generator(seed_name))
        available[ids] = False
        role_ids[role] = position(ids, f"{role}_cell_positions")
    maximum = getattr(cfg, "stream_max_size", 2986)
    _positive_integer(maximum, "stream_max_size")
    stream_pool_size = min(maximum, int(available.sum()) // 9)
    ids = _choose(torch.where(available)[0], 9 * stream_pool_size, generator("stream_sources"))
    stream_pool = position(ids, "stream_cell_positions")
    flags, groups, strengths, segments = [], [], [], []
    domains = generator("stream_domains")
    novel_count, group = 0, 0
    while len(flags) < stream_pool_size and novel_count < cfg.novel_count:
        group += 1
        probability = mixture_probability(group, cfg.novel_probability, cfg.ramp_groups)
        branch = (torch.rand(cfg.nominal_batch, generator=domains) < probability).tolist()
        start = len(flags)
        for novel in branch[:stream_pool_size - start]:
            flags.append(novel)
            groups.append(group)
            strengths.append(drift_strength(group) if novel else 0.)
            novel_count += int(novel)
            if novel:
                segments.append(dict(start=start, end=len(flags), group=group, probability=probability,
                                     strength=drift_strength(group), novel_count=novel_count))
                start = len(flags)
            if novel_count == cfg.novel_count:
                break
        if start < len(flags):
            segments.append(dict(start=start, end=len(flags), group=group, probability=probability,
                                 strength=drift_strength(group), novel_count=novel_count))
    role_ids["stream"] = stream_pool[:len(flags)]

    partition_ids = []
    partition_generator = generator("evaluation_partition")
    for digit in range(10):
        ids = torch.where(test_labels == digit)[0]
        ids = ids[torch.randperm(len(ids), generator=partition_generator)]
        cut = len(ids) // 2
        partition_ids.extend((ids[:cut] if partition == "development" else ids[cut:]).tolist())
    panel_ids = _choose(torch.tensor(partition_ids, dtype=torch.long), 18 * cfg.eval_size,
                        generator("evaluation_sources"))
    evaluation_ids = {domain: position(panel_ids[i * 9 * cfg.eval_size:(i + 1) * 9 * cfg.eval_size],
                                      f"evaluation_{domain}_cell_positions")
                      for i, domain in enumerate(DOMAINS)}
    complete = novel_count == cfg.novel_count
    plan = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, family=family, replica=replica,
                partition=partition, seeds=seeds, class_probabilities=probabilities.tolist(),
                source_sampling=dict(law="uniform_without_replacement",
                    class_draw="remaining_source_count_weighted", class_draws_independent=False,
                    marginal_class_law="fixed_empirical_source_split",
                    training_source_counts=frequencies.long().tolist(),
                    test_partition_source_counts=torch.bincount(test_labels[partition_ids], minlength=10).tolist(),
                    cross_replica_source_reuse=True, panel_weighting="uniform_over_grids"),
                stream_pool_ids=stream_pool, stream_pool_size=stream_pool_size,
                stream_complete=complete, stream_status="complete" if complete else "source_pool_exhausted",
                stream_novel_count=novel_count, stream_target_novel_count=cfg.novel_count,
                stream_novel=flags, stream_groups=groups, stream_strengths=strengths, segments=segments,
                reference_max_size=cfg.reference_max_size, evaluation_ids=evaluation_ids,
                evaluation_strengths=dict(base=[0.] * cfg.eval_size,
                    novel=cell_data._uniform_strengths(cfg.eval_size, seeds["evaluation_strengths"])),
                diagnostic_novel_strengths=cell_data._uniform_strengths(cfg.diagnostic_novel_size,
                    seeds["diagnostic_novel_strengths"]),
                diagnostic_validation_strengths=cell_data._uniform_strengths(cfg.init_validation_size,
                    seeds["diagnostic_validation_strengths"]),
                rendering=dict(grid_shape=[3, 3], input_shape=[2, 84, 84], cell_shape=[28, 28],
                    cell_offsets=[list(offset) for offset in CELL_OFFSETS],
                    cell_order="row-major", target_shape=[54],
                    target_order=[f"cell{j}_{name}" for j in range(9)
                                  for name in ("d1x", "d1y", "d2x", "d2y", "d3x", "d3y")],
                    parameter_order=list(PARAMETERS), angle_units="degrees",
                    forward_map="R(alpha) diag(exp(a_x),exp(a_y)) shear(h) z + t",
                    proposal_half_widths=[20., 2., 2., math.log(1.1), math.log(1.1), .1],
                    anchors=[list(anchor) for anchor in ANCHORS], target_units="cell-local pixels",
                    positive_angles="clockwise", canvas_center=[13.5, 13.5],
                    resized_shape=[18, 18], resize_mode="bilinear", antialias=True,
                    support_corners=[list(corner) for corner in SUPPORT_CORNERS], support_bound=13.,
                    query_sampling="inverse-map-bilinear", align_corners=False, padding="zeros",
                    blur_radius=BLUR_RADIUS, blur_channel=1, blur_scope="per-cell-before-assembly",
                    branch_scope="whole-grid", mixture_ramp="smoothstep", strength_groups=200),
                transform_diagnostics={})
    for role in TRAIN_ROLES:
        ids = role_ids[role]
        parameters, diagnostics = _grid_transforms(len(ids), seeds[f"{role}_transforms"])
        plan[f"{role}_ids"], plan[f"{role}_transforms"] = ids, parameters
        indices = torch.tensor(ids, dtype=torch.long).reshape(-1, 9)
        plan[f"{role}_labels"] = labels[indices].tolist()
        plan["transform_diagnostics"][role] = diagnostics
    plan["evaluation_transforms"] = {}
    for domain in DOMAINS:
        parameters, diagnostics = _grid_transforms(cfg.eval_size, seeds[f"evaluation_{domain}_transforms"])
        plan["evaluation_transforms"][domain] = parameters
        plan["transform_diagnostics"][f"evaluation_{domain}"] = diagnostics
    return plan


def render_pairs(images: Tensor, matrix: Any, translation: Any,
                 check_stop: Callable[[], None] | None = None) -> Tensor:
    """Render (N,9,1,28,28) source cells with the unmodified Phase 10 helper."""
    if check_stop is not None:
        check_stop()
    if images.ndim != 5 or tuple(images.shape[1:]) != (9, 1, 28, 28):
        raise ValueError("grid sources must have shape (N,9,1,28,28)")
    matrix = torch.as_tensor(matrix, dtype=torch.float64, device="cpu")
    translation = torch.as_tensor(translation, dtype=torch.float64, device="cpu")
    if matrix.shape != (len(images), 9, 2, 2) or translation.shape != (len(images), 9, 2):
        raise ValueError("one affine map is required per grid cell")
    pairs = cell_data.render_pairs(images.reshape(-1, 1, 28, 28), matrix.reshape(-1, 2, 2),
                                  translation.reshape(-1, 2), check_stop)
    return assemble_grid(pairs.reshape(len(images), 9, 2, 28, 28))


def blur_pairs(images: Tensor, strengths: Any, blur_max: float,
               check_stop: Callable[[], None] | None = None) -> Tensor:
    """Blur queries independently within cells, with one shared severity per grid."""
    if check_stop is not None:
        check_stop()
    if images.ndim != 4 or tuple(images.shape[1:]) != (2, 84, 84):
        raise ValueError("images must have shape (N,2,84,84)")
    strengths = torch.as_tensor(strengths, dtype=torch.float64, device="cpu")
    if strengths.ndim == 0:
        strengths = strengths.expand(len(images))
    if strengths.shape != (len(images),):
        raise ValueError("strengths must have one value per grid")
    cells = split_grid(images).reshape(-1, 2, 28, 28)
    blurred = cell_data.blur_pairs(cells, strengths.repeat_interleave(9), blur_max, check_stop)
    return images if blurred is cells else assemble_grid(blurred.reshape(len(images), 9, 2, 28, 28))


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("protocol") != PROTOCOL:
        raise ValueError("unsupported grid registration data protocol/schema")
    if plan.get("partition") not in ("development", "confirmation"):
        raise ValueError("invalid grid registration data partition")


def contextualize(base_data: dict[str, Any], cfg: Any, blur_max: float,
                  check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Reuse nominal tensors/targets; change only whole-grid novel queries."""
    if check_stop is not None:
        check_stop()
    plan = base_data["plan"]
    _validate_plan(plan)
    if base_data["blur_max"] != 0:
        raise ValueError("contextualize requires a nominal cache")
    if not math.isfinite(blur_max) or blur_max < 0:
        raise ValueError("blur_max must be finite and nonnegative")
    result = dict(base_data, blur_max=float(blur_max))
    if blur_max == 0:
        return result
    flags = torch.tensor(plan["stream_novel"], dtype=torch.bool)
    strengths = torch.tensor(plan["stream_strengths"], dtype=torch.float64)
    started = time.perf_counter()
    stream = base_data["stream_x"].clone()
    stream[flags] = blur_pairs(stream[flags], strengths[flags], blur_max, check_stop)
    result["stream_x"] = stream
    result["stream_blur_seconds"] = time.perf_counter() - started
    panels = dict(base_data["panels"])
    panels["novel"] = dict(panels["novel"], x=blur_pairs(panels["novel"]["x"],
        plan["evaluation_strengths"]["novel"], blur_max, check_stop))
    result["panels"] = panels
    result["diagnostic_novel_x"] = blur_pairs(base_data["diagnostic_novel_x"],
        plan["diagnostic_novel_strengths"], blur_max, check_stop)
    return result


def materialize(source: MNISTSource, cfg: Any, plan: dict[str, Any], blur_max: float = 0.,
                check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Render frozen CPU float32 grids/targets without downloads, writes or resampling."""
    if check_stop is not None:
        check_stop()
    _validate_plan(plan)
    if not math.isfinite(blur_max) or blur_max < 0:
        raise ValueError("blur_max must be finite and nonnegative")

    def render(ids, parameters, *, test=False):
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1, 9)
        parameters = torch.tensor(parameters, dtype=torch.float64).reshape(len(ids), 9, 6)
        output = torch.empty(len(ids), 2, 84, 84, dtype=torch.float32)
        for start in range(0, len(ids), TRANSFORM_CHUNK):
            if check_stop is not None:
                check_stop()
            selected = ids[start:start + TRANSFORM_CHUNK]
            raw = source.images(selected.reshape(-1).tolist(), test=test).to(device="cpu", dtype=torch.float32)
            raw = raw.reshape(len(selected), 9, 1, 28, 28)
            output[start:start + len(selected)] = render_pairs(raw,
                *affine_from_parameters(parameters[start:start + len(selected)]), check_stop)
        return output, target_tensor(parameters)

    result = dict(plan=plan, blur_max=0., render_seconds={})
    for role in TRAIN_ROLES:
        started = time.perf_counter()
        x, y = render(plan[f"{role}_ids"], plan[f"{role}_transforms"])
        result["render_seconds"][role] = time.perf_counter() - started
        result[f"{role}_x"], result[f"{role}_y"] = x, y
    result["pretrain_x"], result["pretrain_y"] = result["initialization_x"], result["initialization_y"]
    panels = {}
    for domain in DOMAINS:
        ids = torch.tensor(plan["evaluation_ids"][domain], dtype=torch.long).reshape(-1, 9)
        x, y = render(ids, plan["evaluation_transforms"][domain], test=True)
        panels[domain] = dict(x=x, y=y,
            digits=torch.as_tensor(source.test_labels, dtype=torch.long)[ids].clone(),
            weights=torch.full((len(ids),), 1 / len(ids), dtype=torch.float64))
    result["panels"] = panels
    if check_stop is not None:
        check_stop()
    return contextualize(result, cfg, blur_max, check_stop) if blur_max else result
