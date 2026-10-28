"""Frozen affine-registration plans and deterministic CPU pair rendering.

Training-role IDs index the official training split; panel IDs index the test
split. Targets are six landmark displacements in physical pixels, for both
tasks. Neither task/head settings nor blur severity enter the sampling plan.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from mnist_experiment.fisher_memory.data import MNISTSource, _choose, class_weights

from . import PROTOCOL


SCHEMA_VERSION = 1
DOMAINS = ("base", "novel")
ANCHORS = ((-6., -4.), (6., -4.), (0., 8.))
SUPPORT_CORNERS = ((-9.5, -9.5), (-9.5, 9.5), (9.5, -9.5), (9.5, 9.5))
PARAMETERS = ("alpha_degrees", "t_x", "t_y", "a_x", "a_y", "h")
TRANSFORM_CHUNK = 256
PROPOSAL_CHUNK = 1024
BLUR_RADIUS = 6
TRAIN_ROLES = ("initialization", "validation", "reference", "stream",
               "diagnostic_base", "diagnostic_novel")


def _coordinates(value: Any, shape: tuple[int, ...], name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float64)
    if (tensor.device.type != "cpu" or tensor.dtype not in (torch.float32, torch.float64)
            or tensor.ndim < len(shape) or tuple(tensor.shape[-len(shape):]) != shape
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{name} must be finite CPU floating tensors ending in shape {shape}")
    return tensor


def affine_from_parameters(parameters: Any) -> tuple[Tensor, Tensor]:
    """Return forward (A,t) for (...,6) draws in the recorded PARAMETERS order.

    A = R(alpha) diag(exp(a_x), exp(a_y)) [[1,h],[0,1]]. Pixel y
    increases downward, so a positive angle rotates clockwise on screen.
    """
    parameters = _coordinates(parameters, (6,), "parameters")
    alpha, tx, ty, ax, ay, shear = parameters.unbind(-1)
    radians = alpha * (math.pi / 180.)
    cosine, sine = radians.cos(), radians.sin()
    sx, sy = ax.exp(), ay.exp()
    matrix = torch.stack((cosine * sx, cosine * sx * shear - sine * sy,
                          sine * sx, sine * sx * shear + cosine * sy), dim=-1)
    matrix = matrix.reshape(*parameters.shape[:-1], 2, 2)
    translation = torch.stack((tx, ty), dim=-1)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("affine parameters overflowed")
    return matrix, translation


def _affine(matrix: Any, translation: Any) -> tuple[Tensor, Tensor]:
    matrix = _coordinates(matrix, (2, 2), "matrix")
    translation = _coordinates(translation, (2,), "translation").to(matrix.dtype)
    if matrix.shape[:-2] != translation.shape[:-1]:
        raise ValueError("matrix and translation batch shapes must match")
    return matrix, translation


def encode_affine(matrix: Any, translation: Any) -> Tensor:
    """Encode any forward affine map as (d1x,d1y,d2x,d2y,d3x,d3y)."""
    matrix, translation = _affine(matrix, translation)
    anchors = matrix.new_tensor(ANCHORS)
    delta = anchors @ (matrix - torch.eye(2, dtype=matrix.dtype)).transpose(-1, -2)
    return (delta + translation.unsqueeze(-2)).reshape(*matrix.shape[:-2], 6)


def decode_affine(targets: Any) -> tuple[Tensor, Tensor]:
    """Decode arbitrary finite predictions, without range/invertibility projection."""
    targets = _coordinates(targets, (6,), "targets")
    delta = targets.reshape(*targets.shape[:-1], 3, 2)
    # The anchors sum to zero, and their two independent differences span R^2.
    translation = delta.mean(dim=-2)
    first = (delta[..., 1, :] - delta[..., 0, :]) / 12.
    second = (delta[..., 2, :] - .5 * (delta[..., 0, :] + delta[..., 1, :])) / 12.
    matrix = torch.stack((first, second), dim=-1) + torch.eye(2, dtype=targets.dtype)
    return matrix, translation


def support_mask(matrix: Any, translation: Any) -> Tensor:
    """Accept iff the full bilinear support footprint stays within [-13,13]^2."""
    matrix, translation = _affine(matrix, translation)
    corners = matrix.new_tensor(SUPPORT_CORNERS)
    mapped = corners @ matrix.transpose(-1, -2) + translation.unsqueeze(-2)
    return (mapped.abs() <= 13.).all(dim=-1).all(dim=-1)


def _target_summary(targets: Tensor) -> dict[str, Any]:
    mean = targets.mean(0) if len(targets) else targets.new_zeros(6)
    centered = targets - mean
    covariance = centered.T @ centered / max(len(targets) - 1, 1)
    return dict(target_mean=mean.tolist(), target_covariance=covariance.tolist(),
                covariance_ddof=1)


def sample_transforms(count: int, seed: int) -> tuple[Tensor, dict[str, Any]]:
    """Sample prefix-stable accepted float64 parameters and proposal diagnostics.

    Rejections count proposals through the last retained draw, not unused draws
    in the fixed-size RNG buffer. No digit identity, pixels, or global RNG is used.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    limits = torch.tensor((20., 2., 2., math.log(1.1), math.log(1.1), .1), dtype=torch.float64)
    parameters = torch.empty(count, 6, dtype=torch.float64)
    offset, rejected = 0, 0
    while offset < count:
        candidates = (2 * torch.rand(PROPOSAL_CHUNK, 6, generator=generator,
                                     dtype=torch.float64) - 1) * limits
        matrix, translation = affine_from_parameters(candidates)
        indices = torch.where(support_mask(matrix, translation))[0][:count - offset]
        considered = int(indices[-1]) + 1 if len(indices) == count - offset else len(candidates)
        rejected += considered - len(indices)
        parameters[offset:offset + len(indices)] = candidates[indices]
        offset += len(indices)
    targets = encode_affine(*affine_from_parameters(parameters))
    diagnostics = dict(accepted=count, rejected=rejected, proposed=count + rejected,
                       **_target_summary(targets))
    return parameters, diagnostics


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _smoothstep(value: float) -> float:
    return 3 * value * value - 2 * value * value * value


def mixture_probability(group: int, novel_probability: float = .1, ramp_groups: int = 40) -> float:
    """C1 mixture ramp on original one-based arrival groups (not split segments)."""
    _positive_integer(group, "group")
    _positive_integer(ramp_groups, "ramp_groups")
    if ramp_groups < 2 or not math.isfinite(novel_probability) or not 0 < novel_probability <= 1:
        raise ValueError("invalid mixture ramp")
    return novel_probability * _smoothstep(min((group - 1) / (ramp_groups - 1), 1.))


def drift_strength(group: int) -> float:
    """C1 query-blur severity: .5 at original group 1, one at group 200."""
    _positive_integer(group, "group")
    return .5 + .5 * _smoothstep(min((group - 1) / 199., 1.))


def _uniform_strengths(count: int, seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    return (.5 + .5 * torch.rand(count, generator=generator, dtype=torch.float64)).tolist()


def plan_data(source: MNISTSource, cfg: Any, family: str, replica: int,
              partition: str) -> dict[str, Any]:
    """Freeze disjoint source roles, forward transforms, and exact novel events.

    Uses only data/seed fields of cfg; scalar/vector heads, initialization epoch
    choices and usable reference prefixes cannot change the plan. All training
    roles are disjoint, including the full reference_max_size reserve. Validation
    strengths support diagnostic epoch selection without test-panel feedback.
    """
    if partition not in ("development", "confirmation"):
        raise ValueError("partition must be development or confirmation")
    for name in ("init_size", "init_validation_size", "reference_size", "reference_max_size",
                 "max_init_epochs", "max_diagnostic_epochs", "eval_per_class", "novel_count",
                 "nominal_batch", "ramp_groups", "diagnostic_base_size", "diagnostic_novel_size"):
        _positive_integer(getattr(cfg, name), name)
    if cfg.reference_size > cfg.reference_max_size or cfg.nominal_batch > 4:
        raise ValueError("invalid reference reserve or nominal group size")
    mixture_probability(1, cfg.novel_probability, cfg.ramp_groups)
    family = family.split("--", 1)[0]
    labels = torch.as_tensor(source.train_labels, dtype=torch.long, device="cpu")
    test_labels = torch.as_tensor(source.test_labels, dtype=torch.long, device="cpu")
    for values in (labels, test_labels):
        if values.ndim != 1 or not len(values) or bool(((values < 0) | (values > 9)).any()):
            raise ValueError("source labels must be nonempty vectors of digits 0 through 9")
    frequencies = torch.bincount(labels, minlength=10).double()
    if bool((frequencies == 0).any()):
        raise ValueError("all ten training digits are required")
    probabilities = frequencies / frequencies.sum()
    names = ("initialization_data", "validation_data", "reference", "stream_sources",
             "stream_digits", "stream_domains", "diagnostic_base_data", "diagnostic_novel_data",
             "evaluation_transforms", "evaluation_strengths", "diagnostic_novel_strengths",
             "diagnostic_validation_strengths", "initialization_model", "initialization_order",
             "factors", "replay", "probes", "compression",
             *(f"{role}_transforms" for role in TRAIN_ROLES))
    seeds = {name: cfg.seed_for(family, replica, name) for name in names}
    seeds["epochs"] = [cfg.seed_for(family, replica, f"epoch:{i}") for i in range(cfg.max_init_epochs)]
    seeds["diagnostic_epochs"] = [cfg.seed_for(family, replica, f"diagnostic_order:{i}")
                                  for i in range(cfg.max_diagnostic_epochs)]
    seeds["evaluation_partition"] = cfg.seed_for("evaluation-partition", 0, "strata")
    seeds["evaluation_classes"] = [cfg.seed_for(family, replica, f"evaluation:{k}") for k in range(10)]

    def generator(name):
        return torch.Generator().manual_seed(seeds[name])

    available = torch.ones(len(labels), dtype=torch.bool)
    role_ids = {}
    for role, count, seed_name in (("initialization", cfg.init_size, "initialization_data"),
                                   ("validation", cfg.init_validation_size, "validation_data"),
                                   ("reference", cfg.reference_max_size, "reference"),
                                   ("diagnostic_base", cfg.diagnostic_base_size, "diagnostic_base_data"),
                                   ("diagnostic_novel", cfg.diagnostic_novel_size, "diagnostic_novel_data")):
        role_ids[role] = _choose(torch.where(available)[0], count, generator(seed_name))
        available[role_ids[role]] = False
    stream_generator = generator("stream_sources")
    buckets = [_choose(torch.where(available & (labels == digit))[0],
                       int((available & (labels == digit)).sum()), stream_generator) for digit in range(10)]
    offsets = [0] * 10
    digit_generator, domain_generator = generator("stream_digits"), generator("stream_domains")
    stream_ids, stream_labels, stream_novel, groups, strengths, segments = [], [], [], [], [], []
    novel_count, group = 0, 0
    while novel_count < cfg.novel_count:
        group += 1
        probability = mixture_probability(group, cfg.novel_probability, cfg.ramp_groups)
        digits = torch.multinomial(probabilities, cfg.nominal_batch, replacement=True,
                                   generator=digit_generator).tolist()
        flags = (torch.rand(cfg.nominal_batch, generator=domain_generator) < probability).tolist()
        start = len(stream_ids)
        for digit, novel in zip(digits, flags, strict=True):
            if offsets[digit] >= len(buckets[digit]):
                raise ValueError("stream exhausted a digit; refusing duplicate source exposures")
            stream_ids.append(buckets[digit][offsets[digit]])
            offsets[digit] += 1
            stream_labels.append(digit)
            stream_novel.append(novel)
            groups.append(group)
            strengths.append(drift_strength(group) if novel else 0.)
            novel_count += int(novel)
            if novel:
                segments.append(dict(start=start, end=len(stream_ids), group=group, probability=probability,
                                     strength=drift_strength(group), novel_count=novel_count))
                start = len(stream_ids)
            if novel_count == cfg.novel_count:
                break
        if start < len(stream_ids):
            segments.append(dict(start=start, end=len(stream_ids), group=group, probability=probability,
                                 strength=drift_strength(group), novel_count=novel_count))
    role_ids["stream"] = stream_ids

    evaluation_ids = []
    partition_generator = generator("evaluation_partition")
    for digit in range(10):
        ids = torch.where(test_labels == digit)[0]
        ids = ids[torch.randperm(len(ids), generator=partition_generator)]
        cut = len(ids) // 2
        ids = ids[:cut] if partition == "development" else ids[cut:]
        evaluation_ids.extend(_choose(ids, cfg.eval_per_class,
                              torch.Generator().manual_seed(seeds["evaluation_classes"][digit])))

    plan = dict(schema_version=SCHEMA_VERSION, protocol=PROTOCOL, family=family,
                replica=replica, partition=partition, seeds=seeds,
                class_probabilities=probabilities.tolist(), stream_labels=stream_labels,
                stream_novel=stream_novel, stream_groups=groups, stream_strengths=strengths,
                segments=segments, reference_max_size=cfg.reference_max_size,
                evaluation_ids={domain: list(evaluation_ids) for domain in DOMAINS},
                evaluation_strengths=dict(base=[0.] * len(evaluation_ids),
                    novel=_uniform_strengths(len(evaluation_ids), seeds["evaluation_strengths"])),
                diagnostic_novel_strengths=_uniform_strengths(cfg.diagnostic_novel_size,
                    seeds["diagnostic_novel_strengths"]),
                diagnostic_validation_strengths=_uniform_strengths(cfg.init_validation_size,
                    seeds["diagnostic_validation_strengths"]),
                rendering=dict(parameter_order=list(PARAMETERS), angle_units="degrees",
                    forward_map="R(alpha) diag(exp(a_x),exp(a_y)) shear(h) z + t",
                    proposal_half_widths=[20., 2., 2., math.log(1.1), math.log(1.1), .1],
                    anchors=[list(anchor) for anchor in ANCHORS], target_units="pixels",
                    target_order=["d1x", "d1y", "d2x", "d2y", "d3x", "d3y"],
                    positive_angles="clockwise", canvas_center=[13.5, 13.5],
                    resized_shape=[18, 18], resize_mode="bilinear", antialias=True,
                    support_corners=[list(corner) for corner in SUPPORT_CORNERS], support_bound=13.,
                    query_sampling="inverse-map-bilinear", align_corners=False,
                    padding="zeros", blur_radius=BLUR_RADIUS, blur_channel=1,
                    mixture_ramp="smoothstep", strength_groups=200), transform_diagnostics={})
    for role in TRAIN_ROLES:
        ids = role_ids[role]
        parameters, diagnostics = sample_transforms(len(ids), seeds[f"{role}_transforms"])
        plan[f"{role}_ids"] = ids
        plan[f"{role}_transforms"] = parameters.tolist()
        plan["transform_diagnostics"][role] = diagnostics
    parameters, diagnostics = sample_transforms(len(evaluation_ids), seeds["evaluation_transforms"])
    plan["evaluation_transforms"] = {domain: parameters.tolist() for domain in DOMAINS}
    plan["transform_diagnostics"]["evaluation"] = diagnostics
    return plan


def _images(images: Tensor, channels: int) -> None:
    if (images.device.type != "cpu" or images.ndim != 4
            or images.shape[1:] != (channels, 28, 28)
            or images.dtype not in (torch.float32, torch.float64)):
        raise ValueError(f"images must be CPU float32/float64 tensors of shape (N,{channels},28,28)")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("images must contain finite intensities in [0,1]")


def render_pairs(images: Tensor, matrix: Any, translation: Any,
                 check_stop: Callable[[], None] | None = None) -> Tensor:
    """Center antialiased 18x18 references and inverse-sample forward affine queries.

    Channel zero is the reference, channel one the query. For align_corners=False
    the centered pixel coordinate q becomes q/14 in the sampling grid. Targets
    describe the forward map, not this inverse grid. No support projection occurs.
    """
    if check_stop is not None:
        check_stop()
    _images(images, 1)
    matrix, translation = _affine(matrix, translation)
    if matrix.shape != (len(images), 2, 2) or translation.shape != (len(images), 2):
        raise ValueError("one affine map is required per image")
    inverse, info = torch.linalg.inv_ex(matrix)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
        raise ValueError("rendering requires invertible affine maps")
    output = images.new_empty((len(images), 2, 28, 28))
    coordinates = torch.arange(28, dtype=matrix.dtype) - 13.5
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(1, 784, 2)
    for start in range(0, len(images), TRANSFORM_CHUNK):
        if check_stop is not None:
            check_stop()
        batch = images[start:start + TRANSFORM_CHUNK]
        reference = F.pad(F.interpolate(batch, size=(18, 18), mode="bilinear",
                                       align_corners=False, antialias=True), (5, 5, 5, 5))
        end = start + len(batch)
        grid = (pixels - translation[start:end, None, :]) @ inverse[start:end].transpose(1, 2)
        grid = (grid / 14.).reshape(-1, 28, 28, 2).to(images.dtype)
        query = F.grid_sample(reference, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        output[start:end] = torch.cat((reference, query), dim=1)
    if check_stop is not None:
        check_stop()
    return output


def blur_pairs(images: Tensor, strengths: Any, blur_max: float,
               check_stop: Callable[[], None] | None = None) -> Tensor:
    """Blur only query channel 1 with sigma=strength*blur_max, fixed radius six.

    Normalized separable kernels use zero padding. Strengths are per pair (or a
    scalar), in [0,1]. Zero blur/strength is an exact identity; inputs are never
    mutated. This also supports held-out validation for diagnostic epoch choice.
    """
    if check_stop is not None:
        check_stop()
    _images(images, 2)
    if not math.isfinite(blur_max) or blur_max < 0:
        raise ValueError("blur_max must be finite and nonnegative")
    strengths = torch.as_tensor(strengths, dtype=torch.float64, device="cpu")
    if strengths.ndim == 0:
        strengths = strengths.expand(len(images))
    if (strengths.shape != (len(images),) or not bool(torch.isfinite(strengths).all())
            or bool(((strengths < 0) | (strengths > 1)).any())):
        raise ValueError("strengths must have one finite value in [0,1] per pair")
    if not len(images) or blur_max == 0 or not bool((strengths > 0).any()):
        return images
    output = images.clone(memory_format=torch.contiguous_format)
    coordinates = torch.arange(-BLUR_RADIUS, BLUR_RADIUS + 1, dtype=torch.float64)
    for start in range(0, len(images), TRANSFORM_CHUNK):
        if check_stop is not None:
            check_stop()
        batch = images[start:start + TRANSFORM_CHUNK, 1:2]
        count = len(batch)
        sigma = blur_max * strengths[start:start + count]
        safe_sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
        kernels = torch.exp(-.5 * (coordinates[None, :] / safe_sigma[:, None]).square())
        kernels[sigma == 0] = (coordinates == 0).double()
        kernels = (kernels / kernels.sum(dim=1, keepdim=True)).to(images.dtype)
        horizontal = F.conv2d(batch.reshape(1, count, 28, 28), kernels.reshape(count, 1, 1, -1),
                              padding=(0, BLUR_RADIUS), groups=count)
        query = F.conv2d(horizontal, kernels.reshape(count, 1, -1, 1),
                         padding=(BLUR_RADIUS, 0), groups=count)
        output[start:start + count, 1:2] = query.reshape_as(batch)
    if check_stop is not None:
        check_stop()
    return output


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("protocol") != PROTOCOL:
        raise ValueError("unsupported registration data protocol/schema")
    if plan.get("partition") not in ("development", "confirmation"):
        raise ValueError("invalid registration data partition")


def contextualize(base_data: dict[str, Any], cfg: Any, blur_max: float,
                  check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Share nominal tensors/labels and replace only novel query channels.

    Always use the nominal materialize result; stacking contexts is rejected.
    cfg is retained for the common runner API, not used for data/model choices.
    """
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
    """Render all frozen assets as CPU float32 pairs and (N,6) pixel targets.

    reference_x contains the full reference_max_size reserve; the runner chooses
    its usable nested prefix. No downloads, writes, resampling, or task slicing.
    """
    if check_stop is not None:
        check_stop()
    _validate_plan(plan)
    if not math.isfinite(blur_max) or blur_max < 0:
        raise ValueError("blur_max must be finite and nonnegative")

    def render(ids, parameters, *, test=False):
        if check_stop is not None:
            check_stop()
        parameters = torch.tensor(parameters, dtype=torch.float64).reshape(-1, 6)
        matrix, translation = affine_from_parameters(parameters)
        images = source.images(ids, test=test).to(device="cpu", dtype=torch.float32)
        pairs = render_pairs(images, matrix, translation, check_stop)
        return pairs, encode_affine(matrix, translation).float()

    result = dict(plan=plan, blur_max=0., render_seconds={})
    for role, output in (("initialization", "pretrain"), ("validation", "validation"),
                         ("reference", "reference"), ("stream", "stream"),
                         ("diagnostic_base", "diagnostic_base"), ("diagnostic_novel", "diagnostic_novel")):
        started = time.perf_counter()
        x, y = render(plan[f"{role}_ids"], plan[f"{role}_transforms"])
        result["render_seconds"][role] = time.perf_counter() - started
        result[f"{output}_x"] = x
        if role != "reference":
            result[f"{output}_y"] = y
    panels = {}
    for domain in DOMAINS:
        ids = plan["evaluation_ids"][domain]
        if domain == "novel":
            x, y = panels["base"]["x"], panels["base"]["y"]
            if (ids != plan["evaluation_ids"]["base"]
                    or plan["evaluation_transforms"][domain] != plan["evaluation_transforms"]["base"]):
                raise ValueError("registration panels must share source IDs and transforms")
        else:
            x, y = render(ids, plan["evaluation_transforms"][domain], test=True)
        digits = torch.as_tensor(source.test_labels, dtype=torch.long)[ids].clone()
        panels[domain] = dict(x=x, y=y, digits=digits,
                              weights=class_weights(digits, plan["class_probabilities"]))
    result["panels"] = panels
    return contextualize(result, cfg, blur_max, check_stop) if blur_max != 0 else result
