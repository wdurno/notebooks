"""Immutable CPU render plans for continuous single-digit registration drift.

No images are copied into assets. IDs use one study-wide space: training IDs
are raw indices, and official-test IDs are offset by the training split size.
Rendering reuses only the frozen Phase 10 geometry, never its memory model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory_registration.data import (
    ANCHORS, PARAMETERS, SUPPORT_CORNERS, affine_from_parameters, encode_affine,
    render_pairs, support_mask,
)

from .config import Config, TRAIN_ROLES, content_hash


DATA_SCHEMA_VERSION = 1
PARTITION_PROTOCOL = "registration-study-stratified-24k-1k-3k-2k-15k-15k-test-halves-v1"
EVAL_ROLES = ("evaluation_development", "evaluation_fresh")
load_cached = load_data


def _labels(raw_data: Any, split: str) -> Tensor:
    value = raw_data[f"{split}_y"] if isinstance(raw_data, Mapping) else getattr(raw_data, f"{split}_labels")
    if (not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.long
            or value.ndim != 1 or not len(value) or bool(((value < 0) | (value > 9)).any())
            or bool((torch.bincount(value, minlength=10) == 0).any())):
        raise ValueError("both source splits must have nonempty CPU int64 labels for all ten digits")
    return value


def _apportion(count: int, capacities: Tensor) -> Tensor:
    """Largest remainders with exact totals and no capacity overflow."""
    if count > int(capacities.sum()):
        raise ValueError("insufficient disjoint source capacity")
    ideal = capacities.double() * count / int(capacities.sum())
    allocated = ideal.floor().long()
    remainder = count - int(allocated.sum())
    order = torch.argsort(ideal - allocated, descending=True, stable=True)
    allocated[order[:remainder]] += 1
    return allocated


def _partitions(raw_data: Any, cfg: Config) -> tuple[dict[str, Tensor], dict[str, int]]:
    train_labels, test_labels = _labels(raw_data, "train"), _labels(raw_data, "test")
    seeds = {name: cfg.seed_for("study-partition", 0, name) for name in ("train", "test")}
    buckets: dict[str, list[Tensor]] = {role: [] for role in (*TRAIN_ROLES, *EVAL_ROLES)}
    # Proportional scaling permits genuinely tiny synthetic CPU acceptance data.
    # On canonical MNIST these totals are exactly 24000/1000/3000/2000/15000/15000.
    remaining = (_apportion(len(train_labels), torch.tensor(cfg.role_counts))
                 if len(train_labels) <= sum(cfg.role_counts) else None)
    if remaining is None or bool((remaining < 1).any()):
        raise ValueError("source split must fit the study allocation and populate every training role")
    generator = torch.Generator(device="cpu").manual_seed(seeds["train"])
    for digit in range(10):
        ids = torch.where(train_labels == digit)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        counts = _apportion(len(ids), remaining)
        start = 0
        for role, size in zip(TRAIN_ROLES, counts.tolist(), strict=True):
            buckets[role].append(ids[start:start + size])
            start += size
        remaining -= counts
    generator = torch.Generator(device="cpu").manual_seed(seeds["test"])
    for digit in range(10):
        ids = torch.where(test_labels == digit)[0]
        if len(ids) < 2:
            raise ValueError("each test class needs at least two sources for dev/fresh isolation")
        ids = ids[torch.randperm(len(ids), generator=generator)] + len(train_labels)
        cut = len(ids) // 2
        buckets[EVAL_ROLES[0]].append(ids[:cut])
        buckets[EVAL_ROLES[1]].append(ids[cut:])
    return {role: torch.cat(parts) for role, parts in buckets.items()}, seeds


def replica_assignment(cfg: Config, replica: int, cohort: str) -> dict[str, Any]:
    """Predetermine balanced sign/length blocks without looking at outcomes."""
    if cohort not in ("development", "main", "fresh", "stationary"):
        raise ValueError("cohort must be development, main, fresh, or stationary")
    cfg.seed_for(cohort, replica, "assignment")
    sign = -1 if replica % 2 == 0 else 1
    factor = 1. if cohort == "development" else cfg.schedule_factors[(replica // 2) % 2]
    return dict(sign=sign, factor=float(factor), block=replica % 4,
                assignment="index-balanced-sign-fast-length-slow-v1")


def make_schedule(cfg: Config, stationary: bool = False, *, factor: float = 1.,
                  hold_factor: float = 1.) -> tuple[Tensor, list[dict[str, Any]]]:
    """Half-cosine h(t), including both ramp endpoints and all hold batches.

    Length factor scales the whole path. The optional common hold extension
    changes only holds; it cannot silently stretch the ramps.
    """
    cfg.validate()
    if factor not in cfg.schedule_factors or hold_factor not in (1, 2):
        raise ValueError("factor must be frozen at 1/1.25 and hold_factor at 1/2")
    hold = math.ceil(cfg.hold_steps * factor * hold_factor)
    ramp = math.ceil(cfg.ramp_steps * factor)
    phase = torch.linspace(0., 1., ramp, dtype=torch.float64)
    rise = (1 - torch.cos(torch.pi * phase)) / 2
    rise[0], rise[-1] = 0., 1.
    parts = (("low_hold_initial", torch.zeros(hold, dtype=torch.float64)), ("rise", rise),
             ("high_hold", torch.ones(hold, dtype=torch.float64)), ("return", rise.flip(0)),
             ("low_hold_final", torch.zeros(hold, dtype=torch.float64)))
    segments, start = [], 0
    for name, values in parts:
        segments.append(dict(name=name, start=start, end=start + len(values),
                             arrivals=len(values), observations=len(values) * cfg.samples_per_step,
                             stationary=stationary))
        start += len(values)
    h = torch.cat([values for _, values in parts])
    return (torch.zeros_like(h) if stationary else h), segments


def _covariance(values: Tensor) -> list[list[float]]:
    centered = values - values.mean(0)
    return (centered.T @ centered / max(len(values) - 1, 1)).tolist()


def sample_transforms(means: Tensor | float, seed: int, *, count: int | None = None,
                      angle_std: float = 6., angle_limit: float = 50.) -> tuple[Tensor, dict[str, Any]]:
    """Draw fixed-support truncated normals, then reject invalid full affines.

    Six independent uniforms propose angle/translation/log-scale/shear. Inverse
    CDF sampling truncates the Gaussian exactly before the common geometric
    rejection. A rejected observation retains its own assigned mean on retry.
    Diagnostics count only consumed proposals, including every rejection.
    """
    means = torch.as_tensor(means, dtype=torch.float64, device="cpu")
    if means.ndim == 0 and type(count) is int and count > 0:
        means = means.expand(count)
    if (means.ndim != 1 or not len(means) or not bool(torch.isfinite(means).all())
            or bool((means.abs() > angle_limit).any())):
        raise ValueError("one finite in-support angle mean is required per observation")
    if (type(seed) is not int or not 0 <= seed < 2**63 or not math.isfinite(angle_std)
            or angle_std <= 0 or not math.isfinite(angle_limit) or angle_limit <= 0):
        raise ValueError("invalid transform seed, angle standard deviation, or support")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = torch.empty(len(means), 6, dtype=torch.float64)
    proposal_counts = torch.zeros(len(means), dtype=torch.long)
    pending = torch.arange(len(means))
    normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
    lower = normal.cdf((-angle_limit - means) / angle_std)
    upper = normal.cdf((angle_limit - means) / angle_std)
    limits = torch.tensor((2., 2., math.log(1.1), math.log(1.1), .1), dtype=torch.float64)
    edges = torch.linspace(-angle_limit, angle_limit, 41, dtype=torch.float64)
    proposed_histogram = torch.zeros(40, dtype=torch.float64)
    for _ in range(10000):
        if not len(pending):
            break
        uniforms = torch.rand(len(pending), 6, generator=generator, dtype=torch.float64)
        quantiles = lower[pending] + uniforms[:, 0] * (upper[pending] - lower[pending])
        quantiles = quantiles.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
        angles = means[pending] + angle_std * normal.icdf(quantiles)
        candidates = torch.cat((angles[:, None], (2 * uniforms[:, 1:] - 1) * limits), dim=1)
        proposed_histogram += torch.histogram(angles, bins=edges)[0]
        proposal_counts[pending] += 1
        accepted = support_mask(*affine_from_parameters(candidates))
        parameters[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]
    else:
        raise RuntimeError("geometric rejection exceeded 10000 proposals for an observation")
    targets = encode_affine(*affine_from_parameters(parameters))
    accepted_histogram = torch.histogram(parameters[:, 0], bins=edges)[0]
    proposed = int(proposal_counts.sum())
    bin_width = float(edges[1] - edges[0])
    diagnostics = dict(
        accepted=len(means), proposed=proposed, rejected=proposed - len(means),
        acceptance_fraction=len(means) / proposed, proposal_counts=proposal_counts,
        angle_bin_edges=edges.tolist(), proposed_angle_counts=proposed_histogram.long().tolist(),
        accepted_angle_counts=accepted_histogram.long().tolist(),
        proposed_angle_density=(proposed_histogram / (proposed * bin_width)).tolist(),
        accepted_angle_density=(accepted_histogram / (len(means) * bin_width)).tolist(),
        parameter_mean=parameters.mean(0).tolist(), parameter_covariance=_covariance(parameters),
        target_mean=targets.mean(0).tolist(), target_covariance=_covariance(targets),
        covariance_ddof=1, nuisance_independence="proposal-only; geometric conditioning couples factors",
        truncation="inverse-normal-cdf", angle_std=angle_std, angle_limit=angle_limit,
    )
    return parameters, diagnostics


def _source_images(raw_data: Any, ids: Tensor, *, test: bool) -> Tensor:
    if isinstance(raw_data, Mapping):
        source = raw_data["test_x" if test else "train_x"]
        if (not isinstance(source, Tensor) or source.device.type != "cpu" or source.ndim != 4
                or source.shape[1:] != (1, 28, 28) or not source.is_floating_point()):
            raise ValueError("raw images must be normalized CPU floating (N,1,28,28) tensors")
        images = source[ids].float()
    else:
        images = raw_data.images(ids.tolist(), test=test).to(device="cpu", dtype=torch.float32)
    return images


def render_panel(raw_data: Any, panel: Mapping[str, Any], indices: Any = None) -> tuple[Tensor, Tensor]:
    """Render selected plan rows as CPU float32 (N,2,28,28), (N,6).

    IDs/angles/digits remain in the audit plan, not in model inputs. ``indices``
    may be a slice or int64 positions; a scalar selects a batch of size one.
    No transforms are redrawn, and the raw source is never modified.
    """
    ids, transforms = panel["ids"], panel["transforms"]
    if indices is not None:
        ids, transforms = ids[indices], transforms[indices]
    ids = ids.reshape(-1)
    transforms = transforms.reshape(-1, 6)
    if len(ids) != len(transforms) or ids.dtype != torch.long or ids.device.type != "cpu":
        raise ValueError("one CPU int64 source ID and transform is required per observation")
    test = panel["test"]
    if type(test) is not bool:
        raise ValueError("render plans require an explicit test-split flag")
    source_ids = ids - (len(_labels(raw_data, "train")) if test else 0)
    source_count = len(_labels(raw_data, "test" if test else "train"))
    if bool(((source_ids < 0) | (source_ids >= source_count)).any()):
        raise ValueError("source IDs do not belong to the panel's declared split")
    matrix, translation = affine_from_parameters(transforms)
    if not bool(support_mask(matrix, translation).all()):
        raise ValueError("render plans contain geometrically invalid transforms")
    if not len(ids):
        return torch.empty(0, 2, 28, 28), torch.empty(0, 6)
    images = _source_images(raw_data, source_ids, test=test)
    pairs = render_pairs(images, matrix, translation)
    return pairs, encode_affine(matrix, translation).float()


def render_batch(raw_data: Any, stream: Mapping[str, Any], step: int) -> tuple[Tensor, Tensor]:
    """Render exactly one new arrival; no schedule metadata enters the return."""
    if type(step) is not int or not 0 <= step < len(stream["ids"]):
        raise ValueError("step must index an existing arrival")
    return render_panel(raw_data, stream, indices=step)


def get_batch(raw_data: Any, assets: Mapping[str, Any], role: str,
              positions: Any = None) -> dict[str, Tensor]:
    """CPU learner callback payload; stream positions index flat observations."""
    if role in assets["streams"]:
        stream = assets["streams"][role]
        panel = dict(ids=stream["ids"].reshape(-1),
                     transforms=stream["transforms"].reshape(-1, 6),
                     digits=stream["digits"].reshape(-1), test=False)
    else:
        panel = assets["panels"][role]
    inputs, targets = render_panel(raw_data, panel, indices=positions)
    ids, digits = panel["ids"], panel["digits"]
    if positions is not None:
        ids, digits = ids[positions], digits[positions]
    result = dict(inputs=inputs, targets=targets, digits=digits.reshape(-1), ids=ids.reshape(-1))
    if "weights" in panel:
        result["weights"] = panel["weights"] if positions is None else panel["weights"][positions].reshape(-1)
    return result


def make_assets(cfg: Config, replica: int, cohort: str, raw_data: Any,
                angle: float | None = None, hold_factor: float = 1) -> dict[str, Any]:
    """Resolve every policy-independent source ID/transform before learning.

    Both schedules share a source order, initialization, and evaluation panels.
    Each schedule uses every online source once at most. Replicas may reuse
    benchmark sources, but no training role crosses another anywhere in study.
    Calibration aliases the exact already-seen initialization transforms.
    """
    cfg.validate()
    assignment = replica_assignment(cfg, replica, cohort)
    angle = cfg.angle_max if angle is None else angle
    if type(angle) not in (int, float) or not math.isfinite(angle) or not 0 < angle <= cfg.angle_limit:
        raise ValueError("angle must be a finite positive mean within the fixed support")
    h, segments = make_schedule(cfg, factor=assignment["factor"], hold_factor=hold_factor)
    partitions, partition_seeds = _partitions(raw_data, cfg)
    train_labels, test_labels = _labels(raw_data, "train"), _labels(raw_data, "test")
    raw_counts = torch.bincount(train_labels, minlength=10)
    probabilities = raw_counts.double() / len(train_labels)
    seeds: dict[str, int] = {}

    def seed(name: str) -> int:
        seeds[name] = cfg.seed_for(cohort, replica, name)
        return seeds[name]

    def choose(role: str, count: int, name: str) -> Tensor:
        pool = partitions[role]
        if count > len(pool):
            raise ValueError(f"{role} has only {len(pool)} disjoint sources; requested {count}")
        generator = torch.Generator(device="cpu").manual_seed(seed(name))
        return pool[torch.randperm(len(pool), generator=generator)[:count]]

    for name in ("initialization_model", "initialization_order", "numerical", "bootstrap"):
        seed(name)
    panels: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}

    def panel(name: str, ids: Tensor, mean: float, *, test: bool = False) -> dict[str, Any]:
        parameters, info = sample_transforms(mean, seed(f"{name}:transforms"), count=len(ids),
                                            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit)
        digits = test_labels[ids - len(train_labels)] if test else train_labels[ids]
        result = dict(ids=ids, transforms=parameters, params=parameters, test=test, digits=digits,
                      proposal_counts=info["proposal_counts"],
                      hash=content_hash(dict(ids=ids.tolist(), transforms=parameters.tolist(), test=test)))
        panels[name] = result
        diagnostics[name] = {key: value for key, value in info.items() if key != "proposal_counts"}
        return result

    for name, role, count in (("init", "initialization", cfg.initialization_size),
                               ("validation", "validation", cfg.validation_size),
                               ("reference", "reference", cfg.reference_max_size)):
        panel(name, choose(role, count, f"{name}:sources"), 0.)
    panels["initialization"] = panels["initial"] = panels["init"]
    calibration_generator = torch.Generator(device="cpu").manual_seed(seed("noise_calibration"))
    calibration_indices = torch.randperm(cfg.initialization_size, generator=calibration_generator)[:cfg.calibration_size]
    initial = panels["init"]
    calibration_transforms = initial["transforms"][calibration_indices]
    panels["calibration"] = dict(ids=initial["ids"][calibration_indices],
                                 transforms=calibration_transforms, params=calibration_transforms,
                                 digits=initial["digits"][calibration_indices], test=False,
                                 proposal_counts=initial["proposal_counts"][calibration_indices])
    diagnostic_ids = choose("diagnostic", cfg.capability_size + 3 * cfg.diagnostic_size, "diagnostic:sources")
    panel("diagnostic", diagnostic_ids[:cfg.capability_size], assignment["sign"] * angle)
    panels["capability"] = panels["diagnostic"]
    for i, (name, strength) in enumerate((("diagnostic_nominal", 0.), ("diagnostic_mid", .5),
                                         ("diagnostic_shifted", 1.))):
        start = cfg.capability_size + i * cfg.diagnostic_size
        panel(name, diagnostic_ids[start:start + cfg.diagnostic_size], assignment["sign"] * angle * strength)
    evaluation_role = EVAL_ROLES[0] if cohort == "development" else EVAL_ROLES[1]
    evaluation_ids = []
    for digit in range(10):
        pool = partitions[evaluation_role]
        pool = pool[test_labels[pool - len(train_labels)] == digit]
        if cfg.eval_per_class > len(pool):
            raise ValueError(f"{evaluation_role} has insufficient digit {digit} sources")
        generator = torch.Generator(device="cpu").manual_seed(seed(f"evaluation:{digit}:sources"))
        evaluation_ids.append(pool[torch.randperm(len(pool), generator=generator)[:cfg.eval_per_class]])
    evaluation = torch.cat(evaluation_ids)
    for name, mean in (("old", 0.), ("shifted", assignment["sign"] * angle)):
        result = panel(name, evaluation, mean, test=True)
        result["weights"] = probabilities[result["digits"]] / cfg.eval_per_class

    online_role = "online_development" if cohort == "development" else "online_confirmation"
    stream_ids = choose(online_role, len(h) * cfg.samples_per_step, "stream:sources").reshape(-1, cfg.samples_per_step)
    stream_digits = train_labels[stream_ids]
    streams: dict[str, Any] = {}
    stream_metadata: dict[str, Any] = {}
    # One common random number seed, reset for each law. Rejection differences
    # can change coupling, but each accepted marginal retains the specified law.
    transform_seed = seed("stream:transforms")
    for name, stationary in (("main", False), ("stationary", True)):
        strength, resolved_segments = make_schedule(cfg, stationary=stationary,
            factor=assignment["factor"], hold_factor=hold_factor)
        mu = assignment["sign"] * angle * strength
        means = mu[:, None].expand(-1, cfg.samples_per_step).reshape(-1)
        parameters, info = sample_transforms(means, transform_seed,
                                             angle_std=cfg.angle_std, angle_limit=cfg.angle_limit)
        parameters = parameters.reshape(len(strength), cfg.samples_per_step, 6)
        segment_names = [segment["name"] for segment in resolved_segments
                         for _ in range(segment["end"] - segment["start"])]
        stream_hash = content_hash(dict(ids=stream_ids.tolist(), transforms=parameters.tolist(),
                                        h=strength.tolist(), mu=mu.tolist()))
        streams[name] = dict(ids=stream_ids, digits=stream_digits, transforms=parameters,
            params=parameters, test=False, h=strength, mu=mu, segments=segment_names,
            segment_ranges=resolved_segments,
            arrival_ids=torch.arange(len(strength)), hash=stream_hash,
            proposal_counts=info["proposal_counts"].reshape_as(stream_ids))
        stream_metadata[name] = dict(hash=stream_hash, stream_hash=stream_hash,
            schedule_hash=content_hash(dict(h=strength.tolist(), mu=mu.tolist(), segments=resolved_segments)),
            arrival_count=len(strength), observation_count=stream_ids.numel(),
            unique_id_count=stream_ids.unique().numel(), replacement=False,
            class_counts=torch.bincount(stream_digits.flatten(), minlength=10).tolist(),
            segments=resolved_segments)
        diagnostics[f"stream_{name}"] = {key: value for key, value in info.items() if key != "proposal_counts"}
    metadata = dict(schema_version=DATA_SCHEMA_VERSION, protocol=cfg.protocol, replica=replica,
        cohort=cohort, partition_protocol=PARTITION_PROTOCOL, partition_seeds=partition_seeds,
        partition_hash=content_hash({role: ids.tolist() for role, ids in partitions.items()}),
        role_counts={role: len(ids) for role, ids in partitions.items()},
        raw_train_class_counts=raw_counts.tolist(), class_probabilities=probabilities.tolist(),
        train_size=len(train_labels), test_size=len(test_labels), test_id_offset=len(train_labels),
        evaluation_role=evaluation_role, assignment=assignment, angle_max=float(angle),
        schedule_factor=assignment["factor"], hold_factor=hold_factor,
        resolved_hold_steps=segments[0]["arrivals"], resolved_ramp_steps=segments[1]["arrivals"],
        resolved_num_p_steps=len(h), seeds=seeds, streams=stream_metadata,
        calibration=dict(source="initialization-training-subset", additional_labels=0,
            size=cfg.calibration_size, initialization_indices=calibration_indices.tolist()),
        reference=dict(primary_size=cfg.reference_size, reserve_size=cfg.reference_max_size,
                       nested_prefix=True), transform_diagnostics=diagnostics,
        panel_hashes={name: value["hash"] for name, value in panels.items() if "hash" in value},
        rendering=dict(parameter_order=list(PARAMETERS), anchors=[list(z) for z in ANCHORS],
            support_corners=[list(z) for z in SUPPORT_CORNERS], support_bound=13.,
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
            angle_law="fixed-support truncated normal followed by geometric rejection",
            nuisance_half_widths=[2., 2., math.log(1.1), math.log(1.1), .1],
            forward_map="R(alpha) diag(exp(a_x),exp(a_y)) shear(h) z + t",
            target_order=["d1x", "d1y", "d2x", "d2y", "d3x", "d3y"],
            target_units="pixels", positive_angles="clockwise", canvas_center=[13.5, 13.5],
            resized_shape=[18, 18], antialias=True, interpolation="bilinear",
            query_sampling="inverse-map", align_corners=False, padding="zeros",
            changing_blur=False, controller_covariates="images only; targets after action"))
    result = dict(panels=panels, streams=streams, partitions=partitions, seeds=seeds,
                  metadata=metadata, calibration_indices=calibration_indices,
                  class_probabilities=probabilities, schedule_factor=assignment["factor"],
                  sign=assignment["sign"])
    for name in ("init", "validation", "reference", "calibration"):
        result[f"{name}_ids"] = panels[name]["ids"]
        result[f"{name}_indices"] = calibration_indices if name == "calibration" else panels[name]["ids"]
    return result
