"""Frozen source and transform plans for padded registration under smooth drift.

Every law reuses a six-uniform transform basis. Geometry is asserted, never
rejected, so amplitudes and current-law panels preserve exact source coupling.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory_registration.data import (
    ANCHORS, PARAMETERS, SUPPORT_CORNERS, affine_from_parameters, encode_affine,
)

from ..registration.data import EVAL_ROLES, _covariance, _labels, _partitions, _source_images
from .config import Config, TRAIN_ROLES, content_hash


DATA_SCHEMA_VERSION = 1
PARTITION_PROTOCOL = "discrepancy-study-stratified-24k-1k-3k-2k-15k-15k-test-halves-v1"
CANVAS_SIZE = 40
SUPPORT_BOUND = 19.
load_cached = load_data


def replica_assignment(cfg: Config, replica: int, cohort: str) -> dict[str, Any]:
    if cohort not in ("development", "main", "fresh", "stationary"):
        raise ValueError("cohort must be development, main, fresh, or stationary")
    cfg.seed_for(cohort, replica, "assignment")
    sign = -1 if replica % 2 == 0 else 1
    factor = 1. if cohort == "development" else cfg.schedule_factors[(replica // 2) % 2]
    return dict(sign=sign, factor=float(factor), block=replica % 4,
                assignment="index-balanced-sign-fast-length-slow-v1")


def make_schedule(cfg: Config, stationary: bool = False, *, factor: float = 1.,
                  hold_factor: float = 1.) -> tuple[Tensor, list[dict[str, Any]]]:
    """Two holds joined by endpoint-inclusive half-cosine outward/return ramps."""
    cfg.validate()
    if factor not in cfg.schedule_factors or hold_factor not in (1, 2):
        raise ValueError("factor must be frozen at 1/1.25 and hold_factor at 1/2")
    if type(stationary) is not bool:
        raise ValueError("stationary must be a boolean")
    hold = math.ceil(cfg.hold_steps * factor * hold_factor)
    ramp = math.ceil(cfg.ramp_steps * factor)
    rise = (1 - torch.cos(torch.pi * torch.linspace(0., 1., ramp, dtype=torch.float64))) / 2
    rise[0], rise[-1] = 0., 1.
    parts = (("low_hold_initial", torch.zeros(hold, dtype=torch.float64)), ("rise", rise),
             ("return", rise.flip(0)), ("low_hold_final", torch.zeros(hold, dtype=torch.float64)))
    segments, start = [], 0
    for name, values in parts:
        segments.append(dict(name=name, start=start, end=start + len(values),
                             arrivals=len(values), observations=len(values) * cfg.samples_per_step,
                             stationary=stationary))
        start += len(values)
    h = torch.cat([values for _, values in parts])
    return (torch.zeros_like(h) if stationary else h), segments


def support_mask(matrix: Any, translation: Any) -> Tensor:
    """Conservative bilinear support must lie in [-19,19]^2 on a 40px canvas."""
    matrix = torch.as_tensor(matrix, device="cpu")
    translation = torch.as_tensor(translation, device="cpu", dtype=matrix.dtype)
    if (not matrix.is_floating_point() or matrix.shape[-2:] != (2, 2)
            or translation.shape != (*matrix.shape[:-2], 2)
            or not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(translation).all())):
        raise ValueError("finite matching CPU affine matrices and translations required")
    corners = matrix.new_tensor(SUPPORT_CORNERS)
    mapped = corners @ matrix.transpose(-1, -2) + translation.unsqueeze(-2)
    return (mapped.abs() <= SUPPORT_BOUND).all(dim=-1).all(dim=-1)


def sample_transforms(means: Tensor | float, seed: int, *, count: int | None = None,
                      angle_std: float = 6., angle_limit: float = 70.,
                      uniforms: Tensor | None = None) -> tuple[Tensor, dict[str, Any]]:
    """Inverse-CDF truncated angle plus five independent uniform nuisances.

    Reusing the same uniforms changes only angles as their assigned means move.
    An invalid support footprint raises: silently conditioning the law is forbidden.
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
    if uniforms is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        uniforms = torch.rand(len(means), 6, generator=generator, dtype=torch.float64)
    if (not isinstance(uniforms, Tensor) or uniforms.device.type != "cpu"
            or uniforms.dtype != torch.float64 or uniforms.shape != (len(means), 6)
            or not bool(torch.isfinite(uniforms).all())
            or bool(((uniforms < 0) | (uniforms >= 1)).any())):
        raise ValueError("uniform basis must be CPU float64 (N,6) in [0,1)")
    normal = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64),
                                        torch.tensor(1., dtype=torch.float64))
    lower = normal.cdf((-angle_limit - means) / angle_std)
    upper = normal.cdf((angle_limit - means) / angle_std)
    quantiles = lower + uniforms[:, 0] * (upper - lower)
    quantiles = quantiles.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
    angles = means + angle_std * normal.icdf(quantiles)
    limits = means.new_tensor((2., 2., math.log(1.1), math.log(1.1), .1))
    parameters = torch.cat((angles[:, None], (2 * uniforms[:, 1:] - 1) * limits), dim=1)
    matrix, translation = affine_from_parameters(parameters)
    if not bool(support_mask(matrix, translation).all()):
        raise ValueError("configured affine proposal exceeds padded support; renderer defect")
    targets = encode_affine(matrix, translation)
    edges = torch.linspace(-angle_limit, angle_limit, 41, dtype=torch.float64)
    histogram = torch.histogram(angles, bins=edges)[0]
    density = (histogram / (len(means) * float(edges[1] - edges[0]))).tolist()
    diagnostics = dict(
        accepted=len(means), proposed=len(means), rejected=0, acceptance_fraction=1.,
        proposal_counts=torch.ones(len(means), dtype=torch.long), uniforms=uniforms,
        angle_bin_edges=edges.tolist(), proposed_angle_counts=histogram.long().tolist(),
        accepted_angle_counts=histogram.long().tolist(), proposed_angle_density=density,
        accepted_angle_density=density, parameter_mean=parameters.mean(0).tolist(),
        parameter_covariance=_covariance(parameters), target_mean=targets.mean(0).tolist(),
        target_covariance=_covariance(targets), covariance_ddof=1,
        nuisance_independence="independent angle/translation/log-scale/shear; no rejection",
        truncation="inverse-normal-cdf", angle_std=angle_std, angle_limit=angle_limit,
        untruncated_mass_outside_mean=float((lower + 1 - upper).mean()),
        uniform_hash=content_hash(uniforms.tolist()),
    )
    return parameters, diagnostics


def render_pairs(images: Tensor, matrix: Any, translation: Any,
                 check_stop: Callable[[], None] | None = None) -> Tensor:
    """Pad the 18px antialiased source before inverse bilinear affine sampling."""
    if check_stop is not None:
        check_stop()
    if (images.device.type != "cpu" or images.ndim != 4 or images.shape[1:] != (1, 28, 28)
            or images.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any())):
        raise ValueError("images must be normalized CPU floating (N,1,28,28)")
    matrix = torch.as_tensor(matrix, dtype=torch.float64, device="cpu")
    translation = torch.as_tensor(translation, dtype=torch.float64, device="cpu")
    if matrix.shape != (len(images), 2, 2) or translation.shape != (len(images), 2):
        raise ValueError("one affine map is required per image")
    if not bool(support_mask(matrix, translation).all()):
        raise ValueError("render plans contain geometrically invalid padded transforms")
    inverse, info = torch.linalg.inv_ex(matrix)
    if bool((info != 0).any()) or not bool(torch.isfinite(inverse).all()):
        raise ValueError("rendering requires invertible affine maps")
    output = images.new_empty((len(images), 2, CANVAS_SIZE, CANVAS_SIZE))
    coordinates = torch.arange(CANVAS_SIZE, dtype=torch.float64) - (CANVAS_SIZE - 1) / 2
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(1, CANVAS_SIZE**2, 2)
    for start in range(0, len(images), 256):
        if check_stop is not None:
            check_stop()
        batch = images[start:start + 256]
        reference = F.pad(F.interpolate(batch, size=(18, 18), mode="bilinear",
                                       align_corners=False, antialias=True), (11, 11, 11, 11))
        end = start + len(batch)
        grid = (pixels - translation[start:end, None, :]) @ inverse[start:end].transpose(1, 2)
        grid = (grid / (CANVAS_SIZE / 2)).reshape(-1, CANVAS_SIZE, CANVAS_SIZE, 2).to(images.dtype)
        query = F.grid_sample(reference, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        output[start:end] = torch.cat((reference, query), dim=1)
    if check_stop is not None:
        check_stop()
    return output


def render_panel(raw_data: Any, panel: Mapping[str, Any], indices: Any = None) -> tuple[Tensor, Tensor]:
    ids, transforms = panel["ids"], panel["transforms"]
    if indices is not None:
        ids, transforms = ids[indices], transforms[indices]
    ids, transforms = ids.reshape(-1), transforms.reshape(-1, 6)
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
        raise ValueError("render plans contain geometrically invalid padded transforms")
    if not len(ids):
        return torch.empty(0, 2, CANVAS_SIZE, CANVAS_SIZE), torch.empty(0, 6)
    images = _source_images(raw_data, source_ids, test=test)
    return render_pairs(images, matrix, translation), encode_affine(matrix, translation).float()


def render_batch(raw_data: Any, stream: Mapping[str, Any], step: int) -> tuple[Tensor, Tensor]:
    if type(step) is not int or not 0 <= step < len(stream["ids"]):
        raise ValueError("step must index an existing arrival")
    return render_panel(raw_data, stream, indices=step)


def _panel(ids: Tensor, digits: Tensor, transforms: Tensor, uniforms: Tensor, *,
           test: bool, **extra: Any) -> dict[str, Any]:
    return dict(ids=ids, digits=digits, transforms=transforms, params=transforms,
                uniforms=uniforms, test=test, proposal_counts=torch.ones(len(ids), dtype=torch.long),
                hash=content_hash(dict(ids=ids.tolist(), transforms=transforms.tolist(), test=test)), **extra)


def current_panel(assets: Mapping[str, Any], schedule: str, accepted_step: int) -> dict[str, Any]:
    """Independent panel for the most recent accepted batch (zero is nominal)."""
    if schedule not in assets["streams"]:
        raise ValueError("current-law panel requires a known schedule")
    stream = assets["streams"][schedule]
    if type(accepted_step) is not int or not 0 <= accepted_step <= len(stream["ids"]):
        raise ValueError("current-law accepted_step must lie in [0, arrival_count]")
    mu = 0. if accepted_step == 0 else float(stream["mu"][accepted_step - 1])
    old = assets["panels"]["old"]
    if mu == 0.:
        return old
    for name in ("shifted", "shifted25", "shifted45"):
        endpoint = assets["panels"][name]
        if mu == endpoint["mean"]:
            return endpoint
    settings = assets["metadata"]["rendering"]
    transforms, _ = sample_transforms(mu, assets["seeds"]["evaluation:transforms"],
        count=len(old["ids"]), angle_std=settings["angle_std"], angle_limit=settings["angle_limit"],
        uniforms=old["uniforms"])
    return _panel(old["ids"], old["digits"], transforms, old["uniforms"], test=True,
                  weights=old["weights"], mean=mu, schedule=schedule, accepted_step=accepted_step)


def panel_for_role(assets: Mapping[str, Any], role: str) -> dict[str, Any]:
    if not isinstance(role, str):
        raise ValueError("batch role must be a string")
    if role.startswith("current:"):
        parts = role.split(":")
        if len(parts) != 3 or not parts[2].isdigit():
            raise ValueError("current role must be current:<schedule>:<accepted_step>")
        return current_panel(assets, parts[1], int(parts[2]))
    if role in assets["streams"]:
        stream = assets["streams"][role]
        return dict(ids=stream["ids"].reshape(-1), transforms=stream["transforms"].reshape(-1, 6),
                    digits=stream["digits"].reshape(-1), test=False)
    return assets["panels"][role]


def get_batch(raw_data: Any, assets: Mapping[str, Any], role: str,
              positions: Any = None) -> dict[str, Tensor]:
    panel = panel_for_role(assets, role)
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
    """Seal one amplitude's plans; initialization and uniforms pair amplitudes."""
    cfg.validate()
    assignment = replica_assignment(cfg, replica, cohort)
    angle = cfg.angle_max if angle is None else angle
    if type(angle) not in (int, float) or angle not in cfg.amplitudes:
        raise ValueError("angle must be one of the two frozen excursion amplitudes")
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

    def panel(name: str, ids: Tensor, mean: float, *, test: bool = False,
              basis: str | None = None, uniforms: Tensor | None = None) -> dict[str, Any]:
        parameters, info = sample_transforms(mean, seed(f"{basis or name}:transforms"), count=len(ids),
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit, uniforms=uniforms)
        digits = test_labels[ids - len(train_labels)] if test else train_labels[ids]
        result = _panel(ids, digits, parameters, info["uniforms"], test=test, mean=float(mean))
        panels[name] = result
        diagnostics[name] = {key: value for key, value in info.items()
                             if key not in ("proposal_counts", "uniforms")}
        return result

    for name, role, count in (("init", "initialization", cfg.initialization_size),
                             ("validation", "validation", cfg.validation_size),
                             ("reference", "reference", cfg.reference_max_size)):
        panel(name, choose(role, count, f"{name}:sources"), 0.)
    panels["initialization"] = panels["initial"] = panels["init"]
    generator = torch.Generator(device="cpu").manual_seed(seed("noise_calibration"))
    calibration_indices = torch.randperm(cfg.initialization_size, generator=generator)[:cfg.calibration_size]
    initial = panels["init"]
    panels["calibration"] = _panel(initial["ids"][calibration_indices], initial["digits"][calibration_indices],
        initial["transforms"][calibration_indices], initial["uniforms"][calibration_indices], test=False, mean=0.)
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
    old = panel("old", evaluation, 0., test=True, basis="evaluation")
    old["weights"] = probabilities[old["digits"]] / cfg.eval_per_class
    for amplitude in cfg.amplitudes:
        endpoint = panel(f"shifted{amplitude:g}", evaluation, assignment["sign"] * amplitude,
                         test=True, basis="evaluation", uniforms=old["uniforms"])
        endpoint["weights"] = old["weights"]
    panels["shifted"] = panels[f"shifted{angle:g}"]
    panels["current"] = old

    online_role = "online_development" if cohort == "development" else "online_confirmation"
    stream_ids = choose(online_role, len(h) * cfg.samples_per_step, "stream:sources").reshape(-1, cfg.samples_per_step)
    stream_digits = train_labels[stream_ids]
    streams: dict[str, Any] = {}
    stream_metadata: dict[str, Any] = {}
    transform_seed = seed("stream:transforms")
    for name, stationary in (("main", False), ("stationary", True)):
        strength, resolved_segments = make_schedule(cfg, stationary=stationary,
            factor=assignment["factor"], hold_factor=hold_factor)
        mu = assignment["sign"] * angle * strength
        parameters, info = sample_transforms(mu.repeat_interleave(cfg.samples_per_step), transform_seed,
                                             angle_std=cfg.angle_std, angle_limit=cfg.angle_limit)
        parameters = parameters.reshape(len(strength), cfg.samples_per_step, 6)
        segment_names = [segment["name"] for segment in resolved_segments
                         for _ in range(segment["end"] - segment["start"])]
        stream_hash = content_hash(dict(ids=stream_ids.tolist(), transforms=parameters.tolist(),
                                        h=strength.tolist(), mu=mu.tolist()))
        streams[name] = dict(ids=stream_ids, digits=stream_digits, transforms=parameters,
            params=parameters, uniforms=info["uniforms"].reshape_as(parameters), test=False,
            h=strength, mu=mu, segments=segment_names, segment_ranges=resolved_segments,
            arrival_ids=torch.arange(len(strength)), hash=stream_hash,
            proposal_counts=info["proposal_counts"].reshape_as(stream_ids))
        stream_metadata[name] = dict(hash=stream_hash, stream_hash=stream_hash,
            schedule_hash=content_hash(dict(h=strength.tolist(), mu=mu.tolist(), segments=resolved_segments)),
            uniform_hash=info["uniform_hash"], arrival_count=len(strength), observation_count=stream_ids.numel(),
            unique_id_count=stream_ids.unique().numel(), replacement=False,
            class_counts=torch.bincount(stream_digits.flatten(), minlength=10).tolist(), segments=resolved_segments)
        diagnostics[f"stream_{name}"] = {key: value for key, value in info.items()
                                        if key not in ("proposal_counts", "uniforms")}
    metadata = dict(schema_version=DATA_SCHEMA_VERSION, protocol=cfg.protocol, replica=replica,
        cohort=cohort, partition_protocol=PARTITION_PROTOCOL, partition_seeds=partition_seeds,
        partition_hash=content_hash({role: ids.tolist() for role, ids in partitions.items()}),
        role_counts={role: len(ids) for role, ids in partitions.items()},
        raw_train_class_counts=raw_counts.tolist(), class_probabilities=probabilities.tolist(),
        train_size=len(train_labels), test_size=len(test_labels), test_id_offset=len(train_labels),
        evaluation_role=evaluation_role, assignment=assignment, angle_max=float(angle), amplitudes=list(cfg.amplitudes),
        schedule_factor=assignment["factor"], hold_factor=hold_factor,
        resolved_hold_steps=segments[0]["arrivals"], resolved_ramp_steps=segments[1]["arrivals"],
        resolved_num_p_steps=len(h), seeds=seeds, streams=stream_metadata,
        calibration=dict(source="initialization-training-subset", additional_labels=0,
            size=cfg.calibration_size, initialization_indices=calibration_indices.tolist()),
        reference=dict(primary_size=cfg.reference_size, reserve_size=cfg.reference_max_size, nested_prefix=True),
        transform_diagnostics=diagnostics,
        panel_hashes={name: value["hash"] for name, value in panels.items()},
        current_evaluation=dict(role_format="current:<schedule>:<accepted_step>",
            time="most-recent-accepted-batch; initialization nominal", fixed_source_ids=True,
            uniform_hash=content_hash(old["uniforms"].tolist()), shared_endpoint_uniforms=True,
            source_role=evaluation_role, additional_training_labels=0),
        rendering=dict(parameter_order=list(PARAMETERS), anchors=[list(z) for z in ANCHORS],
            support_corners=[list(z) for z in SUPPORT_CORNERS], support_bound=SUPPORT_BOUND,
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
            angle_law="fixed-support truncated normal; no geometric rejection",
            nuisance_half_widths=[2., 2., math.log(1.1), math.log(1.1), .1],
            forward_map="R(alpha) diag(exp(a_x),exp(a_y)) shear(h) z + t",
            target_order=["d1x", "d1y", "d2x", "d2y", "d3x", "d3y"],
            target_units="pixels", positive_angles="clockwise", canvas_center=[19.5, 19.5],
            canvas_size=CANVAS_SIZE, resized_shape=[18, 18], antialias=True, interpolation="bilinear",
            query_sampling="inverse-map", align_corners=False, padding="zeros",
            changing_blur=False, controller_covariates="images only; targets after action"))
    result = dict(panels=panels, streams=streams, partitions=partitions, seeds=seeds,
                  metadata=metadata, calibration_indices=calibration_indices,
                  class_probabilities=probabilities, schedule_factor=assignment["factor"], sign=assignment["sign"])
    for name in ("init", "validation", "reference", "calibration"):
        result[f"{name}_ids"] = panels[name]["ids"]
        result[f"{name}_indices"] = calibration_indices if name == "calibration" else panels[name]["ids"]
    return result
