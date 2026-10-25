"""Paired angle-regression inputs; plans contain IDs and degrees, never images.

Training IDs refer to the official training split; evaluation IDs refer to the
official test split. Evaluation IDs/angles are dictionaries keyed by domain.
All materialized images and degree targets are CPU float32 tensors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mnist_experiment.fisher_memory.data import MNISTSource, class_weights
from src.seeding import derive_component_seed

from .config import Config


SCHEMA_VERSION = 1
DOMAINS = ("base", "novel")
ROTATION_CHUNK = 256
_SHAPE = .25
_MEDIAN = 40.
_MAXIMUM = 90.
_TRUNCATION_MASS = .5 * (1 + math.erf(math.log(_MAXIMUM / _MEDIAN) / _SHAPE / math.sqrt(2)))


def sample_angles(count: int, domain: str, seed: int) -> Tensor:
    """Return CPU float32 degrees, with independent novel sign/magnitude RNGs.

    Novel magnitudes are rejected above 90 degrees, never clipped. A longer
    draw preserves the shorter draw's prefix. Neither this function nor
    planning/materialization advances PyTorch's global RNG.
    """
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if domain not in DOMAINS:
        raise ValueError("domain must be base or novel")
    generator = torch.Generator().manual_seed(derive_component_seed(seed, f"{domain}:magnitude"))
    if domain == "base":
        return (20 * torch.rand(count, generator=generator, dtype=torch.float64) - 10).float()
    sign_generator = torch.Generator().manual_seed(derive_component_seed(seed, "novel:sign"))
    signs = 2 * torch.randint(0, 2, (count,), generator=sign_generator) - 1
    magnitude = torch.empty(count, dtype=torch.float64)
    offset = 0
    while offset < count:
        # Fixed draw sizes make prefixes independent of a requested sample count.
        candidates = _MEDIAN * torch.exp(_SHAPE * torch.randn(
            1_024, generator=generator, dtype=torch.float64))
        accepted = candidates[candidates <= _MAXIMUM][:count - offset]
        magnitude[offset:offset + len(accepted)] = accepted
        offset += len(accepted)
    return (magnitude * signs).float()


def angle_density(values: Any, domain: str) -> np.ndarray:
    """Analytic component density per degree, returned as a float64 NumPy array.

    Accepts scalars, array-likes, and CPU tensors; preserves the input shape.
    This is the component PDF, not the time-varying mixture PDF. At the support
    endpoints we use the interior value (their values do not affect integrals).
    """
    if domain not in DOMAINS:
        raise ValueError("domain must be base or novel")
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    angles = np.asarray(values, dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError("density coordinates must be finite")
    magnitude = np.abs(angles)
    if domain == "base":
        return np.where(magnitude <= 10., .05, 0.)
    support = (magnitude > 0.) & (magnitude <= _MAXIMUM)
    safe_magnitude = np.where(support, magnitude, _MEDIAN)
    density = np.exp(-.5 * (np.log(safe_magnitude / _MEDIAN) / _SHAPE) ** 2)
    density /= 2 * safe_magnitude * _SHAPE * math.sqrt(2 * math.pi) * _TRUNCATION_MASS
    return np.where(support, density, 0.)


def rotate_batch(images: Tensor, angles: Tensor, check_stop: Callable[[], None] | None = None) -> Tensor:
    """Rotate NCHW MNIST tensors by per-image degrees, positive counterclockwise.

    CPU-only, deterministic, bilinear, no expansion, same center and explicit
    zero-fill mask as ``rotate_mnist_batch``. Preserves floating image dtype.
    Rendering is chunked; ``check_stop`` may raise at chunk boundaries. Empty
    batches are supported. This function does not resize or normalize images.
    """
    if check_stop is not None:
        check_stop()
    if images.device.type != "cpu" or images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("images must be CPU tensors of shape (N, 1, 28, 28)")
    if images.dtype not in (torch.float32, torch.float64):
        raise ValueError("images must have float32 or float64 dtype")
    angles = torch.as_tensor(angles, dtype=torch.float64, device="cpu")
    if angles.shape != (len(images),) or not bool(torch.isfinite(angles).all()):
        raise ValueError("angles must have one finite degree value per image")
    output = torch.empty_like(images, memory_format=torch.contiguous_format)
    coordinates = torch.arange(28, dtype=images.dtype) - 13.5
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    base_grid = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(1, 784, 3)
    for start in range(0, len(images), ROTATION_CHUNK):
        if check_stop is not None:
            check_stop()
        batch = images[start:start + ROTATION_CHUNK]
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("images must be finite")
        radians = angles[start:start + len(batch)] * (math.pi / 180.)
        cosine, sine = radians.cos().to(images.dtype), radians.sin().to(images.dtype)
        theta = torch.zeros((len(batch), 2, 3), dtype=images.dtype)
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
        grid = base_grid.expand(len(batch), -1, -1).bmm(theta.transpose(1, 2) / 14.)
        # torchvision's explicit fill=[0.] also multiplies by interpolated coverage.
        padded = torch.cat((batch, torch.ones_like(batch)), dim=1)
        sampled = F.grid_sample(padded, grid.reshape(-1, 28, 28, 2), mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        output[start:start + len(batch)] = sampled[:, :1] * sampled[:, 1:]
    if check_stop is not None:
        check_stop()
    return output


def _choose(ids: Tensor, count: int, generator: torch.Generator) -> list[int]:
    if count > len(ids):
        raise ValueError("insufficient disjoint source observations")
    return ids[torch.randperm(len(ids), generator=generator)[:count]].tolist()


def plan_data(source: MNISTSource, cfg: Config, family: str, replica: int,
              partition: str) -> dict[str, Any]:
    """Create a JSON-safe immutable sampling recipe with exact novel checkpoints.

    ``partition`` is explicitly development or confirmation, independent of the
    family name. Segments use half-open indices and one-based nominal groups;
    ``novel_count`` is the cumulative count at that segment's end. Evaluation
    domains share test source IDs but have independently sampled fixed angles.
    The recorded initialization_model/factors seeds are cfg.seed_for(..., name)
    directly. Other numerical components and each pretraining epoch are named
    independently too; no model/treatment configuration enters the stream RNGs.
    """
    if partition not in ("development", "confirmation"):
        raise ValueError("partition must be development or confirmation")
    for name in ("init_size", "init_validation_size", "reference_size", "init_epochs",
                 "eval_per_class", "novel_count", "nominal_batch", "ramp_groups"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.nominal_batch > 4 or cfg.ramp_groups < 2 or not 0 < cfg.novel_probability <= 1:
        raise ValueError("invalid nominal group size or drip schedule")
    labels = torch.as_tensor(source.train_labels, dtype=torch.long, device="cpu")
    test_labels = torch.as_tensor(source.test_labels, dtype=torch.long, device="cpu")
    for values in (labels, test_labels):
        if values.ndim != 1 or not len(values) or bool(((values < 0) | (values > 9)).any()):
            raise ValueError("source labels must be a nonempty vector of digits 0 through 9")
    frequencies = torch.bincount(labels, minlength=10).double()
    if bool((frequencies == 0).any()):
        raise ValueError("all ten training digits are required")
    probabilities = frequencies / frequencies.sum()
    names = (
        "initialization_data", "initialization_angles", "validation_data", "validation_angles",
        "reference", "reference_angles", "stream_sources", "stream_digits", "stream_domains",
        "stream_angles_base", "stream_angles_novel", "evaluation_angles_base", "evaluation_angles_novel",
        "initialization_model", "initialization_order", "factors", "replay", "probes", "compression",
    )
    seeds = {name: cfg.seed_for(family, replica, name) for name in names}
    seeds["epochs"] = [cfg.seed_for(family, replica, f"epoch:{epoch}") for epoch in range(cfg.init_epochs)]
    seeds["evaluation_partition"] = cfg.seed_for("evaluation-partition", 0, "strata")
    seeds["evaluation_classes"] = [cfg.seed_for(family, replica, f"evaluation:{k}") for k in range(10)]

    def generator(name: str) -> torch.Generator:
        return torch.Generator().manual_seed(seeds[name])

    available = torch.ones(len(labels), dtype=torch.bool)
    role_ids = {}
    for role, count, seed_name in (("initialization", cfg.init_size, "initialization_data"),
                                   ("validation", cfg.init_validation_size, "validation_data"),
                                   ("reference", cfg.reference_size, "reference")):
        role_ids[role] = _choose(torch.where(available)[0], count, generator(seed_name))
        available[role_ids[role]] = False
    stream_generator = generator("stream_sources")
    buckets = [_choose(torch.where(available & (labels == k))[0],
                       int((available & (labels == k)).sum()), stream_generator) for k in range(10)]
    offsets = [0] * 10
    digit_generator, domain_generator = generator("stream_digits"), generator("stream_domains")
    stream_ids, stream_labels, stream_novel, segments = [], [], [], []
    novel_count, group = 0, 0
    while novel_count < cfg.novel_count:
        group += 1
        probability = cfg.novel_probability * min((group - 1) / (cfg.ramp_groups - 1), 1.)
        digits = torch.multinomial(probabilities, cfg.nominal_batch, replacement=True,
                                   generator=digit_generator).tolist()
        flags = (torch.rand(cfg.nominal_batch, generator=domain_generator) < probability).tolist()
        start = len(stream_ids)
        for digit, novel in zip(digits, flags, strict=True):
            if offsets[digit] == len(buckets[digit]):
                raise ValueError("stream exhausted a digit; refusing duplicate source exposures")
            stream_ids.append(buckets[digit][offsets[digit]])
            offsets[digit] += 1
            stream_labels.append(digit)
            stream_novel.append(novel)
            novel_count += int(novel)
            if novel:
                segments.append(dict(start=start, end=len(stream_ids), group=group,
                                     probability=probability, novel_count=novel_count))
                start = len(stream_ids)
            if novel_count == cfg.novel_count:
                break
        if start < len(stream_ids):
            segments.append(dict(start=start, end=len(stream_ids), group=group,
                                 probability=probability, novel_count=novel_count))

    stream_flags = torch.tensor(stream_novel, dtype=torch.bool)
    stream_angles = torch.empty(len(stream_ids), dtype=torch.float32)
    for domain, mask in (("base", ~stream_flags), ("novel", stream_flags)):
        stream_angles[mask] = sample_angles(int(mask.sum()), domain, seeds[f"stream_angles_{domain}"])
    evaluation_ids = []
    partition_generator = generator("evaluation_partition")
    for digit in range(10):
        ids = torch.where(test_labels == digit)[0]
        ids = ids[torch.randperm(len(ids), generator=partition_generator)]
        cut = len(ids) // 2
        ids = ids[:cut] if partition == "development" else ids[cut:]
        evaluation_ids.extend(_choose(ids, cfg.eval_per_class,
                                      torch.Generator().manual_seed(seeds["evaluation_classes"][digit])))
    plan = dict(schema_version=SCHEMA_VERSION, family=family, replica=replica, partition=partition,
                seeds=seeds, stream_ids=stream_ids, stream_angles=stream_angles.tolist(),
                stream_labels=stream_labels, stream_novel=stream_novel, segments=segments,
                class_probabilities=probabilities.tolist(),
                evaluation_ids={domain: list(evaluation_ids) for domain in DOMAINS},
                evaluation_angles={domain: sample_angles(len(evaluation_ids), domain,
                                    seeds[f"evaluation_angles_{domain}"]).tolist() for domain in DOMAINS})
    for role, ids in role_ids.items():
        plan[f"{role}_ids"] = ids
        plan[f"{role}_angles"] = sample_angles(len(ids), "base", seeds[f"{role}_angles"]).tolist()
    return plan


def materialize(source: MNISTSource, cfg: Config, plan: dict[str, Any],
                check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Reconstruct CPU inputs from the plan without sampling, downloading, or writing.

    ``cfg`` keeps the runner's common interface; all data choices come from the
    plan, including fixed angles. Targets are degrees, not normalized model
    units. Reference images are base-domain inputs, without observed-target
    Fisher construction. Panel digit labels are audit strata, not model inputs.
    """
    if check_stop is not None:
        check_stop()
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported regression data schema")
    if plan.get("partition") not in ("development", "confirmation"):
        raise ValueError("invalid regression data partition")

    def render(ids: list[int], degrees: list[float], *, test: bool = False) -> tuple[Tensor, Tensor]:
        if check_stop is not None:
            check_stop()
        y = torch.tensor(degrees, dtype=torch.float32)
        x = source.images(ids, test=test).to(device="cpu", dtype=torch.float32)
        return rotate_batch(x, y, check_stop), y

    result = {"plan": plan}
    for role, output in (("initialization", "pretrain"), ("validation", "validation"),
                         ("reference", "reference"), ("stream", "stream")):
        x, y = render(plan[f"{role}_ids"], plan[f"{role}_angles"])
        result[f"{output}_x"] = x
        if role != "reference":
            result[f"{output}_y"] = y
    panels = {}
    for domain in DOMAINS:
        ids = plan["evaluation_ids"][domain]
        x, y = render(ids, plan["evaluation_angles"][domain], test=True)
        digits = torch.as_tensor(source.test_labels, dtype=torch.long)[ids].clone()
        panels[domain] = dict(x=x, y=y, digits=digits,
                              weights=class_weights(digits, plan["class_probabilities"]))
    result["panels"] = panels
    return result
