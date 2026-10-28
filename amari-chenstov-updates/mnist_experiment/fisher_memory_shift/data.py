"""Paired continuous-shift plans and deterministic, cache-friendly CPU rendering.

Plans contain IDs, degree targets and per-image strengths, never model inputs.
Training-role IDs index the official training split; evaluation IDs index the
official test split. Novel membership is sampling metadata, not a feature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from mnist_experiment.fisher_memory.data import MNISTSource, class_weights
from mnist_experiment.fisher_memory_regression import data as regression_data
from mnist_experiment.fisher_memory_regression.data import rotate_batch

from . import PROTOCOL
from .config import Config, Context, default_contexts

SCHEMA_VERSION = 1
DOMAINS = ("base", "novel")
TRANSFORM_CHUNK = 256
BLUR_RADIUS = 6


def sample_angles(count: int, domain: str, seed: int) -> Tensor:
    """Draw prefix-stable CPU float32 Uniform[-60,60] degrees for either domain."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if domain not in DOMAINS:
        raise ValueError("domain must be base or novel")
    generator = torch.Generator().manual_seed(seed)
    return (120 * torch.rand(count, generator=generator, dtype=torch.float64) - 60).float()


def drift_strength(group: int) -> float:
    """Smooth strength on original one-based nominal groups, not split segments."""
    if type(group) is not int or group < 1:
        raise ValueError("group must be a positive integer")
    v = min((group - 1) / 199., 1.)
    return .5 + .5 * (3 * v * v - 2 * v * v * v)


def _uniform_strengths(count: int, seed: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    return (.5 + .5 * torch.rand(count, generator=generator, dtype=torch.float64)).tolist()


def plan_data(source: MNISTSource, cfg: Config, family: str, replica: int,
              partition: str) -> dict[str, Any]:
    """Plan all paired assets before treatment, independent of context and stage.

    Reuse the established source/drip/test-pool sampler, replacing every old
    angle law explicitly. Diagnostic IDs are drawn from unused training sources
    and cannot affect the online digit/domain RNGs. Epoch seeds cover the full
    extension ceiling, so choosing a later common epoch stage cannot resample.
    """
    cfg.validate()
    family = family.split("--", 1)[0]
    plan = regression_data.plan_data(source, cfg, family, replica, partition)
    plan["protocol"] = PROTOCOL
    plan["schema_version"] = SCHEMA_VERSION
    plan["rendering"] = dict(angle_law="uniform-degrees", angle_min=-60., angle_max=60.,
                             order=["rotation", "gamma", "blur"], blur_radius=BLUR_RADIUS,
                             blur_padding="zeros", strength_groups=200)
    seeds = plan["seeds"]
    seeds["epochs"] = [cfg.seed_for(family, replica, f"epoch:{epoch}")
                       for epoch in range(cfg.max_init_epochs)]
    for role in ("initialization", "validation", "reference"):
        plan[f"{role}_angles"] = sample_angles(
            len(plan[f"{role}_ids"]), "base", seeds[f"{role}_angles"]).tolist()
    flags = torch.tensor(plan["stream_novel"], dtype=torch.bool)
    stream_angles = torch.empty(len(flags), dtype=torch.float32)
    for domain, mask in (("base", ~flags), ("novel", flags)):
        stream_angles[mask] = sample_angles(int(mask.sum()), domain, seeds[f"stream_angles_{domain}"])
    plan["stream_angles"] = stream_angles.tolist()
    groups = [index // cfg.nominal_batch + 1 for index in range(len(flags))]
    plan["stream_groups"] = groups
    plan["stream_strengths"] = [drift_strength(group) if novel else 0.
                                for group, novel in zip(groups, plan["stream_novel"], strict=True)]
    for segment in plan["segments"]:
        segment["strength"] = drift_strength(segment["group"])

    # Both fixed panels share source IDs AND angles; only the rendering differs.
    for name in ("evaluation_angles_base", "evaluation_angles_novel"):
        seeds.pop(name)
    for name in ("evaluation_angles", "evaluation_strengths", "diagnostic_base_data",
                 "diagnostic_novel_data", "diagnostic_base_angles", "diagnostic_novel_angles",
                 "diagnostic_novel_strengths"):
        seeds[name] = cfg.seed_for(family, replica, name)
    seeds["diagnostic_epochs"] = [cfg.seed_for(family, replica, f"diagnostic_order:{epoch}")
                                  for epoch in range(cfg.max_diagnostic_epochs)]
    count = len(plan["evaluation_ids"]["base"])
    angles = sample_angles(count, "base", seeds["evaluation_angles"]).tolist()
    plan["evaluation_angles"] = {domain: list(angles) for domain in DOMAINS}
    plan["evaluation_strengths"] = dict(base=[0.] * count,
        novel=_uniform_strengths(count, seeds["evaluation_strengths"]))

    available = torch.ones(len(source.train_labels), dtype=torch.bool)
    for role in ("initialization", "validation", "reference", "stream"):
        available[plan[f"{role}_ids"]] = False
    for domain, count in (("base", cfg.diagnostic_base_size), ("novel", cfg.diagnostic_novel_size)):
        role = f"diagnostic_{domain}"
        generator = torch.Generator().manual_seed(seeds[f"{role}_data"])
        ids = regression_data._choose(torch.where(available)[0], count, generator)
        available[ids] = False
        plan[f"{role}_ids"] = ids
        plan[f"{role}_angles"] = sample_angles(count, domain, seeds[f"{role}_angles"]).tolist()
        plan[f"{role}_strengths"] = ([0.] * count if domain == "base" else
                                     _uniform_strengths(count, seeds[f"{role}_strengths"]))
    return plan


def transform(images: Tensor, context: Context | None, strengths: Tensor,
              check_stop: Callable[[], None] | None = None) -> Tensor:
    """Apply gamma then continuous-sigma separable blur to already rotated pixels.

    Kernels have fixed radius six and unit mass, with zero padding. Per-image
    grouped convolutions avoid quantizing strengths or materializing a kernel
    per pixel. Inputs are not mutated or renormalized. Zero strength, zero blur,
    and an absent context are exact identities.
    """
    if check_stop is not None:
        check_stop()
    if images.device.type != "cpu" or images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("images must be CPU tensors of shape (N, 1, 28, 28)")
    if images.dtype not in (torch.float32, torch.float64):
        raise ValueError("images must have float32 or float64 dtype")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("images must contain finite intensities in [0,1]")
    strengths = torch.as_tensor(strengths, dtype=torch.float64, device="cpu")
    if strengths.ndim == 0:
        strengths = strengths.expand(len(images))
    if (strengths.shape != (len(images),) or not bool(torch.isfinite(strengths).all())
            or bool(((strengths < 0) | (strengths > 1)).any())):
        raise ValueError("strengths must have one finite value in [0,1] per image")
    if context is not None and not isinstance(context, Context):
        raise ValueError("context must be a Context or None")
    if context is None or not len(images) or not bool((strengths > 0).any()):
        return images
    output = torch.empty_like(images, memory_format=torch.contiguous_format)
    coordinates = torch.arange(-BLUR_RADIUS, BLUR_RADIUS + 1, dtype=torch.float64)
    for start in range(0, len(images), TRANSFORM_CHUNK):
        if check_stop is not None:
            check_stop()
        batch = images[start:start + TRANSFORM_CHUNK]
        strength = strengths[start:start + len(batch)]
        if context.gamma_max != 1:
            gamma = torch.pow(context.gamma_max, strength).to(images.dtype)
            batch = batch.pow(gamma[:, None, None, None])
        if context.blur_max != 0:
            sigma = context.blur_max * strength
            safe_sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
            kernels = torch.exp(-.5 * (coordinates[None, :] / safe_sigma[:, None]).square())
            kernels[sigma == 0] = (coordinates == 0).double()
            kernels = (kernels / kernels.sum(dim=1, keepdim=True)).to(images.dtype)
            count = len(batch)
            horizontal = F.conv2d(batch.reshape(1, count, 28, 28),
                kernels.reshape(count, 1, 1, -1), padding=(0, BLUR_RADIUS), groups=count)
            batch = F.conv2d(horizontal, kernels.reshape(count, 1, -1, 1),
                             padding=(BLUR_RADIUS, 0), groups=count).reshape_as(batch)
        output[start:start + len(batch)] = batch
    if check_stop is not None:
        check_stop()
    return output


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("protocol") != PROTOCOL:
        raise ValueError("unsupported continuous-shift data protocol/schema")
    if plan.get("partition") not in ("development", "confirmation"):
        raise ValueError("invalid continuous-shift data partition")


def contextualize(base_data: dict[str, Any], cfg: Config, context: Context | None,
                  check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Share nominal assets and replace only novel stream, panel and diagnostic x.

    Always pass the nominal result of ``materialize(..., context=None)`` here;
    applying a context to an already contextualized cache is rejected.
    """
    if check_stop is not None:
        check_stop()
    plan = base_data["plan"]
    _validate_plan(plan)
    if base_data.get("context") is not None:
        raise ValueError("contextualize requires a nominal cache")
    if context is not None and not isinstance(context, Context):
        raise ValueError("context must be a Context or None")
    result = dict(base_data)
    result["context"] = context.mapping() if context is not None else None
    if context is None:
        return result
    flags = torch.tensor(plan["stream_novel"], dtype=torch.bool)
    strengths = torch.tensor(plan["stream_strengths"], dtype=torch.float64)
    stream = base_data["stream_x"].clone()
    stream[flags] = transform(stream[flags], context, strengths[flags], check_stop)
    result["stream_x"] = stream
    panels = dict(base_data["panels"])
    panels["novel"] = dict(panels["novel"])
    panels["novel"]["x"] = transform(panels["novel"]["x"], context,
                                     plan["evaluation_strengths"]["novel"], check_stop)
    result["panels"] = panels
    result["diagnostic_novel_x"] = transform(base_data["diagnostic_novel_x"], context,
                                             plan["diagnostic_novel_strengths"], check_stop)
    return result


def materialize(source: MNISTSource, cfg: Config, plan: dict[str, Any],
                context: Context | None = None,
                check_stop: Callable[[], None] | None = None) -> dict[str, Any]:
    """Render nominal CPU float32 assets once, optionally applying a context.

    Degree targets are unchanged by rendering. The old runner keys are retained;
    diagnostic_base_x/y and diagnostic_novel_x/y are separate learnability data.
    This function never downloads, writes, samples, or changes the plan.
    """
    if check_stop is not None:
        check_stop()
    _validate_plan(plan)

    def render(ids, degrees, *, test=False):
        if check_stop is not None:
            check_stop()
        y = torch.tensor(degrees, dtype=torch.float32)
        x = source.images(ids, test=test).to(device="cpu", dtype=torch.float32)
        return rotate_batch(x, y, check_stop), y

    result = dict(plan=plan, context=None)
    for role, output in (("initialization", "pretrain"), ("validation", "validation"),
                         ("reference", "reference"), ("stream", "stream"),
                         ("diagnostic_base", "diagnostic_base"),
                         ("diagnostic_novel", "diagnostic_novel")):
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
    return contextualize(result, cfg, context, check_stop) if context is not None else result
