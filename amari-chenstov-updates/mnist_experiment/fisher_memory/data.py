"""Paired real-MNIST inputs, FIFO audit cohorts, and held-out panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.mnist_data import load_mnist_datasets
from mnist_experiment.rotated_mnist.config import RotationConfig
from mnist_experiment.rotated_mnist.transform import rotate_mnist_batch, tensor_content_hash

from .config import Config, content_hash


ROTATION = RotationConfig((0., 30., 0., 30.), 40, "bilinear", False, 0.)


class MNISTSource:
    def __init__(self, root: Path | str) -> None:
        self.train, self.test = load_mnist_datasets(Path(root), download=False)
        self.train_labels = torch.as_tensor(self.train.targets, dtype=torch.long)
        self.test_labels = torch.as_tensor(self.test.targets, dtype=torch.long)

    def images(self, ids: list[int], *, test: bool = False) -> Tensor:
        data = self.test.data if test else self.train.data
        return data[ids].unsqueeze(1).to(torch.float32).div_(255)


def _choose(ids: Tensor, count: int, generator: torch.Generator) -> list[int]:
    if count > ids.numel():
        raise ValueError("insufficient disjoint observations")
    return ids[torch.randperm(ids.numel(), generator=generator)[:count]].tolist()


def _draw_labels(labels: Tensor, count: int, generator: torch.Generator,
                 prevalence: float | None = None) -> list[int]:
    frequencies = torch.bincount(labels, minlength=10).double()
    if prevalence is not None:
        frequencies[:9] *= (1 - prevalence) / frequencies[:9].sum()
        frequencies[9] = prevalence
    else:
        frequencies /= frequencies.sum()
    return torch.multinomial(frequencies, count, replacement=True, generator=generator).tolist()


def plan_data(source: MNISTSource, cfg: Config, family: str, replica: int, kind: str) -> dict[str, Any]:
    """Sample IDs without replacement; different treatments never resample assets."""
    if kind not in ("rotation", "mixture"):
        raise ValueError("unknown asset kind")
    component = lambda name: cfg.seed_for(family, replica, f"{kind}:{name}")
    labels = source.train_labels
    gen = lambda name: torch.Generator().manual_seed(component(name))
    initial_candidates = torch.where(labels != 9)[0] if kind == "mixture" else torch.arange(len(labels))
    initial = _choose(initial_candidates, cfg.init_size, gen("initialization_data"))
    available = torch.ones(len(labels), dtype=torch.bool)
    available[initial] = False
    reference_candidates = torch.where(available & (labels != 9 if kind == "mixture" else True))[0]
    reference = _choose(reference_candidates, cfg.reference_size, gen("reference"))
    available[reference] = False
    available_ids = torch.where(available)[0]
    stream_gen = gen("stream")
    buckets = {k: _choose(available_ids[labels[available_ids] == k],
                         int((labels[available_ids] == k).sum()), stream_gen) for k in range(10)}
    offsets = {k: 0 for k in range(10)}
    if kind == "mixture":
        values = [0., *cfg.schedule("digit9_mixture")]
        width = cfg.mixture_batch
    else:
        values = [None] * cfg.rotation_steps
        width = cfg.rotation_batch
    indices, targets = [], []
    for prevalence in values:
        row_labels = _draw_labels(labels, width, stream_gen, prevalence)
        row = []
        for k in row_labels:
            if offsets[k] >= len(buckets[k]):
                raise ValueError("stream exhausted a class; refusing duplicate unique exposures")
            row.append(buckets[k][offsets[k]])
            offsets[k] += 1
        indices.extend(row)
        targets.extend(row_labels)

    # Development and confirmation have disjoint test-image halves per class.
    eval_ids = []
    fixed_gen = torch.Generator().manual_seed(cfg.seed_for("evaluation-partition", 0, "strata"))
    for k in range(10):
        bucket = torch.where(source.test_labels == k)[0]
        bucket = bucket[torch.randperm(len(bucket), generator=fixed_gen)]
        cut = len(bucket) // 2
        bucket = bucket[cut:] if family == "confirmation" else bucket[:cut]
        eval_ids.extend(_choose(bucket, cfg.eval_per_class, gen(f"evaluation:{k}")))
    probabilities = torch.bincount(labels, minlength=10).double()
    probabilities /= probabilities.sum()
    return dict(schema_version=1, kind=kind, family=family, replica=replica,
                initialization_ids=initial, reference_ids=reference, stream_ids=indices,
                stream_labels=targets, evaluation_ids=eval_ids,
                class_probabilities=probabilities.tolist(),
                seeds={**{name: component(name) for name in ("initialization_data", "reference", "stream",
                       "initialization_model", "initialization_order", "factors")},
                       "epochs": [cfg.seed_for(family, replica, f"{kind}:epoch:{epoch}") for epoch in range(cfg.init_epochs)],
                       "evaluation_partition": cfg.seed_for("evaluation-partition", 0, "strata"),
                       "evaluation_classes": [component(f"evaluation:{k}") for k in range(10)]},
                schedules={e: cfg.schedule(e) for e in cfg.environments
                           if (e == "digit9_mixture") == (kind == "mixture")})


def materialize(source: MNISTSource, cfg: Config, plan: dict[str, Any]) -> dict[str, Any]:
    raw = source.images(plan["stream_ids"])
    labels = torch.tensor(plan["stream_labels"], dtype=torch.long)
    streams = {}
    for environment, values in plan["schedules"].items():
        if environment == "digit9_mixture":
            streams[environment] = raw
        else:
            width = cfg.rotation_batch
            streams[environment] = torch.cat([
                rotate_mnist_batch(raw[j * width:(j + 1) * width], angle, ROTATION)
                for j, angle in enumerate(values)])
    panel_ids = plan["evaluation_ids"]
    return dict(plan=plan, plan_hash=content_hash(plan), streams=streams, targets=labels,
                panel_inputs=source.images(panel_ids, test=True),
                panel_targets=source.test_labels[panel_ids].clone(),
                stream_hashes={e: tensor_content_hash(x) for e, x in streams.items()})


def class_weights(labels: Tensor, probabilities: list[float]) -> Tensor:
    counts = torch.bincount(labels, minlength=10).double()
    probs = torch.tensor(probabilities, dtype=torch.float64)
    if bool(((counts == 0) & (probs > 0)).any()):
        raise ValueError("holdout missing required class stratum")
    return probs[labels] / counts[labels].clamp_min(1)


def panel(assets: dict[str, Any], environment: str, coordinate: float) -> tuple[Tensor, Tensor, Tensor]:
    x, y = assets["panel_inputs"], assets["panel_targets"]
    probabilities = list(assets["plan"]["class_probabilities"])
    if environment == "digit9_mixture":
        denominator = sum(probabilities[:9])
        probabilities = [(1 - coordinate) * p / denominator for p in probabilities[:9]] + [coordinate]
    else:
        x = rotate_mnist_batch(x, coordinate, ROTATION)
    return x, y, class_weights(y, probabilities)


def audit_cohorts(cfg: Config, environment: str) -> list[dict[str, Any]]:
    steps = len(cfg.schedule(environment))
    width = cfg.batch_size(environment)
    burn = width if environment == "digit9_mixture" else 0
    window = min(8, max(1, steps // 5)) if cfg.smoke else 8
    result = []
    for index, fraction in enumerate((.1, .3, .5, .7)):
        end = max(window, int(round(steps * fraction)))
        start = max(0, end - window)
        ids = list(range(burn + start * width, burn + end * width))
        # Include partial-batch eviction if an explicit extension changes m.
        last_eviction_step = end + (cfg.replay_capacity + width - 1) // width
        first_eviction_step = start + 1 + cfg.replay_capacity // width
        result.append(dict(cohort_id=f"c{index + 1}", ids=ids, start_step=start + 1,
                           end_step=end, first_eviction_step=first_eviction_step,
                           eviction_step=last_eviction_step,
                           coordinates=cfg.schedule(environment)[start:end],
                           horizon=min(2 * cfg.replay_capacity, max(0, (steps - last_eviction_step) * width))))
    return result
