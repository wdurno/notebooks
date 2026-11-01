"""CPU-only paired digit-9 assets, with fixed study-wide role isolation.

Training IDs index ``data['train_x']``. Test IDs are offset by the training
size, so all role memberships inhabit a single collision-free ID space.
Streams store (arrival, observation) IDs/labels rather than duplicate images;
materialize a batch with ``data['train_x'][assets['stream_ids'][t]]``.
Nothing here trains, downloads, uses CUDA, writes artifacts, or advances a
global random generator. The caller seals the returned assets immutably.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.mnist_data import load_mnist_datasets

from .config import Config, content_hash


DATA_SCHEMA_VERSION = 1
PARTITION_PROTOCOL = "stratified-train-20-10-20-50-test-halves-v1"
TRAIN_ROLES = ("initialization", "validation", "reference", "online")
EVAL_ROLES = ("evaluation_development", "evaluation_fresh")


def load_data(data_root: Path) -> dict[str, Tensor]:
    """Load cached canonical MNIST as CPU NCHW float32 [0,1] and long labels.

    A missing cache raises the dataset loader's error; downloading is forbidden.
    In-memory callers can supply the same four-tensor dictionary directly.
    """
    train, test = load_mnist_datasets(Path(data_root), download=False)
    return {
        "train_x": train.data.cpu().unsqueeze(1).to(torch.float32).div_(255),
        "train_y": torch.as_tensor(train.targets, dtype=torch.long, device="cpu"),
        "test_x": test.data.cpu().unsqueeze(1).to(torch.float32).div_(255),
        "test_y": torch.as_tensor(test.targets, dtype=torch.long, device="cpu"),
    }


def _validate_data(data: dict[str, Tensor]) -> None:
    for split in ("train", "test"):
        x, y = data[f"{split}_x"], data[f"{split}_y"]
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise ValueError("data must contain CPU tensors")
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise ValueError("asset generation accepts CPU data only")
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28) or not x.is_floating_point():
            raise ValueError("images must be floating NCHW tensors with shape (N,1,28,28)")
        if y.dtype != torch.long or y.ndim != 1 or len(y) != len(x) or not len(y):
            raise ValueError("labels must be nonempty long vectors matching the images")
        if bool(((y < 0) | (y > 9)).any()) or bool((torch.bincount(y, minlength=10) == 0).any()):
            raise ValueError("both source splits must contain all ten digits")
        # Bound temporary validation masks independently of the full MNIST size.
        for chunk in x.split(512):
            if not bool(torch.isfinite(chunk).all()) or bool(((chunk < 0) | (chunk > 1)).any()):
                raise ValueError("images must be finite and normalized to [0,1]")


def _partitions(data: dict[str, Tensor], cfg: Config) -> tuple[dict[str, Tensor], dict[str, int]]:
    # Study seeds never depend on replica, cohort, treatment, or configured sizes.
    seeds = {name: cfg.seed_for("study-partition", 0, name) for name in ("train", "test")}
    buckets: dict[str, list[Tensor]] = {role: [] for role in (*TRAIN_ROLES, *EVAL_ROLES)}
    for split in ("train", "test"):
        generator = torch.Generator(device="cpu").manual_seed(seeds[split])
        labels = data[f"{split}_y"]
        for digit in range(10):
            ids = torch.where(labels == digit)[0]
            ids = ids[torch.randperm(len(ids), generator=generator)]
            if split == "train":
                sizes = (max(1, len(ids) // 5), max(1, len(ids) // 10), max(1, len(ids) // 5))
                if sum(sizes) >= len(ids):
                    raise ValueError("each training class needs at least four images for role isolation")
                start = 0
                for role, size in zip(TRAIN_ROLES, (*sizes, len(ids) - sum(sizes)), strict=True):
                    buckets[role].append(ids[start:start + size])
                    start += size
            else:
                if len(ids) < 2:
                    raise ValueError("each test class needs at least two images for dev/fresh isolation")
                ids = ids + len(data["train_y"])
                cut = len(ids) // 2
                buckets[EVAL_ROLES[0]].append(ids[:cut])
                buckets[EVAL_ROLES[1]].append(ids[cut:])
    return {role: torch.cat(parts) for role, parts in buckets.items()}, seeds


def make_schedule(cfg: Config, stationary: bool = False) -> tuple[Tensor, list[dict[str, Any]]]:
    """Return exact float64 prevalences and zero-based, half-open segments.

    Each ramp includes both endpoints, including the adjacent hold's endpoint.
    The underlying half cosine has zero endpoint slope; no boundary is hidden
    by shortening a hold or inserting an extra arrival.
    """
    cfg.validate()
    if stationary:
        return torch.full((cfg.num_p_steps,), cfg.p_low, dtype=torch.float64), [
            dict(name="stationary_low", start=0, end=cfg.num_p_steps)
        ]
    phase = torch.linspace(0., 1., cfg.ramp_steps, dtype=torch.float64)
    rise = cfg.p_low + (cfg.p_high - cfg.p_low) * (1 - torch.cos(torch.pi * phase)) / 2
    rise[0], rise[-1] = cfg.p_low, cfg.p_high
    parts = (
        ("low_hold_initial", torch.full((cfg.hold_steps,), cfg.p_low, dtype=torch.float64)),
        ("rise", rise),
        ("high_hold", torch.full((cfg.hold_steps,), cfg.p_high, dtype=torch.float64)),
        ("return", rise.flip(0)),
        ("low_hold_final", torch.full((cfg.hold_steps,), cfg.p_low, dtype=torch.float64)),
    )
    start, segments = 0, []
    for name, values in parts:
        segments.append(dict(name=name, start=start, end=start + len(values)))
        start += len(values)
    return torch.cat([values for _, values in parts]), segments


def make_assets(cfg: Config, replica: int, cohort: str, data: dict[str, Tensor]) -> dict[str, Any]:
    """Materialize treatment-independent assets and a JSON-safe audit manifest.

    ``cohort`` is development, main/fresh, or stationary (fresh evaluation).
    Initialization, validation, and reference use p_low and no replacement;
    only online sampling permits replacement. Non-9 probabilities are frozen
    from *raw training* counts, never recomputed after role partitioning.
    Reusing this dictionary pairs every treatment, regardless of its action.
    """
    cfg.validate()
    if cohort not in ("development", "main", "fresh", "stationary"):
        raise ValueError("cohort must be development, main, fresh, or stationary")
    cfg.seed_for(cohort, replica, "replica")  # Validate the replica before sampling.
    _validate_data(data)
    partitions, partition_seeds = _partitions(data, cfg)
    raw_counts = torch.bincount(data["train_y"], minlength=10)
    non9 = raw_counts[:9].double() / raw_counts[:9].sum()
    seeds: dict[str, int] = {}

    def generator(name: str) -> torch.Generator:
        seeds[name] = cfg.seed_for(cohort, replica, name)
        return torch.Generator(device="cpu").manual_seed(seeds[name])

    # These are consumed by the parent learner/analysis, not by data sampling.
    for name in ("initialization_model", "initialization_order", "numerical", "bootstrap"):
        seeds[name] = cfg.seed_for(cohort, replica, name)

    def sample(role: str, p: Tensor, name: str, replacement: bool = False) -> tuple[Tensor, Tensor]:
        flags = torch.rand(p.numel(), generator=generator(f"{name}:mixture"), dtype=torch.float64) < p.flatten()
        labels = torch.multinomial(non9, p.numel(), replacement=True, generator=generator(f"{name}:non9"))
        labels[flags] = 9
        ids = torch.empty_like(labels)
        pool = partitions[role]
        for digit in range(10):
            positions = torch.where(labels == digit)[0]
            candidates = pool[data["train_y"][pool] == digit]
            rng = generator(f"{name}:inputs:{digit}")
            if replacement:
                selected = torch.randint(len(candidates), (len(positions),), generator=rng)
            else:
                if len(positions) > len(candidates):
                    raise ValueError(f"{role} partition has insufficient digit {digit} images; "
                                     "increase source data or reduce the requested sample")
                selected = torch.randperm(len(candidates), generator=rng)[:len(positions)]
            ids[positions] = candidates[selected]
        return ids.reshape(p.shape), labels.reshape(p.shape)

    result: dict[str, Any] = {"partitions": partitions, "non9_probabilities": non9}
    role_counts = {}
    for name, role, size in (("init", "initialization", cfg.initialization_size),
                             ("validation", "validation", cfg.validation_size),
                             ("reference", "reference", cfg.reference_size)):
        p = torch.full((size,), cfg.p_low, dtype=torch.float64)
        ids, labels = sample(role, p, name)
        result[f"{name}_ids"], result[f"{name}_y"] = ids, labels
        result[f"{name}_x"] = data["train_x"][ids].clone()
        role_counts[name] = torch.bincount(labels, minlength=10).tolist()

    eval_role = EVAL_ROLES[0] if cohort == "development" else EVAL_ROLES[1]
    test_offset = len(data["train_y"])
    eval_parts = []
    for digit in range(10):
        candidates = partitions[eval_role]
        candidates = candidates[data["test_y"][candidates - test_offset] == digit]
        if cfg.eval_per_class > len(candidates):
            raise ValueError(f"{eval_role} has fewer than eval_per_class={cfg.eval_per_class} digit {digit} images")
        order = torch.randperm(len(candidates), generator=generator(f"evaluation:{digit}"))
        eval_parts.append(candidates[order[:cfg.eval_per_class]])
    result["eval_ids"] = torch.cat(eval_parts)
    result["eval_source_ids"] = result["eval_ids"] - test_offset
    result["eval_x"] = data["test_x"][result["eval_source_ids"]].clone()
    result["eval_y"] = data["test_y"][result["eval_source_ids"]].clone()

    stream_metadata = {}
    for prefix, stationary in (("stream", False), ("stationary_stream", True)):
        p, segments = make_schedule(cfg, stationary=stationary)
        ids, labels = sample("online", p[:, None].expand(-1, cfg.samples_per_step), prefix, replacement=True)
        realized = (labels == 9).sum(dim=1)
        expected = p * cfg.samples_per_step
        seen: set[int] = set()
        unique_counts = []
        for row in ids.tolist():
            seen.update(row)
            unique_counts.append(len(seen))
        segment_ids = torch.empty(cfg.num_p_steps, dtype=torch.long)
        for index, segment in enumerate(segments):
            start, end = segment["start"], segment["end"]
            segment_ids[start:end] = index
            segment.update(arrivals=end - start, observations=(end - start) * cfg.samples_per_step,
                           expected_9s=float(expected[start:end].sum()),
                           realized_9s=int(realized[start:end].sum()),
                           unique_ids=int(ids[start:end].unique().numel()))
        result.update({
            f"{prefix}_ids": ids, f"{prefix}_y": labels, f"{prefix}_p": p,
            f"{prefix}_segments": segments, f"{prefix}_segment_ids": segment_ids,
            f"{prefix}_arrival_ids": torch.arange(cfg.num_p_steps),
            f"{prefix}_expected_cumulative_9s": expected.cumsum(0),
            f"{prefix}_realized_cumulative_9s": realized.cumsum(0),
            f"{prefix}_cumulative_unique_ids": torch.tensor(unique_counts),
        })
        stream_metadata[prefix] = dict(
            schedule_hash=content_hash(p.tolist()),
            stream_hash=content_hash(dict(ids=ids.tolist(), labels=labels.tolist(), p=p.tolist())),
            arrival_count=cfg.num_p_steps, observation_count=ids.numel(),
            unique_id_count=len(seen), expected_9s=float(expected.sum()),
            realized_9s=int(realized.sum()), class_counts=torch.bincount(labels.flatten(), minlength=10).tolist(),
            replacement=True, segments=segments,
        )
    result["seeds"] = seeds
    result["metadata"] = dict(
        schema_version=DATA_SCHEMA_VERSION, protocol=cfg.protocol,
        partition_protocol=PARTITION_PROTOCOL, partition_seeds=partition_seeds,
        partition_hash=content_hash({role: ids.tolist() for role, ids in partitions.items()}),
        replica=replica, cohort=cohort, evaluation_role=eval_role, seeds=seeds,
        train_size=test_offset, test_size=len(data["test_y"]), test_id_offset=test_offset,
        image_contract="NCHW-[0,1]-no-augmentation-v1", raw_train_class_counts=raw_counts.tolist(),
        non9_probabilities=non9.tolist(), initial_prevalence=cfg.p_low,
        role_class_counts=role_counts, streams=stream_metadata,
    )
    return result
