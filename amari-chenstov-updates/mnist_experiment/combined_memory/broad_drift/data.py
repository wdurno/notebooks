"""Whole-stream smooth rotation with an outbound-only acquisition clock.

Reuse the original disjoint source allocator, fixed holdouts and renderer.
Replace its temporary stream law and discard all Bernoulli draws before sealing
assets. Legacy ``novel`` clocks now mean outbound acquisition, not membership
of a mixture component. Return observations never advance that clock.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .. import data as original
from ..data import make_schedule, render_batch, render_panel, sample_transforms
from .config import content_hash


DATA_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = original.SOURCE_SCHEMA_VERSION
PARTITION_PROTOCOL = original.PARTITION_PROTOCOL
TRAIN_ROLES = original.TRAIN_ROLES
PANEL_ROLES = original.PANEL_ROLES
load_cached = original.load_cached


def _panel_hash(panel):
    # Unlike the legacy hash, include scalar law/clock metadata and boundaries.
    return content_hash({key: original._tensor_hash(value) if isinstance(value, torch.Tensor) else value
                         for key, value in panel.items() if key != "hash"})


def _source_hashes():
    root = Path(__file__).resolve().parents[3]
    paths = ("mnist_experiment/combined_memory/__init__.py",
             "mnist_experiment/combined_memory/config.py",
             "mnist_experiment/combined_memory/broad_drift/__init__.py",
             "mnist_experiment/combined_memory/broad_drift/config.py",
             "mnist_experiment/combined_memory/broad_drift/data.py")
    return {**original._source_hashes(),
            **{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}}


def make_assets(cfg, replica, cohort, raw_data):
    """Compact paired plans; only evaluation_partition uses the old namespace."""
    cfg.validate()
    assets = original.make_assets(cfg, replica, cohort, raw_data)
    seeds, stream = assets["seeds"], assets["stream"]
    for name in ("stream:mixture", "current:mixture"):
        seeds.pop(name)
    stream.pop("mixture_uniforms")
    basis = assets["panels"]["current"]
    basis.pop("mixture_uniforms")
    basis.pop("novel")
    basis.update(h=0., p=0., accepted_step=0)

    h, segments = make_schedule(cfg, stationary=cohort == "stationary")
    means = h.repeat_interleave(cfg.samples_per_step)[:, None] * cfg.angle_max * assets["signs"]
    transforms, info = sample_transforms(means.reshape(-1), seeds["stream:transforms"],
        angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
        uniforms=stream["uniforms"].reshape(-1, 6))
    acquisition = torch.zeros(cfg.stream_size, dtype=torch.bool)
    start, end = cfg.hold_steps, cfg.hold_steps + cfg.ramp_steps
    if cohort != "stationary":
        acquisition[start * cfg.samples_per_step:end * cfg.samples_per_step] = True
    counts = acquisition.reshape(cfg.num_p_steps, cfg.samples_per_step).sum(1)
    cumulative = counts.cumsum(0)
    for segment in segments:
        count = int(counts[segment["start"]:segment["end"]].sum())
        segment.update(outbound_count=count, acquisition_count=count,
                       expected_novel=float(count), realized_novel=count)
    stream.update(transforms=transforms.reshape(-1, 9, 6), h=h, p=(h > 0).double(),
                  segments=segments, acquisition=acquisition, novel=acquisition,
                  outbound_count=int(acquisition.sum()), cumulative_acquisition=cumulative,
                  cumulative_novel=cumulative, expected_cumulative_novel=cumulative.double())
    for role in TRAIN_ROLES:
        assets[role]["hash"] = _panel_hash(assets[role])
    for panel in assets["panels"].values():
        panel["hash"] = _panel_hash(panel)

    metadata = assets["metadata"]
    metadata.pop("asset_hash")
    metadata["transform_diagnostics"]["stream"] = {
        key: value for key, value in info.items()
        if key not in ("uniforms", "proposal_counts", "uniform_hash")}
    metadata.update(schema_version=DATA_SCHEMA_VERSION, source_schema_version=SOURCE_SCHEMA_VERSION,
        config_hash=cfg.scientific_hash, source_hashes=_source_hashes(),
        panel_hash_protocol="tensor-content-and-scalar-metadata-v1",
        stream_law=dict(name="whole-stream-truncated-normal-v1", bernoulli_component=False,
                        angle_mean="cell_sign * angle_max * h", angle_std=cfg.angle_std,
                        angle_limit=cfg.angle_limit,
                        p_meaning="indicator(h > 0); whole changing law, not acquisition"),
        evaluation_partition=dict(seed=seeds["evaluation_partition"], seed_root=181106,
            protocol="plan18-combined-memory-v1", cohort="study-partition", replica=0,
            component="test", preserved_from_original_study=True),
        acquisition=dict(unit="unique_outbound_grids", mask="stream.acquisition",
            cumulative="stream.cumulative_acquisition", budget=cfg.outbound_budget,
            outbound_count=stream["outbound_count"], start=start, end=end,
            bounds="half-open zero-based batches", zero_accepted_step=start,
            endpoint_accepted_step=end, return_advances_clock=False,
            applicable=cohort != "stationary"),
        legacy_aliases={
            "config.novel_probability": "unused allocator compatibility field; not a law probability",
            "config.analysis_novel_count": "acquisition.budget",
            "stream.novel": "stream.acquisition; outbound mask, not law membership",
            "stream.cumulative_novel": "stream.cumulative_acquisition",
            "stream.expected_cumulative_novel": "deterministic cumulative acquisition (float64)",
            "schedule.expected_novel": "schedule.outbound_count (float)",
            "schedule.realized_novel": "schedule.outbound_count",
            "segments.expected_novel": "segments.acquisition_count (float)",
            "segments.realized_novel": "segments.acquisition_count",
            "trajectory.novel_count": "cumulative outbound acquisition; zero for stationary"})
    metadata["source_hash"] = content_hash(metadata["source_hashes"])
    metadata["schedule"].update(hash=original._tensor_hash(h), segments=segments,
        outbound_count=stream["outbound_count"], expected_novel=float(stream["outbound_count"]),
        realized_novel=stream["outbound_count"], acquisition_hash=original._tensor_hash(acquisition))
    metadata["asset_hash"] = content_hash(dict(metadata=metadata,
        panels={role: assets[role]["hash"] for role in TRAIN_ROLES},
        evaluation={role: panel["hash"] for role, panel in assets["panels"].items()},
        signs=original._tensor_hash(assets["signs"])))
    return assets


def current_panel(assets, accepted_step, cfg):
    """Whole current law at a committed batch, with frozen sources/uniforms.

    ``p`` indicates nonnominal means only. Holdouts have no acquisition mask.
    ``accepted_step=0`` is the nominal frozen-initialization reference.
    """
    stream = assets["stream"]
    if type(accepted_step) is not int or not 0 <= accepted_step <= len(stream["h"]):
        raise ValueError("accepted_step must lie in [0, arrival_count]")
    basis = assets["panels"]["current"]
    h = 0. if accepted_step == 0 else float(stream["h"][accepted_step - 1])
    means = (h * cfg.angle_max * assets["signs"]).expand(len(basis["ids"]), 9)
    transforms, _ = sample_transforms(means.reshape(-1), assets["seeds"]["current:transforms"],
        angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
        uniforms=basis["uniforms"].reshape(-1, 6))
    result = dict(basis, transforms=transforms.reshape(-1, 9, 6), h=h,
                  p=float(h > 0), accepted_step=accepted_step)
    result["hash"] = _panel_hash(result)
    return result
