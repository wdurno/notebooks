"""Compact paired grid plans with disjoint sources and separate arrival clocks.

Assets contain source IDs, forward-affine parameters and named seeds, never
rendered trajectories. Official-test IDs are offset by the training split size.
The inherited padded renderer and non-rejecting truncated-normal law are reused
unchanged; labels remain cell-local physical landmark displacements.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from mnist_experiment.diagonal_lfu.data import load_data
from mnist_experiment.fisher_memory_registration.data import (
    ANCHORS, PARAMETERS, SUPPORT_CORNERS, affine_from_parameters, encode_affine,
)
from mnist_experiment.kalman_pi.discrepancy import data as padded_data
from mnist_experiment.kalman_pi.registration.data import _labels

from .config import content_hash


DATA_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 1
PARTITION_PROTOCOL = "plan18-uniform-train-without-replacement-test-halves-v1"
CANVAS_SIZE = padded_data.CANVAS_SIZE
SUPPORT_BOUND = padded_data.SUPPORT_BOUND
TRAIN_ROLES = ("init", "validation", "reference", "diagnostic", "stream")
PANEL_ROLES = ("old", "novel", "current")
sample_transforms = padded_data.sample_transforms
render_pairs = padded_data.render_pairs
support_mask = padded_data.support_mask
load_cached = load_data


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)),
                             sort_keys=True, separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _panel_hash(panel):
    hashes = {key: _tensor_hash(value) for key, value in panel.items()
              if isinstance(value, torch.Tensor)}
    return content_hash(dict(test=panel["test"], tensors=hashes))


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = ("mnist_experiment/combined_memory/data.py", "mnist_experiment/combined_memory/model.py",
             "mnist_experiment/combined_memory/curvature.py",
             "mnist_experiment/kalman_pi/discrepancy/data.py",
             "mnist_experiment/kalman_pi/registration/data.py",
             "mnist_experiment/fisher_memory_registration/data.py",
             "mnist_experiment/fisher_memory_registration/model.py",
             "mnist_experiment/fisher_memory_grid_registration/model.py",
             "mnist_experiment/fisher_memory_stm_capacity/model.py",
             "mnist_experiment/diagonal_lfu/data.py",
             "src/mnist_data.py", "src/parameters.py", "src/seeding.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def make_schedule(cfg, stationary=False):
    """One-based ramp j=1..R; half-open zero-based batch segment bounds.

    No duplicated turning point or central plateau: outbound ends at one;
    return starts just below one. Each quiet interval has its full length.
    Stationary replicas keep all four segment boundaries and both clocks.
    """
    cfg.validate()
    if type(stationary) is not bool:
        raise ValueError("stationary must be a boolean")
    phase = torch.arange(1, cfg.ramp_steps + 1, dtype=torch.float64) / cfg.ramp_steps
    cosine = torch.cos(torch.pi * phase)
    outbound, returning = (1 - cosine) / 2, (1 + cosine) / 2
    outbound[-1], returning[-1] = 1., 0.
    parts = (("low_hold_initial", torch.zeros(cfg.hold_steps, dtype=torch.float64)),
             ("rise", outbound), ("return", returning),
             ("low_hold_final", torch.zeros(cfg.hold_steps, dtype=torch.float64)))
    segments, start = [], 0
    for name, values in parts:
        segments.append(dict(name=name, start=start, end=start + len(values),
                             arrivals=len(values), observations=len(values) * cfg.samples_per_step,
                             stationary=stationary))
        start += len(values)
    h = torch.cat([values for _, values in parts])
    return (torch.zeros_like(h) if stationary else h), segments


def make_assets(cfg, replica, cohort, raw_data):
    """Resolve treatment-independent sources/transforms without rendering.

    Training sources are one uniform draw without replacement, not a balanced
    class draw. Replicas may reuse benchmark images; roles within a replica
    cannot. Study-wide test halves are fixed before any development outcome.
    """
    cfg.validate()
    if cohort not in ("development", "main", "fresh", "confirmation", "stationary"):
        raise ValueError("unknown Plan 18 cohort")
    cfg.seed_for(cohort, replica, "replica")
    train_y, test_y = _labels(raw_data, "train"), _labels(raw_data, "test")
    sizes = dict(init=cfg.init_size, validation=cfg.validation_size, reference=cfg.reference_size,
                 diagnostic=cfg.diagnostic_size, stream=cfg.stream_size)
    required = 9 * sum(sizes.values())
    if required > len(train_y):
        raise ValueError(f"insufficient disjoint training sources: need {required}, have {len(train_y)}")
    if 27 * cfg.eval_size > len(test_y) // 2:
        raise ValueError("insufficient disjoint test sources in the assigned half")
    seeds = {}

    def seed(name):
        seeds[name] = cfg.seed_for(cohort, replica, name)
        return seeds[name]

    def generator(name):
        return torch.Generator(device="cpu").manual_seed(seed(name))

    for name in ("initialization_model", "initialization_order", "factors", "replay", "probes",
                 "numerical", "diagnostics", "bootstrap", "noise_calibration", "condition_order"):
        seed(name)
    for epoch in range(cfg.max_init_epochs):
        seed(f"initialization_order:epoch:{epoch}")
    for step in range(cfg.num_p_steps):
        seed(f"replay:{step}")
        seed(f"probes:{step}")
    base = torch.tensor([1] * 4 + [-1] * 5, dtype=torch.long)
    global_sign = -1 if replica % 2 == 0 else 1
    signs = base[torch.randperm(9, generator=generator("cell_signs"))] * global_sign
    train_ids = torch.randperm(len(train_y), generator=generator("training_sources"))[:required]
    allocation, offset = {}, 0
    for role, size in sizes.items():
        allocation[role] = train_ids[offset:offset + 9 * size].reshape(size, 9)
        offset += 9 * size

    seeds["evaluation_partition"] = cfg.seed_for("study-partition", 0, "test")
    test_order = torch.randperm(len(test_y), generator=torch.Generator(device="cpu").manual_seed(
        seeds["evaluation_partition"]))
    cut = len(test_y) // 2
    test_pool = test_order[:cut] if cohort == "development" else test_order[cut:]
    selected = test_pool[torch.randperm(len(test_pool), generator=generator("evaluation_sources"))
                         [:27 * cfg.eval_size]] + len(train_y)
    test_ids = {role: selected[i * 9 * cfg.eval_size:(i + 1) * 9 * cfg.eval_size].reshape(-1, 9)
                for i, role in enumerate(PANEL_ROLES)}
    transform_diagnostics = {}

    def panel(name, ids, means, *, test=False, uniforms=None):
        means = torch.as_tensor(means, dtype=torch.float64).expand(len(ids), 9).reshape(-1)
        transforms, info = sample_transforms(means, seed(f"{name}:transforms"),
            angle_std=cfg.angle_std, angle_limit=cfg.angle_limit,
            uniforms=None if uniforms is None else uniforms.reshape(-1, 6))
        labels = test_y[ids - len(train_y)] if test else train_y[ids]
        result = dict(ids=ids, digits=labels, transforms=transforms.reshape(-1, 9, 6), test=test,
                      uniforms=info["uniforms"].reshape(-1, 9, 6))
        result["hash"] = _panel_hash(result)
        transform_diagnostics[name] = {key: value for key, value in info.items()
                                      if key not in ("uniforms", "proposal_counts", "uniform_hash")}
        return result

    result = {role: panel(role, allocation[role], 0.) for role in TRAIN_ROLES[:-1]}
    h, segments = make_schedule(cfg, stationary=cohort == "stationary")
    p = h * cfg.novel_probability
    mixture_uniforms = torch.rand(cfg.stream_size, generator=generator("stream:mixture"), dtype=torch.float64)
    novel = mixture_uniforms < p.repeat_interleave(cfg.samples_per_step)
    means = (novel.double() * h.repeat_interleave(cfg.samples_per_step))[:, None] * cfg.angle_max * signs
    stream = panel("stream", allocation["stream"], means)
    realized = novel.reshape(cfg.num_p_steps, cfg.samples_per_step).sum(1)
    segment_ids = torch.empty(cfg.num_p_steps, dtype=torch.long)
    for i, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        segment_ids[start:end] = i
        segment.update(expected_novel=float((p[start:end] * cfg.samples_per_step).sum()),
                       realized_novel=int(realized[start:end].sum()))
    stream.update(novel=novel, mixture_uniforms=mixture_uniforms, h=h, p=p, segments=segments,
                  segment_ids=segment_ids, samples_per_step=cfg.samples_per_step,
                  cumulative_grids=torch.arange(1, cfg.num_p_steps + 1) * cfg.samples_per_step,
                  cumulative_novel=realized.cumsum(0),
                  expected_cumulative_novel=(p * cfg.samples_per_step).cumsum(0))
    stream["hash"] = _panel_hash(stream)
    result["stream"] = stream
    old = panel("old", test_ids["old"], 0., test=True)
    severity = .5 + .5 * torch.rand(cfg.eval_size, generator=generator("novel:severity"), dtype=torch.float64)
    novel_panel = panel("novel", test_ids["novel"], severity[:, None] * cfg.angle_max * signs, test=True)
    novel_panel["h"] = severity
    novel_panel["hash"] = _panel_hash(novel_panel)
    current = panel("current", test_ids["current"], 0., test=True)
    current["mixture_uniforms"] = torch.rand(cfg.eval_size, generator=generator("current:mixture"),
                                            dtype=torch.float64)
    current["novel"] = torch.zeros(cfg.eval_size, dtype=torch.bool)
    current["hash"] = _panel_hash(current)
    result.update(panels=dict(old=old, novel=novel_panel, current=current), signs=signs, seeds=seeds)
    source_hashes = _source_hashes()
    role_hashes = {role: _tensor_hash(ids) for role, ids in {**allocation, **test_ids}.items()}
    metadata = dict(
        schema_version=DATA_SCHEMA_VERSION, source_schema_version=SOURCE_SCHEMA_VERSION,
        protocol=cfg.protocol, replica=replica, cohort=cohort, partition_protocol=PARTITION_PROTOCOL,
        evaluation_role="development" if cohort == "development" else "confirmation",
        seeds=seeds, train_size=len(train_y), test_size=len(test_y), test_id_offset=len(train_y),
        assignment=dict(global_sign=global_sign, cell_signs=signs.tolist(),
                        rule="index-balanced-global-flip-of-permuted-four-positive-five-negative-v1"),
        source_sampling=dict(law="uniform_without_replacement", class_balancing=False,
                             within_replica_role_disjoint=True, cross_replica_source_reuse=True,
                             finite_population_independent=False, panel_weighting="uniform_over_grids",
                             observer_approximation="finite-population draws are not exactly independent"),
        raw_train_class_counts=torch.bincount(train_y, minlength=10).tolist(),
        role_source_counts={role: ids.numel() for role, ids in {**allocation, **test_ids}.items()},
        role_class_counts={role: torch.bincount(train_y[ids.flatten()], minlength=10).tolist()
                           for role, ids in allocation.items()},
        role_source_hashes=role_hashes, source_allocation_hash=content_hash(role_hashes),
        test_partition_hash=_tensor_hash(test_order), source_hashes=source_hashes,
        source_hash=content_hash(source_hashes),
        rendering=dict(input_shape=[2, 120, 120], grid_shape=[3, 3], cell_shape=[40, 40],
                       resized_shape=[18, 18], canvas_center=[19.5, 19.5], padding=[11] * 4,
                       cell_order="row-major", parameter_order=list(PARAMETERS),
                       target_order=[f"cell{cell}_{name}" for cell in range(9)
                                     for name in ("d1x", "d1y", "d2x", "d2y", "d3x", "d3y")],
                       target_units="cell-local pixels", anchors=[list(a) for a in ANCHORS],
                       support_corners=[list(c) for c in SUPPORT_CORNERS], support_bound=SUPPORT_BOUND,
                       angle_std=cfg.angle_std, angle_limit=cfg.angle_limit, angle_max=cfg.angle_max,
                       forward_map="R(alpha) diag(exp(a_x),exp(a_y)) shear(h) z + t",
                       angle_units="degrees", positive_angles="clockwise", blur=False,
                       query_sampling="inverse-map-bilinear", align_corners=False, antialias=True,
                       angle_law="fixed-support truncated normal; no geometric rejection"),
        schedule=dict(hash=_tensor_hash(h), arrivals=cfg.num_p_steps, grids=cfg.stream_size,
                      constituent_pairs=9 * cfg.stream_size, expected_novel=float(p.sum() * cfg.samples_per_step),
                      realized_novel=int(novel.sum()), stationary=cohort == "stationary", segments=segments),
        transform_diagnostics=transform_diagnostics)
    metadata["asset_hash"] = content_hash(dict(protocol=cfg.protocol, schema_version=DATA_SCHEMA_VERSION,
        source_schema_version=SOURCE_SCHEMA_VERSION, source_hash=metadata["source_hash"],
        source_allocation=metadata["source_allocation_hash"],
        panels={role: value["hash"] for role, value in result.items() if role in TRAIN_ROLES},
        evaluation={role: value["hash"] for role, value in result["panels"].items()},
        seeds=seeds, signs=signs.tolist()))
    result["metadata"] = metadata
    return result


def render_panel(raw, panel, indices=None, check=lambda: None):
    """Bounded CPU rendering: (N,2,120,120) inputs and (N,54) pixel targets."""
    check()
    ids, transforms = panel["ids"], panel["transforms"]
    if (ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 2
            or ids.shape[1] != 9 or transforms.shape != (*ids.shape, 6)):
        raise ValueError("grid plans require CPU int64 (N,9) IDs and (N,9,6) transforms")
    if type(panel["test"]) is not bool:
        raise ValueError("render plans require an explicit test-split flag")
    if indices is not None:
        ids, transforms = ids[indices], transforms[indices]
    ids, transforms = ids.reshape(-1, 9), transforms.reshape(-1, 9, 6)
    split = "test" if panel["test"] else "train"
    source_ids = ids - (len(raw["train_y"]) if panel["test"] else 0)
    source = raw[f"{split}_x"]
    if bool(((source_ids < 0) | (source_ids >= len(source))).any()):
        raise ValueError("source IDs do not belong to the panel's declared split")
    output = torch.empty(len(ids), 2, 120, 120, dtype=torch.float32)
    targets = torch.empty(len(ids), 54, dtype=torch.float32)
    for start in range(0, len(ids), 16):
        check()
        selected = source_ids[start:start + 16]
        matrices, translations = affine_from_parameters(transforms[start:start + len(selected)])
        images = source[selected.reshape(-1)].float()
        cells = render_pairs(images, matrices.reshape(-1, 2, 2), translations.reshape(-1, 2), check)
        output[start:start + len(selected)] = cells.reshape(-1, 3, 3, 2, 40, 40).permute(
            0, 3, 1, 4, 2, 5).reshape(-1, 2, 120, 120)
        targets[start:start + len(selected)] = encode_affine(matrices, translations).reshape(-1, 54).float()
    check()
    return output, targets


def render_batch(raw, assets, step, check=lambda: None):
    """Return one complete arrival, never splitting the novel-count clock."""
    stream = assets["stream"]
    if type(step) is not int or not 0 <= step < len(stream["h"]):
        raise ValueError("step must index an existing arrival")
    batch = stream["samples_per_step"]
    return render_panel(raw, stream, slice(step * batch, (step + 1) * batch), check)


def current_panel(assets, accepted_step, cfg):
    """Independent current-source pool at the latest accepted law, zero=nominal.

    ``accepted_step`` is the count of committed batches, not a zero-based batch
    index. Sources and mixture/transform uniforms are frozen across evaluations.
    """
    stream = assets["stream"]
    if type(accepted_step) is not int or not 0 <= accepted_step <= len(stream["h"]):
        raise ValueError("accepted_step must lie in [0, arrival_count]")
    basis = assets["panels"]["current"]
    h = 0. if accepted_step == 0 else float(stream["h"][accepted_step - 1])
    p = cfg.novel_probability * h
    novel = basis["mixture_uniforms"] < p
    means = novel.double()[:, None] * h * cfg.angle_max * assets["signs"]
    transforms, _ = sample_transforms(means.reshape(-1), assets["seeds"]["current:transforms"],
        angle_std=cfg.angle_std, angle_limit=cfg.angle_limit, uniforms=basis["uniforms"].reshape(-1, 6))
    result = dict(basis, transforms=transforms.reshape(-1, 9, 6), novel=novel,
                  h=h, p=p, accepted_step=accepted_step)
    result["hash"] = _panel_hash(result)
    return result
