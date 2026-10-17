"""Paired source fitting, disjoint data roles, and resumable initial Fisher assets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from mnist_experiment.rotated_mnist.config import RotationConfig
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model
from mnist_experiment.rotated_mnist.schedule import resolve_shaped_rotation_schedule
from mnist_experiment.rotated_mnist.transform import rotate_mnist_batch, tensor_content_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.seeding import derive_component_seed

from .artifacts import UnitStore, file_hash
from .config import REPO_ROOT, Study, canonical_hash
from .evaluation import evaluate, source_quality
from .fisher import model_fisher
from .model import build_model
from .optimizer import fit_restricted


ASSET_REQUIRED = ("assets.pt", "summary.json", "metrics.json")
SOURCE_REQUIRED = (*ASSET_REQUIRED, "source_checkpoint.pt", "data_plan.pt")
SEED_COMPONENTS = (
    "initialization", "partition", "evaluation", "source_order", "source_rng",
    "warmup", "stream_membership", "stream_non_nine", "stream_nine",
    "rotation_observations", "reference", "model_labels", "numerical", "bootstrap",
)


def named_seeds(study: Study, phase: str, index: int, family: str) -> dict[str, int]:
    """No architecture width, schedule, or exposure branch enters paired seeds."""
    return {name: study.seed(f"{phase}:{family}:{name}", index) for name in SEED_COMPONENTS}


def _cpu(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _json(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def partition_ids(study: Study, train_targets: Tensor, test_targets: Tensor, *,
                  family: str, seeds: dict[str, int]) -> dict[str, Any]:
    """Reserve exact-size source pools, then disjoint online/reference/dev pools.

    Mixture source slots are selected among non-nines, not selected among all
    digits and silently shortened by filtering. Test IDs use a train-size offset
    so identities remain unambiguous across the two MNIST dataset splits.
    """
    if family not in {"rotation", "mixture"}:
        raise ValueError("unknown environment family")
    rng = _generator(seeds["partition"])
    all_ids = torch.arange(len(train_targets))
    eligible = all_ids if family == "rotation" else all_ids[train_targets != 9]
    source_count = study.source_size + study.source_validation_size
    total = source_count + study.online_pool_size + study.reference_pool_size + study.development_evaluation_size
    if source_count > len(eligible) or total > len(train_targets):
        raise ValueError("insufficient MNIST observations for disjoint configured partitions")
    source = eligible[torch.randperm(len(eligible), generator=rng)[:source_count]]
    available = torch.ones(len(train_targets), dtype=torch.bool)
    available[source] = False
    remaining = all_ids[available]
    remaining = remaining[torch.randperm(len(remaining), generator=rng)]
    online_stop = study.online_pool_size
    reference_stop = online_stop + study.reference_pool_size
    development_stop = reference_stop + study.development_evaluation_size
    if study.final_evaluation_size > len(test_targets):
        raise ValueError("final evaluation size exceeds the MNIST test split")
    test_ids = torch.randperm(len(test_targets), generator=_generator(seeds["evaluation"]))[
        :study.final_evaluation_size]
    ids = {
        "source": source[:study.source_size],
        "source_validation": source[study.source_size:],
        "online": remaining[:online_stop],
        "reference": remaining[online_stop:reference_stop],
        "development_evaluation": remaining[reference_stop:development_stop],
        "final_evaluation": test_ids + len(train_targets),
    }
    return {"ids": ids, "hash": canonical_hash({key: tensor_content_hash(value)
                                                 for key, value in ids.items()}),
            "train_size": len(train_targets), "test_size": len(test_targets),
            "id_convention": "train index; test index plus train_size",
            "seeds": seeds,
            "class_counts": {key: torch.bincount(
                test_targets[value - len(train_targets)] if key == "final_evaluation" else train_targets[value],
                minlength=10).tolist() for key, value in ids.items()}}


def _draw_pool(pool: Tensor, count: int, seed: int) -> Tensor:
    """Shuffle without replacement, restarting only after pool exhaustion."""
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    if not len(pool):
        raise ValueError("cannot draw observations from an empty class pool")
    rng = _generator(seed)
    parts = []
    for start in range(0, count, len(pool)):
        parts.append(pool[torch.randperm(len(pool), generator=rng)[:min(len(pool), count - start)]])
    return torch.cat(parts)


def _exposure(ids: Tensor, labels: Tensor) -> dict[str, Any]:
    count, unique = ids.numel(), ids.unique().numel()
    return {"presentations": count, "unique_observations": unique,
            "duplicates": count - unique, "duplicate_fraction": 1 - unique / count,
            "realized_nines": int((labels == 9).sum()),
            "observation_ids_hash": tensor_content_hash(ids),
            "targets_hash": tensor_content_hash(labels)}


def stream_ids(study: Study, targets: Tensor, partitions: dict, *, family: str,
               seeds: dict[str, int]) -> dict[str, Any]:
    """Resolve paired identities without any model-dependent label selection."""
    pool = partitions["ids"]["online"]
    if family == "rotation":
        rotation = RotationConfig((0.0, 30.0, 0.0, 30.0),
                                  study.rotation_transitions_per_arrow, "bilinear", False, 0.0)
        count = 3 * study.rotation_transitions_per_arrow
        ids = _draw_pool(pool, count * study.rotation_samples,
                         seeds["rotation_observations"]).reshape(count, study.rotation_samples)
        streams = {}
        for kind in ("linear", "sigmoid"):
            schedule = resolve_shaped_rotation_schedule(rotation, kind=kind, sigmoid_kappa=8.0)
            # The source model is at angle zero; all 120 transitions are arrivals.
            positions = list(schedule.angles_degrees[1:])
            streams[f"rotation_{kind}"] = {
                "observation_ids": ids.clone(), "targets": targets[ids], "positions": positions,
                "schedule": schedule.to_mapping(), "schedule_hash": schedule.content_hash,
                "exposure": _exposure(ids, targets[ids]), "rotation": rotation.to_mapping(),
            }
        return {"streams": streams, "warmup_ids": torch.empty(0, dtype=torch.long)}
    if family != "mixture":
        raise ValueError("unknown environment family")
    non_nine = pool[targets[pool] != 9]
    warmup = _draw_pool(non_nine, study.warmup_samples, seeds["warmup"])
    # Warm-up IDs cannot reappear as treatment observations, even after exhaustion.
    non_nine = non_nine[~torch.isin(non_nine, warmup)]
    sizes = sorted(set([*study.mixture_samples, study.scarcity_samples]))
    maximum = max(sizes)
    positions = [0.1 * (step + 1) / study.mixture_steps for step in range(study.mixture_steps)]
    uniforms = torch.rand((study.mixture_steps, maximum), dtype=torch.float64,
                          generator=_generator(seeds["stream_membership"]))
    is_nine = uniforms < torch.tensor(positions, dtype=torch.float64)[:, None]
    ids = torch.empty(is_nine.shape, dtype=torch.long)
    ids[is_nine] = _draw_pool(pool[targets[pool] == 9], int(is_nine.sum()), seeds["stream_nine"])
    ids[~is_nine] = _draw_pool(non_nine, int((~is_nine).sum()), seeds["stream_non_nine"])
    streams = {}
    for size in sizes:
        prefix = ids[:, :size].clone()
        labels = targets[prefix]
        streams[f"mixture_m{size}"] = {
            "observation_ids": prefix, "targets": labels, "positions": positions,
            "schedule_hash": canonical_hash(positions),
            "exposure": {**_exposure(prefix, labels), "expected_nines": size * sum(positions),
                         "cumulative_realized_nines": (labels == 9).sum(1).cumsum(0).tolist(),
                         "cumulative_expected_nines": (size * torch.tensor(positions, dtype=torch.float64)).cumsum(0).tolist()},
            "development_only": size == study.scarcity_samples,
            "sampling": "Bernoulli membership; empirical non-nine shuffled pool; per-arrival prefixes",
            "membership_uniforms_hash": tensor_content_hash(uniforms),
        }
    return {"streams": streams, "warmup_ids": warmup,
            "warmup_exposure": _exposure(warmup, targets[warmup])}


def _images(dataset, ids: Tensor) -> Tensor:
    # The existing loader applies ToTensor only; preserve its [0,1] convention.
    data = getattr(dataset, "data", None)
    if isinstance(data, Tensor) and data.dtype == torch.uint8 and data.ndim == 3:
        return data[ids].unsqueeze(1).float().div_(255)
    return torch.stack([dataset[int(index)][0] for index in ids]).float().cpu()


def asset_unit(store: UnitStore, phase: str, index: int, family: str, width: int, *,
               source_only: bool = False, extended_epochs: bool = False) -> dict:
    if family not in {"rotation", "mixture"} or isinstance(width, bool) or width not in {0, 5, 6}:
        raise ValueError("assets require rotation/mixture and width 0, 5, or 6")
    return store.unit(phase, "source" if source_only else "assets", index, environment=family,
                      detail={"width": width, "source_only": source_only,
                              "extended_epochs": extended_epochs, "asset_schema_version": 1,
                              "seeds": named_seeds(store.study, phase, index, family),
                              "resolved_config": store.study.mapping(),
                              "source_epoch_selection": "minimum source-validation NLL; earliest tie",
                              "rotation_sigmoid_sharpness": 8.0, "mixture_p_max": 0.1,
                              "warmup_ewc_strength": 19.0, "warmup_ridge": 0.0})


def _fit_source(store, session, model, data, seeds, *, epochs, parent_checkpoint=None):
    study = store.study
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    checkpoint_path = session.path / "source_checkpoint.pt"
    checkpoint = _load(checkpoint_path) if checkpoint_path.exists() else parent_checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=study.source_learning_rate)
    rng = _generator(seeds["source_order"])
    cuda_devices = [] if device.type == "cpu" else [device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seeds["source_rng"])
        if checkpoint is None:
            initial_quality = source_quality(model, data["validation_inputs"], data["validation_targets"])
            checkpoint = {"epoch": 0, "model": _cpu(model.state_dict()),
                          "initial_model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
                          "best_model": _cpu(model.state_dict()), "best_epoch": 0,
                          "best_quality": initial_quality, "initial_quality": initial_quality,
                          "metrics": [], "order_rng_state": rng.get_state(),
                          "torch_rng_state": torch.get_rng_state(),
                          "cuda_rng_states": [torch.cuda.get_rng_state(d).cpu() for d in cuda_devices]}
            session.write_torch("source_checkpoint.pt", checkpoint)
        else:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            rng.set_state(checkpoint["order_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            for dev, state in zip(cuda_devices, checkpoint["cuda_rng_states"], strict=True):
                torch.cuda.set_rng_state(state, device=dev)
            if not checkpoint_path.exists():
                session.write_torch("source_checkpoint.pt", checkpoint)
        for epoch in range(checkpoint["epoch"], epochs):
            store.limit.check()
            started = time.monotonic()
            model.train()
            order = torch.randperm(len(data["inputs"]), generator=rng)
            train_sum = 0.0
            for ids in order.split(study.source_batch_size):
                x = data["inputs"][ids].to(device=device, dtype=dtype)
                y = data["targets"][ids].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite source-training NLL")
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach()) * len(ids)
            quality = source_quality(model, data["validation_inputs"], data["validation_targets"])
            row = {"stage": "source_epoch", "epoch": epoch + 1,
                   "training_nll": train_sum / len(order),
                   "validation_nll": quality["nll"], "validation_accuracy": quality["accuracy"],
                   "wall_seconds": time.monotonic() - started}
            checkpoint["metrics"].append(row)
            if quality["nll"] < checkpoint["best_quality"]["nll"]:
                checkpoint.update(best_model=_cpu(model.state_dict()), best_epoch=epoch + 1,
                                  best_quality=quality)
            checkpoint.update(epoch=epoch + 1, model=_cpu(model.state_dict()),
                              optimizer=_cpu(optimizer.state_dict()), order_rng_state=rng.get_state(),
                              torch_rng_state=torch.get_rng_state(),
                              cuda_rng_states=[torch.cuda.get_rng_state(d).cpu() for d in cuda_devices])
            session.write_torch("source_checkpoint.pt", checkpoint)
            store.limit.check()
    model.load_state_dict(checkpoint["best_model"])
    model.eval()
    return checkpoint


def _initial_fisher(store, session, model, layout, inputs, targets, observation_ids,
                    seeds, *, family, name):
    """Atomic cumulative chunk checkpoints, with independently named model draws."""
    study = store.study
    path = session.path / f"{name}_checkpoint.pt"
    state = _load(path) if path.exists() else {
        "count": 0, "outer_sum": torch.zeros(layout.total_numel, layout.total_numel, dtype=torch.float64),
        "chunks": [], "observation_ids": torch.empty(0, dtype=torch.long),
    }
    parameter = next(model.parameters())
    pool = inputs.to(device=parameter.device, dtype=parameter.dtype)
    weights = (targets != 9).double() if family == "mixture" else None
    while state["count"] < study.initial_fisher_samples:
        store.limit.check()
        start = state["count"]
        count = min(study.fisher_chunk_size, study.initial_fisher_samples - start)
        seed = derive_component_seed(seeds["model_labels"], f"{name}:chunk:{start}")
        matrix, diagnostics = model_fisher(model, layout, pool, weights,
                                           sample_size=count, seed=seed, exact_labels=False,
                                           chunk_size=study.fisher_chunk_size)
        matrix = matrix.detach().double().cpu()
        if matrix.shape != state["outer_sum"].shape or not torch.isfinite(matrix).all():
            raise ValueError("invalid initial Fisher chunk")
        sampled_ids = observation_ids[diagnostics["input_indices"].cpu()]
        state["outer_sum"] += count * matrix
        state["observation_ids"] = torch.cat((state["observation_ids"], sampled_ids))
        state["chunks"].append({"start": start, "stop": start + count, "seed": seed,
                                 "seeds": diagnostics["seeds"],
                                 "observation_ids_hash": tensor_content_hash(sampled_ids),
                                 "min_eigenvalue_before_psd": diagnostics["min_eigenvalue_before_psd"],
                                 "negative_spectral_mass": diagnostics["negative_spectral_mass"],
                                 "psd_projection_frobenius": diagnostics["psd_projection_frobenius"]})
        state["count"] += count
        session.write_torch(path.name, state)
        store.limit.check()
    matrix = state["outer_sum"] / state["count"]
    matrix = (matrix + matrix.mT) / 2
    ids = state["observation_ids"]
    diagnostics = {"estimator": "sampled_model_labels", "input_draw_count": state["count"],
                   "unique_input_draws": ids.unique().numel(),
                   "duplicate_fraction": 1 - ids.unique().numel() / len(ids),
                   "sampling_with_replacement": True, "chunks": state["chunks"],
                   "observation_ids_hash": tensor_content_hash(ids),
                   "reference_pool_hash": tensor_content_hash(observation_ids),
                   "input_measure": "empirical_non_nine" if family == "mixture" else "empirical_all_digits"}
    return matrix, diagnostics, ids


def ensure_assets(store: UnitStore, phase: str, index: int, family: str, width: int, *,
                  resume: bool = True, source_only: bool = False,
                  extended_epochs: bool = False) -> Path:
    """Return a sealed directory containing assets.pt, summary.json and metrics.json.

    Width zero is the old 487-coordinate diagnostic reference. Source-only units
    omit streams, Fisher estimation and warm-up. Full units reuse the exact sealed
    source unit and record its hash; mixture branches share one full-space p=0
    update and one Fisher recomputed at that post-update state. All saved tensors
    are on CPU. The caller owns the run lock and final-cohort freeze checks.
    """
    study = store.study
    unit = asset_unit(store, phase, index, family, width, source_only=source_only,
                      extended_epochs=extended_epochs)
    required = SOURCE_REQUIRED if source_only else ASSET_REQUIRED
    session = store.begin(unit, required, resume=resume)
    if session is None:
        return store.paths(unit)[0]
    seeds = named_seeds(study, phase, index, family)
    parent = None
    if not source_only:
        parent = ensure_assets(store, phase, index, family, width, resume=True,
                               source_only=True, extended_epochs=extended_epochs)
        store.limit.check()
    data_root = Path(study.data_root)
    train, test = load_mnist_datasets(data_root if data_root.is_absolute() else REPO_ROOT / data_root,
                                     download=False)
    train_targets, test_targets = dataset_targets(train), dataset_targets(test)
    plan_path = session.path / "data_plan.pt"
    if plan_path.exists():
        plan = _load(plan_path)
    elif parent is not None:
        plan = _load(parent / "data_plan.pt")
        session.write_torch("data_plan.pt", plan)
    else:
        plan = partition_ids(study, train_targets, test_targets, family=family, seeds=seeds)
        # Source-only selection does not need to draw or build online streams.
        session.write_torch("data_plan.pt", plan)
    ids = plan["ids"]
    if width == 0:
        model, layout = build_gauge_fixed_model(seeds["initialization"], device=study.device,
                                                dtype=getattr(torch, study.dtype))
    else:
        model, layout = build_model(seeds["initialization"], width=width, device=study.device,
                                    dtype=getattr(torch, study.dtype))
    layout_mapping = layout.metadata()
    validation_inputs = _images(train, ids["source_validation"])
    validation_targets = train_targets[ids["source_validation"]]
    if source_only:
        parent_checkpoint = None
        parent_hashes = []
        if extended_epochs:
            base_unit = asset_unit(store, phase, index, family, width, source_only=True)
            base = store.completed(base_unit, SOURCE_REQUIRED)
            if base is not None:
                parent_checkpoint = _load(base / "source_checkpoint.pt")
                parent_hashes.append(canonical_hash(base_unit))
        epochs = study.source_epoch_ceiling if extended_epochs else study.source_epochs
        checkpoint = _fit_source(store, session, model, {
            "inputs": _images(train, ids["source"]), "targets": train_targets[ids["source"]],
            "validation_inputs": validation_inputs, "validation_targets": validation_targets,
        }, seeds, epochs=epochs, parent_checkpoint=parent_checkpoint)
        quality = checkpoint["best_quality"]
        history = checkpoint["metrics"]
        recent = [row["validation_nll"] for row in history[-3:]]
        summary = {"phase": phase, "replica_index": index, "family": family, "width": width,
                   "source_only": True, "seeds": seeds, "partition_hash": plan["hash"],
                   "parameter_count": layout.total_numel, "source_quality": quality,
                   "source_nll": quality["nll"], "source_accuracy": quality["accuracy"],
                   "selected_epoch": checkpoint["best_epoch"], "completed_epochs": checkpoint["epoch"],
                   "initial_source_quality": checkpoint["initial_quality"], "epoch_quality": history,
                   "convergence": {"criterion": "last three validation NLL range <= .001 (descriptive)",
                                   "settled": len(recent) == 3 and max(recent) - min(recent) <= 0.001},
                   "epoch_selection_panel": "source_validation", "parent_hashes": parent_hashes,
                   "partition_class_counts": plan["class_counts"], "parameter_layout": layout_mapping,
                   "source_model_hash": tensor_content_hash(layout.flatten_module(model, detach=True)),
                   "resolved_config": study.mapping()}
        assets = {"schema_version": 1, "initial_model": _cpu(model.state_dict()),
                  "pretraining_initial_model": checkpoint["initial_model"],
                  "source_model": _cpu(model.state_dict()), "source_quality": quality,
                  "parameter_layout": layout_mapping, "partition_ids": ids,
                  "source_validation_inputs": validation_inputs,
                  "source_validation_targets": validation_targets, "seeds": seeds}
        session.write_torch("assets.pt", assets)
        session.write_json("summary.json", summary)
        session.write_json("metrics.json", history)
        return store.finish(session, required)
    source_assets = _load(parent / "assets.pt")
    if source_assets["parameter_layout"] != layout_mapping:
        raise ValueError("source asset parameter layout mismatch")
    model.load_state_dict(source_assets["initial_model"])
    model.eval()
    is_final = phase == "phase5"
    evaluation_ids = ids["final_evaluation"] if is_final else ids["development_evaluation"]
    evaluation_inputs = _images(test if is_final else train,
                                evaluation_ids - len(train_targets) if is_final else evaluation_ids)
    evaluation_targets = (test_targets[evaluation_ids - len(train_targets)] if is_final
                          else train_targets[evaluation_ids])
    reference_inputs, reference_targets = _images(train, ids["reference"]), train_targets[ids["reference"]]
    stream_plan_path = session.path / "stream_plan.pt"
    stream_plan = _load(stream_plan_path) if stream_plan_path.exists() else stream_ids(
        study, train_targets, plan, family=family, seeds=seeds)
    if not stream_plan_path.exists():
        session.write_torch("stream_plan.pt", stream_plan)
    metrics = [{"stage": "source_initialization", "position": 0.0,
                "source_observations": len(ids["source"]), "warmup_observations": 0,
                "treatment_observations": 0,
                **evaluate(model, evaluation_inputs, evaluation_targets, family=family, position=0.0)}]
    fisher, fisher_diagnostics, fisher_ids = _initial_fisher(
        store, session, model, layout, reference_inputs, reference_targets, ids["reference"],
        seeds, family=family, name="source_fisher")
    source_fisher_diagnostics = fisher_diagnostics
    warmup_diagnostics = None
    source_model = _cpu(model.state_dict())
    warmup_ids = stream_plan["warmup_ids"]
    warmup_inputs = _images(train, warmup_ids) if len(warmup_ids) else torch.empty(0, 1, 28, 28)
    warmup_targets = train_targets[warmup_ids]
    if family == "mixture":
        warmup_path = session.path / "warmup_checkpoint.pt"
        if warmup_path.exists():
            warmup = _load(warmup_path)
            model.load_state_dict(warmup["model"])
            warmup_diagnostics = warmup["diagnostics"]
        else:
            store.limit.check()
            parameter = next(model.parameters())
            anchor = layout.flatten_module(model, detach=True).clone()
            fit = fit_restricted(model, layout, warmup_inputs.to(parameter), warmup_targets.to(parameter.device),
                                 torch.ones(len(warmup_ids), device=parameter.device), anchor,
                                 19.0 * fisher.to(parameter.device),
                                 torch.arange(layout.total_numel, device=parameter.device),
                                 max_iter=study.max_iter, max_eval=study.max_eval,
                                 tolerance_grad=study.tolerance_grad, tolerance_change=study.tolerance_change)
            warmup_diagnostics = {key: value for key, value in fit.items() if not isinstance(value, Tensor)}
            session.write_torch("warmup_checkpoint.pt", {"model": _cpu(model.state_dict()),
                                                         "diagnostics": _cpu(warmup_diagnostics)})
            store.limit.check()
        metrics.append({"stage": "post_warmup", "position": 0.0,
                        "source_observations": len(ids["source"]), "warmup_observations": len(warmup_ids),
                        "treatment_observations": 0,
                        **evaluate(model, evaluation_inputs, evaluation_targets, family=family, position=0.0)})
        fisher, fisher_diagnostics, fisher_ids = _initial_fisher(
            store, session, model, layout, reference_inputs, reference_targets, ids["reference"],
            seeds, family=family, name="post_warmup_fisher")
    streams = {}
    for name, stream in stream_plan["streams"].items():
        store.limit.check()
        shape = stream["observation_ids"].shape
        images = _images(train, stream["observation_ids"].flatten()).reshape(*shape, 1, 28, 28)
        if family == "rotation":
            rotation = RotationConfig.from_mapping(stream["rotation"])
            images = torch.stack([rotate_mnist_batch(batch, angle, rotation)
                                  for batch, angle in zip(images, stream["positions"], strict=True)])
        streams[name] = {**stream, "inputs": images}
    initial_source_quality = source_quality(model, validation_inputs, validation_targets)
    parent_unit = asset_unit(store, phase, index, family, width, source_only=True,
                             extended_epochs=extended_epochs)
    summary = {"phase": phase, "replica_index": index, "family": family, "width": width,
               "source_only": False, "seeds": seeds, "parameter_count": layout.total_numel,
               "parameter_layout": layout_mapping, "partition_hash": plan["hash"],
               "partition_class_counts": plan["class_counts"], "parent_hashes": [canonical_hash(parent_unit)],
               "source_parent": str(parent.relative_to(store.root)),
               "source_parent_assets_hash": file_hash(parent / "assets.pt"),
               "source_quality": source_assets["source_quality"],
               "source_nll": source_assets["source_quality"]["nll"],
               "source_accuracy": source_assets["source_quality"]["accuracy"],
               "post_warmup_source_quality": initial_source_quality,
               "evaluation_split": "test" if is_final else "development_train",
               "evaluation_ids_hash": tensor_content_hash(evaluation_ids),
               "source_fisher": source_fisher_diagnostics, "initial_fisher": fisher_diagnostics,
               "warmup": warmup_diagnostics, "warmup_ewc_strength": 19.0, "warmup_ridge": 0.0,
               "warmup_exposure": stream_plan.get("warmup_exposure"),
               "streams": {name: {key: value for key, value in stream.items()
                                   if key not in {"inputs", "targets", "observation_ids"}}
                           for name, stream in streams.items()}, "resolved_config": study.mapping()}
    assets = {"schema_version": 1, "initial_model": _cpu(model.state_dict()), "source_model": source_model,
              "initial_fisher": fisher, "initial_fisher_observation_ids": fisher_ids,
              "parameter_layout": layout_mapping, "partition_ids": ids, "seeds": seeds,
              "evaluation_inputs": evaluation_inputs, "evaluation_targets": evaluation_targets,
              "evaluation_ids": evaluation_ids, "reference_inputs": reference_inputs,
              "reference_targets": reference_targets, "reference_ids": ids["reference"],
              "source_validation_inputs": validation_inputs, "source_validation_targets": validation_targets,
              "warmup_inputs": warmup_inputs, "warmup_targets": warmup_targets,
              "warmup_observation_ids": warmup_ids, "streams": streams,
              "source_quality": source_assets["source_quality"], "parent_hashes": summary["parent_hashes"]}
    session.write_torch("assets.pt", _cpu(assets))
    session.write_json("summary.json", _json(summary))
    session.write_json("metrics.json", metrics)
    checkpoints = ["source_fisher_checkpoint.pt"]
    if family == "mixture":
        checkpoints.extend(("warmup_checkpoint.pt", "post_warmup_fisher_checkpoint.pt"))
    return store.finish(session, (*required, "data_plan.pt", "stream_plan.pt", *checkpoints))
