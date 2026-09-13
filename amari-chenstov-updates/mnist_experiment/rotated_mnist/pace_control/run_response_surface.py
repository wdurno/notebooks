"""Execute the immutable Plan 10 Phase 1d finite-EWC response surface."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.ewc import build_optimizer, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.initialization import state_dict_hash
from src.mnist_data import load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.seeding import derive_component_seed

from ..phase4_metrics import materialize_base_panel, materialize_rotated_panel
from ..phase5_double_lap_artifacts import load_completed_double_lap_run
from ..run import _learner_optimizer_config
from ..run_phase3 import _fresh_fisher
from ..transform import rotate_mnist_batch, tensor_content_hash
from .artifacts import Plan10RunStore, file_sha256, read_json, validate_completed
from .response_surface import analyze_response_surface
from .response_surface_config import (
    ResponseSurfaceConfig,
    load_response_surface_config,
)


REQUIRED = (
    "config.json",
    "source_contract.json",
    "prerequisite_contract.json",
    "evaluation_panel.json",
    "anchor_contracts.json",
    "anchor_states.pt",
    "fit_records.json",
    "response_surface.json",
    "gate.json",
    "summary.json",
)


@dataclasses.dataclass(frozen=True)
class _Source:
    path: Path
    loaded: Any
    initial_state: dict[str, Tensor]
    initial_fisher: LowRankDiagonalFisher
    stream_inputs: Tensor
    stream_targets: Tensor
    parameters: Tensor
    rows: tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class _Anchor:
    step: int
    angle_degrees: float
    direction_to_next: int
    leg_id: int
    q: float
    parameter: Tensor
    fisher: LowRankDiagonalFisher
    parameter_hash: str
    fisher_hash: str
    fisher_trace: float

    def contract(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "angle_degrees": self.angle_degrees,
            "direction_to_next": self.direction_to_next,
            "leg_id": self.leg_id,
            "q": self.q,
            "parameter_hash": self.parameter_hash,
            "fisher_hash": self.fisher_hash,
            "fisher_trace": self.fisher_trace,
        }

    def artifact(self) -> dict[str, Any]:
        return {
            **self.contract(),
            "parameter": self.parameter,
            "fisher": self.fisher.artifact_mapping(),
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("cache/mnist_experiment/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/plan10/phase1d"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _hash_parts(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _fisher_hash(fisher: LowRankDiagonalFisher) -> str:
    return _hash_parts(
        tensor_content_hash(fisher.factor),
        tensor_content_hash(fisher.residual_diagonal),
    )


def _load_source(config: ResponseSurfaceConfig, repo_root: Path) -> _Source:
    source_path = repo_root / config.source_run_path
    loaded = load_completed_double_lap_run(source_path)
    if loaded.manifest["run_id"] != config.source_run_id:
        raise ValueError("Phase 1d source run ID differs from its manifest")
    if config.source_schedule not in loaded.stream_plans:
        raise ValueError("Phase 1d source schedule is unavailable")
    if config.source_condition not in loaded.trajectory_metrics[config.source_schedule]:
        raise ValueError("Phase 1d source condition is unavailable")
    model_states = torch.load(
        source_path / "model_states.pt", map_location="cpu", weights_only=True
    )
    initial_state = model_states[config.source_schedule][config.source_condition][
        "initial"
    ]
    fisher_value = torch.load(
        source_path / "initial_fisher.pt", map_location="cpu", weights_only=True
    )
    initial_fisher = representation_from_artifact(fisher_value["representation"])
    if not isinstance(initial_fisher, LowRankDiagonalFisher) or initial_fisher.rank != 8:
        raise ValueError("Phase 1d requires the source rank-8-plus-diagonal Fisher")
    stream = torch.load(
        source_path / "stream_tensors.pt", map_location="cpu", weights_only=True
    )[config.source_schedule]
    trajectories = torch.load(
        source_path / "trajectories.pt", map_location="cpu", weights_only=True
    )[config.source_schedule][config.source_condition]
    parameters = trajectories["parameters"]
    rows = loaded.trajectory_metrics[config.source_schedule][config.source_condition]
    if parameters.shape != (len(rows), 512):
        raise ValueError("Phase 1d source parameter trajectory is incompatible")
    return _Source(
        path=source_path,
        loaded=loaded,
        initial_state=initial_state,
        initial_fisher=initial_fisher,
        stream_inputs=stream["inputs"],
        stream_targets=stream["targets"],
        parameters=parameters,
        rows=rows,
    )


def _source_contract(source: _Source, config: ResponseSurfaceConfig) -> dict[str, Any]:
    names = (
        "config.json",
        "manifest.json",
        "partitions.json",
        "stream_plans.json",
        "stream_tensors.pt",
        "initial_fisher.pt",
        "model_states.pt",
        "trajectories.pt",
        "trajectory_metrics.json",
    )
    return {
        "source_run_id": source.loaded.manifest["run_id"],
        "source_config_hash": source.loaded.manifest["config_hash"],
        "source_schedule": config.source_schedule,
        "source_condition": config.source_condition,
        "source_manifest_status": source.loaded.manifest["status"],
        "source_file_sha256": {
            name: file_sha256(source.path / name) for name in names
        },
        "source_partition_hash": source.loaded.partitions.content_hash,
        "source_schedule_hash": source.loaded.stream_plans[
            config.source_schedule
        ].schedule.content_hash,
        "initial_state_hash": state_dict_hash(source.initial_state),
        "initial_fisher_hash": _fisher_hash(source.initial_fisher),
        "endogenous_anchor_state_reconstructed": True,
    }


def _validate_prerequisite(
    config: ResponseSurfaceConfig, repo_root: Path
) -> dict[str, Any]:
    if config.prerequisite_run_path is None:
        return {"required": False, "passed": True}
    path = repo_root / config.prerequisite_run_path
    manifest = validate_completed(path, required=REQUIRED)
    if manifest["run_id"] != config.prerequisite_run_id:
        raise ValueError("Phase 1d prerequisite run ID differs from its manifest")
    summary = read_json(path / "summary.json")
    if config.stage == "expansion":
        passed = bool(summary.get("expansion_recommended"))
        decision_field = "expansion_recommended"
    else:
        passed = bool(summary.get("gate_pass"))
        decision_field = "gate_pass"
    if not passed:
        raise ValueError(
            f"Phase 1d prerequisite did not pass {decision_field}: {path.name}"
        )
    return {
        "required": True,
        "passed": True,
        "path": config.prerequisite_run_path,
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "decision_field": decision_field,
        "decision_value": passed,
        "summary_sha256": file_sha256(path / "summary.json"),
    }


def _reconstruct_anchors(
    source: _Source,
    config: ResponseSurfaceConfig,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> list[_Anchor]:
    model, layout = build_canonical_model(
        config.replica_seed, device=device, dtype=training_dtype
    )
    model.load_state_dict(source.initial_state)
    fisher = source.initial_fisher.to(device=device, dtype=matrix_dtype)
    q = 1.0 / source.loaded.config.data.initialization_size
    anchors = []
    anchor_steps = set(config.anchor_steps)
    for step in range(max(config.anchor_steps) + 1):
        parameter = source.parameters[step].to(device=device, dtype=training_dtype)
        layout.copy_vector_to_module(model, parameter)
        row = source.rows[step]
        observed_parameter_hash = tensor_content_hash(parameter.cpu())
        if observed_parameter_hash != row["parameter_hash"]:
            raise RuntimeError(f"source parameter hash mismatch at step {step}")
        reconstructed_trace = float(fisher.diagonal_vector().sum())
        expected_trace = float(row["fisher_update"]["previous_trace"])
        if not math.isclose(
            reconstructed_trace, expected_trace, rel_tol=2e-6, abs_tol=2e-6
        ):
            raise RuntimeError(
                f"source Fisher trace mismatch at step {step}: "
                f"{reconstructed_trace} != {expected_trace}"
            )
        expected_q_before = 1.0 / float(row["controller"]["effective_size"])
        if not math.isclose(q, expected_q_before, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"source pre-update q mismatch at step {step}")
        if step in anchor_steps:
            anchors.append(
                _Anchor(
                    step=step,
                    angle_degrees=float(row["angle_degrees"]),
                    direction_to_next=int(row["direction_to_next"]),
                    leg_id=int(row["leg_id"]),
                    q=q,
                    parameter=parameter.detach().cpu(),
                    fisher=fisher.to(device="cpu"),
                    parameter_hash=observed_parameter_hash,
                    fisher_hash=_fisher_hash(fisher),
                    fisher_trace=reconstructed_trace,
                )
            )
        if step == max(config.anchor_steps):
            break
        inputs = source.stream_inputs[step].to(device=device, dtype=training_dtype)
        targets = source.stream_targets[step].to(device=device)
        fresh = _fresh_fisher(
            model, layout, inputs, targets, matrix_dtype=matrix_dtype
        )
        if step > 0:
            update = blend_archive_fisher(
                fisher,
                fresh,
                blend_gain=config.fixed_pi,
                rank=source.loaded.config.fisher.rank,
                lanczos_seed=derive_component_seed(
                    source.loaded.config.replica_seed,
                    "plan5_double_lap_update_lanczos:"
                    f"schedule={config.source_schedule}:"
                    f"condition={config.source_condition}:step={step}",
                ),
            )
            fisher = update.representation
        q = (1.0 - config.fixed_pi) ** 2 * q + config.fixed_pi**2 / config.batch_size
        expected_q = float(row["controller_acceptance"]["state_after"]["q"])
        if not math.isclose(q, expected_q, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"source q mismatch at step {step}")
    if tuple(anchor.step for anchor in anchors) != config.anchor_steps:
        raise RuntimeError("Phase 1d failed to reconstruct every frozen anchor")
    return anchors


def _materialize_index_batch(dataset: Any, indices: Tensor) -> tuple[Tensor, Tensor]:
    inputs = []
    targets = []
    for index in indices.tolist():
        image, target = dataset[index]
        inputs.append(image.to(dtype=torch.float32, device="cpu"))
        targets.append(int(target))
    return torch.stack(inputs).contiguous(), torch.tensor(targets, dtype=torch.long)


def _evaluation_panels(
    source: _Source,
    anchors: list[_Anchor],
    config: ResponseSurfaceConfig,
    test_dataset: Any,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
) -> tuple[dict[float, tuple[Tensor, Tensor]], dict[str, Any]]:
    base_inputs, base_targets = materialize_base_panel(
        test_dataset,
        source.loaded.partitions.evaluation,
        num_workers=config.runtime.num_workers,
    )
    angles = sorted(
        {
            round(anchor.angle_degrees, 9)
            for anchor in anchors
        }
        | {
            round(
                anchor.angle_degrees + anchor.direction_to_next * pace, 9
            )
            for anchor in anchors
            for pace in config.pace_degrees
        }
    )
    panels = {}
    panel_hashes = {}
    for angle in angles:
        inputs, targets, content_hash = materialize_rotated_panel(
            base_inputs, base_targets, angle, source.loaded.config.rotation
        )
        panels[angle] = (
            inputs.to(device=device, dtype=training_dtype),
            targets.to(device=device),
        )
        panel_hashes[str(angle)] = content_hash
    contract = {
        "sample_count": int(base_targets.numel()),
        "source": "phase5_test_evaluation_partition",
        "source_partition_hash": source.loaded.partitions.content_hash,
        "base_inputs_hash": tensor_content_hash(base_inputs),
        "base_targets_hash": tensor_content_hash(base_targets),
        "angles_degrees": angles,
        "rotated_panel_hashes": panel_hashes,
        "used_for_anchor_selection": False,
        "used_for_fitting": False,
    }
    return panels, contract


def _evaluate_nll_accuracy(
    model: nn.Module,
    panel: tuple[Tensor, Tensor],
    *,
    batch_size: int,
) -> tuple[float, float]:
    inputs, targets = panel
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, targets.numel(), batch_size):
            batch_inputs = inputs[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            logits = model(batch_inputs)
            total_loss += float(
                nn.functional.cross_entropy(
                    logits, batch_targets, reduction="sum"
                )
            )
            total_correct += int((logits.argmax(dim=1) == batch_targets).sum())
    model.train(was_training)
    return total_loss / targets.numel(), total_correct / targets.numel()


def _unit_id(anchor_step: int, pace: float, replicate: int) -> str:
    return f"anchor={anchor_step}:pace={pace:.9g}:replicate={replicate}"


def _finite_mapping(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_mapping(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_mapping(item) for item in value)
    return True


def _run_units(
    source: _Source,
    anchors: list[_Anchor],
    panels: dict[float, tuple[Tensor, Tensor]],
    train_dataset: Any,
    config: ResponseSurfaceConfig,
    session: Any,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
    resume: bool,
) -> tuple[list[dict[str, Any]], float]:
    progress_path = session.working_path / "progress.pt"
    if resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        records = list(progress["records"])
        completed = set(progress["completed_units"])
        prior_seconds = float(progress.get("wall_time_seconds", 0.0))
    else:
        records = []
        completed = set()
        prior_seconds = 0.0
    model, layout = build_canonical_model(
        config.replica_seed, device=device, dtype=training_dtype
    )
    model.load_state_dict(source.initial_state)
    optimizer_config = _learner_optimizer_config(source.loaded.config)
    pool = torch.tensor(source.loaded.partitions.online, dtype=torch.long)
    base_batches = {}
    for anchor in anchors:
        for replicate in range(config.replicate_count):
            seed = derive_component_seed(
                config.replica_seed,
                f"phase1d_batch:anchor={anchor.step}:replicate={replicate}",
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            choices = torch.randint(
                pool.numel(), (config.batch_size,), generator=generator
            )
            indices = pool[choices]
            inputs, targets = _materialize_index_batch(train_dataset, indices)
            base_batches[(anchor.step, replicate)] = (
                inputs,
                targets,
                indices.tolist(),
                seed,
            )

    units = [
        (anchor, pace, replicate)
        for anchor in anchors
        for pace in config.pace_degrees
        for replicate in range(config.replicate_count)
    ]
    started = time.perf_counter()
    newly_completed = 0
    for unit_index, (anchor, pace, replicate) in enumerate(units, start=1):
        identifier = _unit_id(anchor.step, pace, replicate)
        if identifier in completed:
            continue
        endpoint = round(
            anchor.angle_degrees + anchor.direction_to_next * pace, 9
        )
        old_angle = round(anchor.angle_degrees, 9)
        base_inputs, targets, indices, batch_seed = base_batches[
            (anchor.step, replicate)
        ]
        inputs = rotate_mnist_batch(
            base_inputs, endpoint, source.loaded.config.rotation
        ).to(device=device, dtype=training_dtype)
        targets_device = targets.to(device=device)
        anchor_parameter = anchor.parameter.to(device=device, dtype=training_dtype)
        anchor_fisher = anchor.fisher.to(device=device, dtype=matrix_dtype)
        layout.copy_vector_to_module(model, anchor_parameter)
        fresh = _fresh_fisher(
            model, layout, inputs, targets_device, matrix_dtype=matrix_dtype
        )
        lanczos_seed = derive_component_seed(
            config.replica_seed,
            f"phase1d_lanczos:anchor={anchor.step}:pace={pace}:replicate={replicate}",
        )
        for pi in config.pi_values:
            row = {
                "unit_id": identifier,
                "anchor_step": anchor.step,
                "anchor_angle_degrees": anchor.angle_degrees,
                "direction_to_next": anchor.direction_to_next,
                "endpoint_angle_degrees": endpoint,
                "pace_degrees": pace,
                "within_physical_bounds": (
                    config.physical_pace_minimum
                    <= pace
                    <= config.physical_pace_maximum
                ),
                "replicate_index": replicate,
                "pi": pi,
                "batch_seed": batch_seed,
                "batch_indices": indices,
                "batch_targets": targets.tolist(),
                "lanczos_seed": lanczos_seed,
                "next_nll": None,
                "next_accuracy": None,
                "old_angle_nll": None,
                "old_angle_accuracy": None,
                "failure": None,
            }
            try:
                layout.copy_vector_to_module(model, anchor_parameter)
                update = blend_archive_fisher(
                    anchor_fisher,
                    fresh,
                    blend_gain=pi,
                    rank=source.loaded.config.fisher.rank,
                    lanczos_seed=lanczos_seed,
                )
                optimizer = build_optimizer(model, optimizer_config)
                proposal = take_ewc_proposal(
                    model,
                    layout,
                    inputs,
                    targets_device,
                    update.representation.to(dtype=training_dtype),
                    optimizer_config,
                    optimizer,
                    adaptation_weight=pi,
                    penalty_anchor=anchor_parameter,
                )
                next_nll, next_accuracy = _evaluate_nll_accuracy(
                    model,
                    panels[endpoint],
                    batch_size=config.evaluation_batch_size,
                )
                if endpoint == old_angle:
                    old_nll, old_accuracy = next_nll, next_accuracy
                else:
                    old_nll, old_accuracy = _evaluate_nll_accuracy(
                        model,
                        panels[old_angle],
                        batch_size=config.evaluation_batch_size,
                    )
                row.update(
                    {
                        "next_nll": next_nll,
                        "next_accuracy": next_accuracy,
                        "old_angle_nll": old_nll,
                        "old_angle_accuracy": old_accuracy,
                        "fresh_fisher_trace": float(torch.trace(fresh)),
                        "proposal_fisher_trace": float(
                            update.representation.diagonal_vector().sum()
                        ),
                        "fisher_update": update.lanczos.mapping(),
                        "proposal": proposal.metrics_mapping(),
                    }
                )
                if not _finite_mapping(row):
                    raise RuntimeError("fit record contains nonfinite diagnostics")
            except Exception as exc:  # Preserve a failed cell for the frozen gate.
                row["failure"] = f"{type(exc).__name__}: {exc}"
            records.append(row)
        completed.add(identifier)
        newly_completed += 1
        elapsed = prior_seconds + time.perf_counter() - started
        if (
            newly_completed % config.checkpoint_interval_units == 0
            or unit_index == len(units)
        ):
            session.write_torch(
                "progress.pt",
                {
                    "records": records,
                    "completed_units": sorted(completed),
                    "wall_time_seconds": elapsed,
                },
            )
        if newly_completed % max(1, config.checkpoint_interval_units * 2) == 0:
            rate = newly_completed / max(time.perf_counter() - started, 1e-9)
            remaining = len(units) - len(completed)
            print(
                json.dumps(
                    {
                        "stage": config.stage,
                        "completed_units": len(completed),
                        "total_units": len(units),
                        "eta_seconds": remaining / max(rate, 1e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records, prior_seconds + time.perf_counter() - started


def run_response_surface(
    config: ResponseSurfaceConfig,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    repo_path = Path(repo_root)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    source = _load_source(config, repo_path)
    if source.loaded.config.runtime.dtype != config.runtime.dtype:
        raise ValueError("Phase 1d dtype must match the historical source")
    matrix_dtype = resolve_dtype(source.loaded.config.fisher.matrix_dtype)
    prerequisite = _validate_prerequisite(config, repo_path)
    session = Plan10RunStore(output_root).begin(
        run_id=config.run_id,
        run_kind=f"phase1d_{config.stage}_finite_ewc_response_surface",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_path,
        experiment_config=config,
        resume=resume,
    )
    total_started = time.perf_counter()
    session.write_json("source_contract.json", _source_contract(source, config))
    session.write_json("prerequisite_contract.json", prerequisite)
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    anchors = _reconstruct_anchors(
        source,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    anchor_contracts = [anchor.contract() for anchor in anchors]
    session.write_json("anchor_contracts.json", anchor_contracts)
    session.write_torch(
        "anchor_states.pt", {str(anchor.step): anchor.artifact() for anchor in anchors}
    )
    panels, panel_contract = _evaluation_panels(
        source,
        anchors,
        config,
        test_dataset,
        device=device,
        training_dtype=training_dtype,
    )
    session.write_json("evaluation_panel.json", panel_contract)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records, fit_wall_seconds = _run_units(
        source,
        anchors,
        panels,
        train_dataset,
        config,
        session,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
        resume=resume,
    )
    surface, gate, summary = analyze_response_surface(
        records, anchor_contracts, config
    )
    summary.update(
        {
            "fit_wall_time_seconds": fit_wall_seconds,
            "total_wall_time_seconds": time.perf_counter() - total_started,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
    )
    gate["summary"] = summary
    session.write_json("fit_records.json", records)
    session.write_json("response_surface.json", surface)
    session.write_json("gate.json", gate)
    session.write_json("summary.json", summary)
    path = session.complete(REQUIRED)
    return path


def main() -> None:
    arguments = parse_arguments()
    config = load_response_surface_config(arguments.config)
    path = run_response_surface(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=Path.cwd(),
        download=arguments.download,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {"path": str(path), **read_json(path / "summary.json")},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
