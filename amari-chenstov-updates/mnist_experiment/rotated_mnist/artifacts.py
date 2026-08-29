"""Immutable artifact lifecycle for the detachable Plan 5 namespace."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from src.seeding import SEED_SCHEMA_VERSION, derive_seed_map

from .config import ARTIFACT_SCHEMA_VERSION, RotatedExperimentConfig
from .data import (
    RotatedPartitions,
    RotatedStreamPlan,
    validate_stream_tensors,
)
from .transform import tensor_content_hash
from src.initialization import state_dict_hash


MANIFEST_SCHEMA_VERSION = 1
PLAN5_SEED_COMPONENTS = (
    "plan5_model_initialization",
    "plan5_data_partition",
    "plan5_initialization_loader",
    "plan5_online_stream",
    "plan5_reference_stream",
    "plan5_evaluation_panel",
    "plan5_transform_cache",
    "plan5_optimizer",
    "plan5_numerical_randomization",
)
REQUIRED_ARTIFACTS = (
    "partitions.json",
    "stream_plan.json",
    "stream_tensors.pt",
    "initialization_metrics.json",
    "trajectory_metrics.json",
    "trajectory.pt",
    "model_states.pt",
    "run_summary.json",
)


class RotatedArtifactError(RuntimeError):
    """Base error for Plan 5 run lifecycle and validation."""


class RotatedCompletedRunError(RotatedArtifactError):
    """Raised when a completed Plan 5 run would be mutated."""


class RotatedIncompleteRunError(RotatedArtifactError):
    """Raised when an incomplete Plan 5 run requires explicit resumption."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RotatedArtifactError(f"could not read {path}: {exc}") from exc


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def runtime_metadata(
    repo_root: str | Path,
    config: RotatedExperimentConfig,
) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    try:
        import torchvision
    except ImportError:
        torchvision_version = None
    else:
        torchvision_version = torchvision.__version__
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "torchvision_version": torchvision_version,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
        },
        "requested_device": config.runtime.device,
        "dtype": config.runtime.dtype,
        "git": _git_metadata(Path(repo_root)),
    }


def _safe_path(root: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError("artifact path must be a nonempty relative path")
    return root / relative


@dataclasses.dataclass
class RotatedRunSession:
    run_id: str
    working_path: Path
    final_path: Path
    _completed: bool = False

    @property
    def path(self) -> Path:
        return self.final_path if self._completed else self.working_path

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        if self._completed:
            raise RotatedCompletedRunError(f"run {self.run_id} is complete")
        destination = _safe_path(self.working_path, relative_path)
        if destination.name in {"config.json", "manifest.json", "COMPLETED"}:
            raise RotatedArtifactError(f"{destination.name} is lifecycle-managed")
        _write_json(destination, value)
        return destination

    def write_torch(self, relative_path: str | Path, value: Any) -> Path:
        if self._completed:
            raise RotatedCompletedRunError(f"run {self.run_id} is complete")
        destination = _safe_path(self.working_path, relative_path)
        if destination.name in {"config.json", "manifest.json", "COMPLETED"}:
            raise RotatedArtifactError(f"{destination.name} is lifecycle-managed")
        buffer = io.BytesIO()
        torch.save(value, buffer)
        _atomic_write(destination, buffer.getvalue())
        return destination

    def complete(self, required: Iterable[str | Path] = REQUIRED_ARTIFACTS) -> Path:
        if self._completed:
            raise RotatedCompletedRunError(f"run {self.run_id} is complete")
        if self.final_path.exists():
            raise RotatedCompletedRunError(
                f"completed destination already exists: {self.final_path}"
            )
        missing = [
            str(name)
            for name in required
            if not _safe_path(self.working_path, name).is_file()
        ]
        if missing:
            raise RotatedArtifactError(
                f"cannot complete rotated run; missing artifacts: {missing}"
            )
        manifest = _read_json(self.working_path / "manifest.json")
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        _write_json(self.working_path / "manifest.json", manifest)
        _atomic_write(self.working_path / "COMPLETED", b"")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.working_path, self.final_path)
        self._completed = True
        return self.final_path


class RotatedRunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.incomplete_root = self.root / ".incomplete"

    def begin(
        self,
        config: RotatedExperimentConfig,
        repo_root: str | Path,
        *,
        resume: bool = False,
    ) -> RotatedRunSession:
        config.validate()
        final_path = self.root / config.run_id
        working_path = self.incomplete_root / config.run_id
        if final_path.exists():
            if (final_path / "COMPLETED").is_file():
                raise RotatedCompletedRunError(
                    f"rotated run already completed: {config.run_id}"
                )
            raise RotatedArtifactError(
                f"final rotated run exists without COMPLETED: {final_path}"
            )
        if working_path.exists():
            if not resume:
                raise RotatedIncompleteRunError(
                    f"incomplete rotated run exists; pass --resume: {config.run_id}"
                )
            if _read_json(working_path / "config.json") != config.to_mapping():
                raise RotatedArtifactError("incomplete rotated run config differs")
            manifest = _read_json(working_path / "manifest.json")
            if manifest.get("status") != "incomplete":
                raise RotatedArtifactError("incomplete rotated manifest is invalid")
            return RotatedRunSession(config.run_id, working_path, final_path)

        working_path.mkdir(parents=True, exist_ok=False)
        _write_json(working_path / "config.json", config.to_mapping())
        _write_json(
            working_path / "manifest.json",
            {
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "artifact_schema_version": config.artifact_schema_version,
                "metric_schema_version": config.metric_schema_version,
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "run_id": config.run_id,
                "experiment": config.experiment,
                "replica_id": config.replica_id,
                "config_hash": config.config_hash,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "seeds": derive_seed_map(
                    config.replica_seed, PLAN5_SEED_COMPONENTS
                ),
                "runtime": runtime_metadata(repo_root, config),
            },
        )
        return RotatedRunSession(config.run_id, working_path, final_path)


@dataclasses.dataclass(frozen=True)
class LoadedRotatedRun:
    path: Path
    config: RotatedExperimentConfig
    manifest: dict[str, Any]
    partitions: RotatedPartitions
    stream_plan: RotatedStreamPlan
    initialization_metrics: dict[str, Any]
    trajectory_metrics: tuple[dict[str, Any], ...]
    run_summary: dict[str, Any]
    stream_inputs: torch.Tensor | None = None
    stream_targets: torch.Tensor | None = None


def load_completed_run(
    path: str | Path,
    *,
    load_stream_tensors: bool = False,
    validate_tensor_artifacts: bool = False,
) -> LoadedRotatedRun:
    run_path = Path(path)
    if not (run_path / "COMPLETED").is_file():
        raise RotatedIncompleteRunError(f"rotated run is incomplete: {run_path}")
    for name in REQUIRED_ARTIFACTS:
        if not (run_path / name).is_file():
            raise RotatedArtifactError(f"completed rotated run is missing {name}")
    config_value = _read_json(run_path / "config.json")
    if not isinstance(config_value, Mapping):
        raise RotatedArtifactError("rotated config artifact must be an object")
    config = RotatedExperimentConfig.from_mapping(config_value)
    manifest = _read_json(run_path / "manifest.json")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("metric_schema_version") != config.metric_schema_version
        or manifest.get("status") != "completed"
        or manifest.get("config_hash") != config.config_hash
        or manifest.get("run_id") != config.run_id
        or run_path.name != config.run_id
    ):
        raise RotatedArtifactError("rotated run manifest is incompatible")
    partitions = RotatedPartitions.from_mapping(
        _read_json(run_path / "partitions.json")
    )
    stream_plan = RotatedStreamPlan.from_mapping(
        _read_json(run_path / "stream_plan.json")
    )
    if stream_plan.partition_hash != partitions.content_hash:
        raise RotatedArtifactError("rotated stream and partition hashes differ")
    metrics_value = _read_json(run_path / "trajectory_metrics.json")
    if not isinstance(metrics_value, list) or len(metrics_value) != stream_plan.schedule.num_points:
        raise RotatedArtifactError("rotated trajectory metrics are incompatible")
    for step, row in enumerate(metrics_value):
        if not isinstance(row, dict) or row.get("step") != step:
            raise RotatedArtifactError("rotated trajectory step ordering is invalid")
    stream_inputs = None
    stream_targets = None
    if load_stream_tensors:
        try:
            tensors = torch.load(
                run_path / "stream_tensors.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError) as exc:
            raise RotatedArtifactError(f"could not load stream tensors: {exc}") from exc
        if not isinstance(tensors, dict) or set(tensors) != {"inputs", "targets"}:
            raise RotatedArtifactError("rotated stream tensor artifact is invalid")
        stream_inputs = tensors["inputs"]
        stream_targets = tensors["targets"]
        validate_stream_tensors(stream_plan, stream_inputs, stream_targets)
    initialization_metrics = _read_json(run_path / "initialization_metrics.json")
    run_summary = _read_json(run_path / "run_summary.json")
    if not isinstance(initialization_metrics, dict) or not isinstance(run_summary, dict):
        raise RotatedArtifactError("rotated scalar artifacts are incompatible")
    if (
        run_summary.get("schedule_hash") != stream_plan.schedule.content_hash
        or run_summary.get("stream_plan_hash") != stream_plan.content_hash
        or run_summary.get("partition_hash") != partitions.content_hash
        or run_summary.get("num_points") != stream_plan.schedule.num_points
        or run_summary.get("num_transitions")
        != stream_plan.schedule.num_transitions
    ):
        raise RotatedArtifactError("rotated run summary provenance is incompatible")
    for step, row in enumerate(metrics_value):
        if (
            row.get("angle_degrees") != stream_plan.schedule.angles_degrees[step]
            or row.get("condition") != config.learner.condition
            or "current_nine_ovr_accuracy" not in row
            or "current_environment_accuracy" not in row
        ):
            raise RotatedArtifactError("rotated trajectory metric contract failed")
    if validate_tensor_artifacts:
        try:
            trajectory = torch.load(
                run_path / "trajectory.pt", map_location="cpu", weights_only=True
            )
            states = torch.load(
                run_path / "model_states.pt", map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError) as exc:
            raise RotatedArtifactError(
                f"could not load rotated tensor artifacts: {exc}"
            ) from exc
        if not isinstance(trajectory, dict) or set(trajectory) != {
            "parameters",
            "displacements",
        }:
            raise RotatedArtifactError("rotated trajectory tensor artifact is invalid")
        parameters = trajectory["parameters"]
        displacements = trajectory["displacements"]
        if (
            parameters.ndim != 2
            or parameters.shape[0] != stream_plan.schedule.num_points
            or displacements.shape != parameters[:-1].shape
            or not torch.equal(displacements, parameters[1:] - parameters[:-1])
        ):
            raise RotatedArtifactError("rotated trajectory tensor identity failed")
        trajectory_payload = (
            tensor_content_hash(parameters)
            + tensor_content_hash(displacements)
            + stream_plan.content_hash
        )
        import hashlib

        if (
            hashlib.sha256(trajectory_payload.encode("ascii")).hexdigest()
            != run_summary.get("trajectory_hash")
        ):
            raise RotatedArtifactError("rotated trajectory hash does not match")
        if not isinstance(states, dict) or set(states) != {"initial", "final"}:
            raise RotatedArtifactError("rotated model-state artifact is invalid")
        if (
            state_dict_hash(states["initial"])
            != run_summary.get("initial_model_state_hash")
            or state_dict_hash(states["final"])
            != run_summary.get("final_model_state_hash")
        ):
            raise RotatedArtifactError("rotated model-state hash does not match")
    return LoadedRotatedRun(
        path=run_path,
        config=config,
        manifest=manifest,
        partitions=partitions,
        stream_plan=stream_plan,
        initialization_metrics=initialization_metrics,
        trajectory_metrics=tuple(metrics_value),
        run_summary=run_summary,
        stream_inputs=stream_inputs,
        stream_targets=stream_targets,
    )
