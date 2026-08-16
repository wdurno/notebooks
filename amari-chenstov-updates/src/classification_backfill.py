"""Immutable inference-only classification summaries for saved trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

CLASSIFICATION_BACKFILL_SCHEMA_VERSION = 1
CLASSIFICATION_BACKFILL_DIRECTORY = "classification_backfills"


class ClassificationBackfillError(RuntimeError):
    """Raised when a classification backfill is absent or incompatible."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def default_classification_backfill_root(repo_root: str | Path) -> Path:
    return (
        Path(repo_root)
        / "cache"
        / "mnist_experiment"
        / CLASSIFICATION_BACKFILL_DIRECTORY
    )


def classification_backfill_identity(
    source_run_id: str, source_trajectory_sha256: str
) -> str:
    payload = json.dumps(
        {
            "schema_version": CLASSIFICATION_BACKFILL_SCHEMA_VERSION,
            "source_run_id": source_run_id,
            "source_trajectory_sha256": source_trajectory_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:16]
    return f"classification-v1__{source_run_id}__{suffix}"


def _source_identity(source_run_path: Path) -> tuple[str, str, Path]:
    trajectory_path = source_run_path / "phase8_trajectories.pt"
    if not (source_run_path / "COMPLETED").is_file():
        raise ClassificationBackfillError(
            f"source controller run is incomplete: {source_run_path}"
        )
    if not trajectory_path.is_file():
        raise ClassificationBackfillError(
            f"source trajectory is missing: {trajectory_path}"
        )
    source_run_id = source_run_path.name
    return source_run_id, _sha256_file(trajectory_path), trajectory_path


def classification_backfill_path(
    source_run_path: str | Path, output_root: str | Path
) -> Path:
    source_run = Path(source_run_path)
    run_id, trajectory_hash, _ = _source_identity(source_run)
    return Path(output_root) / classification_backfill_identity(
        run_id, trajectory_hash
    )


def load_classification_backfill(
    source_run_path: str | Path,
    output_root: str | Path,
) -> Mapping[str, Any] | None:
    """Load a compatible completed sidecar, returning None when not yet built."""

    source_run = Path(source_run_path)
    root = Path(output_root)
    if not root.is_dir():
        return None
    run_id = source_run.name
    if not any(root.glob(f"classification-v1__{run_id}__*")):
        return None
    run_id, trajectory_hash, _ = _source_identity(source_run)
    path = root / classification_backfill_identity(
        run_id, trajectory_hash
    )
    if not path.exists():
        return None
    if not (path / "COMPLETED").is_file():
        return None
    try:
        artifact = json.loads(
            (path / "classification_metrics.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassificationBackfillError(
            f"classification backfill is unreadable: {path}"
        ) from exc
    if artifact.get("schema_version") != CLASSIFICATION_BACKFILL_SCHEMA_VERSION:
        raise ClassificationBackfillError(
            f"unsupported classification backfill schema: {path}"
        )
    if artifact.get("source_run_id") != run_id:
        raise ClassificationBackfillError(
            f"classification backfill source run mismatch: {path}"
        )
    if artifact.get("source_trajectory_sha256") != trajectory_hash:
        raise ClassificationBackfillError(
            f"classification backfill trajectory mismatch: {path}"
        )
    return artifact


def overlay_classification_backfill(
    rows: Sequence[Mapping[str, Any]],
    source_run_path: str | Path,
    output_root: str | Path,
) -> list[dict[str, Any]]:
    """Overlay sidecar fields on scalar rows without mutating source artifacts."""

    artifact = load_classification_backfill(source_run_path, output_root)
    if artifact is None:
        return [dict(row) for row in rows]
    indexed = {
        (str(row["method"]), int(row["step"])): row
        for row in artifact.get("rows", ())
    }
    if len(indexed) != len(artifact.get("rows", ())):
        raise ClassificationBackfillError("classification backfill has duplicate rows")
    results = []
    for source in rows:
        key = (str(source["method"]), int(source["step"]))
        derived = indexed.get(key)
        if derived is None:
            raise ClassificationBackfillError(
                f"classification backfill is missing {key}"
            )
        results.append(
            {
                **source,
                **{
                    name: value
                    for name, value in derived.items()
                    if name not in {"method", "step", "p"}
                },
                "classification_metric_source": str(
                    artifact["backfill_id"]
                ),
            }
        )
    return results


def build_classification_backfill(
    source_run_path: str | Path,
    *,
    repo_root: str | Path,
    output_root: str | Path,
    data_root: str | Path,
    replica_root: str | Path,
    device_name: str,
    resume: bool = False,
) -> Path:
    """Evaluate every saved state and atomically publish a classification sidecar."""

    import torch
    from torch.utils.data import Subset
    from tqdm.auto import tqdm

    from .artifacts import collect_runtime_metadata
    from .classification_metrics import (
        nine_environment_metrics,
        select_nine_classification_metrics,
    )
    from .config import load_config
    from .initialization import evaluate_classifier, load_replica_bundle_for_config
    from .mnist_data import load_mnist_datasets
    from .mnist_model import resolve_device, resolve_dtype
    from .results_analysis import load_phase8_controller_run

    source_path = Path(source_run_path)
    validated = load_phase8_controller_run(source_path)
    config = load_config(source_path / "config.json")
    run_id, trajectory_hash, trajectory_path = _source_identity(source_path)
    backfill_id = classification_backfill_identity(run_id, trajectory_hash)
    root = Path(output_root)
    final_path = root / backfill_id
    working_path = root / ".incomplete" / backfill_id
    expected_source = {
        "backfill_schema_version": CLASSIFICATION_BACKFILL_SCHEMA_VERSION,
        "backfill_id": backfill_id,
        "source_run_id": run_id,
        "source_config_hash": config.config_hash,
        "source_metric_schema_version": validated.metrics[
            "phase8_metric_schema_version"
        ],
        "source_trajectory_sha256": trajectory_hash,
    }
    if (final_path / "COMPLETED").is_file():
        loaded = load_classification_backfill(source_path, root)
        if loaded is None:
            raise ClassificationBackfillError(
                f"completed classification backfill is invalid: {final_path}"
            )
        return final_path
    if final_path.exists():
        raise ClassificationBackfillError(
            f"classification backfill path is invalid: {final_path}"
        )
    if working_path.exists() and not resume:
        raise ClassificationBackfillError(
            f"incomplete classification backfill exists; pass resume: {working_path}"
        )
    if not working_path.exists():
        working_path.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            working_path / "manifest.json",
            {
                **expected_source,
                "status": "incomplete",
                "started_at": _utc_now(),
                "completed_at": None,
                "requested_device": device_name,
                "runtime": collect_runtime_metadata(repo_root, config),
            },
        )
        _atomic_json(working_path / "partial_states.json", {"states": []})
    else:
        manifest = json.loads(
            (working_path / "manifest.json").read_text(encoding="utf-8")
        )
        mismatched = {
            name: (manifest.get(name), value)
            for name, value in expected_source.items()
            if manifest.get(name) != value
        }
        if mismatched:
            raise ClassificationBackfillError(
                f"incomplete classification backfill mismatch: {mismatched}"
            )

    device = resolve_device(device_name)
    dtype = resolve_dtype(config.runtime.training_dtype)
    _, test_dataset = load_mnist_datasets(data_root, download=False)
    loaded_bundle = load_replica_bundle_for_config(
        replica_root, config, device=device
    )
    evaluation_dataset = Subset(
        test_dataset, loaded_bundle.partitions.evaluation
    )
    trajectory = torch.load(
        trajectory_path, map_location="cpu", weights_only=False
    )
    p_values = tuple(float(value) for value in trajectory["p_values"])
    methods = tuple(str(method) for method in trajectory["conditions"])
    partial_path = working_path / "partial_states.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    states = list(partial.get("states", ()))
    state_index = {
        (str(row["method"]), int(row["parameter_index"])): row
        for row in states
    }
    expected_state_count = sum(
        int(trajectory["conditions"][method]["parameters"].shape[0])
        for method in methods
    )
    progress = tqdm(
        total=expected_state_count,
        initial=len(state_index),
        desc="classification backfill",
        unit="state",
    )
    model = loaded_bundle.model.to(device=device, dtype=dtype)
    try:
        for method in methods:
            parameters = trajectory["conditions"][method]["parameters"]
            for parameter_index, parameter in enumerate(parameters):
                key = (method, parameter_index)
                if key in state_index:
                    continue
                vector = parameter.to(device=device, dtype=dtype)
                loaded_bundle.layout.copy_vector_to_module(model, vector)
                evaluation = evaluate_classifier(
                    model,
                    evaluation_dataset,
                    batch_size=config.initialization.batch_size,
                    device=device,
                    dtype=dtype,
                    num_workers=0,
                    nine_prevalence=p_values[min(parameter_index, len(p_values) - 1)],
                )
                row = {
                    "method": method,
                    "parameter_index": parameter_index,
                    "non_nine_accuracy": evaluation["non_nine_accuracy"],
                    **select_nine_classification_metrics(evaluation),
                }
                states.append(row)
                state_index[key] = row
                _atomic_json(partial_path, {"states": states})
                progress.update(1)
    finally:
        progress.close()

    rows = []
    for method in methods:
        parameter_count = int(
            trajectory["conditions"][method]["parameters"].shape[0]
        )
        for step, p in enumerate(p_values):
            stage_fields = {}
            for stage, parameter_index in (
                ("before", step),
                ("after", min(step + 1, parameter_count - 1)),
            ):
                state = state_index[(method, parameter_index)]
                weighted = nine_environment_metrics(
                    prevalence=p,
                    recall=float(state["nine_recall"]),
                    false_positive_rate=float(state["nine_false_positive_rate"]),
                    non_nine_accuracy=float(state["non_nine_accuracy"]),
                )
                selected = {
                    **{
                        name: state[name]
                        for name in (
                            "nine_true_positive_count",
                            "nine_false_positive_count",
                            "nine_true_negative_count",
                            "nine_false_negative_count",
                        )
                    },
                    **weighted,
                }
                stage_fields.update(
                    {f"{stage}_{name}": value for name, value in selected.items()}
                )
            rows.append({"method": method, "step": step, "p": p, **stage_fields})

    artifact = {
        "schema_version": CLASSIFICATION_BACKFILL_SCHEMA_VERSION,
        "backfill_id": backfill_id,
        "source_run_id": run_id,
        "source_config_hash": config.config_hash,
        "source_metric_schema_version": validated.metrics[
            "phase8_metric_schema_version"
        ],
        "source_trajectory_sha256": trajectory_hash,
        "holdout_partition_hash": loaded_bundle.partitions.content_hash,
        "row_count": len(rows),
        "metric_contract": {
            "replica_unit": True,
            "prevalence_coordinate": "row_p",
            "nine_accuracy_legacy_semantics": "nine_recall",
            "precision_null_when_undefined": True,
        },
        "rows": rows,
    }
    _atomic_json(working_path / "classification_metrics.json", artifact)
    manifest = json.loads(
        (working_path / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "actual_device": str(device),
            "row_count": len(rows),
        }
    )
    _atomic_json(working_path / "manifest.json", manifest)
    (working_path / "COMPLETED").touch()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(working_path, final_path)
    return final_path
