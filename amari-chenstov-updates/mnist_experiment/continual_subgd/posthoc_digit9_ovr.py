"""Artifact-only digit-9 one-vs-rest evaluation for completed Phase 5 paths."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from mnist_experiment.rotated_mnist.artifacts import _read_json
from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model

from .artifacts import UnitStore, file_hash
from .config import Plan13Study, canonical_hash
from .environments.digit9_mixture import nine_ovr_metrics
from .rotation import runtime


DIGIT9_OVR_SCHEMA_VERSION = "plan13-digit9-ovr-v1"
DIGIT9_OVR_REQUIRED = ("metrics.json", "summary.json", "checks.json")
DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
DEFAULT_NOTEBOOK = Path("mnist_experiment/continual_subgd_results.ipynb")


def _phase5_inputs(
    store: UnitStore,
) -> tuple[dict[int, Path], list[tuple[dict[str, Any], Path]], str]:
    ledger_path = store.root / "ledgers" / "phase5.json"
    ledger = _read_json(ledger_path)
    if ledger.get("study_hash") != store.study.config_hash:
        raise RuntimeError("Phase 5 ledger does not belong to the configured study")
    assets: dict[int, Path] = {}
    trajectories = []
    fingerprints = []
    for item in ledger["items"]:
        if item["action"] not in {"mixture_assets", "mixture_trajectory"}:
            continue
        required = tuple(item["required"])
        completed = store.completed(item["unit"], required)
        if completed is None:
            raise RuntimeError(f"Phase 5 input is incomplete: {item['unit']}")
        fingerprints.append(
            {
                "path": completed.relative_to(store.root).as_posix(),
                "integrity": _read_json(completed / "integrity.json"),
            }
        )
        index = int(item["unit"]["index"])
        if item["action"] == "mixture_assets":
            assets[index] = completed
        else:
            trajectories.append((item, completed))
    if len(assets) != 16 or len(trajectories) != 80:
        raise RuntimeError(
            f"expected 16 Phase 5 asset units and 80 trajectories; "
            f"found {len(assets)} and {len(trajectories)}"
        )
    digest = canonical_hash(
        {
            "schema_version": DIGIT9_OVR_SCHEMA_VERSION,
            "ledger_sha256": file_hash(ledger_path),
            "inputs": sorted(fingerprints, key=lambda value: value["path"]),
        }
    )
    return assets, trajectories, digest


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> torch.Tensor:
    return torch.cat(
        [model(inputs[start : start + chunk_size]).argmax(dim=1) for start in range(0, len(inputs), chunk_size)]
    )


def _normalized_auc(rows: list[dict[str, Any]], field: str) -> float:
    ordered = sorted(rows, key=lambda row: int(row["post_burn_in_observations"]))
    left = float(ordered[0]["post_burn_in_observations"])
    right = float(ordered[-1]["post_burn_in_observations"])
    if right <= left:
        raise ValueError("OvR trajectory needs at least two distinct evaluation points")
    area = 0.0
    for previous, current in zip(ordered, ordered[1:]):
        width = float(current["post_burn_in_observations"]) - float(
            previous["post_burn_in_observations"]
        )
        area += width * (float(previous[field]) + float(current[field])) / 2
    return area / (right - left)


def _sidecar_name(index: int, condition: str) -> str:
    safe_condition = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in condition
    )
    return f"rows/{index:05d}__{safe_condition}.json"


def run_digit9_ovr_posthoc(
    store: UnitStore,
    *,
    resume: bool,
) -> Path:
    asset_paths, trajectory_inputs, input_digest = _phase5_inputs(store)
    unit = store.unit(
        "phase5_posthoc",
        "digit9_ovr",
        1,
        environment="digit9_mixture",
        detail={
            "schema_version": DIGIT9_OVR_SCHEMA_VERSION,
            "phase5_input_digest": input_digest,
            "positive_class": 9,
            "current_accuracy_definition": "p_t * TPR_9 + (1 - p_t) * TNR_9",
            "balanced_accuracy_definition": "(TPR_9 + TNR_9) / 2",
        },
    )
    session = store.begin(unit, DIGIT9_OVR_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, DIGIT9_OVR_REQUIRED)
        assert completed is not None
        return completed

    by_replica: dict[int, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for item, path in trajectory_inputs:
        by_replica[int(item["unit"]["index"])].append((item, path))

    device, training_dtype, _ = runtime(store.study)
    all_rows = []
    recall_errors = []
    for index in sorted(by_replica):
        assets = torch.load(
            asset_paths[index] / "assets.pt", map_location="cpu", weights_only=False
        )
        evaluation_inputs = assets["evaluation_inputs"].to(
            device=device, dtype=training_dtype
        )
        evaluation_targets = assets["evaluation_targets"].to(device=device)
        model, layout = build_gauge_fixed_model(
            store.study.seed("phase5:posthoc_digit9_ovr", index),
            device=device,
            dtype=training_dtype,
        )
        model.eval()
        for item, trajectory_path in sorted(
            by_replica[index], key=lambda value: value[0]["unit"]["condition"]
        ):
            condition = str(item["unit"]["condition"])
            sidecar_name = _sidecar_name(index, condition)
            sidecar_path = session.working_path / sidecar_name
            if sidecar_path.is_file():
                sidecar = _read_json(sidecar_path)
                if (
                    sidecar.get("schema_version") != DIGIT9_OVR_SCHEMA_VERSION
                    or sidecar.get("replica_index") != index
                    or sidecar.get("condition") != condition
                ):
                    raise RuntimeError(f"incompatible resumable OvR sidecar: {sidecar_path}")
                all_rows.extend(sidecar["rows"])
                recall_errors.append(float(sidecar["max_abs_recall_error"]))
                print(f"[digit9-ovr] reused replica {index:02d} / {condition}")
                continue

            trajectory = torch.load(
                trajectory_path / "trajectory.pt", map_location="cpu", weights_only=False
            )
            original_rows = _read_json(trajectory_path / "metrics.json")
            parameters = trajectory["parameters"]
            if len(parameters) != len(original_rows):
                raise RuntimeError(
                    f"parameter/metric length mismatch for {trajectory_path}"
                )
            derived_rows = []
            max_abs_recall_error = 0.0
            for parameter, original in zip(parameters, original_rows):
                layout.copy_vector_to_module(
                    model, parameter.to(device=device, dtype=training_dtype)
                )
                predictions = _predict(model, evaluation_inputs)
                metrics = nine_ovr_metrics(
                    predictions, evaluation_targets, float(original["p"])
                )
                max_abs_recall_error = max(
                    max_abs_recall_error,
                    abs(metrics["nine_ovr_recall"] - float(original["p1_accuracy"])),
                )
                derived_rows.append(
                    {
                        "replica_index": index,
                        "condition": condition,
                        "step": int(original["step"]),
                        "post_burn_in_step": int(original["post_burn_in_step"]),
                        "post_burn_in_observations": int(
                            original["post_burn_in_observations"]
                        ),
                        "p": float(original["p"]),
                        **metrics,
                    }
                )
            sidecar = {
                "schema_version": DIGIT9_OVR_SCHEMA_VERSION,
                "phase5_input_digest": input_digest,
                "replica_index": index,
                "condition": condition,
                "max_abs_recall_error": max_abs_recall_error,
                "rows": derived_rows,
            }
            session.write_json(sidecar_name, sidecar)
            all_rows.extend(derived_rows)
            recall_errors.append(max_abs_recall_error)
            print(f"[digit9-ovr] completed replica {index:02d} / {condition}")

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in all_rows:
        grouped[row["condition"]][int(row["replica_index"])].append(row)
    condition_summary = []
    for condition in sorted(grouped):
        replica_rows = grouped[condition]
        condition_summary.append(
            {
                "condition": condition,
                "replicas": len(replica_rows),
                "mean_nine_ovr_accuracy_auc": statistics.fmean(
                    _normalized_auc(rows, "nine_ovr_accuracy")
                    for rows in replica_rows.values()
                ),
                "mean_nine_ovr_balanced_accuracy_auc": statistics.fmean(
                    _normalized_auc(rows, "nine_ovr_balanced_accuracy")
                    for rows in replica_rows.values()
                ),
                "mean_nine_ovr_recall_auc": statistics.fmean(
                    _normalized_auc(rows, "nine_ovr_recall")
                    for rows in replica_rows.values()
                ),
                "mean_nine_ovr_specificity_auc": statistics.fmean(
                    _normalized_auc(rows, "nine_ovr_specificity")
                    for rows in replica_rows.values()
                ),
            }
        )
    expected_rows = sum(
        len(_read_json(path / "metrics.json")) for _, path in trajectory_inputs
    )
    checks = {
        "phase5_input_digest": input_digest,
        "trajectory_count": len(trajectory_inputs),
        "metric_row_count": len(all_rows),
        "expected_metric_row_count": expected_rows,
        "replicas_per_condition": {
            condition: len(replica_rows) for condition, replica_rows in grouped.items()
        },
        "max_abs_recall_error_against_phase5": max(recall_errors, default=math.inf),
        "all_finite": all(
            math.isfinite(float(value))
            for row in all_rows
            for key, value in row.items()
            if key.startswith("nine_ovr_")
        ),
    }
    checks["passed"] = (
        checks["trajectory_count"] == 80
        and checks["metric_row_count"] == checks["expected_metric_row_count"]
        and set(checks["replicas_per_condition"].values()) == {16}
        and checks["max_abs_recall_error_against_phase5"] <= 1e-7
        and checks["all_finite"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"digit-9 OvR post-hoc checks failed: {checks}")
    summary = {
        "schema_version": DIGIT9_OVR_SCHEMA_VERSION,
        "study_hash": store.study.config_hash,
        "phase5_input_digest": input_digest,
        "environment": "digit9_mixture",
        "condition_summary": condition_summary,
    }
    session.write_json("metrics.json", all_rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, DIGIT9_OVR_REQUIRED)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    completed = run_digit9_ovr_posthoc(store, resume=arguments.resume)
    from .refresh_notebook import refresh

    refresh(store, arguments.notebook)
    print(completed)


if __name__ == "__main__":
    main()


__all__ = [
    "DIGIT9_OVR_REQUIRED",
    "DIGIT9_OVR_SCHEMA_VERSION",
    "run_digit9_ovr_posthoc",
]
