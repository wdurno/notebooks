"""Create the immutable artifact-only Plan 6 trend-variance audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .phase5_single_lap_artifacts import load_completed_single_lap_run
from .phase6_artifacts import Phase6RunStore
from .phase6_config import load_phase6_debias_config
from .phase6_debias import (
    reconstruct_debiased_recommendations,
    summarize_debias_rows,
)


REQUIRED = (
    "source_contract.json",
    "debiased_recommendations.json",
    "audit_summary.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/phase6/debias"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(config_path: Path, output_root: Path, *, resume: bool) -> Path:
    config = load_phase6_debias_config(config_path)
    repo_root = Path(__file__).parents[2]
    source_path = repo_root / config.source_run_path
    source = load_completed_single_lap_run(source_path)
    if source.config.run_id != config.source_run_id:
        raise ValueError("loaded source run does not match the frozen contract")
    if source.config.data.samples_per_step != config.deployed_batch_size:
        raise ValueError("source deployed batch size differs from Plan 6")
    if source.config.schedule_kinds != config.schedule_kinds:
        raise ValueError("source schedules differ from Plan 6")

    session = Phase6RunStore(
        output_root,
        run_kind="phase6_artifact_only_trend_variance_audit",
        label="Phase 6 debias audit",
    ).begin(config, repo_root, resume=resume)
    session.write_json(
        "source_contract.json",
        {
            "source_run_id": source.config.run_id,
            "source_config_hash": source.config.config_hash,
            "source_manifest_sha256": _sha256(source.path / "manifest.json"),
            "source_metrics_sha256": _sha256(source.path / "trajectory_metrics.json"),
            "condition": config.condition,
            "schedule_hashes": source.run_summary["schedule_hashes"],
            "deployed_batch_size": config.deployed_batch_size,
            "transition_alignment": "row t weights the transition t to t+1",
        },
    )

    all_rows = {}
    summaries = {}
    for schedule in config.schedule_kinds:
        all_rows[schedule] = {}
        summaries[schedule] = {}
        source_rows = source.trajectory_metrics[schedule][config.condition]
        for scale in config.variance_scales:
            key = f"variance_scale_{scale:g}"
            rows = reconstruct_debiased_recommendations(
                source_rows,
                batch_size=config.deployed_batch_size,
                variance_scale=scale,
                pi_min=config.pi_min,
                pi_max=config.pi_max,
            )
            all_rows[schedule][key] = rows
            summaries[schedule][key] = summarize_debias_rows(
                rows,
                live_start_step=source.config.controller.cold_start_steps,
            )
    primary = {
        schedule: summaries[schedule]["variance_scale_1"]
        for schedule in config.schedule_kinds
    }
    gate = "continue_with_caution"
    session.write_json("debiased_recommendations.json", all_rows)
    session.write_json(
        "audit_summary.json",
        {
            "run_kind": "phase6_artifact_only_trend_variance_audit",
            "config_hash": config.config_hash,
            "source_run_id": config.source_run_id,
            "primary_variance_scale": 1.0,
            "primary_summaries": primary,
            "sensitivity_summaries": summaries,
            "gate": gate,
            "gate_reason": (
                "the correction is finite and material, but does not by itself "
                "explain the large realized EDR actions"
            ),
            "approximation_limits": [
                "the scalar variance energy is propagated across changing Fishers",
                "the uncertainty-scale estimate may contain trend prediction error",
                "deterministic trend curvature is not sampling variance",
            ],
        },
    )
    return session.complete(required=REQUIRED)


def main() -> None:
    arguments = parse_arguments()
    path = run(arguments.config, arguments.output_root, resume=arguments.resume)
    print(json.dumps({"path": str(path), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
