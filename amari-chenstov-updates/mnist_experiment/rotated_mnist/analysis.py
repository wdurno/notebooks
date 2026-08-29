"""Read-only lightweight summaries for rotated-MNIST notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import (
    LoadedRotatedRun,
    RotatedArtifactError,
    RotatedIncompleteRunError,
    load_completed_run,
)


def discover_completed_runs(root: str | Path) -> tuple[Path, ...]:
    root_path = Path(root)
    incomplete_root = root_path / ".incomplete"
    incomplete = (
        tuple(path for path in incomplete_root.iterdir() if path.is_dir())
        if incomplete_root.is_dir()
        else ()
    )
    if incomplete:
        names = ", ".join(sorted(path.name for path in incomplete))
        raise RotatedIncompleteRunError(
            f"rotated-MNIST run root contains incomplete runs: {names}"
        )
    if not root_path.exists():
        return ()
    paths = tuple(
        sorted(
            path
            for path in root_path.iterdir()
            if path.is_dir() and path.name != ".incomplete"
        )
    )
    for path in paths:
        if not (path / "COMPLETED").is_file():
            raise RotatedArtifactError(
                f"rotated-MNIST run lacks COMPLETED marker: {path}"
            )
    return paths


def load_completed_summaries(root: str | Path) -> tuple[LoadedRotatedRun, ...]:
    return tuple(load_completed_run(path) for path in discover_completed_runs(root))


def summary_rows(runs: tuple[LoadedRotatedRun, ...]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run.config.run_id,
            "replica_id": run.config.replica_id,
            "condition": run.run_summary["condition"],
            "num_points": run.run_summary["num_points"],
            "samples_per_step": run.run_summary["samples_per_step"],
            "final_current_nine_ovr_accuracy": run.run_summary[
                "final_current_nine_ovr_accuracy"
            ],
            "final_current_environment_accuracy": run.run_summary[
                "final_current_environment_accuracy"
            ],
        }
        for run in runs
    ]
