"""Plan 13 Phase 6A: exploratory ten-positive low-prevalence study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import UnitStore
from .config import Plan13Study
from .phase6_low_prevalence import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_NOTEBOOK,
    DEFAULT_ROOT,
    PHASE6A,
    PHASE6A_CUDA_SMOKE,
    PHASE6A_SMOKE,
    PRODUCTION_REPLICAS,
    build_ledger,
    expected_schedule,
    low_prevalence_data_config,
    phase6a_conditions,
    run,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.smoke and arguments.cuda_smoke:
        raise ValueError("choose either --smoke or --cuda-smoke")
    if arguments.max_units is not None and arguments.max_units < 1:
        raise ValueError("--max-units must be positive")
    if arguments.max_wall_seconds is not None and arguments.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    smoke = arguments.smoke or arguments.cuda_smoke
    phase = (
        PHASE6A_CUDA_SMOKE
        if arguments.cuda_smoke
        else PHASE6A_SMOKE if arguments.smoke else PHASE6A
    )
    replicas = (1,) if smoke else tuple(range(1, PRODUCTION_REPLICAS + 1))
    completed, exhausted = run(
        store,
        phase=phase,
        smoke=smoke,
        replicas=replicas,
        data_root=arguments.data_root,
        notebook=arguments.notebook,
        resume=arguments.resume,
        max_units=arguments.max_units,
        max_wall_seconds=arguments.max_wall_seconds,
    )
    print(f"[phase6a] completed_now={completed} exhausted={exhausted}")


if __name__ == "__main__":
    main()


__all__ = [
    "PHASE6A",
    "PRODUCTION_REPLICAS",
    "build_ledger",
    "expected_schedule",
    "low_prevalence_data_config",
    "phase6a_conditions",
]
