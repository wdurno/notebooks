"""Execute the immutable Plan 10 Phase 1 feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan10RunStore
from .config import load_feasibility_config
from .feasibility import build_feasibility_map


REQUIRED = (
    "config.json",
    "source_contract.json",
    "fisher_speed_map.json",
    "finite_step_validation.json",
    "oracle_route.json",
    "summary.json",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/plan10/phase1"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path.cwd()
    config = load_feasibility_config(arguments.config)
    session = Plan10RunStore(arguments.output_root).begin(
        run_id=config.run_id,
        run_kind="phase1_fisher_speed_feasibility",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        experiment_config=config,
        resume=arguments.resume,
    )
    map_rows, validation_rows, route_rows, summary, source_contract = build_feasibility_map(config, repo_root)
    session.write_json("source_contract.json", source_contract)
    session.write_json("fisher_speed_map.json", map_rows)
    session.write_json("finite_step_validation.json", validation_rows)
    session.write_json("oracle_route.json", route_rows)
    session.write_json("summary.json", summary)
    path = session.complete(REQUIRED)
    print(json.dumps({"path": str(path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
