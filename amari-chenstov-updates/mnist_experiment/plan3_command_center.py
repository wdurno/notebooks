"""Preview the staged Plan 3 handoff without creating run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase9 import parse_replica_indices
from src.plan3 import load_plan3_spec, parse_stage_names, preview_plan3


DEFAULT_SPEC = Path(__file__).with_name("plan3_profiles.json")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planning-only Plan 3 command center")
    parser.add_argument("command", choices=("preview",))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--stages", help="comma-separated stage names")
    parser.add_argument("--replicas", help="indices such as 6-10 or 6,8")
    parser.add_argument(
        "--details",
        action="store_true",
        help="include per-condition estimands and planning costs",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    spec = load_plan3_spec(arguments.spec)
    stages = parse_stage_names(arguments.stages, spec)
    replicas = parse_replica_indices(
        arguments.replicas,
        default_start=spec.replica_indices[0],
        default_count=len(spec.replica_indices),
    )
    preview = preview_plan3(
        spec,
        stage_names=stages,
        replica_indices=replicas,
    )
    if not arguments.details:
        preview.pop("conditions")
    print(json.dumps(preview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
