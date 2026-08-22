"""Execute the immutable Plan 4 Fisher-risk score-only diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.plan4_fisher_analysis import write_fisher_diagnostic


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-count", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path(__file__).parents[1]
    destination = write_fisher_diagnostic(
        repo_root,
        source_count=arguments.source_count,
        device_name=arguments.device,
    )
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "path": str(destination.relative_to(repo_root)),
                "screening_schedule": summary["screening_schedule"],
                "source_count": summary["source_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
