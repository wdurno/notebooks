"""Execute immutable Plan 9 retrospective studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan9RunStore
from .config import load_retrospective_config
from .retrospective import run_e9_1, run_e9_2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/plan9"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    repo_root = Path(__file__).parents[3]
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_root = arguments.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    config = load_retrospective_config(config_path)
    session = Plan9RunStore(output_root / config.study).begin(
        run_id=config.run_id,
        run_kind=f"plan9_{config.study}_retrospective",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        runtime_config=config,
        resume=arguments.resume,
    )
    try:
        result = run_e9_1(config, repo_root) if config.study == "e9_1" else run_e9_2(config, repo_root)
        rows, vectors, summary, source_contract = result
        session.write_json("source_contract.json", source_contract)
        session.write_json("audit_rows.json", rows)
        session.write_torch("vectors.pt", vectors)
        session.write_json("summary.json", summary)
        path = session.complete(
            ("source_contract.json", "audit_rows.json", "vectors.pt", "summary.json")
        )
    except KeyboardInterrupt:
        session.mark_interrupted("SIGINT")
        raise
    except BaseException as exc:
        session.mark_interrupted(type(exc).__name__)
        raise
    print(json.dumps({"run_id": config.run_id, "path": str(path), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
