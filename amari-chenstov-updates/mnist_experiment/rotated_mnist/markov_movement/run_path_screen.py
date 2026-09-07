"""Execute immutable Plan 9 E9.10 opportunity-qualified path screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan9RunStore
from .config import load_path_screen_config
from .path_screen import run_path_screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/plan9"),
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).parents[3]
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_root = arguments.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    config = load_path_screen_config(config_path)
    session = Plan9RunStore(output_root / config.study).begin(
        run_id=config.run_id,
        run_kind="plan9_e9_10_opportunity_path_screen",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        runtime_config=config,
        resume=arguments.resume,
    )
    try:
        rows, summary, contract = run_path_screen(config, repo_root)
        session.write_json("source_contract.json", contract)
        session.write_json("screen_rows.json", rows)
        session.write_json("summary.json", summary)
        path = session.complete(("source_contract.json", "screen_rows.json", "summary.json"))
    except KeyboardInterrupt:
        session.mark_interrupted("SIGINT")
        raise
    except BaseException as exc:
        session.mark_interrupted(type(exc).__name__)
        raise
    print(json.dumps({"run_id": config.run_id, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
