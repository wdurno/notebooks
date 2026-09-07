"""Execute Plan 9 E9.14 structured-gain calibration and pivot gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan9RunStore
from .gain import run_gain_calibration
from .gain_config import load_gain_calibration_config


REQUIRED = ("source_contract.json", "calibration_rows.json", "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("cache/mnist_experiment/rotated_mnist/plan9"))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).parents[3]
    config_path = arguments.config if arguments.config.is_absolute() else repo_root / arguments.config
    output_root = arguments.output_root if arguments.output_root.is_absolute() else repo_root / arguments.output_root
    config = load_gain_calibration_config(config_path)
    session = Plan9RunStore(output_root / config.study).begin(
        run_id=config.run_id,
        run_kind="plan9_e9_14_gain_calibration",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        runtime_config=config,
        resume=arguments.resume,
    )
    try:
        rows, summary, contract = run_gain_calibration(config, repo_root)
        session.write_json("source_contract.json", contract)
        session.write_json("calibration_rows.json", rows)
        session.write_json("summary.json", summary)
        path = session.complete(REQUIRED)
    except KeyboardInterrupt:
        session.mark_interrupted("SIGINT")
        raise
    except BaseException as exc:
        session.mark_interrupted(type(exc).__name__)
        raise
    print(json.dumps({"run_id": config.run_id, "path": str(path), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
