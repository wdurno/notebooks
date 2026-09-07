"""Execute Plan 9 E9.13 dual-timescale structured-gain estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan9RunStore
from .gain import E9_13_REQUIRED, run_gain_estimator
from .gain_config import load_gain_estimator_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("cache/mnist_experiment/rotated_mnist/plan9"))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).parents[3]
    config_path = arguments.config if arguments.config.is_absolute() else repo_root / arguments.config
    output_root = arguments.output_root if arguments.output_root.is_absolute() else repo_root / arguments.output_root
    config = load_gain_estimator_config(config_path)
    session = Plan9RunStore(output_root / config.study).begin(
        run_id=config.run_id,
        run_kind="plan9_e9_13_gain_estimator",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        runtime_config=config,
        resume=arguments.resume,
    )
    try:
        rows, predictions, summary, contract = run_gain_estimator(config, repo_root)
        session.write_json("source_contract.json", contract)
        session.write_json("solver_rows.json", rows)
        session.write_torch("predictions.pt", predictions)
        session.write_json("summary.json", summary)
        path = session.complete(E9_13_REQUIRED)
    except KeyboardInterrupt:
        session.mark_interrupted("SIGINT")
        raise
    except BaseException as exc:
        session.mark_interrupted(type(exc).__name__)
        raise
    print(json.dumps({"run_id": config.run_id, "path": str(path), "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
