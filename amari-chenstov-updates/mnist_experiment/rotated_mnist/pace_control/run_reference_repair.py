"""Execute the immutable Plan 10 Phase 1c reference-repair screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Plan10RunStore
from .reference_config import load_reference_repair_config
from .reference_repair import run_reference_repair_screen


REQUIRED = (
    "config.json",
    "source_contract.json",
    "reference_states.pt",
    "fit_metrics.json",
    "heldout_panel.json",
    "finite_risk_pairs.json",
    "heldout_nll_matrix.pt",
    "summary.json",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("cache/mnist_experiment/datasets"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/plan10/phase1c/screen"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    repo_root = Path.cwd()
    config = load_reference_repair_config(arguments.config)
    session = Plan10RunStore(arguments.output_root).begin(
        run_id=config.run_id,
        run_kind="phase1c_reference_repair_screen",
        config=config.to_mapping(),
        config_hash=config.config_hash,
        repo_root=repo_root,
        experiment_config=config,
        resume=arguments.resume,
    )
    references, fits, panel, pairs, summary, source_contract, nll_matrix = run_reference_repair_screen(
        config,
        repo_root,
        session,
        data_root=arguments.data_root,
        download=arguments.download,
    )
    session.write_json("source_contract.json", source_contract)
    session.write_torch("reference_states.pt", references)
    session.write_json("fit_metrics.json", fits)
    session.write_json("heldout_panel.json", panel)
    session.write_json("finite_risk_pairs.json", pairs)
    session.write_torch("heldout_nll_matrix.pt", nll_matrix)
    session.write_json("summary.json", summary)
    path = session.complete(REQUIRED)
    print(json.dumps({"path": str(path), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
