from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .schemas import ExperimentRunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment_interface",
        description="Run PiCar experimental phases with UUID-scoped data/model persistence.",
    )
    parser.add_argument("--phase", choices=("init", "tune", "retask"), required=True)
    parser.add_argument(
        "--picar-host",
        type=str,
        default=None,
        help="PiCar API host:port (overrides PICAR_V_HOST if provided).",
    )
    parser.add_argument(
        "--deterministic-coding",
        action="store_true",
        help="Use deterministic policy text decoding (do_sample=False).",
    )
    parser.add_argument("--fixed-t", type=float, default=None, help="Pin interpolation t in [0, 1].")
    parser.add_argument(
        "--t-step",
        type=float,
        default=0.001,
        help="Linear t increment per environment step when traversing.",
    )
    parser.add_argument(
        "--t-log-every",
        type=int,
        default=1,
        help="Print t progress every N steps when traversing.",
    )
    parser.add_argument(
        "--load-snapshot",
        type=Path,
        default=None,
        help="Optional snapshot path or directory to initialize model state from.",
    )
    parser.add_argument(
        "--reward-prompt",
        type=str,
        default="find the red ball",
        help="Reward task prompt text (retask phase uses this directly).",
    )
    parser.add_argument("--snapshot-keep", type=int, default=3, help="Maximum snapshots to retain per run.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for run data (default: demo/data).")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Root directory for model snapshots (default: demo/model).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fit-iters", type=int, default=32)
    parser.add_argument("--train-every-steps", type=int, default=16)
    parser.add_argument("--min-replay-size", type=int, default=32)
    parser.add_argument("--memorize-every-steps", type=int, default=64)
    parser.add_argument("--memorize-n", type=int, default=64)
    parser.add_argument(
        "--memorize-random-idx",
        action="store_true",
        help="Sample random replay indices during SSR memorization.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help="Set runtime logging verbosity.",
    )
    return parser


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    from .run_controller import run_experiment

    config = ExperimentRunConfig(
        phase=args.phase,
        picar_host=args.picar_host,
        deterministic_coding=bool(args.deterministic_coding),
        fixed_t=args.fixed_t,
        t_step=args.t_step,
        t_log_every=args.t_log_every,
        load_snapshot=args.load_snapshot,
        reward_prompt=args.reward_prompt,
        snapshot_keep=args.snapshot_keep,
        data_root=args.data_root,
        model_root=args.model_root,
        batch_size=args.batch_size,
        fit_iters=args.fit_iters,
        train_every_steps=args.train_every_steps,
        min_replay_size=args.min_replay_size,
        memorize_every_steps=args.memorize_every_steps,
        memorize_n=args.memorize_n,
        memorize_random_idx=bool(args.memorize_random_idx),
    )
    summary = run_experiment(config)
    print(
        f"[experiment] done uuid={summary.run_uuid} steps={summary.steps_completed} "
        f"data={summary.data_run_dir} model={summary.model_run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
