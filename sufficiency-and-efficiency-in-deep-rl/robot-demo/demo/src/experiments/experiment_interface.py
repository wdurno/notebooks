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
        "--update-mode",
        choices=("auto", "batch", "online"),
        default="auto",
        help="Choose the training backend (`auto`: init=batch, tune/retask=online).",
    )
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
    parser.add_argument(
        "--history-window",
        type=int,
        default=12,
        help="Number of recent messages kept in rolling context.",
    )
    parser.add_argument(
        "--prompt-token-window",
        type=int,
        default=512,
        help="Token budget cap for model prompts (0 disables token-based truncation).",
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--all-images",
        dest="all_images",
        action="store_true",
        help="Attach an image placeholder to every retained user message.",
    )
    image_group.add_argument(
        "--latest-image-only",
        dest="all_images",
        action="store_false",
        help="Only attach an image placeholder to the most recent user message (default).",
    )
    parser.set_defaults(all_images=False)
    parser.add_argument(
        "--init-t",
        type=float,
        default=0.0,
        help="Initial interpolation t in [0, 1] when traversing (phase tune with no --fixed-t).",
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
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--load-snapshot",
        type=Path,
        default=None,
        help="Optional snapshot path or directory to initialize model state from.",
    )
    load_group.add_argument(
        "--load-latest-from-model-root",
        action="store_true",
        help="Load the most recent snapshot found under --model-root.",
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
    parser.add_argument("--epochs", type=int, default=1, help="Optimizer steps per training trigger in phase 2/3.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Optimizer learning rate.")
    parser.add_argument(
        "--fit-iters",
        type=int,
        default=None,
        help=(
            "Iterations per `model.fit(...)` call. If omitted, auto-resolves to "
            "ceil(replay_size / batch_size) at each training trigger."
        ),
    )
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
        update_mode=args.update_mode,
        picar_host=args.picar_host,
        deterministic_coding=bool(args.deterministic_coding),
        history_window=int(args.history_window),
        all_images=bool(args.all_images),
        prompt_token_window=int(args.prompt_token_window),
        init_t=float(args.init_t),
        fixed_t=args.fixed_t,
        t_step=args.t_step,
        t_log_every=args.t_log_every,
        load_snapshot=args.load_snapshot,
        load_latest_from_model_root=bool(args.load_latest_from_model_root),
        reward_prompt=args.reward_prompt,
        snapshot_keep=args.snapshot_keep,
        data_root=args.data_root,
        model_root=args.model_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
