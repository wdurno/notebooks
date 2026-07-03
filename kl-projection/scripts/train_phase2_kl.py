"""Train the offline phase 2 KL-projection policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from picar_kl.phase2.train import Phase2TrainingConfig, run_phase2_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the phase 2 KL-projection LSTM policy.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=None,
        help="Phase 1 data root. May be repeated. Defaults to artifacts/data/phase1.",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/data/phase2/encodings"))
    parser.add_argument("--cache-model-name", default="qwen2.5-vl-3b")
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("artifacts/manifests/tracked/models/vlm_models.json"),
    )
    parser.add_argument("--cache-encoder-id", default="qwen2.5-vl-get-image-features-pooler-output-v1")
    parser.add_argument("--cache-config-hash", default="output_dtype=float16")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/runs/phase2"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/models/phase2"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument("--prediction-steps", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--conditioning-dim", type=int, default=32)
    parser.add_argument("--conditioning-hidden-dim", type=int, default=128)
    parser.add_argument("--token-type-dim", type=int, default=8)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--allow-partial-cache", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_phase2_training(
        Phase2TrainingConfig(
            data_roots=tuple(args.data_root or [Path("artifacts/data/phase1")]),
            cache_root=args.cache_root,
            cache_model_name=args.cache_model_name,
            cache_manifest_path=args.cache_manifest,
            cache_encoder_id=args.cache_encoder_id,
            cache_config_hash=args.cache_config_hash,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            run_name=args.run_name,
            context_steps=args.context_steps,
            prediction_steps=args.prediction_steps,
            window_stride=args.window_stride,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_dim=args.model_dim,
            conditioning_dim=args.conditioning_dim,
            conditioning_hidden_dim=args.conditioning_hidden_dim,
            token_type_dim=args.token_type_dim,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            seed=args.seed,
            max_windows=args.max_windows,
            save_checkpoint=not args.no_checkpoint,
            allow_partial_cache=bool(args.allow_partial_cache),
        )
    )
    print(
        "[phase2-train] "
        f"run_id={result.run_id} final_loss={result.final_loss:.6f} "
        f"windows={result.window_count} valid_steps={result.valid_steps} summary={result.summary_path}",
        flush=True,
    )
    if result.checkpoint_path is not None:
        print(f"[phase2-train] checkpoint={result.checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
