"""Replay a fitted Phase 2 policy against cached Phase 1 encodings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picar_kl.phase2.runtime import Phase2ReplayConfig, replay_phase2_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a Phase 2 policy against cached Phase 1 data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=None,
        help="Phase 1 data root. May be repeated. Defaults to artifacts/data/phase1.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/models/phase2"))
    parser.add_argument("--fit-id", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--allow-partial-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-steps", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = replay_phase2_policy(
        Phase2ReplayConfig(
            data_roots=tuple(args.data_root or [Path("artifacts/data/phase1")]),
            checkpoint_root=args.checkpoint_root,
            fit_id=args.fit_id,
            checkpoint_path=args.checkpoint_path,
            cache_root=args.cache_root,
            device=args.device,
            max_windows=args.max_windows,
            allow_partial_cache=bool(args.allow_partial_cache),
        )
    )
    payload = result.to_dict()
    print(
        "[phase2-replay] "
        f"fit_id={result.fit_id} device={result.device} "
        f"windows={result.window_count} replayed_steps={result.replayed_step_count} "
        f"checkpoint={result.checkpoint_path}",
        flush=True,
    )
    for step in payload["steps"][: max(0, int(args.print_steps))]:
        print(
            "[phase2-replay] "
            f"window={step['window_index']} target={step['target_index']} "
            f"step={step['step_index']} source={step['source_action_name']} "
            f"pred={step['action']['action_name']} latency={step['latency_seconds']:.6f}",
            flush=True,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[phase2-replay] output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
