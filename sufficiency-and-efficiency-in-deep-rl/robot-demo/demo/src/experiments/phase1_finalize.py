from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

try:
    from src.model import ModelConfig, ModelObservation, PiCarActionModel, Transition, TransitionReplayBuffer
except ModuleNotFoundError:
    from model import ModelConfig, ModelObservation, PiCarActionModel, Transition, TransitionReplayBuffer

from .snapshot_store import SnapshotStore, resolve_snapshot_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase1FinalizeConfig:
    data_runs: list[Path]
    model_root: Path | None = None
    load_snapshot: Path | None = None
    epochs: int = 1
    batch_size: int = 8
    fit_iters: int | None = None
    prompt_token_window: int = 512
    progress_every: int = 0
    memorize_random_idx: bool = False
    snapshot_keep: int = 3
    run_uuid: str | None = None


@dataclass(frozen=True)
class Phase1FinalizeSummary:
    run_uuid: str
    model_run_dir: Path
    snapshot_path: Path
    source_run_count: int
    transitions_loaded: int
    fit_calls: int
    memorized_count: int


def default_demo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase1_finalize",
        description="Finalize phase-1 data by offline tuning and full memorization.",
    )
    parser.add_argument(
        "--data-runs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more phase-1 run directories under demo/data.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Model root directory (default: demo/model).",
    )
    parser.add_argument(
        "--load-snapshot",
        type=Path,
        default=None,
        help="Optional snapshot path/dir to initialize from before tuning.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of offline epochs over replay sampling.")
    parser.add_argument("--batch-size", type=int, default=8, help="Replay batch size for each fit call.")
    parser.add_argument(
        "--fit-iters",
        type=int,
        default=None,
        help=(
            "Iterations per `model.fit(...)` call. If omitted, defaults to "
            "ceil(replay_size / batch_size) for an approximate one-pass epoch."
        ),
    )
    parser.add_argument(
        "--prompt-token-window",
        type=int,
        default=512,
        help="Token budget cap for prompts during offline finalize fit (0 disables token-based truncation).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Log fit progress every N epochs (0 = auto cadence).",
    )
    parser.add_argument(
        "--memorize-random-idx",
        action="store_true",
        help="Use random replay indices during final SSR memorization.",
    )
    parser.add_argument("--snapshot-keep", type=int, default=3, help="Snapshot retention count.")
    parser.add_argument("--run-uuid", type=str, default=None, help="Optional explicit output UUID.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Set runtime logging verbosity.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    config = Phase1FinalizeConfig(
        data_runs=[Path(path) for path in args.data_runs],
        model_root=args.model_root,
        load_snapshot=args.load_snapshot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fit_iters=args.fit_iters,
        prompt_token_window=args.prompt_token_window,
        progress_every=args.progress_every,
        memorize_random_idx=bool(args.memorize_random_idx),
        snapshot_keep=args.snapshot_keep,
        run_uuid=args.run_uuid,
    )
    summary = run_phase1_finalize(config)
    LOGGER.info(
        f"[phase1_finalize] done uuid={summary.run_uuid} "
        f"sources={summary.source_run_count} transitions={summary.transitions_loaded} "
        f"fit_calls={summary.fit_calls} memorized={summary.memorized_count} "
        f"snapshot={summary.snapshot_path}",
    )
    return 0


def run_phase1_finalize(config: Phase1FinalizeConfig) -> Phase1FinalizeSummary:
    _validate_config(config)
    run_uuid = config.run_uuid or str(uuid4())
    model_root = (config.model_root or (default_demo_root() / "model")).resolve()
    model_run_dir = model_root / run_uuid
    model_run_dir.mkdir(parents=True, exist_ok=False)
    source_runs = resolve_data_runs(config.data_runs)
    transitions_estimate = sum(count_step_rows(path) for path in source_runs)
    if transitions_estimate <= 0:
        raise ValueError("No `source=step` rows found in the provided data runs.")

    replay_buffer = TransitionReplayBuffer(capacity=transitions_estimate)
    transitions_loaded = 0
    max_step_index = 0
    for run_dir in source_runs:
        for transition in iter_transitions_from_run(run_dir):
            replay_buffer.add(transition)
            transitions_loaded += 1
            max_step_index = max(max_step_index, int(transition.next_observation.step_index))

    if transitions_loaded <= 0:
        raise ValueError("No transitions could be reconstructed from the provided runs.")

    model = PiCarActionModel(
        replay_buffer=replay_buffer,
        config=ModelConfig(
            model_dir=model_root,
            prompt_token_window=(
                int(config.prompt_token_window) if int(config.prompt_token_window) > 0 else None
            ),
        ),
    )
    snapshot_store = SnapshotStore(model_run_dir, max_keep=config.snapshot_keep)

    loaded_snapshot_path = None
    if config.load_snapshot is not None:
        loaded_snapshot_path = resolve_snapshot_path(config.load_snapshot)
        result = snapshot_store.load_into_model(snapshot_path=loaded_snapshot_path, model=model)
        LOGGER.info(
            f"[phase1_finalize] loaded snapshot={result.path} "
            f"trainable_keys={len(result.loaded_trainable_keys)} ssr={result.has_ssr_state}",
        )

    memorized_count = len(replay_buffer)
    _set_model_optimization_mode(model)
    model.memorize(n=memorized_count, random_idx=config.memorize_random_idx, disable_tqdm=True)
    LOGGER.info("[phase1_finalize] memorize done count=%d", memorized_count)
    _log_memory_diagnostics(stage="after_memorize_before_fit", model=model)

    _set_model_optimization_mode(model)

    effective_fit_iters = _resolve_fit_iters(
        replay_size=len(replay_buffer),
        batch_size=int(config.batch_size),
        fit_iters=config.fit_iters,
    )
    optimizer_steps_planned = int(config.epochs)
    fit_calls = 0
    last_pi = None
    last_loss = None
    LOGGER.info(
        f"[phase1_finalize] start epochs={config.epochs} replay_size={len(replay_buffer)} "
        f"batch_size={config.batch_size} fit_iters={effective_fit_iters} "
        f"optimizer_steps_planned={optimizer_steps_planned}",
    )
    batch_size = min(max(1, config.batch_size), len(replay_buffer))
    for epoch_idx in range(config.epochs):
        pi, loss = model.fit(batch_size=batch_size, iters=effective_fit_iters)
        fit_calls += 1
        last_pi = float(pi)
        last_loss = float(loss)
        if _should_log_epoch_progress(
            epoch_number=epoch_idx + 1,
            total_epochs=int(config.epochs),
            progress_every=config.progress_every,
        ):
            LOGGER.info(
                f"[phase1_finalize] epoch={epoch_idx + 1}/{config.epochs} "
                f"optimizer_steps={fit_calls}/{optimizer_steps_planned} "
                f"fit_iters={effective_fit_iters} last_pi={last_pi:.4f} "
                f"last_loss={last_loss:.6f}",
            )

    snapshot_path = snapshot_store.save_snapshot(
        model=model,
        replay_buffer=replay_buffer,
        step_index=max_step_index,
        t=0.0,
        reason="phase1-finalize",
        memorize_count=memorized_count,
    )
    if hasattr(replay_buffer, "clear"):
        replay_buffer.clear(memorized_count)

    metadata = {
        "uuid": run_uuid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "loaded_snapshot": str(loaded_snapshot_path) if loaded_snapshot_path is not None else None,
        "source_runs": [str(path) for path in source_runs],
        "source_run_count": len(source_runs),
        "transitions_loaded": transitions_loaded,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "fit_iters": effective_fit_iters,
        "fit_iters_requested": config.fit_iters,
        "prompt_token_window": config.prompt_token_window,
        "progress_every": config.progress_every,
        "fit_calls": fit_calls,
        "memorized_count": memorized_count,
        "snapshot_path": str(snapshot_path),
        "max_source_step_index": max_step_index,
        "snapshot_keep": config.snapshot_keep,
    }
    snapshot_store.write_run_metadata(metadata)
    return Phase1FinalizeSummary(
        run_uuid=run_uuid,
        model_run_dir=model_run_dir,
        snapshot_path=snapshot_path,
        source_run_count=len(source_runs),
        transitions_loaded=transitions_loaded,
        fit_calls=fit_calls,
        memorized_count=memorized_count,
    )


def resolve_data_runs(data_runs: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in data_runs:
        run_dir = Path(path).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Data run directory not found: {run_dir}")
        observations_path = run_dir / "observations.jsonl"
        if not observations_path.is_file():
            raise FileNotFoundError(f"Missing observations file: {observations_path}")
        resolved.append(run_dir)
    if not resolved:
        raise ValueError("At least one data run directory must be provided.")
    return resolved


def count_step_rows(run_dir: Path) -> int:
    count = 0
    observations_path = Path(run_dir) / "observations.jsonl"
    with observations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload.get("source", "")).lower() == "step":
                count += 1
    return count


def iter_transitions_from_run(run_dir: Path) -> Iterable[Transition]:
    observations_path = Path(run_dir) / "observations.jsonl"
    prev_observation: ModelObservation | None = None
    with observations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {observations_path}:{line_number}") from exc
            observation = observation_from_row(run_dir, row)
            source = str(row.get("source", "")).lower()
            if prev_observation is None:
                prev_observation = observation
                continue
            if source != "step":
                prev_observation = observation
                continue

            action = row.get("action") or {}
            executed_action_vector = _optional_action_vector(action.get("executed_action_vector"))
            reward = row.get("reward")
            if executed_action_vector is None or reward is None:
                prev_observation = observation
                continue
            transition = Transition(
                observation=prev_observation,
                executed_action_vector=executed_action_vector,
                reward=float(reward),
                next_observation=observation,
                done=bool(row.get("done", False)),
                target_text=build_target_text(action=action, observation_t=float(prev_observation.t)),
                logp_beta_sum=_optional_float(action.get("logp_beta_sum")),
                target_action_name=_optional_str(action.get("agentic_action_name")),
                agentic_action_vector=_optional_action_vector(action.get("agentic_action_vector")),
                actor_action_vector=_optional_action_vector(action.get("actor_action_vector")),
                metadata={
                    "source_run_dir": str(run_dir),
                    "source_line": line_number,
                    "source_row_metadata": row.get("metadata", {}),
                },
            )
            yield transition
            prev_observation = observation


def observation_from_row(run_dir: Path, row: dict[str, Any]) -> ModelObservation:
    image_rel_path = row.get("image_path")
    if not image_rel_path:
        raise ValueError("Observation row is missing `image_path`.")
    image_path = Path(run_dir) / str(image_rel_path)
    image_rgb = load_image_array(image_path)
    messages = row.get("messages", [])
    metadata = row.get("metadata", {})
    if not isinstance(messages, list):
        messages = []
    if not isinstance(metadata, dict):
        metadata = {}
    return ModelObservation(
        image_rgb=image_rgb,
        messages=messages,
        t=float(row.get("t", 0.0)),
        last_reward=float(row.get("last_reward", 0.0)),
        done=bool(row.get("done", False)),
        step_index=int(row.get("step_index", 0)),
        metadata=metadata,
    )


def load_image_array(path: Path) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for loading stored observations") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing image blob: {path}")
    payload = np.load(path, allow_pickle=False)
    if "image" not in payload:
        raise ValueError(f"Missing `image` array in blob: {path}")
    return payload["image"]


def build_target_text(*, action: dict[str, Any], observation_t: float) -> str | None:
    generated_text = str(action.get("generated_text", "") or "").strip()
    action_name = _optional_str(action.get("agentic_action_name"))
    if action_name and observation_t < 1.0:
        return json.dumps(
            {"action": action_name, "say": generated_text},
            ensure_ascii=True,
        )
    if generated_text:
        return generated_text
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _optional_action_vector(value: Any) -> dict[str, float] | None:
    if value is None or not isinstance(value, dict):
        return None
    vector = {}
    for key in ("pan", "tilt", "turn", "drive"):
        vector[key] = float(value.get(key, 0.0))
    return vector


def _validate_config(config: Phase1FinalizeConfig) -> None:
    if int(config.epochs) < 1:
        raise ValueError(f"--epochs must be >= 1, got {config.epochs}")
    if int(config.batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {config.batch_size}")
    if config.fit_iters is not None and int(config.fit_iters) < 1:
        raise ValueError(f"--fit-iters must be >= 1, got {config.fit_iters}")
    if int(config.prompt_token_window) < 0:
        raise ValueError(
            "--prompt-token-window must be >= 0 (0 disables token truncation), "
            f"got {config.prompt_token_window}"
        )
    if int(config.progress_every) < 0:
        raise ValueError(f"--progress-every must be >= 0, got {config.progress_every}")
    if int(config.snapshot_keep) < 1:
        raise ValueError(f"--snapshot-keep must be >= 1, got {config.snapshot_keep}")


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(message)s")
        return None
    root_logger.setLevel(level)
    return None


def _set_model_optimization_mode(model: Any) -> None:
    set_optimization_mode = getattr(model, "set_optimization_mode", None)
    if callable(set_optimization_mode):
        set_optimization_mode()
        return None
    if hasattr(model, "train"):
        model.train()
    return None


def _resolve_fit_iters(*, replay_size: int, batch_size: int, fit_iters: int | None) -> int:
    if fit_iters is not None:
        return max(1, int(fit_iters))
    safe_batch_size = max(1, int(batch_size))
    return max(1, int(math.ceil(int(replay_size) / float(safe_batch_size))))


def _should_log_epoch_progress(epoch_number: int, total_epochs: int, progress_every: int) -> bool:
    if epoch_number <= 0:
        return False
    if epoch_number >= total_epochs:
        return True
    interval = int(progress_every) if int(progress_every) > 0 else max(1, int(total_epochs) // 10)
    return (epoch_number % interval) == 0


def _log_memory_diagnostics(*, stage: str, model: Any) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return None
    try:
        import torch
    except Exception:
        return None

    def _tensor_mib(value: Any) -> float:
        if value is None or not isinstance(value, torch.Tensor):
            return 0.0
        return float(value.numel() * value.element_size()) / (1024.0 * 1024.0)

    tensor_names = ("ssr_low_rank_matrix", "ssr_residual_diagonal", "ssr_center", "ssr_prev_center")
    for name in tensor_names:
        tensor = getattr(model, name, None)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            LOGGER.debug("[mem] stage=%s tensor=%s present=False", stage, name)
            continue
        LOGGER.debug(
            "[mem] stage=%s tensor=%s present=True shape=%s dtype=%s device=%s size_mib=%.2f",
            stage,
            name,
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            _tensor_mib(tensor),
        )

    if not torch.cuda.is_available():
        return None
    device = getattr(model, "device", torch.device("cuda"))
    try:
        if isinstance(device, torch.device):
            if device.type == "cuda":
                device_index = device.index if device.index is not None else torch.cuda.current_device()
            else:
                device_index = torch.cuda.current_device()
        else:
            device_index = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_index) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(device_index) / (1024.0 * 1024.0)
        max_allocated = torch.cuda.max_memory_allocated(device_index) / (1024.0 * 1024.0)
        max_reserved = torch.cuda.max_memory_reserved(device_index) / (1024.0 * 1024.0)
        LOGGER.debug(
            "[mem] stage=%s cuda_device=%d allocated_mib=%.2f reserved_mib=%.2f "
            "max_allocated_mib=%.2f max_reserved_mib=%.2f",
            stage,
            int(device_index),
            float(allocated),
            float(reserved),
            float(max_allocated),
            float(max_reserved),
        )
    except Exception:
        LOGGER.debug("[mem] stage=%s cuda_stats=unavailable", stage, exc_info=True)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
