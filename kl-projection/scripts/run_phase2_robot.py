"""Run live Phase 2 robot execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from picar_kl.context import ContextConfig
from picar_kl.models.visual import QwenVisualTokenEncoder
from picar_kl.phase1.run import NoSpeaker, NoSpeechSource
from picar_kl.phase2.live import Phase2LiveRunConfig, run_phase2_live
from picar_kl.phase2.runtime import load_phase2_policy
from picar_kl.reward import ConstantRewardScorer, FrozenVLMRewardScorer, RewardConfig
from picar_kl.robot.client import PiCarClient, PiCarClientConfig
from picar_kl.speech.adapters import SpeechServiceSpeaker, StreamingSpeechSource
from picar_kl.speech.config import SpeechConfig, SpeechStreamConfig
from picar_kl.speech.service import SpeechService
from picar_kl.vlm.qwen import QwenPhase1Config, QwenPhase1Controller
from picar_kl.vlm.runtime import QwenRuntime, QwenRuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 live robot execution.")
    parser.add_argument("--picar-host", required=True, help="Robot host:port.")
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/data/phase2-live"))
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run. Default: run until Ctrl-C.")
    parser.add_argument("--task-prompt", default="find the red ball")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/models/phase2"))
    parser.add_argument("--fit-id", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--k", type=int, default=None, help="Must equal fitted prediction_steps when provided.")
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--vlm-manifest",
        type=Path,
        default=Path("artifacts/manifests/tracked/models/vlm_models.json"),
    )
    parser.add_argument("--vlm-model-name", default="qwen2.5-vl-3b")
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--visual-output-dtype", default="float16")
    parser.add_argument("--history-window", type=int, default=180)
    parser.add_argument("--prompt-token-window", type=int, default=8000)
    parser.add_argument("--enable-reward", dest="reward", action="store_true", default=True)
    parser.add_argument("--no-reward", dest="reward", action="store_false")
    parser.add_argument("--reward-prompt-id", default="reward_prompt_1")
    parser.add_argument("--enable-speech-input", dest="speech_input", action="store_true", default=True)
    parser.add_argument("--no-speech-input", dest="speech_input", action="store_false")
    parser.add_argument("--enable-speech-output", dest="speech_output", action="store_true", default=True)
    parser.add_argument("--no-speech-output", dest="speech_output", action="store_false")
    parser.add_argument("--allow-speech-downloads", action="store_true")
    parser.add_argument("--speech-amplitude-threshold", type=float, default=0.015)
    parser.add_argument("--speech-silence-seconds", type=float, default=0.8)
    parser.add_argument("--speech-min-seconds", type=float, default=0.25)
    parser.add_argument("--speech-callback-queue-size", type=int, default=256)
    parser.add_argument("--speech-utterance-queue-size", type=int, default=32)
    parser.add_argument("--x-resize", type=int, default=None)
    parser.add_argument("--y-resize", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    loaded_policy = load_phase2_policy(
        checkpoint_root=args.checkpoint_root,
        fit_id=args.fit_id,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    robot = PiCarClient(
        PiCarClientConfig(
            host=args.picar_host,
            x_resize=args.x_resize,
            y_resize=args.y_resize,
        )
    )
    runtime_config = QwenRuntimeConfig(
        model_name=args.vlm_model_name,
        model_root=args.model_root,
        manifest_path=args.vlm_manifest,
        allow_downloads=bool(args.allow_model_downloads),
    )
    runtime = QwenRuntime.from_config(runtime_config)
    phase1_config = QwenPhase1Config(
        model_name=args.vlm_model_name,
        model_root=args.model_root,
        manifest_path=args.vlm_manifest,
        allow_downloads=bool(args.allow_model_downloads),
    )
    vlm = QwenPhase1Controller(
        model=runtime.model,
        processor=runtime.processor,
        config=phase1_config,
        runtime=runtime,
    )
    visual_encoder = QwenVisualTokenEncoder(runtime, output_dtype=args.visual_output_dtype)
    reward_scorer = (
        FrozenVLMRewardScorer(
            RewardConfig(
                prompt_id=args.reward_prompt_id,
                allow_downloads=bool(args.allow_model_downloads),
            ),
            runtime=runtime,
        )
        if args.reward
        else ConstantRewardScorer(task_text=args.task_prompt)
    )

    speech_config = SpeechConfig(
        model_dir=args.model_root,
        allow_downloads=bool(args.allow_speech_downloads),
    )
    speech_service = SpeechService(speech_config) if args.speech_input or args.speech_output else None
    speech_source = NoSpeechSource()
    if args.speech_input:
        speech_source = StreamingSpeechSource.from_service(
            speech_service,
            SpeechStreamConfig(
                sample_rate=speech_config.audio.sample_rate,
                channels=speech_config.audio.channels,
                dtype=speech_config.audio.dtype,
                blocksize=speech_config.audio.blocksize,
                amplitude_threshold=float(args.speech_amplitude_threshold),
                silence_seconds=float(args.speech_silence_seconds),
                min_speech_seconds=float(args.speech_min_seconds),
                callback_queue_size=max(1, int(args.speech_callback_queue_size)),
                utterance_queue_size=max(1, int(args.speech_utterance_queue_size)),
            ),
        )
    speaker = SpeechServiceSpeaker(speech_service) if args.speech_output else NoSpeaker()

    if hasattr(speech_source, "start"):
        speech_source.start()
    try:
        summary = run_phase2_live(
            Phase2LiveRunConfig(
                data_root=args.data_root,
                task_prompt=args.task_prompt,
                checkpoint_root=args.checkpoint_root,
                fit_id=args.fit_id,
                checkpoint_path=args.checkpoint_path,
                device=args.device,
                max_steps=None if args.steps is None else max(1, int(args.steps)),
                k=args.k,
                context=ContextConfig(
                    history_window=int(args.history_window),
                    prompt_token_window=int(args.prompt_token_window),
                ),
                metadata={
                    "runner": "scripts/run_phase2_robot.py",
                    "reward_enabled": bool(args.reward),
                },
            ),
            robot=robot,
            vlm=vlm,
            visual_encoder=visual_encoder,
            loaded_policy=loaded_policy,
            speech_source=speech_source,
            speaker=speaker,
            reward_scorer=reward_scorer,
            prompt_tokenizer=runtime,
        )
    finally:
        if hasattr(speech_source, "stop"):
            speech_source.stop()
    print(
        f"[phase2-live] {summary.status} uuid={summary.run_uuid} "
        f"fit_id={summary.fit_id} steps={summary.steps_completed} data={summary.run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
