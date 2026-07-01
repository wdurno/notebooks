"""CLI for phase 1 smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from picar_kl.phase1.run import FixedActionVLMController, NoSpeaker, NoSpeechSource, Phase1RunConfig, run_phase1
from picar_kl.robot.client import PiCarClient, PiCarClientConfig
from picar_kl.speech.adapters import SpeechServiceSpeaker, StreamingSpeechSource
from picar_kl.speech.config import SpeechConfig, SpeechStreamConfig
from picar_kl.speech.service import SpeechService
from picar_kl.vlm.qwen import QwenPhase1Config, QwenPhase1Controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run phase 1 VLM-only data generation.")
    parser.add_argument("--picar-host", required=True, help="Robot host:port.")
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/data/phase1"))
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run. Default: run until Ctrl-C.")
    parser.add_argument("--task-prompt", default="find the red ball")
    parser.add_argument("--smoke-fixed-action", default=None, help="Use a fixed action instead of Qwen.")
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--vlm-manifest",
        type=Path,
        default=Path("artifacts/manifests/tracked/models/vlm_models.json"),
    )
    parser.add_argument("--vlm-model-name", default="qwen2.5-vl-3b")
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--enable-speech-input", dest="speech_input", action="store_true", default=True)
    parser.add_argument("--no-speech-input", dest="speech_input", action="store_false")
    parser.add_argument("--enable-speech-output", dest="speech_output", action="store_true", default=True)
    parser.add_argument("--no-speech-output", dest="speech_output", action="store_false")
    parser.add_argument("--allow-speech-downloads", action="store_true")
    parser.add_argument("--speech-amplitude-threshold", type=float, default=0.015)
    parser.add_argument("--speech-silence-seconds", type=float, default=0.8)
    parser.add_argument("--speech-min-seconds", type=float, default=0.25)
    parser.add_argument("--x-resize", type=int, default=None)
    parser.add_argument("--y-resize", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    robot = PiCarClient(
        PiCarClientConfig(
            host=args.picar_host,
            x_resize=args.x_resize,
            y_resize=args.y_resize,
        )
    )
    if args.smoke_fixed_action:
        vlm = FixedActionVLMController(args.smoke_fixed_action)
        controller_name = "fixed-action-cli"
    else:
        vlm = QwenPhase1Controller.from_config(
            QwenPhase1Config(
                model_name=args.vlm_model_name,
                model_root=args.model_root,
                manifest_path=args.vlm_manifest,
                allow_downloads=bool(args.allow_model_downloads),
            )
        )
        controller_name = "qwen2.5-vl"

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
            ),
        )
    speaker = SpeechServiceSpeaker(speech_service) if args.speech_output else NoSpeaker()

    if hasattr(speech_source, "start"):
        speech_source.start()
    try:
        summary = run_phase1(
            Phase1RunConfig(
                data_root=args.data_root,
                task_prompt=args.task_prompt,
                max_steps=None if args.steps is None else max(1, int(args.steps)),
                metadata={"controller": controller_name},
            ),
            robot=robot,
            vlm=vlm,
            speech_source=speech_source,
            speaker=speaker,
        )
    finally:
        if hasattr(speech_source, "stop"):
            speech_source.stop()
    print(f"[phase1] {summary.status} uuid={summary.run_uuid} steps={summary.steps_completed} data={summary.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
