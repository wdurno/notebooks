"""Diagnose local microphone, STT, and TTS wiring."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from picar_kl.speech.config import SpeechConfig
from picar_kl.speech.errors import MissingDependencyError, SpeechError
from picar_kl.speech.service import SpeechService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record one utterance and print speech I/O diagnostics.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--playback", action="store_true", help="Synthesize and play the transcript when non-empty.")
    parser.add_argument("--keep-wav", type=Path, default=None, help="Copy the recorded wav to this path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service = SpeechService(SpeechConfig(model_dir=args.model_dir))
    print("Detected audio devices:")
    for idx, device in enumerate(_query_devices()):
        print(f"  [{idx}] {device}")
    print("Press Enter, speak clearly, then press Enter again to stop.", flush=True)
    input()

    try:
        recording = service.audio.record_until_enter()
        stats = _wav_stats(recording.wav_path)
        print(
            "recording "
            f"path={recording.wav_path} duration={recording.duration_seconds:.3f}s "
            f"rms={stats['rms']:.6f} peak={stats['peak']:.6f}"
        )
        if args.keep_wav is not None:
            args.keep_wav.parent.mkdir(parents=True, exist_ok=True)
            args.keep_wav.write_bytes(recording.wav_path.read_bytes())
            print(f"saved wav={args.keep_wav}")
        transcription = service.stt.transcribe_file(recording.wav_path)
        print(
            "transcription "
            f"text={transcription.text!r} language={transcription.language!r} "
            f"language_probability={transcription.language_probability!r} "
            f"duration={transcription.duration_seconds!r} segments={transcription.segments!r}"
        )
        if args.playback:
            if transcription.text.strip():
                service.speak(transcription.text)
                print("playback=ok")
            else:
                print("playback=skipped-empty-transcript")
    except MissingDependencyError as exc:
        print(f"missing_dependency={exc}")
        return 2
    except SpeechError as exc:
        print(f"speech_error={exc}")
        return 1
    return 0


def _query_devices() -> list[str]:
    try:
        import sounddevice as sd
    except Exception as exc:
        return [f"sounddevice unavailable: {exc}"]
    return [str(device) for device in sd.query_devices()]


def _wav_stats(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return {"rms": 0.0, "peak": 0.0}
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
        "peak": float(np.max(np.abs(audio))),
    }


if __name__ == "__main__":
    raise SystemExit(main())
