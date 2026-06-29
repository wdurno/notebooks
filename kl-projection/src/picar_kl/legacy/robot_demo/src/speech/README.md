# Speech Subsystem

This package implements requirement 1 from `demo/spec.md`: speech-to-text,
text-to-speech, and local microphone/speaker integration that can be reused by
the rest of the demo.

## Design goals

- Keep speech code isolated from robot-control and model-training code.
- Prefer CUDA when available, but run correctly on CPU-only systems.
- Store large speech artifacts under `demo/model/`, not in source code.
- Make live audio interaction available to the user, not just file-based model APIs.
- Keep import-time behavior lightweight so the package can be configured before
  optional dependencies are installed.

## Module layout

### `config.py`

Defines the main configuration dataclasses:

- `SpeechConfig`: top-level speech configuration
- `STTConfig`: faster-whisper runtime settings
- `TTSConfig`: Piper voice and synthesis settings
- `AudioIOConfig`: microphone and speaker settings

This module also defines default path resolution for `demo/` and `demo/model/`
and the CUDA-or-CPU device selection policy.

### `manifests.py`

Loads project-controlled metadata from `demo/model/manifests/`.

These manifests define:

- default STT model identifiers
- default TTS voice identifiers
- expected target directories under `demo/model/`
- source repository identifiers and required files

### `model_store.py`

Implements model asset resolution and download behavior.

Responsibilities:

- resolve speech assets relative to `demo/model/`
- validate whether the required local files already exist
- download missing STT or TTS assets into the correct directories
- keep temporary downloads under `demo/model/cache/`

This module is the boundary between application code and model-file management.

### `audio_io.py`

Implements the thin OS-facing audio layer.

Responsibilities:

- enumerate audio devices
- record microphone audio into temporary WAV files
- play synthesized WAV files through the speaker or headset
- convert floating-point samples into WAV-compatible PCM

This module intentionally hides platform-specific behavior behind a small
interface so higher-level code does not need to care whether it is running on
macOS or Ubuntu.

### `stt.py`

Implements speech-to-text using `faster-whisper`.

Responsibilities:

- load the configured Whisper model from `demo/model/stt/...`
- select CUDA or CPU compute mode
- transcribe WAV files
- return structured transcription metadata for downstream consumers

The output type is `TranscriptionResult`, which is suitable for later reuse in
the RL environment and experiment logging.

### `tts.py`

Implements text-to-speech using Piper voice assets stored under
`demo/model/tts/...`.

Responsibilities:

- resolve the configured Piper voice files
- invoke Piper to synthesize WAV output
- return synthesis metadata including the output path and sample rate

The output type is `SynthesisResult`.

### `service.py`

Provides the high-level orchestration layer used by the rest of the demo.

Responsibilities:

- `listen_once()`: microphone input to transcription
- `speak(text)`: text to speaker playback
- `round_trip()`: microphone input to transcription to synthesized playback

This is the intended integration point for the future RL environment loop.

### `errors.py`

Defines speech-specific error types so callers can distinguish:

- missing optional dependencies
- missing model assets
- audio device failures
- backend synthesis/runtime failures

## Runtime flow

The common round-trip path is:

1. `SpeechService` records microphone audio through `AudioIO`
2. `FasterWhisperSTT` transcribes the recording
3. `PiperTTS` synthesizes the recognized text
4. `AudioIO` plays the generated audio back to the user

## Storage conventions

Speech models follow the storage layout from `demo/model/README.md`:

- STT: `demo/model/stt/whisper-large-v3-turbo-int8/`
- TTS: `demo/model/tts/piper/en_US-lessac-medium/`

Project-maintained metadata lives in:

- `demo/model/manifests/stt_models.json`
- `demo/model/manifests/tts_voices.json`

## Dependency strategy

Most heavy dependencies are imported lazily inside the modules that need them.
That keeps the package importable in partially configured environments, which is
useful during development on machines that do not yet have the audio or speech
stack installed.

## Intended future integration

The speech subsystem is designed to plug into the future environment loop in
`demo/src/env` without exposing backend-specific details. The environment should
depend on `SpeechService` or the lower-level STT/TTS classes, not on raw audio
or model-download logic.
