# Model Storage Layout

This directory stores large model artifacts that should not be committed to git.
It is organized by subsystem so speech, vision-language, and future model families
can coexist without filename collisions or ambiguous ownership.

## Design goals

- Keep downloaded model weights separate from source code.
- Make it obvious which files belong to STT, TTS, or VLM workloads.
- Leave room for multiple model variants and fine-tuned outputs.
- Support automatic downloads on first use.
- Keep temporary download state separate from finalized artifacts.

## Directory structure

```text
model/
  stt/
    whisper-large-v3-turbo-int8/
  tts/
    piper/
  vlm/
    qwen2.5-vl-3b/
  cache/
    downloads/
  manifests/
```

## Subdirectories

### `stt/`

Speech-to-text model storage.

Each STT model should live in its own directory. For example,
`stt/whisper-large-v3-turbo-int8/` should contain the files required by the
selected Whisper inference backend, such as quantized weights, tokenizer assets,
and backend-specific config files.

### `tts/`

Text-to-speech model storage.

`tts/piper/` is the root for Piper voices. Each voice should live in its own
subdirectory so `.onnx` voice files and their matching config files remain grouped
together.

Example:

```text
tts/
  piper/
    en_US-lessac-medium/
      en_US-lessac-medium.onnx
      en_US-lessac-medium.onnx.json
```

### `vlm/`

Vision-language model storage.

`vlm/qwen2.5-vl-3b/` is intended to hold both the base model and any fine-tuning
outputs. A practical convention is:

```text
vlm/
  qwen2.5-vl-3b/
    base/
    finetunes/
```

Recommended usage:

- `base/`: immutable upstream model download.
- `finetunes/`: experiment outputs, checkpoints, adapters, and merged exports.

This separation prevents accidental training writes into the canonical base model
directory and makes cleanup simpler.

### `cache/`

Temporary storage used during downloads or model preparation.

`cache/downloads/` is the right place for partial downloads, staging files, or
backend caches that should not be treated as finalized model artifacts.

The intended pattern is:

1. Download into `cache/downloads/`.
2. Validate the result if checksums or expected filenames are known.
3. Move the finalized files into the target model directory.

### `manifests/`

Project-controlled metadata about expected models.

This directory is intended for small files you maintain, such as:

- expected filenames
- model identifiers
- checksums
- source URLs
- default voice or model selections

Vendor model files do not belong here.

## Recommended conventions

- Do not commit downloaded model weights, checkpoints, adapters, or caches.
- Keep one directory per model or voice to avoid collisions.
- Treat base downloads as read-mostly.
- Store fine-tuning outputs in clearly named experiment directories.
- Resolve paths relative to this `model/` directory or through an environment
  variable override such as `DEMO_MODEL_DIR`.

## Example future layout

```text
model/
  stt/
    whisper-large-v3-turbo-int8/
      ...
  tts/
    piper/
      en_US-lessac-medium/
        ...
  vlm/
    qwen2.5-vl-3b/
      base/
        ...
      finetunes/
        2026-02-28-captioning-v1/
          metadata.json
          checkpoints/
          exports/
  cache/
    downloads/
  manifests/
    stt_models.json
    tts_voices.json
    vlm_models.json
```
