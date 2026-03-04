# RL Environment Package

This package implements requirement 3 from `demo/spec.md`: a Gym-like
environment loop that coordinates the PiCar control API, the speech subsystem,
the PiCar action model, frozen VLM reward scoring, replay-buffer updates, and
experiment persistence under `demo/data/`.

## Design goals

- Keep robot orchestration out of the model and speech packages.
- Make the runtime loop smooth for a human operator by using continuous speech
  intake instead of step-blocking microphone prompts.
- Keep reward scoring separate from the trainable policy by using a frozen base
  VLM and a prompt registry.
- Persist enough run state to debug or resume experiments after `Ctrl-C`.
- Keep the fast unit test suite independent from the physical robot and heavy
  model downloads.

## Package layout

### `config.py`

Defines the environment configuration:

- PiCar API connection details
- speech-stream runtime parameters
- reward prompt selection
- training cadence and interpolation schedule
- experiment output directories

### `schemas.py`

Defines env-side dataclasses used for:

- queued speech events
- reward prompt specs and reward results
- per-step training summaries
- persisted experiment paths

### `speech_stream.py`

Implements continuous microphone listening for hands-free use.

The runtime model is:

1. read audio chunks from a background input stream
2. split chunks into utterances using lightweight VAD-style silence detection
3. transcribe finished utterances with the existing STT backend
4. push recognized text into a non-blocking queue consumed by the env loop

### `rewarding.py`

Implements the frozen reward scorer and prompt registry.

Responsibilities:

- define built-in reward prompts
- parse strict JSON or numeric reward outputs
- clip rewards to the configured range
- score frames with the untuned base VLM

### `picar_bridge.py`

Adapts the local `picar-v-rl-env` repo into a small client used by this demo.
This wrapper intentionally ignores the legacy `(x, y, r)` metadata returned by
`/img`; the demo relies on the frozen VLM scorer instead.

### `persistence.py`

Creates experiment directories under `demo/data/` and writes:

- `metadata.json`
- JSONL event logs
- replay snapshots
- per-step frame blobs
- model checkpoints and trainable-weight exports

### `picar_env.py`

Defines the main Gym-like environment class.

Core methods:

- `reset()`
- `step()`
- `run_forever()`
- `close()`

## Game loop

One control iteration follows this order:

1. drain any newly recognized operator utterances from the speech queue
2. capture the latest camera frame from the PiCar
3. score that frame with the frozen reward VLM
4. assemble Qwen-style message history with operator text and reward context
5. query the trainable `PiCarActionModel`
6. speak any generated assistant text through TTS
7. send the executed action vector to `apply_vector`
8. capture the post-action frame for the replay transition
9. append the transition to replay and optionally train every `K` steps

The loop runs until the user sends `Ctrl-C`.

The environment also rate-limits `apply_vector` calls so the Raspberry Pi sees
at most one vector command every 0.5 seconds by default.

## Shutdown behavior

`KeyboardInterrupt` is treated as normal termination. On shutdown the env:

1. stops the continuous speech worker
2. flushes step logs and metadata
3. saves a replay snapshot
4. saves model checkpoints and trainable weights
5. returns the robot camera to `look-forward`

## Testing strategy

Unit tests focus on pure logic and fake dependencies:

- message assembly
- reward parsing and clipping
- prompt registry selection
- speech queue draining and utterance segmentation
- run-directory creation and checkpoint planning
- env step bookkeeping with fake robot, fake scorer, and fake speaker

Robot control, live audio, and real Qwen execution remain integration concerns.
