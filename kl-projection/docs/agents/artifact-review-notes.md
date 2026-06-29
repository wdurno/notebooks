# Artifact Review Notes

## Abstract

These notes track review of existing assets referenced by `/docs/agents/existing-artifacts.md`.
The review is motivated by the current experiment in `/README.md`: a two-speed VLM/LSTM robotics hierarchy, KL-projection initialization, and phase 3 online finalization.
Old assets should be mined for useful mechanics, data, tests, and interfaces, but stale architecture should not be preserved accidentally.

## Review Plan

1. Intent mapping.
   Map each old asset to the new experimental phases.
   Classify pieces as reuse, adapt, or discard.

2. Phase 1 data and data generation.
   Keep phase 1 alive as a first-class capability.
   Review old generated data, its persistence format, and the old VLM-only control path.
   The new project must support both importing old data and generating more phase 1 data.

3. Action and control interfaces.
   Review the old action names, one-hot/action-vector conversions, continuous control vector, and PiCar `/apply_vector` contract.
   These are central to KL targets and LSTM outputs.

4. Robot runtime split.
   Review the Raspberry Pi Flask server and GPU-server client.
   Preserve a light robot install and avoid importing heavy AI dependencies on the Raspberry Pi.

5. Model and training mechanics.
   Review the old Qwen/LoRA, replay buffer, Actor-Critic, SSR, EWC, and Lanczos mechanics.
   Separate reusable phase 3 machinery from the old single-VLM `t` interpolation design.

6. Speech and coherency.
   Review STT/TTS, language history, and VLM-based reward/coherency scoring.
   Preserve conversational robotics where practical.

7. Tests.
   Classify old tests into copy-now, adapt-later, and obsolete.
   Bootstrap the new test suite around stable interfaces first.

8. Math notebook.
   Extract implementation-facing guidance from the Amari-Chentsov notebook.
   Focus on single-observation batches, optimal `pi`, and forgetting diagnostics.

## Review Principles

Phase 1 remains active.
Do not treat old data as enough by default.
The codebase should let the experimenter collect more VLM-only data if phase 2 needs it.

The old demo had a different model architecture.
It moved from agentic VLM actions to value-head actions using a `t` interpolation.
The new experiment uses a sparse VLM conditioning head and a per-step LSTM.
Keep useful mechanics, but avoid letting the old `t` architecture shape the new implementation.

Raw robot observations belong in ignored artifact storage.
Distilled results belong in `/experiments/runs/` only when they are small and under the 1MB per-run limit.

## Findings

### Phase 1 Data

The old demo has explicit phase 1 runs under:

- `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data/phase1/f0725eab-fe64-479e-8f88-995d38c24dba`
- `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data/phase1/04f639c9-478a-4cb2-ab4d-086813915793`

The second run has 382 completed steps and is likely the most useful initial dataset.
The format is:

```text
run_dir/
  run_meta.json
  observations.jsonl
  images/
    step_000000.npz
    step_000001.npz
    ...
```

Each observation row stores:

- `source`, usually `reset` or `step`.
- `step_index`.
- `messages`.
- `user_texts`.
- `reward`.
- `last_reward`.
- `action`.
- `action_receipt`.
- `training`.
- `image_path`.

The `action` object stores:

- `agentic_action_name`.
- `agentic_action_vector`.
- `actor_action_vector`.
- `executed_action_vector`.
- `critic_value`.
- `generated_text`.
- `logp_beta_sum`.

For the new experiment, phase 1 should preserve the useful parts of this schema but does not need to preserve old `t` semantics.
The new phase 1 writer should log VLM action distributions directly, not just action names or executed vectors.
For old data compatibility, one-hot distributions can be reconstructed from `agentic_action_name`.

### Data Loading Implications

The old `phase1_finalize.py` already reconstructs transitions from `observations.jsonl`.
That code is a good reference for a new dataset loader.
The new loader should likely expose a phase 2 sample containing:

- prior image or image path.
- next image or image path when useful.
- message/history context.
- user text.
- VLM action distribution.
- executed action vector.
- generated robot text.
- raw latency events when available in new data.

Old data does not appear to have the new latency metric explicitly.
It may have timestamps sufficient for rough step elapsed time, but fresh phase 1 generation should record raw wall-clock latency events intentionally.

### Phase 1 Generation

Keep phase 1 alive.
The old `PiCarGymEnv` shows the needed live loop:

1. Drain speech.
2. Capture current frame.
3. Score/reward with frozen VLM.
4. Build VLM messages.
5. Ask model for action and text.
6. Speak generated text.
7. Apply vector to robot.
8. Capture next frame.
9. Persist observation and action records.

The new phase 1 loop should use VLM-only action generation and persistence without LSTM involvement.
It should write compatible records so phase 2 can train from old and new runs together.

### Action Space

The old action space is a strong reuse candidate.
It defines eight actions:

1. `drive-left`
2. `drive-right`
3. `drive-forward`
4. `drive-backward`
5. `look-left`
6. `look-right`
7. `look-up`
8. `look-forward`

It also defines the continuous robot control vector keys:

```text
pan, tilt, turn, drive
```

Bounds are:

- `pan`: `[-1, 1]`
- `tilt`: `[0, 1]`
- `turn`: `[-1, 1]`
- `drive`: `[-1, 1]`

For the new experiment, the action distribution should live over the eight action names.
The distribution-to-control-vector mapping can use the old weighted sum logic.
The robot API should receive only the final continuous vector.

The old `t` interpolation helpers are historically useful but should not define the new architecture.
The new model hierarchy should instead combine VLM conditioning with LSTM per-step action distributions.

### Robot Control

The old PiCar API supports both discrete action routes and `/apply_vector`.
The `/apply_vector` route is the best match for the new experiment because both the VLM distribution and LSTM distribution can deterministically map into a continuous action vector.

The GPU-server client should keep retry and command-spacing behavior.
The Raspberry Pi server should stay light and should not import AI dependencies.

The PiCar API repo is compact and should be integrated mostly as robot-side code plus a server-side client.
The robot API imports hardware-facing `car.py` from `api_main.py` at module import time.
That is acceptable on the Raspberry Pi, but the new package should isolate this path so normal server imports and unit tests do not require `picar`, camera hardware, or servo libraries.

Prefer this split:

- robot-side Flask app and hardware adapter: light install, Raspberry Pi only.
- server-side PiCar client: requests/Pillow/numpy only, no AI imports.
- AI orchestration: server only.

The old `run_api.py` simply imports `car_env.api_main` and runs Flask.
The new package can use a console script or a small `scripts/` wrapper, but the same minimal shape is fine.

### Tests

The old unit tests for action space and PiCar bridge are good copy/adapt candidates.
They cover:

- one-hot action round trips.
- action-to-vector conversion.
- vector interpolation/clamping behavior.
- robust image response parsing.
- HTTP timeout retry behavior.
- command-rate spacing.

These should be among the first new unit tests after scaffolding.

### Model and Training Mechanics

The old model path has useful mechanics but stale architecture.
Reusable pieces:

- fake backbone patterns for fast tests.
- model/replay-buffer schemas.
- snapshot/persistence ideas.
- Qwen model store and manifest conventions.
- QLoRA load path, with server-only heavy dependencies.
- Actor-Critic loss ideas for phase 3.

Stale pieces:

- `t` interpolation from agentic action to actor action.
- single VLM actor/value-head architecture as the main policy.
- phase names `init`, `tune`, `retask` as the primary new experiment structure.

The new model stack should be organized around:

- VLM action distribution for phase 1.
- VLM conditioning head emitted every `K` steps.
- LSTM action distribution emitted every step.
- KL projection from VLM distributions to LSTM distributions.
- optional phase 3 QLoRA, EWC, replay, and Actor-Critic loss.

### SSR, EWC, and Lanczos

`src/core/lanczos.py` and `src/core/ssr_agent.py` are important phase 3 candidates.
They implement low-rank Fisher/EWC-like machinery with a residual diagonal.
The old code assumes concrete subclasses define a mean-scaled `loss`.
That assumption is compatible with future phase 3 work.

The online SSR extension is especially relevant.
It caches a current gradient, updates sufficient statistics with an EMA-like gain, and exposes `optimal_pi`.
This aligns with the math notebook's single-observation batch mechanics and forgetting diagnostic.

Adapt with care:

- old SSR classes are coupled to model classes through inheritance.
- phase 3 should probably start with composition or a narrow adapter if possible.
- EWC must be disableable with `lambda = 0`.
- online consolidation may be expensive, so implementation should keep rank and update cadence configurable.

### Math Notebook

The Amari-Chentsov notebook is compact and mostly conceptual.
Implementation-facing content:

- single-observation batches justify online sufficient statistic updates with low activation memory.
- `pi` can be interpreted as a control/gain, not only a sample proportion.
- locally optimal `pi` can demand too much forgetting.
- high `pi` should be treated as an overwhelm/forgetting diagnostic, not automatically as permission to overwrite memory.

This supports phase 3 design, not phase 2 KL projection.

### Model and Speech Manifests

The old demo stores small JSON manifests under `model/manifests`.
These are good candidates for the new repo's `/artifacts/manifests/tracked` directory.

Existing manifest concepts:

- VLM default: `qwen2.5-vl-3b`.
- VLM repo: `Qwen/Qwen2.5-VL-3B-Instruct`.
- STT default: `whisper-large-v3-turbo-int8`.
- STT repo: `Zoont/faster-whisper-large-v3-turbo-int8-ct2`.
- TTS default voice: `en_US-lessac-medium`.
- TTS repo: `rhasspy/piper-voices`.

Actual model files should remain ignored under artifact/model storage.
The manifests are not user-facing runtime configs, but they are safe-to-version changing defaults.

The old model-store code downloads into a cache directory, validates required files, then copies into finalized model directories.
That pattern is worth preserving.
In the new repo, model-store code should resolve paths from project config or environment variables rather than hard-coded old `demo/model` roots.
