# Phase 1-2 Build Notes

## Abstract

These notes plan the phase 1 and phase 2 build-out for the `picar_kl` package.
The build target is a working repo that no longer depends on the old source repos for ordinary development.
Existing artifacts should be copied into this repo, refactored into the current experiment's architecture, and documented in the current experiment's terms.

The near-term product is:

1. Generate new phase 1 VLM-only robot data.
2. Reuse copied phase 1 data from old experiments.
3. Train an LSTM with KL projection.
4. Run the robot with sparse VLM control and fast per-step LSTM action generation.
5. Run copied/refactored integration tests before real robot validation.

Phase 3 is intentionally deferred, but phase 3 source material must be preserved.
Do not discard mathematical results, including negative numerical results.

## Build Principles

Use full copies of useful old code and data, then refactor inside this repo.
Do not modify old source repos.
After this build, the old source repos should not be needed for normal phase 1/2 use.
Copy relevant old content as-is before refactoring, but do not copy code or tests that are no longer relevant to this experiment.
Real runtime paths should fail loudly when required dependencies or assets are missing.
Test doubles are allowed only as explicit test doubles, never as silent fallbacks in production paths.

Prefer stable interfaces before model complexity.
The action space, data records, PiCar client, and test harness should be boring and dependable.

Keep phase 1 alive.
Old data is useful but may not be enough.
The experimenter must be able to generate more VLM-only data.

Keep the old `t` interpolation architecture out of the new model shape.
The new architecture is VLM/LSTM hierarchy:

- VLM fires every `K` steps.
- VLM owns language and sparse conditioning.
- LSTM emits action distributions every step.
- KL projection trains LSTM from VLM action distributions.

Use check-ins between build phases.
The scope is large enough that later phases should be refined after earlier results are visible.

Use split install paths.
The GPU server does not need a true package build during normal development; use `pip install -r requirements-server.txt`.
The Raspberry Pi robot should install a built wheel with robot-light dependencies only.
Keep heavyweight training dependencies, including `torch`, off the base robot install.

## Target Repository Shape

Expected root additions:

```text
src/
  picar_kl/
tests/
  unit/
  integration/
config/
experiments/
  configs/
  runs/
  reports/
artifacts/
  data/
  manifests/
    ephemeral/
    tracked/
  models/
notebooks/
docs/
  humans/
scripts/
pyproject.toml
requirements-server.txt
requirements-robot.txt
.gitignore
```

Generated or large paths should be ignored by git:

- `/build/`
- `/dist/`
- `*.egg-info/`
- `/artifacts/data/`
- `/artifacts/models/`
- `/artifacts/manifests/ephemeral/`

Tracked manifests belong under `/artifacts/manifests/tracked/`.

## Build Phase A: Scaffold and Import Baseline

Goal: create the new repo shell and copy source material into safe locations.

Tasks:

1. Create `pyproject.toml` for one package named `picar_kl`.
2. Create requirement files:
   - `requirements-server.txt` for GPU server, VLM, speech, training, and experiment execution.
   - `requirements-robot.txt` for Raspberry Pi Flask server only.
3. Create `.gitignore` matching `AGENTS.md`.
4. Create package directories under `src/picar_kl/`.
5. Copy old PiCar API code into a robot-facing package namespace.
6. Copy old robot-demo server-side source into a staging namespace or refactored modules.
7. Copy old integration tests into `tests/integration/`, then refactor imports.
8. Copy old unit tests that protect reused mechanics into `tests/unit/`, then refactor imports.
9. Copy model manifests into `/artifacts/manifests/tracked/`.
10. Copy full old data into `/artifacts/data/legacy_robot_demo/`.
11. Copy existing phase 1 subset into `/artifacts/data/phase1/` for convenience.
12. Copy math notebook/result material into `/notebooks/` or `/docs/humans/`, rewritten for the current experiment's context.

Check-in A:
Confirm directory structure, package name, copied artifact placement, and gitignore behavior before deeper refactors.

Phase A is only a staged import.
It preserves useful source material and data inside this repo, but it does not make the copied architecture the new architecture.

## Refactor Roadmap

Refactoring happens by extracting narrow modules from staged legacy copies, testing the extracted behavior, then moving runtime paths onto the extracted modules.
The staged legacy tree is reference material during this process.

Phase B extracts stable interfaces:

- action names and action distributions.
- distribution-to-robot-vector conversion.
- phase 1 record schemas and old-data compatibility loaders.
- latency event records.
- PiCar client/server boundaries.

Phase C rewrites phase 1 runtime around those interfaces:

- VLM-only control loop.
- observation storage.
- speech hooks where practical.
- fake VLM and fake robot tests.
- manual robot tests marked with `integration` and `robot`.
- robot wheel build and Flask server entry point.

Phase C.5 pays down avoidable legacy debt before phase 2:

- extract speech from staged legacy code into first-class `picar_kl.speech` modules.
- remove ordinary runtime dependence on `picar_kl.legacy` where practical.
- keep staged legacy code only as provenance, phase 3 reference material, or explicitly marked integration coverage.
- keep data collection last, after cleanup and tests.

Phase C.6 restores shared VLM context and reward machinery:

- extract prompt/history construction into first-class `picar_kl.context` modules.
- extract frozen-VLM reward scoring into first-class `picar_kl.reward` modules.
- add a shared Qwen runtime so action generation and reward scoring never load two Qwen copies.
- restore bounded rolling history, persistent goals, reward/status context, and assistant reply history.
- enforce a hard prompt token budget through shared context code, not phase-specific code.
- raise the default prompt token budget from 512 to 8000.
- record prompt-budget metadata so phase 3 can trust VRAM-related guarantees.

Phase D builds new phase 2 training code:

- visual encoder protocol.
- VLM conditioning head.
- LSTM policy.
- KL projection dataset, loss, and training script.
- old and new phase 1 data loaders.

Phase E builds new phase 2 robot execution:

- sparse VLM refresh every `K` steps.
- per-step LSTM action distributions.
- raw latency logging.
- coherency records for later scoring.
- fake end-to-end integration test before manual robot validation.

The exit criterion is practical: ordinary phase 1 and phase 2 use should import `picar_kl` modules, not staged legacy modules.
Legacy modules may remain temporarily as provenance and phase 3 reference material, but should not be on the main runtime path after their behavior has been extracted.

## Build Phase B: Core Interfaces

Goal: establish stable contracts used by phase 1 and phase 2.

Modules to build or refactor:

```text
picar_kl/actions.py
picar_kl/latency.py
picar_kl/records.py
picar_kl/data/phase1.py
picar_kl/robot/client.py
picar_kl/robot/server.py
```

Core contracts:

1. Action names are the existing eight PiCar actions.
2. VLM and LSTM action distributions are probability vectors over those eight actions.
3. Robot control vectors use `pan`, `tilt`, `turn`, and `drive`.
4. Action distributions deterministically map to robot control vectors by weighted action-vector combination.
5. Fresh records include raw latency events.
6. Old records can be loaded by reconstructing one-hot distributions from `agentic_action_name`.

Tests:

1. Action round trips.
2. Distribution validation.
3. Distribution-to-vector conversion.
4. Old phase 1 row loading.
5. Image `.npz` loading.
6. Latency event serialization.
7. PiCar client retries and command spacing.

Check-in B:
Confirm data schemas and action-distribution semantics before building model training around them.

## Build Phase C: Phase 1 Data Generation

Goal: run the robot with VLM-only control and produce fresh phase 1 data.

Phase 1 loop:

1. Drain speech events.
2. Capture image from robot.
3. Score current state with the frozen/base VLM reward scorer when enabled.
4. Build VLM control messages through shared context machinery.
5. Generate VLM action distribution and robot speech.
6. Convert action distribution to continuous robot vector.
7. Apply vector to robot.
8. Capture next image when useful.
9. Persist record with image path, messages, user text, action distribution, executed vector, generated text, reward/status data, and raw latency events.

Important behavior:

- No LSTM involvement.
- Language stays owned by VLM.
- Action output is constrained to the small action set.
- Existing STT/TTS features should be retained where practical.
- Phase 1 should use shared context/reward/runtime modules once Phase C.6 is complete.
- Data format should be compatible with old data through the phase 1 loader.

Likely modules:

```text
picar_kl/vlm/control.py
picar_kl/speech/
picar_kl/phase1/run.py
picar_kl/io/observation_store.py
```

Scripts:

```text
scripts/run_phase1.py
scripts/build_robot_wheel.py
```

Tests:

- Unit tests with fake VLM and fake robot client.
- Integration test adapted from old PiCar env smoke test.
- Manual robot test remains marked `integration` and `robot`.

Packaging:

- Server path: `pip install -r requirements-server.txt`.
- Robot path: build a wheel and install robot extras on the Raspberry Pi.
- Base package dependencies must remain robot-light.
- `torch` and VLM/training dependencies belong in server requirements or server extras.
- Add a `picar-kl-robot-server` entry point before robot install docs are considered complete.

Check-in C:
Experimenter can generate a small phase 1 run, inspect files, and confirm the record shape before phase 2 training depends on it.

## Build Phase C.5: Legacy Debt Cleanup Before Phase 2

Goal: reduce avoidable dependence on staged legacy code before Phase D training code is built.
Phase D should build on current `picar_kl` interfaces, not copied old repo paths.

Scope:

1. Mark Phase C complete and move residual work into this debt phase.
2. Extract the legacy speech package into first-class `picar_kl.speech` modules.
3. Replace `LegacySpeechConfig`, `LegacySpeechSource`, and `LegacySpeaker` names with neutral current-project names or compatibility aliases.
4. Update Phase 1 CLI speech hooks to use the extracted speech implementation.
5. Update manual speech integration tests to import `picar_kl.speech` directly instead of modifying `sys.path` to import legacy `speech.*`.
6. Keep model assets canonical under `/artifacts/models/`.
7. Keep tracked speech manifests under `/artifacts/manifests/tracked/models/`.
8. Preserve staged legacy speech code only as provenance until ordinary runtime and tests no longer need it.
9. Document any remaining `picar_kl.legacy` runtime imports explicitly.
10. Run the default unit suite and focused speech tests after extraction.
11. Collect more Phase 1 data only after this cleanup is stable.

Extraction assessment:

- Speech extraction is small-to-medium, not huge.
- The legacy speech implementation is roughly 580 lines across config, audio I/O, STT, TTS, model store, manifests, service, and errors.
- It is mostly self-contained around server-side dependencies: `sounddevice`, `faster-whisper`, `piper-tts`, `huggingface_hub`, and `numpy`.
- The main risks are manual audio-device behavior and Piper subprocess behavior, both already covered by manual integration tests.

Acceptance criteria:

1. Phase 1 CLI speech paths do not import `picar_kl.legacy.robot_demo.src.speech`.
2. Manual speech round-trip imports `picar_kl.speech` directly.
3. Unit tests cover speech manifest fallback, model-store missing-asset behavior, queue/empty speech sources, and speaker/source adapters.
4. `~/.venv/bin/python -m pytest -q` passes.
5. Data collection remains possible after cleanup.

Check-in C.5:
Confirm whether any remaining legacy imports are acceptable before starting Phase D.

### Phase C.5 Speech and Processor Extraction Completed

Implemented:

- extracted speech modules into `/src/picar_kl/speech/`:
  - `audio_io.py`
  - `config.py`
  - `errors.py`
  - `manifests.py`
  - `model_store.py`
  - `service.py`
  - `stt.py`
  - `tts.py`
- changed speech defaults so canonical model assets live under `/artifacts/models/`.
- changed speech manifest fallback to tracked manifests under `/artifacts/manifests/tracked/models/`.
- changed Phase 1 CLI speech hooks to use `SpeechConfig`, `SpeechServiceSource`, and `SpeechServiceSpeaker`.
- kept `LegacySpeechConfig`, `LegacySpeechSource`, and `LegacySpeaker` as compatibility aliases only.
- changed manual speech round-trip integration to import `picar_kl.speech` directly.
- changed manual PiCar env integration to import `picar_kl.speech` directly for audio/STT/TTS.
- extracted the Qwen image-only processor fallback into `/src/picar_kl/vlm/processor_loader.py`.
- changed `picar_kl.vlm.qwen` to use the extracted processor fallback instead of legacy model code.

Added tests:

- speech model-store missing STT/TTS asset behavior.
- speech model-store existing TTS asset behavior.
- Qwen image-only processor image-token expansion behavior.

Verification:

- `~/.venv/bin/python -m pytest -q tests/unit/speech tests/unit/vlm` passed with 13 tests.
- `~/.venv/bin/python -m pytest -q` passed with 42 tests.
- manual speech and PiCar env integration tests collect cleanly.

Remaining legacy imports outside staged provenance:

- `/tests/integration/robot_demo/test_picar_env_manual.py` still imports legacy `env.*` and `model.*` modules intentionally, because it is a preserved old-env smoke test.
- `/tests/unit/models/test_legacy_model_store.py` still imports legacy model-store modules intentionally, because it verifies old VLM manifest fallback behavior.

Current ordinary runtime status:

- Phase 1 speech runtime no longer imports `picar_kl.legacy.robot_demo.src.speech`.
- Qwen Phase 1 controller no longer imports the legacy processor fallback.

Continuous speech UX and open-ended collection fix:

- extracted old continuous speech stream UX into `/src/picar_kl/speech/stream.py`.
- Phase 1 now defaults to continuous speech input, TTS output, and unlimited steps.
- use `Ctrl-C` to end an open-ended collection run.
- completed steps are written as the robot runs.
- run metadata records `status`, `steps_completed`, and `ended_at`.
- `--no-speech-input` and `--no-speech-output` disable verbal communication when needed.
- Phase 1 drains completed utterances into `user_texts` each step.
- CLI exposes `--speech-amplitude-threshold`, `--speech-silence-seconds`, and `--speech-min-seconds` for room tuning.

Post-extraction live Phase 1 smoke:

- run `f279d875-45de-483b-b945-505edd3837cc` collected after speech and processor extraction.
- 5 records and 5 image blobs were written.
- phase 1 loader read all records and images successfully.
- action sequence was `look-left`, then repeated `drive-forward`.
- action distributions remained valid one-hot vectors.
- total per-step latency stayed roughly 0.9-1.1 seconds.
- image capture remained fast.
- Qwen decision latency stayed roughly 0.4 seconds after the first step.
- no record-shape regression was observed after extraction.

## Build Phase C.6: Shared Context, Reward, and Qwen Runtime

Goal: restore the useful old VLM environment behavior without trapping it inside `picar_kl.phase1`.
The same machinery must support live phase 1 collection, phase 2 KL-projection dataset construction, and phase 3 fine tuning.

Architectural rule:

- `picar_kl.context` owns structured episode context and token-budget enforcement.
- `picar_kl.reward` owns reward prompts and frozen/base VLM reward scoring.
- `picar_kl.vlm.runtime` owns shared Qwen model and processor loading.
- Phase packages orchestrate these modules but do not own them.

Target modules:

```text
picar_kl/context/
  __init__.py
  budget.py
  config.py
  messages.py
  protocols.py
  state.py
picar_kl/reward/
  __init__.py
  config.py
  prompts.py
  schemas.py
  scoring.py
picar_kl/vlm/runtime.py
```

Context behavior:

1. Store structured episode state, not only rendered prompt strings.
2. Preserve a persistent goal system message derived from the reward task.
3. Add one user/status message per control step containing:
   - operator speech.
   - `step_index`.
   - `last_reward`.
   - `current_reward`.
   - `reward_prompt_id`.
4. Add assistant replies containing selected action and spoken text.
5. Enforce `history_window` before token-budgeting.
6. Enforce `prompt_token_window` after messages are assembled.
7. Preserve mandatory system messages during token truncation.
8. Prefer the most recent messages when context must be dropped.
9. Clip the latest text message when that is the only way to preserve new operator input.
10. Default `history_window` remains 180.
11. Default `prompt_token_window` becomes 8000.

Memory and VRAM guarantee:

- `prompt_token_window` is the hard prompt-memory contract shared by phases.
- `0` or `None` disables token truncation only when explicitly requested.
- live records should include configured window values and measured prompt length when available.
- phase 2 and phase 3 dataset builders should use the same context builder so training prompts respect the same budget collected online.

Reward behavior:

1. Restore the old frozen/base VLM reward scorer.
2. Use the red-ball reward prompt as the default.
3. Parse `{"reward": number}` while retaining the old robust numeric fallback.
4. Clip reward to the configured reward range.
5. Store raw reward text, unclipped reward, clipped reward, prompt id, and task text.
6. Keep command-following reward shaping available as a current-project module, but do not force it into phase 2 training until we intentionally model it.

Shared Qwen runtime behavior:

1. Load Qwen model and processor once.
2. Give the action controller and reward scorer access to the same model/processor.
3. Reward scoring must use base/frozen inference.
4. If adapters are present later, reward scoring enters a context that disables adapters.
5. If adapters are absent, that context is a no-op.
6. Runtime should expose token counting needed by `picar_kl.context`.

Phase 1 migration:

1. Replace `last_generated_text`-only context with `EpisodeContext`.
2. Score current image before building control messages.
3. Feed reward/status/history into the action VLM.
4. Persist reward fields and context metadata in each observation record.
5. Persist enough message/context state for phase 2 and phase 3 reconstruction.
6. Keep open-ended collection, continuous speech, and Ctrl-C-safe metadata behavior.

Phase 2 preparation:

1. Dataset code should call the shared context builder rather than reimplementing prompt reconstruction.
2. Old data without reward/status fields remains loadable.
3. Missing reward/status fields should be explicit, not silently fabricated.
4. KL projection can start from action distributions and images, but context reconstruction must already be available for future phase 3 compatibility.

Tests:

1. Reward text parsing and clipping.
2. Reward scorer uses shared runtime and disables adapters when available.
3. Context history trimming preserves persistent goals.
4. Token-budget truncation keeps mandatory messages and recent context.
5. Latest operator message clipping works under tight token budgets.
6. Phase 1 fake run records rewards and bounded context.
7. Old phase 1 data still loads when reward fields are missing.
8. Shared runtime is not duplicated when both action and reward paths are enabled.

Check-in C.6:
Run a short real robot Phase 1 collection with reward scoring enabled.
Inspect reward text, clipped reward, prompt length metadata, action latency, and qualitative coherency before starting Phase D.

### Phase C.6 Shared Context and Reward Implementation Pass

Implemented:

- added `/src/picar_kl/context/` for shared VLM context configuration, message construction, token-budgeting, and rolling episode state.
- changed the default shared `history_window` to 180.
- changed the default shared `prompt_token_window` to 8000.
- added `/src/picar_kl/reward/` for reward schemas, prompt registry, robust reward parsing, constant test scorer, and frozen/base VLM reward scoring.
- added `/src/picar_kl/vlm/runtime.py` so action generation and reward scoring can share one loaded Qwen model and processor.
- changed `QwenPhase1Controller` to consume the shared Qwen runtime.
- changed Phase 1 runtime to:
  - score the current frame before action generation.
  - build prompts through `EpisodeContext`.
  - persist bounded messages, reward result metadata, context-budget metadata, and reward latency.
  - keep continuous speech, open-ended runs, and Ctrl-C-safe metadata behavior.
- changed Phase 1 CLI to:
  - expose `--history-window`, defaulting to 180.
  - expose `--prompt-token-window`, defaulting to 8000.
  - enable frozen/base VLM reward scoring by default for real Qwen runs.
  - use one shared Qwen runtime for both action generation and reward scoring.
  - retain fixed-action smoke runs without loading Qwen.

Added tests:

- context history trimming and persistent-goal preservation.
- context token-budget truncation and latest-message clipping.
- reward parsing, clipping, and shared-runtime reward scoring.
- Qwen runtime prompt-token counting and adapter-disabling context.
- Phase 1 record expectations for reward and context metadata.

Verification:

- `~/.venv/bin/python -m pytest -q tests/unit/context tests/unit/reward tests/unit/vlm tests/unit/phase1/test_run.py` passed with 14 tests.
- `~/.venv/bin/python -m pytest -q` passed with 53 tests.

Remaining check-in:

- run a short real robot Phase 1 collection with reward scoring enabled.
- inspect reward raw text, clipped reward, prompt token metadata, action latency, and qualitative coherency.


## Build Phase D: Phase 2 Offline KL Projection

Goal: train an LSTM from old and new phase 1 data using mandatory Qwen visual-token encodings.
This phase builds the offline KL-projection pipeline, not the robot runtime.

Design decision:

- Do not mean-pool visual vectors as the primary representation.
- Preserve the per-image visual-token sequence produced by the Qwen visual stack or projection path.
- Reduce token width with a learned projection, not by destroying the token sequence.
- Feed the LSTM a continuous token stream across robot steps.
- Emit action distributions only at learned action-readout tokens placed at step boundaries.

Per-step token stream shape:

```text
visual_token_1 + Dh + previous_action + token_type=visual
visual_token_2 + Dh + previous_action + token_type=visual
...
visual_token_N + Dh + previous_action + token_type=visual
action_readout + Dh + previous_action + token_type=readout -> action logits
```

Definitions:

- `N`: number of visual tokens for the image.
- `Dv`: visual-token feature width.
- `Dh`: VLM conditioning-head width.
- `previous_action`: previous action distribution or executed action representation.
- `action_readout`: learned token indicating that the LSTM should emit one action distribution for the current robot step.

The LSTM hidden state is not reset between steps within a training sequence.
This lets the LSTM carry temporal state while still receiving explicit previous-action information.
`Dh` is constant for each `K`-step segment.

Dataset behavior:

1. Load multiple phase 1 runs.
2. Support copied old data.
3. Support new data with explicit action distributions.
4. Reconstruct old one-hot distributions from `agentic_action_name`.
5. Reconstruct bounded VLM context with shared `picar_kl.context` machinery when records contain enough source material.
6. Precompute mandatory real visual-token encodings from phase 1 images.
7. Cache visual-token encodings under ignored artifacts so training does not recompute Qwen features every epoch.
8. Record encoding shape metadata and fail loudly if later runs mix incompatible visual-token shapes without an explicit adapter.
9. Provide sequences of visual-token encodings, previous actions, VLM head conditioning, readout positions, and target action distributions.

Old-data compatibility:

- old data has images and action names.
- old action names deterministically reconstruct one-hot action distributions.
- old data lacks newer latency events, context-budget metadata, and usually full reward-result/raw reward text.
- these missing fields are not blockers for phase 2 KL projection.
- missing fields must remain explicit rather than silently fabricated.

Model behavior:

1. Qwen visual stack provides per-image visual-token encodings.
2. A learned visual projection maps `[Dv] -> [Dmodel]` per token.
3. A VLM conditioning head emits one `Dh` vector per `K`-series.
4. `Dh` is appended to each visual token and readout token in that `K`-series.
5. Previous action representation is appended to each token for the current step.
6. A learned action-readout token is appended after each image's visual tokens.
7. The LSTM consumes the continuous token stream across steps.
8. The action head reads LSTM hidden states at readout positions only.
9. The LSTM emits action distributions over the canonical 8-action set.
10. KL loss trains LSTM distributions against VLM target distributions.

Phase D.1: Encoding Cache

Goal: turn phase 1 images into reusable Qwen visual-token encodings.

Tasks:

1. Implement a real Qwen visual-token encoder behind an explicit protocol.
2. Decide and document the chosen Qwen tensor source after inspecting actual model outputs.
3. Preserve visual-token sequences rather than mean-pooling them.
4. Store encoding tensors under ignored `artifacts/data/phase2/encodings/`.
5. Store a cache manifest with:
   - source run id.
   - source image path.
   - model name and manifest path.
   - image shape.
   - encoding shape.
   - tensor dtype.
   - encoder version/config hash.
6. Add `scripts/precompute_phase2_encodings.py`.

Tests:

1. Cache manifest round trip.
2. Cache hit/miss behavior.
3. Dataset refuses missing encodings in production mode.
4. Real image-to-Qwen encoding smoke test marked `gpu`.

Phase D.2: Dataset and Collation

Goal: produce token-stream training batches from phase 1 records and cached encodings.

Tasks:

1. Load old and new phase 1 records.
2. Resolve each record's cached visual-token encoding.
3. Reconstruct target action distributions.
4. Build previous-action inputs.
5. Group records into fixed or configurable sequence windows.
6. Build readout-position masks.
7. Preserve enough metadata to trace losses back to source run/step.
8. Handle old data's missing reward/context/latency fields explicitly.

Tests:

1. Old action-name rows reconstruct one-hot targets.
2. New explicit distributions pass through unchanged.
3. Collation returns visual tokens, previous actions, targets, readout masks, and metadata.
4. Variable run lengths and short runs are handled.

Phase D.3: LSTM KL Model

Goal: implement the offline KL-projection model around the token-stream design.

Likely modules:

```text
picar_kl/models/visual.py
picar_kl/models/vlm_head.py
picar_kl/models/lstm_policy.py
picar_kl/training/kl_projection.py
```

Tasks:

1. Implement learned visual-token projection.
2. Implement VLM conditioning head producing `Dh`.
3. Implement learned action-readout token.
4. Append `Dh`, previous action, and token-type features to visual/readout tokens.
5. Run LSTM across the full token stream.
6. Emit action logits at readout positions only.
7. Compute KL loss against target action distributions.
8. Keep model dimensions configurable.

Tests:

1. LSTM forward pass shape checks.
2. Readout-mask action logits shape checks.
3. KL loss numerical behavior.
4. Training step updates trainable parameters.
5. Gradients reach visual projection, VLM head, LSTM, and action head.

Phase D.4: Offline Training Script

Goal: train from precomputed encodings and phase 1 records.

Likely modules and scripts:

```text
picar_kl/phase2/cache.py
picar_kl/phase2/dataset.py
picar_kl/phase2/train.py
scripts/train_phase2_kl.py
```

Tasks:

1. Add a training config/dataclass.
2. Load phase 1 records and encoding cache manifests.
3. Build dataset and dataloader.
4. Train for a configurable number of epochs.
5. Save compact run outputs under `experiments/runs/`, keeping each run under 1MB.
6. Save large model checkpoints under ignored artifacts if needed.
7. Log KL loss and basic dataset statistics.

Tests:

1. Tiny precomputed-encoding fixture can train for a few steps.
2. Loss/logging output shape is stable.
3. Experiment-run output stays compact.

Check-in D:
Confirm the offline pipeline consumes copied old data plus new data, uses mandatory cached Qwen visual-token encodings, and decreases KL loss on a tiny dataset before adding robot execution.

### Phase D.1 Encoding Cache Implementation Pass

Implemented:

- added `/src/picar_kl/models/visual.py`:
  - `VisualTokenEncoding`
  - `VisualTokenEncoder` protocol
  - `QwenVisualTokenEncoder`
- chose the Qwen tensor source:
  - `Qwen2_5_VLModel.get_image_features(...).pooler_output`
  - in Transformers 5.2 this returns one `[num_visual_tokens, hidden_dim]` tensor per image.
  - despite the `pooler_output` name, this is not mean-pooled to one vector.
  - these are the projected image embeddings Qwen scatters into the language sequence.
- added `/src/picar_kl/phase2/cache.py`:
  - cache config.
  - cache entry schema.
  - JSON manifest round trip.
  - compressed `.npz` tensor storage.
  - cache keys based on source image, source step, model, manifest, encoder id, and config hash.
- added `/scripts/precompute_phase2_encodings.py`.

Cache location:

- default cache root is `/artifacts/data/phase2/encodings/`.
- tensor files are stored under `tensors/`.
- manifest is stored as `manifest.json`.

Added tests:

- visual-token encoding shape and dtype validation.
- Qwen visual-token encoder preserves a token sequence in a fake Qwen-shaped unit test.
- encoding cache stores tensors and manifests.
- cache keys distinguish steps.

Verification:

- `~/.venv/bin/python -m pytest -q tests/unit/models/test_visual.py tests/unit/phase2/test_cache.py` passed with 4 tests.
- `~/.venv/bin/python -m py_compile scripts/precompute_phase2_encodings.py src/picar_kl/models/visual.py src/picar_kl/phase2/cache.py` passed.
- `~/.venv/bin/python -m pytest -q` passed with 60 tests.
- real Qwen precompute smoke passed:
  - command: `PYTHONPATH=src ~/.venv/bin/python scripts/precompute_phase2_encodings.py --data-root artifacts/data/phase1 --cache-root artifacts/data/phase2/encodings --limit 1`
  - source run: `04f639c9-478a-4cb2-ab4d-086813915793`, step `0`.
  - image shape: `[120, 160, 3]`.
  - encoding shape: `[24, 2048]`.
  - dtype: `float16`.
  - compressed tensor size: roughly `76K`.
  - cache root size after one entry: roughly `88K`.

Remaining check-in:

- inspect whether `[24, 2048]` remains stable across newer phase 1 runs before broad precompute jobs.

### Phase D.2 Dataset and Collation Implementation Pass

Implemented:

- added `/src/picar_kl/phase2/dataset.py`:
  - `Phase2StepExample`
  - `Phase2SequenceExample`
  - `Phase2Batch`
  - `load_phase2_sequences`
  - `phase2_step_from_record`
  - `collate_phase2_sequences`
- dataset construction requires cached visual-token encodings.
- old action-name records reconstruct one-hot targets through the canonical action space.
- new explicit action distributions pass through the existing record loader.
- previous action defaults to `look-forward` at run starts.
- previous action then tracks the previous target action distribution within each run.
- collation returns:
  - visual tokens: `[batch, steps, visual_tokens, visual_dim]`
  - previous actions: `[batch, steps, action_dim]`
  - target distributions: `[batch, steps, action_dim]`
  - readout mask: `[batch, steps]`
  - step mask: `[batch, steps]`
  - trace metadata per source step.
- incompatible visual-token shapes fail loudly.
- missing context/reward/latency fields remain explicit in metadata.

Added tests:

- old and new records load into phase 2 sequences.
- old action names reconstruct one-hot target distributions.
- previous-action inputs are correct across steps.
- short final sequences pad with masks.
- missing cached encodings fail loudly.
- incompatible visual-token shapes fail loudly.

Verification:

- `~/.venv/bin/python -m pytest -q tests/unit/phase2` passed with 5 tests.
- `~/.venv/bin/python -m pytest -q` passed with 63 tests.

Remaining check-in:

- run a small integration-style dataset build using real cached encodings from multiple newer phase 1 records before D.3 model work.

Follow-up verification:

- ran bounded real-Qwen precompute against newer run 375c751d-e0e3-4288-a247-d0d673596adf.
- encoded 16 newer-run records.
- combined cache now covers 16 imported-run records and 16 newer-run records.
- all 32 usable cached steps share visual-token shape [24, 2048].
- dataset collation produced:
  - visual tokens: [8, 4, 24, 2048].
  - previous actions: [8, 4, 8].
  - target distributions: [8, 4, 8].
  - readout mask: [8, 4].
- target distribution row sums remained exactly 1.0.
- note: loaders must use the same cache config as precompute; current precompute sets config_hash=output_dtype=float16.


### Phase D.3 LSTM KL Model Implementation Pass

Implemented:

- added /src/picar_kl/models/vlm_head.py:
  - VLMConditioningHeadConfig
  - VLMConditioningHead
- added /src/picar_kl/models/lstm_policy.py:
  - LSTMPolicyConfig
  - TokenStreamLSTMPolicy
  - LSTMPolicyOutput
- added /src/picar_kl/training/kl_projection.py:
  - masked_action_kl_loss
  - train_kl_projection_step
  - KLProjectionStepResult
- model preserves the visual-token sequence.
- model projects visual-token width with a learned projection.
- model appends:
  - Dh conditioning.
  - previous action distribution.
  - learned token-type features.
- model appends a learned action-readout token after each image visual-token sequence.
- LSTM runs across the flattened token stream.
- action logits are emitted only from per-step readout positions.
- KL loss is masked over valid readout positions.

Added tests:

- LSTM forward pass shape checks.
- readout and token mask shape checks.
- zero-conditioning smoke path.
- nonzero Dh changes logits.
- synthetic VLM conditioning head gradients flow through the policy.
- KL loss prefers a matching target distribution.
- KL loss rejects an empty valid-step mask.
- one training step updates both policy and conditioning-head parameters.

Verification:

- ~/.venv/bin/python -m pytest -q tests/unit/models/test_lstm_policy.py tests/unit/training/test_kl_projection.py passed with 7 tests.
- ~/.venv/bin/python -m pytest -q tests/unit/phase2 tests/unit/models tests/unit/training passed with 17 tests.
- ~/.venv/bin/python -m py_compile src/picar_kl/models/vlm_head.py src/picar_kl/models/lstm_policy.py src/picar_kl/training/kl_projection.py passed.
- ~/.venv/bin/python -m pytest -q passed with 70 tests.


### Phase D.4 Offline Training Script Implementation Pass

Implemented:

- added /src/picar_kl/phase2/train.py:
  - Phase2TrainingConfig
  - Phase2TrainingResult
  - run_phase2_training
  - phase2_batch_to_tensors
- added /scripts/train_phase2_kl.py.
- trainer loads phase 1 records through the phase 2 dataset path.
- trainer resolves cached visual-token encodings using the full cache identity:
  - cache root.
  - model name.
  - manifest path.
  - encoder id.
  - config hash.
- trainer builds the token-stream LSTM policy from cached encoding shape.
- trainer now produces Dh inside the trainable model from prefix windows.
- trainer writes compact run summaries under experiments/runs/phase2 by default.
- trainer writes model checkpoints under ignored artifacts/models/phase2 by default.
- trainer supports explicit partial-cache mode for smoke/trial runs.
- strict cache mode remains the default.

Added tests:

- batch-to-tensor conversion casts visual encodings to float32 for training.
- window-to-tensor conversion separates prefix context from target KL segment.
- tiny synthetic cached dataset trains for multiple epochs.
- compact summary is written and stays below 1 MB.
- checkpoint output is written under the configured checkpoint root.
- checkpoint writing can be disabled.

Verification before joint-Dh correction, superseded by the design-correction verification below:

- ~/.venv/bin/python -m pytest -q tests/unit/phase2/test_train.py passed with 3 tests.
- ~/.venv/bin/python -m pytest -q tests/unit/phase2/test_dataset.py tests/unit/phase2/test_train.py passed with 6 tests.
- ~/.venv/bin/python -m pytest -q tests/unit/phase2 tests/unit/models tests/unit/training passed with 20 tests.
- PYTHONPATH=src ~/.venv/bin/python scripts/train_phase2_kl.py --help passed.
- tiny real-cache CLI smoke passed with explicit partial-cache mode:
  - run id: d4-cli-smoke-2.
  - final loss: 2.044505.
  - sequences: 2.
  - valid steps: 8.
  - summary size: 1345 bytes.
- ~/.venv/bin/python -m py_compile src/picar_kl/phase2/dataset.py src/picar_kl/phase2/train.py scripts/train_phase2_kl.py passed.
- ~/.venv/bin/python -m pytest -q passed with 73 tests.


### Phase D.4 Design Correction: Joint Dh Training

Decision:

- Do not persist Dh vectors as final training inputs.
- Dh must be produced inside the trainable model.
- The VLM/conditioning head and LSTM policy must be trained together under the KL objective.
- Cache frozen Qwen visual-token encodings only; they are expensive reusable inputs, not trainable targets.

Windowing rule:

- Each phase 2 training example is split into a prefix segment and a target segment.
- Default shape is K prefix actions followed by K target actions.
- The prefix segment provides context to the trainable conditioning head.
- The target segment receives the generated Dh vector.
- KL divergence is computed only over target-segment action distributions.
- This avoids leaking future actions into Dh while keeping Dh differentiable with the policy.

Implementation update:

- added Phase2WindowExample and Phase2WindowBatch.
- added load_phase2_windows and collate_phase2_windows.
- added PrefixConditioningHead.
- added Phase2KLModel.
- revised run_phase2_training to use prefix-target windows.
- revised CLI options from sequence-length/max-sequences to context-steps/prediction-steps/max-windows.
- current smoke path trains conditioning head plus LSTM jointly.

Verification after correction:

- focused joint-Dh tests passed with 16 tests.
- phase2/model/training tests passed with 23 tests.
- real-cache joint training smoke passed through run_phase2_training:
  - run id: d4-joint-smoke.
  - final loss: 2.0318193435668945.
  - windows: 2.
  - valid target steps: 4.
  - summary size: 1495 bytes.
- full unit suite passed with 76 tests.


## Build Phase D.5: Post-Review Training and Evaluation Changes

Goal: make phase 2 fitting interpretable, repeatable, and useful for long-running experimentation.

Fit modes:

1. `window_sampling_fit`
   - primary fit/evaluation mode.
   - sample random `M*K` windows.
   - use the first `(M-1)K` steps as prefix context.
   - compute KL over the final `K` target steps.
   - use this mode for hold-out metrics and stopping decisions.

2. `full_sequence_fit`
   - runtime-realistic mode.
   - walk complete runs in order.
   - update/use `Dh` as deployment would.
   - use this mode for initialization, finishing, and runtime simulation.
   - report its metrics as sequential/runtime metrics, not simple-random validation claims.

Metrics:

- KL / cross entropy.
- top-1 action accuracy.
- target-action probability.
- entropy of predicted action distribution.
- per-action confusion matrix.
- run count, window count, target-step count, and skipped-window count.

Hold-out and breakout metadata:

- every metric artifact must include split/breakout metadata.
- prefer a structured JSON field named `breakout`.
- `breakout` must identify at least:
  - fit mode.
  - split name, like `train`, `validation`, `test`, or `runtime`.
  - split strategy, like `random_window`, `run_id`, or `contiguous_block`.
  - run ids included.
  - context steps.
  - prediction steps.
  - window stride or sampling policy.
  - random seed when sampling is random.
- filenames may also include split names for readability, but JSON metadata is authoritative.

Artifacts:

- every model fit gets a separate ignored artifact directory under `/artifacts/models/phase2/<fit_id>/`.
- each fit artifact should include:
  - checkpoint.
  - config.
  - metric JSONL.
  - final summary.
  - split/breakout metadata.
  - source cache identity.
- compact distilled summaries go under `/experiments/runs/phase2/<fit_id>/`.
- phase 3 starter checkpoints should reference phase 2 fit ids or checkpoint paths from these artifacts.

Notebook/reporting:

- training must run from scripts, suitable for `tmux`.
- notebooks must consume emitted artifacts only.
- add a fitting report notebook under `/experiments/reports/` after metric artifacts exist.

Implementation D.5:

- added `Phase2Breakout` and action metrics utilities.
- added per-fit metric JSONL under `/artifacts/models/phase2/<fit_id>/metrics.jsonl`.
- added per-fit `config.json`, artifact-local `summary.json`, and compact `/experiments/runs/phase2/<fit_id>/summary.json`.
- added `window_sampling_fit` with random validation split metadata.
- added `full_sequence_fit` with sequential/runtime split metadata.
- added selected, loaded, candidate, and skipped window counts to fit summaries.
- added skipped record counts, including missing-cache and no-action breakouts.
- missing cached encodings now break contiguous training windows when partial cache loading is allowed.
- added `--fit-mode` and `--validation-fraction` to `scripts/train_phase2_kl.py`.

Verification D.5:

- Phase 2 unit tests passed with 14 tests.
- full test suite passed with 81 tests.
- real-cache `window_sampling_fit` smoke passed:
  - run id: `d5-window-smoke-1783097743`.
  - selected windows: 4.
  - loaded windows: 14.
  - candidate windows: 300.
  - skipped windows: 286.
  - train windows: 3.
  - validation windows: 1.
  - valid target steps: 6.
  - final loss: 1.9025554656982422.
- real-cache `full_sequence_fit` smoke passed:
  - run id: `d5-full-smoke-1783097754`.
  - selected windows: 4.
  - loaded windows: 14.
  - candidate windows: 300.
  - skipped windows: 286.
  - train windows: 4.
  - validation windows: 0.
  - valid target steps: 8.
  - final loss: 1.9024055004119873.


Check-in D.5:
Confirm window-sampling metrics are interpretable, fit artifacts are versioned, and reports can be generated without re-running training.


## Build Phase E: Phase 2 Robot Execution

Goal: run the robot with sparse VLM and fast LSTM action generation.

Core decisions:

- default checkpoint selection loads the latest Phase 2 fit artifact.
- allow explicit checkpoint selection by `--fit-id` or `--checkpoint-path`.
- `K == prediction_steps`.
- `prediction_steps` is the number of actions the LSTM predicts per VLM/head cycle.
- keeping `K == prediction_steps` preserves the Phase 2 KL-projection interpretation and should simplify Phase 3 online KL-divergence.
- runtime device defaults to `auto`, selecting CUDA when available.
- keep Phase 1 speech input, speech output, and step logging behavior.
- Phase 2 should add LSTM policy outputs between VLM refreshes, not remove the conversational robot UX.

Runtime loop:

1. Select Phase 2 fit artifact.
2. Load fit summary, config, and checkpoint.
3. Set `K = prediction_steps` from the fit summary unless explicitly provided with the same value.
4. Capture image each step.
5. Encode image each step using the same visual-token cache/runtime identity expected by the fit.
6. Every `K` steps, run the VLM/head path over the current prefix context to produce a fresh `Dh`.
7. Keep `Dh` constant for exactly the next `K` LSTM target steps.
8. Run the LSTM each step.
9. Convert the LSTM action distribution to a robot vector.
10. Apply the vector.
11. Log action probabilities, selected action, `Dh` metadata, latency, speech context, VLM text, and coherency-relevant records.

Build Phase E.1: Offline Policy Loader and Replay Smoke

Goal: prove the fitted Phase 2 policy can be loaded and used before touching live robot movement.

Tasks:

- add a Phase 2 fit artifact loader.
- support latest-fit, `fit_id`, and direct checkpoint path selection.
- reconstruct `Phase2KLModel` from checkpoint metadata.
- load checkpoint weights onto `auto`, `cuda`, or `cpu`.
- expose a replay API that consumes cached Phase 1 visual encodings.
- replay a Phase 1 run/window sequence through the policy.
- emit valid action distributions and timing records.
- add unit tests for loader selection and replay output shape/probability validity.
- add a script-level smoke command for replay.

Implementation E.1:

- added `picar_kl.phase2.runtime`.
- added Phase 2 fit artifact selection by latest fit, `fit_id`, or direct checkpoint path.
- added checkpoint loading onto `auto`, `cuda`, or `cpu`.
- added CPU-safe checkpoint reconstruction for `Phase2KLModel`.
- added offline replay over cached Phase 1 visual encodings.
- replay emits valid `ActionRecord` distributions, selected action names, vectors, conditioning norms, and latency estimates.
- added `scripts/run_phase2_replay.py`.
- documented replay in `/docs/humans/run_phase2.md`.

Verification E.1:

- runtime unit tests passed with 3 tests.
- Phase 2 unit tests passed with 19 tests.
- full unit suite passed with 86 tests.
- real-artifact replay smoke passed via API:
  - fit id: `phase2-window-20260703T170537Z`.
  - device: `cpu`.
  - windows: 2.
  - replayed steps: 8.
  - first predicted action: `look-forward`.
- direct script invocation could not be smoke-tested inside this sandbox because the command wrapper failed before Python started with `bwrap: loopback: Failed RTM_NEWADDR`; script compilation passed.


Build Phase E.2: Live Phase 2 Runtime Loop

Goal: run the live robot with VLM refreshes every `K` steps and LSTM actions every step.

Tasks:

- add `scripts/run_phase2_robot.py`.
- reuse Phase 1 robot client, speech listener, TTS, context history, and logging patterns.
- run Qwen/head only at cycle boundaries.
- run visual encoding and LSTM policy per step.
- enforce `K == prediction_steps`.
- record raw wall-clock latency events separately for image capture, visual encoding, VLM/head refresh, LSTM action, robot request, speech input, and speech output.
- continue writing data incrementally so `Ctrl-C` preserves completed steps.

Implementation E.2:

- added `picar_kl.phase2.live`.
- added live Phase 2 execution with VLM bootstrap steps followed by LSTM actions.
- enforce `K == prediction_steps` at runtime.
- keep Phase 1 speech input, speech output, reward scoring, context rendering, and incremental observation logging.
- share one Qwen runtime across VLM decisions, visual-token encoding, and frozen-VLM reward scoring in the script runner.
- record separate latency events for image capture, visual encoding, reward scoring, context rendering, VLM bootstrap, VLM/head refresh, LSTM action, robot action, and speech output.
- added `scripts/run_phase2_robot.py`.
- documented live Phase 2 execution in `/docs/humans/run_phase2.md`.

Verification E.2:

- live runtime unit tests passed with 2 tests.
- live/runtime Phase 2 tests passed with 5 tests.
- Phase 2 unit tests passed with 21 tests.
- full unit suite passed with 88 tests.
- script compilation passed.
- direct script invocation could not be smoke-tested inside this sandbox because the command wrapper failed before Python started with `bwrap: loopback: Failed RTM_NEWADDR`.

Build Phase E.3: Integration Tests

Goal: test live-execution wiring before a real robot trial.

Tests:

- fake robot/fake visual encoding/fake policy end-to-end loop.
- offline replay smoke using a real Phase 2 checkpoint and cached Phase 1 encodings.
- manual robot smoke adapted from old integration tests.
- optional real VLM path marked `gpu`.

Metrics:

- raw wall-clock latency events.
- coherency records for later VLM judging.
- action distributions from LSTM and cycle-level VLM/head metadata.
- metric aggregation must tolerate old phase 1 data that lacks explicit latency events.
- when latency events are missing, metric code may report unavailable values or derive rough timestamp intervals explicitly labeled as estimates.

Scripts:

```text
scripts/run_phase2_replay.py
scripts/run_phase2_robot.py
```

Check-in E:
Experimenter reviews replay smoke, then runs full integration test, then real robot phase 2 trial.

## Build Phase F: Documentation and Handoff

Goal: make the repo usable without the old codebases.

Docs to write:

1. `/docs/humans/setup.md`
2. `/docs/humans/build.md`
3. `/docs/humans/robot_install.md`
4. `/docs/humans/run_phase1.md`
5. `/docs/humans/train_phase2.md`
6. `/docs/humans/run_phase2_robot.md`
7. `/docs/humans/data_layout.md`

Also update:

- `/README.md` if the user wants top-level usage notes.
- `/docs/agents/artifact-review-notes.md` with implementation findings.
- this file with build notes and deviations.

Check-in F:
Confirm phase 1 and phase 2 commands are clear enough for the experimenter to run.

## Integration Test Import Plan

Bring over old robot-demo integration tests and refactor them:

1. Speech round-trip manual test.
2. PiCar environment manual smoke test.
3. Vision stream manual test.

Expected markers:

- `integration`
- `robot`
- `gpu` when real VLM is involved.

Default `pytest` should not require hardware.
Manual integration tests should provide clear setup instructions.

## Existing Artifact Import Plan

Old sources to copy:

1. `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/src`
2. `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/tests`
3. `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/data`
4. `../sufficiency-and-efficiency-in-deep-rl/robot-demo/demo/model/manifests`
5. `../../picar-v-rl-env/src`
6. `../../picar-v-rl-env/run_api.py`
7. `../amari-chenstov-updates/mathematical_overview.ipynb`

Do not copy large model weights.
Do copy small manifests.
Do copy existing phase 1 data because the experimenter says it is not too large.

For copied code, record provenance according to `/docs/agents/import-provenance-notes.md`.

## Remaining Concerns

1. Full copy then refactor is workable but increases short-term duplication.
   Mitigation: copy relevant content first, immediately put stable tests around imports, then collapse duplicate code by module family.

2. The old VLM code may be expensive to exercise in tests.
   Mitigation: keep explicit fake test doubles first-class and mark real VLM tests as `gpu`.

3. Old phase 1 data lacks explicit latency records and may only support rough timestamp-derived latency.
   Mitigation: compatibility loader accepts it; new phase 1 generation records raw latency events; metric code labels missing or estimated latency clearly.

4. The real Qwen visual encoding interface may shape LSTM data too early.
   Mitigation: define a visual encoder protocol and test with explicit fake encodings before binding to Qwen internals.

5. Robot hardware imports can break server-side tests if not isolated.
   Mitigation: robot Flask app imports hardware code only on robot-side entry paths.

6. Phase 2 depends on a VLM head not present in old code.
   Mitigation: implement the conditioning head behind a small module with fake tests before real Qwen integration.

7. Packaging can accidentally pull server/training dependencies onto the Raspberry Pi.
   Mitigation: keep base dependencies robot-light, build a wheel only for robot install, and keep server development on `requirements-server.txt`.

## Current Recommendation

Proceed with the build in check-pointed phases.
Before Phase C runtime work, fix the dependency split so `torch` is no longer a base package dependency.
Then build the phase 1 loop and robot Flask entry point together.
Do not start phase 2 model training until copied data loads cleanly, action distribution semantics are tested, and server-only dependencies are isolated from robot install.

## Build Notes

### Phase A Completed

Phase A scaffold and import pass has been executed.

Created:

- `/src/picar_kl/`
- `/tests/unit/`
- `/tests/integration/`
- `/config/`
- `/experiments/configs/`
- `/experiments/runs/`
- `/experiments/reports/`
- `/artifacts/manifests/tracked/`
- `/artifacts/manifests/ephemeral/`
- `/artifacts/models/`
- `/artifacts/data/`
- `/notebooks/`
- `/docs/humans/`
- `/scripts/`
- `/pyproject.toml`
- `/requirements-server.txt`
- `/requirements-robot.txt`
- `/.gitignore`

Copied:

- old robot-demo source into `/src/picar_kl/legacy/robot_demo/src/`
- old robot-demo tests into `/src/picar_kl/legacy/robot_demo/tests/`
- old robot-demo integration tests into `/tests/integration/robot_demo/`
- full old robot-demo data into `/artifacts/data/legacy_robot_demo/`
- old phase 1 data into `/artifacts/data/phase1/`
- old model manifests into `/artifacts/manifests/tracked/models/`
- old PiCar API code into `/src/picar_kl/robot/legacy_car_env/`
- old PiCar API runner into `/src/picar_kl/robot/legacy_run_api.py`
- Amari-Chentsov notebook into `/notebooks/imported/`

Added provenance:

- `/docs/agents/phase-a-import-provenance.md`
- `/docs/humans/math-context.md`
- local README files in staged legacy directories.

Adjusted:

- copied integration tests now point at `/src/picar_kl/legacy/robot_demo/src/`.
- default pytest path is `/tests/unit/`, so integration tests remain manual.

Verification:

- `~/.venv/bin/python -m pytest -q` passed with the scaffold smoke test.
- full old data and phase 1 subset copies are present locally and remain gitignored.

### Phase B Core Interface Pass Completed

Implemented:

- `/src/picar_kl/actions.py`
- `/src/picar_kl/latency.py`
- `/src/picar_kl/records.py`
- `/src/picar_kl/data/phase1.py`
- `/src/picar_kl/robot/client.py`
- `/src/picar_kl/robot/server.py`

Established:

- canonical eight-action order.
- JSON-friendly action distributions.
- deterministic distribution-to-vector conversion.
- raw wall-clock latency event records.
- phase 1 record types for new data.
- legacy row conversion from `agentic_action_name` to one-hot distributions.
- recursive phase 1 run discovery for old root UUID runs and nested `phase1` runs.
- `.npz` image loading.
- retrying PiCar HTTP client.
- robot-side Flask app factory without top-level hardware imports.

Added unit tests for:

- action round trips.
- distribution validation.
- distribution-to-vector conversion.
- latency serialization and timer behavior.
- legacy action and observation conversion.
- nested phase 1 run discovery.
- phase 1 `.npz` image loading.
- PiCar client route, vector, JPEG, and timeout behavior.

Verification:

- `~/.venv/bin/python -m pytest -q` passed with 20 tests.
- `~/.venv/bin/python -m py_compile` passed for new Phase B modules.

Phase C starting backlog:

- remove `torch` from base package dependencies and keep it server-side.
- add a robot wheel build script.
- add a `picar-kl-robot-server` entry point.
- choose the exact new phase 1 writer layout.
- adapt speech hooks behind explicit interfaces.
- move the VLM-only runtime off staged legacy modules.
- add fake VLM/fake robot end-to-end tests for the phase 1 loop.

### Phase C Initial Runtime Pass Completed

Implemented:

- moved `torch` out of base package dependencies and into server extras.
- added `picar-kl-robot-server` entry point.
- added `picar-kl-phase1` entry point.
- added `/scripts/build_robot_wheel.py`.
- added `/src/picar_kl/robot/app.py`.
- added `/src/picar_kl/io/observation_store.py`.
- added `/src/picar_kl/vlm/control.py`.
- added `/src/picar_kl/phase1/run.py`.
- added `/src/picar_kl/phase1/cli.py`.
- added `/docs/humans/robot_install.md`.

Established:

- server path remains `pip install -r requirements-server.txt`.
- robot path uses a built wheel with robot extras.
- robot server hardware imports are deferred until the command runs without `--dry-run`.
- phase 1 runtime is protocol-driven for robot, VLM, speech source, and speaker.
- phase 1 records now include image blobs, messages, user texts, action distributions, action receipts, generated text, and raw latency events.
- fixed-action VLM controller exists only for smoke tests and loop validation.

Added unit tests for:

- pyproject dependency split and entry points.
- phase 1 observation store.
- VLM message construction and action parsing.
- fake robot/fake VLM phase 1 end-to-end loop.

Verification:

- `~/.venv/bin/python -m pytest -q` passed with 27 tests.
- `~/.venv/bin/python -m py_compile` passed for new Phase C modules and build script.

Initial pass handoff:

- wire the real Qwen2.5-VL-3B phase 1 controller.
- adapt speech hooks behind explicit interfaces.
- decide whether `picar-kl-phase1` should allow fixed-action smoke mode by default or require an explicit flag.
- run a robot-side dry-run/manual install check on the Raspberry Pi.
- adapt copied integration tests to the new runtime path.

### Phase C Controller and Speech Pass Completed

Implemented:

- `/src/picar_kl/models/store.py`.
- `/src/picar_kl/vlm/qwen.py`.
- `/src/picar_kl/speech/adapters.py`.
- `/docs/humans/run_phase1.md`.
- fake-runtime integration test under `/tests/integration/phase1/`.

Established:

- Qwen2.5-VL phase 1 controller loads from tracked VLM manifests and local model assets.
- model downloads are opt-in.
- Qwen controller defaults to JSON action parsing and returns action distributions.
- Qwen processor has a fallback to the copied image-only processor helper.
- fixed-action mode is explicit via `--smoke-fixed-action`.
- speech input and output are explicit CLI flags.
- speech adapters sit behind the existing phase 1 protocols.
- a new-runtime integration test exercises fake robot plus fake VLM without hardware.

Added unit tests for:

- model manifest resolution and missing asset errors.
- Qwen generated-action parsing with fake model and processor.
- speech source and speaker adapters.

Verification:

- `~/.venv/bin/python -m pytest -q` passed with 34 tests.
- `~/.venv/bin/python -m pytest -q tests/integration/phase1/test_phase1_fake_runtime.py` passed.
- `~/.venv/bin/python -m py_compile` passed for new controller/speech modules.
- `~/.venv/bin/python scripts/build_robot_wheel.py --dist-dir /tmp/picar-kl-wheel-test` passed.
- wheel metadata confirmed `torch` is server-extra only, not a base dependency.

Remaining Phase C work:

- run a robot-side dry-run/manual install check on the Raspberry Pi.
- run a server-side Qwen smoke test once local model assets are available.
- adapt copied hardware integration tests to the new robot server and phase 1 commands.

### Phase C Vision Integration Refactor Completed

Implemented:

- refactored `/tests/integration/robot_demo/vision_test.py` to use `picar_kl.robot.client.PiCarClient`.
- updated `/tests/integration/robot_demo/README.md` for the new vision test command.

Established:

- `vision_test.py` is now a robot Flask server and camera smoke test.
- it no longer imports legacy env, legacy model, reward scorer, Qwen, STT, or TTS.
- it validates frame shape and dtype.
- desktop frame display is optional via `PICAR_SHOW_VISION_WINDOW=1`.

Verification:

- `~/.venv/bin/python -m pytest -q` passed with 34 tests.
- `~/.venv/bin/python -m pytest -q tests/integration/robot_demo/vision_test.py` skipped cleanly without `PICAR_V_HOST`.
- `~/.venv/bin/python -m py_compile tests/integration/robot_demo/vision_test.py` passed.

Recommended manual sequence:

1. Build wheel: `~/.venv/bin/python scripts/build_robot_wheel.py`.
2. Copy and install the wheel on Raspberry Pi; use `pip install -r requirements-robot.txt` for fresh dependency setup and `pip install --force-reinstall --no-deps /home/pi/picar_kl-0.1.0-py3-none-any.whl` for code-only reinstalls.
3. Start robot server on Raspberry Pi: `picar-kl-robot-server --host 0.0.0.0 --port 5000`.
4. Run vision test from server machine: `PICAR_V_HOST=<host:port> ~/.venv/bin/python -m pytest -q tests/integration/robot_demo/vision_test.py -s`.

Remaining Phase C work:

- record the live robot install, camera, speech, and env smoke results.
- run a small phase 1 data-generation smoke against the live robot.
- keep the `legacy/...` paths as short-term debt only; move model assets under `/artifacts/models/` as refactors continue.

### Phase C Live Robot and Legacy Integration Check Completed

Validated:

- built the robot wheel and installed it on the Raspberry Pi.
- started `picar-kl-robot-server` on the PiCar.
- added `--no-camera` for hardware-control checks when camera debugging blocks startup.
- diagnosed the original USB camera as faulty after V4L streaming hung on both the Raspberry Pi and Ubuntu tower.
- validated the replacement PiCar camera with `v4l2-ctl` and the Flask `/img` path.
- ran the refactored manual vision integration test against the live Raspberry Pi server.
- ran the manual speech round-trip test after routing speech models to `/artifacts/models/`.
- ran the manual PiCar env smoke test after routing VLM models to `/artifacts/models/`.

Implemented during this check:

- robot server startup no longer imports the legacy camera-opening environment path.
- legacy speech manifests fall back to tracked manifests under `/artifacts/manifests/tracked/models/`.
- legacy VLM model manifests fall back to tracked manifests under `/artifacts/manifests/tracked/models/`.
- copied legacy model downloads are ignored at `/src/picar_kl/legacy/robot_demo/model/`.

Notes:

- `/src/picar_kl/legacy/robot_demo/model/` is accidental generated state, not canonical project state.
- canonical model assets belong under `/artifacts/models/`.
- do not delete generated model assets without explicit experimenter approval.
- the env smoke test exercised the legacy Qwen-backed reward path; a separate Qwen controller smoke is only useful if the next live phase 1 run does not use `picar-kl-phase1` with the real Qwen controller.

Verification after doc and integration-path cleanup:

- `~/.venv/bin/python -m pytest -q` passed with 38 tests.
- `~/.venv/bin/python -m pytest -q tests/integration/phase1/test_phase1_fake_runtime.py` passed.

### Phase C Live Phase 1 Smoke Completed

Run:

- `b69c16ec-73bf-42c9-af64-10c1ec219cd7`
- command path: `picar_kl.phase1.cli` with real Qwen2.5-VL controller, live PiCar server, `steps=2`, `160x120` images.

Validated:

- `run_meta.json` records phase, task prompt, uuid, and controller metadata.
- `observations.jsonl` contains two step records.
- image blobs load through the phase 1 loader as `120x160x3 uint8`.
- action distributions are valid one-hot probability vectors.
- executed vectors match action mappings.
- robot receipts are recorded in action metadata.
- raw latency events are present for image capture, VLM decision, action application, and speaker.

Quality notes:

- first image capture was slow, likely camera warm-up; second capture was fast.
- VLM decision latency was sub-second in this tiny run.
- frames are very bright; future collection should watch exposure and red-ball visibility.
- the red signal appeared small and near the left/top-left of frame.
- action/text coherency was imperfect on step 1: action was `drive-left`, text was `Looking left.`

Remaining Phase C work:

- collect more operator-supervised phase 1 runs once scene lighting and ball placement look good.
- use frozen-Qwen evaluation, not handcrafted ball-coordinate labels, as the primary data-quality and reward signal.

### Phase C Longer Live Phase 1 Sample Reviewed

Run:

- `99ba808e-4531-434b-a028-138aa0f9d67a`
- real Qwen2.5-VL controller, live PiCar server, `steps=10`, `160x120` images.
- robot was allowed to drive around a living room with the red ball initially in front.

Validated:

- `observations.jsonl` contains 10 step records.
- 10 image blobs are present and load through the phase 1 loader.
- action distributions are valid one-hot vectors.
- total per-step latency was roughly 0.9-1.1 seconds in this run.
- image capture was fast after startup.
- VLM decision latency was roughly 0.4 seconds after the first step.

Behavioral notes:

- action sequence was `look-left`, then repeated `drive-forward`.
- generated text was broadly coherent with selected actions.
- red object signal was visible for most frames, moved toward the left edge, then disappeared near the end.
- Qwen did not correct left once the ball drifted toward the edge, so this is useful but noisy phase 1 data.

Data-quality note:

- do not depend on handcrafted `(ball_x, ball_y, ball_radius)` labels. Legacy CV ball detection exists as old robot-environment machinery, but the current experiment should treat frozen-Qwen evaluation as the primary reward/data-quality signal. Cheap image heuristics are acceptable for ad hoc inspection only.
