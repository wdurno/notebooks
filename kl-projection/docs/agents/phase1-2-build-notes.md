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

Goal: train an LSTM from old and new phase 1 data.

Dataset behavior:

1. Load multiple phase 1 runs.
2. Support copied old data.
3. Support new data with explicit action distributions.
4. Reconstruct old one-hot distributions from `agentic_action_name`.
5. Reconstruct bounded VLM context with shared `picar_kl.context` machinery when records contain enough source material.
6. Provide sequences of visual encodings, prior actions, VLM head conditioning, and target action distributions.

Model behavior:

1. VLM visual stack provides per-image encodings.
2. VLM conditioning head emits one vector per `K`-series.
3. LSTM receives visual encodings, prior action representation, and the constant VLM head vector for each step in the `K`-series.
4. LSTM emits action distributions every step.
5. KL loss trains LSTM distributions against VLM target distributions.

Practical simplification for first build:

- Implement the phase 2 training pipeline with clean interfaces and a fake/small encoder path for tests.
- Keep the real Qwen visual encoding path server-only and optional in tests.
- Put visual encoding behind an explicit protocol so training code does not depend on Qwen internals.
- The real Qwen encoder should fail clearly if Qwen dependencies, CUDA, or required model assets are unavailable.
- Fake encoders should be used only in unit tests and lightweight integration tests.

Example protocol shape:

```python
class VisualEncoder(Protocol):
    output_dim: int

    def encode(self, images: Sequence[Any]) -> torch.Tensor:
        ...
```

Likely modules:

```text
picar_kl/models/visual.py
picar_kl/models/vlm_head.py
picar_kl/models/lstm_policy.py
picar_kl/training/kl_projection.py
picar_kl/phase2/train.py
```

Scripts:

```text
scripts/train_phase2_kl.py
```

Tests:

1. Dataset sequence collation.
2. KL loss shape and numerical behavior.
3. LSTM forward pass.
4. Training step updates LSTM parameters.
5. Old data compatibility smoke test.

Check-in D:
Confirm offline loss decreases on a small copied/fake dataset before adding robot execution.

## Build Phase E: Phase 2 Robot Execution

Goal: run the robot with sparse VLM and fast LSTM action generation.

Runtime loop:

1. Capture image each step.
2. Encode image for LSTM each step.
3. Every `K` steps, run VLM language/strategy path and conditioning head.
4. Keep VLM head vector constant for the next `K` steps.
5. Run LSTM each step.
6. Convert LSTM distribution to robot vector.
7. Apply vector.
8. Log action latency and coherency-relevant data.

Metrics:

- raw wall-clock latency events.
- coherency records for later VLM judging.
- KL/training metrics when applicable.
- metric aggregation must tolerate old phase 1 data that lacks explicit latency events.
- when latency events are missing, metric code may report unavailable values or derive rough timestamp intervals explicitly labeled as estimates.

Scripts:

```text
scripts/run_phase2_robot.py
```

Integration tests:

- Fake robot/fake VLM end-to-end loop.
- Manual robot smoke test adapted from old integration tests.
- Optional real VLM path marked `gpu`.

Check-in E:
Experimenter runs full integration test and then real robot phase 2 trial.

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
