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
10. Copy existing phase 1 data into `/artifacts/data/phase1/`.
11. Copy math notebook/result material into `/notebooks/` or `/docs/humans/`, rewritten for the current experiment's context.

Check-in A:
Confirm directory structure, package name, copied artifact placement, and gitignore behavior before deeper refactors.

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
3. Build VLM control messages.
4. Generate VLM action distribution and robot speech.
5. Convert action distribution to continuous robot vector.
6. Apply vector to robot.
7. Capture next image when useful.
8. Persist record with image path, messages, user text, action distribution, executed vector, generated text, and raw latency events.

Important behavior:

- No LSTM involvement.
- Language stays owned by VLM.
- Action output is constrained to the small action set.
- Existing STT/TTS features should be retained where practical.
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
```

Tests:

- Unit tests with fake VLM and fake robot client.
- Integration test adapted from old PiCar env smoke test.
- Manual robot test remains marked `integration` and `robot`.

Check-in C:
Experimenter can generate a small phase 1 run, inspect files, and confirm the record shape before phase 2 training depends on it.

## Build Phase D: Phase 2 Offline KL Projection

Goal: train an LSTM from old and new phase 1 data.

Dataset behavior:

1. Load multiple phase 1 runs.
2. Support copied old data.
3. Support new data with explicit action distributions.
4. Reconstruct old one-hot distributions from `agentic_action_name`.
5. Provide sequences of visual encodings, prior actions, VLM head conditioning, and target action distributions.

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
3. `/docs/humans/run_phase1.md`
4. `/docs/humans/train_phase2.md`
5. `/docs/humans/run_phase2_robot.md`
6. `/docs/humans/data_layout.md`

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

## Current Recommendation

Proceed with the build in check-pointed phases.
Start with Phase A and Phase B.
Do not start phase 2 model training until copied data loads cleanly and action distribution semantics are tested.

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
- phase 1 data copy is present locally and remains gitignored.
