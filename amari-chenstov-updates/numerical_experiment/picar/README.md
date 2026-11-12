# Plan 19 PiCar Experiment

See [the plan](plan19.md) for theory and the scientific contract, and
[the handoff](PLAN19_HANDOFF.md) for current implementation and acceptance
status. Phase 0 is software acceptance, not evidence of robot competence.

## Environment

Run commands from the repository root. A fresh checkout does not need another
project's code or model directory. Python 3.12 is the tested interpreter.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r numerical_experiment/picar/requirements-server.txt
```

Use `requirements.txt` instead for CPU smoke/simulation without Qwen, or
`requirements-report.txt` for artifact-only reporting. Choose a Torch/CUDA
wheel appropriate for the machine and check PyTorch CUDA availability before
GPU work. The tested host uses Torch 2.10.0+cu128; pinning Python packages does
not install an NVIDIA driver. A working existing venv can be used instead.

PyBullet 3.2.7 may need a source build on Python 3.12. Its upstream build
chooses compiler parallelism from CPU/RAM counts. The acceptance host limits
that build to two compiler workers; this does not modify physics algorithms.
Allow enough RAM/time for installation and do not run a large build alongside
a memory-constrained experiment without checking resources.

## Model Assets

Explicit Qwen loading downloads the pinned Hugging Face revision into
`cache/numerical_experiment/picar/models/` by default and retains it for reuse.
The exact revision and file checksums are recorded in run metadata. Imports,
CPU smoke tests, and notebook/reporting commands never download models.

An existing checkpoint directory may be supplied with `--model-path` to avoid
another download. This is an optional optimization, not a dependency on that
directory for other users. `--offline` forbids downloads and must fail if the
requested files are absent. Do not edit source to insert a personal model path.

## Boundaries

- All robot-side service code is in `demo/picar`; the Pi does not need Qwen or Torch.
- Shared interfaces and simulator code are in `src/robotics`.
- Shared numerical functions are in `src/continual/picar_math.py`.
- The runtime does not import the earlier `kl-projection` checkout.
- Generated artifacts and model weights belong in ignored `cache/` or an explicitly provided archive root.
- The physical robot remains disarmed by default. Software acceptance uses a fake driver.

## Acceptance Commands

Run the complete bounded CPU suite and paired smoke pipeline:

```bash
.venv/bin/python -m numerical_experiment.picar.acceptance \
  --output cache/numerical_experiment/picar/phase0/acceptance-v1
```

Add `--resume` after an interruption. Completed units are verified and skipped;
changing scientific source or configuration requires a new output directory.
The fake-service test binds an ephemeral loopback port, never a physical robot.

For actual quantized Qwen, use the separate bounded CUDA check. This command
downloads and retains the pinned weights when no existing path is supplied:

```bash
.venv/bin/python -m numerical_experiment.picar.qwen_acceptance \
  --output cache/numerical_experiment/picar/phase0/qwen-check-v1
```

Optionally append `--model-path <checkpoint-directory>`; use `--offline` when
network access is forbidden. GPU acceptance checks a small input, not the full
production replay/context envelope. The adapted suffix uses FP32 computation
over frozen 4-bit weights; the frozen lower prefix remains reduced-precision.

For a manually bounded smoke/restart check and artifact-only reporting:

```bash
.venv/bin/python -m numerical_experiment.picar.run smoke \
  --output cache/numerical_experiment/picar/phase0/smoke-v1 --max-events 3
.venv/bin/python -m numerical_experiment.picar.run smoke \
  --output cache/numerical_experiment/picar/phase0/smoke-v1 --resume
.venv/bin/python -m numerical_experiment.picar.run status \
  --output cache/numerical_experiment/picar/phase0
.venv/bin/python -m numerical_experiment.picar.run report \
  --output cache/numerical_experiment/picar/phase0
```

`run stop --output <run-directory>` requests a checkpoint-boundary halt. The
next explicit `run smoke --resume` clears that request. An external archive
is optional via `--archive-root <mounted-directory>`; loss of that mount must
stop writing, not silently redirect data onto the primary disk.

See the handoff for measured results and remaining calibration work. Do not
substitute the tiny model or fake teacher into a production condition when a
dependency is missing. Phase 0 does not authorize Phases 1--5 or robot motion.

## Phase 1 Resource Check

The Phase 1 runner performs staged CUDA profiles, simulator feasibility,
frozen-role sanity checks, and an independent-clock control/update pilot. It
then writes conditional whole-study forecasts and stops for review. It never
launches the learning phases or connects to the physical robot.

```bash
.venv/bin/python -u -m numerical_experiment.picar.phase1_run \
  --output cache/numerical_experiment/picar/phase1/resource-v1 \
  --max-hours 3
```

An existing model may be reused via `--model-path <checkpoint-directory>`;
otherwise the pinned model is downloaded into the project cache. Add
`--offline` to prohibit downloads. Resume the same campaign with `--resume`:
completed resource units are verified and skipped, while changed model/source
recipes receive separate units. `run stop --output <campaign-directory>`
requests a halt at the next unit boundary; the current bounded probe finishes
first. A forced halt can require repeating an incomplete diagnostic unit.

To show both Phase 0 and Phase 1 in the artifact-only notebook:

```bash
.venv/bin/python -m numerical_experiment.picar.run report \
  --output cache/numerical_experiment/picar
```

## Phase 1 Fix 1

This versioned readiness pass audits native/custom Qwen parity, clearer role
interfaces, short RGB sequences, and cooperative online updates. It never
starts the scientific learning phases or moves the physical robot.

```bash
.venv/bin/python -u -m numerical_experiment.picar.phase1_fix \
  --output cache/numerical_experiment/picar/phase1/fix1-v2 \
  --scene cache/numerical_experiment/picar/phase1/resource-v2/units/scene-audit/summary.json \
  --max-hours 3
```

Supply an existing model with `--model-path <checkpoint-directory>` if useful;
otherwise the pinned project-local download remains the default. `--offline`
forbids downloads. Add `--resume` after an interruption. Completed units are
verified/skipped; partial role panels and accelerated teacher rollouts resume
from checkpoints. A partial short real-time timing pilot retries as a whole
diagnostic, not as a scientific learning trajectory. `run stop --output
<campaign-directory>` requests a bounded stop. Changed numerical/role source
requires a new campaign version; do not overwrite a completed attempt.

The three-hour repair cap includes recorded earlier completed repair campaigns.
The original Phase 1 forecasts remain conditional until a competent teacher,
useful reward path and common execution recipe are established.

New repair runs opt into `RunStore(..., checkpoint_policy="journal-v1")`.
This retains append-only evidence/events, the current and previous recovery
states, and explicit `checkpoint(..., milestone=True)` states. Only superseded
state payloads in an incomplete journal run are pruned after durable pointer
publication; existing completed/legacy artifacts are untouched. Journal resume
must select the same policy. Default `RunStore` behavior remains `legacy`.

## Phase 1 Fix 2

The separate section "Phase 1 Fix 2: Restore Dense Visual Rewards And
The Robot Workflow" in `plan19.md` is implemented, with bounded diagnostics
completed under `phase1/fix2-v2`.
It covers generated numeric rewards, contact semantics, causal conversational
history, public time remaining, return resolution and sustained-load checks.
It also requires the default-off retired-data sleep option and actual
actuator/camera plus speech interfaces, accepted using injected hardware and
recorded inputs. Physical operation remains separate.

The implementation entry point is now available. Its bounded campaign runs
software acceptance, stored RGB reward panels, staged teacher rollouts and
real-time pipeline timing, then stops at the resource check-in:

```bash
.venv/bin/python -u -m numerical_experiment.picar.phase1_fix2 \
  --output cache/numerical_experiment/picar/phase1/fix2-v2 \
  --max-hours 3
```

Add `--model-path <checkpoint-directory>` for explicit model reuse and
`--offline` to prohibit downloads. Default loading retains the pinned model
inside this project's cache. Add `--resume` for the same configuration/source
version; complete units are verified/skipped. `run stop --output
<campaign-directory>` requests a checkpointed halt. Original Fix 1 artifacts
are unaffected. Changed numerical/role code requires a new output version.
The cumulative repair budget includes Fix 1, with separate per-campaign totals.
The initial `fix2-v1` attempt is preserved. Its parser rejected valid fenced
JSON; `fix2-v2` records the corrected syntax handling and retired-context mask.

`--stage software`, `--stage rewards`, or `--stage teacher` permits a bounded
interim check-in; only the default `all` stage seals the finished campaign.
The short timing pilot restarts an incomplete diagnostic attempt from paired
initialization while retaining its journal traces. It is not a scientific
trajectory or a claim of identical wall-clock behavior after restart.

Retired sleep is exposed through `archive_sleep.ArchiveSleepSession`: its
explicit configuration uses `retired_archive_replay=True` and
`retired_replay_protocol="supplemental-rehearsal-v1"`. It requires a stopped
sleep interval, compatible frozen likelihood identity, a checked archive and
whole-owner checkpoint callbacks. The session records source counts/I/O and
never repeats retirement or adds new observations. The primary production
comparison rejects this option. `test_picar_archive_sleep.py` provides a
small executable integration example including exact crash recovery.

Physical driver and typed/recorded-speech entry points are documented in
`demo/picar/README.md`. Fix 2 tests those interfaces with injected ports and
recorded inputs; no robot, microphone or speaker is accessed automatically.

Acceptance passed 637 tests. Generated scores work but remain coarse; the
ported teacher succeeded in 3/6 unobstructed trials and the obstacle task in
0/12 across two interfaces. Both real-time student pilots had zero non-startup
fallbacks, with all pending labels drained during stopped work. This does not
establish student competence or sustained uninterrupted update throughput.
The notebook preserves failed checks as well as successes. Conditional
whole-study costs are 21.4--30.8 h nominal or 11.6--17.2 h at the smaller size.
Task readiness remains unresolved; the runner stops for review, not Phase 2.

## Saved-Image Questions

Run independent frozen-Qwen recognition questions on existing PNG or RGB
NumPy frames, without simulation, control prompts, conversation history or
training. Include a ball-absent image to check leading-question effects.

```bash
.venv/bin/python -u -m numerical_experiment.picar.visual_questions \
  --output cache/numerical_experiment/picar/diagnostics/visual-questions-v1 \
  --image initial=<saved-camera-frame> \
  --image absent=<saved-ball-absent-frame>
```

Default loading uses the pinned project-local model cache; explicit reuse
uses `--model-path <checkpoint-directory>`, optionally with `--offline`.
Each of the six questions starts a fresh conversation. Raw answers, image
hashes, prompts, preprocessing grids and model provenance are retained in
the immutable run; `responses.csv` is the lightweight readout. Use `--resume`
for an interrupted unchanged run, or a new output path for changed inputs.
These are qualitative diagnostics, not a calibrated depth-estimation test
or a scientific learning comparison. No additional phase is started.

## Phase 1 Fix 3

The bounded centering-reward/curriculum repair includes frozen-role diagnostics,
an easy teacher bootstrap and two short Replay/EWC student pilots. It stops for
the task/resource check-in before the main STM comparison. Run from the checkout:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar.phase1_fix3 \
  --output cache/numerical_experiment/picar/phase1/fix3-v1 --resume
```

Default loading downloads the pinned model into the project cache. Optional
`--model-path <checkpoint-directory> --offline` reuses an existing local model.
`--stage software`, `--stage rewards` and `--stage teacher` allow bounded
check-ins; resuming with `--stage all` continues the same immutable configuration.
A `STOP` file in the campaign root or SIGTERM requests a durable stop. Remove
the stop file before explicit resumption. Changed scientific sources/settings
require a new output directory; completed units remain immutable. The notebook
loads completed summaries and reports pending work without running models.
