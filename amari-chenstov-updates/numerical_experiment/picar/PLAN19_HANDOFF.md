# Plan 19 Phase 0 Handoff

## Status

Phase 0 bounded implementation acceptance passed on 2026-11-09: **280 tests,
zero failures/errors/skips**, paired CPU smoke, and actual quantized-Qwen CUDA
acceptance. No scientific learning trajectories or physical motor operations
were started. Do not start Phases 1--5 automatically.

Report: [plan19_results.ipynb](plan19_results.ipynb), pre-rendered from artifacts.

`mathematical_overview.ipynb` must remain unchanged. Initial SHA-256:
`eae6e00ce327334cca56755714d1965f21e0ff1dc4e8d3d14487eaa2b7ec8487`.

Plan 18 work already exists in the dirty worktree and is unrelated. Do not
revert, relocate, or incorporate its changes in this phase.

## Delegation

| Slice | Requested model/effort | Agent | State |
|---|---|---|---|
| 0A/B and bounded numerical integration review | `gpt-6-astra`, `xhigh` | Kierkegaard | Complete; closed |
| 0C model/language wiring and 0F reporting | `gpt-6.1-sol`, `xhigh` | Hubble | Complete; closed |
| 0D/E robot, storage, independent worker and control | `gpt-6.1-sol`, `xhigh` | Goodall | Complete; closed |
| Configuration, learner, acceptance, CLI and handoff | Coordinator | Main thread | Accepted |

Worker token/credit usage is not exposed by the current tool response and is
therefore unknown. Do not infer usage from output length.

## Environment Notes

- Use the user's existing `~/.venv/bin/python`; do not upgrade the model stack.
- GPU: RTX 4070, 12 GB. PyTorch 2.10.0+cu128 sees CUDA outside the sandbox;
  sandbox-only `is_available=False` is not a driver failure.
- Qwen defaults to a pinned download retained in the project's ignored
  `cache/numerical_experiment/picar/models/`, independent of working directory.
  `--model-path` is an optional explicit reuse override; `--offline` forbids
  downloads. The default Hub route is unit-tested with a mocked download.
  Actual GPU acceptance used explicitly supplied existing weights and their
  verified file checksums, without downloading a duplicate copy.
- Installed PyBullet 3.2.7, Gymnasium 1.4.0, imageio 2.38.1 and
  imageio-ffmpeg 0.8.1. PyBullet's local source build used two compiler workers;
  physics code was unchanged. Torch/Transformers/bitsandbytes were not upgraded.
  `pip check` was clean after installation. Full versions are in run manifests.
- Phase 0 artifacts belong under `cache/numerical_experiment/picar/`. No
  private external-drive path should enter source or tracked configs.

## Next-Phase Checklist

1. Read `plan19.md` Phase 0 and `NUMERICAL_CONTRACT.md`.
2. Await authorization for Phase 1. Its resource/task gates are not yet passed.
3. Use `preflight --cuda` and the real-model acceptance entry point before larger
   workloads. The candidate context-8/batch-4 envelope is unverified.
4. Establish physical mapping, scripted feasibility, teacher/evaluator skill,
   inference period, fallback rates, context and label eligibility before
   binding the real-time controller to scientific acquisition.
5. Calibrate FD strength by gradient ratios and trajectories, not the smoke
   default. Match retention in development before sealing fresh recipes.
6. Enforce cumulative campaign time/storage accounting in the stage scheduler,
   in addition to the implemented per-run quota and free-space admission.
   Pending/context bytes count; replay/sleep do not create new evidence.

## Acceptance Artifacts

Paths below are relative to `cache/numerical_experiment/picar/phase0/`:

- `acceptance-v3`: final immutable bundle. 280 tests passed;
  paired bootstrap/control/FD smoke and CUDA check passed. Total invocation
  was 52.52 s, including tests, loading and artifact work. Its summary is
  authoritative; child timing is subtracted from parent accounting to avoid
  double-counting in the report.
- `acceptance-v1` and `acceptance-v2`: earlier bundles retained. The final
  revision also executes notebook cells from the notebook's own directory.
- `qwen-acceptance-v1`: retained first diagnostic. Forward/update/Fisher worked,
  but reduced-precision top-block arithmetic did not support stable FD.
- `qwen-acceptance-v2`: corrected check. Across half/base/double epsilon,
  penalty max/min was 1.0472; gradient-norm ratio 1.0398; gradient cosines
  exceeded .9999. Both banks received gradients. Reserved memory was 6.715 GiB;
  inference .618 s, one update .517 s, eviction Fisher .179 s. The final bundle
  independently repeated the corrected check against its source snapshot.
- `visual-acceptance-v1`: synchronized fixed-seed ego/top-down video and PNGs,
  visually inspected. The top-down stream is never a learner input.

Qwen probes use one 64x48 image/history step, top-L=1, hidden size 16, and
processor limits 56x56--112x112. They do not admit the full production workload.
The CPU smoke uses accelerated physics and a fake teacher/zero potential.
These are software results, not robot competence or sample-efficiency results.

## Corrections Before Science

1. FP32 adapted Qwen suffix, rotary embeddings and final norm; frozen lower
   blocks remain BF16 and packed weights remain quantized. Small-output KL
   arithmetic uses FP64 to avoid cancellation near equality.
2. Each condition deep-copies bootstrap optimizer state. Fresh replica RNGs
   are distinct from bootstrap streams and paired between treatments; resumed
   streams retain their consumed state.
3. Zero-based record indices become observation counts for replay ages, giving
   the newest item age zero. Retired predecessors are masked in replay context.
4. Physics restore does not invent frame IDs. Completed artifacts also reject
   new child runs or frame stores beneath them.
5. Control mixtures accept float32 normalization roundoff but reject invalid
   distributions. No implicit steering-triggered forward motion remains.
6. Relative artifact paths resolve from the project, not the notebook's
   working directory. The resource inspector is named `preflight.py` to avoid
   shadowing Python's standard `profile` module during kernel startup.

Tests cover numerical identities, both-endpoint/both-bank gradients, frozen
roles, independent physics under blocked updates, delayed labels, exactly-once
retirement, and forced halts around action/label/optimizer/retirement commits.
No interrupted episode gets a fabricated zero return. Startup/missed-deadline
safe-neutral intervals are tagged, not fabricated Gaussian action labels.

## Entry Points And Scope

- Numerical contract: `NUMERICAL_CONTRACT.md`, `src/continual/picar_math.py`.
- Models and one-time recorded teacher KL: `models.py`, `teacher.py`.
- Likelihood/optimizer/FD/EWC integration: `learner.py`.
- Shared backend, independent clocked physics, physical client: `src/robotics/`.
- Maintained lightweight onboard service with fake-driver tests: `demo/picar/`.
- Real-time control/update ownership and bounded queues: `control.py`, `runtime.py`.
- Transactional artifacts/replay: `storage.py`, `schemas.py`.
- Acceptance trajectory, CLI and profiler: `engine.py`, `run.py`, `acceptance.py`,
  `preflight.py`, `qwen_acceptance.py`.
- Artifact-only statistics/notebook/video: `analysis.py`, `report.py`, `preview.py`.

The smoke runner deliberately rejects production configurations. Its short
deadlines, fake roles and accelerated physics must not become a research
recipe. `ControlSession` supplies the independent real-time acquisition and
update machinery for later stage runners; full scientific acquisition,
eligibility, calibration and study orchestration remain Phase 1 onward work.
Do not treat `configs/plan19.json` as a selected or resource-admitted recipe.

## Commands

Run from the repository root. Fresh-clone installation is in README; no other
checkout is required. Use new output names for changed code/configurations.

```bash
~/.venv/bin/python -m numerical_experiment.picar.run status \
  --output cache/numerical_experiment/picar/phase0

~/.venv/bin/python -m numerical_experiment.picar.run report \
  --output cache/numerical_experiment/picar/phase0

~/.venv/bin/python -m numerical_experiment.picar.preflight --cuda
```

The README acceptance command supports `--resume`. `run smoke` additionally
supports `--max-events` for recovery checks. `run stop --output <run-directory>`
requests a checkpoint-boundary stop; explicit smoke resume clears that request.
Reporting never unpickles checkpoints, trains, downloads, or repairs artifacts.
Fake-service tests bind localhost only and may need socket permission outside
a restrictive sandbox. No hardware adapter has been actuated or calibrated.
