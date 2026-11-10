# Plan 19 Handoff: Phase 1 Check-In

## Status

**Fix 2 complete, stopped at the task/resource check-in (2026-11-09).**
`phase1/fix2-v2/orchestration/COMPLETED` seals the corrected campaign. All six
software deliverables are present; integrated acceptance passed **637 tests**.
The notebook is artifact-only and includes the full Fix 2 diagnostic section
and conclusion. No Phase 2--5, physical motion or scientific STM comparison
has started. No worker needs to remain running. Preserve the already-dirty
worktree; do not revert earlier implementations or artifacts.

- Contact-aligned generated scores are valid and separate absence from close
  approach, but strict far/intermediate/close ordering passed only 5/12
  development and 8/12 held-out pairs (80% guide unmet).
- Ported Qwen: 3/6 unobstructed successes. Longer grounded prompt: 0/6.
  Full obstacle task: 0/12 across both interfaces, all collisions. Teacher
  rollouts are accelerated diagnostics, not real-time-control admission.
  Fresh 20-layout validation was not admitted.
- Each real-time student/control pilot completed three episodes, 44 updates
  (28 online, 16 stopped drain/sleep), 36 retirements and zero non-startup
  fallbacks. Peak label backlog 17, final zero. Students were untrained;
  all episodes collided. These are execution tests, not learning results.
- Return/bin gradient checks pass with 41 bins. Retired-data sleep is
  implemented/default off and crash-tested. Pi hardware/speech interfaces
  have injected-input acceptance; real calibration/STT/actuation untested.
- Corrected campaign charged 17.9 minutes; both Fix 2 attempts 21.3 minutes;
  all Fix 1+2 repairs 31.1 minutes, below the three-hour repair cap.
- Updated conditional study forecasts: **21.4--30.8 h** nominal, or
  **11.6--17.2 h** at the smaller permitted scope. Notebook totals explicitly
  add 10.6 minutes of earlier acceptance/profiling omitted by the sealed
  forecast. Fit timings include cooperative waits; these are conditional
  serial forecasts, not readiness or compute approval.

Next decision: settle teacher/reward readiness and a study scope with the
user. Do not automatically start Phase 2. Completed `phase1_fix2 --resume`
must use the same configuration/model/source identity and only verifies/skips
the campaign. New numerical/role changes require a new campaign version.
Exact source snapshots are retained in both attempts under
`source-snapshots/verified-source-v1`; the completed corrected campaign also
has its final provenance source archive. Rebuild the notebook with the
artifact-only `picar.run report` command in README.

Closeout verified 60 completed units, three source archives, unchanged
completion markers and a completed-resume skip without importing Torch.
All eight notebook code cells execute artifact-only. Focused reporting tests
pass (24), all four preselected videos decode with motion, and the GPU has
no remaining compute process. Fix 1's section and the theory notebook retain
their original hashes. Report-only grouping/figure-label changes are outside
the frozen scientific source identity; all 45 scientific source files match.

**Recognition follow-up (2026-11-10):** at the user's request, the new
`visual_questions` entry point queried four saved initial images plus a
ball-absent control, six independent questions each. The separate immutable
run `diagnostics/visual-questions-v1` contains all 30 raw answers and a sealed
CSV (89.3 active seconds). Spontaneous red-sphere recognition passed on all
four positive images. Direct presence/location questions rejected the absent
control, while distance questions hallucinated its position. Apparent depth
relationships and metric distances remain unreliable. No main experiment
scope changed. Three focused unit tests pass; saved frames/CSV/seals verify,
and completed resume skips without importing Torch. The GPU is free.

**First Fix 2 attempt retained:** `phase1/fix2-v1` checkpointed a requested
stop after 201.8 active seconds, with 20 complete units. Software acceptance
passed 635 tests. Qwen often generated valid action/reward JSON inside a
single Markdown fence, which the initial strict parser rejected. Those
teacher rows are interface failures, not measured navigation failures.
The new parser accepts only bare JSON or one complete JSON fence, keeping
keys/numeric validation strict. The timing pilot now masks retired context
predecessors. Both corrections receive a new source/configuration version
`phase1-fix2-v2` under `phase1/fix2-v2`; never overwrite v1 artifacts.
All three Sol sidecars have finished and closed. New resource forecasting
uses measured generated-role, fit, Fisher and durable journal costs.

### Historical Planning And Prior Acceptance Notes

The following notes describe earlier states and are superseded by the
completed Fix 2 status above; they are retained for continuity.

**Planning update (2026-11-09):** the user requested a further amendment
as a separate **Phase 1 Fix 2** for review, with Sol to execute subsequently.
Read "Phase 1 Fix 2: Restore Dense Visual Rewards And The Robot Workflow"
in `plan19.md` before making changes. It is **planned, not implemented or
executed**. This turn changes documentation only; no experiment was launched.
The earlier unexecuted draft was nested under Fix 1; the user corrected that
numbering. Fix 1's specification, execution history and artifact names are
preserved. Append future material amendments rather than rewriting that work.

The draft restores generated 0--10 RGB reward scores, proposes explicit
`ball-contact-v1` success semantics and accumulated `dense-visual-v1` rewards,
and tests simple approach before the full obstacle task. It requires causal
conversation/coaching context, a public task countdown, return bounds/bin
resolution checks, and sustained queue/timing accounting. Historical safe-stop
and potential-shaping versions remain intact. The normalized joint likelihood,
STM FD and diagonal eviction-only EWC mathematics remain the same family;
new reward labels/scales and initialization need their own version.

All related corrections are explicit implementation deliverables. Item 5
implements default-off retired-data sleep with supplemental-rehearsal
accounting and exact resume tests; primary scientific conditions keep it off.
Item 6 implements the actual actuator/camera adapter and typed/speech event
bridge with injected hardware and recorded-input tests. Real calibration,
microphone use and physical actuation are not part of this acceptance run.

Expected additional active compute is 0.9--2.4 h, with 4--9 h of implementation
estimated separately. Keep Fix 2 charges distinct, counting them together
with Fix 1's 9.8 minutes against the existing three-hour repair envelope and
overall study budget. Proposed output family:
`phase1/fix2-v1`. There is no new runnable command yet; the existing
`phase1_fix --output .../fix1-v2 --resume` only verifies the old campaign.
After implementation/readiness, stop for the resource check-in; no automatic
Phase 2--5 or physical motion. The added software features do not authorize
additional scientific treatment arms or real-world deployment tests.

**Completed prior work:** Phase 1 Fix 1 is implemented and its two bounded campaigns are
complete: `phase1/fix1-v1` and `phase1/fix1-v2`. Native/split parity and the
revised stop/left/right instruction checks pass. Context-4 cooperative pilots
had zero fallbacks at .75/1/1.25 s; scripted feasibility passed 10/10 at .75/1 s
but only 8/10 at 1.25 s. These are two-update burst diagnostics, not sustained
online-load admission. Teacher navigation and the visual potential remain
unready. No fresh teacher validation or Phase 2--5 learning was launched.

Repair compute was 590.4 recorded seconds across both passes. A Tk backend
abort during v2 was recovered through `--resume` with Agg; completed units
were preserved/skipped and the incomplete short timing pilot retried. The
new notebook section and Plan 19 execution record describe both successes and
remaining blockers. The separate Fix 2 draft above addresses that scope decision;
its execution still awaits the user's next instruction.

New code: `role_repair.py` (versioned grounding, causal multi-image input,
FP32 answer readout, native reference/generation); `phase1_fix*.py` (bounded
resumable diagnostics and artifact-only reporting). `MemoryLearner.fit`
accepts a graph-safe `cooperate` callback; `ControlSession.update(cooperative=True)`
uses it and skips expired work before GPU inference. Parameters remain
unchanged at every yield; failed updates preserve actor exploration RNG
consumption. Multi-step events stop yielding before their first mutation.

`RunStore(..., checkpoint_policy="journal-v1")` is opt-in; default legacy
behavior stays unchanged. It retains append-only events and current/previous
full states plus explicit milestones. Journal resume must select the same
policy. Only superseded state payloads in incomplete runs are pruned after
durable CURRENT publication. Never prune completed data. The v2 provenance
unit includes a checksummed source archive and recovery notes.

Phase 0 acceptance passed, and the user subsequently authorized Phase 1 only,
followed by a resource check-in. Phase 1 diagnostics are finished on
2026-11-09, **but readiness gates failed**. The original Phase 1 suite passed
361 tests; the final Fix 1 regression suite passes **442 tests**. No Phase 2--5 learning or physical motor operations
were started. Do not start them automatically. All Phase 1 processes and
coding agents have exited; the GPU is free.

The measured whole-study ranges are **38.9--55.7 h** nominal (six pairs,
2,400 new transitions/arm), or **19.0--28.0 h** at the smaller permitted size
(four pairs, 1,200/arm). Neither is safely admitted under 20 hours. These
conditional engineering forecasts include evaluation/sleep/checkpoint I/O;
teacher rescue or more optimizer steps can increase them.

Report: [plan19_results.ipynb](plan19_results.ipynb), pre-rendered from artifacts.

`mathematical_overview.ipynb` must remain unchanged. Initial SHA-256:
`eae6e00ce327334cca56755714d1965f21e0ff1dc4e8d3d14487eaa2b7ec8487`.

The worktree was clean when this Phase 1 turn began. Current uncommitted
changes are the Phase 1 implementation, exact model optimizations, tests,
notebook and notes. Do not revert unrelated later user changes.

## Phase 1 Findings And Next Decision

- Main immutable campaign:
  `cache/numerical_experiment/picar/phase1/resource-v2`. Its `orchestration`
  directory is complete; separate `units/` retain nine CUDA probes, scene
  audit, role/control checks and two scope forecasts. Re-running this completed
  campaign verifies/skips it; changed diagnostics require new units/versions.
  Main campaign execution was 364 s, plus earlier probes and engineering/tests.
- Memory shortlist only: top-L=1, hidden 64, LTM/STM ranks 4/1, context 8,
  replay batch 1, one score draw and one FD probe, 160x120 input. Peak reserved
  7.11 GiB, headroom 3.85 GiB. Top-2, batch-4 and context-16 exhausted VRAM;
  top-4/batch-8 were skipped. This does not establish student competence.
- Cold inference .790 s, control fit 1.714 s, FD fit 1.980 s, full-context
  eviction Fisher .866 s, frozen roles ~.17 s each. One learner checkpoint
  is 8.22 MB; a durable transaction measured .056 s. Draw counts 2/4 and
  two FD probes were measured, not selected for efficacy.
- Exact opt-in frozen-prefix CPU memoization (two entries/256 MB bound)
  reuses only frozen computations within events; all trainable suffix and
  FD-endpoint gradients remain live. Probes clear caches before every event
  and use distinct new/replay inputs. Frozen role extraction now avoids
  computing discarded earlier RGB rows while preserving the final row's
  entire recorded text/action context. Equivalence and invalidation tests
  pass; initial uncached `resource-v1` remains retained.
- Privileged scripted controller passed 10/10 endpoint/mirrored layouts and
  camera/action mapping checks. Its synchronized clips were visually checked.
  It is never a source of unreported primary teacher demonstrations.
- Frozen Qwen: 0/2 A and 0/2 B teacher rollouts succeeded (all collided).
  Completion potential ranked collisions above stopped success. These small
  diagnostics identify an inadequate current role path, not a precise estimate
  of Qwen's general capability. Bootstrap must not claim retention of an
  unlearned skill. Next development needs bounded role/prompt/observation
  repairs before substantive learning, as allowed by the plan.
- Real-time worker reached 20 simulated seconds/4,800 physics ticks while
  two ~2.1 s updates ran. At a 1.25 s control period, 6/17 non-startup intervals
  used safe fallback (35.3%). Worker wall time was 22.68 s. This is not admission
  for continuous one-update-per-acquisition operation; improve/profile the
  scheduling envelope without pausing physics or fabricating policy labels.
- Storage: raw training records/frames project to 13.5 / 4.7 GB nominal/small,
  but every-full-snapshot retention projects to 1.55 TB / 393 GB for fresh
  training alone. A bounded checkpoint/event-journal policy and explicit
  archive root are needed before large work. Do not silently use the drive
  or delete anything under an existing COMPLETED marker. Current Phase 1
  artifacts occupy about 95 MB.
- No physical measurements supplied. No sim-to-real validation claimed.

Probes use model initialization seed 191101, simulated profile history seed
190101 and named learner streams from `resolved(recipe).seeds(0)`; source and
model-file checksums are in immutable unit provenance. Role rollouts use
seeds 4/5 in both endpoint domains; the live worker uses 191107.

The next conversation should settle task readiness repairs and the acceptable
compute envelope, not automatically launch a 39--56 h run. The smoke FD
coefficient remains uncalibrated for scientific use. Full production-stage
orchestration still belongs to later work.

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
2. Phase 1 authorization has been used; await the requested check-in before
   later learning. Its resource shortlist fits but task/deadline gates failed.
3. Use `preflight --cuda` after node changes. Context-8/batch-1 is profiled;
   context-8/batch-4 exhausted VRAM and is not an admitted recipe.
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
