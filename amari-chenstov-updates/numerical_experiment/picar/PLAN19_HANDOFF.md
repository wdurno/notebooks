# Plan 19 Handoff: Phase 3 Development

## Paused For Node Handover (2026-11-12, 21:56 UTC)

**Both tmux workers have exited cleanly and CUDA has no remaining compute
process.** The user requested the pause; do not resume until instructed.
STOP files exist in BOTH `phase3/development-v1` and
`phase3/development-retention-v1`. The latter stopped while waiting; it has
not taken over, sealed the source, initialized a model or acquired data.

Two controls are complete and immutable: ridge .1 (`pilot-e50a89383d05`,
627 captures) and .03 (`pilot-6ff65bb6993d`, 625 captures). Both went from
5/20 easy-A successes to 0/20 at midpoint/final, with 0/10 hard-B throughout.
The ridge-.3 control (`pilot-470d6557397f`) is checkpointed at **330/640
incoming intervals, midpoint evaluation**. Resume its saved evaluation and
trajectory; do not restart it or classify its missing endpoints as failures.

The source parent durable charge is **8,658.508928239 s (2.405 h)**; prior
study plus this charge is **29,378.133737569 s (8.161 h)**. About 3.595 h of
the shared six-hour Phase 3 ceiling remains, including final verification.
Idle node-borrowing time must not be charged. This is a remaining allowance,
not an estimate that every proposed candidate will fit.

Restart: intentionally remove BOTH STOP files, launch the original campaign
using the command below, and launch the conditional supervisor with `--wait`
using its command below. Supply the same optional model-path/offline overrides
from the source `orchestration/config.json`. Use separate tmux sessions as
before. The source must finish its third ridge candidate; the supervisor then
decides whether to leave the original running or checkpoint/seal it and take
over. After a takeover, never resume the original writer again.

Shutdown verification checked all seven completed unit seals, all 516 parent
journal entries and all 1,035 active-pilot journal entries, including the
required recovery snapshots. All 69 frozen historical scientific sources and
the mathematical overview remain unchanged. The notebook was refreshed by
the runner's clean-stop handler. Independent supervisor/reporting acceptance
passed **59 tests in 49.26 s**; its agent also passed 78 combined runtime,
analysis and probe checks. All coding agents are closed. No FD development,
Phase 4 or physical robot operation has started.

## Phase 3 Authorized (2026-11-12)

The user authorized Phase 3 only. The campaign started at 19:39 UTC in tmux
session `plan19-phase3`, at
`cache/numerical_experiment/picar/phase3/development-v1`. Acceptance passed
109 tests in 159.47 s. CUDA initialization completed and is sealed at
`units/bootstrap-715cd4052b9c` (850.57 s, fresh KL/NLL/Fisher). The recorded
temperature is 2.8375266913896, action variance 1. All 247 original actions
were forward; no new teacher data were acquired. Scratch calibration selected
LR .003, two optimizer steps and exploration SD 8. The first ridge-.1 pilot
is complete at `units/pilot-e50a89383d05`: 627 captures, no pending updates;
easy-A 5/20 initially, 0/20 at midpoint/final; hard-B 0/10 throughout.
These are NEW development layouts, not Phase 2's 7/20 panel. The lower-ridge
.03 candidate is now running at `units/pilot-6ff65bb6993d`; .3 follows.
The new independent package is
`numerical_experiment/picar_phase3`; the scientific sources from Phases 1/2
and the mathematical overview remain frozen. Two reused Sol sidecars own
the development runtime/tests and reporting/tests. The coordinator owns
calibration, selection, budget, scheduler and GPU execution. Both sidecars
have finished and are closed. Runtime tests passed 30 checks and 42 earlier
regressions; the coordinator independently passed 47 runtime/probe checks.
Seven minutes of conservative pre-execution CPU validation and five minutes
of closeout allowance count inside the six-hour phase ceiling.

### Conditional Retention Repair Armed

After the first pilot's 25-point retention loss and zero hard-task gain, the
coordinator is preparing a separate `picar_phase3_retention` supervisor. The
original three ridge candidates CONTINUE UNCHANGED. Independent retention and
reporting acceptance passed 59 tests in 49.26 s. Before the user-requested
pause, the frozen supervisor waited in tmux `plan19-phase3-retention`
(launched about 21:28 UTC), logging to
`phase3/development-retention-v1/run.log` under the PiCar cache root.
It will take over only after all three are complete and none
provides a retained hard-task gain; otherwise the original schedule continues.
Do not resume the original past that gate without reviewing this amendment.

The refined common repair is LR .001, steps 1, pi .01, exploration SD 4 and
action variance 16 (same tau 2.8375266913896/ridge selected from original
controls). This replaces the unexecuted higher-plasticity repair. It needs
fresh KL/NLL/Fisher initialization from the recorded demonstrations because
Sigma_A changes; old Fisher/panels cannot be transplanted. No teacher queries
or retired-data online replay are added. The same six-hour Phase 3 budget
carries across the source handover; waiting overlaps existing work and is
not charged twice. No Phase 4 is authorized. See the appended plan refinement.

Supervisor resume command (same model-path/offline overrides as the source):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase3_retention.campaign \
  --output cache/numerical_experiment/picar/phase3/development-retention-v1 \
  --source-campaign cache/numerical_experiment/picar/phase3/development-v1 \
  --resume --wait
```

Before the three-control gate, resume both the original runner and waiting
supervisor. After the source orchestration is sealed as
`superseded_after_ridge_screen`, resume ONLY the supervisor. Never remove the
source STOP after takeover. New-run STOP/SIGTERM preserves the current repair
checkpoint. The waiting supervisor performs cheap discovery and validates all
seals before taking ownership; idle waiting is not charged twice.

The appended Phase 3 execution recipe records the exact calibration and
candidate sequence. Use feature plus projection RMS, a fresh initial Fisher
for any changed likelihood scale, three ridge candidates and up to two
documented repair rounds. Attempt two development seeds and report any budget
omissions. Initial/midpoint/final fixed-layout evaluation is descriptive, not
confirmation or a sample-savings estimate. The maximum Phase 3 charge is six
hours inside the cumulative 20-hour target; prior charge is 20,719.62480906 s
(5.76 h). No automatic Phase 4 or physical actuation is authorized.

The standalone resume command, once acceptance passes, is:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase3.campaign \
  --output cache/numerical_experiment/picar/phase3/development-v1 \
  --teacher-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --prior-review cache/numerical_experiment/picar/phase2/readiness-rms-v1/reviews/closeout-v1 \
  --resume
```

Supply the same optional model-path/offline overrides recorded in the resolved
configuration. Default loading downloads the pinned model into project cache.
Never run a duplicate writer. STOP/SIGTERM checkpoint; remove STOP only before
intentional resume. All old completed runs remain immutable. The runtime
source/config fingerprint is frozen when the campaign starts; report-only
edits may continue. Check `orchestration/progress.json` and `run.log` for state.

## Phase 2 Complete (2026-11-12)

**Latest state, 17:56 UTC: complete and stopped for review. GPU is free; no
tmux job remains.** Both original baseline pilots and both normalization
rescues retained 7/20 easy and 0/10 hard successes. Captures were 620/619 for
the baselines, 635 for feature RMS and 623 for feature plus projection RMS.
Each finished with zero pending updates. The original orchestration is sealed
as `superseded_after_baselines`; the rescue orchestration is sealed as
`ready_for_checkin` with 12 completed child units. Completion is NOT a
readiness pass: the operational hard-panel improvement guide was not met.

Feature RMS alone left the initialization projection saturated. Adding
projection RMS produced zero saturated coordinates on the archived probes,
median local tanh slope .462 and maximum instruction logit difference .051
(originally about 4.77e-7). This repaired measured local sensitivity, not
instruction following or navigation learning. No FD comparison was run.

Phase 2 campaigns charged 12,290 seconds (3.41 h). Including a five-minute
conservative final-verification allowance, Phase 2 totals 3.50 h, inside its
shared five-hour cap. The conservative cumulative study charge is 5.76 h, leaving 14.24 h
of the 20 h target before additional development. Measured processing was
2.92 s/captured interval. Four pairs of 1,200 intervals/condition would take
7.8--11.7 h processing plus 1.41 h evaluation under hypothetical FD cost
multipliers 1--2; six pairs of 2,400 would take 23.4--35.1 h plus 2.12 h.
FD overhead is UNMEASURED and later development/closeout are extra. These are
conditional forecasts, not admitted studies. First resolve common-controller
learning. No Phase 3, additional rescue or physical robot work is authorized.

Closeout passed 124 focused Phase 2 tests and 51 final reporting tests. All
23 pre-closeout seals verified, and all ten notebook code cells executed
without importing Torch. Completed `--resume` verified unchanged seals without
Torch or GPU work. All 63 scientific sources match the archived fingerprint;
the mathematical overview is unchanged. Final outcome and sensitivity plots
were visually inspected. The separately sealed `reviews/closeout-v1` records
the verification and extra allowance without mutating the campaign forecast.
The complete Phase 2 artifact tree occupies about 854 MB.
The final report refresh exposed a metadata-only closeout review being treated
as a scientific experiment version. A reporting-only classification fix and
regression test resolved it; no sealed artifact or scientific source changed.
All ten notebook cells and all 24 seals (including closeout) verified afterward.

### Completed Execution Details And Recovery History

The user authorized Phase 2 only, ending at the readiness/forecast check-in.
Implementation is in `numerical_experiment/picar_phase2`, separate from the
frozen Phase 1 numerical sources. See the appended Phase 2 execution recipe
in `plan19.md`: compatible initial asset reuse, 20 easy/10 hard fixed layouts,
two 640-interval baseline pilots, at most two bounded common rescue recipes,
five active compute hours maximum. No FD comparison, Phase 3 or physical
operation is authorized. Preserve all prior artifacts and the theory notebook.

Two bounded Sol 6.1 Extra High sidecars implement the evaluated runtime/tests
and artifact-only reporting/tests. The coordinator owns configuration,
scientific integration, scheduler, resource limits and GPU execution. Their
write sets do not overlap. This records the plan's explicit routing settings;
token/credit usage is unavailable, not inferred.

Both sidecars are now complete/closed. Runtime recovery tests and reporting
acceptance passed; integration additionally fixed the language unit's full
checkpoint publication and action-variance compatibility checks before
freezing sources. The original campaign ran in tmux session `plan19-phase2`:
`cache/numerical_experiment/picar/phase2/readiness-v1`. Its `run.log` captures
the invocation, and `orchestration/progress.json` tracks the current stage.
Five minutes of conservative pre-execution validation reserve are included
in the charged phase allowance. The runtime source archive and resolved
recipe are immutable child artifacts; report-only improvements are excluded
from the scientific fingerprint.

Resume command from the checkout (also supply the same optional model-path
and offline overrides recorded in `orchestration/config.json`):

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase2.campaign \
  --output cache/numerical_experiment/picar/phase2/readiness-v1 \
  --bootstrap-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --resume
```

Do not launch a second writer while tmux is running. A root STOP file or
SIGTERM requests a checkpointed halt; remove STOP before intentional resume.
The initial asset has passed the compatibility/disjointness audit. No
repeated KL or unrecorded use of retired data is allowed. The original
52 scientific sources and mathematical overview retain their hashes.

### Targeted Conditioning Rescue Handover (2026-11-12)

The first 640-interval baseline completed at its episode boundary after 620
captures: easy-A remained 7/20, hard-B remained 0/10. The second planned seed
finished after 619 captures with the same outcomes. Initial frozen teacher readiness was also 7/20 easy-A; both
teacher and student succeeded on 7/11 clear and 0/9 blocked layouts.

The language probe changes Qwen features but barely changes action logits
(maximum difference about 4.77e-7). An artifact-only CPU reconstruction found
all projected coordinates saturated at abs(tanh(z)) > .99; median absolute
preactivation was 11.59 and median local tanh derivative zero in float32.
The sealed diagnostic is `readiness-v1/reviews/feature-health-v1`. This is a
concrete conditioning concern, not isolated causal proof or an STM result.

The appended plan amendment replaces the two UNEXECUTED ridge/floor rescues
with feature RMS normalization, then optionally feature plus projection RMS
normalization. All other scientific settings and parameter counts stay fixed.
Each new model reconstructs the 247 original demonstrations and receives fresh
weights, optimizer state, one-time KL, likelihood fit and Fisher. No historical
model-dependent Fisher is transplanted into a different forward model.

The supervisor ran in tmux session `plan19-phase2-rms` and is now complete:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar_phase2_rescue.campaign \
  --output cache/numerical_experiment/picar/phase2/readiness-rms-v1 \
  --source-campaign cache/numerical_experiment/picar/phase2/readiness-v1 \
  --teacher-source cache/numerical_experiment/picar/phase1/fix3-v2/units/bootstrap-5f42cbe3df8a \
  --resume --wait
```

Use the same optional model-path/offline overrides as the original command.
Its log is `readiness-rms-v1/run.log`. It waits for BOTH baseline seals, stops
the original writer before the old rescues, and seals that orchestration as
`superseded_after_baselines`. It then owns the remaining Phase 2 work. Both
versions share ONE five-hour active-compute ceiling; this is not a fresh
five-hour allowance. Waiting overlaps the original campaign and is not charged
twice. Neither campaign automatically starts Phase 3.

The following describes interruption recovery during execution; both runs
are now complete and must not be reopened for additional learning. A completed
`--resume` verifies and returns without importing Torch or starting new work.
After a reboot before both baselines finish, resume the original command above
AND this waiting supervisor. After the original orchestration is sealed,
resume only the rescue command. Never launch duplicate writers. Remove the
appropriate root STOP file only for intentional resumption. Both numerical
source sets are frozen now; use a new version for any material correction.
Only reporting-only changes can be made without changing their fingerprints.
Both implementation sidecars have finished and are closed; 62 focused rescue,
health and manager tests passed before the supervisor was launched.

## Previous Check-In (2026-11-11)

**Fix 3 is complete and stopped for review. GPU is free; no worker remains.**
`cache/numerical_experiment/picar/phase1/fix3-v2/orchestration/COMPLETED`
seals 23 child units. Both real student pilots ran completely. Each retained
2/2 easy and 2/2 middle successes; hard success remained 0/2 before and after.
This supplies a useful approach bootstrap, not demonstrated learning of the
new obstacle task. The common learner must acquire the changed skill before
an STM comparison is informative. Phases 2--5 and physical operation remain
pending. No further parameter or prompt sweep was started.

The notebook has the complete chronological Fix 3 section, compact success
plot, reward contact sheet, before/after clips and a concise conclusion.
Final reporting fixes prevent a null live timestamp on the old stopped
attempt from superseding the completed campaign, verify pairing using the
recorded seed/geometry, and keep legends off the data.

Closeout reran all 19 focused reporting tests and executed all nine notebook
code cells without importing Torch. The final two-student success figure was
visually checked. Completed resume verifies/skips without importing Torch;
the GPU has no remaining compute process. The mathematical overview retains
its original SHA-256.

Scientific Python sources remain frozen and match the campaign fingerprint.
`units/provenance-*/sources.tar.gz` preserves the runtime code. Keep unrelated
dirty Plan 18 files and `.gitignore` changes. The mathematical overview is
unchanged. A completed resume verifies/skips without importing Torch:

```bash
~/.venv/bin/python -u -m numerical_experiment.picar.phase1_fix3 \
  --output cache/numerical_experiment/picar/phase1/fix3-v2 --resume
```

Supply the same optional `--model-path` and `--offline` values recorded in
`orchestration/config.json`. A root `STOP` file requests a checkpointed halt;
remove it before an intentional resume. Pending episodes retain their state
and never receive fabricated failure returns. The CLI ends after the two
bounded Replay/EWC student pilots and the task/resource check-in, not the
full STM comparison or physical operation.

- The stopped `fix3-v1` attempt and its exact source snapshot are retained:
  combined proximity/centering judgments saturated almost all visible scores
  at 10. It charged 264.7 seconds and trained no student.
- The permitted targeted repair separates the exact Fix 2 contact/proximity
  query from a horizontal-image-coordinate query. Camera pan discounts the
  half-point centering addition. Development approach ordering is 4/6 and
  centering ordering 4/6; validation approach ordering is 4/6, centering 2/6.
  The signal is useful but coarse, with off-axis detection errors. No further
  prompt search is authorized by this amendment.
- Software acceptance passed 705 tests. A preceding incomplete software unit
  retains `pytest-initial-report-failure.txt` for a reporting-string assertion;
  it was corrected without changing scientific settings.
- The geometry audit has 58.2% clear early layouts and 100% blocked late
  layouts. Privileged diagnostic pursuit reached contact in 12/12 checked
  level/gain cases; none of those controls enter demonstrations.
- Return-bin calibration selected 161 bins to preserve more of the small
  centering signal. Model context 4, hidden width 64, top-L 1, LTM/STM ranks
  4/1, replay capacity 8, new/replay batch 1/1, FD strength zero in this pilot.
- Prior repairs/recognition plus the stopped attempt were 37.0 minutes at
  v2 admission. V2 has its own two-hour ceiling inside the cumulative
  three-hour repair envelope and 20-hour overall study ledger.
- Actual CUDA recovery acceptance halted one-time distillation at step 95
  after 292 captured transitions (44 episodes, 247 executed teacher examples,
  16 successes). A fresh GPU process restored parameters, KL Adam tensors and
  named RNG states exactly; the check is in `fix3-v2/gpu-recovery-check.json`.
  The STOP flag was removed and the same immutable configuration resumed.
- Corrected campaign compute was 80.5 min; both Fix 3 attempts total 84.9 min.
  Cumulative recorded repairs/recognition were 117.5 min; a conservative
  further 5 min reserve covers development/verification work outside those
  invocations. Including earlier acceptance/profiling, the conservative
  overall study charge is about 2.22 h. Budget caps remain unchanged.
- The bootstrap took 33.2 min across invocations; do not use only its final
  invocation's 12.2 min as its cost. Pilots took 17.2 and 17.6 min, captured
  305/300 intervals, and completed 276 updates each (255 online, 21 stopped).
  Ordinary non-startup misses were 2/278 and 1/277. Final replay has 8 entries,
  with no pending updates and exactly-once retirement accounting.
- Review artifacts: `review/integrity.json`, `review/video-checks.json`, and
  `review/resource-forecast.json`. Audit verified 32 completed units across
  attempts, 1,898 frame blobs, 203 media files and two source archives with
  unchanged completion markers. All six clips decode and show motion.
- Conditional Replay/EWC processing throughput is 2.95 s/captured interval:
  four pairs x 1,200/arm would take about 7.9 h processing; six x 2,400 about
  23.6 h. These exclude additional FD cost, development, evaluation and
  closeout, and do not admit a study whose new-task learning gate is unmet.

## Reader-Requested Hard-Task Recording (2026-11-12)

The old `seed-8-B/scene.mp4` gallery clips are privileged scripted feasibility
runs, not learned-student runs. The latest Fix 1 version records success at
13.5 s. Original Fix 3 final hard evaluations instead all collided at
4.8--5.7 s; only the easy evaluation panel originally had video capture.

At the user's request, `numerical_experiment.picar_evaluation_video` recorded
a separate no-learning evaluation of student 193701's final checkpoint on
hard layout 195803. Output: `diagnostics/fix3-final-hard-video-v1` (completed).
The synchronized camera/top-down MP4 reproduced a collision at 7.23 s versus
5.30 s in the original trial. This is explicitly supplemental footage with
fresh real-time scheduling, not recovery of the original trial's footage.
All learner parameters, optimizer, Fisher, anchors, RNGs and counters were
checked unchanged. Six focused tests pass, the original campaign numerical
sources remain unchanged, and the GPU is free. The successful invocation took
22.8 s; a preceding sandbox-restricted CUDA load failed before any rollout.
Original results and completion seals are untouched; no main-study result
or treatment comparison was added. This separate CLI stays outside the
frozen campaign's source tree and accepts an optional model path as usual.

## Historical Fix 2 Check-In

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
