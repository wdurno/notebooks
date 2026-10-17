# Plan 14 Execution Handoff

Updated October 16, 2026, at 17:10 PDT after final verification.

## Complete: Node Available

**Plan 14 is complete, including final notebook reporting.** There is no
remaining planned compute. Do not restart the main experiment or add replicas
unless the user requests a new, separately specified extension. All 2,304 distinct final
trajectories are sealed: 32 paired replicas for each of 18 distinct policies
in each of the four separate conditions (576 trajectories per condition).
All 72 policy/condition cells have exactly 32 replicas, without missing pairs.
There are 5,736 sealed scientific units overall; no Phase 5 checkpoint remains
in staging. Main worker session 34866 / PID 4867 exited **0**, and disk
supervisor session 5168 exited **0**. Host checks at 16:47:55 PDT verified
the lock free, worker absent and no GPU compute processes. Final checks at
**17:10:00 PDT** repeated these checks after reporting and tests: all seven
phases are completed, all 31 frozen scientific hashes still match, the driver
hash is unchanged, and the sealed inventory is unchanged. Free disk is
**47.16 GiB**, after final unit-test temporary outputs. The node is available.

The sealed Phase 6 closeout is:

```text
phase6/closeout__0__shared__None__6042ea5185c03592c608/
```

Its required-file checksums and the completed driver invocation checksums
were independently verified after normal exit. The driver invocation is
sealed at `execution/report_resume/1792159237589724691-4867-39e98a48242a/`:
`study_complete=true`, unchanged scientific sources, 157 duplicate forced
refreshes coalesced and 57 reports rendered, elapsed invocation 35,203.03 s
(9.78 h, including startup/replay and reporting). Its final sealed-unit-ID
inventory SHA256 is
`b00176266999ba1edad8f1ed352fd8217b7696464558160a6cced7e383e06c1a`.
Older interrupted invocation records and obsolete Phase 0 staging remain
untouched; their presence does not mean the frozen final cohort is incomplete.

**Final reporting completed:** the reviewed supplemental CLI was
launched only after the main writer and supervisor exited. Native session
**18622** (`mnist_experiment.akaike_score_gate_reporting`) exited **0** after
its full read-only integrity scan and four Phase 3 figure insertions. Its
verified host PID 18466 is absent. It used sandbox namespace PID **2**; the
lock file still contains that stale namespace PID but its lock is free.
**Never signal host PID 2** or infer liveness from this stale text.

The final notebook is `mnist_experiment/akaike_score_gate_results.ipynb`:
**29 cells, 164 PNG figure outputs, zero error outputs**. Phases 0--6 are in
order; the four tagged supplementary plots lie within Phase 3, before
Phase 4. They were exported and visually inspected, with nonblank PNG/pixel
checks. A new artifact-grounded **Conclusion** follows the Phase 6 heading,
distinguishing selected primary effects, secondary factorial evidence,
compute/retention tradeoffs, and unresolved-fit limitations. All four primary
table means and intervals were checked directly against sealed closeout data.
Cell IDs, JSON structure, Python source parsing and ordering checks passed.
The optional `nbformat` package is absent in the venv; no library-level
nbformat validation or notebook-kernel execution was performed.

Final notebook SHA256:
`6e20077d45920fc87da6b0f1ff696fb7b718a7a5169eb849cca271cc7fa5c6cd`.
Image-review manifest:
`cache/mnist_experiment/akaike_score_gate/closeout_report_review_20261017T000729352806Z/manifest.json`
(directory timestamp is UTC). The complete unit suite passed again:
**411 passed in 85.77 seconds**. Main worker, disk supervisor, reporting CLI,
and final test sessions have all exited successfully. No frozen scientific
source, setting, stopping rule, or completed scientific artifact was changed
to produce the final report. No top-level theory primer or git commit was made.

Do not run the original notebook renderer afterward: it removes both the
supplement and manually added conclusion. The separate supplemental CLI is
idempotent and preserves unrelated cells, but there is no need to rerun it
on this completed notebook. Future analysis should load sealed artifacts;
new experiments require their own specification and immutable run identity.

### Fresh-Cohort Findings

Primary comparisons are the development-selected adaptive policy against its
frozen fixed-depth control, not an after-the-fact choice of the best adaptive
cell. Differences below are **ASG minus fixed mean NLL-AUC**; positive favors
the fixed control. These are nominal condition-specific paired 95% bootstrap
intervals (32 replicas), not simultaneous coverage or significance stopping.

| Condition | ASG / Fixed | Mean Difference | 95% Interval |
| --- | --- | ---: | --- |
| Linear rotation | `asg_lr_diag` / `depth4_lr` | +0.05657 | [+0.04648, +0.06825] |
| Sigmoid rotation | `asg_lr_diag` / `depth4_lr` | +0.06586 | [+0.05578, +0.07696] |
| Mixture m18, expected 9.9 nines | `asg_lr_diag` / `depth1_diag` | +0.00468 | [-0.00348, +0.01176] |
| Mixture m182, expected 100.1 nines | `asg_diag_lr` / `depth5_diag` | +0.02976 | [+0.01447, +0.04782] |

The selected policies did not show primary NLL superiority: three comparisons
favor fixed controls, and the small-mixture comparison is inconclusive.
There are meaningful tradeoffs: selected rotation ASG uses about 55 active
parameters vs 239 and 20 s online vs 96 s, with stronger old-data retention,
but worse current NLL. The larger-mixture selected ASG uses 86.4 vs 461
parameters, yet online times are similar (10.60 vs 10.67 s). Do not confuse
parameter savings with scalable or end-to-end compute savings.

There is a separate predeclared factorial signal: low-rank current gate
curvature reduces rotation NLL-AUC averaged across historical EWC choices
(linear -0.04221, CI [-0.05333, -0.03269]; sigmoid -0.05239,
CI [-0.06217, -0.04353]), with a substantial interaction with historical EWC.
Do not claim every adaptive variant loses in current NLL: `asg_lr_lr` means
are 0.54477 / 0.59361 for linear/sigmoid, below fixed-depth-4's 0.56284 /
0.61875. However, its mean retention changes 0.02809 / 0.03041 exceed the
frozen 0.02 bound; this is not an unqualified admissible winner. These cell
means are descriptive, not replacements for the selected primary policies.
All accepted unresolved fits remain included. Many deeper mixture controls
have unresolved fits at nearly every arrival, so the results are operational
policy comparisons, not certified MLE or isolated statistical-dimension laws.

## October 16 Execution (Historical, 07:00 PDT)

The user explicitly requested resuming after interrupting the preceding
turn. Actual host checks at 06:59--07:00 PDT October 16 found the lock free,
former PID 6366 absent, and no surviving writer. Root `running` / `phase5`
was stale bookkeeping, not liveness. CUDA float64 works in `~/.venv`
(PyTorch 2.10.0+cu128, RTX 4070); about 10.95 GiB GPU memory was free.
All 31 frozen scientific hashes and driver version 1 SHA256 match. No
implementation or scientific settings changed since the 411-test suite
passed in 86.20 seconds during the preceding continuation; that result is
retained rather than rerunning unchanged tests. The notebook has zero errors
and last refreshed at 22:40:49 PDT October 15.

The unchanged reporting-only resume command below was launched at about
**07:00 PDT October 16**. Its actual held lock and exact argv are verified:
**PID 4867, native exec session 34866**. The independent 30-second disk
supervisor is **session 5168**, verifying this exact writer/module and
requesting SIGTERM below 22 GiB free. Free disk at launch was **52.70 GiB**;
preserve the 20-GiB reserve. New invocation provenance is staging under
`execution/report_resume/.incomplete/1792159237589724691-4867-39e98a48242a/`.
Earlier interrupted invocation records remain intact and incomplete.
Historical session numbers/PIDs below must not be reused or signaled.

Phases 0--4 are complete. There are **5,284 sealed units overall**, including
Phase 5 **1,981 units / 1,875 of 2,304 distinct final trajectories**.
Replicas 0--25 are complete across all four conditions; replica 26 has
three linear fixed policies sealed. Exactly **429 trajectories** remain.
The CPU-verified recovery checkpoint is:

```text
phase5/.incomplete/
trajectory__26__rotation_linear__depth4_diag__9b65d87f85aa6342fc6d/checkpoint.pt
```

It is at zero-based step **106**, stage **`fit`**, episode start 96, depth 4,
106 accepted fit calls, zero unresolved fits, and an existing archive
reservation. Its SHA256 is
`43122e68fb771a87e7ed4368fc0791006ce636f9c573155cd2a0b558f29380b8`,
4,136,307 bytes, modification time `1792129603973543654` ns. Resume the
pending fit with model/anchor/archive, seeds, prior accepted operations and
charged budget intact. Initial sealed-unit-ID inventory SHA256 is
`dc9b625cb9a6b49aaf8ae6495b6bf9a67df366c7b2bda50be6f6ba31ecf4fa3d`.
Do not exclude unresolved states elsewhere in the cohort. Exclude interrupted
calendar spans only from active-time extrapolation, retaining operation times
and all scientific results.

Continue the unchanged 32-replica final cohort through Phase 5, then Phase 6,
then the reviewed separate Phase 3 reporting supplement after the final
ordinary refresh releases the lock. Initial full integrity/report validation
and completed-unit replay still take time. The previous tested-driver run
needed 62.46 minutes to its first recovered seal, versus roughly 3 h 51 min
for a prior unoptimized continuation; this is not a controlled estimate of
reporting's isolated speedup. The new archive is larger. Monitor actual
lock/argv, I/O, checkpoint/report timestamps, errors, and disk rather than
trajectory counts alone. Stop responsiveness during completed-unit replay
remains inherited and slow. Do not change frozen scientific sources,
configuration, cohort, convergence rules, or significance stopping.

### Progress At 09:34 PDT

The recovered replica-26 linear `depth4_diag` trajectory sealed at
**08:03:22 PDT**, **62.76 minutes** after this invocation's recorded start.
It retained all accepted work and finished with **0 unresolved fits out of
120**. Recorded active online/evaluation work is 45.97/42.36 s; exclude only
its interrupted calendar span from runtime extrapolation. This restart time
includes verification/replay and completion of the saved trajectory, not an
isolated measurement of reporting speedup.

The same verified PID 4867 / session 34866 continues without pipeline errors.
There are **5,363 sealed units overall**, Phase 5 **2,060 units / 1,950 of
2,304 trajectories** (354 left); the progress JSON counter may lag a freshly
sealed unit. Replicas 0--26 are complete across all four conditions, and
replica 27 is advancing through linear rotation. The notebook refreshed at
09:24:27 PDT; free disk is 51.99 GiB. All 31 frozen hashes still match.
The initial step-106 checkpoint above has completed and is now immutable;
inspect the current `.incomplete` trajectory on any later resume rather than
expecting that old checkpoint to remain in staging. Continue unchanged through
Phase 6 and final reporting-supplement integration.

### Progress At 11:07 PDT

The same verified writer and supervisor continue unchanged without pipeline
errors. Phase 5 has **2,139 units / 2,025 of 2,304 distinct trajectories**
sealed (279 left), with **5,442 sealed units overall**. Replicas 0--27 are
complete across all four conditions; replica 28 is progressing through
linear rotation. The notebook last refreshed at 10:50:57 PDT; free disk is
51.31 GiB. All 31 frozen scientific hashes were checked again and still
match. Continue the fixed remaining four replicas, then Phase 6 and the
separate final report supplement; no conditions or replicas were dropped.

### Progress At 15:10 PDT

All **31 paired replicas per condition** (indices 0--30) are complete:
Phase 5 has **2,232 of 2,304 distinct trajectories** sealed, with the final
**72 remaining** for replica 31. There are 5,659 sealed units overall and
2,356 Phase 5 units, including shared assets. The final replica follows.
The same PID 4867 / exec session 34866 and supervisor session 5168 continue
without pipeline errors. The notebook refreshed at 15:05:43 PDT; free disk
is 49.30 GiB. Preserve the frozen cohort and all accepted scientific states.
Phase 6 and the separate final Phase 3 reporting supplement remain pending.

## October 15 Continuation (Historical, 19:18 PDT)

The user's new **"Please continue working"** authorizes resumption and
supersedes the previous halt. Actual host preflight at 19:16 PDT verified
the run lock free, former PID 12071 absent, exact checkpoint and sealed-unit
inventory hashes unchanged, all 31 frozen scientific hashes matching, and
CUDA float64 working in `~/.venv` (PyTorch 2.10.0+cu128, RTX 4070).
There was about 11.01 GiB of free GPU memory. The notebook has zero error
outputs; its last refresh before launch was 18:13:15 PDT. The 411-test unit
suite passed again in **86.20 seconds**. No implementation or scientific
configuration changed.

The tested reporting-only resume command below was launched at about
**19:18 PDT October 15**. The real held lock and exact argv are verified:
**actual PID 6366, native exec session 90322**. Its new independent disk
supervisor is **session 60320**, checking this exact writer/module every
30 seconds and requesting SIGTERM below 22 GiB free; preserve the 20-GiB
reserve. Free disk after unit tests and at launch was **52.49 GiB**. The new
invocation provenance is staging at
`execution/report_resume/.incomplete/1792117099329706403-6366-f56b694c203a/`.
The older interrupted invocation remains intact and incomplete. Historical
native session numbers and PIDs below are not live identities.

Phases 0--4 remain complete. Resume the unchanged 32-replica final cohort
from **1,743 of 2,304 trajectories** sealed (561 remain), then Phase 6,
then the reviewed separate Phase 3 report supplement only after the final
ordinary refresh releases the lock. The saved replica-24 linear `asg_lr_lr`
checkpoint remains step 33, stage `evaluate`, depth 1, with one unresolved
fit retained. Preserve its model, anchor, archive, seeds, and charged budget.
Do not repeat accepted operations or exclude unresolved scientific states.

Startup performs initial full report/integrity validation, completed-unit
verification, and progress replay; the old `phase5` progress label during
the initial scan is not evidence that new training has begun. Monitor actual
lock/argv, I/O, checkpoint/notebook timestamps, errors, and disk. Report-only
coalescing is tested, but the preceding invocation was stopped before its
first new scientific unit: end-to-end restart savings remain unmeasured.
The driver still inherits slow stop checks in completed-unit validation.
Do not alter the frozen package or stopping rules to change that behavior.

### Progress At 20:51 PDT

The saved replica-24 linear `asg_lr_lr` trajectory sealed at **20:20:46 PDT**,
**62.46 minutes** after this invocation's recorded start (19:18:19 PDT).
This is a measured end-to-end restart time, including validation/replay and
completion of the saved trajectory, not a controlled estimate of the
coalescer's isolated speedup. The prior successful continuation took roughly
3 h 51 min. Do not attribute the entire difference solely to reporting.
The recovered trajectory retained **4 unresolved fits out of 147**; all its
scientific results remain included. Recorded active online/evaluation work
is 52.16/42.42 s. Exclude only its interrupted calendar span from runtime
extrapolation, retaining its operation times and accepted scientific states.

At 20:51 PDT the same verified writer remains active without pipeline error:
**5,159 sealed units overall**, Phase 5 **1,856 units / 1,758 of 2,304 final
trajectories** (546 left). The active partial trajectory is replica 24 /
sigmoid / `depth5_lr`, advancing normally. The notebook refreshed at
20:44:01 PDT; free disk is 52.40 GiB. All 31 frozen source hashes still
match. Continue the unchanged cohort, then closeout and the final reporting
supplement. The first-new-work measurement above supersedes the startup-only
"unmeasured" status earlier in this historical continuation narrative.

## October 15 Stop (Historical, 18:21 PDT)

At this historical stop, resumption required a new user request (now received).
The user initially requested a
graceful halt: SIGTERM was sent at **17:53:16 PDT October 15** to the verified
writer, PID 12071 / native session 95593. The frozen completed-unit validation
path does not check its stop flag, so it continued read-only archive replay.
The user first chose to wait, then asked about early-shutdown risk and
explicitly approved **"Yes, stop it now"**. This later approval superseded the
earlier instruction to wait for a clean exit.

At **18:21:42 PDT**, after verifying the exact held lock/argv, unchanged sealed
inventory, checkpoint bytes and timestamp, all scientific source hashes, and
historical staging files, only that verified process was sent SIGKILL. The
worker exited **137**, not a graceful exit 0. The disk supervisor (session
4418) exited **0**. Post-stop host checks at **18:22:24 PDT** confirmed the
lock free, PID 12071 absent, and no GPU compute processes. The node is free.
There was no new accepted scientific work during this invocation.

The scientific checkpoint SHA256 before and after stopping is
`0ce68c4b18e3455541533695e32222ad090951fc4c6d8f065470c8d63ed6db08`,
4,132,339 bytes, with unchanged modification time
`1792066054075362250` ns. It is still replica 24 / linear / `asg_lr_lr`,
step 33, stage `evaluate`, depth 1, with one unresolved fit retained. There
are still 5,144 sealed units, including Phase 5's 1,841 units / **1,743 of
2,304 trajectories** (561 remain). The sealed-unit-ID inventory SHA256 is
`8799f8ff01064414af0d84945c07f8413dc4943858a13ea31c0f6bbaea69f9da`.
All 31 frozen scientific source hashes match; 52.02 GiB is free. Phase 6
and the separate Phase 3 reporting supplement remain pending.

Root `progress.json` still says `running` / `phase5`: this is stale
bookkeeping from the forced stop, not a live writer. The notebook last
refreshed at **18:13:15 PDT**, with zero error outputs. No terminal refresh
or invocation-provenance finalization ran. Leave
`execution/report_resume/.incomplete/1792110135571661270-12071-452ce82bc4d1/`
intact; it contains config, manifest, exact driver source, and initial
inventory only. Do not fabricate its completion or final refresh counts.
No completed scientific artifact was deleted, overwritten, or repaired.

The reporting-only driver version 1 is implemented, reviewed, and tested
(411 passing tests), and the resume command below remains valid. Its scope
does **not** fix inherited slow stop checks during archive replay. A future
stop-responsiveness improvement would need separate implementation/testing;
do not silently edit the frozen package to obtain it. Restart overhead still
includes initial/full validation, unit checks, and progress bookkeeping.
No end-to-end restart-speedup measurement is available: this invocation was
stopped before reaching new scientific work. Do not claim a measured speedup.
Native session numbers/PIDs below are historical; recheck real lock/argv on
the next authorized resume, then start a new identity-specific disk supervisor.

## Approved Driver And Historical Continuation

The user approved implementing and using the separately versioned report-only
resume driver after tests passed. **All 411 unit tests passed in 110.49 s**:
365 existing tests plus 46 new driver tests. The independent test agent's work
was reviewed and the agent closed. Generated-image CPU comparisons exercised
both a fixed diagonal policy and actual adaptive LR/LR gate probes, with and
without provenance: model tensors, scientific metrics, gates, snapshots,
checkpoints, and charged budgets agree. Tests also cover completed-artifact
immutability, corruption, random-state preservation, refresh failures,
heartbeat, terminal checks, CLI guards, and invocation provenance. All 31
frozen scientific source hashes remain unchanged.

At launch, the held writer lock and exact command were verified: **actual
PID 12071, native exec session 95593**. Its independent 30-second disk
supervisor was **session 4418**, verifying module
`mnist_experiment.akaike_score_gate_resume` and requesting SIGTERM below
22 GiB. Both have now exited, as recorded above. Free disk at launch was
52.03 GiB; preserve the 20-GiB reserve. The invocation began at about
**17:22 PDT**.
Use the following command for subsequent resumes; the original CLI remains a
compatible fallback but does not acquire this reporting optimization:

```bash
/home/evan/.venv/bin/python -m mnist_experiment.akaike_score_gate_resume \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb --resume
```

Driver version 1 SHA256 is
`62ad4b51964af124e9b375bd33be14a53591df337e98358044bb3079e8c5501e`.
Its separate invocation provenance is staging under
`execution/report_resume/.incomplete/1792110135571661270-12071-452ce82bc4d1/`;
normal exit seals this record outside scientific phase directories. Its
completion does not imply scientific cohort completion. The driver changes
only redundant forced report refreshes, retaining initial/final checks,
normal unit validation and the original 900-second heartbeat. It does not
change scientific dispatch, estimator logic, settings, seeds, budgets, or
statistical stopping. Initial validation still costs time: measure the first
new trajectory rather than assuming an instantaneous restart.

**Cutover details:** SIGTERM was sent to the verified original PID 4306 at
17:20:10 PDT. Its frozen completed-unit replay has no stop checks, so it
remained in read-only Phase 2 replay. At 17:21:42 PDT, after rechecking the
exact lock/argv, unchanged sealed inventory, checkpoint bytes and timestamps,
all scientific hashes, and historical staging files, only that process was
terminated with SIGKILL (native session exit 137, not a graceful exit 0).
Its supervisor exited normally; lock release and PID absence were verified
before launching the new driver. No accepted scientific work was discarded.
The unchanged checkpoint SHA256 is
`0ce68c4b18e3455541533695e32222ad090951fc4c6d8f065470c8d63ed6db08`;
the sorted sealed-unit-ID inventory SHA256 is
`8799f8ff01064414af0d84945c07f8413dc4943858a13ea31c0f6bbaea69f9da`.
The old October 7 Phase 0 staging directory
`phase0/.incomplete/pipeline_validation__0__shared__None__d59e5e0ec900b176bc5f/`
contains only historical config/manifest files and was left untouched.

At cutover Phases 0--4 remain complete, with 5,144 sealed units overall;
Phase 5 has 1,841 units / **1,743 of 2,304 distinct trajectories** sealed.
Replicas 0--23 are complete across all four conditions; replica 24 has 15
linear policies sealed. There are 561 final trajectories left, including
the saved `asg_lr_lr` step-33 `evaluate` checkpoint described below. Continue
the unchanged cohort through Phase 6. Apply the separate reviewed Phase 3
reporting supplement only after the worker's final refresh and lock release.
Monitor actual lock/argv, I/O, errors, checkpoint/report timestamps, and disk.
Never change stopping rules from interim results or mutate sealed artifacts.

## October 15 Continuation (Historical, 16:00 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 15:58--15:59 PDT October 15 found the lock free,
former PID 13471 absent, and no surviving Plan 14 worker. Root
`status="running"` was stale bookkeeping. CUDA float64 works in `~/.venv`
on the RTX 4070 (PyTorch 2.10.0+cu128), all 31 frozen scientific hashes
match, and the resolved study hash is unchanged. Free disk was 54.05 GiB;
GPU free memory was about 10.97 GiB. The notebook last refreshed at
12:24:51 PDT, with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 19918, actual PID 4306**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
57873**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
94450/37928 and PID 13471 are historical. Native session numbers can be
reused across turns: session 19918 here is the new PID 4306 invocation,
not the historical PID 4193 run. Always recheck the real lock/command.

Inventory is unchanged from the preceding restart: Phases 0--4 complete,
5,144 sealed units overall, Phase 5 1,841 units / **1,743 of 2,304 distinct
final trajectories**. Replicas 0--23 are complete across all four conditions;
replica 24 has 15 linear policies sealed. Exactly 561 final trajectories
remain unsealed. The preceding invocation ended during archive/report
replay, before adding trajectories or advancing the saved checkpoint.

```text
phase5/.incomplete/
trajectory__24__rotation_linear__asg_lr_lr__a761c4034c3aa7eb9818/checkpoint.pt
```

The CPU-verified checkpoint remains step 33, stage `evaluate`, episode
start 32, depth 1, 41 fit calls, one unresolved fit, and an existing archive
reservation. Resume the pending evaluation with model, anchor, archive,
seeds, and budget intact; do not repeat accepted fits, gate probes, or
archive charges. Retain unresolved states in scientific results. Exclude
only interrupted calendar spans from active-compute projections, not
their scientific data or recorded operation times.

Continue the unchanged 32-replica cohort through Phase 5, then Phase 6,
then the reviewed report-only Phase 3 supplement after the final ordinary
refresh releases the lock. Restart replay is a substantial limitation:
the last successful continuation needed about 3 hours 51 minutes before
new seals, and short later windows have ended during replay with no new
training. The user was informed of roughly four-hour restart overhead.
An optional asynchronous question asks whether to investigate a separately
versioned report-only optimization preserving artifact checks, frozen
scientific sources, and settings. The user approved **investigation** at
about 16:56 PDT. Read-only source review identified at least 48 forced report
refreshes in reused Phase 2 audits, plus 96 completed Phase 5 condition-block
refreshes before this checkpoint, in addition to phase boundaries. Every
refresh calls `analysis.load_report` across the whole catalog and rewrites
the notebook even when no scientific unit changed. The current notebook is
67.08 MiB; serialized figure outputs total 49.15 MiB, within the existing
64-MiB cache. Plot caching is working; blindly enlarging it is not the fix.
The frozen `UnitStore.completed` separately verifies every reused unit and
caches those checks only within an invocation.

A separately versioned reporting-only resume wrapper appears viable: coalesce
unchanged forced refreshes, retain initial/final full checks and normal unit
validation, and keep the 15-minute reporting heartbeat. It must not change
scientific scheduling, source hashes, seeds, settings, budget/fit accounting,
or sealed artifacts. Full integrity-scan overhead remains unavoidable.
A second optional asynchronous question authorized implementation, testing,
and adoption only after equivalence and artifact-safety tests pass (user
reply about 17:02 PDT). A candidate now exists outside the frozen package:
`mnist_experiment/akaike_score_gate_resume.py`. It subclasses only reporting
refresh behavior and delegates all scientific methods, unit validation,
CLI arguments, locks, and signal handling to the frozen executor. Initial,
changed-data/status, missing-notebook, terminal/error, and 900-second
heartbeat refreshes remain. Separate immutable invocation provenance under
`execution/report_resume/` stores its exact source, hash, original scientific
hashes, command, environment, inventory, refresh counters, and outcome;
these records are not scientific phase units or additional replicas.
A disjoint test agent (Boole, id `01a14207-75a2-7460-aa81-5a59a4a54117`)
is implementing `test/unit/akaike_score_gate/test_resume_driver.py`.
**Tests/deployment remain pending; the active worker still runs the original
unchanged CLI.** Do not silently alter
the frozen 31 Python files or scientific behavior. Any approved reporting
optimization needs separate provenance and focused verification.

Only Markdown and the separately versioned candidate driver/tests changed
for this continuation. All frozen sources/configuration remain unchanged;
the 365 existing tests passed again in 78.45 seconds before adoption.
Candidate eligibility was also checked read-only against the actual frozen
study. The new test agent's equivalence/safety suite remains in progress;
do not adopt until it passes and is reviewed. The plan records the approved
report-only amendment and candidate CLI. Monitor actual
lock/command, I/O, notebook/checkpoint timestamps, errors, and disk; unchanged
trajectory counts alone do not establish a stall. Never mutate/delete sealed
units or change stopping rules based on interim results.

## October 15 Continuation (Historical, 09:43 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 09:42 PDT October 15 found the lock free, former
PID 4516 absent, and no surviving Plan 14 worker. Root `status="running"`
was stale bookkeeping. CUDA float64 works in `~/.venv` on the RTX 4070
(PyTorch 2.10.0+cu128), all 31 frozen scientific source hashes match, and
the resolved study hash is unchanged. Free disk was 54.14 GiB; GPU free
memory was about 10.90 GiB. The notebook last refreshed at 04:53:07 PDT,
with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 94450, actual PID 13471**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
37928**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
68186/24892 and PID 4516 are historical; never trust or signal stale PIDs.

Phases 0--4 are complete. At this restart there are 5,144 sealed units;
Phase 5 has 1,841 units / **1,743 of 2,304 distinct final trajectories**.
Replicas 0--23 are complete across all four conditions (1,728 trajectories).
Replica 24 has 15 linear policies sealed. Exactly 561 final trajectories
remain unsealed, including this saved partial unit:

```text
phase5/.incomplete/
trajectory__24__rotation_linear__asg_lr_lr__a761c4034c3aa7eb9818/checkpoint.pt
```

The CPU-verified checkpoint is zero-based step 33, stage `evaluate`,
episode start 32, depth 1, 41 fit calls, one unresolved fit, and an existing
archive reservation. Resume the pending evaluation with the accepted model,
episode anchor, archive, seeds, and budget intact; do not repeat accepted
fits, gate probes, or consolidation charges. Keep the unresolved fit and
all other unresolved states in the scientific analysis with explicit
limitations. Exclude only this interrupted unit's calendar span from active
timing projections, retaining every scientific result and operation time.

Continue the unchanged 32-replica cohort and promoted repair through Phase 5,
then Phase 6. Apply the reviewed report-only Phase 3 supplement only after
the main worker's final refresh releases the lock. Startup checks the large
sealed archive and refreshes completed condition blocks. The last successful
restart took about 3 hours 51 minutes to its first new seal; this slightly
larger completed cohort also needs several hours of replay. Use actual
lock/command, I/O, and notebook/checkpoint timestamps for liveness rather
than trajectory counts alone. Only Markdown changed; all frozen sources/
configuration and the 365 previously passing tests remain unchanged. Never
mutate/delete sealed units or change stopping rules based on interim results.

## October 15 Continuation (Historical, 00:09 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 00:08 PDT October 15 found the lock free, former
PID 4371 absent, and no surviving Plan 14 worker. Root `status="running"`
was stale bookkeeping. CUDA float64 works in `~/.venv` on the RTX 4070
(PyTorch 2.10.0+cu128), all 31 frozen scientific source hashes match, and
the resolved study hash is unchanged. Free disk was 54.78 GiB; GPU free
memory was about 10.99 GiB. The notebook last refreshed at 22:49:52 PDT
October 14, with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 68186, actual PID 4516**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
24892**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
12989/71166 and PID 4371 are historical; never trust or signal stale PIDs.

Inventory is unchanged from the preceding restart: Phases 0--4 complete,
5,079 sealed units overall, Phase 5 1,776 units / **1,682 of 2,304 distinct
final trajectories**. Replicas 0--22 are complete in all four conditions;
replica 23 has its linear block and eight sigmoid policies sealed. Exactly
622 final trajectories remain unsealed. The previous invocation ended
during archive/report replay, before adding any trajectory or advancing
the saved checkpoint. Its reporting/integrity work must replay again;
do not interpret unchanged scientific counts as new experimental failure.

```text
phase5/.incomplete/
trajectory__23__rotation_sigmoid__depth1_lr__187d471e9937cfe3506d/checkpoint.pt
```

The CPU-verified checkpoint remains zero-based step 78, stage `arrival`,
episode start 64, depth 1, 78 fit calls, zero unresolved fits, and an
existing archive reservation. Resume with the accepted model, anchor,
archive, seeds, and budget intact; do not repeat accepted fits or archive
charges. Exclude only this interrupted unit's calendar span from active
timing projections, keeping all scientific results and recorded operation
times. Retain unresolved optimizer states in other units with explicit
limitations; do not filter them from effects or uncertainty estimates.

Continue the unchanged 32-replica cohort and promoted repair through Phase 5,
then Phase 6. Apply the reviewed report-only Phase 3 supplement only after
the main worker's final refresh releases the lock. Startup validates the
large archive and refreshes every completed condition block; allow several
hours before new seals. Use actual lock/command, I/O, and notebook/
checkpoint timestamps for liveness, not trajectory counts alone. Only
Markdown changed; all frozen sources/configuration and the 365 previously
passing tests remain unchanged. Never mutate/delete sealed artifacts or
change statistical stopping rules based on interim results.

Milestone at 04:13 PDT October 15: the saved replica-23 sigmoid `depth1_lr`
trajectory sealed at 03:59:42 PDT, about 3 hours 51 minutes after restart.
It retained 120 fit calls with zero unresolved fits and recorded active
online/evaluation times 23.56/41.40 seconds. Exclude only its interrupted
calendar span from timing projections, not its scientific data or actual
operation times. Phase 5 now has 1,689 sealed trajectories / 5,086 units
overall, with replica 23's sigmoid block still progressing. The notebook
refreshed at 03:59 PDT; free disk is 54.73 GiB. The verified writer and
disk supervisor remain active, with no reported pipeline errors.

## October 14 Continuation (Historical, 19:41 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 19:40 PDT October 14 found the lock free, former
PID 4193 absent, and no surviving Plan 14 worker. Root `status="running"`
was stale bookkeeping. CUDA float64 works in `~/.venv` on the RTX 4070
(PyTorch 2.10.0+cu128), all 31 frozen scientific source hashes match, and
the resolved study hash is unchanged. Free disk was 54.69 GiB; GPU free
memory was about 10.97 GiB. The notebook last refreshed at 16:03:26 PDT,
with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 12989, actual PID 4371**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
71166**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
19918/56504 and PID 4193 are historical; never trust or signal stale PIDs.

Phases 0--4 are complete. At this restart there are 5,079 sealed units;
Phase 5 has 1,776 units / **1,682 of 2,304 distinct final trajectories**.
Replicas 0--22 are complete in all four conditions (1,656 trajectories).
Replica 23 has its 18-policy linear block and eight sigmoid policies sealed.
Exactly 622 final trajectories remain unsealed, including this saved unit:

```text
phase5/.incomplete/
trajectory__23__rotation_sigmoid__depth1_lr__187d471e9937cfe3506d/checkpoint.pt
```

The CPU-verified checkpoint is zero-based step 78, stage `arrival`, episode
start 64, depth 1, 78 fit calls, zero unresolved fits, and an existing
archive reservation. Resume the next arrival with the accepted model,
episode anchor, archive, seeds, and budget intact; do not repeat accepted
fits or consolidation charges. Exclude only this interrupted unit's calendar
span from active-compute projections, keeping all scientific results and
recorded operation times. Unresolved optimizer states elsewhere remain
included in every analysis with explicit limitations.

Continue the unchanged 32-replica cohort and promoted repair through Phase 5,
then Phase 6. Apply the reviewed report-only Phase 3 supplement only after
the main worker's final refresh releases the lock. Startup validates the
large sealed archive and regenerates reports. The previous restart needed
about three hours before its first new seal; this larger completed cohort
requires more block/report replay. Use real lock/command, I/O, and notebook/
checkpoint timestamps rather than trajectory counts alone for liveness.
Only Markdown changed; all frozen sources/configuration and the 365
previously passing tests remain unchanged. Never mutate or delete sealed
artifacts or change stopping rules based on interim results.

## October 14 Continuation (Historical, 02:14 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 02:13 PDT October 14 found the lock free, former
PID 4413 absent, and no surviving Plan 14 worker. Root `status="running"`
was stale bookkeeping. CUDA float64 works in `~/.venv` on the RTX 4070
(PyTorch 2.10.0+cu128), all 31 frozen scientific source hashes match, and
the resolved study hash is unchanged. Free disk was 60.11 GiB; GPU free
memory was about 10.98 GiB. The notebook last refreshed at 01:10:09 PDT,
with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 19918, actual PID 4193**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
56504**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
6728/77608 and PID 4413 are historical; never trust or signal stale PIDs.

Phases 0--4 are complete. At this restart there are 4,481 sealed units;
Phase 5 has 1,178 units / **1,116 of 2,304 distinct final trajectories**.
Replicas 0--14 are complete in all four conditions (1,080 trajectories).
Replica 15 has both 18-policy rotation blocks complete; its mixture
conditions are next. Exactly 1,188 final trajectories remain unsealed.
There is no incomplete unit directory or trajectory checkpoint to restore:
reuse all sealed units and continue the next queued work. Both rotations
have 16 complete replicas; each mixture has 15. Keep the conditions separate.

Continue the unchanged 32-replica cohort and promoted repair through Phase 5,
then Phase 6. Apply the reviewed report-only Phase 3 supplement only after
the main worker's final refresh releases the lock. Startup validates the
large sealed archive and regenerates reports; the previous restart took
about 2 hours 42 minutes to its first new seal, and this larger completed
cohort needs more block/report replay. Use actual lock/command, I/O, and
notebook/checkpoint timestamps to assess progress, not trajectory counts
alone. Only Markdown changed; all frozen sources/configuration and the
365 previously passing tests remain unchanged. Never mutate or delete
sealed artifacts, drop unresolved optimizer states, or change stopping
rules based on interim results.

Milestone at 05:20 PDT October 14: the first new trajectory sealed at
05:14:48 PDT (`replica 15 / mixture_m18 / depth1_diag`, ten fits, zero
unresolved), about three hours after restart. No sealed trajectories were
retrained. Phase 5 has 1,134 sealed trajectories / 4,501 units overall;
replica 15's smaller-mixture block is complete and its larger mixture is
next. The notebook refreshed at 05:13 PDT; free disk is 59.89 GiB.
The verified writer/supervisor continue, and no pipeline error is reported.
There was no partial unit at this restart, so no new trajectory needs a
pause-span timing exclusion merely because startup replay was lengthy.

Milestone at 08:50 PDT October 14: replicas 0--17 are complete across all
four conditions (1,296 trajectories), and replica 18's linear block is
running. Phase 5 has 1,308 sealed trajectories / 4,685 units overall.
The notebook refreshed at 08:41 PDT; free disk is 58.27 GiB. The same
verified PID 4193 / session 19918 writer and session 56504 disk supervisor
continue. No pipeline error is reported; the frozen cohort and analysis
are unchanged. Phase 6 and final Phase 3 supplement integration remain
pending. Unresolved optimizer states remain included in scientific results.

Milestone at 11:24 PDT October 14: replicas 0--19 are complete across all
four conditions (1,440 trajectories), and replica 20's linear block is
running. Phase 5 has 1,443 sealed trajectories / 4,828 units overall.
The notebook refreshed at 11:19 PDT; free disk is 56.95 GiB. The same
verified writer/supervisor continue, with no reported pipeline errors or
scientific source/configuration changes. Continue the unchanged final
cohort; Phase 6 and the report-only Phase 3 supplement remain pending.

Milestone at 14:30 PDT October 14: replicas 0--21 are complete across all
four conditions (1,584 trajectories), and replica 22's linear block is
running. Phase 5 has 1,597 sealed trajectories / 4,990 units overall.
The notebook refreshed at 14:24 PDT; free disk is 55.58 GiB. The same
verified writer and disk supervisor continue, with no pipeline errors
reported. The fixed cohort, source hashes, and configuration remain
unchanged; Phase 6 and final Phase 3 supplement integration still follow.

## October 13 Continuation (Historical, 17:19 PDT)

The user explicitly requested resuming after interrupting the prior turn.
Actual host preflight at 17:18 PDT October 13 found the lock free, former
PID 4795 absent, and no surviving Plan 14 worker. Root `status="running"`
was stale bookkeeping. CUDA float64 works in `~/.venv` on the RTX 4070
(PyTorch 2.10.0+cu128), all 31 frozen scientific source hashes match, and
the resolved study hash is unchanged. Free disk was 62.26 GiB; GPU free
memory was about 10.99 GiB. The notebook last refreshed at 16:19:32 PDT,
with 160 figure outputs and zero error outputs.

The unchanged full-study `--resume` writer now holds the actual lock:
**native exec session 6728, actual PID 4413**, verified by exact
module/config/root command. The 30-second disk supervisor is **session
77608**; it requests SIGTERM below 22 GiB and exits when this verified
writer releases the lock. Preserve the 20-GiB reserve. Old sessions
90549/84220 and PID 4795 are historical; never trust or signal stale PIDs.

Phases 0--4 are complete. At this restart there are 4,228 sealed units;
Phase 5 has 925 units / **875 of 2,304 distinct final trajectories**.
Replicas 0--11 are complete in all four conditions (864 trajectories).
Replica 12 has 11 linear policies sealed. Exactly 1,429 final trajectories
remain unsealed, including this saved partial trajectory:

```text
phase5/.incomplete/
trajectory__12__rotation_linear__depth4_lr__1d8a2c217e4eb32d94cf/checkpoint.pt
```

The CPU-verified checkpoint is zero-based step 11, stage `fit`, episode
start 0, depth 4, 11 fit calls, one unresolved fit, and an existing archive
reservation. Retain that unresolved state in all scientific diagnostics;
it is not a pipeline failure or a reason to drop the trajectory. Resume
the pending fit with its accepted model, episode anchor, archive, seeds,
and budget intact; do not repeat prior accepted fitting or consolidation.
Exclude only this interrupted unit's calendar span from active-compute
projections, keeping every scientific result and recorded operation time.

Continue the unchanged 32-replica cohort and promoted repair through Phase 5,
then Phase 6. Apply the reviewed report-only Phase 3 supplement only after
the main worker's final refresh releases the lock. Startup validates the
large sealed archive and regenerates reports; allow at least roughly two
hours before new seals and use real lock/command, I/O, and notebook/
checkpoint timestamps to assess progress. More completed Phase 5 blocks
now need replay than at the previous restart. Only Markdown changed;
all frozen sources/configuration and the 365 previously passing tests
remain unchanged. Never delete or mutate sealed artifacts.

Milestone at 20:23 PDT October 13: the saved replica-12 linear `depth4_lr`
trajectory sealed at 20:00:58 PDT, about 2 hours 42 minutes after restart.
It records 120 fit calls, 15 unresolved fits, and active online/evaluation
times 110.60/42.14 seconds. Keep those unresolved states in the results;
they are optimizer-health limitations, not grounds for dropping the unit.
Exclude only its interrupted calendar span from timing projections.
Phase 5 now has 887 sealed trajectories / 4,240 units overall, with replica
12's linear block complete and sigmoid underway. The notebook refreshed
at 20:13 PDT; free disk is 62.19 GiB. The same verified writer and disk
supervisor remain active, and no pipeline error is reported.

Milestone at 22:51 PDT October 13: replicas 0--13 are complete across every
condition (1,008 trajectories), and replica 14's linear block is running.
Phase 5 has 1,019 sealed trajectories / 4,380 units overall. The notebook
refreshed at 22:45 PDT; free disk is 60.95 GiB. No pipeline error is reported.
The verified PID 4413 / session 6728 writer and session 77608 disk supervisor
continue. The cohort, analysis, sources, and configuration are unchanged;
Phase 6 and final Phase 3 supplement integration are still pending.

## October 13 Continuation (Historical, 07:12 PDT)

The user explicitly requested resuming. Actual host preflight at 07:11 PDT
October 13 found the lock free, former PID 4447 absent, and no surviving
writer. Root `status="running"` was stale bookkeeping, not liveness.
CUDA float64 works in `~/.venv` on the RTX 4070 (PyTorch 2.10.0+cu128),
all 31 frozen scientific source hashes match, and the resolved study hash
is unchanged. Free disk was 65.59 GiB; GPU free memory was about 11.03 GiB.

The unchanged full-study `--resume` writer now owns the actual lock:
**native exec session 90549, actual PID 4795**, verified by its exact
module/config/root command. The 30-second disk supervisor is **session
84220**, requests SIGTERM below 22 GiB, and exits when this verified writer
releases the lock. Preserve the 20-GiB reserve. Old sessions 18569/28527
and PID 4447 are historical; never trust or signal stale PIDs.

Phases 0--4 are complete. At this restart there are 3,856 sealed units;
Phase 5 has 553 units / **523 of 2,304 distinct final trajectories**.
Replicas 0--6 are complete in all four conditions. Replica 7 has all 18
linear policies and one sigmoid policy sealed. Exactly 1,781 final
trajectories remain unsealed, including this saved partial trajectory:

```text
phase5/.incomplete/
trajectory__7__rotation_sigmoid__depth2_diag__3dba5dee09f44e100e85/checkpoint.pt
```

The CPU-verified checkpoint is zero-based step 62, stage `evaluate`,
episode start 32, depth 2, 63 fit calls, zero unresolved fits, and an
existing archive reservation. Resume the pending evaluation with its
accepted model, episode anchor, archive, seeds, and work budget intact;
do not repeat or recharge accepted fitting/consolidation work.
For timing projections only, exclude this interrupted unit's calendar
start-to-finish span. Keep all its scientific results and recorded active
operation times. The fixed 32-replica ledger, conditions, promoted repair,
and statistical stopping rules are unchanged.

Continue Phase 5, then Phase 6, then apply the reviewed report-only Phase 3
supplement after the main worker's final refresh releases the lock. The
restart initially validates the large sealed archive and regenerates reports;
allow roughly two hours or more before new seals, using real lock/command,
I/O, and checkpoint/notebook timestamps rather than counts alone for liveness.
Only Markdown changed for this restart. The 365 previously passing unit
tests remain applicable; no frozen Python source or configuration changed.

Milestone at 09:47 PDT October 13: the saved `depth2_diag` sigmoid trajectory
sealed at 09:24:45 PDT, about 2 hours 12 minutes after restart. It has 120
fit calls, zero unresolved fits, and recorded active online/evaluation times
11.08/42.73 seconds. Its interrupted calendar span is not an active-compute
measurement. Phase 5 now has 535 sealed trajectories / 3,868 units overall;
replica 7's sigmoid block is still progressing. The notebook refreshed at
09:43 PDT; free disk is 65.51 GiB. No execution errors are reported, and
the verified writer/supervisor remain active. Continue the unchanged cohort.

Milestone at 13:15 PDT October 13: replicas 0--9 are complete in every
condition (720 trajectories), and replica 10's linear block is running.
Phase 5 has 723 sealed trajectories / 4,068 units overall. The notebook
refreshed at 13:09 PDT; free disk is 63.68 GiB. There are no reported
execution errors. The same verified writer and disk supervisor continue;
do not launch a duplicate or change the frozen cohort. Final reporting
and the separate Phase 3 supplement still remain pending.

## October 12 Continuation (Historical, 20:18 PDT)

The user requested continuing after borrowing the node. Preflight at 20:17
PDT October 12 verified the actual lock was free, former PID 5816 absent,
CUDA float64 working in `~/.venv` on the RTX 4070 (PyTorch 2.10.0+cu128),
all 31 frozen scientific hashes matching, and the resolved study hash
unchanged. Free disk was 68.82 GiB; GPU free memory was about 10.96 GiB.
The saved Phase 5 checkpoint still matches the state documented below:
replica 2, sigmoid `asg_lr_lr`, arrival 102, episode start 96, depth 3,
128 fit calls, zero unresolved fits, and the existing archive reservation.

The unchanged full-study `--resume` writer was restarted by 20:18 PDT:
**native exec session 18569, actual PID 4447**. Its held lock and exact
module/config/root command were verified outside the sandbox. Do not
duplicate it; a stored PID/session is not a liveness guarantee. The old
sessions 96818/38630 are finished. The earlier user pause is revoked by
this explicit resume request.

An independent lightweight disk supervisor is active in **session 28527**.
Every 30 seconds it verifies this writer's actual lock owner and command,
requests SIGTERM if free space falls below 22 GiB, and exits when the worker
releases the lock. This gives 2 GiB of headroom above the 20-GiB reserve.
The frozen runner itself has no disk guard. Never delete sealed artifacts
or signal an unverified/reused PID.

Restart inventory is unchanged: Phases 0--4 complete, 3,490 sealed units
overall, Phase 5 187 units / 177 of 2,304 distinct trajectories. The fixed
32-replica-per-condition ledger and promoted `repair_score128` are already
frozen; do not recalibrate or change stopping rules. Resume the saved
arrival without repeating accepted fits or consolidation charges. Continue
Phase 5, then Phase 6, then final report-supplement integration after the
main worker exits and performs its last refresh.

Initial work validates the large completed archive. Root progress may
temporarily retain `interrupted` and then show earlier phase labels while
their sealed units are checked, not retrained. The preceding restart took
roughly 2 hours 12 minutes to its first new seal. Phase 4 now has a frozen
decision, but reporting and integrity checks still take substantial time.
Use actual lock/command, I/O, and notebook/checkpoint timestamps to assess
progress; unchanged trajectory counts alone do not establish a stall.
Only Markdown changed for this continuation; all scientific sources,
configuration, and earlier completed artifacts remain unchanged.

Milestone at 22:07 PDT October 12: Phase 5 has 180 sealed trajectories
and 3,493 units overall; replica 2's linear and sigmoid blocks are complete.
The saved `asg_lr_lr` sigmoid trajectory finished successfully, retaining
accepted fits and its episode/budget state. Its marker is 22:03:42 PDT,
about 1 hour 46 minutes after restart; total recorded online/evaluation
work is 48.75/41.93 seconds, with 147 fit calls and zero unresolved fits.
Free space is 68.80 GiB; the
notebook refreshed at 22:03 PDT. Phases 0--4 were reused, not retrained.
No scientific source/configuration or earlier completed artifact changed.

For timing extrapolation only, exclude the calendar start-to-finish span of
`phase5/trajectory__2__rotation_sigmoid__asg_lr_lr__bbb4e1f82b93a968e1df`.
Its manifest spans the node-reuse pause and startup validation. Keep this
trajectory in every scientific result, and retain its recorded online/
evaluation/fit times, which accumulate actual work. Apply the same rule to
any subsequently paused unit; never infer active compute from paused
manifest timestamps or remove such units from paired effects.

Milestone at 23:45 PDT October 12: all four conditions have four complete
fresh replicas (0--3, 288 trajectories), and replica 4's linear block is
running. Phase 5 has 292 sealed trajectories / 3,613 units overall at this
snapshot. The notebook refreshed at 23:38 PDT; free disk is 67.68 GiB.
No errors or source/configuration changes are reported. Keep the fixed
32 replicas per condition and the existing paired analysis; these are
descriptive interim results, not a significance-based stopping opportunity.

## October 12 Pause (Historical)

The user needs to repurpose the node. At 16:04:43 PDT on October 12
(23:04:43 UTC), SIGTERM was sent only after verifying PID 5816's held run
lock and exact module/root command. The runner reported that accepted work
was checkpointed, refreshed the notebook, and exited with code 0. At
16:07 PDT, the actual lock was free, PID 5816 absent, and no Plan 14
experiment or reporting worker remained. The disk supervisor also exited
with code 0. **The node is free of Plan 14 workers.** Native sessions
96818/38630 and monitoring cell 51 are finished; do not poll or reuse them.
Do not resume until the user asks.

Root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
Root progress is `status="interrupted"`, Phase 5, 3,490 sealed units.
The nested Phase 5 status remains `running` because it is unfinished, not
because a process survives. Stale PID text is not evidence of a held lock.

| Phase | Sealed units | Status |
| --- | ---: | --- |
| 0 | 4 | Complete |
| 1 | 604 | Complete |
| 2 | 48 | Complete |
| 3 | 1,777 | Complete |
| 4 | 870 | Complete: 816 panels, 16 repair trajectories, four local repair audits, profiling, selection |
| 5 | 187 | Partial: 177 trajectories, plus shared source/assets |
| 6 | 0 | Pending |

The immutable final ledger is already frozen: 32 new paired replicas per
condition, four separate conditions, 2,432 nominal assignments / 2,304
distinct trajectories including `repair_score128`. Exactly 2,127 distinct
trajectories remain unsealed, including the saved partial one. Do not
recalibrate, change the cohort, or perform significance-based stopping.
Replicas 0 and 1 are complete in all four conditions (72 trajectories each).
Replica 2 has all 18 linear policies and 15 of 18 sigmoid policies complete;
its mixture conditions have not started. Keep these denominators distinct.

Saved checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase5/.incomplete/
trajectory__2__rotation_sigmoid__asg_lr_lr__bbb4e1f82b93a968e1df/checkpoint.pt
```

The CPU-checked state is replica 2, sigmoid, `asg_lr_lr`, zero-based step
102, stage `arrival`, episode start 96, selected depth 3 (114 parameters).
Arrival 101 is complete, with 408 treatment observations evaluated. Resume
at arrival 102, retaining the current episode's anchor, archive, accepted
model, numerical seeds, work budget, and existing archive reservation.
There are 128 fit calls, zero unresolved fits or gate denials in this saved
trajectory, and its latest restricted residual is 8.62591e-6 (converged).
Do not repeat arrival 101 or charge/consolidate the reserved episode twice.
This is the ordinary 64-score LR-gate policy, not the promoted 128-score
diagonal-gate repair. Configuration remains width 5, float64, rank 8, c=1,
half-life eight, cadence two, omega 39, ridge ratio 0.1, solver ceilings
200/300, residual tolerance 1e-5, episode length 32, principal budget, and
no calibration snapshots. Its charge-table hash is saved in the checkpoint.

All 31 frozen scientific hashes still match the Phase 1 ledger. No code,
scientific configuration, or completed artifact was manually changed.
The latest invocation recorded 7.147 active hours before its final refresh;
its first new seal followed about 2 hours 12 minutes of validation/replay.
Future resume may again spend hours checking completed artifacts and
refreshing reports; inspect actual lock/command, I/O, and notebook/checkpoint
timestamps rather than treating unchanged unit counts as a stall.

Notebook: `mnist_experiment/akaike_score_gate_results.ipynb`, refreshed at
16:06:34 PDT (23:06:34 UTC), 160 figures, zero error outputs. Free disk is
68.78 GiB. Preserve all sealed artifacts and a 20-GiB free-space reserve.
After launching the next verified writer, restart a lightweight supervisor
that verifies that writer's actual lock/command and requests SIGTERM below
22 GiB. The frozen runner has no native disk-space guard. Never signal an
old PID without rechecking its owner/command.

Use the unchanged command on the next user-authorized resume:

```bash
/home/evan/.venv/bin/python -m mnist_experiment.akaike_score_gate.orchestrate \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb \
  --resume
```

Verify the actual lock is free, CUDA float64 works in `~/.venv`, and frozen
sources/configuration still match before launching. Reuse every sealed unit
and the frozen Phase 4 selection; finish Phase 5, then Phase 6. The 365 unit
tests (including 27 reporting-extension tests) already passed during the
previous continuation; only Markdown changed since that verification.

The reviewed report-only Phase 3 supplement is still pending real-notebook
integration. Do not run its full integrity scan during this node-reuse pause.
After the experiment finally completes and its last refresh releases the
lock, run the reporting-extension CLI described below and verify its four
figures and phase order. Another ordinary refresh overwrites the supplement,
so apply it last. Do not modify the 31 frozen Python files to integrate it.

## October 12 Continuation (Historical)

The user authorized documentation reconciliation followed by resuming the
unchanged study. Preflight at 08:53 PDT on October 12 verified the actual
lock was free, former PID 4615 absent, CUDA float64 working on the RTX 4070
in `~/.venv` (PyTorch 2.10.0+cu128), all 31 frozen source hashes unchanged,
and the resolved study hash matching. The root's old `running` status is
stale bookkeeping, not process liveness. There are no incomplete trajectory
checkpoints. Do not assume the previous session/PID remains usable.

Inventory: 3,145 sealed units; Phases 0--3 complete (4/604/48/1,777 units).
Phase 4 has 712 units and 680 trajectories: all 544 principal trajectories
plus 136 of its 272 additional panels. The next work is the remaining 136
panel trajectories, triggered bounded repairs, development selection, then
the fresh Phase 5 cohort and Phase 6 closeout. Phase 5 has not started and
no Phase 4 repair/selection artifact is frozen yet.

The plan's summary table is reconciled with measured forecasts: Phase 5
35--50 active hours for 2,176 distinct principal trajectories, Phase 6
0.5--1 hour artifact-only closeout; the obsolete 6--14-hour Phase 5 estimate
and whole-study subtotal are no longer presented as current budgets. The
Phase 5--6 subtotal excludes remaining development, repairs, and repeated
restart validation. Development repairs retain a provisional 4--8-hour
allowance; one promoted gate-only repair adds 128 final trajectories and
approximately 2--4 hours. No scope, cohorts, estimands, or stopping rules changed.

Storage preflight: 72.23 GiB free; study 506 GiB, chiefly Phase 2 (178 GiB)
and Phase 3 (281 GiB), both complete. Routine Phase 4 units average 6.2 MiB.
Phase 5's no-calibration-snapshot trajectories and shared larger evaluation
assets project 20--22 GiB, with 25--30 GiB reserved conservatively for all
remaining work. Keep the external 20-GiB reserve monitor active. The frozen
runner has no native disk-space guard; do not confuse documentation with
implemented protection. Signal only a verified live lock owner/command, and
never prune or mutate sealed artifacts. Main notebook/report-extension
integration instructions below remain applicable after the final refresh.

Only Markdown changed for this reconciliation. The unchanged `--resume`
writer started at approximately 08:55 PDT and was verified at 08:56 PDT:
**exec session 96818, actual PID 5816**, held lock and expected command.
Initial work validates the existing catalog; no new unit has sealed yet.
Do not duplicate this worker. Session identifiers are not portable and PID
text can outlive/refer to another process; recheck the actual lock/command.
A separate lightweight disk supervisor checks every 30 seconds, verifies
this same lock owner and exact module/root command, and requests SIGTERM
if free space falls below 22 GiB (2-GiB headroom above the 20-GiB reserve).
Its native exec session is **38630**. It exits automatically when the worker
releases the lock. This supervisor
does not import PyTorch or alter artifacts/frozen sources. Verify its launch
session from the active tool state rather than guessing an old identifier.

Milestone at 11:19 PDT on October 12 (18:19 UTC): new Phase 4 trajectories
are sealing again, with 716 of the 816 distinct panel trajectories complete
and 3,181 sealed units overall. The notebook refreshed at 11:18 PDT; free
space is 71.94 GiB. Startup validation plus completed Phase 4 block/report
replay took roughly 2 hours 12 minutes before the first new seal, longer
than the prior restart's 1 hour 20 minutes because more development blocks
now exist. That is restart overhead, not online-policy runtime. Scientific
sources/configuration and every earlier completed unit remain unchanged.
No Phase 4 repair/selection or Phase 5 ledger is frozen yet.

Milestone at 11:54 PDT October 12: all 816 panel trajectories are sealed;
the first repair trajectory has also sealed (817 Phase 4 trajectories).
`phase4_repair_candidates.json` contains just `repair_score128`: LR EWC,
diagonal current gate, c=1, half-life eight, cadence two, rank eight,
128 current-score draws rather than 64. Its bounded four-replica/four-condition
development test and first-replica local audits are running. There is no
rank/cadence repair candidate, and no Phase 4 selection or final ledger yet.
Root/main worker and disk-supervisor sessions remain 96818/38630; verify
actual PID 5816 and its lock before any intervention. Free space is about
71 GiB; scientific sources and cohorts remain unchanged.

Phase 4 completed at 12:16 PDT October 12; Phase 5 started at 12:17 PDT.
The repair candidate freeze to development selection interval was 20.65
uninterrupted minutes (0.344 hours), not the old provisional 4--8 hours.
All 832 trajectories (816 panels + 16 repair), four repair audits, profiling,
and selection are sealed; Phase 4 has 870 units. The promoted repair is
`repair_score128`. Promotion is a practical-tie outcome, not an NLL win:
mean development NLL-AUC 0.57153 vs baseline 0.57072, mean active parameters
56.125 vs 56.375, under tie tolerances 0.005 nats and 2% timing. The four
paired development replicas per condition are not fresh efficacy evidence.

Frozen Phase 5: 32 new replicas per condition, four separate conditions,
2,432 nominal assignments / 2,304 distinct trajectories including the
promoted repair. The ledger records no significance stopping. Per-condition
ASG/fixed selections: rotations `asg_lr_diag` vs `depth4_lr`; mixture m18
`asg_lr_diag` vs `depth1_diag`; mixture m182 `asg_diag_lr` vs `depth5_diag`.
The factorial policies and all other declared controls remain in the cohort;
the repair is separately reported. Principal 35--50 hours plus its promoted
repair's 2--4 hours gives 37--54 hours for the whole frozen Phase 5 and
37.5--55 with closeout, excluding repeated restart validation. Do not add
the promoted-repair component twice. Six fresh trajectories were complete
at 12:29 PDT; source/assets are shared within replica/family. Free space
is about 70 GiB. Main worker 96818 / verified PID 5816 and supervisor 38630
remain active. Final notebook-supplement integration still waits for the
worker's last refresh and lock release; all science sources remain frozen.

First complete fresh replica: all 72 trajectories (18 distinct policies
per condition) sealed by 13:34:14 PDT October 12 (20:34:14 UTC). At the
14:07 PDT inspection, 90 fresh trajectories were sealed overall and 69.54
GiB remained free. The notebook has 160 figure outputs and no errors;
the separate Phase 3 supplement has still not been applied.

Replica-0 mean unit wall / online / evaluation seconds: rotation linear
100.91 / 45.68 / 42.49; sigmoid 101.44 / 47.02 / 41.42; mixture m18
14.19 / 12.01 / 1.01; mixture m182 12.40 / 10.26 / 1.00. Rotation artifacts
average 6.22 MiB and mixtures 6.65/6.66 MiB, including retained checkpoints.
These first-replica rates corroborate, but do not certify, the time/storage
forecasts. Each condition has only one complete fresh replica so far, not
18 independent efficacy replicas. Keep the frozen 32 per condition.

Fit-call unresolved counts in this replica: linear 317/2,114, sigmoid
310/2,120, m18 42/172, m182 100/180. These are optimizer operations, not
trajectory/replica failure rates. All accepted states remain in the fixed
cohort; do not filter unconverged controls or treat them as certified MLEs.
Main 96818 / verified PID 5816 and disk supervisor 38630 continue unchanged.

## October 11 Continuation (Historical)

The user requested resumption after node reuse. At 18:17 PDT on October 11,
the actual lock was free. Its old PID 4457 had been reused by the VS Code
Codex host, not the experiment. Never signal a stored PID without verifying
the actual lock and command line. CUDA float64 execution in `~/.venv`, all
31 frozen source hashes, and the saved Phase 4 checkpoint were verified.
The unchanged full-study `--resume` command below was restarted in
**exec session 97523, PID 4615**; its expected command line and held lock
were verified outside the sandbox. Session identifiers are not portable.

Restart inventory: Phases 0--3 complete; Phase 4 has 13 trajectories and two
paired source/asset units; 2,448 units overall. Continue replica 0's linear
`depth6_lr` at step 88, stage `arrival`, with its existing episode state.
Do not repeat earlier fits or consolidations. All remaining Phases 4--6,
bounded repairs, and report closeout remain authorized without review blockers.
Free disk: 77.6 GiB. Retain the 20-GiB reserve and phase-boundary checks.

No scientific source/configuration or completed artifact changed. Markdown
notes are updated; parent review of the reporting extension described below
will proceed during validation, with no real-root notebook writes while the
worker is active. The prior launch-to-first-new-seal interval was about
3 hours 21 minutes; Phase 3 is now complete, so dispatch can reuse its frozen
decision rather than replay candidate generation. Still expect substantial
integrity/reporting work. Check actual process/lock, I/O, and notebook/checkpoint
timestamps rather than inferring a stall from unchanged trajectory counts.

Reporting review completed during this continuation. The separately versioned
`mnist_experiment/akaike_score_gate_reporting.py` was reviewed against the
actual audit schema. The glossary now clarifies resampling observations and
their actual labels, and distinguishes accepted optimizer-state variation
from certified MLE variance when fits fail. Forecast guides preserve the
data ranges instead of stretching realized-gain axes to large forecasts;
panel counts are labeled paired rows/recorded values, not implied replicas.
Only this new module and its tests were edited, outside the frozen source set.

Independent verification: all **365 Plan 14 unit tests passed in 104.57 s**,
including **27 reporting-extension tests**. All 31 scientific hashes still
match. The real Phase 3 scalar config/summary/audit/decision files were checked
against their integrity digests without reading tensor artifacts. The helper
passes the actual assignment/decision checks, reproduces 128 history fits
per condition and unresolved counts 16/11/2/22, and preserves missing/unresolved
forecast coverage. The actual notebook's phase headings are insertion-compatible.
Four preview plots were rendered separately; risk/covariance plots were visually
inspected. Latest reviewed previews:

```text
cache/mnist_experiment/akaike_score_gate/report_review_20261012T012636Z/
```

The main notebook remains untouched by the supplement while the writer is
live. Final integration is still required after the main worker exits and
its final refresh: run the CLI described under the historical pause below,
which performs a full integrity scan and rejects an active writer. Verify
the four embedded outputs and phase order afterwards. Earlier statements
that parent review/tests are pending are historical; notebook integration
is the only remaining reporting-extension task. Never patch frozen Python
sources or scientific artifacts to integrate this report-only addition.

Milestone at 20:19 PDT on October 11 (03:19 UTC on October 12): the first
complete Phase 4 replica is sealed in all four conditions (68 distinct
principal trajectories). Phase 4 has 70 trajectories and 76 units; the root
has 2,509 units. Replica 1's linear rotation is running. The saved arrival-88
full-depth trajectory completed; this continuation's first new marker is
19:36:57 PDT, about 1 hour 20 minutes after launch. Reusing the completed
Phase 3 decision reduced dispatch overhead relative to candidate replay.
Free disk: 75.8 GiB. No scientific source/configuration changed.

First-replica timing means: uninterrupted rotation units 79.49/82.53 seconds
(linear/sigmoid), mixture units 13.54/11.51 seconds (m18/m182). Exclude the
one known paused unit's 8-hour start-to-end interval from timing extrapolation
only; it remains in every scientific comparison. Online work and evaluation
are recorded separately. Optimization limitations remain: unresolved fits
166/2,030 linear, 158/2,046 sigmoid, 38/172 m18, 95/169 m182 in this replica,
largely concentrated in deep fixed LR controls. These are fit-call counts,
not independent replicas or final efficacy rates. Keep all accepted states.

`plan14.md` now reforecasts Phase 4 panels at 8--12 active hours before
triggered repairs (existing 4--8-hour repair allowance still provisional).
Phase 5 provisionally needs 35--50 active hours for 2,176 principal
trajectories, plus about 2--4 hours if one repair adds 128 assignments. Its
10k-image final panel is 2.5 times the 4k-image development panel; extrapolating
the measured 17-second rotation evaluation component adds about 25 seconds
per rotation. Count source/assets, checkpoints, and progressive reports too.
Refine before its final ledger from all Phase 4 rates. No sample counts,
cohort membership, or significance/early-stopping rules are changed.

Milestone at 03:19 PDT on October 12 (10:19 UTC): all 544 principal
development trajectories are sealed, with 136 in each condition (eight
replicas, 17 distinct policies). Scarcity/strict-budget panels are running;
at this checkpoint, Phase 4 has 547 trajectories and 579 units, 3,012 units
overall. No repair candidates or Phase 4 selection are frozen yet. The live
worker remains exec session 97523 / verified PID 4615; do not duplicate it.

All-principal mean uninterrupted unit times are 76.22/77.99 seconds for
linear/sigmoid (excluding the one known paused trajectory only from timing),
12.31/10.67 seconds for m18/m182. Mean rotation evaluation remains about
16.9 seconds. These support the current 35--50-hour provisional Phase 5
forecast, including its larger panel and fresh source/assets/reporting.
Keep cohort counts fixed; recheck when development selection/repairs complete.

Fit-call failures across these principal data are 1,601/16,069 linear,
1,667/16,200 sigmoid, 348/1,386 m18, and 654/1,386 m182. They are optimizer
operations, not independent trajectory counts or final inference. All
assigned trajectories and accepted states remain included. The notebook
has 74 figure outputs and no errors. Free disk: 71.9 GiB. Source code,
scientific configuration, and completed artifacts are unchanged.

## Latest Pause (Historical)

The user requested wrapping up compute to borrow the node. At 11:39:06 PDT
on October 11 (18:39:06 UTC), SIGTERM was sent to the verified orchestrator
PID 4457 after checking its actual held run lock and command line. It
checkpointed accepted work, refreshed the notebook, and exited with code 0.
At 11:40:45 PDT, the actual lock was free, the former process was absent,
and no Plan 14 experiment or reporting worker remained. **The node is free
of Plan 14 workers.** Native session 80100 and monitoring cell 39 are finished;
do not poll/reuse them. The completed reporting agent was also closed.

Root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
Notebook: `mnist_experiment/akaike_score_gate_results.ipynb`.
Root progress is `status="interrupted"`, Phase 4, 2,448 sealed units.
Nested Phase 4 still says `running` because its phase is unfinished, not
because a worker survives. Stale PID text alone does not imply a held lock.

- Phases 0--3 are complete: 4, 604, 48, and 1,777 units respectively.
  Phase 3 includes all 960 trajectories, 768 shadows, 16 variance cases,
  and its sealed calibration decision. Do not repeat that work.
- The frozen pointer is `decisions/phase3_calibration.json`: c=1,
  history half-life eight arrivals, gate cadence two arrivals. Its aggregate
  development constraint flag is false; individual folds/conditions can fail
  retention. No final efficacy claim is established.
- All 16 base variance-audit fits converged. Resampled refit failures are
  linear 16/128, sigmoid 11/128, m18 2/128, m182 22/128: 51/512 overall.
  No case has every refit unresolved. Keep all accepted states and report
  these as conditional finite-pool history diagnostics, not final replicas.
- Phase 4 has 15 sealed units: one source, one asset, and 13 trajectories
  from replica 0's linear condition. The next `depth6_lr` trajectory is saved
  at zero-based step 88, stage `arrival`, episode start 64, depth 6, with
  88 fit calls, 73 unresolved fits, and zero gate denials. Last recorded
  generic gate status is `budget_denied`; this is a fixed-depth policy with
  no gate. Preserve fit-health flags; do not infer an adaptive closed gate.
  Resume processes arrival 88, retaining the existing episode anchor/archive
  and reserved consolidation work; it must not repeat previous arrivals.
- Saved configuration is principal budget, episode length 32, width 5,
  LR EWC rank 8, score samples 64, omega 39, ridge ratio 0.1, solver
  max_iter 200/max_eval 300, residual tolerance 1e-5, float64. Fixed policies
  canonicalize unused gate cadence to 1. Snapshot steps are empty, as planned.
- Phase 4 still has its remaining principal/panel assignments and bounded
  repairs ahead. Aliases imply 816 distinct panel trajectories before repairs.
  Phase 5 has not started: 2,176 distinct principal trajectories, plus 128
  if one repair is promoted. Phase 6 and report closeout also remain.
- This invocation recorded 24,288.12 seconds (6.75 hours), before the final
  report refresh. The first new seal was at 08:15:00 PDT, about 3 hours
  21 minutes after launch. That launch-to-seal interval includes validation,
  reporting, and finishing the saved trajectory, not repeated sealed training.
  Do not interpret it as an exact measurement of pure validation cost.
- Free disk: 77.6 GiB. Maintain the 20-GiB reserve and review storage at
  phase boundaries. Do not delete or mutate sealed artifacts. Routine
  later trajectories have snapshots disabled; local repair checks can retain
  a bounded number. Phase 4 forecast remains provisional at 6--16 hours
  plus triggered repairs; remeasure after its first complete paired replica.
- Final notebook refresh completed at 11:39:58 PDT, with 57 figure outputs
  and no error outputs. All 31 frozen source hashes match. No scientific
  implementation/configuration or completed artifact was manually changed.

Saved checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase4/.incomplete/
trajectory__0__rotation_linear__depth6_lr__8d9ce3549c84179aaea0/checkpoint.pt
```

Wait for the user's next resume. Verify actual lock availability and process
identity, CUDA in `~/.venv`, and frozen source hashes, then use the unchanged
full-study `--resume` command in **Resume** below. It reuses Phases 0--3,
continues arrival 88, and runs remaining Phases 4--6 without another review
blocker. Expect substantial replay overhead from revalidating the larger
catalog. Check actual worker/lock, I/O, notebook/checkpoint timestamps, not
unchanged counts alone. Never relax convergence criteria, bypass source/parent
checks, reduce the fixed fresh cohort, or stop/extend it for significance.

### Reporting Extension Pending Review

The parallel agent completed new files outside the frozen source set:

- `mnist_experiment/akaike_score_gate_reporting.py`
- `test/unit/akaike_score_gate/test_reporting_extension.py`

Agent-reported result: 26 focused unit tests passed; frozen source hashes
remain unchanged. Parent code review, independent test execution, and
real-artifact integration are **not yet done**. The agent was closed after
completion (id `01a12c35-40f1-7981-9cc4-427be674611f`). Review these changes
while the next GPU continuation runs; do not duplicate its implementation.

API: `build_phase3_supplement(report)` creates tagged summary/plot cells from
verified scalar report data, with actual Phase 3 field names, separate
conditions, unresolved/missing coverage, conditional covariance limitations,
fit counts, and provenance. CLI, after review and only without a live writer:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate_reporting \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb
```

It rejects active writers, requires completed Phase 3, and inserts its own
tagged cells before Phase 4 without removing other cells; repeated invocation
should be idempotent. It retains the full integrity scan, so do not casually
run it on the large real root while the user needs the node. No real-root CLI
or notebook mutation was performed. During closeout, verify the plots against
actual audit scalars, run focused tests, and integrate after the main worker
exits. Frozen notebook refreshes discard the supplement, so reapply it after
the final refresh or a future authorized pause. Leave frozen experiment Python
sources untouched so the completed study remains normally resumable.

## Previous Continuation (Historical)

The user requested resumption after temporary node reuse. At 04:54 PDT on
October 11, the previous worker was absent and the actual run lock was free.
CUDA float64 execution in `~/.venv`, all 31 frozen source hashes, and the
saved checkpoint below were verified. The unchanged full-study command in
**Resume** was restarted with `--resume`: **exec session 80100, PID 4457**.
Its expected command line and held lock were verified outside the sandbox.
Session identifiers are not portable; verify the actual lock and command line
before launching another writer or signaling any stored PID.

Restart inventory is unchanged from the historical pause below: 1,635
sealed units overall, including 947 Phase 3 trajectories (979 Phase 3 units).
The saved current-batch-only linear `asg_lr_lr` trajectory is at step 113,
stage `gate`, depth 1, after its accepted fit. Resume continues that gate.
Thirteen trajectories remain, followed by 768 local shadows and 16
history-variance cases. Phases 4--6 remain authorized without another review
blocker; no final calibration decision exists yet. Free disk: 145.2 GiB.

Only Markdown notes were manually edited. Keep the existing float64 config,
output root, residual certificate, source hashes, and fixed fresh cohort.
Preserve all sealed artifacts and monitor storage. Replay now traverses a
larger catalog than the prior restart, whose first new seal followed about
2 hours 46 minutes of verification/reporting. Expect substantial overhead;
unchanged counts alone are not a stalled worker. Verify liveness through the
actual lock/process, I/O, notebook timestamps, and checkpoint advancement.
The earlier timing table is explicitly outdated as a forecast. Do not silently
shrink the study or bypass integrity checks to meet that allowance.

Milestone at 10:10 PDT on October 11 (17:10 UTC): all 960 Phase 3 trajectories
are sealed, including the entire 64-trajectory current-batch-only ablation.
This continuation added the 13 remaining trajectories, beginning with the
saved gate checkpoint. Its first completion marker is timestamped 08:15:00
PDT, about 3 hours 21 minutes after launch; preceding replay was integrity/
reporting overhead, not repeated training. Local shadows are now running,
with 27 sealed cases and 1,675 units overall at this milestone. No variance
cases or final calibration decision exist yet; Phases 4--6 have not started.
Free disk: 124.0 GiB. Notebook error outputs remain absent. Keep monitoring
storage and audit/fit health. The first 43 shadows subsequently measured
a median 1.69 seconds per audit unit, excluding driver loading and progressive
report refreshes; they are much quicker than the current-batch ablation.
The live worker remains exec session 80100 / verified PID 4457.

Storage check at 10:44 PDT on October 11: 566 of 768 local shadows are
sealed. The first 561 units occupy 30.89 GiB (mean 56.38 MiB), projecting
about 11.4 GiB for their remaining assignments. Free disk is 94.3 GiB.
Variance-case sizes still need measurement. Routine Phase 4/5 trajectories
have snapshots disabled; only bounded local repair checks retain histories.
Those later paths are therefore much smaller than current-batch controls.
The earlier 80-GiB monitor threshold is an inspection alert, not an automatic
scientific stop. After this inspection, continue monitoring with a 20-GiB
reserve and recheck required storage at phase boundaries; gracefully halt
if remaining work would exceed available space. Never prune completed units
or weaken artifact requirements to fit storage. No source code changed.

Milestone at 11:00 PDT on October 11 (18:00 UTC): all 768 local shadows are
sealed, along with the first three of 16 history-variance cases. There are
2,419 sealed units overall (1,763 Phase 3 units). All three cases have a
converged restricted fit and zero unresolved histories among their 32 refits
each. Remaining cases must still be checked; these are conditional research
diagnostics, not final-replica evidence. The first three variance units occupy
0.81 GiB. Free disk: about 82 GiB. No final calibration decision exists yet.

`plan14.md` now publishes a pre-Phase-4 forecast of 6--16 active hours for
the unchanged principal and scarcity/budget panels, plus triggered bounded
repairs (earlier allowance 4--8 hours). This extrapolates the baseline
Phase 3 medians (rotation 113--156 seconds; mixture 9--20 seconds by policy),
recognizing routine Phase 4 has no snapshot histories and other fixed depths/
budget tiers need measurement. Refine this after first paired development
replicas. Reforecast Phase 5 before its final ledger, counting its larger
holdout and fresh cohort. Its original 6--14-hour allowance is unvalidated.

Phase 3 completed at 11:16:55 PDT on October 11; Phase 4 began at 11:17:41
PDT. Phase 3 has 1,777 sealed units: 960 trajectories, 16 sources, 16 assets,
768 shadows, 16 variance cases, and one calibration decision. All phases
through Phase 3 total 2,433 units; the root counter briefly lagged at 2,432.
The frozen pointer is `decisions/phase3_calibration.json`, not a JSON file
directly under the root. It selects c=1, half-life eight arrivals, cadence two
arrivals. Its tie-aware development rule reports no aggregate constraint
failure; do not infer every condition/replica satisfies retention. Out-of-fold
selection uses only confirmation replicas 4--7 for scoring (cadence two in
three folds, cadence four in one), not a final efficacy test.

All 16 base variance-audit fits converged. Unresolved resampled refits are
16/128 linear, 11/128 sigmoid, 2/128 mixture m18, and 22/128 mixture m182;
51/512 overall. No state has every refit unresolved. Preserve all assigned
accepted states and label these conditional finite-pool covariance diagnostics,
not independent replicas. Free disk at Phase 4 entry: 77.8 GiB. Continue the
20-GiB reserve and phase-boundary storage checks. Routine later paths have
snapshots disabled. All scientific source/configuration remains unchanged.

With c=1, the `asg_uncalibrated` threshold comparator aliases `asg_lr_lr`
(same history/cadence). Phase 4 therefore has 544 principal +272 additional
scarcity/budget trajectories =816 distinct panel trajectories, before repairs.
Phase 5 projects 2,176 distinct principal trajectories, plus 128 if one repair
is promoted. Nominal ledgers still record the full policy list; aliases do not
add independent information. Use the unchanged worker (80100 / PID 4457).

Reporting follow-up: the refreshed notebook has 52 figures and no error
outputs, but Phase 3 shadow/variance charts are missing. `_audit_figures`
expects generic pairs such as `forecast_gain` / `realized_gain`, whereas the
actual schema has `realized_weighted_training_gain`, `realized_next_gain`,
and nested covariance errors. The scalar artifacts are complete. Do not
patch the frozen renderer while this run is active. Complete artifact-only
Phase 3 diagnostic plots during report closeout using a separately versioned
reporting extension outside the frozen source set, preserving compute-run
compatibility and pairing. Add focused tests for the real audit schema.

## Latest Pause (Historical)

The user requested halting for node reuse. At 03:55 PDT on October 11
(10:55 UTC), SIGTERM was sent to the verified orchestrator PID 4559 after
checking its actual held run lock and command line. It checkpointed accepted
work, completed its final notebook refresh, and exited with code 0. At
03:56:59 PDT, the actual run lock was available, the former process no
longer existed, and no Plan 14 experiment/reporting worker was present.
**The node is free of Plan 14 workers.** Exec session 96322 and monitoring
cell 31 are finished; do not reuse them. Stale PID text is not a live lock.

Main root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
Notebook: `mnist_experiment/akaike_score_gate_results.ipynb`.
Root progress reports `status="interrupted"`, Phase 3, and 1,635 sealed units.
Its nested Phase 3 status still says `running` because the phase is incomplete;
it does not indicate a surviving worker.

- Phases 0--2 remain complete: four validation units, 604 Phase 1 units,
  and 48 Phase 2 audits. Their sealed results were reused without retraining.
- Phase 3 has 947 trajectories, 16 sources, and 16 assets: 979 sealed units.
  This continuation added 157 trajectories. The initial screen and all four
  shortlisted candidates' confirmation are complete; the shortlist is frozen.
- Three of four current-batch-only replicas are complete in all conditions.
  Replica 3's linear `asg_lr_lr` trajectory is checkpointed at zero-based
  step 113, stage `gate`, episode start 113, depth 1. It has 182 fit calls,
  nine unresolved fits, zero gate denials, and an accepted converged current
  fit. Archive compute is reserved and the probe is scheduled. Resume performs
  this gate next; do not repeat the already accepted fit.
- This is the current-batch-only protocol: `episode_arrivals=1`, c=1,
  half-life=8, cadence=1, rank=8, score samples=64, rotation omega=39,
  ridge ratio=0.1, max_iter=200, max_eval=300, residual tolerance=1e-5.
- Thirteen Phase 3 trajectories remain, including this checkpoint: one
  linear, four sigmoid, and eight mixture assignments. Then follow all
  768 baseline local shadows and 16 history-variance cases (32 refits each).
  No final calibration decision exists. Phases 4--6 have not started but
  remain authorized after the user's next resume request.
- Recorded invocation time is 65,653.83 seconds (18.24 hours), before the
  final report refresh. This includes substantial verification/checkpointing/
  reporting overhead and is not policy compute or a remaining-time estimate.
  The first new trajectory sealed about 2 hours 46 minutes after launch.
- Free disk: 145.2 GiB. Monitor storage; current-batch rotation units retain
  about 3.6 GiB each. Preserve every completed artifact, including checkpoints.
- Final notebook refresh completed at 03:56:30 PDT, with 44 figure outputs
  and no error outputs. All 31 frozen source hashes match. Only Markdown
  status notes were manually edited. Scientific sources/configuration remain
  unchanged; the previously recorded 338 passing unit tests still apply.

Saved checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase3/.incomplete/
trajectory__3__rotation_linear__asg_lr_lr__3121241f2f3ce8d5df85/checkpoint.pt
```

Wait for the user's next resume. Verify actual lock availability, worker
identity, CUDA in `~/.venv`, and frozen source hashes, then use the unchanged
full-study `--resume` command in **Resume** below. Continue this gate and the
remaining Phases 3--6 without another review blocker. Do not weaken the
residual certificate, change the frozen cohort, retrain sealed units, or
optimize frozen Python sources in place. Restart replay may now take longer
than the last 2-hour-46-minute overhead; completed trajectory counts can stay
unchanged during validation/reporting. Verify liveness through the actual
worker/lock, I/O, notebook timestamps, and checkpoint advancement.

The timing table in `plan14.md` is explicitly marked outdated as a forecast:
the current-batch ablation alone projects roughly 16--20 hours from measured
early rates, and later-phase rates are still unmeasured. Keep research
overhead separate from online-policy compute and publish revised forecasts
as later phases run. No scope was reduced to meet optimistic estimates.

## Previous Continuation (Historical)

The user requested resumption again after an interruption. At 09:40 PDT on
October 10, the actual run lock was free and the previous worker PID 4632
no longer existed. Root progress still said `running`; it was stale, not
evidence of a surviving process. CUDA float64 execution in `~/.venv` and
all 31 frozen source hashes were verified before the unchanged full-study
command in **Resume** below was restarted with `--resume`.

At 09:41 PDT, the new worker was verified as **exec session 96322, PID 4559**,
with the expected command line and held run lock outside the sandbox.
Session identifiers are not portable. Always verify the actual lock and
command line before launching another writer or signaling a stored PID.

Restart state:

- Phases 0--2 remain complete: four validation units, 604 Phase 1 units,
  and 48 Phase 2 audits, all reused without retraining.
- Phase 3 has 790 sealed trajectories, 16 sources, and 16 assets: 822 units
  and 1,478 sealed units overall. The 608-trajectory screen and four-candidate
  shortlist are frozen. Baseline and c=2 confirmation are complete;
  cadence-2 confirmation is in progress on replica 5's sigmoid rotation.
- The saved `asg_lr_diag` trajectory is at zero-based step 87, stage
  `advance`, depth 1, with 90 fit calls, zero unresolved fits, and zero
  gate denials. Resume advances the accepted step without repeating its fit.
- Remaining cadence confirmation, 64 current-batch-only trajectories,
  768 baseline local shadows, and 16 history-variance cases still follow.
  Phase 3 still projects 960 distinct trajectories. No final calibration
  decision exists; Phases 4--6 remain authorized but have not started.
- Free disk at restart: 267.1 GiB. Monitor storage and preserve every sealed
  artifact. Do not shorten the fresh cohort or stop for significance.

Current saved checkpoint at restart:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase3/.incomplete/
trajectory__5__rotation_sigmoid__asg_lr_diag__86767e527f607590c267/checkpoint.pt
```

Only Markdown status notes were manually changed. Scientific implementation,
configuration, residual certificate, and sealed artifacts remain unchanged.
The last restart took 2 hours 29 minutes to reach its first new seal; replay
now traverses a larger catalog and may take longer. Unchanged trajectory
counts during integrity checking/report refresh are not by themselves a
stall. Keep replay cost separate from new training and verify liveness via
the actual worker/lock, I/O, notebook timestamps, and checkpoint advancement.

Milestone at 13:42 PDT on October 10 (20:42 UTC): Phase 3 has 848 sealed
trajectories and 880 units, with 1,536 sealed units overall. Cadence-2
confirmation is complete and the cadence-4 candidate is underway. This
continuation has added 58 trajectories. The first new completion marker is
timestamped 12:27:20 PDT, about 2 hours 46 minutes after launch; the
completed-catalog replay did not retrain sealed units. Free disk is 259.5 GiB,
and the refreshed notebook has no error outputs. Current-batch-only runs,
all local shadows and history-variance audits, and Phases 4--6 remain ahead.
The worker remains exec session 96322 / verified PID 4559. No scientific
implementation, configuration, or completed artifact was manually changed.

Milestone at 19:44 PDT on October 10 (02:44 UTC on October 11): all candidate
confirmation is complete. Phase 3 has 914 sealed trajectories (946 units),
with 1,602 units overall. One complete current-batch-only replica across the
four conditions is sealed; replica 1's linear rotation is in progress.
Free disk: 212.6 GiB. Shadows and variance cases have not started.

The current-batch-only ablation is materially slower and larger than the
screened protocols: the first complete replica took approximately four hours,
suggesting 16--20 hours for all 64 trajectories at those rates. A representative
rotation trajectory required about 117 seconds of online work but 24 minutes
to seal, with approximately 3.6 GiB in its required completed artifacts.
Repeated history checkpointing and report reads dominate that overhead.
`plan14.md` now explicitly marks its previous Phase 3 and subtotal timing
ranges as outdated forecasts. Keep scope unchanged, monitor disk, and replace
the whole-study forecast after later-phase rates are measured. Do not delete
sealed checkpoints or optimize frozen Python sources in place.

Milestone at 03:46 PDT on October 11 (10:46 UTC): Phase 3 has 947 sealed
trajectories (979 units), with 1,635 units overall. Three of four
current-batch-only replicas are complete in all four conditions; replica 3's
linear `asg_lr_lr` trajectory is running, followed by its sigmoid and mixture
conditions. Current checkpoint directory:

```text
phase3/.incomplete/trajectory__3__rotation_linear__asg_lr_lr__3121241f2f3ce8d5df85/
```

Free disk: 144.0 GiB. Storage growth matches the large per-episode histories;
no sealed artifacts have been pruned or modified. All 31 frozen source hashes
match. The notebook has 44 rendered figure outputs and no error outputs;
a representative all-variant development chart was visually checked.
No local shadows or history-variance cases are sealed yet. The live worker
remains exec session 96322 / verified PID 4559; remaining phases are still
authorized. No code or scientific settings changed during this continuation.

## Previous Continuation (Historical)

The user requested resumption after a chat interruption. At 00:24 PDT on
October 10, the actual run lock was free and the previous orchestrator was
no longer running, despite stale root progress saying `running`. Its old
PID 4514 had been reused by the Codex host, not the experiment. **Never signal
a PID from stale lock text without checking the actual lock and command line.**
The interrupted monitoring cell no longer exists. Old exec session 44413 is
not the current worker and must not be reused.

CUDA float64 execution in `~/.venv` and all 31 frozen source hashes were
verified. At 00:26 PDT on October 10 (07:26 UTC), the unchanged full-study
command in **Resume** below was restarted with `--resume`, in exec session
39665, PID 4632. The new worker's command line and held lock were verified
outside the restricted sandbox. Session identifiers are not portable;
sandbox `/proc` does not expose the host worker.

Restart state:

- Phases 0--2 are complete: four validation units, 604 Phase 1 units, and
  48 Phase 2 audits. Their sealed scientific results remain unchanged.
- Phase 3 has 724 units: 692 trajectories, 16 sources, and 16 assets.
  There are 1,380 sealed units overall. The stale top-level counter was
  1,378; inventory, not that lagged counter, establishes completed work.
- The initial screen's 608 trajectories and `phase3_shortlist.json` are
  frozen. Baseline confirmation is on replica 7's mixture conditions.
  Twelve baseline trajectories remain, including the saved m18 run.
- The other three shortlisted candidates, 64 current-batch-only runs,
  768 baseline local shadows, and 16 history-variance cases still follow.
  The projected Phase 3 total remains 960 distinct trajectories, not the
  earlier 1,024 upper bound. No final calibration decision exists yet.
- The saved trajectory is replica 7 / mixture m18 / `asg_diag_diag`,
  with c=1, half-life=8, cadence=1, rank=8, and 64 model-score draws.
  Its checkpoint is at zero-based step 8, stage `fit`, depth 4, with
  16 fit calls, one unresolved fit, and zero gate denials. The next fit
  has not been accepted; resume continues that stage once.
- Free disk: 279.8 GiB. Preserve sealed artifacts and monitor storage.
  Phases 4--6 remain authorized but have not started.

Current saved checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase3/.incomplete/
trajectory__7__mixture_m18__asg_diag_diag__cb48c0854557614a6a00/checkpoint.pt
```

Keep the existing configuration, output root, float64 arithmetic, 1e-5
residual certificate, and frozen source/parent checks. Only Markdown notes
were edited; no scientific implementation, configuration, or sealed artifact
was changed. The full remaining sequence is authorized without a review
blocker. Restart verification/reporting again precedes new work and may take
over two hours. Root phase/status and notebook timestamps may reflect replay;
the actual lock and worker identity establish liveness. Do not stop/extend
the fresh cohort for significance or confuse unresolved fits with closed gates.

Milestone at 03:29 PDT on October 10 (10:29 UTC): Phase 3 has 723 sealed
trajectories and 755 units; the root has 1,411 units. The saved m18 trajectory
completed, and baseline confirmation is finished for replicas 4--7 in all
four conditions. The c=2 / half-life=8 / cadence=1 candidate is now on replica
5's linear rotation. The other cadence candidates, current-batch-only runs,
and all shadows/variance audits still follow. No final calibration decision
exists, and Phases 4--6 have not started. Free disk: 276.2 GiB. No scientific
settings or completed artifacts changed during this continuation.
The first newly sealed trajectory's completion marker is timestamped 02:55:05
PDT, about 2 hours 29 minutes after launch. This restart's longer replay cost
reflects the larger completed catalog and its reporting, not repeated training.

## Previous Continuation (Historical)

The user requested resumption. At 18:50 PDT on October 9 (01:50 UTC on
October 10), CUDA float64 execution in `~/.venv`, availability of the actual
run lock, and all 31 frozen source hashes were verified. The unchanged
full-study command in **Resume** below was restarted with `--resume`, in
exec session 44413, PID 4514. The worker's command line and held lock were
verified outside the restricted sandbox. Session identifiers are not
portable, and sandbox `/proc` does not expose the host worker.
**Verify the actual lock before launching another writer or signaling a PID.**

At restart, Phases 0--2 were complete and Phase 3 had 599 sealed trajectories
(1,271 sealed units overall). The saved sigmoid-rotation trajectory described
below is the next unfinished unit. Nine initial-screen assignments remain,
including this checkpoint, before shortlist confirmation and research audits.
Phases 4--6 are authorized to follow without another review blocker.
Free disk at restart: 289.4 GiB; preserve sealed artifacts and monitor storage.

The previous restart took about 2 hours 15 minutes to reach new compute
because completed blocks trigger large integrity reads and report refreshes.
Expect substantial replay overhead before this saved gate advances. Root
progress can temporarily retain the historical interruption or show earlier
phases while the dispatcher checks their completed units. Unchanged counts
alone do not indicate a stalled worker. Check its actual lock, I/O, notebook
timestamps, and checkpoint advancement. No implementation or configuration
was changed to bypass checks or shorten the scientific design.

Only Markdown status notes changed during restart. Continue with the same
float64 configuration, source hashes, 1e-5 residual certificate, four separate
conditions, fixed fresh cohort, and immutable result root. Do not pool smoke
runs, retrain completed screens, or stop/extend the final cohort for favorable
significance. The earlier pause below is historical, not the live worker state.

Milestone at 21:13 PDT on October 9 (04:13 UTC on October 10): all 608
initial-screen trajectories are sealed, and `phase3_shortlist.json` is frozen.
Phase 3 has 624 units including its eight sources and eight assets; the root
has 1,280 units. This restart completed the nine remaining screen assignments,
including the saved sigmoid gate. The worker continues into confirmation.
No final `phase3_calibration` decision exists yet.

The frozen shortlist contains the baseline (c=1, half-life=8, cadence=1),
plus (c=2, half-life=8, cadence=1), (c=1, half-life=8, cadence=2), and
(c=1, half-life=8, cadence=4). Confirmation covers development replicas 4--7
in all four conditions and all four EWC/gate representation combinations.
All shortlisted candidates share half-life 8, so their full-depth controls
deduplicate: 256 new adaptive trajectories plus 32 controls, rather than the
352-trajectory upper bound. With the 64 current-batch-only runs, Phase 3 now
projects 960 distinct trajectories in total. The 768 baseline local shadows
and 16 history-variance cases are still required. These counts reflect the
frozen selection and legitimate reuse, not a reduced scientific design.

Free disk at this milestone: 288.4 GiB. Restart verification and completed-block
reporting again took over two hours before new scientific work resumed. Keep
that overhead separate from training, and continue storage/fit-health checks.

## Latest Pause (Historical)

The user explicitly requested another temporary pause. At 15:56 PDT on
October 9 (22:56 UTC), SIGTERM was sent to the verified orchestrator PID 4489.
It checkpointed accepted work, refreshed the notebook, and exited with code 0.
The actual run lock is available and the former worker process no longer
exists. **No Plan 14 experiment or reporting worker remains running.** Exec
session 60282 is completed; do not poll or reuse it. The monitoring cell was
also terminated. Stale PID text in `.run.lock` does not indicate a live worker.

Main root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
Notebook: `mnist_experiment/akaike_score_gate_results.ipynb`.
Root `progress.json` reports `status="interrupted"`, Phase 3, and 1,271 sealed
units. Its nested Phase 3 status can still say `running`; the phase is
incomplete, but the worker has exited.

- Phases 0--2 remain complete: four validation units, 604 Phase 1 units,
  and 48 Phase 2 Fisher audits. Their scientific results were reused.
- Phase 3 now has 599 sealed trajectories, eight sources, and eight assets
  (615 units). This invocation added 22 completed trajectories. The initial
  screen has 608 distinct assignments; nine remain, including the checkpoint.
- No `phase3_shortlist.json` or final calibration selection exists yet.
  Shortlist confirmation, current-batch-only runs, 768 baseline local shadows,
  and 16 history-variance cases still follow. Phases 4--6 have not started.
- The saved trajectory is replica index 3, sigmoid rotation, `asg_lr_lr`:
  c=1, half-life=8, cadence=4, rank=8, and 64 score samples. Its checkpoint is
  at zero-based step 36, stage `gate`, depth 4, with 48 fit calls and zero
  unresolved fits or gate denials. Resume continues the gate after the
  accepted fit; it must not repeat that fit.
- Recorded invocation time is 10,038.95 seconds (2.79 hours), including
  substantial restart verification/reporting overhead and new trajectory
  work. This is not cumulative study training time or a remaining-time
  prediction. The final report refresh adds a little more wall time.
- Free disk: 289.4 GiB. Preserve every sealed artifact and monitor storage,
  particularly during shadows and fresh-cohort execution.
- Final notebook refresh completed at 15:57:40 PDT with no error outputs.
  All 31 frozen source hashes still match. Only Markdown status notes were
  manually changed; no scientific implementation or configuration changed.
  The previously recorded 338 passing unit tests remain applicable.

Active checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase3/.incomplete/
trajectory__3__rotation_sigmoid__asg_lr_lr__0c69062634dd4d0aeab2/checkpoint.pt
```

Wait for the user's next resume request. Then verify CUDA and the actual lock,
and use the unchanged command in **Resume** below. It continues this
checkpoint and all remaining authorized Phases 3--6 without an additional
review blocker. Keep `default_float64.json` and the existing output root.
Never bypass source/parent compatibility checks, weaken the 1e-5 residual
certificate, retrain completed screens, or change the fixed fresh cohort.

Operational caveat: this restart took about 2 hours 15 minutes to reach its
first newly sealed trajectory because the dispatcher revalidated large
artifacts and force-refreshed reports while traversing completed blocks.
The worker was active throughout; unchanged counts alone were not evidence
of a stall. Keep this cost distinct from new scientific training. The frozen
source was not modified to optimize this overhead. The latest forecasting
tables below remain provisional and do not predict remaining calendar time.

## Previous Continuation (Historical)

The user requested resumption. At 13:09 PDT on October 9 (20:09 UTC), CUDA
float64 execution in `~/.venv` and availability of the actual run lock were
verified. The unchanged full-study command below was restarted with
`--resume`, in exec session 60282, PID 4489. The worker's command line and
held lock were verified outside the restricted sandbox; sandbox `/proc`
does not expose the host worker. Session identifiers are not portable.
**Verify the actual lock before another writer or before signaling a worker.**

At restart, Phases 0--2 were complete and Phase 3 had 577 sealed trajectories.
The saved linear-rotation trajectory below continues before the remaining
31 initial-screen assignments, shortlist confirmation, and research audits.
Phases 4--6 are authorized to follow without an intermediate review blocker.
The first reporting/checksum refresh can take time; root progress may still
show the historical interruption until dispatch resumes. Do not infer worker
state from stale PID text or a stale progress counter alone.

Free disk at restart: 292.6 GiB. Preserve all sealed outputs and monitor
storage. Only Markdown status notes changed; no scientific configuration,
implementation source, or completed artifact was changed by this restart.
The timing forecasts and scientific cautions below remain applicable.

Restart milestone: by 15:25 PDT on October 9 (22:25 UTC), the saved
`asg_diag_lr` linear trajectory was sealed and Phase 3 had 578 trajectories.
The next trajectory, replica 2 / linear rotation / `asg_lr_diag`, was actively
checkpointing. Root status is `running`, Phase 3, with 1,250 sealed units.
The completed-phase replay finished its Phase 2 checks at 14:06 PDT and then
traversed existing Phase 3 candidate blocks. It took about 2 hours 15 minutes
from launch to the first newly sealed trajectory: large integrity reads and
forced report refreshes on completed blocks add substantial restart overhead.
This is not new scientific training or a reason to recreate sealed units.
The resume dispatcher updates phase timestamps during verification, so the
historical scientific completion times below are distinct from latest replay
timestamps. CUDA's known adaptive-pooling nondeterminism warning appeared
when actual fitting resumed; the tested resume tolerance remains applicable.

## Previous Pause (Historical)

The user explicitly requested stopping for temporary node reuse. At 12:04
PDT on October 9 (19:04 UTC), SIGTERM was sent to the verified orchestrator
PID 4560. It checkpointed accepted work, refreshed the notebook, and exited
with code 0. The actual run lock is available and the former worker process
no longer exists. **No Plan 14 experiment or reporting worker remains running.**
Exec session 71884 is completed; do not attempt to poll or reuse it. The old
PID text in `.run.lock` is harmless and is not evidence of an active worker.

Main root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
Notebook: `mnist_experiment/akaike_score_gate_results.ipynb`.
Root `progress.json` now reports `status="interrupted"`, Phase 3, and
1,249 sealed units. The unfinished phase's nested status can still say
`running`; that describes an incomplete phase, not a live process.

- Phases 0--2 are complete: four main validation units, 604 Phase 1 units,
  and all 48 Phase 2 Fisher audits.
- Phase 3 has 577 sealed trajectories, eight sources, and eight assets.
  These cover the first four screening replicas. The initial screen has
  608 distinct trajectory assignments; 31 remain, including the checkpointed
  trajectory.
- `phase3_shortlist.json` and final calibration selection do not exist yet.
  Larger-cohort confirmation, current-batch-only runs, local shadows, and
  history-variance cases still follow. Phases 4--6 have not started.
- The interrupted trajectory is replica 2, linear rotation, `asg_diag_lr`,
  with c=1, half-life=8, cadence=4, rank=8, and 64 score samples. It is saved
  at zero-based step 49, stage `evaluate`, after 60 fits and zero unresolved
  fits. Resume should evaluate the already accepted state, not refit it.
- Current invocation active time was 61,076.62 seconds (16.97 hours),
  including its remaining Phase 1 work, Phase 2, and part of Phase 3. This
  is neither cumulative study compute nor remaining-time prediction.
- About 291 GiB remains available. Phase 2's immutable outputs alone occupy
  178 GiB; continue checking storage. No sealed artifacts were deleted.
- Notebook refresh completed with no error outputs. Representative Phase 3
  learning/capacity plots were visually checked. No efficacy conclusion or
  new significance claim was made during screening.

Active checkpoint:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase3/.incomplete/
trajectory__2__rotation_linear__asg_diag_lr__42ba0bf749868f9734bc/checkpoint.pt
```

Restart using the **Resume** command below after checking CUDA and the actual
lock. Keep `default_float64.json`, the existing output root, and `--resume`.
The command revalidates frozen parents, skips sealed units, continues this
checkpoint, then executes the remaining authorized Phases 3--6 without a
review blocker. Do not reset the output root, change source hashes, weaken
the 1e-5 certificate, or recreate the completed screens. Only Markdown status
notes were manually edited in this continuation; implementation/configuration
sources are unchanged. The latest unit suite remains 338 passed, recorded
below; no new code change required rerunning it.

Latest timing forecast: 18--30 active hours for Phase 3, 45--78 for the full
default study, and 49--86 including bounded repairs. Phase 3 has accrued
about 13.2 hours since its start, but confirmation/audit rates remain
unmeasured. Calendar intervals while the user repurposes the node do not
count as active compute. Keep the four scientific conditions separate and
the fresh evaluation cohort fixed; do not extend or stop it for significance.

## Earlier Continuation (Historical)

At 02:55 PDT on October 9, Phase 3 had sealed 162 trajectories plus eight
sources and eight assets, after about four hours of execution. No shortlist
or final calibration decision was frozen yet. The full existing design
contains up to 1,024 distinct trajectories: 608 initial screen trajectories
(512 adaptive plus 96 full-depth controls across three half-lives), up to
352 additional confirmation trajectories, and 64 current-batch-only runs.
All 128 baseline adaptive trajectories then receive six local shadows
(768 audits), followed by 16 history-variance cases with 32 refits each.
The main worker remains exec session 71884 / PID 4560.

`plan14.md` now forecasts 18--30 active hours for Phase 3, 45--78 hours
for the whole default study, and 49--86 including bounded repairs. These
replace the optimistic 3--8-hour Phase 3 assumption; scientific scope,
source hashes, and settings remain unchanged. Later-phase rates are still
unmeasured. Disk inspection: Phase 0 128 MiB, Phase 1 41 GiB, Phase 2
178 GiB, and Phase 3 19 GiB, with 339 GiB available. Storage is adequate
under current projections but must be monitored, especially as shadow
and fresh-cohort artifacts accumulate. Never delete sealed artifacts to
free space without the user's explicit direction.

Phase 2 completed at 22:50 PDT on October 8 (05:50 UTC on October 9),
after approximately 2 hours 51 minutes: all 48 preassigned Fisher audits
sealed, with 9,440 recorded audit rows. No case has its
`all_restricted_fits_unresolved` flag set. Unresolved restricted rows remain
recorded: linear rotation 0, sigmoid rotation 345, mixture m18 348, and
mixture m182 230. These row counts include repeated fixed-state diagnostics,
not independent replica failure rates. Twelve Phase 2 figures were rendered
and representative geometry, forecast, and attribution figures visually
checked; the notebook has no error outputs. Phase 3 started at 22:51 PDT
in the same exec session 71884 / PID 4560 and is proceeding autonomously.
Phase 3's calibration and selection are not yet complete; Phases 4--6 remain.

Earlier milestone: Phase 1 completed at 19:59 PDT on October 8 (02:59 UTC
on October 9), with 604 sealed units: 492 trajectories (360 solver pilots
plus baseline/replacement trajectories), 48 source fits, eight assets,
48 local depth audits, and eight protocol/selection units. The frozen
`phase1_selection.json` contains 112 selected-retention driver assignments
and all 48 preassigned audit cases. Phase 2 then started in the same worker,
exec session 71884 / PID 4560; verify the lock before another writer.
Top-level progress counts can lag throughout an invocation, not just within
Phase 1. The phase status and sealed inventory are more useful live signals.

Descriptive local depth audits (12 preassigned states per condition, four
replicas with three states each) selected varying depths. All 48 head-only
fits were certified; 257 of 288 fits across six depths were certified.
Mean independent scoring-panel regret of head-only relative to the
selection-panel-chosen local depth was 0.01525 nat for mixture m18,
0.02608 for mixture m182, 0.02647 for linear rotation, and 0.03979 for
sigmoid rotation. These are local research diagnostics, not online efficacy,
independent replica intervals, or an achievable oracle performance claim.

The latest restart followed another temporary node reuse. At 19:06 PDT on
October 8 (02:06 UTC on October 9), the old worker's lock was confirmed
released and CUDA was verified. The unchanged full-study command was
restarted with `--resume`, in exec session 71884, PID 4560. Verify the live
lock before launching another writer; session identifiers are not portable.
Phase 1 had 474 sealed trajectories at this restart: all 360 solver pilots
plus fixed-depth baselines and selected-retention replacements. Its next
mixture trajectory was checkpointed at step 5, stage `fit`. At that restart,
Phase 1's depth audits and Phases 2--6 were still pending. No scientific
settings or completed artifacts were changed by this restart.

The solver decision is frozen at 200 iterations and the unchanged 1e-5
residual certificate. Across its complete 120-trajectory pilot grid, 2,500
of 7,800 fits (32.05%) were unresolved; online-policy work was 18,547.73
seconds (5.15 hours). `unresolved_at_ceiling` is explicitly true. Do not
interpret unresolved gate opportunities as ordinary closed-gate decisions.
The family-common retention decision keeps rotation's omega 39 / ridge
ratio 0.1 and selects mixture's omega 19 / ridge ratio 0.01. Both satisfy
the pilot mean source-retention constraint; these are development decisions,
not final efficacy evidence.

Earlier continuation: the user requested resumption after repurposing the
node. At 06:25 PDT on
October 8 (13:25 UTC), the old worker's lock was confirmed released and CUDA
was verified on the RTX 4070. The unchanged full-study command below was
restarted with `--resume`, in exec session 93423, PID 4892. Verify the live
lock before launching another writer; session identifiers are not portable.

At this restart, Phase 1 had sealed all 120 trajectories at the 50-iteration
ceiling and 104 of 120 at the 100-iteration ceiling. The next trajectory,
replica 1 / linear rotation / depth 6, was checkpointed at step 112, stage
`advance`, with 113 fits. No completed artifacts or scientific settings were
changed. Phase 0 remains complete; Phases 2--6 have not started. The planned
200-iteration screen is still required. Top-level `progress.json` counters
lag within Phase 1, so inspect sealed inventories and active checkpoints as
well. The historical checkpoints and timing measurements below are context,
not the current worker state.

At approximately 10:27 PDT on October 8, the complete 100-iteration screen
was sealed: 120 trajectories, 7,800 fits, 3,256 unresolved (41.74%),
12,582.43 seconds of online-policy work, and 1,014.56 seconds of research
evaluation. The planned 200-iteration screen had sealed 68 trajectories
(4,420 fits, 1,284 unresolved). Its incomplete trajectory was advancing
normally. These partial 200-iteration results are not a paired full-screen
comparison. Notebook phase headings and outputs were checked for errors;
the development-variant figures render and keep variants separate.

Continuation: the replacement agent verified the frozen source and study
hashes and resumed the full run at 23:25 UTC on October 7. The state below
documents the handoff checkpoint; consult `progress.json` for live status.

Execution milestone: at approximately 02:32 UTC on October 8 (October 7 in
the client's timezone), all 120 retention-pilot trajectories at the
50-iteration ceiling were sealed. The sealed optimizer-pilot summary records
7,800 fits, with 3,834 unresolved (49.15%) at the unchanged 1e-5 certificate.
The planned common 100-iteration pass is starting. The resumed invocation
has accrued about 3.1 hours; those 120 trajectories record 2.20 hours of
online-policy work and 17.2 minutes of research evaluation. Other setup,
validation, checkpointing, and reporting add wall time. Phase 1 is not yet
complete, and these pilots are not final ASG evidence. The main orchestrator
remains active, unbounded through Phase 6, in exec session 96498; verify its
live lock before attempting another writer. During Phase 1 the top-level
progress counter lags; use sealed inventories and checkpoints too.

At 04:44 UTC, the first replica's 60 trajectories at 100 iterations were
sealed: 1,579/3,900 fits unresolved (40.49%), versus 1,815/3,900 (46.54%) at
50 iterations for this replica. Online-policy work is 6,167.39 seconds;
research evaluation is 505.12 seconds. The half-pass took about 2.2 hours.
The second replica is now running. This already guarantees the complete
100-iteration pass will require the planned 200-iteration pass, but all 120
assignments still run. The updated provisional forecast in `plan14.md` is
13--20 hours for Phase 1, 30--61 for all phases, or 34--69 including bounded
repairs. Only initial pilot rates are measured; later-phase estimates remain
provisional. No scientific protocol or source code was changed.

## Original Handoff Checkpoint (Historical)

- The user authorized **all of Plan 14**, including autonomous development,
  bounded repairs, fresh evaluation, and rolling reporting. They requested a
  checkpoint-safe handoff, not cancellation of the study.
- The main process was stopped gracefully with SIGTERM at 20:34:47 UTC.
  It exited successfully. **No experiment or reporting process remains running.**
  The old `.run.lock` PID text can remain after exit; the actual lock is released.
- Main root: `cache/mnist_experiment/akaike_score_gate/default_float64`.
- Main notebook: `mnist_experiment/akaike_score_gate_results.ipynb`.
- Phase 0 is complete. Phase 1 is interrupted, with **62 sealed units** in the
  main root. Phases 2-6 have not started in the main study.
- Phase 1 has completed 24 initial source fits, their 24 common extensions,
  two source/architecture decisions, and all eight paired asset bundles.
- Its first retention-pilot trajectory is checkpointed at zero-based step 61,
  stage `evaluate`, after 62 accepted fit calls, with zero unresolved fits so
  far. This is one head-only pilot, not general optimizer-health evidence.
- Main invocation elapsed time was 972 seconds, mostly source fitting and
  architecture checks. This is not an estimate of remaining study time.
- All implementation agents are closed. User previously authorized parallel
  agents, but routine execution should not require new implementation work.

## Resume

Run from `/home/evan/Documents/notebooks/amari-chenstov-updates`:

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.orchestrate \
  --config mnist_experiment/akaike_score_gate/configs/default_float64.json \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb \
  --resume
```

CUDA access requires an escalated `exec_command` in this environment:
`sandbox_permissions="require_escalated"`, prefix
`["/home/evan/.venv/bin/python"]`. Sandbox-only PyTorch reports CUDA unavailable.
Do not silently substitute CPU. The machine has a working RTX 4070 and
PyTorch 2.10.0+cu128 in `~/.venv`.

Optional limits: `--max-wall-seconds 10800`, `--max-units N`, or
`--through-phase N`. The unbounded command continues all authorized phases.
The script skips sealed units, revalidates frozen decisions, and resumes the
incomplete trajectory. Replaying the early phase dispatch does not retrain
completed sources. Initial notebook/checksum refreshes can take time.

To stop, send SIGTERM to the verified orchestrator PID in `.run.lock` and wait
for process exit. SIGINT is also handled. Abrupt machine halts are supported by
atomic checkpoints, but a graceful signal is preferable when available.

The original handoff's paused trajectory (historical) was:

```text
cache/mnist_experiment/akaike_score_gate/default_float64/phase1/.incomplete/
trajectory__0__rotation_linear__depth1_lr__5add3b0589c117076ac3/checkpoint.pt
```

## Frozen Source Decision

The common source-training protocol extended every architecture, family, and
replica from 15 to 30 epochs. Each retains its best source-validation epoch.
No final-test outcomes were used. Four paired source replicas per family:

| Architecture | Parameters | Rotation NLL | Rotation accuracy | Mixture NLL | Mixture accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Width 5, selected | 501 | 0.236892 | 92.9625% | 0.228767 | 93.3125% |
| Width 6, fallback | 559 | 0.232926 | 93.0875% | 0.207133 | 93.8250% |
| Width 0, old shallow diagnostic | 487 | 0.169175 | 95.1875% | 0.121755 | 96.3125% |

Width 5 meets the prespecified equal-family tolerance relative to width 6:
within .02 nat and one accuracy percentage point. **The capacity-limitation
flag is true** because both new architectures trail the older shallow
diagnostic. Keep this caveat visible, but proceed as planned; do not silently
replace the selected architecture or make shift severity more favorable.
There are no completed treatment comparisons or ASG efficacy findings yet.

The sealed source decision is under `phase1/architecture_selection__*/`.
The complete `phase1_selection.json` pointer does not exist yet: common solver
and retention selection, driver trajectories, and local depth audits remain.

## Verification Completed

- Latest full unit suite: **338 passed in 59.72 seconds**:
  `~/.venv/bin/python -m pytest test/unit/akaike_score_gate -q`.
- Real cached-MNIST CPU and CUDA interruption tests passed for both environment
  families and fixed/adaptive policies: eight cases. Adaptive cases actually
  evaluate gates. Continuous/resumed science and charged work agree within
  1e-6; measured timings and peak memory are excluded from that comparison.
- A second completed resume does no scientific work and preserves checksums.
- The earlier all-phase CUDA integration smoke completed Phases 0-6 with 532
  sealed units, including fresh-cohort integrity and final report export.
  Root: `cache/mnist_experiment/akaike_score_gate/cuda_endtoend_v2`.
  Notebook: that root's `report/smoke_results.ipynb`.
- That integration smoke used an earlier float32 reduced-size configuration
  and is **pipeline evidence only**, not the resolved scientific experiment.
  Do not resume it as the main study or pool its outcomes with main results.
- Main source figures and smoke figures were visually inspected. The main
  notebook was refreshed automatically during the graceful stop.

## Numerical Preflight And Resolved Configuration

The original float32 prototype hit L-BFGS numerical plateaus. A paired audit
reused five identical saved prefit states, buffers, anchors, and precisions:

| Arithmetic | Change tolerance | Fits meeting unchanged 1e-5 residual |
| --- | ---: | ---: |
| Float32 | 1e-9 | 1/5 |
| Float32 | 1e-12 | 1/5 |
| Float64 | 1e-9 | 1/5 |
| Float64 | 1e-12 | 5/5 |

Consequently the main configuration freezes float64 model arithmetic and
`tolerance_change=1e-12` for all treatments and controls. The residual
criterion stays 1e-5. `default_float64.json` inherits other settings from
`default.json`; **do not accidentally execute the latter alone**.
Evidence: `cache/mnist_experiment/akaike_score_gate/precision_audit_v1` and
the reproducible `numerical_audit.py` module. This is a five-state numerical
check, not a population claim about convergence.

The main operation profile has 846 cells with repeated measurements. Frozen
per-arrival base allowances are about 0.285493 seconds for rotations and
0.531568 seconds for mixtures; principal policy budgets are twice these.
These are deterministic profiled-work allowances, not total research runtime
or kill timers. Research-only references, evaluations, and shadows are extra.
Do not estimate the entire study merely by multiplying these allowances.

Gate restart validation uses ridge ratio 100 and a higher iteration ceiling
to ensure an actual gate is exercised. Those are **validation-only** settings,
not the scientific retention settings. The main Phase 1 pilot selects the
common solver ceiling and family retention as specified in the plan.

## Scientific Boundaries

Read `mnist_experiment/plan14.md`, root `AGENTS.md`, and
`mnist_experiment/AGENTS.md` before changing experimental behavior.

- One likelihood Fisher; loss gradients are negative scores. This plan
  explicitly uses direct consolidation rather than LFU for the archive.
- Four separate conditions: linear rotation, sigmoid rotation, and mixtures
  with 9.9 or 100.1 expected treatment nines. The 4.4-nine condition is a bounded
  Phase 4 development panel, not a fifth final-cohort condition.
- Mixture paths run from p=.01 through .10, with 18 or 182 examples per arrival.
  Rotations have 120 arrivals with four examples each. Eight common warm-up
  examples are excluded from treatment-exposure axes and clearly disclosed.
- Six nested suffix depths; the historical anchor and precision are fixed
  within each evidence episode. Episode consolidation happens once. Model
  label draws used to estimate Fisher are not additional observed examples.
- EWC representation (diagonal/LR+diagonal) and current-gate representation
  form a factorial. The small dense implementation/full-score probes do not
  demonstrate large-network compute savings by themselves.
- Gate evaluation requires a resolved restricted fit. Unresolved fits and
  denied probes are not silently counted as ordinary closed gates.
- Selection uses development mean NLL-AUC and the retention constraint,
  with the specified timing/capacity tie breaks. Do not tune for a flattering
  frozen-layer count or stop because an intermediate result is significant.
- Final cohort: 32 fresh paired replicas per condition, 18 nominal policies,
  plus at most one promoted gate-only repair. Identical uncalibrated c=1
  policies are explicitly aliased, not extra observations.
- Final effects are treatment minus control: negative NLL difference favors
  treatment. Intervals are condition-specific paired bootstrap intervals,
  not simultaneous coverage guarantees.

## Remaining Execution

The orchestrator implements this sequence; no manual phase-selection decision
or additional user approval is needed for the prespecified work:

1. Finish Phase 1: 120 retention-pilot trajectories per admitted common solver
   ceiling (50, then 100/200 only if the unresolved fraction exceeds 5%),
   starting/replacement depth drivers, 48 preassigned local depth audits, and
   frozen common retention per family. Keep the 1e-5 residual criterion.
2. Phase 2: 48 preassigned Fisher/decision audit cases, 16 independent low-score
   banks each, same-bank representations, finite high-sample references with
   stability checks, and eight historical audits.
3. Phase 3: eight development replicas per condition, bounded c/h/cadence
   screening, four-candidate shortlist, current-batch ablation, local shadows,
   history variance, and frozen calibration. Screening replicas cannot serve
   as untouched out-of-fold validation; code excludes them accordingly.
4. Phase 4: fixed-depth/factorial comparisons, scarcity/compute panel, at most
   two diagnostic-triggered repairs, at most one promotion. Score/rank repairs
   have measured probe charges and local shadow checks before promotion.
5. Phase 5: execute the fixed fresh cohort, never automatically extend it or
   stop on favorable p-values. Phase 6: fresh integrity audit and synthesis.

Original planning estimate: 18-44 active hours, or 22-52 including bounded
repairs, plus initial engineering. These are provisional, not measured
remaining times. Source selection took about 15 minutes. Publish updated
forecasts using full-size pilot trajectories and audit rates once available;
do not infer them from tiny smoke runtimes or silently shrink the cohort.

## Monitoring And Reporting

```bash
~/.venv/bin/python -m mnist_experiment.akaike_score_gate.refresh_notebook \
  --output-root cache/mnist_experiment/akaike_score_gate/default_float64 \
  --notebook mnist_experiment/akaike_score_gate_results.ipynb
```

Normally let the running orchestrator refresh after paired blocks. A separate
refresh is useful after a stop; avoid competing report writers. `--force-fresh`
bypasses report checksum/figure caches for final verification. The notebook
loads artifacts only; never add model training, repair, downloads, or large
matrix computations to it.

Watch `progress.json`, `lab_log.json`, sealed summaries, and checkpoints.
During Phase 1 the top-level completed-unit counter can lag until a phase
boundary or stop; notebook inventory and sealed units show actual progress.
The top-level status is authoritative after a stop even if the nested phase
still says `running`. At handoff it is `interrupted`, not a software failure.

The current renderer caches unchanged phase figures in-process. The older
integration smoke lacked that cache and spent substantial time rerendering.
It is not a useful forecast for main reporting overhead.

Disk at handoff: roughly 573 GB available; main root 2.5 GB. Monitor free disk
as references, snapshots, and replica bundles accumulate. Preserve completed
artifacts. Other preflight roots (`default`, `gpu_gate_resume_*`, etc.) are
separate evidence, not inputs to the scientific cohort.

## Engineering Cautions

- **Do not casually edit namespace Python files after resumption.** The main
  Phase 1 ledger freezes broad source hashes, including reporting modules.
  Altering them makes an ordinary resume fail compatibility checks. Prefer
  artifact-only inspection and Markdown notes. A necessary defect repair
  requires an explicit, documented compatibility/versioning decision; never
  remove the checks or overwrite sealed artifacts to force continuation.
- All new Plan 14 code, tests, plan, and main notebook are currently untracked.
  No commit was made. No previous plans' source files were changed. Do not
  revert user changes or commit without request.
- Recent verified fixes include advance-boundary checkpoint restoration,
  c=1 alias identity, screening/OOF separation, repaired-probe work charging,
  fresh final integrity checks, scalar closeout metrics, and source-versioned
  validation checkpoint selection. Focused tests cover these.
- Per-arrival fit diagnostics now reset when no fit occurs. Effective
  dimension is labeled as the last probe preview for the selected coordinates,
  not a post-fit curvature recalculation.
- A prior tentative claim of a CUDA checkpoint tensor-device bug was withdrawn:
  the loop already transfers those tensors before use. Do not repeat that
  diagnosis. The actual numerical issue was the float32 convergence plateau.
- CUDA adaptive-average-pooling emits a nondeterminism warning under
  `warn_only=True`. Resume is tested to tolerance, not claimed bit-exact.
- Key modules: `orchestrate.py` (phase dispatch), `development.py` (Phases 1/2),
  `trajectory.py` (checkpoint state machine), `fisher_audit.py` and
  `calibration_audit.py` (research diagnostics), `phase0.py`/`validation.py`
  (preflight), `analysis.py`/`refresh_notebook.py` (artifact reporting), and
  `closeout.py` (final integrity and paired inference).

Resume the existing study, monitor the real data, and keep the user informed.
Do not restart the implementation or launch a new experimental design.
