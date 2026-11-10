# Plan 18 Restart Notes

## Phase 8 Complete (Latest State)

Phase 8 finished on 2026-11-10 at 18:39:31 UTC. Do not resume scientific
computation: all three historical acceptance paths, 24 fresh blurred paths
and 16 fresh no-blur paths are complete. The final full-checksum audit passed
90 units before sealing its own result; all eight paired blocks and retirement
ledgers passed. Completion is in stm_reproduction_v1/study_completed.json.
Additional charged compute was 6.632 hours, within the 20-hour allowance.
The final notebook generation is 77b57faab78ceaed7402f5f6.

Fresh F1-R2 average novel-risk difference is -0.058952, nominal paired 95%
CI [-0.097241, -0.022023], favorable in 7/8 replicas. F1-R1 also favors F1.
The sample-saving point is 60.4296%, matched compute .472868x, S=1.27794,
and equal-budget compute 1.15399x. Ordinary inverse intervals are withheld:
107/2000 draws have a target already met at initialization. Matched-budget
retention qualification remains unestablished; full-budget retention passes.
The blur-context interaction is -0.031769 [-0.066725, -0.004184]. These are
frozen policy comparisons with different ridge values, not isolated FD effects.

Artifact review prompted by the user's frozen-retention question: the old
holdout and unregularized metric are identical across policies and time.
Mean per-replica initial old RMS is 1.361337 px. F1 endpoint change is
-0.002134 px, exploratory paired 95% CI [-0.006119, 0.002964], under blur;
without blur it is -0.008364 [-0.013293, -0.001425]. These endpoint contrasts
reuse the final 2000 whole-replica bootstrap indices. Over the entire blurred
path, the prespecified old RMS deterioration is instead +0.007790 px.
The stream supplies 986.25 nominal grids on average alongside 100 novel grids,
so improving the old-domain fit beyond finite pretraining is possible.
No new training, retuning, or changes to sealed analysis were performed.

The following recovery and launch entries are historical progress notes.

Reboot recovery on 2026-11-10: CUDA is available on the RTX 4070 with
PyTorch 2.10.0+cu128. Independently checked all 37 completed unit seals/file
checksums and the interrupted fresh replica 4 R2 checkpoint (n=94); all passed.
The scientific source hash still matches the acceptance record below. The old
writer lock was free and no tmux server survived the reboot. Resumed the same
production root in tmux `plan18-phase8`, appending to the existing log. The new
worker advanced past n=94 and completed replica 4 R2, reaching 15/24 blurred
and 0/16 no-blur paths complete. Fresh replica 5 initialization is underway;
historical acceptance remains 3/3. All 96 current focused unit tests passed.
The saved forecast is about 4.64 hours remaining, with 16.98 hours left in the
incremental allowance and 566.8 GB free. No recipes, seeds, allocation or frozen
scientific sources changed. Read live status.json for newer progress; the
worker proceeds through 8B, 8C and 8D without another launch or approval.

The user authorized Phase 8 on 2026-11-09: historical Plan 15 fidelity,
eight fresh three-policy blurred blocks, then paired two-policy no-blur blocks.
Read Phase 8 at the foot of plan18.md and reproduction/AGENTS.md. This phase
explicitly restores the historical rank-64-plus-diagonal EWC and frozen recipes.
Earlier Phase 7 is complete and must not be resumed as the Phase 8 experiment.

Implementation: mnist_experiment/combined_memory/reproduction.
Production root: cache/mnist_experiment/plan18/stm_reproduction_v1.
Authorization: cache/mnist_experiment/plan18/phase8_authorization.json; additional
72,000 seconds above the 92,898.41614728434-second family baseline. Budget
survives restarts. Use ~/.venv/bin/python; CUDA works outside the sandbox.

All 69 historical scientific source hashes match, as do the recorded/current
PyTorch 2.10.0+cu128 and CUDA 12.8 versions. Final focused acceptance passed
93 unit tests, actual SIGTERM/resume with bit-identical model and Fisher state,
and complete source-matched CPU/CUDA smoke runs (25 units each). A preliminary
report/analysis schema mismatch was repaired before source freeze; the failed
attempt is preserved and charged. The independent historical statistics audit
passed 218 checks, reproducing 41.443273% savings and S=.604889247383.
Acceptance is sealed in phase8_acceptance.json with scientific source hash
11b110408f31b3928ecff982a5d75656a575ecedfd67f9199227e230f7d27ea6.
The preparation ledger charges all preparatory wall time once, net of the
separately tracked smoke roots. The launch narrative below is historical;
the reboot-recovery entry above and live status supersede its progress counts.

Production launched in tmux session `plan18-phase8`; console output is
cache/mnist_experiment/plan18/phase8-execution.log. The driver first reruns all
three policies on original confirmation replica 0 and compares them with the
immutable original. Fresh execution is gated on fidelity. It will then finish
24 blurred and 16 no-blur paths automatically with notebook updates. Check
status.json and the log for progress newer than these launch notes. Never
launch a second writer; resume the same root after a halt.

Phase 8A is complete: original replica 0 replay passed for R2, R1 and F1,
all 15 numeric comparisons in each case with zero absolute discrepancy,
including initial/final parameters, risk curves, objective/gradient values and
represented Fisher checks. All 298 accepted updates and 853 retirements agree
exactly in each policy. This verifies the historical computation; fresh
statistical replication remains separate. Phase 8B has completed fresh replica 0
(3/24 blurred paths); replica 1 initialization is underway. Its one-block mean
novel risks are R2=2.02094793, R1=1.99384410, F1=1.93039504. These are descriptive
health observations, not cohort inference or a reason to alter the allocation.
The sealed fresh-manifest reserves 24,207.74 seconds (6.724 h), including
contingency/closeout, against 69,502.63 seconds then remaining. Fixed allocation
is 24 blurred plus 16 no-blur paths. A reporting-only refinement makes native fidelity rows and final closeout
clearer (30 reporter tests passed); scientific source identity is unchanged.
After the first two complete fresh blocks (6/24 blurred paths), a reporting-only
restart loaded the tested native fidelity/outcome display and corrected an
unsupported inverse being labelled a retention failure. SIGTERM exited cleanly,
the same root resumed with unchanged scientific source hash/recipes, and the
log remains append-only. See phase8_reporting_restart.json. No completed unit
or scientific output was altered. The current reporter tests passed 32 cases.
At four complete blurred blocks (12/24 paths), a second clean reporting-only
reload displayed the reference-quality diagnostic beside the large point S.
The reference improved only 2.48% from initialization at this interim look,
below the fixed 5% diagnostic; do not promote the point saving to an established
sample-efficiency result. See phase8_reporting_reload_2.json. Both reloads
preserved the same scientific source identity and immutable completed paths.
Fresh replica 4 is underway; use status.json for newer counts.
Exploratory artifact context: original R2 frozen novel risks were
[2.9577, 2.4373, 5.8618, 5.5447, 1.9722, 1.9974, 1.9958, 2.5458],
mean 3.1641, SD 1.6056. The first four fresh values are
[1.9664, 2.9152, 2.0121, 1.7898], mean 2.1709, SD .5054. Original replicas
2/3 had nominal risks 2.0854/2.0155, so their large novel risks describe poor
initial blur transfer rather than comparably poor nominal fits. Recompute
the fresh distribution after all eight; keep this descriptive/post hoc and
do not change seeds, pretraining or policies. It may contextualize limited
reference learning headroom, but is not a demonstrated causal explanation.
The partial inverse is unstable (some baseline targets are already met at
initialization and some FD paths miss their endpoint targets); intervals remain
withheld instead of dropping those draws. The fixed cohort continues unchanged.

Runner:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.reproduction.run --resume
```

Use `--stage status`, `--stage stop`, or `--stage report` for operational commands.
After a cooperative STOP, resume with `--resume --clear-stop`. Smoke runs must
write a separate root and notebook. Preserve all unrelated robotics changes.

## Phase 7 Terminal State (Historical)

PHASE 7 IS COMPLETE. The worker exited normally; do not resume scientific
computation or launch another writer. Phases 0--6 remain immutable history.

Root: cache/mnist_experiment/plan18/low_ridge_v1.
Report: mnist_experiment/plan18_results.ipynb, chronological Phase 7 at bottom.
Current immutable report pointer: low_ridge_v1/report/latest.json.
Console log: cache/mnist_experiment/plan18/phase7-execution.log.
Completion: low_ridge_v1/study_completed.json, 2026-11-09T09:20:43Z.
All 114 trajectories are available: 64 development (55 imported, 9 new),
40 fresh and 10 stationary. Eight changing plus two stationary five-arm blocks
are complete. Final independent validation verified 85 completed units with
full checksums, all 51 frozen scientific sources, all 31 historical notebook
cells, public/sealed notebook agreement, and exact paired/exposure ledgers.
The protected mathematical notebook is unchanged. Nothing is pending.

The four-member calibration passed without refinements: A/B/C ridge1, D/E .1,
B fixed at 1, FD strength .5691316767656495, D/E process scale 1e-5.
Both all-arm development retention ranges passed (.072215/.050080 px).
B remains the development retention reference; A is the efficiency reference.
No production repair, retuning, extra replica or allocation change occurred.

Final nominal paired 95% results (eight fresh replicas, four per sign):
- B-A acquisition MSE: -.006947 [-.017254, .001510] px squared; 5/8 favorable.
- B-A Havg: -.012435 [-.021655, -.004428] px. Endpoint H1216:
  -.005625 [-.026220, .021425]. Both support .10-pixel two-sided equivalence.
- B endpoint-budget saving: .862% [.050%, 1.280%], matched compute 1.553x
  [1.514x, 1.589x], S=.005546 [.000318, .008261]. B's sustained crossing is
  1205.52 grids versus A's own 1216. First crossings are B616.80/A603.99.
  This is a small sustained endpoint benefit, not restored large STM savings.
- C-B acquisition MSE: +.00951 [.00571, .01355]; E-B +.09645
  [.08731, .10683]. D/E retain more old data while acquiring less well.
  Only 4/10 pairwise fresh contrasts support both retention-equivalence checks.
  One-sided retention support in the efficiency table is not equivalence.
- D/E point savings against A are 1.13%/1.23%; broad inverse intervals reach
  ~50% through early/late crossing switches. E bootstrap median is 1.22%.
  C has two unsupported draws out of 2000; inverse CI is withheld, not repaired
  by discarding them. Do not pool Phase 6/7 targets or historical confirmation.

Current report generation: 323e48802bd682803498a6e4. It preserves byte-identical
analysis.json from worker-final generation e7c01902b890efcdd3f6783c. Post-exit
report-only clarification exposes precise crossing counts, B's small saving
and cost, and the one-/two-sided retention distinction. All earlier report
generations remain intact. Prediction/bootstrap and requested pi/gain/motion
and effective-ridge/movement figures were visually checked.

Final unit acceptance: 158 passed, including 33 reporting tests. Earlier
CPU actual-SIGTERM/resume parity, failed-calibration gate and completed CUDA
smoke phase7_smoke_cuda_v1 passed. All needed test/report sessions have exited.
Additional charged compute is 6.22 hours; cumulative family charge 25.81 hours.
The immutable authorization baseline is 70,504.28523658379 seconds, additional
allowance 72,000 seconds, cumulative ceiling 142,504.2852365838 seconds.
Never reset it. Family artifacts remain below 4 GB, with about 574 GB free.

Optional artifact-only report rebuild, from the repository root:

```bash
~/.venv/bin/python -m mnist_experiment.combined_memory.low_ridge.run \
  --config mnist_experiment/combined_memory/low_ridge/configs/production.json \
  --root cache/mnist_experiment/plan18/low_ridge_v1 --report-only
```

Keep the sealed authorization, historical dependencies and completed units.
Further scientific exploration requires a new user decision, not a resume.

## Historical Phase 6 Completion

Phase 6 completed at 2026-11-08T22:35:29Z under retention_match_v1. Its latest
immutable report is 8960658bfe81951b6668b306; study_completed.json seals its
final outcome. No further Phase 6 computation is queued.

All 30 confirmation paths completed: four fresh changing and two stationary
five-arm blocks. Development has 55 paths (37 imports, 18 new); total available
trajectories are 85. The full final audit and follow-up publication check
verified all 71 completed units, 46 frozen scientific sources, six paired
blocks, the passing calibration seal, and the public notebook. The final
post-run report adds the requested Phase 6 pi/gain/motion diagnostics and a
short CI explanation, preserving byte-identical analysis and both earlier
figures. It passed 36 focused reporting tests and a preservation check covering
the earlier notebook cells, both report seals, and all frozen sources.
Final focused acceptance passed all 168 unit tests; earlier CPU lifecycle and
complete CUDA smoke acceptance remain valid. The worker exited cleanly.

Fresh retention passes both nominal .10-pixel equivalence checks in 10/10
pairwise contrasts. E-A rotated-risk difference is -.24630
[-.25939,-.23321] px squared, about 2.66% lower risk; B-A is a tiny worsening,
.00346 [.00267,.00425]. With only four fresh replicas, intervals are exploratory.
E's 49.1% [3.0,50.3] endpoint-budget saving and S=.370 [.016,.377] concern
*sustained* target quality: E crosses around 619 grids, A persistently around
1182 after an earlier crossing near 629. Do not describe this as twice the
initial learning speed. B's small nominal saving matches A's own early
persistent crossing and does not demonstrate additional timing benefit.

Artifact review: 49.1% is the persistent crossing of mean curves, not a median.
In the original 2,000 sign-stratified paired draws, 37.85% switch to a late
crossing (~3% saved), the rest to an early crossing (~48--50%). Bootstrap
median is 48.31%. Only four fresh replicas support this inference.
Lower ridge has a development lead: paired B-A acquisition MSE is -.04885
at c=1 versus -.00037 at c=10, with similar mean retention. Plan 15 head-r1
development likewise benefits at c=1 but not c=3.16228. See the plan's post-run
review for exact numbers and cross-study caveats. This is not fresh evidence
for a new recipe, and no additional experiment is authorized or running.

Cumulative charge including closeout/testing is about 19.58 active hours,
within the approved conditional 21.5-hour ceiling and ultimately below 20.
The extension admitted conservative reserve; it did not reset the ledger.
Study conditions/recipes/tolerance and completed numerical artifacts never
changed during confirmation. No extra trajectories are automatically approved.

## Historical Phase 6 Execution Notes

PHASE 6 EXECUTION AUTHORIZED (2026-11-08); IMPLEMENTATION ACCEPTANCE PASSED.
The new namespace is combined_memory.retention_match and production root is
cache/mnist_experiment/plan18/retention_match_v1. Production is starting in tmux
session `plan18-phase6`; check status.json and the worker before any restart.
The historical studies below remain complete. Do not resume them for new work.
Phase 6 uses the original cumulative 20-hour ceiling during calibration;
the user explicitly approved a conditional 21.5-hour cumulative ceiling on
2026-11-08 after calibration passes, with no change to the 30 confirmation
trajectories, tolerance, or methods. The 500-GB ceiling is unchanged. The notebook
receives a chronological Phase 6 section after the Phase 5 failure conclusion.

The separately versioned resource continuation passed 13 checks, including a
real-store reopen above 20 hours with byte-identical prior scientific units.
Do not invoke it until a passing four-member selection is sealed and the
original worker has exited. If confirmation cannot fit under 20 hours, use:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.retention_match.tools.continue_budget \
  --config mnist_experiment/combined_memory/retention_match/configs/production.json \
  --root cache/mnist_experiment/plan18/retention_match_v1 \
  --approved-ceiling-hours 21.5 \
  --approval-note "User approved conditional 21.5-hour ceiling on 2026-11-08" \
  --clear-stop
```

This revalidates the actual matching evidence and unchanged allocation,
preserves any earlier calibration-budget closure as history, and archives the
approval/wrapper source in decision-resource-extension-21p5. Scientific sources
stay frozen. The continuation WAS ACTIVATED at about 2026-11-08T19:37Z,
after the original worker cleanly closed its budget-limited invocation.
The immutable approval is sealed; the restarted worker reports 77,400 seconds
as the cumulative ceiling. Use this continuation command, not the original
20-hour entry point, for any subsequent restart of this confirmation.

Calibration PASSED on all four members at 2026-11-08T19:36Z. Both refinement
rounds completed; 55 development paths are available (37 imported, 18 new).
Selected ridges A/B/C/D/E are 10/10/10/1/.31622776601683794. Both all-arm ranges
pass: Havg=.07544112058211416 px, endpoint=.02827549376781935 px. The seal is
decision-selection. Confirmation keeps four fresh changing and two stationary
five-arm blocks. Its fixed forecast with reserve is 13,813 seconds (3.84 h).
Cumulative charge at the first closure was 16.57 h. Inspect current status for
the newer confirmation count; fresh inference must not alter recipes.

Historical calibration progression: all ten initial new development paths completed,
but the joint starting tuple narrowly failed Havg (.1050799153 > .10); endpoint
range .0369907604 passed. No confirmation has started. Two E refinement recipes
are admitted, four members each: .01 (automatic first round) and
.31622776601683794 (reviewed second round). The other per-arm allowances are
unused. Keep both rounds and complete every admitted member before selection.

The review recovered paired historical E/ridge-.1 and ridge-1 responses from
member 0 that the automatic proposer omitted because the latter was incomplete
on four members. Havg brackets A; both endpoint responses are already in its
.10-pixel band. The midpoint is only a proposal and must pass the unchanged
four-member joint gate. `decision-refinement-0-1` archives review source and
evidence. Six review tests and 32 reporting tests passed. No frozen scientific
source, completed unit, tolerance, or candidate allowance was changed.

The checkpointed restart following that review resumes the first E/.01 path
at member 0, accepted batch 818. New development count is 47 so far (37 imported
and 10 starting paths), with eight refinement paths queued/in progress. The
presenter now counts those queued paths explicitly. Cumulative compute was
about 15.78 h at this review; remaining confirmation may require a separately
authorized extension if the fixed allocation and reserve no longer fit.

Acceptance: 142 focused unit tests, two CPU lifecycle tests (including actual
SIGTERM/resume and failed calibration with zero confirmation paths), and a
complete CUDA smoke study passed. The finalized CUDA acceptance root is
cache/mnist_experiment/plan18/phase6_smoke_cuda_v1; it is complete and immutable.
The early phase6_smoke_cpu_v1 root predates final scheduler fixes; retain it
but do not resume it. CPU lifecycle tests used isolated temporary roots.

Original calibration command from the repository root (historical; use the
approved continuation command above for current confirmation recovery):

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.retention_match.run \
  --config mnist_experiment/combined_memory/retention_match/configs/production.json \
  --root cache/mnist_experiment/plan18/retention_match_v1 --resume --clear-stop
```

The detached log is execution-20261108T1738Z.log under the production root.
No fresh path is allowed until all four development members support both
five-arm ranges <=.10 px and the passing selection is sealed. Calibration may
close with an explicit budget/feasibility status rather than launch unmatched
confirmation. Keep the original 20-hour cap during calibration; the approved
conditional continuation above may raise it to 21.5 afterward. It was about 14.71 h at CUDA
acceptance completion. Do not change scientific sources while the worker runs.

Early development made a clean reporting-only restart at C/member-2 accepted
batch 234. A and B/member-2 are complete (39 total development paths including
37 imports). The presenter now withholds stratified-bootstrap intervals and
supported-saving flags while either fresh sign stratum has fewer than two
replicas; the final four-replica estimates are unchanged. Thirty-one reporting
tests passed. This note is in the notebook and plan. Resume reloads the new
report source while retaining all scientific hashes/checkpoints and recipes.

PHASE 5 IS COMPLETE. POST-REBOOT VERIFICATION PASSED (2026-11-08).
The detached run completed normally before the reboot, at
2026-11-08T04:44:45Z (November 7, 8:44 p.m. Pacific). All 77 Phase 5 trajectories
are complete: 37 development, 30 fresh, 10 stationary. Six fresh and two
stationary five-arm paired blocks passed the final integrity audit. There is
nothing to resume. No experiment or GPU compute process is running.

Post-reboot read-only verification checked all 107 completed units with full
checksums, all scientific source hashes, the eight-block completion record,
and the sealed final report. Focused reporting tests passed (17). The notebook
now labels unsupported numerical estimates "not estimable" and shows no active
batch instead of "pending/pending". The rebuilt analysis.json is byte-identical
to the prior completed generation; only presentation changed. Historical report
generations and Phase 0--4 notebook cells remain preserved.

Total cumulative Plan 18 charge, including this audit, tests and report rebuild,
is about 14.58 active hours, within the original 20-hour ceiling. No additional
trajectories are authorized automatically. Current completion marker:
cache/mnist_experiment/plan18/broad_drift_v1/study_completed.json.
Current report: mnist_experiment/plan18_results.ipynb.

## Calibration Review And Proposed Phase 6

On 2026-11-08 the user requested an explicit Phase 5 failure conclusion and a
new phase for review. Phase 5 completed computationally, but failed its intended
five-arm retention comparison: B's ridge was refined to 1000 while A/C/D/E
remained at 10/10/1/.1. The retrospective conclusion was published through
the versioned reporting path; original scientific artifacts and numeric results
remain unchanged. This is a calibration-design failure, not a reboot failure.
The new report generation is `64a9b4923ca6748d27be2e8f`. Nineteen focused
reporting tests passed. Publication checks verified byte-identical analysis,
unchanged previous report files and 111 scientific seals/identity files, and
unchanged notebook figures and cell sources except the final conclusion.

Phase 6 at the foot of plan18.md was subsequently authorized for implementation
and execution. It fixes A at ridge 10 initially, starts B at ridge 10, requires
all-arm ranges <=.10 pixels for both outbound retention summaries, verifies
on two additional paired development replicas, and permits only a passing
calibration seal to launch confirmation. Default confirmation is four fresh
plus two stationary five-arm blocks. The lean forecast is 4.75--5.85 additional
active hours, close to the roughly 5.42 hours remaining under the existing cap;
refinement can require a separately approved compute extension. No new
experimental trajectories have been launched.

The remainder of these notes records execution history. Commands and "running"
milestones below are historical; do not restart this completed study.
The current protocol is `combined_memory.broad_drift`, with a new scientific
root `cache/mnist_experiment/plan18/broad_drift_v1`. Do not resume production_v1
for new work. If the tmux worker has exited or the node has restarted, check
GPU availability and durable status before using this command from the repository
root. It is ALREADY RUNNING now; this command is for recovery, not a second launch:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.broad_drift.run \
  --config mnist_experiment/combined_memory/broad_drift/configs/broad_drift.json \
  --root cache/mnist_experiment/plan18/broad_drift_v1 --resume --clear-stop
```

Read Phase 5 at the foot of plan18.md. All incoming grids rotate with the same
slow cosine schedule; outbound acquisition is 1,216 grids rather than 100 rare
grids. Same five arms, repeated development retention calibration, new seeds,
and the ORIGINAL fixed study-wide development/confirmation test partition.
The cumulative 20-hour/500-GB ceilings span both protocols. Prior charged
compute was about 7.384 hours. Historical scientific source hashes still match.
GPU is healthy; direct execution sees CUDA, while the sandbox hides its device
nodes. Phases 5A-5B passed and Phase 5C has resumed. The STOP marker was cleared
by the normal resume API after checkpoint/source validation.

The initial `--max-units 2` production invocation completed both A readiness
paths and all numerical/detectability audits, then exited cleanly. The normal
unlimited-unit command above is now running in tmux through calibration, fresh
paired exploration, and closeout. Do not resume an old completed tool session.
Inspect status.json/PID for the active unit; durable state is authoritative
across conversation/tool-session restarts.

The versioned generator/configuration, batch loop, observer
gain logging, offline detectability diagnostic, scheduler and cumulative-budget
runtime are implemented. Data/config tests passed (31), core CPU checks passed
(5), explicit direct-CUDA core checks passed (5), and inherited numerical,
recovery, scheduler and reporting checks passed (74 with one CUDA-only skip).
Both fixed-replay and combined adaptive/transport paths resume exactly. The
zero-invocation-wall-limit CLI exited cleanly at startup. Additional diagnostic
checks passed (six core tests), both actual CPU SIGTERM/resume CLI lifecycle
tests passed, and the full CUDA smoke study completed 30 paths, calibration,
reporting and four five-arm blocks with final checksums/exposure-clock audit.
The smoke root is `cache/mnist_experiment/plan18/phase5_smoke_cuda_v1`; it is
complete and immutable. No source mismatches were found. Acceptance work is recorded in
the family acceptance ledger; engineering time is separate. Original scientific
sources have not changed.

Reporting acceptance passed 49 combined unit checks (including 13 new focused
tests). Subsequent legend/settings cleanup passed all 13 focused tests again.
Those CPU tasks overlapped active GPU invocations and are already covered by
their wall-clock ledger. Gain-logging overhead is measured separately from
standalone learning time; the inherited filter state/action is exactly unchanged.
The public notebook now contains near-top Phase 5 progress, all sealed earlier
history, and a chronological Phase 5 section. New production current-law and
acquisition metadata were checked directly: all-grid drift, 1,216 outbound
grids, fixed test partition, 59,364 distinct training-role sources per replica.

Readiness froze float32 FD, max_iter=100/max_eval=150, lambda0=.05691316767656495.
All four doubled-solve checks showed zero further objective/prediction movement;
no numerical repair was required. Calibration FD strengths are lambda0 times
{.1,1,10}; no fresh recipes are selected yet. Both complete A readiness paths
remain reusable calibration evidence. First initialization took 249.6 s and
first online path 288.9 s (147.9 s standalone learning).

Paired frozen-anchor score diagnostics support stronger signal with broader
drift: split motion/batch-noise ratios were about 15--1,364 times the rare-law
ratios across checked states/replicas. Absolute broad ratios remain small
(roughly .00013--.00126). Fixed-anchor mapping residual energy/noise ratios
were 11.9, 85.1, 155.1 and 96.0 at the four audit states. The public notebook
now highlights this modelling qualification; it is not an arithmetic defect
or evidence of predictive benefit. Continue the approved best bets; no process
or filter change was made. Cumulative charge at readiness exit was about
7.749 active hours, leaving about 12.25 hours of the common cap.

Development reporting qualification: rotated risk falls then rises near peak.
Before any fresh outcome, report.py gained a reference-clock diagnostic showing
the reference's own persistent endpoint-quality crossing alongside allocated N.
It preserves the approved S and does not change training/calibration/targets;
it prevents attributing an identical earlier crossing to a method advantage.
The plan records the qualification. A synthetic identical-policy regression
test covers this nonmonotone case. Original analysis.py and all scientific
source seals remain unchanged. Do not omit this caveat in final interpretation.

Calibration milestone: 11 development trajectories are complete (7 B, 4 A).
Read status.json for newer progress. The provisional paired STM incumbent is
FD=.5691316767656495, ridge=10; paired Havg=2.3571 and H1216=4.4625 px, far
above the .50-pixel preference. Report.py now highlights absolute retention
separately from inter-method matching. This reference may change under the
bounded optional refinement; it is NOT a frozen fresh recipe yet.
Recorded shadow innovation scores still rank eta=1e-5 first and 1e-4 second;
the actual selection decision remains in the scheduler. All reporting checks
pass (14 focused tests after adding the reference-clock diagnostic). No further
scientific code, source law, target, controller, or numerical policy changed.

Adaptive-calibration milestone: 22 paths complete. Fixed-weight calibration
is finished and eta=1e-5 D/E screens are in progress; eta=1e-4 follows. First
D/c=.1 path mean pi=.0038115, no bounds/fallbacks; first E/c=.1 mean pi=.0038125,
no bounds/fallbacks, maximum normalized clipped mass 4.10e-7. First C/c=.1
had nonzero corrections (mean norm 22.4) with no clipping. These are development
health observations, not fresh effects. Reporting now shows the latest completed
development candidate per arm for pi/gain/LFU health before a complete fresh
block exists, clearly labeled with replica/ridge/process scale; no development
record enters fresh inference. This fallback passed 15 focused reporting tests.
Read status.json for newer counts; the command remains running unchanged.

Reporting-module refresh: paused the main invocation cleanly at E/development-0,
ridge10/eta1e-5, accepted batch693/928 (23 complete paths). Scientific hashes,
checkpoint SHA256, finite model/nonnegative archive, FIFO/exposure ledger, and
693 observer assimilations all passed. The long-running interpreter had cached
an older report module while report.py was edited. Some superseded report
generations can therefore contain older presentation with a newer file hash;
scientific units/metrics are unaffected. execution_notes.json records the issue.
The next normal resume loads current reporting code and rebuilds from sealed
scalar exports. Do not edit report.py during the resumed invocation; defer
closeout presentation changes until it exits or perform another controlled
reload. All 15 focused report tests passed before the refresh.

The reporting fix now pins source bytes at module import and includes the
actual broad report, inherited report, and analysis sources in each new sealed
report generation. Later on-disk edits cannot masquerade as executed cached
code. The reporting-only changes remain outside scientific identity; training,
metrics, initialization and calibration code are unchanged. Current tests cover
an on-disk edit after import and verify the archived bytes/hashes remain those
actually loaded. Resume uses the same scientific root and accepted checkpoint.

The imported-source provenance regression and all other focused report checks
passed: 16 tests. The source-pinning repair is recorded in execution_notes.json.
The normal GPU command has resumed from batch693 with current reporting code;
no scientific trajectory was repeated. Sessions 60119 and 71544 are finished.
Use durable status/runtime files for the current PID and continuation state.

## User-Requested Graceful Halt (Historical Restart Boundary)

The stop request was honored at an atomic accepted-batch boundary. Status is
paused, the timing lease has running=false, and nvidia-smi reports no compute
processes. Invocation/tool session 46216 is finished; do not poll it. Its saved
runtime PID is historical, not evidence of a live worker.

- Complete development trajectories: 25. Fresh: 0. Stationary: 0.
- Active unit: path-development-000-D-2f269884b7dade-e1124238.
- Recipe: D (STM FD + velocity Kalman), ridge=.1,
  FD strength=.5691316767656495, process_scale=1e-4, development replica 0.
- Accepted batch: 240/928; outbound acquisition: 320/1216 grids.
- Replay: 32 grids; retired: 928 grids. Their concatenated IDs are exactly
  range(960); observer assimilations equal the accepted batch count.
- Checkpoint: checkpoint-da2c6e87ffee440eb33fd8ab672698cf.pt in that unit.
- SHA256: 2239c955d42c9edd33c7da19ffbb2f7e79e4b357215b9c5f616d291b60a8cac3.

Read-only halt verification passed checkpoint SHA256, finite model and archive,
nonnegative diagonal, contiguous exposure/retirement ledger, acquisition clock,
observer assimilation count, and all scientific source hashes. An artifact-only
report rebuild completed successfully, preserving old report generations and
the STOP marker. Both bounded tasks were charged through the acceptance ledger.
Cumulative Plan 18 active time is about 9.79 hours, leaving about 10.21 hours
of the original 20-hour allowance (node downtime is excluded).

Next action after authorization: resume the current D/eta=1e-4 calibration
checkpoint using the top-of-file command. Continue the approved paired D/E
screens and bounded retention refinement, freeze recipes/cohort using measured
timing, then fresh/stationary comparisons and closeout. No scientific settings,
selection rule, or completed trajectory changed for this pause. Preserve the
absolute-retention and nonmonotone reference-clock qualifications above.

## Detached Tmux Resumption (Historical Launch)

The user authorized resumption and unattended execution while discussing the
next plan. CUDA is healthy. Bounded preflight passed the saved checkpoint hash,
finite/nonnegative model/archive state, exposure ledger, observer clock, and all
scientific source hashes; it is charged to the family acceptance ledger.

- tmux session: plan18-phase5 (normal server, accessible outside the sandbox).
- Launch: 2026-11-07T23:58Z; initial worker PID 4898 (historical after exit).
- Log: cache/mnist_experiment/plan18/broad_drift_v1/tmux-20261107T235805Z.log.
- The command uses the user's venv, --resume --clear-stop, no max-units limit,
  and the unchanged cumulative 20-hour ceiling (about 10.21 active hours remained).
- Bash pipefail preserves the Python exit status through tee. The pane remains
  available after exit for inspection. No model call or open editor is needed
  for the existing Python scheduler to proceed. Unexpected faults still require
  review/resumption; tmux does not recover a node reboot by itself.

Startup observation confirmed the saved D trajectory advanced from batch 240
to 339/928 without errors. Durable status is Running, Phase 5C. No scientific
configuration or completed artifact was changed for the detached launch.

Monitor from a normal terminal:

```bash
tmux attach -t plan18-phase5
```

Detach with Ctrl-b followed by d. Notebook progress continues automatically;
status.json/runtime.json are the durable current state. To halt gracefully from
another terminal, run the following from the repository root, then wait for the
worker to exit before repurposing the GPU:

```bash
~/.venv/bin/python -m mnist_experiment.combined_memory.broad_drift.run \
  --config mnist_experiment/combined_memory/broad_drift/configs/broad_drift.json \
  --root cache/mnist_experiment/plan18/broad_drift_v1 --request-stop
```

Do not edit sealed scientific code or the imported report module while this
worker is active. Design of the next plan can proceed in separate documentation;
do not start competing GPU experiments. Inspect the exit/status and finish any
necessary narrative interpretation after the unattended run completes.

HISTORICAL PHASES 0-4 ARE COMPLETE. Execution was authorized 2026-11-06; all Phases 0-4
finished with 33 development paths, six complete fresh paired replicas (30
paths), and two stationary paired replicas (10 paths). No process remains
running. See the final closeout below; earlier notes are historical milestones.
The scientific contract is in plan18.md; the report will be plan18_results.ipynb.

- Use ~/.venv/bin/python. PyTorch 2.10.0+cu128 sees the RTX 4070 through
  direct execution. The sandbox hides /dev/nvidia* and reports no CUDA device;
  this is not a broken driver. nvidia-smi is healthy (580.178.04).
- New implementation: mnist_experiment/combined_memory/. Historical sources
  and mathematical_overview.ipynb are protected.
- Final acceptance: 77 Plan 18 unit tests passed (one CUDA-only test skipped
  in the CPU invocation), six explicit CUDA recovery/parity tests passed,
  and 51 inherited filter/artifact tests passed. Tiny CPU and CUDA full pipelines each completed 30 paths,
  calibration, sample-equivalence analysis and notebook publication. These are
  smoke evidence only, not experimental findings.
- Both real CLI lifecycle tests passed: SIGTERM after the first accepted batch,
  exact resumed/uninterrupted scientific-state agreement, immutable completed
  units, report isolation, and clean zero-invocation-wall-budget exit.
- Initial GPU recovery failures were test-harness issues: missing production
  deterministic setup and a misplaced test assertion. Corrected checks passed.
- A startup wall-limit traceback found by integration acceptance was repaired.
- Production root is complete. Smoke revisions are preserved in
  their own roots and must not be resumed using later incompatible code.
- Planned root: cache/mnist_experiment/plan18/production_v1. Every production
  batch is an atomic accepted transaction; use --resume after interruption.
- 20 cumulative active compute hours and 500 GB artifacts/dependencies, with
  20 GB free reserve. Wall-clock accounting survives new source revisions.
- Two bounded coding agents completed data/model/curvature and reporting.
  Integration review repaired sealed-ridge-proposal replay, reference re-ranking,
  metric-code provenance, smoke report isolation and controlled-exit reporting.
- No historical scientific source or mathematical_overview.ipynb was changed.

The eventual run command is:

```bash
~/.venv/bin/python -m mnist_experiment.combined_memory.run \
  --config mnist_experiment/combined_memory/configs/production.json --resume
```

Use --request-stop to set the durable stop marker. Resume afterward with
--resume --clear-stop. SIGTERM/SIGINT also release at a recoverable boundary.
--max-units N limits newly completed trajectories, not total trajectories.
The family ledger includes acceptance, smoke, failed/retried work and all
production roots. Downtime does not consume the allowance.

Final acceptance root: cache/mnist_experiment/plan18/smoke_cuda_v2 (completed).
Production root: cache/mnist_experiment/plan18/production_v1.
Two full development A trajectories completed (928 batches each). Initial fits
took 258-259 seconds; online paths 259-283 seconds including research I/O and
shadow observers, versus 141-168 seconds of standalone learning. Actual novel
counts were 168 and 148, both supporting the first-100 analysis.

Numerical policy is sealed: lambda0=0.4286051792826472, float32 FD, 100 solver
iterations/150 closure calls. FD relative errors were below 1.7e-5; paired
endpoint arithmetic agreed near float64 precision; doubled solves changed no
audited endpoint. Fixed-anchor mapping and archive-quadratic qualifications are
in execution_notes.json and the reporting notebook. No new estimator was added.
Both initialization checkpoints selected epoch 45. Average validation NLL
improvement from epoch 30 to 45 was about 1.71%; retain the common 45-epoch fit.

The readiness invocation paused cleanly via --max-units 2, without a STOP marker.
Resume with the command above without --max-units. Do not start a second CLI
while the family-wide writer lease is held. About 0.474 cumulative active hours
were charged at the boundary, with over 570 GB actual free disk. The production
root held about 30 MB. Replace remaining-runtime forecasts with calibration
timings once the FD and combined paths finish.
No online C estimator, discrepancy controller, dense Fisher, or automatic
scientific-scope extension is authorized.

## Reboot Recovery, November 6

The user restarted the node and requested resumption. Read-only integrity
validation passed for all 17 existing units, including eight completed
development trajectories. Scientific source hashes match the sealed identity;
the protected mathematical notebook's hash is unchanged. CUDA/driver is healthy.

Resume path-development-001-B-17e6dc1584c38b-e1124238 from accepted batch 120/928
(FD strength 0.4286051792826472, ridge 1). Its checksum matches latest.json;
model/archive are finite, 32 FIFO grids plus 448 uniquely retired grids exactly
account for 480 incoming grids, and both composition states are preserved.
No STOP marker is present. Use the normal --resume command above. The durable
timing lease conservatively charges the interrupted chunk; downtime is excluded.
Approximately 1.04 cumulative active hours have been charged before this resume.

Recovery succeeded and the interrupted ninth trajectory completed normally.
The paired development screen chose the STM reference FD strength
0.04286051792826472 at ridge 1. Replay/EWC ridge calibration has completed;
transport calibration is in progress. These local choices are recomputed
deterministically until the final selection.json is sealed; no fresh recipe
is frozen yet. The first low-ridge transport path completed with no nonfinite
optimizer trials (398 seconds learning versus 343 for its matched nontransport
recipe; 493 versus 437 seconds online wall time). About 1.50 cumulative active
hours are charged at this milestone. No efficacy stopping or scope change.

Additional restart acceptance now covers stopping immediately after an Adam
minibatch and during initial reference-Fisher accumulation. Both resumed fits
exactly match uninterrupted scientific state; completed initialization performs
no model construction/training on reuse. The focused CPU recovery suite passed
7 tests with the CUDA-only parity test skipped (already separately validated).
These tests ran alongside the active GPU invocation, so their elapsed time is
covered by that invocation rather than charged a second time. No scientific
source or configuration changed.

The fixed-weight A/B/C calibration screen is complete. The causal readiness
observer shortlist retained process scales 1e-5 and 1e-4 (1e-3 ranked third).
Both D/E arms now receive their predeclared ridge screens and paired finalists.
No fresh outcomes exist and no final selection.json is frozen. Approximately
1.89 cumulative active hours are charged at this milestone. Production source
identity remains unchanged; invocation 73129 is running in this conversation.

## Frozen Selection, November 6

selection.json was sealed at 2026-11-07T02:58:59 UTC (November 6 Pacific).
All five development matching violations are zero at the declared 0.10-pixel
tolerance; no optional ridge refinement was needed. This is development matching,
not evidence of fresh retention equivalence. Recipes:

| Arm | Ridge ratio | FD strength | KF process scale |
| --- | --- | --- | --- |
| A Replay/EWC | 1 | 0 | unused |
| B STM incumbent | 1 | 0.04286051792826472 | unused |
| C STM + transport | 1 | 0.04286051792826472 | unused |
| D STM + adaptation | 0.1 | 0.04286051792826472 | 1e-5 |
| E Combined | 0.1 | 0.04286051792826472 | 1e-5 |

Six changing and two stationary replicas, five arms each, are frozen (40 paths).
Do not retune from fresh outcomes. The conservative forecast was 18,008 seconds
(5.00 hours), including a 25% cushion; 3.32 cumulative active hours were charged
at selection. The scheduler interleaves full paired blocks and stationary checks.
Use the same resume command/source identity; completed calibration and selection
units are immutable. Inference excludes any incomplete five-arm fresh block.

## Graceful Stop, November 6 (Latest State)

The user requested the node. The CLI durable stop request completed and the
main invocation exited normally with status paused and exit code 0. NVIDIA's
compute-process query is empty: the experiment released the GPU. Invocation
73129 and the stop-request invocation 28781 are finished; do not poll them.

- 33 development trajectories, one fresh trajectory (replica 0, arm A), and
  zero stationary trajectories are complete. No complete fresh five-arm block
  exists yet, so no fresh comparative inference is justified.
- Active incomplete unit: path-fresh-000-B-54a346b3643b0b-e1124238.
  Accepted step 296/928; 11 novel grids observed. The checkpoint pointer's
  SHA256 matches its file:
  e5b943ecf103df509c4566a2d50ce81de1b0181b6ee6e48d1ab9b8a00770091b.
- Model and archive are finite; the diagonal is nonnegative. The 32 FIFO IDs
  plus 1,152 unique retired IDs account for all 1,184 incoming grids. Both q
  states, controller state, metrics and RNG state are in the checkpoint.
- Cumulative active charge: 12,559.4 seconds (3.489 hours), excluding downtime;
  approximately 16.51 authorized hours remain. The latest conservative online
  forecast is 15,040 seconds (4.18 hours), plus any initialization/audit costs
  omitted by that progress forecast. The frozen selection forecast included
  all eight initial fits and a 25% cushion (5.00 hours at selection).
- STOP is intentionally present. Do not delete/rewrite scientific checkpoints,
  selection.json, source_snapshot, or completed units. No scientific code,
  settings, estimator, or scope was changed during this resume.

Restart from the repository root after the user releases the node:

```bash
~/.venv/bin/python -u -m mnist_experiment.combined_memory.run \
  --config mnist_experiment/combined_memory/configs/production.json \
  --root cache/mnist_experiment/plan18/production_v1 --resume --clear-stop
```

CUDA requires direct execution if the sandbox still hides /dev/nvidia*. Verify
GPU availability and source/checkpoint integrity, then use this exact root.
Completed calibration, initialization and arm A will be reused; arm B resumes
from the accepted batch, followed by all remaining frozen paired blocks.
Continue to final audit, concise notebook interpretation, and plan closeout.

## Resumed After Reservation (Latest State)

The user authorized resumption. Full read-only validation passed for all 47
existing units (46 completed, one incomplete). Scientific source hashes and the
protected mathematical notebook are unchanged. The batch-296 checkpoint checksum,
finite/nonnegative archive, and contiguous exposure ledger all passed again.
GPU/driver is healthy and actual free storage is about 577.5 GB.

The normal production command with --resume --clear-stop is active in invocation
22396. It reuses the frozen selection and completed artifacts; no scientific
settings changed. The bounded integrity check was charged to the family ledger.
Resume the planned six changing and two stationary paired blocks, then final
audit and reporting. Do not poll the older stopped invocations.

## Reporting Guard Repair (Latest State)

Invocation 22396 finished the first stationary block, then exited with a
reporting-only ValueError: 48 completed paths exceeded the 64 MiB aggregate
scalar-JSON input guard. No scientific unit was lost or partially completed.
The measured consumed exports total about 67 MB; the full cohort projects
comfortably below the new bounded 256 MiB guard. Only report.py and regression
tests changed; report.py is deliberately outside scientific source identity.
The immutable analysis implementation and all scientific settings are unchanged.

The focused report/analysis/scheduler suite passed 48 tests, including guard
enforcement and full-cohort input headroom. A production --report-only rebuild
also passed, preserving previous report generations. Both bounded tasks were
charged to the family ledger. execution_notes.json records this presentation
repair for the reader. The public notebook is current through the first
stationary block; no fresh efficacy stopping or retuning occurred.

Production resumed normally in invocation 59952 (22396 is finished). Completed
paths: development 33, fresh 10, stationary 5. Next is fresh replica 2; retain
the frozen six-changing/two-stationary cohort. About 4.93 cumulative hours are
charged, with more than 15 hours authorized remaining. Continue to closeout.

Latest progress in invocation 59952: three fresh paired blocks plus one
stationary block are complete (20/40 post-calibration paths). Fresh replica 3
initialization is underway. About 5.44 cumulative active hours are charged.
The reporting-only guard repair has handled subsequent publication normally;
no further changes or numerical failures have been observed. Read status.json
for progress newer than this milestone. Continue the remaining frozen blocks.

## Final Closeout (Original Study)

Invocation 59952 completed normally with exit code 0. The study completion
marker is cache/mnist_experiment/plan18/production_v1/study_completed.json,
dated 2026-11-07T08:55:55Z. All 73 trajectories are complete: 33 development,
30 fresh and 10 stationary. The final integrity decision verified 99 prior
units with full checksums, all eight five-arm paired blocks, finite archive
diagonals, and exact exposure/retirement ledgers. No incomplete block remains.

Final reporting presentation distinguishes unsupported inverse estimates from
pending work and marks progress Complete. The conclusion describes the new-data
versus retention trade-off, not an automatic multiplication of historical gains.
Final unit acceptance: 85 passed, one CUDA-only test skipped in the CPU run
(CUDA recovery/parity was separately validated before production). The production
report-only rebuild passed. All needed test/report/experiment sessions are
finished; do not poll 59952, 77595 or 37679.

Principal findings (nominal paired 95% intervals, six fresh replicas):
- E-B novel risk: +0.317 [-0.089, 0.717]; E-A: +0.220 [-0.223, 0.620].
  No demonstrated extra novel-learning advantage from the full combination.
- E-B old RMS difference at 100 novel grids: -0.150 [-0.239, -0.067] pixels;
  E-A: -0.110 [-0.201, -0.020]. The endpoint retention improvement is real
  exploratory evidence, not additional new-data sample efficiency.
- B-A point saving: 15.0%, matched-quality compute 1.45x, S=0.103.
  14.1% inverse-bootstrap failures and absent matched-budget retention support
  mean no ordinary inverse CI or demonstrated efficiency gain is claimed.
- C-A point saving: 2.85%, S=0.0153. D/E do not reach the common A target within
  100 novel grids. Equal-data E compute is 1.93x A and 1.24x B.

Cumulative active charge after final tests/reporting is 7.384 hours, excluding
node downtime; no further compute allocation is automatically used. Artifact
family is about 0.65 GB plus 0.067 GB cached MNIST dependencies; actual free
space is about 577 GB. The only production repair was reporting's 64-to-256 MiB
JSON input guard. Scientific identity, frozen settings, analysis.py and the
protected mathematical notebook remain unchanged. Preserve all completed units.

Public report: mnist_experiment/plan18_results.ipynb. The execution record at
the foot of plan18.md summarizes findings and qualifications. There is nothing
left to resume for this plan. Rebuilding the artifact-only report is optional:

```bash
~/.venv/bin/python -m mnist_experiment.combined_memory.run \
  --config mnist_experiment/combined_memory/configs/production.json \
  --root cache/mnist_experiment/plan18/production_v1 --report-only
```

Future scientific changes or added cohorts require a new explicit decision;
do not edit completed scientific artifacts or resume a new experiment silently.

## Post-Hoc Controller Audit

The user questioned the very low adaptive pi and requested that its plot use
pi rather than "New-evidence weight". Reporting now names pi explicitly and
explains that noise_scale is the one-observation term V, batch noise is V/4,
and movement includes both motion_energy and uncertainty_energy. This is a
presentation change only; no scientific sources, settings or units changed.

Read-only audit of the six fresh D/E pairs (11,136 recorded actions): direct
reconstruction of (S + q_ref V)/(S + q_ref V + V/4) agreed with raw_pi to
3.47e-18 maximum absolute error. The q_ref recursion agreed exactly. There
were no fallbacks, bound activations, covariance repairs, or floored observer
coordinates. For D, outward-ramp mean pi was .003304; the mean normalized
motion-plus-uncertainty ratio 4S/V was .0007497. D/E mean pi across the whole
path was approximately .003493. These are low estimates, not clipping artifacts.

A separate CPU audit recomputed fresh replica 0's 256 calibration scores,
checked two scores against J^T(y-f)/4, and replayed all 928 incoming batches
using independent NumPy Kalman matrix equations (subtractive covariance update,
not the production Joseph implementation). Calibration variance differed from
the saved GPU result by 6.79e-16 relatively; final mean/covariance/raw variance
relative differences were 8.40e-15 / 2.43e-15 / 1.72e-16. Innovation-ratio
maximum absolute difference was 3.78e-15. This took about six seconds after
loading and did not fit a model or modify any artifact. No arithmetic or
batch-normalization defect was found in the checked path.

Important interpretation: Plan 17 Phase 7 moved the entire incoming rotation
law, whereas Plan 18 uses a rare rotating branch (p=.125 h). Only 943 of 22,272
fresh grids were novel, and 85.06% of four-grid batches contained none. The
observer tracks all incoming scores, not the conditional novel holdout. Mixture
heterogeneity and diluted mean motion are plausible causes of weak detected
velocity; this is not evidence that novel-domain drift is absent. The selected
process scale 1e-5 and the already recorded fixed-anchor mapping discrepancy
are additional modelling concerns. Do not mislabel these hypotheses as proven
causes or silently retune using fresh confirmation results.

Focused observer, scheduler and reporting tests passed: 66 tests, including
the new pi-label check. No further experiments are authorized by this audit.

## Supplemental B-A Reporting Contrast

At the user's request, Fresh Paired Effects now begins with B-A, the direct
STM FD versus Replay/EWC comparison. It is a post-run reporting supplement,
using the original whole-paired-replica bootstrap seed, draw count and actual
global-sign strata. The frozen analysis.py and scientific source identity
remain unchanged. Reporting exports the additional row and its provenance;
all previously reported analysis fields were checked to be identical after
removing the supplemental entries. Prior report generations remain intact.

Six-pair B-A results (nominal 95% intervals): novel risk -0.0969 [-0.3170,
0.0206] pixels squared; Havg +0.00209 [-0.00322, 0.01049] pixels; H100
+0.03996 [-0.01039, 0.12963] pixels; current RMS -0.01163 [-0.03278,
-0.00020] pixels. Novel-risk mean improvement is driven by one replica;
the other five differences are slightly positive. The current-mixture RMS
contrast excludes zero nominally, not as a familywise significance claim.
Focused reporting/analysis/scheduler tests passed: 52. No new experiment ran.
