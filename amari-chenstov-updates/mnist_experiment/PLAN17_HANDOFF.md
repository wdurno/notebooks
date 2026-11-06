# Plan 17 Continuation

Execution authorized on 2026-11-01, with parallel coding agents permitted.
Read `plan17.md` and `kalman_pi/AGENTS.md`. The independent allocation is
20 cumulative active compute hours. Do not resume another historical study.

## Current State

Post-closeout interpretation update: the user requested an explanation of
velocity Kalman, inspection of the middle weight dip, and a practical conclusion
about reducing dependence on unavailable tuning trials. The notebook now has
66 cells, including a concise final Phase 7 application conclusion. Latest report:
`discrepancy_v1/report/generation-1742ed392e8d7eab3b7e322f`; audit
`closeout_validation-1b5c10079c07.json`. All 38 reporting tests passed; the full
audit rechecked 207 units, 42 scientific sources and 52 inherited cells.
All earlier report files, scalar results, figures and scientific seals are
unchanged. The earlier closeout generation below remains preserved. Current
notebook SHA256: `20f656f7b69995df1e3667f075528fe783b8b7cf6b27a64a7daa7ac3423c1555`.
The all-batch weight review and its definitions are recorded at the foot of
`plan17.md`. No new experiment was run or authorized.

**Phase 7 execution was authorized on 2026-11-05 with a separate 20-active-hour
cap. Implementation is under `kalman_pi/discrepancy/`; final acceptance passed
75 unit tests and six integration tests (81 total, 154.64 seconds), including
actual process death and publication rollback. Execution and closeout are complete in
`discrepancy_v1`: all 40 development and 80 fresh paths, 20 paired fresh blocks,
and 207 sealed units validate. No experimental worker remains. At
2026-11-05 18:03:08 UTC, selection froze common
ridge zero, velocity eta .0001, discrepancy eta .00001, eight changing replica
identities shared across 25/45 degrees and four stationary identities: all
80 fresh paths / 20 four-policy blocks. The conservative sealed fresh forecast
was 7.916 additional active hours plus the reserved closeout hour; approximately
2.97 Phase 7 hours had been charged then, and 9.336 including final checks.
No efficacy gate or optional eta extension
was used. Do not tune or expand the fresh cohort from interim results.**
PyTorch CUDA works outside the sandbox. Phase 6 was rechecked read-only: all
113 completed units and all scientific source hashes remain intact.
The corrected CUDA CLI acceptance completed all 69 units / 36 tiny paths,
preserving all six pre-stop seals and every scientific source hash.
Phase 7 compute is persistent in `phase7_acceptance_compute.json` plus independent
`plan17-phase7` run roots. Do not reset earlier ledgers.

Scientific-closeout report: `discrepancy_v1/report/generation-c77d0410aa477483de140421`.
Scientific-closeout audit: `closeout_validation-08fd332a337f.json` under that study
root (the latest pointer now selects the interpretation update above). It verified all 207 units, 42 scientific
sources, 20 paired blocks, 52 inherited cells/metadata and four new nonblank
figures. The notebook has 65 cells / 19 embedded figures. All 49 earlier seals
are unchanged. Actual completed `--resume` returned `already_completed` and
preserved all 207 seals and notebook SHA256
`7f353edb07be6191179dd99dc3ece1ab46f96c292c330279ad8184cb593aa52d`.
Control status confirms `writer_active: false`. Do not launch new learning.

Velocity-minus-fixed current RMS gaps are -.014253 [-.017766, -.011069]
at 25 degrees, -.013067 [-.018030, -.008104] at 45, and -.013940
[-.023410, -.007169] on stationary paths. Both ramp directions favor velocity
at each amplitude; whole-path one-pixel accuracy gains are about +.81 / +.79 pp.
Discrepancy-minus-fixed gaps are +.092896 / +.079946 / +.078882 pixels,
with every nominal interval above zero. Its mean weights are near .19 and it
hits the upper bound on 83.0% of changing / 73.7% of stationary batches on
average within trajectories. Fixed-anchor consistency is poor; do not claim a
unique cause or a mathematical refutation. Velocity has worse fixed endpoint
panel error, so this is not a joint or retention-matched win. No S claim.

The reporting-only final revisions compacted the table and fixed titles,
stationary shading and signal-axis readability. Scientific code/settings and
all historical cells are unchanged. Retained test failures were a legacy
disclaimer-string assertion and a new plotting test that inspected axes after
the shared helper cleared the figure. Final 81-test suite passed. All costs,
including failed checks and 120 seconds of conservative read-only closeout
overhead, are charged; historical 18.619 hours stay separate. Production is
about 2.205 GB, total cache about 171.1 GB, free disk about 577.6 GB.
The near-top notebook clock is the last scientific snapshot, preceding the
separately charged final checks. Nothing is queued after this completed phase.

The new four conditions are fixed .025, covariance-only, velocity Kalman, and
conditional-discrepancy Kalman. Discrepancy uses actual learner parameters and
predicted level covariance including the cross term; it never adds the marginal
old q variance again. Both amplitudes (25/45) use padded 40px images, 448 parameters,
two 352-batch holds and two 304-batch ramps. Current-law holdout risk is primary.
Resolved production and tiny CPU/CUDA configurations are in `discrepancy/configs/`.
The planned production root is `cache/mnist_experiment/plan17/discrepancy_v1`.
Completed CUDA acceptance root is `discrepancy_acceptance_cuda_v2`, resumed after
an intentional six-unit stop. The first CUDA root exposed a deterministic
adaptive-pooling backward limitation. On the fixed 20x20 feature map, AvgPool2d(10)
is mathematically equivalent to adaptive 2x2 pooling; CPU value/gradient checks
agree within float64 rounding. Keep deterministic execution enabled. The
69-unit CPU v1 acceptance completed before this equivalent operator replacement
and the drift/selected-ridge solver audit refinement; preserve it as prototype
acceptance, not a source-compatible resume target. Current CPU integration checks
also verify that three of four doubled-budget solver audits cover drift.
All six pre-stop CUDA v2 completion hashes are retained for recovery verification.
After acceptance and source freeze, execute/resume using:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.discrepancy.run \
  --config mnist_experiment/kalman_pi/discrepancy/configs/production.json \
  --root cache/mnist_experiment/plan17/discrepancy_v1 \
  --allocation-hours 20 --resume
```

Add `--clear-stop` only for an explicitly requested restart after a STOP marker.
The scheduler freezes settings before fresh outcomes and targets 80 fresh paths
with the balanced 48-path resource fallback. It preserves the completed Phase 6
notebook generation, adds a chronological Phase 7 section, and never trains from
the notebook. No tmux is required for this local session.

Phase 7 final audit support is in the reporting/acceptance-only module. It
passed on CUDA v2 (69 units) and final production (207 units, 42 sources,
80 fresh paths, 20 paired blocks, 52 inherited notebook cells). All 49 early production completion seals
were rechecked after learning and remain unchanged. The following command is
verified. All seven runtime/acceptance unit cases passed in 1.10 seconds
while production calibration was running, within its active wall-clock charge;
no separate overlapping acceptance reservation was created.

```bash
~/.venv/bin/python -m mnist_experiment.kalman_pi.discrepancy.acceptance \
  --timeout 300 --audit-root cache/mnist_experiment/plan17/discrepancy_v1
```

It validates completed units, source/selection identities, all trajectory rows,
the completion marker, frozen cohort, inherited notebook cells/metadata and
nonblank figures.
Each invocation appends a uniquely named audit record; `closeout_latest.json`
points to it. Acceptance now holds the family allocation lock until reserved
test time is settled, so do not run it concurrently with a production worker.

### Historical Phase 6 State

**Phases 0--5E are complete. Charged historical compute is approximately
14.318 / 20 hours. Phase 6 execution was authorized on 2026-11-04, with a new
20-hour maximum allowance. Phase 6A--6E is complete, using approximately
4.30 / 20 additional active hours (about 18.62 hours including history).** Its
single-digit registration design is in `plan17.md`. Do not launch a historical
command to execute it or reset the historical ledger. The new package is
`kalman_pi/registration/`, with separate source/config identities and acceptance.
The user restored the driver and authorized continuation after integrity checks.
Those checks passed: 41 CPU seals, 62 checkpoint files, 31 scientific sources,
all 39 historical notebook cells, and a new 101-test CPU pass. PyTorch CUDA
works outside the GPU-restricting sandbox. CUDA stop/resume acceptance completed
41 units / 18 tiny paths with all six pre-stop seals unchanged. The temporary
Phase 6A hold was lifted and production finished in `registration_v1`.
No Phase 6 worker remains. The later Phase 7 authorization above supersedes the
historical instruction against starting a subsequent study.

### Phase 6 Final State

All 16 development and 48 fresh paths are complete: eight changing plus four
stationary four-arm blocks, 113 sealed units. Frozen settings: angle 25 degrees,
ridge .01, fixed pi .025, EMA H8, Kalman eta .0001. No ridge companions were
needed. Completed `--resume` returns `already_completed` without training.

Changing Kalman minus fixed RMS gaps: old -.001328 [-.006407, .003929], shifted
+.008807 [.003300, .014313] pixels. No joint applied benefit: fixed has a small
shifted-error advantage, with no established Kalman old-domain gain. Covariance
and EMA improve old retention at greater shifted-error cost. Stationary Kalman
old gap -.002773 [-.005794, .000248], shifted +.059840 [.039512, .080167]; its
shifted panel is untrained-domain generalization, not acquisition. All CIs are
nominal exploratory whole-replica bootstrap intervals. No equivalence or S claim.
Changing Kalman learning/wall ratios .8651 [.8400, .8906] / .9700 [.9612, .9789].

Final tests: 158 unit and six integration passes. Full-file/source validation
passed all 551 current accepted/historical units; all 31 pre-repair production
seals remain identical. All 39 inherited notebook cells and metadata survive;
the canonical notebook has 52 cells / 15 nonblank PNGs. Final generation:
`registration_v1/report/generation-71e194b6c26aaaa7be639182`.
Final integrity record: `registration_v1/closeout_validation_final.json`.
The actual completed-resume check preserved all 113 scientific completion seals
and the canonical notebook bytes; control status confirms no active writer.
Source manifests are unchanged. Final reporting restored saved diagnostics,
fixed initial-action omission/frozen means, simplified direction-pooled legends
without grid interpolation, and preserved small-effect CI precision.

The post-run nested noise audit reused already-seen labels only: 1024 versus
2048 examples, Fisher-weighted noise ratios .99664 / 1.01822. It is a correlated
consistency diagnostic, not truth or retuning. The bounded audit/publication CLI
is `kalman_pi.registration.acceptance --closeout --root <root>`; repeat calls
reuse its source-keyed noise artifact. All audit invocations are separately
charged in `phase6_acceptance_compute.json`. Production is about .67 GB; disk
free exceeds 540 GiB. Retain the earlier prototype roots without pooling them.

### Phase 6 Execution Record

Historical continuation record: all 16 development paths completed. Frozen at
2026-11-04T17:00:40Z: 25 degrees, common ridge .01, EMA H8, Kalman eta .0001,
eight changing plus four stationary blocks, four arms (48 paths). Development
Kalman old RMS gap -.00432 pixels is within the +/- .10 band; no ridge companions.
The production worker was gracefully stopped at the first fresh fixed path
to repair a reporting-only recovery defect. No fresh path had completed.
The then-current combined suite passed **154 tests** (81.19 seconds); the worker
resumed from durable accepted step 98 with `--resume --clear-stop` and finished.

Reporting repair: incomplete live previews can alias an uncommitted metric row,
and scheduler startup temporarily omits the active partial path. Treat their
metrics/presence as provisional in report validation, while keeping identity,
completed trajectories, settings, history and compute monotonicity strict.
No scientific code or learning state changes. Tests for rollback/disappearance,
timing replay and completed-data safety pass, including an actual full-scheduler
checkpoint-abort/publication/resume test. All 31 pre-stop seals and all 31
scientific source hashes are unchanged. The conservative 240-second test/report
charge is in the Phase 6 acceptance ledger. Also corrected group
frozen plot references to the mean initial value, clarified readiness MSE units,
and derived retention-band text from the frozen development candidates. Final
legend/initial-action fixes also passed. Preserve historical cells and generations.

Implementation is present under `kalman_pi/registration/`: standalone config,
448-parameter Gaussian learner, exact integrated-label diagonal Fisher, sealed
source/transform plans, causal common-anchor observers, resumable pretraining
and accepted-arrival updates, bounded scheduler/selection, independent allocation,
artifact-only analysis, and a concise chronological notebook appendix.
The initial CPU suite passed **101 tests**, including a real process death and
exact model/Fisher/q/controller/RNG/predictive continuation, full tiny scheduler,
per-batch pretraining checkpoints, role/pairing isolation, storage enforcement,
readiness fallback and historical report preservation. GPU acceptance also passed
after the reboot; the instructions below record the verified acceptance procedure.

CPU CLI acceptance artifacts are under
`cache/mnist_experiment/plan17/registration_acceptance_cpu_v3`.
The final CLI stop/resume completed **41 sealed units / 18 tiny paths** and
passed full checksum validation. A completed-run resume exits without training.
The last focused post-addition checks pass 13 tests. The post-reboot full suite
again passed 101 tests. The new allowance had charged approximately **0.11 hours**
before reboot; read the ledger for current spending. Production was launched
after successful CUDA acceptance and has completed. Its command below is now
a verified completed-run no-op.
Earlier v1/v2 CPU roots preserve prototype acceptance; their source identities
precede final repairs, so do not resume or pool them. The canonical completed
report is under `cache/mnist_experiment/plan17/registration_v1/report/` and
`plan17_results.ipynb`. That root now has a scientific identity and production
units. All 39 inherited Phase
0--5E notebook cells and prior metadata are preserved verbatim.

Acceptance compute is charged by `phase6_acceptance_compute.json` and separately
clocked CPU smoke roots. These do not reduce the old 20-hour allowance; Phase 6
has its own newly authorized 20 hours. Never reset either ledger. Both production
development capability checks passed at the default 25-degree shift. Replica 0
frozen old/shifted RMS was .741/1.989 pixels and fitted shifted RMS was 1.183;
replica 1 was .761/2.075 and fitted 1.301. Context, selection and all fresh
replication are complete; the final settings/outcomes are recorded above.

The following post-reboot checks and acceptance commands have been completed.
They remain as recovery documentation, not work to repeat automatically:

```bash
nvidia-smi
~/.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

The deferred tiny CUDA acceptance included an intentional
six-unit stop and resume, under the phase allocation:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.registration.run \
  --config mnist_experiment/kalman_pi/registration/configs/smoke_cuda.json \
  --root cache/mnist_experiment/plan17/registration_acceptance_cuda_v1 \
  --allocation-hours 20 --max-units 6 --resume
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.registration.run \
  --config mnist_experiment/kalman_pi/registration/configs/smoke_cuda.json \
  --root cache/mnist_experiment/plan17/registration_acceptance_cuda_v1 \
  --allocation-hours 20 --resume
```

The six prior seals remained unchanged; completed output validates, and
completed resume returned `already_completed` without training. CUDA smoke
stays under its own root; do not treat its tiny data as applied findings.
The authorized production worker was launched with the following command.
Use it for resume only when no worker owns the study:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.registration.run \
  --config mnist_experiment/kalman_pi/registration/configs/production.json \
  --root cache/mnist_experiment/plan17/registration_v1 \
  --allocation-hours 20 --resume
```

Use existing `kalman_pi.control status/stop --root <root>` interfaces. Only add
`--clear-stop` on a user-requested resume after an actual STOP marker. A reboot
releases the writer lease; do not delete completed artifacts or lock files.
Local operation needs no tmux. Significant corrections must be recorded in the
notebook and receive a new scientific identity/root if completed units exist.

The reporting-only Phase 5E publisher now rejects a canonical notebook carrying
the Phase 6 marker. Its historical sealed generations and all three older
studies' scientific source hashes remain unchanged.
The Phase 5 primary root is `retention_v1` (120 development / 48 fresh paths,
205 sealed units); the supplemental root is `retention_supplement_v1` (eight
three-arm blocks / 24 paths, 49 sealed units). Both completed resume commands
return without training. All 356 scientific units across the three production
roots passed final full-file/source/seal checks. The canonical notebook has
11 nonblank embedded figures, preserving every inherited primary cell.

Key supplemental result: removing isotropic ridge improves both accuracies
under fixed pi. Against fixed pi .025 at common zero ridge, Kalman changes
mean 9 recall by +.0277 pp [-2.3712,2.2771] and old accuracy by +2.1144 pp
[1.5616,2.5812]. Evidence is a retention benefit, not an established acquisition
gain or equivalence. Online wall time is 1.0372x [1.0247,1.0489]; learning-only
time is .8560x [.8283,.8819]. Preserve both cost definitions. Primary fresh
retention matching did not generalize; all six acquisition contrasts were
negative. See the notebook and plan closeout tables for all comparisons.

Final validation: `retention_supplement_v1/closeout_validation_final.json`.
Final report: `generation-28161cd6335c51e03e5582cc`. Earlier full suite: 376
passed; final reporting-only suite: 67 passed. Both actual completed-resume
checks preserved all seals and notebook bytes. No Plan 16 sources changed.

## Historical Execution Context

The original implementation,
acceptance, both shadow drivers, all 22 ordinary development paths, and all 48
fresh trajectories are complete. The final root is `production_v4`, using
`configs/calibrated.json`. All 12 fresh four-arm blocks are paired, and all 102
sealed units passed full-file checksum verification. Charged active compute is
approximately 4.14 hours out of 20; shared accounted storage is approximately
1.6 GB out of 500 GB. Do not extend the frozen cohort or launch another study
without user direction. The first root `production_v1` is stopped for
a refresh-linked noise-normalization diagnosis. A versioned fixed-anchor
ablation completed in `production_v2`; its main shadow ratios are now 1.15--1.39
with zero boundary activations. A final bounded full-initialization calibration
check completed in `production_v3` and improved main innovation ratios to
.91--1.16. The last allowed adjacent-decade screen shifts eta to
{.00001,.0001,.001}; prior .01 results remain in v3. The new package reuses
frozen Plan 16 learner/model/data/lifecycle helpers without editing them.
All arms use direct-EMA diagonal Fisher updates, not LFUs, STM or FD KL.
The four policies are fixed, covariance-only, anchor EMA and anchor Kalman.
Selection froze at 2026-11-01 22:04 UTC: c=.025, ridge ratio=.1, EMA H=16,
Kalman eta=.0001. Fresh allocation is eight main and four stationary replicas,
four policies each (48 completed trajectories). The pre-fresh forecast was 9188
seconds (2.55 h), with separately reserved contingency and closeout allowance.
Do not tune or extend from fresh outcomes. The family had used about 1.55 active
hours at the freeze, including the earlier revisions and acceptance checks.

Core CPU tests verify score signs, Joseph covariance PSD, scalar Kalman parity,
Holt recursion, common-anchor recentering, exact nominal concentration and
no duplicated assimilation. Four CPU interruption/resume comparisons reproduce
the full learner/filter state exactly. A mid-refresh interruption rolls back
the entire uncommitted transition. End-to-end CPU and CUDA smoke studies
completed; CUDA resumed after an intentional time-limit stop. The focused suite
passed 173 tests, including scheduler and strict fresh full-block checks.
Parallel review caught and repaired an overflow-safe risk-ratio edge case and
restart-dependent initialization cost forecasting. Final production source is
frozen; reporting/analysis changes remain independently generation-versioned.
The worker was checkpointed and resumed twice to load reporting-only fixes:
chronological shadow regime labels, helper-aware report generation hashes,
unobstructed legends, and a separate frozen reference for each length cohort.
The latest reporting-only suite passes 60 tests. Final presentation emphasizes
full-path effects and all five predefined regimes; endpoint statistics remain
in the detailed analysis. Diagnostic axes respect physical parameter ranges.
No learning state, settings, accepted arrivals or scientific identity changed.
An actual node restart was also recovered during fresh replication: all 21
previously completed fresh trajectories survived, and the interrupted Kalman
path resumed from its accepted-arrival checkpoint. The full cohort then finished.
Smoke notebooks live under their respective run roots, not the canonical report.

## Phase 5 Execution

On 2026-11-02 the user authorized Phase 5 implementation and execution. The
new implementation is under `kalman_pi/retention/`; the historical source
manifest still matches `production_v4` exactly. The completed `production_v4`
run stays sealed; its resume command is not the follow-up entry point.

The reference remains fixed pi .025, ridge .1. Calibrate only each adaptive
policy's ridge, separately by schedule, toward a development retention gap
within +/-0.5 pp. The plan gives bounded grids, fresh cohorts, unmatched-result
handling, and a 9--14 additional active-hour forecast within the remaining
approximately 15.86 hours of the original family allocation. It requires new
explicit pairing across selected unequal ridge values; the historical analysis
groups by common ridge and remains valid for original results. The Phase 5
report builder loads a pinned immutable historical report and appends live
results without rebuilding the old six figures. The old publisher rejects an
attempt to overwrite the combined notebook. The selection receives its own
immutable unit before fresh confirmation, and resume verifies it.
Planning publication passed all 60 reporting tests and preserved all 762
non-report files in the completed root plus all six original figure outputs.
A separate conservative 60-second planning/report charge is appended to the
family acceptance ledger; read that ledger for the exact remaining allocation.

Phase 5 acceptance: the final focused suite passes **95 tests**, including
selection, reporting, scheduler recovery, publisher ownership, and two actual
process-death tests. CPU and CUDA end-to-end tiny studies
each completed 14 development plus eight fresh paths; CUDA resumed after a
bounded six-unit stop. These are smoke acceptance only, not scientific results.
The family acceptance ledger charges 120 seconds for unrooted implementation
tests; smoke roots have their own clocks. The production command is:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.retention.run \
  --config mnist_experiment/kalman_pi/retention/configs/production.json \
  --root cache/mnist_experiment/plan17/retention_v1 \
  --allocation-hours 20 --resume
```

Use the existing `kalman_pi.control status/stop --root <study-root>` commands
for this new root. Add `--clear-stop` only when resuming an explicitly requested
stop. The scheduler runs all remaining Phase 5 sub-phases; no tmux is required
for this local session. A reboot requires this explicit resume command.

Phase 5B completed 120 development paths (114 grid plus six stationary
refinement paths), with no numerical failures. The immutable selection froze
at **2026-11-02 21:49:51 UTC**, before fresh outcomes. All six selected settings
are inside the development +/-0.5 pp band. Selected ridges are main:
covariance=.3, EMA=.3, Kalman=1; stationary: covariance=EMA=.05477225575051661,
Kalman=.17320508075688773. Fixed remains .1 on both schedules. Main old gaps
are +.2525, +.2664, -.3138 pp; stationary gaps are -.0778, -.0346, +.0121 pp.
All development acquisition gaps at these matched settings are negative; retain
them transparently. This does not determine the fresh comparison.
Fresh allocation remains all **eight main plus four stationary blocks / 48
paths**, with no resource reduction. Pre-fresh forecast was 9914.6 seconds
(2.75 h), plus contingency/closeout. The family used about 10.21 hours at
confirmation start. Continue frozen settings without retuning or extension;
the separately approved Phase 5E follows after the primary worker exits.

Early development observation (30 sealed trajectories; not confirmation):
covariance-only at ridge zero gives +3.5241 pp mean 9 recall and +7.8869 pp
mean old accuracy versus fixed pi .025/ridge .1 on four changing replicas.
Its stationary old/9 gaps are +7.3185/-1.9128 pp on two replicas. These are
favorable but retention-unmatched operating points, visible in the full
development map. Preserve and discuss them even if the matching-band rule
selects a different ridge. They do not establish adaptive pi's contribution
separately from ridge, or fresh confirmation; do not silently change the
reference, matching band, grid, or cohort because of this lead.

At 42 completed development paths, Kalman/ridge-zero similarly gives changing
old/9 gaps +7.8757/+4.3997 pp and stationary +8.0447/+0.8872 pp versus the
fixed reference. EMA/ridge-zero is close to covariance-only. These remain
development means, not fresh confirmation or an isolated ridge-vs-policy
comparison. The authorized grid and matching-band selector continue unchanged.

The user approved a separate Phase 5E confirmation of the zero-ridge lead on
2026-11-02. Its scope is eight independent fresh changing-schedule
replicas, three paired arms (fixed pi .025/ridge .1, fixed pi .025/ridge zero,
Kalman/ridge zero), about 1.5--2 additional active hours within the same 20-hour
family cap. Implement it in the nested `retention/supplement/` package so the
running primary scientific source identity stays unchanged. Execute it after
the primary Phase 5 grid and fresh cohort, which continue unchanged.
Seal a separate protocol before its fresh outcomes; preserve the
primary study-wide data-role/dev-fresh partition, with independent new stream
and initialization seeds, and report the supplemental results separately.

The supplemental implementation is under `kalman_pi/retention/supplement/`.
The initial combined suite passed **375 tests** in 98.03 seconds, including
analysis/report acceptance, independent parent code review, complete three-arm
CPU interruption/resume and actual notebook publication. All CPU acceptance
overlapped the already charged primary wall clock; do not double-charge it. Do not
start a second family writer while the primary worker is running. After the
primary completes, finish acceptance and use this separate command:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.retention.supplement.run \
  --config mnist_experiment/kalman_pi/retention/supplement/configs/production.json \
  --root cache/mnist_experiment/plan17/retention_supplement_v1 \
  --allocation-hours 20 --resume
```

It pins the completed primary report, seals its three-arm design before learning,
and appends Phase 5E without rebuilding earlier evidence. Its independent seed
is 171103; the primary partition seed stays 171102. The eight supplemental
blocks do not enlarge or retune the primary confirmation cohort. The original
reference and zero-ridge fixed control both use pi .025; the third arm is
Kalman eta .0001 at zero ridge. Inspect the cumulative ledger before launching;
unstarted supplemental blocks must leave a 0.5-hour reporting reserve.

The primary worker exited cleanly at 2026-11-03 00:22 UTC. A reporting-only
ownership guard in `retention/report.py` now rejects a target notebook containing
`metadata.plan17_zero_ridge_report`. It was added only after worker exit.
Candidate annotations and repeated conclusion text were also simplified without
changing any scalar analysis or scientific unit. The original reporter refuses combined
Phase 5 ownership; the supplemental reporter prevents stale/foreign notebook
replacement and verifies all inherited primary output bytes. Before supplement
launch, the tiny CUDA acceptance in `retention_supplement_smoke_cuda` completed
with its own clock: six paths / two full paired blocks / 13 sealed units after
an intentional six-unit stop and resume. All six prior seals were unchanged.
This smoke root is completed, not a production resume target.

Production Phase 5E began at approximately 2026-11-03 00:32 UTC, using all
eight changing blocks / 24 paths in `retention_supplement_v1`. The worker's
console log is under that root; exec session 2746 exited successfully after
completion at 2026-11-03 02:02 UTC. The primary worker had already exited.
The command above now returns `already_completed`, not additional training.
The family used about 12.81 h at start, with no allocation extension.

Primary closeout: 120 development plus 48 fresh paths, 205 sealed units, all
full-file checksums and source identities verified alongside all 102 original
units. The artifact `retention_v1/phase5_closeout_validation.json` records the
audit, analysis recomputation, all 12 paired fresh blocks, nine nonblank embedded
images and preservation of the six historical PNG outputs. Full suite: 376
passed; final reporting-only suite: 66 passed. Current primary report generation
is `generation-53879f12c043ed81053edced`; the supplement should pin that generation.
All six fresh old-accuracy point gaps exceeded the development band. Kalman
changing 9/old gaps are -7.6946/+1.0961 pp (intervals [-11.0613,-4.9108] and
[-.1234,2.2120]); stationary -1.6677/+.6219 pp ([-3.1445,-.8799] and
[-.4951,1.7981]). Interpret as remaining trade-offs, not matched fresh retention.
No retuning or allocation extension was performed. Family compute was about
12.76 h at primary exit. A separate conservative 180-second acceptance/closeout
entry covers final unrooted checks; CUDA acceptance is separately clocked.

## Scientific Implementation Notes

The v1 main shadow pilot exposed extreme innovation underprediction linked to
anchor refreshes. At step 1280--1281 its eta=.001 noise scale fell from 1.522e9
to 144.1 and raw pi jumped .01270 to .99952. Preserve v1; it has no fresh
replication. The `production_v2/changes.json` ledger records a single-setting
ablation, `anchor_refresh=0`, with unchanged data/learner/observer grids. Its
five-unit probe completed and was inspected before continuing. The v2
fixed-anchor check succeeded in removing the severe calibration artifacts.
Because its calibration subset had zero 9s (nine exist in initialization), v3
uses all 2048 already-seen initialization examples to set the frozen noise
shape. It adds no labels. Its five-unit probe also completed. The selected v4
configuration keeps that repair and shifts the process-noise grid down by one
decade. Its shadow innovation ratios are 1.01--1.16 (main) and 1.04--1.17
(stationary); these are calibration diagnostics, not applied efficacy.

- Actions are frozen before reading their own training batch. Observed-label
  scores at the common anchor inform only subsequent actions; model-label
  integration supplies Fisher curvature, never the discrepancy mean.
- Initialization noise calibration uses all 2048 named, already-seen
  initialization-training examples, not extra labels or holdouts. Earlier v1/v2
  used a 256-example subset; preserve that distinction when comparing revisions.
- Scores, covariance/filter algebra and Fisher accumulation use float64.
  Training uses the shared Plan 16 diagonally scaled physical objective.
- Holt means follow the specified recursion. For shadow innovation scoring,
  its fixed-gain covariance is propagated with measurement noise and zero
  process noise; this covariance is not added to Holt's movement energy.
- Shadow observers share a fixed-policy driver and its actual q. Their
  hypothetical recommendations do not alter the driver or its weight history.
  Driver units are separate from ordinary predictive comparisons and timings.
- Reanchoring leaves velocity/covariance/evidence counts unchanged. Raw score
  variances are reset from paired available examples under the new map. The
  paired mapping-error diagnostic is not an extra assimilation or a p-value.
- Fresh length factors in {1,1.25} are sampled before learning and shared by
  paired arms. Reporting must keep different curve grids visibly separate;
  inference uses complete paired replica paths, never interpolated time points.

## Execution And Recovery

Use `~/.venv/bin/python`. The verified historical execution/recovery command:

```bash
~/.venv/bin/python -u -m mnist_experiment.kalman_pi.run \
  --config mnist_experiment/kalman_pi/configs/calibrated.json \
  --root cache/mnist_experiment/plan17/production_v4 \
  --allocation-hours 20 --resume
```

The completed root now returns `already_completed` without launching learning.
Earlier v1--v3 roots are retained
development diagnostics, not resume targets. Check actual writer state before
launching another invocation; no fresh outcomes were used in these revisions.
Use the same command with `--resume` after interruption. If a cooperative stop
was requested, add `--clear-stop` only when explicitly resuming that request.

```bash
~/.venv/bin/python -m mnist_experiment.kalman_pi.control status --root <study-root>
~/.venv/bin/python -m mnist_experiment.kalman_pi.control stop --root <study-root>
```

The scheduler advanced all phases without conversational polling, in a dedicated
tmux server/session. That worker has exited; there is no session to attach to.
Its log is `production_v4/console.log` under the Plan 17 cache family.
Execution did not auto-resume
after a reboot. No restart loop should override a user-requested stop.

Each completed unit is immutable. Checkpoints retain model/Fisher, previous
parameters, action/q/nominal weight history, observer/anchor/noise states,
metrics, RNGs and active clocks. Scientific source changes require a new root;
report-only edits are separately generation-versioned. Shared storage ceiling
is 500 GB for Plans 16/17, with a 20 GB filesystem reserve. Do not prune evidence.

## Reporting

Canonical notebook: `plan17_results.ipynb`. Computation belongs to entry points,
not the notebook. Keep its progress near the top, stages chronological and
conclusion table concise. Show both accuracy outcomes and real measured cost.
No success gate, significance stopping or automatic allocation extension.

## Results And Artifact Index

Kalman versus fixed on changing schedules gives -0.94 pp [-1.63, -0.27] 9
recall and +1.03 pp [0.57, 1.48] old retention, with 3.4% more measured wall
time. Stationary results show the same acquisition/retention trade-off. Against
covariance-only on changing schedules, Kalman improves both metrics. Its return
ramp gains do not persist through the final low hold. These are exploratory
nominal 95% paired intervals, not a claim of an optimal policy or sample savings.

Under `cache/mnist_experiment/plan17/production_v4/`:

- `study_completed.json`: original scientific closeout marker; preserve it.
- `selection.json`: frozen settings, cohort, and pre-fresh forecast.
- `closeout_validation_final.json`: 102 units checked with full file hashes,
  unchanged seals, 12 paired blocks, six nonblank notebook images, and no-op resume.
- `report/latest.json`: final report generation, scalar analysis, and figures.
- `changes.json`: pre-fresh repair evidence. Earlier roots remain unpooled.

The canonical notebook is fully populated, with a six-row conclusion table and
separate changing/stationary results. Rebuilding it is artifact-only:

```bash
~/.venv/bin/python -m mnist_experiment.kalman_pi.report \
  --root cache/mnist_experiment/plan17/production_v4
```

Phases 0--4 and Phase 5A--E are complete. No Plan 16
sources/artifacts were changed, and no subsequent plan has started.
Use the supplemental report publisher for the current combined notebook:

```bash
~/.venv/bin/python -m mnist_experiment.kalman_pi.retention.supplement.report \
  --root cache/mnist_experiment/plan17/retention_supplement_v1
```

The historical and primary publishers require separate `--notebook`
destinations once Phase 5E owns the canonical notebook. Neither can remove
the new section; all older report generations remain sealed.
