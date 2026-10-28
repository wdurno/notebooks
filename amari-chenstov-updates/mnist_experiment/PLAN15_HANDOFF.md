# Plan 15 Handoff

## Phase 11 Draft

Phase 11 in plan15.md was drafted for review on 2026-10-28: joint nine-digit
registration with 54 targets and a shared CNN/dense head. No Phase 11 code or
experiment has been started. The draft proposes a 20-hour cumulative allocation,
an upfront Fisher/compression timing check, three adaptive controls, and a
32-grid/two-replay-grid resource convention. Execution awaits authorization;
Phase 10 remains complete.

The 2026-10-28 resource amendment adds a 10--20-minute initial assessment
(30-minute active cap, included in 11B and the 20-hour total). Measure exact
CPU/CUDA compression and whole accepted updates before assuming the larger
model is affordable. Keep exact eigh when costs permit; otherwise consider
QR-stabilized block power iteration, reorthogonalized block Lanczos, or a
PSD-aware Nystrom/operator approach according to the measured bottleneck.
The plan specifies trace/diagonal, low-rank-breakdown and exact-reference
acceptance checks. No benchmark or backend implementation has been run yet.

## Phase 10 Complete

Phase 10 completed on 2026-10-27: **230 trajectories** (156 development, 72 fresh
confirmation, two independent pilots), with **12 fresh paired replicas per task**
and 2,000 paired bootstrap draws. There is no pending Phase 10 training. The
production tmux process exited successfully; inspect live state rather than
treating its recorded PIDs as active. The study used approximately **8.4 active
hours including closeout checks**, below the single 20-hour allowance; offline
node pauses did not renew or consume the allowance. New data occupies roughly
1--2 GB, with less than 8 GB in the shared 500 GB budget, including prior Plan 15
artifacts and inputs. Read the final status/storage ledger for exact counts.

Both tasks show small positive budget-averaged acquisition effects:

- Vector replay-minus-FD A: **0.010589 square pixels**, nominal paired 95% CI
  **[0.004562, 0.016832]**; 9/12 favorable, median gain 0.011657. FD uses **2.434
  times** equal-budget replay learning compute.
- Scalar gain: **0.002606 square pixels**, CI **[0.000966, 0.004225]**; 8/12 favorable,
  median 0.002477. FD uses **2.087 times** equal-budget compute.

Full-budget retention is supported and equivalent within the registered
tolerance for both tasks. Vector **14.79% fewer novel pairs** (85.21 versus 100)
at **2.078 times matched-quality compute**, $S=0.0712$, is a **descriptive point
estimate**, not confirmed sample savings. One of 2,000 inverse draws is unresolved,
so the prespecified rule withholds ordinary inverse intervals and matched-budget
certification. No draw was discarded. Scalar matching fails at the endpoint;
1,157/2,000 draws are unresolved. Targets improve frozen risk by 10.75% vector
and 13.65% scalar. These limitations are distinct from failed full-budget retention.

The secondary common-y1 gain contrast favors the vector context: 0.008733,
CI [0.002546, 0.015747] square pixels. This is compatible with the richer-response
lead, not proof of output dimension or STM-to-LTM transfer. Both tasks benefit;
head capacity, pretraining and extra supervision also differ.

All **345 immutable units** passed a full checksum audit with no incomplete
units. The accepted scientific source hash remains
369c81456c0747d160163bd7c60e9a5418128072d312664db4586e7fa9680306.
The final notebook retains all 65 historical cells. Initial CPU/CUDA recovery
acceptance remains valid; final fast unit and report checks passed. Reporting-only
changes add capacity/readiness context, label descriptive savings and explain
withheld inverse uncertainty. The efficiency plot now displays sqrt(mean replica
risk), consistent with the stored inversion, rather than mean replica RMSE;
no stored analysis or learning result changed. Predictive panels retain mean RMSE.
Development stars and relative costs also use the initial grid cohort, avoiding
mixed two-replica points and four-replica selection scores in one frontier.
At the user's request, Interpretation now opens with a compact vector/scalar
comparison table, prominently including S, sample savings, both compute ratios,
retention and inverse failures. The reporting template preserves it on refresh;
no scientific result or uncertainty rule changed.

An analysis-only refresh consumes sealed analyses and performs no training:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory_registration.run \
  --root cache/mnist_experiment/plan15/registration_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb \
  --stage analyze --resume --clear-stop
```

Run this only without another writer. Unlike the standalone --stage report
shortcut, it retains budget-aware supplement publication. Completed scientific
units remain immutable. No new cohort or subsequent phase is authorized here.

## Phase 10 Execution Record

Authorized 2026-10-26: implement and execute Phase 10 under Sol, with one
20-hour active-compute allocation and the shared 500 GB Plan 15 storage cap.
New code is in mnist_experiment/fisher_memory_registration/. Historical
scientific modules are unchanged. The main notebook now has a pending Phase 10
section; its 65 pre-existing cells were verified unchanged.

**Production started 2026-10-26 at about 18:54 UTC** in tmux session
plan15-phase10, root cache/mnist_experiment/plan15/registration_v1 and log
cache/mnist_experiment/plan15/registration_v1.log. Initial PID 14908 was verified
in nvidia-smi. The scheduler passed pretraining/calibration and is running the
policy screen. Both FD optimizer audits passed seven checkpoints each, with no
objective improvement under the doubled solver budget.
Use current process/lease state rather than treating this historical PID as live.

**Latest restart verified 2026-10-26 at about 22:33 PDT.** The old tmux server
and GPU process were absent. The normal resume command restored the vector
replica-2 EWC-only finalist from novel count 79, preserving 138 completed
trajectories and approximately 5.46 charged active hours. That trajectory then
completed and the next finalist started (139 complete). The tmux session name
and root above are unchanged; the resumed PID was 4237, historical only.
The earlier 19:01 PDT restart likewise recovered the 66-trajectory checkpoint.
No scientific settings or source identity changed. Offline time was not charged,
and neither allocation renewed. Inspect live status before restarting.
At that restart, initial-grid and bounded-refinement work was sealed; finalist
development, fresh confirmation and analysis remained. Fourteen reporting checks
passed then, including clear pending cost labels; that reporting-only clarification
loaded on restart. The completed status above supersedes these historical notes.

Formal CPU acceptance passed **121 checks**, including exact resume, hard kills,
pretraining and retirement transactions for both tasks. Full CPU and CUDA
scheduler smokes passed; an accepted-boundary CUDA interruption/resume matched
all 30 common uninterrupted trajectory parameter vectors and retirement ledgers
exactly. The acceptance seal is registration_acceptance_v1, scientific hash
369c81456c0747d160163bd7c60e9a5418128072d312664db4586e7fa9680306.
Acceptance/superseded work charged about 803 seconds to this allocation before
production, including a conservative 600-second preliminary-test allowance.
Tiny smoke outputs are acceptance evidence, not scientific findings.
PyTorch 2.10.0+cu128 accesses the RTX 4070 through ~/.venv/bin/python outside
the restricted sandbox; sandboxed CUDA initialization fails despite nvidia-smi.

Acceptance evidence uses:

- cache/mnist_experiment/plan15/registration_checks_v1/cpu.xml
- cache/mnist_experiment/plan15/registration_smoke_cpu_v1
- cache/mnist_experiment/plan15/registration_smoke_cuda_v1

Earlier registration_dev_smoke_v1/v2/v3 roots preserve pre-acceptance attempts
and must be charged as superseded work. Source hashes are not frozen until
acceptance; never bypass a production source mismatch to resume older code.
The Python scheduler advances 10B--10E without an agent connection. Reporting
source may change independently; scientific source changes require a new source
identity and validated lineage. GPU operations and process inspection require
outside-sandbox tool execution in this environment. A sandboxed status command
can falsely report process_alive=false because /proc access is restricted.

Run from the repository root. For an inactive/stopped job:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory_registration.run \
  --root cache/mnist_experiment/plan15/registration_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb \
  --resume --clear-stop
```

For status or a graceful stop, use the same module/root with --stage status or
--stage stop. Do not start a second writer while the tmux job is running.
After a reboot, explicitly resume; completed scientific units remain immutable.
Restart commands have been tested on the matching-source CPU/CUDA smoke roots.

Source/rendering design: two 28x28 views of a source, fixed 6D affine law in
all roles, query-only continuously changing blur, scalar companion on identical
inputs; covariance 4I, both-block gradients, retirement once. Counts 476/636;
capability fallback 792/1000. Keep complete paired trajectories and all labels,
report physical-pixel errors plus matched-quality savings/cost/retention.

### Planned FD Optimizer Audit Supplement

At about 19:05 UTC, the scheduler stopped cleanly during the first grid FD
trajectory (novel count 29) to complete the planned representative FD optimizer
audits before policy selection. Pretraining passed at 15 epochs: vector risk
ratio 0.2514 and scalar 0.1996 versus zero displacement. Blur 1.8 was retained.
Rank 8 compression passed, maximum checked relative Frobenius error 0.00832;
diagnostic STM effective ranks were 6.14 vector and 1.26 scalar.

The versioned operator mnist_experiment/plan15_registration_audit.py performs
the two planned FD grid recipes at c=sqrt(10), strength=lambda0, with doubled
solver audits; it then execs the original scheduler. It leaves the frozen
scientific package/hash unchanged and records its own hash, trajectory seals,
and measurements in decision-fd-solver-audit-vector/scalar. Audited recipes
are reused by the grid. A focused CPU integration check verifies that these
read-only optimizer audits leave accepted parameters/metrics unchanged.

If interrupted before both audit decisions exist and pass, resume via:

```bash
~/.venv/bin/python -u -m mnist_experiment.plan15_registration_audit \
  --root cache/mnist_experiment/plan15/registration_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb \
  --resume-scheduler
```

Once both decisions pass, the normal scheduler --resume --clear-stop command
above is sufficient. The tmux session remains plan15-phase10; inspect live
processes instead of relying on its earlier PID. This supplement changes
measurement/scheduling, not likelihood, optimization settings or accepted
learning updates. All time is charged to the same 20-hour allocation.

The report now supports rolling complete-pair descriptive confirmation curves,
secondary common-coordinate contrasts and linked sealed geometry. Thirteen
reporting checks passed, in addition to two audit-observer checks. A subsequent
clean resume loaded reporting-only updates; the accepted scientific hash did
not change. Its displayed workload extrapolation is conservative and can exceed
remaining allocation when early weak-ridge settings dominate timings. The
20-hour cap and confirmation reserve are still enforced by the scheduler.

## Phase 9 Complete

Phase 9 finished on 2026-10-26: **340 complete trajectories** (304 development,
36 fresh confirmation), **446 completed immutable units**, and 2,000 paired
bootstrap draws. There is no pending Phase 9 GPU work. The tmux pane exited
successfully (pane_dead=1, exit status 0). Aggregate charged work, including
acceptance and final publication, is approximately **6.96/20 active hours**;
the node's offline interval did not count as compute. New artifacts occupy
approximately **0.83 GB**, with approximately 5.85 GB in the shared 500 GB
budget including earlier Plan 15 data and inputs.

The selected gamma-4.5 confirmation did **not** establish an FD acquisition
advantage: replay-minus-FD budget-mean MSE is **0.05568 square degrees**, paired
95% CI **[-0.15151, 0.29494]**. Full-budget retention is equivalent within the
registered tolerance: FD-minus-replay H is -0.00580 degrees, CI
[-0.01485, 0.00139], and both satisfy the deterioration ceiling.
FD learning cost is 47.60 versus replay's 28.89 seconds, **1.65 times at the
same 100-novel-image budget**, not at matched quality.

**Sample savings and S are unresolved, not zero.** FD's mean endpoint MSE
(402.9575) misses replay's target (402.4652); 1,987/2,000 bootstrap inversions
are unresolved, so no ordinary inverse interval is reported. Replay improves
frozen MSE by only 1.96%, below the 5% practical-target preference. Matched-
budget retention cannot be certified without an evaluable inverse. This is
a negative result for these screened contexts, not a universal rejection of
FD or STM. The concise notebook conclusion reports these distinctions,
secondary applied metrics, label accounting, and learning costs.

Scientific source remains the hash below. All 446 unit seals were verified,
with zero incomplete units; final reporting changes preserve the same seals,
all 53 historical notebook cells, and the frozen Phase 8 notebook. The original
93 CPU/CUDA acceptance checks remain valid; all five final reporting checks
pass. Final changes were reporting-only: clearer practical-target/inverse
warnings, full-versus-matched retention, accounting, and figure labels.

### Execution And Recovery Record

The user authorized all Phase 9 work on 2026-10-25 under Sol, including
implementation and the new aggregate 20-active-hour allocation. The package
is mnist_experiment/fisher_memory_shift/. Production must use CUDA from
the user's ~/.venv; GPU access works outside the restricted sandbox.

Current scientific source:
7a0406428fd36c964abfab0dd61a6b99f0447a3aac4f8e824ccb64d130ac8ee8.
All Phase 8.1 scientific source hashes remain unchanged. The main notebook has
the completed Phase 9 section and top progress entry; all 53 older cells were
verified unchanged (canonical history SHA256
9084283e64df73af20ea4c7f7e6a6daba19516755baf70913d924fd1ce890467).

Implementation includes coherent gamma/blur plans, shared staged nominal
pretraining with Adam-state continuation, old regression learning mathematics,
immutable context/refinement/selection decisions, accepted-update checkpoints,
source/config/data identities, blockwise bootstrap recovery, and a standalone
unattended scheduler. Two bounded parallel coding tasks implemented data/config
and analysis/selection; integration and execution were retained here.

Acceptance now passes **93 checks** (92 CPU, one CUDA), including hard-kill
recovery, atomic boundary stops, source pairing, notebook preservation, exact
Adam-state continuation, and the no-band dominance regression. Both matching
current-source full scheduler smokes are complete. Evidence is sealed under
continuous_shift_acceptance_v2. The older
continuous_shift_acceptance_v1 and smoke roots are preserved and count toward
the same phase budget; they validate the earlier pre-production source only.
The uncompleted dev0 CPU smoke is also preserved, not efficacy evidence.

Production started at approximately **2026-10-26 03:09 UTC** in detached tmux
session **plan15-phase9**. Root: cache/mnist_experiment/plan15/continuous_shift_v1;
log: cache/mnist_experiment/plan15/continuous_shift_v1.log.
The scheduler runs all stages autonomously under the shared 20-hour ledger and
updates the main results notebook. Do not run concurrent writers against the
shared Plan 15 storage budget; its lock deliberately serializes them.

Broad-angle initialization met the readiness target at 15 epochs: validation
MSE is 68.41% lower than zero prediction across the two development replicas.
The planned readiness extensions selected gamma 4.5 and blur 1.8 pixels as the
stronger endpoints, alongside gamma 1.8 and blur 0.6. The strongest blur raises
frozen RMSE by about 12%; gamma shifts remain mild. Those limitations are
visible in the notebook; they do not cancel the full screen. All four reference
and FD gradient calibrations passed; the full-budget 9C screen subsequently completed.
The first-context enlarged solver audits had zero measured objective gap.
A brief reporting-only checkpoint/restart loaded clearer partial-pair labels
and supplemental heatmaps. All 41 preceding completed seals were checked
unchanged; scientific source, settings and the cumulative budget did not change.
The status JSON records counts and stage, but its "running" value is not a
liveness check: after a reboot or hard stop it may be stale. Check the UTC
heartbeat and the live tmux pane before reporting that computation is active.

**Restart verified 2026-10-26, approximately 13:42 UTC.** The node had been
repurposed, and no tmux session survived. Before resumption, all 334 completed
units were validated read-only, including 272 complete trajectories; the
scientific source still matched the frozen identity. The interrupted stronger-
blur replay trajectory had valid current/previous checkpoints at novel counts
83/82. User authorization restored GPU availability. The same root resumed in
plan15-phase9 without clearing a STOP request (none existed), advanced to 100,
and completed trajectory 273. The aggregate ledger remained about 5.8/20 active
hours; disconnected downtime did not become experimental compute. The complete
216-trajectory initial grid and completed refinements are reused, not rerun.
At that restart, remaining work was bounded refinement, any eligible combination,
finalist development, frozen fresh confirmation and analysis. Reporting-only heatmap
layout changes passed the three reporting tests and do not change source
identity or scientific settings.

**Confirmation frozen at approximately 14:14 UTC.** All 304 development
trajectories are complete (216 initial + 64 refinement + 24 finalist).
The optional combination was not admitted: both families did not provide a
feasible positive acquisition lead within the point retention tolerances. Four-replica
finalist selection chose gamma 4.5. The immutable confirmation manifest fixes
12 fresh paired replicas through 100 novel images, all at ridge 10; FD and
replay have capacity 256/replay count 16, and FD strength is the historical
2.6356432540763373. EWC-only remains capacity/replay zero. Development gain
was only 0.2875 square degrees; the replay target improved frozen MSE by 2.82%,
below the 5% practical-target preference, and the development inverse was
unresolved. These warnings do not cancel the planned fresh cohort.
The next work was 9D confirmation, followed by 9E analysis/closeout. A fourth
reporting test covers explicit disclosure of small target improvements; all
four reporting tests passed then. That reporting-only clarification was loaded
in final publication, without changing scientific identity; five now pass.

To inspect or request a graceful halt:

    ~/.venv/bin/python -m mnist_experiment.fisher_memory_shift.run \
      --root cache/mnist_experiment/plan15/continuous_shift_v1 --stage status
    ~/.venv/bin/python -m mnist_experiment.fisher_memory_shift.run \
      --root cache/mnist_experiment/plan15/continuous_shift_v1 --stage stop

For live-process verification, also inspect:

    tmux display-message -p -t plan15-phase9 \
      '#{pane_pid} #{pane_dead} #{pane_dead_status} #{pane_current_command}'

A missing session or pane_dead=1 is not running, regardless of saved status.

The tested interrupted-run command below is retained for recovery context.
No resume is needed now that the phase is complete. After an interruption,
resume the SAME root and acceptance identity:

    ~/.venv/bin/python -u -m mnist_experiment.fisher_memory_shift.run \
      --root cache/mnist_experiment/plan15/continuous_shift_v1 \
      --acceptance-root cache/mnist_experiment/plan15/continuous_shift_acceptance_v2 \
      --notebook mnist_experiment/fisher_memory_results.ipynb \
      --allocation-hours 20 --stage all --resume

Add --clear-stop only when deliberately clearing a STOP request. Resume does
not grant another 20 hours. The initial accepted/test work was charged once;
all acceptance versions share the allocation. For an unattended restart, run
the command in a new/reused detached tmux session and preserve its log.
GPU commands invoked by Codex require access outside the restricted sandbox.

Current runtime decisions and progress belong in the new root, not an old
Phase 8 resume command. Read Phase 9 in the plan for bounded exploration,
retention qualification and score guards. Completed numerical artifacts never
change; meaningful repairs need new identities and documented lineage.

## Phase 9 Draft: Continuous Rendering Shifts

Historical design handoff, superseded by the execution authorization above:
the earlier request authorized plan writing only. Phase 9 is in
[plan15.md](plan15.md#phase-9-continuous-rendering-shifts-and-useful-sample-savings-complete).

The responsibility split is deliberate: Astra authored the design; after the
user authorizes execution, Sol owns implementation, acceptance, calibration,
exploration, confirmation, repair, reporting, and closeout. No further
model-switch or intermediate review is required. Do not resume an old Phase 8
root to start this new data protocol.

Core design:

- Keep the 422-coordinate Gaussian angle-regression model and all ten digits.
  Pretrain the full Uniform[-60,60]-degree task; novelty is now continuous
  gamma/blur rendering change, not extrapolation to an unseen angle range.
- Retain exact 100-novel-image drip trajectories, paired initialization/streams,
  fixed nominal/novel held-out panels, and independently named randomness.
- Hold replay capacity at 256. Calibrate both primary methods over replay
  counts four/sixteen and ridge; cross FD strength with ridge. Keep EWC-only
  and frozen initialization visible.
- Start with two gamma and two blur severities. Bound readiness repairs and
  refinement; at most one combined context is conditional on evidence and time.
  Select one context on development, then normally run 12 fresh paired replicas.
- Present S as fraction of novel samples saved divided by the measured compute
  multiplier at matched prediction quality. Specify inverse-curve uncertainty,
  full-budget and matched-budget retention, and unresolved scores explicitly.
- Keep progress near the notebook top and append a concise Phase 9 section
  after 8.1E. Detailed diagnostics belong in a linked supplement.

Forecast: 5.5--12.9 active compute hours, a proposed new aggregate 20-hour cap,
and approximately 3--6 hours of initial implementation work separately.
All process work, including tests/repairs/restarts, counts toward the cap.
The strict shared 500 GB artifact ceiling remains. Execution must implement
--resume, explicit --clear-stop, immutable decisions/units, accepted-update
checkpoints, and bounded stop-aware validation/reporting.

Proposed code/root identities are fisher_memory_shift/ and
cache/mnist_experiment/plan15/continuous_shift_v1; these do not exist merely
because the plan names them. Sol must publish tested commands after acceptance.
The plan supplies the complete schedule, selection, metric, budget and repair
rules. Preserve all historical evidence and code identities.

## Phase 8.1E Complete

The user authorized the artifact-only exploratory sample-equivalence follow-up
after discussing why Phase 7E's global straight-line fit would distort these
flattening regression curves. It completed October 25 local time (October 26
UTC). **No GPU training was run or remains required.** The original Phase 8.1
confirmation and all 318 source unit seals remain unchanged.

New root: `cache/mnist_experiment/plan15/retention_efficiency_v1`.
Protocol: `plan15-retention-sample-equivalence-v1`. Analysis source hash:
`463fa8b9a5923d14bcfaee624533363114e374fc3c4048956cc879eb318a7c8a`.
Code is in `plan15_retention_efficiency.py` and its reporting companion,
outside the original scientific-source fingerprint. The notebook has three
new cells immediately after Phase 8.1's original conclusion. Parent report
refreshes preserve the new section. The frozen Phase 8 archive is untouched.

The target is replay's **observed** mean new-domain MSE at 100 novel images,
1159.3112 square degrees. Adjacent-checkpoint interpolation places FD's terminal
crossing at **91.4087 images**: **8.5913 saved (8.59%)**, nominal paired-bootstrap
95% interval **[1.8083,21.1794]**. The rule retains every downward entry but
uses the last one that persists through the observed endpoint. This is not a
global-line fit, smoothing, extrapolation, individual first-passage time or a
guarantee of future observations saved. Background exposure changes with the
novel budget; calibration/pretraining labels are excluded.

All 2,000 bootstrap draws are evaluable; **959 (47.95%) have multiple entries**.
The point mean has one entry. Keep this sensitivity warning prominent rather
than treating a nominal inversion interval as newly confirmed sample savings.
Every draw recomputes both mean curves, the replay target, matching budget,
corresponding costs and old-domain endpoint errors using the same paired sample.
All results are retained in 20 immutable 100-draw blocks.

Equal-capability mean learning times are 38.4630 s for FD and 25.0247 s for
replay, ratio **1.5370 [1.3287,1.6402]**; the original equal-budget ratio remains
1.6540. Total arrivals are 983.9636 versus 1050.7500. Old-domain endpoint RMSEs
are 7.3221 versus 7.6818 degrees, difference -0.3596 [-0.8591,-0.0075]. These
are not H and do not extend the original retention-equivalence certificate.

Acceptance passes **39 tests**, including complete three-arm cohort/pairing,
invalid and multiple crossings, bootstrap target recomputation, immutable
block-level resume, stop files and notebook preservation. The follow-up
conservatively charges 60 seconds once for pre-run acceptance; production
analysis/reporting adds only seconds. It shares Phase 8.1's 20-hour ledger,
has a ten-active-minute own cap, and adds about 2.2 MB under the 500-GB ledger.
The original 4.93-hour execution record below is historical, before this addition.
Final closeout confirms all 50 pre-existing notebook cells unchanged, the frozen
archive checksum unchanged, all 318 original unit seals unchanged, and 23
completed follow-up units. Its final plot was visually inspected. Charged
follow-up time including report refresh is about 68 seconds; total Phase 8.1
allocation usage is approximately 4.95 active hours. Both invocations exited
successfully; no follow-up process is running.

To resume/revalidate/republish this completed analysis without training:

```bash
~/.venv/bin/python -u -m mnist_experiment.plan15_retention_efficiency \
  --source-root cache/mnist_experiment/plan15/regression_phase8_1_v1 \
  --root cache/mnist_experiment/plan15/retention_efficiency_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb --resume
```

Add `--clear-stop` only for an intentional STOP-file pause. Do not extend the
cohort, change target/crossing rules or overwrite existing blocks. A new
scientific interpretation requires a separately identified follow-up.

## Phase 8.1 Complete

**Completed 2026-10-25 at approximately 23:44 UTC; presentation closeout followed.**
No further training is required or running. The retained `plan15-phase81` tmux
pane exited with code zero. Artifacts are under
`cache/mnist_experiment/plan15/regression_phase8_1_v1`; the adjacent `.log` file
retains the execution history. The main notebook contains four figures, two
compact tables and a linked human-readable diagnostics supplement.

All **272 trajectories** completed: 144 initial-grid, 80 bounded-refinement,
12 additional finalist-development and 36 fresh-confirmation trajectories.
The 12-replica confirmation cohort was frozen before its first outcome, with
no subsequent retuning or cohort extension. There were zero solver failures;
30 enlarged-solve audit snapshots were recorded during the initial grid.

Primary replay-minus-FD acquisition gain is **18.8931 [15.6257, 22.3599] square
degrees**, nominal paired whole-replica bootstrap 95% interval, favorable in
12/12 replicas. The selected FD policy has **1.40% lower budget-averaged novel
MSE**. FD-minus-replay H is 0.04904 [-0.00253, 0.09847] degrees, supporting the
registered +/-0.20-degree equivalence criterion. All three methods meet the
common retention ceiling by their one-sided 95% upper bounds and satisfy the
measured resource limits. FD's ceiling support is narrow: upper H is 1.31994
versus delta 1.33424 degrees. These are conditional reused-benchmark results,
not an external deployment guarantee.

Selected policies: FD C=256,b=16,c=10,lambda=2.635643; replay/EWC
C=256,b=4,c=13.160740; EWC-only C=b=0,c=39.482220. FD and replay mean learning
times are 41.39 and 25.02 seconds per trajectory (1.65 times). This is a
separately calibrated policy comparison, not an isolated FD-penalty effect or
an equal-compute advantage. No actual-samples-saved estimate was made.
FD versus EWC-only has a positive secondary acquisition interval, but does
not establish retention equivalence; do not extend the primary matching claim.

Aggregate charged active compute is approximately **4.93 hours** out of 20,
including acceptance and report refreshes. New artifacts use about 0.50 GB;
shared budgeted storage is about 4.94 GB out of 500. The sealed closeout checks
317 preceding units; the resulting total is 318 complete units. Independent
post-run checks verified all 2,884 sealed file checksums and all 18 authoritative
sampling plans. Both scientific hashes and all 41 earlier/ten archived notebook
cells are unchanged. Final reporting/archive tests passed 19 checks, and all
four final figures were visually inspected.

The content-addressed supplement is linked from the conclusion. It includes
35 same-(C,b,c) development contrasts, the full 236 development trajectories
in an expandable table, calibration costs, retired-training-cohort diagnostics
and metadata interpretation notes. It adds no confirmation tests or selections.
For artifact-only republishing after future presentation changes, use the
normal command below with `--stage analyze --resume`. Do not restart or extend
confirmation automatically; any further scientific work needs a new design.

Final acceptance: **319 CPU and ten CUDA checks passed**, including abrupt
process kill, accepted-update/eviction/line-search interruption, stop-file
handling and immutable completed replay. Acceptance plus earlier test work is
charged conservatively at about five active minutes. Scientific source hash:
`0cdc9f1dba59112c8b520cdab0a9849dd3df5a46983108b6614fb1bc7898dad8`.
The original Phase 8 scientific hash remains unchanged. The frozen initial
notebook SHA-256 is
`12cffa3c25a745dbfe55f40dc93c1f1b8f2fdb26166e555f1151758c225d823d`.

Restart on 2026-10-25 at approximately 21:20 UTC: the prior process and tmux
server were absent. Resumed the same root with `--resume`, recovering the
interrupted initial-grid trajectory at novel count 96. It completed normally;
the previous 132 trajectories were reused without mutation. No STOP marker was
present and no scientific setting changed. Paused time is not charged as active
compute; the interrupted invocation's durable timing lease is retained.
The old `status.json` reason string can survive a restart; use the current
status, heartbeat and live process rather than that historical reason alone.

Raw-metadata interpretation notes from source review (no numerical change):

- The generic interval helper's `favorable_replicas` field counts values above
  zero. This is favorable for the signed acquisition-gain contrasts, but is
  merely a positive-value count for raw errors, H, and H differences. Do not
  interpret it as favorable for those quantities. It does not enter any interval,
  ceiling, equivalence or policy-selection calculation, and is not shown in the
  notebook's conclusion table.
- The confirmation manifest's auxiliary `seed_components` inventory includes
  `stream_ids` and `stream_novel` labels that are not driving RNG names.
  Authoritative sampling provenance is each sealed `assets-*/plan.json`:
  `seeds.stream_sources`, `seeds.stream_digits`, `seeds.stream_domains`, the
  separately named angle seeds, actual sampled source IDs and applied angles.
  All arms consume those same saved plans. Do not reconstruct a stream from
  the auxiliary aliases. Preserve the frozen manifest and scientific source;
  these are inventory-label caveats, not changes to sampling or pairing.

Two initial reporting-only improvements were tested separately (ten fast checks):
comparison labels use "vs" to avoid suggesting the wrong subtraction sign,
and confirmation captions state total arrivals alongside novel-image counts.
The final artifact-only refresh has loaded these changes and the additional
conclusion/supplement presentation. It did not retrain or mutate sealed units.

During the first replica, a reporting-only checkpoint pause/resume loaded a
full-range plus target-neighborhood plot. A weak-ridge outlier otherwise made
the matching band unreadable. All observations remain visible; no selection,
metric, setting or scientific source changed. Nine focused report/design tests
passed, the updated PNG was visually inspected, and the scientific hash was
rechecked before immediately resuming the same tmux run.

The user authorized all Phase 8.1 work on 2026-10-25 after switching to Sol.
Implementation is in the separate `fisher_memory_retention` package, reusing
the repaired Phase 8 learning engine unchanged. CPU/CUDA acceptance precedes
production under `cache/mnist_experiment/plan15/regression_phase8_1_v1`.
The frozen Phase 8 notebook and all original computational units stay unchanged.

The new scheduler runs all stages unattended, supports accepted-update recovery,
and freezes every development expansion and the confirmation ledger. On a
normal restart, use the command below; add `--clear-stop` only when intentionally
resuming a STOP/SIGTERM pause. Do not resume the old regression Phase 8 root.

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory_retention.run \
  --root cache/mnist_experiment/plan15/regression_phase8_1_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb \
  --allocation-hours 20 --resume
```

Production requires a matching immutable acceptance record at
`cache/mnist_experiment/plan15/retention_acceptance_v1`. The new root records
the full scientific source snapshot in addition to hashes/configuration, so
the experimental method remains recoverable even before a git commit.
The 20-hour allocation is cumulative across Phase 8.1 production/acceptance
roots and invocations, not additional time on each resume. Prior Phase 8
compute is not charged again; all data still share the 500 GB storage cap.

## Phase 8.1 Draft And Notebook Archive

Historical drafting record: on 2026-10-25 the user requested the retention-matched follow-up **draft and
notebook reorganization only**. No Phase 8.1 experimental implementation,
integration smoke, or training is authorized yet. Read the new Phase 8.1 in
[plan15.md](plan15.md) before further work. Its proposed ceiling is the earlier
FD mean, 1.3342351463 degrees, with a separately proposed 0.20-degree matching
tolerance; both remain reviewable design choices, not new results.

The main [results notebook](fisher_memory_results.ipynb) now retains a short
Phase 8 historical conclusion and link, followed by the pending Phase 8.1
section; progress is near the top. All ten original Phase 8 cells, including
their static figures, corrections and conclusion, are preserved unchanged in
[the frozen initial report](fisher_memory_phase8_initial_results.ipynb).
All 41 pre-Phase-8 cells remain unchanged in the main notebook. Snapshot
metadata records original notebook/cell digests, run identity, scientific
source hash and source commit `4337a6f2b095f327b00acab9dac8bdd171123ef0`.

This is a reporting-only migration. The Phase 8 scientific hash below and all
cache artifacts are unchanged. A publisher guard rejects old Phase 8 refreshes
targeting either the frozen snapshot or the reorganized main notebook. Historical
refreshes must use a distinct output path; do not remove the guard to revive the
old main-notebook layout. The archive was copied, not recomputed from checkpoints.

Proposed Phase 8.1 execution is 5--11 active compute hours, capped at 20 only if
authorized. Each method gets independent full-budget ridge calibration, FD
gets crossed strength/ridge calibration, and both replay arms investigate
the same two allocations. Preserve the old outcomes as context, never pool
them into new confirmation or choose replicas by observed retention. No online
retention controller is proposed. Resume engineering starts only after the
user's next instruction, not by resuming the completed Phase 8 root.

## Phase 8 Complete

Phases 8A--8E completed on 2026-10-24. No Phase 8 experiment or test process
remains running. Valid evidence is under
`cache/mnist_experiment/plan15/regression_phase8_v2`; v1 remains superseded.
The completed Phase 8 presentation is now preserved in
[fisher_memory_phase8_initial_results.ipynb](fisher_memory_phase8_initial_results.ipynb),
with five main figures and two compact tables; the main notebook links to it.

Twelve fresh paired replicas cover FD (C=256,b=16), replay/EWC (256,4), and
EWC-only (0,0), all to 100 novel observations. Mean replay-minus-FD acquisition
MSE is -21.91, nominal paired 95% interval [-28.44,-14.49]; only 1/12 pairs favors
FD. The FD(75)-replay(100) contrast is +79.80 [59.75,101.02] square degrees:
the smaller-budget advantage is not supported. FD improves on freezing by
293.92 [267.64,316.07] square degrees, but exceeds the one-degree base-retention
guard: 1.334 [1.162,1.485] degrees. Replay also exceeds it, at 1.591
[1.408,1.769]. Thus this regression study does not confirm an FD sample-efficiency
advantage or a policy meeting all practical constraints. Retention differences
between policies are descriptive; no extra paired significance claim was added.

Selected-policy FD/replay learning-cost ratio is 1.73 (47.77/27.61 seconds at
100 novel / 1048.5 mean total observations). Both satisfy the measured resource
envelope. No linear inversion was registered; use raw curves and the observed
budget contrast. No cohort resizing, optional extension or new target followed
confirmation inspection.

Aggregate active compute is approximately **2.93 hours** out of 20, including
8A, superseded v1 work, repair and both closeout validations. V2 data occupy
about 0.235 GB; shared budgeted storage is about 4.45 GB out of 500.
`closeout-qa-v1` validates 100 trajectory units (20 calibration, including
prefixes/continuations; 44 allocation; 36 confirmation), zero failed solves,
and all 41 preserved historical cells. Maximum final archive trace relative
error is 3.075e-16, diagonal error 1.668e-20. `closeout-qa-v2` rechecks all prior
scientific seals and passes 305 CPU plus nine CUDA recovery tests after a
reporting-only fix that draws figure layout before export, preserving titles
and axis labels. The scientific source hash remains unchanged below.

The artifact-only supplementary entry point now publishes separately sealed
`diagnostics-v2` / `reports/phase8-diagnostics-v2.html`, linked in the notebook.
It covers renderer examples, development resource frontiers, per-digit errors,
and the prospective fit diagnostic. Earlier `diagnostics-v1` HTML remains
unchanged. To refresh/validate the current supplement without learning:

```bash
~/.venv/bin/python -m mnist_experiment.plan15_regression_diagnostics \
  --root cache/mnist_experiment/plan15/regression_phase8_v2 \
  --notebook cache/mnist_experiment/plan15/regression_phase8_v2/reports/phase8-historical-review.ipynb --resume
```

## Phase 8 Numerical Repair

**v1 is superseded; validated v2 is complete.** V1 confirmation replica 3
revealed exploding Fisher archives and nonfinite solver trials in all three
conditions. The legacy Lanczos routine loses Krylov-basis orthogonality on
nearly rank-one matrices: a PSD target with trace 1 was represented with trace
4, and repeated constant-target compression inflated trace above 63,000 in ten
updates. Several initial assets were already inflated; confirmation replica
3 represented trace 3.0534 for a source trace of 1.5571. This is a numerical
defect, not evidence against learning or FD.

All v1 development and partial confirmation artifacts remain immutable under
`regression_phase8_v1`; its STOP marker remains set. The sealed
`compression-failure-evidence` unit retains the reproduction, asset trace audit,
and prior report snapshot. The notebook now discloses the failure and withholds
v1 inference. Impact on older experiments using this shared legacy routine has
not been assessed; their code/results have not been changed.

The regression-specific repair uses exact symmetric eigh on the already-dense
287-coordinate LTM block, retaining eight modes plus the residual diagonal.
It preserves the source diagonal/trace and the intended likelihood Fisher,
eviction EMA and ridge formula. A failed numerical solve now stops before an
arrival can be committed. Protocol v2 changes component-seed namespaces, so
the entire calibration/allocation/confirmation study is restarted with fresh
seeds under `regression_phase8_v2`, without pooling v1 evidence. The shared
20-hour compute limit still charges all v1, repair and acceptance costs.

CPU validation passes 298 checks, including nearly rank-one matrices, repeated
convex-update trace bounds and checkpoint recovery. CUDA recovery passes nine
checks; both tiny full-stage CPU/CUDA smokes pass. Acceptance is sealed under
`regression_repair_acceptance_v2` (307 checks). The exact affected saved model
and its 1,024 inputs reproduce raw trace 1.5571 versus legacy 3.0534; repaired
trace is exactly 1.5571, with zero diagonal error. That reproduction is sealed
under `regression_repair_target_audit_v2`. The new scientific source hash is
`cdc2da2dafb1d9868926c96391165b187074c71697a96c7633dc359c62d728cf`.
Aggregate expenditure was about 1.34 active hours at the v2 restart. Do not
restart v1 with the changed scientific source. The repair-time forecast was
2.5--4 total active hours; the final actual is recorded above.

V2 calibration completed 20 trajectories. It selects c=10,
lambda_F=263.5643254076, h=0.01, with measured FD/replay cost ratio 1.7705
and learning-time envelope 0.05852044 seconds per arrival. All full-budget
solver audits pass; reference retention still misses one degree. The selected
FD strength is interior, so no edge extension ran. The full 34-trajectory v2
allocation map and ten-trajectory refinement are complete (44 trajectories).
Final choices remain FD at (C,b)=(256,16) and replay at (256,4). Development
base RMSE increases are 0.895 and 1.014 degrees, respectively: FD passes the
mean guard; replay narrowly misses and carries its warning. Their new-domain
A(N) means are 1336.79 and 1325.88 square degrees. The sealed development
visual review rejects full-range linear inversion because of systematic early
and late plateaus (residual RMSE 44--49 square degrees).

The fresh 12-replica/36-trajectory confirmation cohort is sealed, without
outcome-based resizing. FD(75) versus replay(100) is the unsupported-development
fallback; the fixed development target is 1160.7384 square degrees. Confirmation,
analysis, supplementary diagnostics and closeout checks are complete. Scientific
hash remains unchanged; no v1 evidence enters v2 intervals.

## Superseded v1 Execution Notes

**Historical v1 status before supersession, not current instructions.** Its
8C was complete and 8D had begun. The full map has 34 trajectories and the
refinement adds six (four development replicas at the selected cell). Both
policies select C=256, b=16. Mean development acquisition MSE is 1295.08 for
replay and 1296.96 for FD; base RMSE increases are 1.371 and 1.311 degrees,
respectively. Neither meets the one-degree guard. These remain development
results, not inference. The 12-replica/36-trajectory confirmation cohort is
sealed, with no outcome-based resizing allowed. Its sample contrast is
FD(75) versus replay(100); 75 is the prespecified unsupported-development
fallback. The fixed development target is 1087.8141 square degrees.

`allocation-fit-review-evidence` and `allocation-visual-review` were sealed
before confirmation registration. The full-range linear fit was rejected:
it misses the late plateau, making inverse target crossings misleading.
The cohort manifest records this prospective decision; use raw curves and
the directly observed budget contrast. After 8D, run `--stage analyze`, then
the supplementary diagnostics entry point. No scientific code or settings
were changed during production.

The user authorized 8B--8E on 2026-10-24. Calibration completed in
`cache/mnist_experiment/plan15/regression_phase8_v1`, using the default
regression configuration and the shared 20-hour allowance (including 8A).
All 22 calibration trajectories and numerical/solver decisions are sealed.
Selected common ridge multiplier: 10; FD strength: 3.1984308793; FD step: 0.01.
Measured FD/replay learning-time ratio: 1.5775. Full-budget reference retention
missed the one-degree guard, so the least-violating valid FD strength was
carried forward with a warning. All selected full-budget solver audits passed.
Aggregate charged time at this transition is about 0.37 hours. The full
eight-cell allocation grid is affordable and is running next in paired batches.
Phase 8C is being inspected in batches of 17 completed trajectories (one
replica across both methods and all eight cells, plus EWC-only), followed by
the planned affordable refinement. Confirmation has not started. Reporting-only
changes make the calibration warning explicit and align heatmap cost flags with
total work / total arrivals. The scientific source hash remains
`bd8b385d05e60efa77600ccb26df34a11161db97a9485e0b6cc3704476b66dde`.
The reporting/supplement tests pass (52 checks). The supplementary entry point
`mnist_experiment.plan15_regression_diagnostics` should run after 8E to publish
renderer examples, development risk/cost frontiers, and per-digit errors.
Measured calibration and first-map costs suggest roughly 1.5--2.5 total active
hours for the planned study, replacing the initial 14--20-hour forecast. The
20-hour hard ceiling remains unchanged; fresh cohort size will still be frozen
from timing before confirmation outcomes are read.
The results notebook updates while work runs. Check the root's `status.json`
and the read-only control status before resuming; never start a second writer.
Inspect the allocation heatmaps before sealing fresh confirmation. All earlier acceptance evidence remains
preserved. This execution note supersedes the pending status in the 8A handoff
below.

## Phase 8A: Regression Implementation

Historical 8A handoff: Phase 8A was implemented before production authorization.
The user's subsequent instruction authorized 8B--8E; see current status above.
The new package is `mnist_experiment.fisher_memory_regression`; read its
`AGENTS.md` and Plan 15's Phase 8 before proceeding. Earlier categorical
implementations and completed scientific artifacts have not been changed.

### Start And Resume

From the repository directory, using the user's venv:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory_regression \
  --root cache/mnist_experiment/plan15/regression_phase8_v2 \
  --notebook mnist_experiment/fisher_memory_results.ipynb \
  --allocation-hours 20 --resume
```

This is the canonical resume command; check current status above before use.
`all` is the default;
explicit stages are `calibrate`, `allocate`, `confirm`, and `analyze`. One root
contains separately sealed stage decisions, paired assets, immutable prefixes,
continuations and complete trajectories. No assistant connection is required.

Repeat the same command after interruption. After a deliberate STOP add
`--clear-stop`; never delete checkpoints or change the root to bypass a stop or
budget. Use `--max-wall-seconds` or `--max-units` for smaller batches. Exit code
2 means a deliberate/resource pause, not completion. A changed scientific
source/configuration requires a new versioned root, not a forced resume.

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.control stop \
  --root cache/mnist_experiment/plan15/regression_phase8_v2
```

Replace `stop` with `status` for a read-only operational check. `status.json`
and `report_snapshot.json` also provide small progress artifacts. CUDA must be
available; production never silently falls back to CPU. In a restricted agent
sandbox, GPU work required an approved unsandboxed Python invocation; the
installed driver and PyTorch CUDA work normally outside that restriction.

### Scientific And Operational Notes

- Gaussian regression from initialization onward: 45-degree fixed likelihood
  scale, 287 LTM + 135 STM parameters, exact conditional-target Fisher $JJ^T$.
  No categorical pretraining or categorical Fisher is used.
- Near-upright training is Uniform[-10,10]; novel training is the signed,
  upper-truncated lognormal in the plan. Unique novel counts and evaluation
  checkpoints are exact; short split segments retain the nominal-four archive
  clock. All evictees, not only sampled replay observations, are consolidated.
- Calibration checks FD values and full gradients in both float64 and the
  actual production dtype. Failed diagnostics are sealed before an error.
  Common solver ceilings can double twice; each attempt has independent
  decision identities and reuses the same streams/assets. Selected full-budget
  policies are re-audited, including late snapshots. Objective gaps exclude the
  Gaussian additive constant; residuals and predictive changes are recorded.
- A failed numerical gate is not a negative scientific result. Sol may repair
  it, version affected evidence, and continue under the plan's autonomy rules.
  Weak initialization and failed retention guards remain visible; they are not
  efficacy-based early-stop rules. Smoke-only solver-gate bypasses are explicit
  and never apply to production.
- Inspect the allocation heatmaps while they accumulate. The deterministic
  near-tie rule selects independently for FD and replay. Confirmation count is
  chosen from timing before its outcomes; the target, budgets, recipes, replica
  list and bootstrap settings are sealed. The initial forecast must be replaced
  by production pilot measurements, not extrapolated from tiny smokes.
- Optional linear fits default to **unregistered**. To register one, stop after
  development trajectories but before cohort sealing and supply `--fit-review`
  on the allocation resume. Its JSON needs `partition: "development"`, the
  canonical JSON hash (`config.content_hash`) of the development selection as
  `selection_hash`, an `approved`
  boolean, `reviewer`, and `reason`. The fit uses all events 1--100. A later
  review cannot change a sealed confirmation manifest. No review is required
  for the primary observed-budget comparison or sample-equivalence figure.
- Keep the final confirmation and diagnostic-only retirement-cohort views
  distinct. Retired exposure ages/counts differ across capacities. The common
  held-out base panel is the primary retention test.

The cumulative 20-hour cap includes the acceptance ledger and every production
or repair root. `--allocation-hours 20` is a total allowance, never another 20
hours per invocation. Hard kills conservatively charge the outstanding bounded
timing lease. All Plan 15 data share the 500 GB cap, with temporary-write and
filesystem reserves; no deletion is authorized. Pretraining recovery repeats
at most sixteen minibatches; online recovery repeats one uncommitted solve.

### Acceptance Evidence

The sealed acceptance record is
`cache/mnist_experiment/plan15/regression_8a_acceptance_v1/units/acceptance/result.json`.
Final scientific source hash starts `bd8b385d05e6`. Charged acceptance is
approximately **468 seconds (7.8 minutes)**, including superseded smokes and
conservative preliminary-test accounting. Phase 8 artifacts occupy about
**45.5 MB**; a tiny full scheduler smoke occupies about 6--7 MB. These do not
justify a production throughput extrapolation. The provisional 2--10 GB
production forecast must be replaced during 8B.

It records scientific source hashes, final test reports, current CPU/CUDA smoke
seals, superseded smoke costs and conservative preliminary-test accounting.
Tiny smokes have only three novel observations, one pretraining epoch and two
solver iterations: **they say nothing about effectiveness or production cost**.
They exercise all scheduler stages, intentional pause/resume, and final analysis.

Focused CPU coverage: 276 tests. Additional CUDA recovery coverage runs in both
float64 and float32. It checks pretraining, mid-fit, retirement, accepted-update,
STOP and SIGTERM recovery; prefix continuation; immutable completion; and
bitwise equality of scientific state to uninterrupted execution on each device.
Expected timing fields and allocator peaks are not equality targets. Unit tests
also cover crash-time charging, data separation, exact count alignment,
independent selection, whole-replica uncertainty and report preservation.

The results notebook appends a concise Phase 8 question/glossary, angle-density
figure, acceptance summary and explicit pending sections after Phase 7E, plus
progress near the top. All 41 historical cells are unchanged; eight cells were
added. Completed-work resume was tested with training and asset creation
disabled: all 331 sealed files per final CPU/CUDA smoke stayed byte-identical.
No experiment writer remains active. The production
report reads small artifacts, never trains or evaluates a model, and does not
perform bootstrap computation. Computational analysis is a separate sealed
stage. No production-quality regression estimate exists yet.

## Phase 7E Efficiency Conclusion (completed)

User approval and execution were on 2026-10-24. This was a CPU-only,
artifact-based follow-up, not more training or confirmation. The final focused
unit/recovery/report suite passes 111 tests. New implementation lives outside
the frozen scientific package in `plan15_efficiency.py` and
`plan15_efficiency_report.py`. The original scientific source hash below is
unchanged. The existing report received only a presentation-order change to
keep this conclusion after refreshed Phase 7 plots.

Analysis root: `cache/mnist_experiment/plan15/regularization_efficiency_v1`.
Its sealed plan retains the fixed configuration, named bootstrap seed,
2,000 whole-paired-replica index sets, original 7D inference, and completion
seals for all 201 development/confirmation units. Separate sealed units retain
the analysis/draws and visual fit review. Completed-work resume skips fitting
and bootstrap; unfinished bootstrap work can safely repeat. The notebook adds
exactly three cells at the end of Phase 7; all 38 earlier cells are unchanged.

**Conclusion:** the original budget-averaged gain remains **0.13017 nats**, with
its original paired 95% interval **[0.10456, 0.15706]** and 12/12 favorable fresh
replicas. Neither the primary interval nor p-value was recomputed. The new
independent-uniform-budget interpretation does not need monotonic learning.
Retention cost, 1.98-times online compute, frozen-reference gain and unbracketed
ridge-grid boundaries remain visible.

**Exploratory sample equivalence:** separate OLS fits on mean rotated-test NLL
use all events 1--100, excluding initialization. Replay's fitted endpoint
target is 1.3190592 nats. FD's model crossing is **43.2493** images, implying
**56.7507 images (56.7507%)** less than replay's 100 under that approximation.
The diagnostic captures the overall decline but has early overshoot and local
reversals; the sealed visual review explicitly qualifies the fit. This is not
mean first-passage time, guaranteed future saving, or STM-to-LTM identification.

**No ordinary 95% inversion interval:** bootstrap draws 848 and 1451 already
meet the fitted replay target at initialization; 2/2,000 are inversion-invalid
under the predeclared guard. All draws and raw crossings are retained. There
are zero out-of-range finite crossings. Do not quietly discard these two draws
or relax the guard to publish an interval. Any changed target/model is a new,
explicitly post hoc analysis, not a correction of the sealed analysis.

Tested completed-work resume, from the repository directory:

```bash
~/.venv/bin/python -u -m mnist_experiment.plan15_efficiency \
  --source-root cache/mnist_experiment/plan15/regularization_confirmation_v1 \
  --root cache/mnist_experiment/plan15/regularization_efficiency_v1 \
  --resume
```

An interrupted job uses the same command; add `--clear-stop` only when explicitly
resuming after a deliberate STOP. The shared budget lock, 500 GB study cap,
100 MB extra-analysis allowance and original 26-hour Phase 7 allocation remain
enforced. Existing fitting/draw/result units and scientific dependencies are
never rewritten. Notebook rendering only reads sealed results; it does not
train, download, evaluate models, or bootstrap. Phase 6 remains deferred.

Final checks: completed-work resume ran with fitting/bootstrap explicitly
disabled and passed; all analysis seals, all 201 original scientific seals,
and all 38 earlier notebook cells stayed unchanged. The diagnostic legend was
visually checked after a rendering-only clipping fix. Final focused suite:
111 passed, plus five targeted renderer checks. The four recorded test
invocations total 130.17 CPU process-seconds, charged once in a sealed
acceptance unit. Including analysis/reporting/recovery validation, 7E adds
approximately **2.4 CPU minutes**, **4.2 MB** of artifacts, and no GPU work;
aggregate Phase 7 expenditure is approximately **9.48 hours**. A separate
append-only clarification corrects a hash-label typo in the acceptance record;
the plan's dependency/source identities and all scientific results are unchanged.
All Phase 7 work is complete; no experiment process remains running.

## Phase 7 Execution (completed)

The new Phase 7 section of [Plan 15](plan15.md) specifies four sub-phases for
Sol to implement and execute: 7A frozen-objective solve audit,
7B paired ridge grid, 7C bounded FD retuning, and 7D fresh confirmation/reporting.
Phase 6 remains provisional and is neither a dependency nor an automatic next job.
The user authorized 7A--7D on 2026-10-23 with the planned initial 26-hour
aggregate allocation. Implementation and CPU/CUDA recovery acceptance passed:
389 tests with no skips/failures, plus a 41-test reporting follow-up. The sealed
manifest is `cache/mnist_experiment/plan15/phase7_acceptance_v1/units/phase7-acceptance-v1/result.json`.
Its 397.837 measured test-process seconds are charged to the shared allocation.
All four phases completed on CUDA on 2026-10-24, with the final cohort sealed
at 12:42 UTC. No Phase 7 work remains pending and no experiment process remains
running. Phase 6 has not started. Scientific source hash:
`26ca52703bfcc4d13f16d1d42b53851f2d31e443aef8bf035f06056f51a64e75`.
Do not change scientific source files in place while this root is active;
use the recorded correction/versioning protocol if a defect needs repair.
The final notebook includes a brief ridge/FD glossary paragraph and labels
the after-7A forecast as a historical estimate, not a live countdown. It also
flags selected ridge-grid boundaries as best tested, without a bracketed optimum.
The collapsed diagnostics also show recorded fitting, eviction, evaluation,
and total online seconds, plus a concise interpretation of the final intervals.
The final 43 focused reporting tests passed, and the scientific source hash
remains unchanged. These reporting-only clarifications have been published
through the tested completed-work resume path, with the final plots inspected.

**Final confirmation:** all 48 trajectories / 12 paired replicas are sealed.
At $c_R=c_F=1$ and $\lambda_F=160.89$, FD's primary acquisition gain over replay
was +0.13017 (paired bootstrap 95% interval [+0.10456, +0.15706], one-sided
sign-flip p=0.00024414), with 12/12 favorable replicas. Mean acquisition NLL
was 1.42354 for replay, 1.29336 for FD, and 1.41341 for the frozen reference.
FD's nominal acquisition gain over freezing was +0.12004 [+0.05893, +0.18800],
with 11/12 favorable replicas; replay's was -0.01013 [-0.07069, +0.05102].
Mixed-stream base retention harm, FD minus replay, was -0.03034
[-0.03716, -0.02438], but FD still worsened base acquisition NLL from
initialization by +0.06461 [+0.05881, +0.07094]. This is an adaptation/retention
trade-off, not forgetting-free learning. Endpoint rotated accuracy was 69.36%
for FD versus 66.46% for replay, descriptively.

Mixed online times averaged 220.85 seconds FD versus 111.42 seconds replay
(1.98 times). Equal-data evidence is not equal-compute superiority, a direct
samples-saved estimate, or a transfer-mechanism identification. Solver failures
and nonfinite trajectories were zero; 7,583 residual-based under-solved updates
remain disclosed. The full fixed-cohort analysis reproduced exactly on resume.
All 201 development/confirmation units passed full checksum validation, and
every completion seal stayed unchanged through standalone and all-stage resume.
Both store writer locks were released. Total active compute was approximately
9.44 hours including acceptance and reporting, with about 0.78 GB stored across
the two scientific roots and acceptance root. Stronger ridge made solves faster
than the inherited timing forecast; the unused allocation is not new scope.

**Development history:** all sixteen initial trajectories and 48 frozen
objective audits sealed successfully. Two audits crossed the predeclared
objective-gap threshold, below the twelve needed for a repair. The sealed
common policy remains 100 iterations / 150 closures; no repair cohort ran.
The 7B ledger covers 64 trajectories, reusing the sixteen compatible 7A units.
Measured online means were 238.05 seconds for replay/EWC and 428.87 seconds for
FD. The sealed timing forecast is 12.42--19.44 active hours remaining after 7A,
within the cumulative allowance. Development storage was about 395 MB at that
transition. The live status and later sealed decisions supersede this snapshot.

The full 64-trajectory ridge grid sealed on 2026-10-24. The immutable shortlist
selects $c_R=c_{F,0}=1$, with both candidates retention-feasible under the
predeclared common reference. Mixed-stream development acquisition NLL means
were 1.30839 for replay and 1.19176 for FD at the inherited coefficient 160.89;
the mean frozen reference was 1.24016. These four-replica development outcomes
are descriptive, not confirmation. The strongest tested ridge won for both
methods, so the optimum is not bracketed. Do not expand the ridge grid.
Phase 7C compared FD coefficients 53.63, 160.89 (reused), and 482.67 at
that one ridge. Their mixed acquisition means were 1.20957, 1.19176, and
1.20631, respectively. The middle coefficient won, so the sealed extension
decision requests no extra coefficient. Final recipes are replay at $c=1$
and FD at $(c,\lambda_F)=(1,160.89)$, with the unchanged 100/150 solver.
The selected ridges coincide, so the same-ridge contrast is the primary
contrast, not another independent finding or additional replay arm.

Development finished at 2026-10-24 09:49 UTC after about 6.54 aggregate active
hours including acceptance. All 80 unique development trajectories and 48
audits are sealed; development storage is about 478 MB. The immutable fresh
confirmation ledger has 12 replicas, both streams, and two recipes: 48
trajectories total. Phase 7D then started automatically. Based on the selected
recipes' measured development timings (about 108 seconds replay and 211 seconds
FD per mixed trajectory), the operational remaining forecast at transition
is approximately 2.5--3.5 active hours including initialization/reporting.
That was the transition forecast, not a change to the immutable historical
after-7A timing artifact. Confirmation took about 2.89 active hours. The final
results above supersede this historical transition snapshot.

The ordinary-policy forecast is 14--24 active compute hours (central estimate
18--20), excluding roughly 3--6 hours of implementation. A proposed 26-hour
aggregate allowance is authorized. One conditional common-ceiling repair
could raise the forecast to 20--40 hours and requires a measured reforecast
within whatever allocation is actually granted. Do not treat per-root ceilings
as independent copies of the overall allowance.

Use the detailed selection, audit, and versioning rules in the plan, not ad hoc
optimizer/regularizer tuning. In particular: reconstruct diagnostic objectives
with their original pre-arrival STM anchors; branches must not alter the live
learner; explicitly assign new seed families to development/confirmation test
partitions; preserve completed Phase 5 evidence; and add the common-ridge replay
control when the final selected FD and replay ridge settings differ. The proposed
Phase 7 entry point is `mnist_experiment.fisher_memory.regularization`.
It runs the complete study autonomously, persists every policy/selection
decision, and checkpoints accepted arrivals and pending objective audits.

The completed production command is retained below for reproducibility. There
is no need to launch it again. From the repository root using the user's venv:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.regularization \
  --root cache/mnist_experiment/plan15/regularization_development_v1 \
  --confirmation-root cache/mnist_experiment/plan15/regularization_confirmation_v1 \
  --allocation-hours 26
```

After any interruption, use the same command with `--resume`. Add `--clear-stop`
only after an intentional STOP request. The 26-hour limit is cumulative across
both roots, previous invocations, and any superseded Phase 7 repair roots;
it is not 26 additional hours per resume. The acceptance compute record also
counts toward that allowance. CUDA access requires ordinary host execution;
the filesystem sandbox hides GPU device files, while PyTorch in the user's
venv detects the RTX 4070 outside it. Do not substitute CPU for production.

The active root's `status.json` and the concise Phase 7 notebook section show
progress. Use `fisher_memory.control stop --root <active-root>` for a cooperative
pause; SIGTERM/Ctrl-C are also cooperative. A hard kill replays at most an
uncommitted arrival or unfinished short audit. Completed units stay immutable.

## Rare Rotated-Domain Confirmation (Phase 5C completed)

The user authorized Phase 5C as written on 2026-10-23, with twelve fresh
four-arm replicas and the predeclared ten-active-hour cumulative cap. The
implementation reuses the Phase 5B trajectory mathematics and fixes FD strength
at its sealed selection, 160.89. Fresh `drip_confirmation` seeds and the
confirmation half of the per-class test-image pool exclude development evidence.
Numerical learning sources are checked against development; only scheduler and
trajectory metadata handling changed. The main notebook receives a concise
live Phase 5C section after the earlier phases. Final inference requires all 48 sealed
trajectories; partial results remain descriptive.

Completed locally on CUDA at 2026-10-23 22:23 UTC after starting at 17:48 UTC.
All 48 trajectories and 62 artifact units are complete and passed full
content-hash validation. The sealed result reproduced exactly and the source
hash still matches. Active runtime was 4.59 hours; artifacts occupy about 284 MB.
No experiment process remains running. Before launch, 244 focused unit/CPU
pipeline tests passed (four expected CUDA-only skips), along with a CUDA
pipeline test. Both CPU and CUDA drills proved exact interrupted/uninterrupted
trajectory equality and unchanged completion seals on resume. Scientific source
hash: `2013757a582afdcceddb36979c77b7c150365117f34f6fde30b8366e53495adb`.
Preserve this source identity for resumption; use a new identity/root for any
scientific correction. Execution used the foreground local session, not `tmux`.

The primary FD gain was +0.4619 NLL-AUC, paired bootstrap 95% interval
[+0.3472, +0.5783], one-sided sign-flip p=0.00024414, with 12/12 favorable
replicas. Mean new-domain acquisition NLL was 2.1161 for replay/EWC versus 1.6542
for FD. Base-retention harm was -0.1566 [-0.1898, -0.1238]; the secondary
interaction was +0.1027 [+0.0512, +0.1585]. The mixed-stream runtime ratio was
1.77 (3.88 versus 6.88 online minutes). Final rotated accuracy averaged 61.98%
versus 65.62%, descriptively. There were zero recorded solver failures/nonfinite
trajectories and 8,318 disclosed under-solved updates. These results establish a
predictive advantage at the tested data/replay budget, not equal-compute
superiority or a memory-transfer mechanism.

The current notebook view is **Phase 5C: Sample and compute efficiency**, at
the bottom after Phase 5B. Results are ordered oldest to newest, and presentation
refreshes preserve that order. Its four-panel figure shows new-domain NLL versus
novel observations, the running acquisition score $A_m^R(n)$, the twelve paired
gains and mean/95% interval, and measured online compute. The score is explicitly
defined as average NLL after novel images 1 through $n$, excluding initialization.
The final means are 2.116 (replay/EWC) and 1.654 (FD), a descriptive 21.8% lower
acquisition loss, not an estimate of the percentage of samples saved. Dashed
lines show the frozen starting model (mean NLL 1.607); a relative efficiency
gain over replay need not be net improvement over initialization throughout
learning. Detailed tables are collapsed, accuracy curves
remain visible, and the older section is labeled as an earlier-study audit.
This presentation lives in reporting-only `report.py`; the frozen analysis and
completed scientific artifacts are unchanged. Refresh the view without training:

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.report \
  --root cache/mnist_experiment/plan15/drip_confirmation_v1 \
  --notebook mnist_experiment/fisher_memory_results.ipynb --phase5c
```

The standalone figure is `reports/phase5c_efficiency.png` inside that run root.

The notebook's **Evaluation protocol and comparable NLLs** subsection separates
training batches from evaluation. All twelve sealed data plans were checked:
each has 1,000 unique test IDs (100 per digit by construction), disjoint from
the Phase 5B tuning panels. The four arms share assets and initial rotated NLL
within each replica. All NLLs use the same fixed 30-degree rotated panel and
training-frequency class weights, with no EWC/Ridge/KL penalty included.
The frozen reference is the initial *rotated test* loss (1.6068), not unrotated
training loss; FD's acquisition score (1.6542) averages checkpoints 1--100,
whereas its final NLL (1.5596) uses only checkpoint 100. The definitions and
equations are included in an expandable note. Batch-aligned plotting is labeled
explicitly. This is a reporting clarification, not a scientific change or rerun.

Command used from the repository root (the run is already complete):

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.drip_confirmation \
  --source-root cache/mnist_experiment/plan15/drip_development_v1 \
  --root cache/mnist_experiment/plan15/drip_confirmation_v1 \
  --allocation-hours 10
```

Resume with `--resume`; add `--clear-stop` after a deliberate STOP request.
The run root's `status.json`, frozen `drip-confirmation-ledger-v1`, accepted
batch checkpoints, and final `drip-confirmation-results-v1` are authoritative.
The initial forecast remains 5--8 hours, with approximately 4.4 hours of online
trajectory work projected from the paired development timings before fresh
initialization, checkpoint/report writes, and validation. No additional condition
or replica is authorized by the Phase 6 guiding questions.

## Rare Rotated-Domain Drip (Phase 5B completed)

Phase 5B development of finite-difference KL sample efficiency in a rare
30-degree rotated domain completed on 2026-10-23. Phase 5C fresh confirmation
completed separately above. The pre-run compute forecast was 5--7 hours for 5B and
5--8 hours for 5C; Phase 5B used 5.21 active hours.
Provisional Phase 6 questions are recorded in `plan15.md`: measured FD overhead,
equal-compute replay comparison, intermittent FD/replay allocation, then limited
buffer/budget sensitivity. These notes do not change Phase 5C or allocate Phase 6
compute. No Phase 6 implementation or run has started.
The Phase 5B runner is `fisher_memory.drip`; its deterministic 100-event stream,
four paired arms, strength selection, immutable artifacts, and live section in
`fisher_memory_results.ipynb` are implemented. The accepted-step cached-MNIST
stop/resume smoke passed, as did 220 focused unit tests (four CUDA skips).

Run from the repository root inside `tmux` with the cumulative eight-hour cap:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.drip \
  --root cache/mnist_experiment/plan15/drip_development_v1 \
  --allocation-hours 8
```

Use the same command plus `--resume` after an unplanned interruption. A requested
cooperative stop is a `STOP` file in that root; resume afterward with both
`--resume --clear-stop`. The root's `status.json`, sealed selection unit, and
completed trajectory units are authoritative. A paused invocation exits 3;
neither pause nor failure should trigger a Phase 5C launch automatically.

Launched in `tmux` session `plan15-drip-5b` on 2026-10-22 and resumed in the
foreground after the node was repurposed. The original console output is in
`cache/mnist_experiment/plan15/drip_development_v1.log`. All 46 trajectories,
including four predeclared upper-boundary probes, and all 55 artifact units
passed full content-hash validation. The sealed pilot selected FD strength
160.89 at the new upper grid boundary. Across the four screen replicas, its
mixed-stream acquisition gain was +0.500 mean 30-degree NLL-AUC over replay/EWC,
but the base-only gain was +0.413 without rotated training images. The
descriptive interaction was +0.087 (3/4 positive), and the mean 0-degree AUC
difference in the mixed stream was -0.128 (improvement). Thus most treatment
lift is not specific to exposure to the rare domain; protection or general
regularization remains a strong explanation. These are selected-on-pilot
development estimates, not a significance test or proof of STM-to-LTM transfer.
See the concise main notebook and sealed selection artifact for the development
readout; fresh confirmation is reported above.

## Sigmoid JVP Repair (completed)

The versioned `fisher_memory.jvp_repair` runner completed a fresh
sigmoid-rotation strength reassessment. Its frozen grid included the exact
finite-difference-selected coefficient, approximately 26.8897, three weaker
settings, and one stronger setting. Three fresh pilot replicas selected a JVP
strength subject to the stated retention margin; a disjoint 12-replica cohort
confirmed it. Each replica had a new paired replay/EWC control. The original
completed roots remain read-only.
See Phase 5A in `plan15.md` for the scientific rules.

The code and tiny cached-MNIST pause/resume pipeline were tested. The repair
completed all 42 trajectories in 2.92 active hours; all 59 units passed full
artifact validation. The pilot selected 26.8897, and the 12 fresh confirmation
replicas favored JVP over replay/EWC on current and historical sigmoid NLL AUC
in all 12 pairs. The mean gains were 0.1145 and 0.1359, respectively. The
command used to launch the completed root from the repository root was:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.jvp_repair \
  --source-root cache/mnist_experiment/plan15/production_v4 \
  --root cache/mnist_experiment/plan15/jvp_repair_v1 \
  --allocation-hours 6
```

For an interrupted run, use the identical command plus `--resume`. A paused
invocation exits with status 3, so `&&` will not silently start a later command.
The live notebook is `mnist_experiment/fisher_memory_jvp_repair_results.ipynb`.
Measured prior sigmoid medians were about 299 seconds per JVP trajectory and
114 seconds per replay/EWC control. The pre-run forecast was 3.5--5 hours wall
clock including assets and validation; realized active time was 2.92 hours.

## Optional Method Extension (completed)

The `fisher_memory.extension` entry point now supports separately sealed JVP
and/or directional-KL Phase 5 trajectories with explicit environment scope,
replica count, cumulative compute allocation, and `--resume`. It reuses the
original `production_v4` replay/EWC summaries as verified read-only paired
controls, freezes pilot-selected strengths, and reproduces the confirmation
seed family, asset-plan hashes, and exact initial learner-state fingerprints in a
new run root. The extension notebook is published separately and reports
progress plus paired current/historical NLL AUC. See the Phase 5 command in
`plan15.md`.

**Completed on 2026-10-19.** `cache/mnist_experiment/plan15/extension_v1`
contains all 72 JVP and directional-KL trajectories (12 paired replicas for
each of three environments and two methods). Its 97 artifact units passed full
content-hash validation; no trajectories are missing or unsealed. The run used
approximately 4.61 cumulative active hours. The main results notebook now
includes a concise Phase 5 comparison, and the separate extension notebook
retains the progress and per-replica view.

Relative to paired replay/EWC, both alternative penalties show positive mean
current and historical NLL-AUC gains in rotation. Directional KL is closest to
the selected finite-difference method at lower measured online cost; JVP is
slower and substantially weaker on sigmoid rotation. Neither alternative shows
a clear digit-9 mixture gain. These are secondary nominal paired comparisons,
not a new familywise-confirmatory claim or proof of STM-to-LTM transfer.

Verification: 216 focused tests passed with four expected CUDA-only skips,
including a cached-MNIST CPU entry-point test that interrupted after one
trajectory and resumed the remaining method. The real sealed source root passed
the extension-ledger preflight for all 72 method/environment/replica jobs before
the extension was launched.

This added scientific source changes the current source hash, so do not attempt
to resume the completed `production_v4` study using the new source tree. Its
sealed artifacts remain the read-only reference, and a reporting-only refresh
does not require resuming its scientific scheduler.

## Status

**Phases 1--4 completed on 2026-10-18 (America/Los_Angeles).**
`production_v4` contains 284 completed trajectories: 140 development and 144
fresh confirmation trajectories (12 replicas x 3 environments x 4 conditions).
All 630 artifact units passed a full content-hash validation. There are no
incomplete units, recorded solver failures, nonfinite results, or missing
primary retention endpoints. Phase 5 was run subsequently, as noted above.
The run consumed approximately 14.5 cumulative active hours, below its 20-hour
allocation; retired diagnostic roots and implementation tests are separate.
The final production artifacts plus budgeted dependencies occupy about 1.9 GB,
far below the strict 500 GB cap. No experiment process should remain running.

### Final Readout

Positive gains below mean replay/EWC minus selected finite-difference KL NLL AUC.
These are nominal paired-bootstrap intervals for each environment separately,
not familywise claims. Development replicas are not pooled into confirmation.

| Environment | Current gain (95% interval) | Historical gain (95% interval) | Favorable current / historical |
| --- | --- | --- | --- |
| Linear rotation | .13985 [.11385, .17453] | .15995 [.12794, .20097] | 12/12; 12/12 |
| Sigmoid rotation | .13725 [.11127, .17301] | .17728 [.14980, .20575] | 12/12; 12/12 |
| Digit-9 mixture | -.00070 [-.00413, .00336] | -.00010 [-.00578, .00610] | 5/12; 6/12 |

The rotation predictive gain is clear in this bounded-solver experiment, but
STM-to-LTM consolidation is not established. Mean current NLL AUC for the
selected method is .51745/.59207 for linear/sigmoid versus .53250/.61964 for
the frozen initial model; these smaller differences are descriptive, not a
new predeclared confirmatory contrast. Much of the gain versus replay/EWC is
therefore protection against negative adaptation. STM removal often helps the
rotation controls, and reducing that harm is not evidence that useful information
transferred. LTM-only loss improves, but ordinary regularization or reduced
capacity remain live explanations. No fresh-arrival intervention was measured.

Mixture results do not show a clear predictive benefit from curvature reduction.
Distillation lowers LTM-only historical loss but does not establish a predictive
gain in the primary full-model outcomes. Streams contained 4--17 observed 9s
(mean 10.33, expected 10.1); each mixture trajectory's final cohort has shorter
declared follow-up, handled without extrapolation and identically across pairs.

Persistent rotation under-solving remains important: approximately 95--97% of
selected-method updates ended above the gradient tolerance despite the repaired
common ceiling. No failed step was silently accepted. Selected-method mean
online trajectory times were about 203/206/169 seconds versus 114/114/88 for
ordinary replay/EWC. The calibrated larger-ceiling replay control used only
124/126/95 seconds; describe it as an allowance control, not an actual equal-time
comparison. Do not claim computational superiority or isolate consolidation
from these controls alone.

Final focused tests passed 209 cases (four CUDA-only skips), including 59
report tests after the reporting-only readout revision. CUDA recovery evidence from
implementation remains recorded below. The concise notebook includes a frozen
reference, a plain-language readout, measured cost, and these interpretation
caveats. A reporting-only refresh can replay this completed root with the resume
command below; it must not create or replace scientific trajectories. The
scientific source hash remains `15b2fd8224d22ebca6391f7e0c2ba81031fa98205df62484a97624e4d8f58c91`.

### Execution History

**Production phases 1--4 started under Sol on 2026-10-17.** The user authorized
the full staged execution after Phase 0. The initial cumulative active-time
allocation is 20 hours, following the proposed allocation below. The run root is
`cache/mnist_experiment/plan15/production_v4`; its status, sealed decisions, and
checkpoints are the authoritative progress record. The scientific source hash
was versioned after the common-solver correction below. GPU access requires execution outside the restricted
sandbox on this node. Resume this root with the following command:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/default.json \
  --root cache/mnist_experiment/plan15/production_v4 \
  --allocation-hours 20 --resume
```

Add `--clear-stop` only when resuming a requested cooperative STOP. Recheck that
no writer owns the root before launching; do not launch a second process after
an assistant disconnection while the first is still running. No change to the
confirmation cohort has been made at launch.

The first attempt, `production_v1`, stopped cleanly after about two minutes.
Its complete default linear trajectory had 100% under-solved updates (gradient
infinity norm above 1e-3), but the scheduler had only triggered the planned
100-iteration/150-closure repair for outright solver failures. Sol corrected
that omission: a majority of under-solved updates now also triggers the one
bounded common-ceiling repair. All later treatments inherit the selected common
ceiling. This changes scheduling, so `production_v1` remains separate diagnostic
evidence; do not resume or pool it with later roots. Significant repair notes persist
across resume and appear in the live notebook. No confirmation data was seen.

`production_v2` also stopped cleanly during Phase 1. The larger ceiling reduced
the default linear mean final gradient from .01375 to .00108, but several rejected
line-search increases near convergence were classified as solver failures.
Version 3 retains the previous accepted iterate for every objective increase,
records the rejected increase and its tolerance, and calls increases no larger
than 16 machine epsilons times max(1, absolute objective) rounding stagnation.
Material increases remain solver failures. CPU float32/float64 regression tests
check both cases. Policy search cannot reselect the original shorter ceiling
after applying its repair. Old development artifacts remain separate; no confirmation
outcomes informed either repair.

Version 3 completed Phase 1, but its mixture directional-KL check falsely rejected
one checkpoint because the fixed local radius .001 had 4.3% gradient-magnitude
truncation error. The independent derivative check passed. On the same saved
checkpoint and direction, radius .00025 reduced that error to 1.08%, with cosine
above .99997. Sol added the bounded mathematical-check ladder
{.001, .0005, .00025, .0001}, recording every attempted radius. All checks must
still pass; failure at every radius leaves the method unavailable. The actual
training direction and radius cap are unchanged. Version 3 stopped cleanly in
early Phase 2, before any confirmation, and remains separate. Version 4 reruns
calibration so the mixture directional method can receive a fair predictive
pilot if its mathematical checks pass. The repeated calibration costs about
half an hour, documented here rather than silently pooling versions.

Version 4 scientific source hash:
`15b2fd8224d22ebca6391f7e0c2ba81031fa98205df62484a97624e4d8f58c91`.
The updated focused suite passed 206 tests (four CUDA-only skips in the CPU
invocation), and the miniature staged scheduler completed through closeout.
The learner corrections also passed all 16 CUDA recovery cases. Verification
records are `repair_v4_unit_tests.xml`, `repair_v3_cuda_recovery.xml`, and
`repair_pipeline_v4` under the Plan 15 cache. The first seven repeated baseline
trajectories matched version 3 exactly on current, historical, prequential, and
initial NLL AUC and realized positive counts. Source/configuration identities in
the production root, rather than this prose, determine resume compatibility.

**Version 4 Phase 1 is complete; Phase 2 is running.** Calibration took about
25 active minutes. All A/B/C methods and distillation passed in all three
environments. The common policy is replay weight .5, ridge multiplier .1,
100 iterations / 150 closures. Two probes are used; the checked FD step is .01
for both rotations and .005 for the mixture. The mixture baseline passed the
acquisition/retention health checks; rotation baseline balance remains unresolved
and is disclosed. The last mixture directional check passed at local audit
radius .00025. No confirmation results exist yet. Later status/ledger artifacts
supersede this checkpoint prose.

Early Phase 2 interpretation question, not a conclusion: the two-replica linear
JVP screen improved mean current/historical NLL as strength increased. At the
largest tested coefficient (53.63), means were .573/.544 versus .800/.810 for
replay/EWC. However, pre-eviction STM reliance was already nearly zero, and
muting STM often helped the unregularized control. Capacity suppression or
ordinary sensitivity regularization is therefore a live alternative to memory
transfer. Revisit the paired pre/post-eviction functional diagnostics at closeout;
do not infer consolidation solely from lower post-eviction reliance. Validation
and confirmation remain pending; no extra control is automatically authorized.

The same screen's frozen-initial reference is even better: mean current NLL AUC
.529 versus .565 for the strongest directional setting. Rotation improvements
over adapting replay/EWC may therefore be protection against negative adaptation,
not useful acquisition. Report both comparisons at closeout and keep the mixture
test central. A reporting-only update adds the stored frozen-model NLL as a
dashed curve and this capacity-suppression caution to the notebook. It passed
57 report tests; the scientific hash stayed unchanged. A clean checkpointed
restart at about 2.26 cumulative active hours loaded the renderer in v4.

At about 3.08 active hours, linear Phase 2 completed (54 total trajectories,
zero solver-failure units). The sealed linear winner is finite-difference KL,
coefficient 53.633903847187014, h=.01, two probes. Fresh validation means were
current/historical .561766/.493613 versus .660280/.634098 for replay/EWC;
the frozen-current reference was .615151. This is two-replica development,
not confirmation. JVP and directional KL also had favorable mean validation
gains; distillation was slightly worse than replay/EWC. Linear compute-matched
replay is frozen at 195 iterations / 292 closures, using the maximum paired
development fit-time ratio 1.94099. Sigmoid Phase 2 is now running independently.

Interpretation clarification: `pre_eviction` is after rehearsal, immediately
before the cohort's first member is evicted. It is not fresh-arrival STM use.
Low reliance there is compatible with the proposed transfer mechanism; it does
not establish that STM was unused from the start. Capacity suppression remains
an alternative, not an established explanation. The reader's glossary and
functional caveat in the report source now say this explicitly. That wording
will load on the next resume, or should be included by a fresh artifact-only
report refresh at closeout; the current process imported the preceding renderer.

Resumed on 2026-10-18 after node preemption at 5.06 cumulative active hours.
The writer lock was free; CUDA was available. The interrupted sigmoid
distillation trajectory recovered its accepted step 98 and continued without
changing scientific identities. This resume also loaded the glossary wording
above. Sigmoid Phase 2 subsequently sealed finite-difference KL, coefficient
26.88965972439316, h=.01, two probes. Its fresh two-replica validation means were
current/historical .598606/.522949 versus .705795/.687189 for replay/EWC;
the frozen-current reference was .718449. Compute-matched replay is fixed at
193 iterations / 290 closures (development fit-time ratio 1.92840).
At about 6.67 active hours, the mixture screen is running, 112 trajectories are
complete, and no completed unit has a recorded solver failure or nonfinite
result. These are development results, not confirmation. The 20-hour cumulative
allocation, fixed 12-replica confirmation target, and scientific hash are
unchanged. Live status and sealed decisions supersede this prose.

**Phase 2 completed; Phase 3 started at about 7.97 active hours on 2026-10-18.**
All three environments selected finite-difference KL. The mixture coefficient is
.0017778404607752718, h=.005, two probes. Its fresh validation current/historical
means were .972700/.791828 versus .965603/.784297 for replay/EWC: the sealed
classification is `diagnostic_no_acquisition_gain`, not a favorable candidate.
It still receives the planned confirmation rather than stopping for efficacy.
Mixture distillation uses .005090545506809547; compute-matched replay uses
199 iterations / 298 closures (development fit-time ratio 1.98548).

The sealed confirmation ledger fixes 12 fresh replicas per environment and
144 trajectories total: ordinary replay/EWC, selected FD, distillation, and
compute-matched replay. Jobs are round-robin by replica/environment. Its
outcome-independent forecast is 29,310.6 seconds (8.14 hours) remaining,
including 20% timing overhead, and 3.27 GB projected artifacts with byte
headroom. The 20-hour active-time allocation is sufficient on that forecast.
Do not alter the cohort from running outcomes or release its inference early.

**At the Phase 0 handoff (2026-10-17),** no production calibration,
predictive pilot, or confirmation cohort had been started. The `development_*` cache roots
are implementation smoke/debug evidence only. Do not pool them with experiments.

The initial implementation lives in `fisher_memory/`. Read its `AGENTS.md` and
[Plan 15](plan15.md). The live, artifact-only report is
[fisher_memory_results.ipynb](fisher_memory_results.ipynb).

## Starting And Resuming

Run from the repository root, using `~/.venv/bin/python`. A persistent terminal
such as tmux is sufficient; the Python scheduler does not depend on an assistant
connection. The user must explicitly allocate production compute before launch.
`<hours>` is a cumulative ceiling across invocations, not additional time.

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/default.json \
  --root cache/mnist_experiment/plan15/production_v4 \
  --allocation-hours <hours>
```

Resume with the same command plus `--resume`. If a STOP file was requested, also
add `--clear-stop`; it is cleared only after identity validation and locking.
Increasing `--allocation-hours` extends the cumulative allowance without changing
scientific configuration. `--max-wall-seconds` limits one invocation;
`--max-units` limits newly completed trajectory units.

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.control status \
  --root cache/mnist_experiment/plan15/production_v4
~/.venv/bin/python -m mnist_experiment.fisher_memory.control stop \
  --root cache/mnist_experiment/plan15/production_v4
```

SIGTERM/Ctrl-C are cooperative. An abrupt kill retains the last committed arrival;
at most the current uncommitted arrival is repeated. Pretraining checkpoints at
most every sixteen batches; initial Fisher accumulation checkpoints every chunk.
Never manually delete a completed unit to force reuse. The scheduler resumes
sealed decisions and skips validated completed work.

Default dataset, notebook, and budget locations can be overridden with
`--data-root`, `--notebook`, and `--budget-root`. A run must be a child of its
shared budget root. One writer across that root prevents concurrent studies from
each spending the same 500 GB allowance. Shared datasets, sibling Plan 15 runs,
checkpoint generations, temporary writes, reports, and orphan report exports
are charged. A free-space reserve is enforced separately.

## Scientific Defaults

- Rank-two additive STM; 487 LTM plus 176 STM parameters. No factor reset/merge.
- Replay capacity 32, weight .5; ridge multiplier .1 times mean archived LTM
  Fisher curvature. EWC pi=.025 for rotations, .05 for the mixture.
- Rotations: 120 arrivals of four images, separately linear and sigmoid.
- Mixture: common p=0 arrival, then 100 arrivals of two images through p=.1;
  10.1 expected online 9s. Fixed evaluation prevalence .1 changes evaluation only.
- Initial fit: 30,000 examples, 15 epochs; 1,024 disjoint reference inputs with
  exact model-label integration. Mixture initialization excludes 9s.
- Bounded L-BFGS: 50 iterations/75 closures; two fixed training probes. These
  numerical/strength defaults await Phase 1 calibration, not confirmation tuning.
- Two screening and two fresh validation replicas per environment; nominal
  confirmation target twelve fresh paired replicas and four conditions.

The scheduler performs the bounded common-policy calibration and one permitted
strength-bracket extension, records its choices, and carries a valid diagnostic
candidate even when efficacy is disappointing. It freezes confirmation before
examining confirmation outcomes. No significance-based stopping or extensions.

## Monitoring And Corrections

Status and sealed accepted metrics update during execution. Notebook publication
occurs at startup, at most fifteen minutes apart during progress, and on exit.
Open the existing notebook during a run; the standalone report writer deliberately
cannot compete with either the active Store lock or the shared storage-budget
lock, even for a different run root. It recounts sibling storage before writing.
Its curves distinguish current holdout,
historical holdout, and evicted training cohorts. Detailed diagnostics stay in
small per-unit JSON and CSV artifacts, not notebook output dumps.

Sol owns subsequent execution and repairs. It need not wait for Astra or the user
to fix a defect. Preserve evidence, version materially changed scientific code or
configuration with a new root, document the change and affected results, and do
not silently retain a confirmation label after outcome-informed redesign.
Reporting-only changes do not change the scientific source hash. Training,
curvature, sampler, scheduler, and artifact-code changes do.

The notebook's smoke plots are deliberately poor predictions from tiny initial
fits. They demonstrate software behavior, not a negative scientific result.
Production-shaped timing measurements also use low-fidelity initial parameters;
Phase 1 must replace extrapolations with actual complete-trajectory costs.

## Verification

The immutable acceptance manifest is
`cache/mnist_experiment/plan15/phase0_acceptance_v1/units/phase0-acceptance/acceptance.json`.
It verifies complete artifacts, source/configuration hashes, and pytest XML.

- Focused unit suite: **195 passed**; its four CUDA-only skips subsequently passed
  in a separate **32-test CUDA mathematical suite**.
- CPU integration: **24 passed**, including SIGTERM and SIGKILL during fitting,
  checkpoint writes, validation, and reporting.
- CUDA integration: **16 passed**, with exact interrupted/uninterrupted equality
  for all six conditions on rotations/mixtures and both initialization stages.
- Final CLI smoke: **18 CPU and 18 CUDA trajectories**, all three environments,
  plus a **34-trajectory miniature staged study** through closeout. No nonfinite
  or solver-failure outcomes in these acceptance trajectories.
- Completed-study fresh-process resume: **9.24 seconds**, without retraining.
- Notebook: **15 cells, six figures**, about 0.75 MB; titles/axes visually checked.

Scientific source hash:
`ec987462b89e49239137163c6c52451c134bbc7c94f53576c24377329a6abbfd`.
Exact CPU, CUDA, and pipeline config hashes and the dataset content hash are in
the acceptance manifest. All final run artifacts are under `acceptance_cpu_v2`,
`acceptance_cuda_v1`, and `acceptance_pipeline_v1` inside the Plan 15 budget root.
Reports and control code are excluded from the scientific hash; rendering-only
fixes do not invalidate completed learning trajectories.

## Timing And Remaining Risks

RTX 4070, PyTorch 2.10.0+cu128, CUDA 12.8, float32 execution. The CUDA timing probe
used production dimensions: 32 replay examples, four arrivals, two probes,
50 iterations/75 closures, and the actual 663-parameter model.

| Operation | Measured seconds |
| --- | ---: |
| Replay/EWC solve | 0.438 |
| JVP solve | 1.324 |
| Finite-difference solve | 0.961 |
| Directional-KL solve | 0.947 |
| Distillation solve | 0.657 |
| Four-example eviction Fisher | 0.012 |
| 1,000-image evaluation | 0.0067 |

These are single, low-fidelity-state timing probes, not predictive pilots or
convergence validation. All used the 75-closure ceiling. Phase 1 must inspect
actual initialized-trajectory residuals and under-solving diagnostics; it may
use the planned bounded common-ceiling repair. Initial Fisher diagnostics compare
a nested prefix to the full reference sample, not to independent truth.

Revised **provisional production forecast: about 10--24 active hours**:
Phase 1 0.5--2 h; Phase 2 4--8 h; Phase 3 5--13 h; Phase 4 0.1--0.5 h.
This extrapolates measured solve costs across the planned 12-replica design,
with initialization, audit, serialization/reporting, and repair headroom. Full
production initialization and complete trajectories have not yet been timed.
Twenty hours is a reasonable proposed first allocation, not an authorization
or guaranteed completion time. Replace the forecast after real Phase 1 timing.

The three final smoke roots total about **40 MB**; source datasets are shared.
Production should be in the few-GB range with this artifact design, with a
conservative target below 50 GB. Admission still enforces the hard **500 GB**
ceiling, including all earlier Plan 15 roots and temporary generations.

All Phase 0 compute has stopped. The original handoff checkpoint was followed by
the user-authorized production launch recorded at the top of this file.
