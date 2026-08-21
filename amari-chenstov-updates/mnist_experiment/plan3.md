# Implementation Plan 3: Memory-Constrained Continual Adaptation

Plan 3 compares Fisher-compressed history, bounded replay, and their hybrid in
the low-data regime established by [plan2.md](plan2.md). It is deliberately
staged: each computational phase changes one scientific factor, then stops for
review before later factors are crossed.

## Motivation

Plan 2 established that a high-quality initial model can use an EWC summary to
adapt more efficiently than a current-batch-only learner when only eight new
observations arrive per environmental step. Plan 3 asks the application-level
question that result enables:

> How much predictive quality can a small-compute learner retain when old
> observations are replaced by a fixed-size Fisher summary, and when is a
> small replay window worth its additional memory and repeated optimization?

The experiment also isolates whether LFUs improve the moving Fisher summary.
Replay and LFU are not crossed until each has earned a narrower setting.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Contracts and frozen handoff | Complete |
| 1 | Replay engine and resource ledger | Complete |
| 2 | Replay-capacity screen | Complete |
| 3 | Hybrid archive and memory matching | Complete |
| 4 | History-mechanism frontier | Complete |
| 5 | LFU isolation | Complete: no LFU selected |
| 6 | Deployment frontier | Complete |
| 7 | Analysis and next-manifold handoff | Pending |

## Frozen Plan 2 Handoff

### Accepted regime

- Canonical 512-parameter CNN and full-network likelihood Fisher.
- Eight new observations at each of 100 evenly spaced values of
  $p=\mathbb P(M_i=1)$ from zero to one.
- Fifty L-BFGS inner iterations per update.
- Rank-8-plus-diagonal Fisher summaries.
- Fixed $\pi=.05$ as the practical EWC policy.
- Adaptive $h=.20$, $\pi_{\min}=.05$ as a portability diagnostic. It exactly
  reproduced fixed $.05$ on confirmatory replicas because it stayed at its
  lower bound.
- No LFU as the accepted baseline, not as a conclusion against LFUs.
- Replicas 6 through 10 as paired Plan 3 development units.

The frozen source-anchor bundle is
`phase9-initial__r0006-r0010__065c98b1061d`. The paired Plan 2 confirmation is
`plan2-low-data__r0006-r0010__7bb69d3a9424`. New conditions reuse their model
initializations, $m=8$ streams, partitions, holdouts, and reference paths; they
write new immutable run artifacts.

### Accepted evidence

Over $p<.5$, fixed and adaptive $.05$ EWC obtained mean environmental-accuracy
AUC $.682$ versus $.532$ for no EWC. Their NLL, Brier, ECE, and non-nine
retention improved in every confirmatory replica. The tested joint Equivalent
Data Multiplier remains bracketed between approximately two and four rather
than identified as a precise scalar.

### Conditions not promoted

- $m\in\{1,2\}$ caused general failure and $m=4$ was borderline.
- $m=16$ and $m=32$ remain the EDM bracket, not the target regime.
- $m=64$ was near parity and $m=128$ favored current-batch learning.
- Fixed $\pi=.10$ was weaker than fixed $.05$ in the target regime.
- $\pi_{\min}=.01$ destabilized the current plug-in controller.
- Adaptive $\pi$ did not demonstrate useful variable actuation on this path.
- Dense Fisher representations and broad optimizer grids are not reopened.

## Governing Contracts

### Replay semantics

At update $t$, the replay likelihood contains the current batch exactly once
and all observations present in the buffer before that update. Every replayed
observation receives ordinary likelihood weight; replay samples are not given
an EWC penalty. After a successfully completed update, the current batch enters
the buffer. An aborted transaction inserts nothing. Repeated MNIST indices are
retained as distinct arrival events, preserving the with-replacement stream.

Bounded replay uses deterministic FIFO eviction. This intentionally favors
recent observations. Pure replay discards evicted observations. The hybrid
instead moves evicted observations into an EWC archive. Active-buffer and
archive observations must remain disjoint so no observation contributes to
both terms at once.

The unbounded replay control never evicts an online observation. It retains at
most 800 observations in this experiment, so it is computationally practical.
It is the unconstrained-memory control, not a predicted accuracy ceiling: stale
observations can bias it toward earlier environmental states. It is also not a
full retraining oracle over the original $p=0$ data.

### Memory semantics

Replay may store MNIST dataset indices physically, but its scientific memory
cost is the logical payload required in deployment. Report both logical payload
bytes and measured resident/serialized bytes. Memory matching uses measured
persistent learner state after common model weights are separated:

- EWC anchor parameters, Fisher factor, Fisher residual diagonal, and
  controller state;
- replay observations, labels, identities, and buffer metadata;
- hybrid EWC state plus replay state.

The primary comparison uses a canonical serialization, not Python object
overhead. For $p=512$, rank $r=8$, and float32 state, the fixed-policy EWC
summary stores its anchor, factor, and residual diagonal in

$$
B_{\mathrm{EWC}}=4p(r+2)=20{,}480\text{ bytes}.
$$

A replay item stores 784 uint8 pixels, one int64 label, and one int64 identity,
or 800 bytes. The FIFO stores 24 bytes of fixed capacity, size, and head-index
metadata. Its derived memory-matched capacity is therefore

$$
B_{\mathrm{match}}
=\left\lfloor\frac{20{,}480-24}{800}\right\rfloor=25.
$$

Report common model and optimizer state separately. Measured resident and
serialized implementation bytes remain secondary outcomes. Peak working
memory is not interchangeable with persistent memory.

### Hybrid weighting contract

The baseline applies fixed $\pi=.05$ between the archived EWC summary and the
entire active likelihood block, with equal likelihood weight inside that block.
Increasing replay capacity therefore improves the active likelihood estimate
without silently changing the selected EWC weight. No code may reinterpret
$\pi$ as a per-observation replay weight or double-count an active sample in
the archive. A composition-adjusted $\pi$ would be a separately named future
condition.

### Outcomes

Preserve Plan 2's expected trajectories over $p_t$:

1. digit-9 OvR accuracy;
2. digit-9 precision at environmental prevalence;
3. digit-9 recall;
4. environmental multiclass accuracy;
5. NLL, Brier score, ECE, and non-nine retention.

Report pointwise uncertainty, first-half AUC, actual total and unique digit-9
exposure, durable acquisition, and discrete EDM brackets. Resource outcomes
are optimizer iterations/evaluations, score gradients, HVPs, wall time, CUDA
elapsed time, persistent bytes, peak CPU/GPU memory, and artifact bytes. Keep
data, compute, latency, and memory as separate efficiency denominators.

## Phase 0: Contracts and Frozen Handoff

### Goal

Turn this handoff into testable configuration and artifact contracts without
implementing replay.

### Scope

1. Validate the frozen replica, initialization, stream, partition, and control
   identities from Plan 2.
2. Freeze replay timing, insertion, eviction, sample weighting, and exposure
   accounting semantics.
3. Define persistent versus transient memory accounting and the exact
   memory-matched-budget calculation.
4. Freeze the staged condition names and named comparisons in
   `plan3_profiles.json`.
5. Preview run count, rough wall time, and artifact storage without writing
   cache artifacts.

### Verification

- Every proposed treatment has a named control and estimand.
- Preview is deterministic and side-effect free.
- The selected-capacity planning placeholder is visibly distinguished from the
  mechanically derived memory-matched capacity.

### Check-in

Approve replay and hybrid weighting semantics before writing the replay engine.

### Completion record

**Status:** Complete (2026-08-18)

- Froze fixed $\pi=.05$ between the EWC archive and the complete active
  likelihood block. Replay observations receive equal likelihood weight inside
  that block; composition-adjusted $\pi$ is not silently introduced.
- Defined canonical persistent memory independently of Python storage. The
  fixed rank-8-plus-diagonal EWC anchor and summary cost 20,480 bytes. A replay
  item costs 800 bytes plus 24 bytes of fixed FIFO metadata, deriving a
  memory-matched capacity of 25 observations.
- Renamed unbounded replay the unconstrained-memory control. It retains all
  online observations but is not assumed to be a predictive upper bound under
  distribution drift.
- Added read-only `plan3_command_center audit-handoff`. It validated five
  completed Phase 9 anchors and all 20 completed Plan 2 confirmation runs,
  including their configuration hashes, master initializations, derived
  streams, paired controls, and reference dependencies. Unrelated unexecuted
  Phase 9 intentions are correctly outside the handoff.
- The source and confirmation manifest SHA-256 values are respectively
  `4de0fbd6a247677aa6d98ebbf4f0cfa7fc36d66629b2237153db8d991621f80c`
  and
  `82b056b754c853067331c9630a032f9556b13d18496670a0e313e831480cda9a`.
- The side-effect-free preview retains 20 new replay-screen runs at 1.64
  provisional sequential hours. All contingent stages currently total 75 new
  runs, 5.49 provisional hours, and 4.19 GiB before phase-gate pruning.
- `python -m pytest -q test/unit/test_plan3.py`: 7 passed.
- `python -m pytest -q test/unit`: 233 passed.
- `python -m mnist_experiment.plan3_command_center audit-handoff`: validated
  without writing artifacts.

**Gate recommendation:** proceed to Phase 1's replay engine and resource
ledger. Pause after CPU/CUDA smoke validation and cost calibration before the
long replay-capacity screen.

## Phase 1: Replay Engine and Resource Ledger

### Goal

Implement deterministic replay and trustworthy resource accounting before any
long screen.

### Scope

1. Add an immutable replay configuration schema with capacities expressed in
   observations and an explicit unbounded value.
2. Implement deterministic FIFO insertion and eviction over recorded stream
   identities.
3. Fit the current batch plus the pre-update buffer as one likelihood dataset;
   prevent duplicate current observations.
4. Keep model initialization, stream, optimizer settings, and holdout pairing
   byte-identical across conditions.
5. Record presented and unique replay counts, optimizer work, logical and
   physical persistent bytes, peak working memory, wall time, and CUDA elapsed
   time by operation.
6. Add CPU and CUDA smoke trajectories for capacities 0, 8, and unbounded.
7. Calibrate the Plan 3 preview cost model from the smoke and a short production
   pilot.

### Verification

- Buffer contents match a hand-calculated FIFO sequence at every step.
- Unbounded replay contains every prior online observation exactly once.
- Repeated optimization is charged as compute but not as unique data exposure.
- Completed Plan 2 runs remain readable and immutable.

### Check-in

Review this tricky implementation, memory ledger, and calibrated wall-time
estimate before launching the replay screen.

### Completion record

**Status:** Complete (2026-08-18)

- Added schema-v13 replay configurations without changing schema-v12 loading,
  mappings, or hashes. The FIFO capacity is an observation count or the
  explicit value `unbounded`.
- Added a dedicated pure-replay runner. It omits Fisher, LFU, and reference
  calculations; optimizes the current batch plus the pre-update buffer; and
  commits arriving events only after successful optimization.
- Checkpoints atomically contain model, optimizer, FIFO, trajectory, and
  accounting state. Source indices are sampled with replacement and are never
  deduplicated; event identities cannot enter the buffer twice.
- The ledger separates arrival events, unique source indices, active objective
  observations, optimizer function evaluations, logical observation bytes,
  physical index-state bytes, serialized state, common model/optimizer state,
  process RSS, peak CUDA allocation, wall time, and CUDA time by operation.
- Completed CPU and CUDA three-step smoke trajectories for capacities 0, 8,
  and unbounded. All six share the frozen stream and initialization. Capacity
  8 retained event IDs 8 through 15 after 16 arrivals; unbounded retained all
  16 events without eviction.
- The CUDA capacity-zero trajectory reproduced the frozen Plan 2 current-only
  control's first three 512-parameter states bit for bit. Removing unused
  Fisher diagnostics therefore changed measured cost without changing the
  learning path.
- Completed one full 100-step CUDA pilot at capacity 32 in `tmux`. It recorded
  792 arrivals, 760 evictions, 32 retained events, and zero accidental current
  event reuse. Every proposal had effective EWC strength zero.
- The pilot took 59.26 seconds: 43.65 seconds for evaluation and 12.47 seconds
  for optimization. Its 3,880 active objective observations and 120,616
  optimizer event-evaluations calibrate the pure-replay preview to 46.789 fixed
  seconds plus 0.003214 seconds per active objective observation. Capacity 32's
  immutable artifacts occupied approximately 1.6 MB.
- The resulting replay-screen estimate is 0.52 sequential GPU-hours and 0.032
  GiB for 20 trajectories. Unbounded replay remains a linear extrapolation and
  therefore the least certain timing estimate. The completed replica-6,
  capacity-32 pilot can be reused in Phase 2.
- Immutable Phase 1 bundle:
  `plan3-phase1__r0006__3930db574a45`.
- `python -m pytest -q test/unit`: 237 passed, 2 skipped.

**Gate recommendation:** proceed to Phase 2 bundle generation, reusing the
completed capacity-32 pilot. Stop again before launching the replay-capacity
screen.

## Phase 2: Replay-Capacity Screen

### Goal

Select a useful bounded replay capacity before crossing replay with EWC or LFU.

### Scope

Using paired replicas 6 through 10, compare:

- reused current-only and fixed-$.05$ EWC/no-LFU references;
- FIFO replay capacities 8, 32, and 128 observations;
- unbounded online replay as the unconstrained-memory control.

The initial capacities represent approximately one, four, and sixteen online
batches. Rank results on predictive quality, repeated optimizer work, latency,
and persistent storage. Unbounded replay remains the unconstrained-memory
control throughout Plan 3; its predictive effect is not assumed positive.

### Verification

- All new conditions use the frozen streams and initializations.
- Replay capacity is the only treatment axis among replay conditions.
- The unbounded condition records no evictions.
- Expected trajectories and paired differences, not one endpoint, drive the
  decision.

### Check-in

Choose the smallest bounded capacity on the useful predictive-cost frontier.
Do not choose LFU, controller, or hybrid settings here.

### Completion record

**Status:** Complete (2026-08-18)

- Prepared and completed immutable bundle
  `plan3-replay-screen__r0006-r0010__16a259169db6`. It contains all 20 replay
  cells over replicas 6 through 10, reuses the accepted replica-6 capacity-32
  pilot, and pairs each cell with the exact completed Plan 2 current-only and
  fixed-$.05$ EWC controls. The analysis validated all 30 condition-replica
  cells before aggregation.
- Over $p<.5$, mean environmental-accuracy AUC rose monotonically from $.650$
  at capacity 8 to $.718$ at 32, $.767$ at 128, and $.772$ with unbounded
  replay. The corresponding EWC and current-only references were $.682$ and
  $.530$.
- Capacity 32 was the smallest replay budget to beat EWC reliably on the main
  classification outcomes. Its paired AUC differences were +$.0357$ for
  environmental accuracy (95% CI $[.0113,.0601]$), +$.0500$ for digit-9 OvR
  accuracy ($[.0109,.0890]$), and +$.1271$ for digit-9 precision
  ($[.0287,.2255]$). Its +$.0931$ recall difference remained uncertain.
- Capacity 128 retained a meaningful predictive gain over 32: +$.0493$
  environmental-accuracy AUC ($[.0325,.0660]$). It nearly saturated unbounded
  replay, whose additional +$.0045$ had a CI spanning zero. Unbounded replay
  slightly improved precision and non-nine retention but slightly reduced
  recall relative to 128.
- The resource frontier is explicit. Mean logical replay state was 6,424,
  25,624, 102,424, and 633,624 bytes for capacities 8, 32, 128, and unbounded.
  Mean optimizer event-evaluations were 35,773, 103,405, 390,043, and
  1,848,525. Thus capacity 128 used about 16% of unbounded storage and 21% of
  its repeated optimizer presentations while matching its environmental
  accuracy within current uncertainty.
- Replay's classification advantage did not imply better likelihood
  calibration. Capacity 32 had NLL AUC 7.68 and ECE AUC .269 versus 1.54 and
  .136 for EWC. This remains a reported tradeoff rather than being hidden by a
  classification-only selection.
- Every unbounded run retained all 792 pre-final-step arrival events with zero
  evictions. Bounded final occupancies and evictions matched their FIFO
  capacities exactly.
- Immutable analysis:
  `phase2__plan3-replay-screen__r0006-r0010__16a259169db6__adfadd6ed85f`.
- `python -m pytest -q test/unit`: 240 passed, 2 skipped before launch.

**Gate recommendation:** select capacity 32 as the smallest useful bounded
replay setting for Phase 3. Preserve capacity 128 as the high-quality bounded
reference and unbounded replay as the unconstrained-memory control. Do not
start the hybrid archive until this selection is reviewed.

## Phase 3: Hybrid Archive and Memory Matching

### Goal

Implement a disjoint active replay window and EWC archive, then derive a true
memory-matched replay condition.

### Scope

1. Maintain a separate archive anchor and Fisher summary representing the
   initialization history plus only observations already evicted from replay.
2. Fit the learner against the archive using the current batch plus pre-update
   replay. After a successful fit, consolidate only the FIFO evictions against
   the prior archive; never recenter the archive at the learner parameters.
3. Move each FIFO eviction into the archive once, after its final active-buffer
   use. A no-eviction step leaves both archive anchor and Fisher unchanged.
4. Treat learner optimization, FIFO insertion/eviction, archive consolidation,
   and archive-Fisher replacement as one transaction. A numerical failure
   commits none of them.
5. Use the same fixed $\pi=.05$ for the learner objective, archive
   consolidation, and no-LFU Fisher blend. Keep archive precision/sample
   accounting explicit.
6. Implement the canonical serialized-byte ledger and verify the derived
   capacity of 25. Any future state-schema change must derive a new capacity
   mechanically; never tune it for accuracy.
7. Add hand-calculated quadratic and tiny-network tests for no loss, duplicate
   loss, or double counting during eviction.

### Verification

- Active and archived observation identities are disjoint and exhaustive.
- Capacity-zero hybrid reduces to the agreed EWC recursion.
- Unbounded hybrid performs no archive insertions after initialization.
- Still-active observations cannot influence the archive anchor through the
  learner fit.
- Memory matching is reproducible from artifact metadata.

### Check-in

Review archive semantics and numerical behavior before the hybrid experiment.

### Completion record

**Status:** Complete (2026-08-20)

- Added schema-14 hybrid runs with a separate archive anchor and rank-8-plus-
  diagonal Fisher. The learner starts from its current parameters but applies
  EWC against the archive anchor; fitting the learner never recenters the
  archive.
- Implemented the clean recursion: fit current plus pre-update replay, stage
  FIFO insertion, consolidate only evictions against the prior archive, blend
  their score Fisher with gain $.05$, then commit learner, FIFO, anchor, and
  Fisher together. No LFU or HVP is used in this phase.
- Added explicit archive/replay identity accounting, checkpoint resumption,
  source-artifact hashing, optimizer-work ledgers, and canonical, measured,
  and serialized memory fields. No-eviction steps preserve the archive
  bit-for-bit.
- Completed immutable bundle `plan3-phase3__r0006__a8ffdd4d6e7b`: eight paired
  CPU/CUDA smokes at capacities 0, 8, 25, and unbounded. The final archived
  online-event counts were respectively 16, 8, 7, and 0 on both devices, with
  zero active/archive overlap and exhaustive event accounting.
- Capacity zero exactly matched the ordinary EWC recursion: the learner and
  archive parameter trajectories were bit-for-bit equal. Unbounded replay
  performed no consolidation. Capacities 8 and 25 left the archive unchanged
  until their first FIFO eviction.
- Canonical final hybrid state was 20,504 bytes at capacity 0, 26,904 bytes at
  capacity 8, and 40,504 bytes at the memory-matched capacity 25. These totals
  include the 20,480-byte archive and 24-byte FIFO metadata.
- `python -m pytest -q test/unit`: 250 passed, 2 skipped.

**Gate recommendation:** proceed to Phase 4 using the validated clean
recursion. Compare one-batch, memory-matched, and selected-capacity hybrids at
capacities 8, 25, and 32 against their pure-replay and fixed-EWC controls. Stop
for review before implementing or running that experiment.

## Phase 4: History-Mechanism Frontier

### Goal

Compare compression, replay, and their hybrid at selected and memory-matched
capacities.

### Scope

Compare fixed-$.05$ EWC/no LFU, one-batch replay, selected replay,
memory-matched replay, one-batch hybrid, selected-capacity hybrid,
memory-matched hybrid, and unbounded replay. The one-batch hybrid has capacity
8 and tests whether a minimal exact waiting room improves EWC. Reuse completed
pure-replay controls. Do not add LFU or adaptive-controller variants.

### Verification

- Each comparison reports incremental persistent bytes and repeated optimizer
  observations alongside predictive effects.
- Hybrid improvements are not caused by active/archive double counting.
- Unbounded replay quantifies the effect of removing the online memory limit;
  stale-history bias may make that effect negative.

### Check-in

Select the history mechanism carried into LFU isolation. A Pareto set is an
acceptable result; do not force one scalar winner across memory and compute.

### Completion record

**Status:** Complete (2026-08-20)

- Completed immutable bundle
  `plan3-history-frontier__r0006-r0010__12582f2d297a`: 20 new trajectories
  across replicas 6 through 10 and 20 reused EWC/replay cells. The new cells
  are pure replay B25 and no-LFU hybrids B8, B25, and B32.
- Every hybrid passed exhaustive per-step event accounting, zero
  active/archive overlap, no-eviction archive immutability, expected final
  archive counts, one score gradient per archived event, and zero HVPs.
- Over $p<.5$, mean environmental-accuracy AUC was $.682$ for EWC, $.650$,
  $.691$, and $.718$ for replay B8/B25/B32, and $.738$, $.768$, and $.774$ for
  hybrid B8/B25/B32. Unbounded replay obtained $.772$.
- Each hybrid improved environmental accuracy over capacity-matched replay.
  The paired gains were +$.087$ at B8, +$.077$ at B25, and +$.056$ at B32;
  all 95% intervals excluded zero. B32 hybrid also improved over EWC by +$.092$
  with 95% CI $[.073,.111]$.
- Hybrid B32 and unbounded replay had no resolved environmental-accuracy
  difference (+$.0026$, 95% CI $[-.033,.038]$). Unbounded replay had better
  digit-9 OvR accuracy and precision, while hybrid B32 had far better NLL
  (1.19 versus 6.11) and ECE (.093 versus .201). This is a real Pareto tradeoff.
- Increasing hybrid capacity from 8 to 25 produced broad predictive and
  calibration gains. B32 improved environmental accuracy over B25 by $.0064$
  (95% CI $[.00005,.0128]$) and NLL by $.0325$, while most individual 9-centric
  differences remained unresolved.
- Canonical persistent state was 26,904, 40,504, and 46,104 bytes for hybrid
  B8/B25/B32, versus 633,624 bytes for unbounded replay. B32 hybrid used 7.3%
  of unbounded storage and 13.7% of its optimizer event-evaluations, although
  its measured trajectory wall time was about 100 seconds versus 72 seconds
  for unbounded replay on this small GPU workload.
- The 20 new trajectories consumed about 30 minutes of measured run time.
  Immutable paired analysis:
  `phase4__plan3-history-frontier__r0006-r0010__12582f2d297a__113026398aa0`.
- `python -m pytest -q test/unit`: 251 passed, 2 skipped.

**Gate recommendation:** carry no-LFU hybrid B32 into Phase 5 as the strongest
bounded predictive configuration and the clearest LFU test bed. Preserve
hybrid B8 and B25 as lower-memory Pareto alternatives; do not rerun them in the
LFU screen unless the Phase 5 effect motivates a targeted follow-up.

## Phase 5: LFU Isolation

### Goal

Measure whether linearized Fisher updates improve the selected moving summary.

### Scope

1. Compare fixed-$.05$ EWC under no LFU, AC-only, and full LFU.
2. Compare the selected hybrid under no LFU and full LFU.
3. Use the same eight-step directional ridge and lagged displacement schedule
   unless a pre-launch diagnostic invalidates them.
4. Preserve pre-projection LFU magnitude, PSD projection, coherence, and
   derivative-cost diagnostics.
5. Keep AC-only diagnostic; promote it only if it reveals a material mechanism
   not explained by full LFU.

### Verification

- Score/HVP sign and lag conventions remain covered by tests.
- LFU conditions differ from their controls only in the applied correction.
- Predictive value is judged together with HVP cost and Fisher diagnostics.

### Preflight record

**Status:** Complete at the preflight gate (2026-08-20); no LFU selected

- The immutable Phase 5 bundle is
  `plan3-lfu-isolation__r0006-r0010__6d8c4ad67265`. Three replica-6 production
  cells completed: EWC AC-only, EWC full LFU, and Hybrid B32 full LFU. The 10
  no-LFU controls remain reused; the other 12 treatment cells were not run.
- In the primary $p<.5$ region, Hybrid B32 full LFU projected away a mean
  33.5% of the candidate Frobenius norm. Projection exceeded 10% on 63% of
  archive updates and 50% on 26%; the maximum was 96.6%. Its directional ridge
  reset on 97.8% of updates, so the intended eight-step averaging almost never
  accumulated.
- EWC full LFU produced materially indefinite candidates on 98% of primary
  steps and reset on 96%. AC-only produced materially indefinite candidates on
  92% and also reset on 96%. AC-only corrections were larger, with mean and
  maximum Frobenius norms 1,409 and 29,635 versus 398 and 6,599 for full LFU.
- The paired pilot environmental-accuracy AUCs were .671 for no-LFU EWC, .545
  for full-LFU EWC, and .517 for AC-only EWC. Hybrid B32 was .736 without LFU
  and .740 with full LFU, but the full-LFU result is projection-defined and is
  not evidence for the intended linearized update.
- Per the predeclared guardrail, the remaining multi-replica compute was not
  launched. The practical recommendation is to carry no LFU into Phase 6;
  reopening LFU requires a newly specified estimator rather than a larger run
  of this failed treatment.
- The Phase 6 decision accepts EMA without LFU. The stopped LFU cells remain
  immutable negative evidence; they are not missing members of a promoted
  treatment matrix.

### Check-in

Choose no LFU or full LFU for deployment. Stop and revise if numerical
projection dominates the proposed correction.

## Phase 6: Deployment Frontier

### Goal

Measure realistic learner costs without oracle-path or high-sample Fisher
diagnostics influencing the runtime comparison.

### Scope

1. Materialize only the selected current-only, fixed EWC/hybrid, adaptive
   EWC/hybrid, bounded replay, and unbounded replay conditions.
2. Remove reference-path and high-sample Fisher diagnostics from the learner
   execution path. Offline holdout evaluation remains allowed and is excluded
   from learner cost.
3. Use the Fisher-update method selected in Phase 5.
4. Retain fixed and adaptive $\pi$ conditions. Adaptive $\pi$ remains a
   diagnostic unless it actuates away from its bounds.
5. Report predictive-memory-compute frontiers and paired uncertainty.

The frozen seven-condition matrix is current-only; fixed EWC; fixed Hybrid
B32; adaptive EWC; adaptive Hybrid B32; Replay B32; and unbounded replay. Both
adaptive conditions use $h=.20$, $\pi_{\min}=.05$, and $\pi_{\max}=.95$.
Fixed conditions use $\pi=.05$. EWC-only is the capacity-zero instance of the
clean hybrid recursion, so current observations enter the compressed archive
only after a successful learner update.

Phase 6 uses a new deployment schema. It loads the high-quality initial
$p=0$ Fisher summary but no reference-optimum path or online high-sample Fisher
diagnostics. Hybrid conditions compute score gradients only when observations
enter the archive, and every condition must record zero HVPs. Adaptive
$\pi_t$ is chosen from prior accepted displacements and the same realized
value weights learner EWC, archive consolidation, and Fisher EMA. Offline
holdout evaluation remains timed separately and is excluded from learner cost.

### Verification

- Removing diagnostics does not alter fixed-policy parameter trajectories
  beyond declared numerical tolerance.
- Learner cost excludes offline evaluation and scientific oracle work.
- Logical storage and measured resident storage are both available.

### Completion record

**Status:** Complete (2026-08-20)

- The immutable seven-condition bundle is
  `plan3-deployment-frontier__r0006-r0010__f21931201ff8`. All 35 paired runs
  completed with no reference path, online Fisher oracle, or HVP. Fixed-policy
  Hybrid B32 reproduced its Phase 4 parameters, displacements, and archive
  anchors exactly on all five replicas. Phase 6 EWC is the newer capacity-zero
  clean archive recursion and has no literal Phase 4 counterpart.
- Over $p<.5$, fixed Hybrid B32 improved environmental-accuracy AUC by .069
  (95% CI [.038, .100]) and 9 OvR AUC by .037 ([.032, .042]) over fixed EWC.
  Its precision gain was .077 ([.047, .107]); its .027 recall gain was not
  resolved at five replicas.
- Against pure Replay B32, fixed Hybrid B32 improved environmental-accuracy
  AUC by .056 ([.033, .080]) and NLL AUC by 6.48. Their 9 OvR, precision, and
  recall differences were unresolved. The hybrid's advantage is therefore
  broader retention and calibration, not faster acquisition of digit 9.
- Fixed Hybrid B32 and unbounded replay were tied on environmental-accuracy
  AUC: hybrid minus unbounded was .003 ([-.033, .038]). Unbounded replay led
  9 OvR by .027 ([.006, .047]) and precision by .090 ([.023, .157]); recall
  was tied. Hybrid led NLL by 4.92 and ECE by .108.
- Adaptive EWC stayed at $\pi_{\min}=.05$ on every step and exactly reproduced
  fixed EWC. Adaptive Hybrid B32 used the lower bound on 81% of steps on
  average, with mean $\pi=.0519$ and mean per-replica maximum $.0571$; its
  predictive effects relative to fixed hybrid were negligible.
- Mean learner-only time and logical persistent state were 48.6 seconds and
  46,104 bytes for fixed Hybrid B32, 11.1 seconds and 25,624 bytes for Replay
  B32, and 23.8 seconds and 633,624 bytes for unbounded replay. Hybrid saves
  13.7x storage versus unbounded replay but costs about twice its learner time
  because archive consolidation is a second optimization.
- The accepted schema-v2 analysis is
  `phase6__plan3-deployment-frontier__r0006-r0010__f21931201ff8__ce7440f3fe91`.
  Its identity includes a content hash so revised analysis definitions cannot
  collide with completed summaries.
- `python -m pytest -q test/unit/test_plan3_analysis.py`: 4 passed.
- `python -m pytest -q test/unit -ra`: 257 passed, 2 CUDA-only tests skipped
  because CUDA was unavailable in the test sandbox.

**Gate recommendation:** carry fixed Hybrid B32, Replay B32, fixed EWC, and
unbounded replay into Phase 7 as a predictive-memory-compute frontier. Treat
adaptive $\pi$ as a diagnostic with no demonstrated value on this linear path.

### Check-in

Choose the practical deployment frontier and decide whether uncertainty needs
more replicas. Reuse replicas unless additional independent units would change
the decision.

## Phase 7: Analysis and Next-Manifold Handoff

### Goal

Produce an artifact-only scientific summary, perform only decision-relevant
confirmatory replication, and identify the next experiment that can test
adaptive $\pi$ under variable speed or curvature.

### Scope

1. Add concise notebook panels for expected trajectories, paired treatment
   effects, EDM, replay occupancy, evictions, and resource frontiers.
2. Link every aggregate to immutable runs and configuration hashes.
3. Separate measured conclusions from assumptions and failed conditions.
4. After Phase 6 freezes the deployment-form conditions, treat Hybrid B32
   versus Replay B32 as a conditional confirmatory target. Use fresh paired
   replicas in predeclared blocks and stop on confidence-interval precision or
   a fixed maximum, not on the first significant result. Environmental
   accuracy AUC is primary; digit-9 OvR accuracy is the principal secondary,
   with precision and recall reported as its required decomposition. Phase 4
   informs variance planning but remains exploratory.
5. Record whether the next experiment should change path speed, curvature,
   model scale, or application domain; do not implement it here.

### Verification

- Notebook execution is artifact-only and fails clearly on incomplete schemas.
- No claim combines data, memory, and compute into an opaque score.
- Every promoted condition has a named control and uncertainty statement.
- Any added confirmation uses the Phase 6-selected LFU method and does not pool
  heterogeneous hybrid definitions.

### Final check-in

Review the evidence before designing a variable-manifold or applied-model plan.
