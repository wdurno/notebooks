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
| 0 | Contracts and frozen handoff | Pending |
| 1 | Replay engine and resource ledger | Pending |
| 2 | Replay-capacity screen | Pending |
| 3 | Hybrid archive and memory matching | Pending |
| 4 | History-mechanism frontier | Pending |
| 5 | LFU isolation | Pending |
| 6 | Deployment frontier | Pending |
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
an EWC penalty. After an accepted update, the current batch enters the buffer.

Bounded replay uses deterministic FIFO eviction. This intentionally favors
recent observations. Pure replay discards evicted observations. The hybrid
instead moves evicted observations into an EWC archive. Active-buffer and
archive observations must remain disjoint so no observation contributes to
both terms at once.

The unbounded replay control never evicts an online observation. It retains at
most 800 observations in this experiment, so it is computationally practical.
It is an expected online-history ceiling, not a guarantee of winning every
nonconvex fit and not a full retraining oracle over the original $p=0$ data.

### Memory semantics

Replay may store MNIST dataset indices physically, but its scientific memory
cost is the logical payload required in deployment. Report both logical payload
bytes and measured resident/serialized bytes. Memory matching uses measured
persistent learner state after common model weights are separated:

- EWC anchor parameters, Fisher factor, Fisher residual diagonal, and
  controller state;
- replay observations, labels, identities, and buffer metadata;
- hybrid EWC state plus replay state.

Report common model and optimizer state separately. Peak working memory is not
interchangeable with persistent memory.

### Hybrid weighting question

The proposed baseline applies fixed $\pi=.05$ between the archived EWC summary
and the active likelihood block, with equal likelihood weight inside that
block. Because replay changes the active block size, Phase 3 must verify this
interpretation before implementation. No code may silently reinterpret
$\pi$ as a per-observation replay weight or double-count an active sample in
the archive.

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
- Planning placeholders are visibly distinguished from selected budgets.

### Check-in

Approve replay and hybrid weighting semantics before writing the replay engine.

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

## Phase 2: Replay-Capacity Screen

### Goal

Select a useful bounded replay capacity before crossing replay with EWC or LFU.

### Scope

Using paired replicas 6 through 10, compare:

- reused current-only and fixed-$.05$ EWC/no-LFU references;
- FIFO replay capacities 8, 32, and 128 observations;
- unbounded online replay.

The initial capacities represent approximately one, four, and sixteen online
batches. Rank results on predictive quality, repeated optimizer work, latency,
and persistent storage. Unbounded replay remains an upper control throughout
Plan 3.

### Verification

- All new conditions use the frozen streams and initializations.
- Replay capacity is the only treatment axis among replay conditions.
- The unbounded condition records no evictions.
- Expected trajectories and paired differences, not one endpoint, drive the
  decision.

### Check-in

Choose the smallest bounded capacity on the useful predictive-cost frontier.
Do not choose LFU, controller, or hybrid settings here.

## Phase 3: Hybrid Archive and Memory Matching

### Goal

Implement a disjoint active replay window and EWC archive, then derive a true
memory-matched replay condition.

### Scope

1. Maintain an EWC archive representing the initialization history plus only
   observations already evicted from replay.
2. Move each FIFO eviction into the archive once, after its final active-buffer
   use.
3. Keep archive precision/sample accounting explicit and verify the agreed
   fixed-$.05$ active-block weighting.
4. Implement the no-LFU archive update first.
5. Derive replay capacity from measured EWC persistent bytes. Replace the
   preview's nominal memory-matched capacity; do not tune it for accuracy.
6. Add hand-calculated quadratic and tiny-network tests for no loss, duplicate
   loss, or double counting during eviction.

### Verification

- Active and archived observation identities are disjoint and exhaustive.
- Capacity-zero hybrid reduces to the agreed EWC recursion.
- Unbounded hybrid performs no archive insertions after initialization.
- Memory matching is reproducible from artifact metadata.

### Check-in

Review archive semantics and numerical behavior before the hybrid experiment.

## Phase 4: History-Mechanism Frontier

### Goal

Compare compression, replay, and their hybrid at selected and memory-matched
capacities.

### Scope

Compare fixed-$.05$ EWC/no LFU, selected replay, memory-matched replay,
selected-capacity hybrid, memory-matched hybrid, and unbounded replay. Reuse
completed controls. Do not add LFU or adaptive-controller variants.

### Verification

- Each comparison reports incremental persistent bytes and repeated optimizer
  observations alongside predictive effects.
- Hybrid improvements are not caused by active/archive double counting.
- Unbounded replay quantifies the loss due to finite online memory.

### Check-in

Select the history mechanism carried into LFU isolation. A Pareto set is an
acceptable result; do not force one scalar winner across memory and compute.

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

### Verification

- Removing diagnostics does not alter fixed-policy parameter trajectories
  beyond declared numerical tolerance.
- Learner cost excludes offline evaluation and scientific oracle work.
- Logical storage and measured resident storage are both available.

### Check-in

Choose the practical deployment frontier and decide whether uncertainty needs
more replicas. Reuse replicas unless additional independent units would change
the decision.

## Phase 7: Analysis and Next-Manifold Handoff

### Goal

Produce an artifact-only scientific summary and identify the next experiment
that can test adaptive $\pi$ under variable speed or curvature.

### Scope

1. Add concise notebook panels for expected trajectories, paired treatment
   effects, EDM, replay occupancy, evictions, and resource frontiers.
2. Link every aggregate to immutable runs and configuration hashes.
3. Separate measured conclusions from assumptions and failed conditions.
4. Record whether the next experiment should change path speed, curvature,
   model scale, or application domain; do not implement it here.

### Verification

- Notebook execution is artifact-only and fails clearly on incomplete schemas.
- No claim combines data, memory, and compute into an opaque score.
- Every promoted condition has a named control and uncertainty statement.

### Final check-in

Review the evidence before designing a variable-manifold or applied-model plan.
