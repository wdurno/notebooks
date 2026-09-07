# Implementation Plan 8: Decomposed EDR Rechallenge

Plan 8 follows the stopped rotated-MNIST EDR challenge in
[plan5.md](plan5.md), the instantaneous-oracle study in
[plan6.md](plan6.md), and the coefficient audit and mathematical repair in
[plan7.md](plan7.md). It does not rewrite any of those experiments. It asks one
narrow follow-up question:

> Did historical EDR fail because it exponentially discounted a known,
> policy-dependent covariance contribution together with the uncertain
> movement contribution, and can separating those terms produce a useful
> online composition recommendation?

The experiment reuses immutable Plan 5 assets only where doing so strengthens
pairing. Every revised adaptive learner begins at the common initial model and
recomputes its complete closed-loop trajectory. No model, Fisher, controller,
or optimizer state may be resumed from a historical EDR trajectory.

Plan 8 is a versioned rechallenge rather than a revision of Plan 5. Its source,
configuration, artifact, test, and notebook additions must remain removable
without changing the meaning or execution of Plans 1--7.

## Motivation

For the applied fixed-batch EWC model, let $q_t$ be the pre-transition
concentration of the normalized weights represented by the old summary:

$$
q_t=\sum_i w_{t,i}^2,
\qquad
q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m_t}.
$$

Under the local calibrated-summary and equal-covariance-shape assumptions, the
centered marginal Fisher risk can be written

$$
R_t(\pi)
=(1-\pi)^2\left(S_t+q_tD_t\right)
+\pi^2\frac{D_t}{m_t},
$$

where

$$
S_t=\|d\theta_t^\star\|_{\mathcal I_t}^2,
\qquad
D_t=\operatorname{tr}(\mathcal I_tK_t).
$$

Define the normalized movement premium

$$
\rho_t:=\frac{m_tS_t}{D_t}.
$$

The complete marginal recommendation and its covariance-only component are

$$
\pi_t^{\mathrm{marg}}
=\frac{\rho_t+m_tq_t}{\rho_t+m_tq_t+1},
\qquad
\pi_t^{\mathrm{cov},q}
=\frac{m_tq_t}{1+m_tq_t}.
$$

When a constant action $c$ and fixed batch size $m$ have reached weight
stationarity,

$$
q_\infty(c)=\frac{c}{m(2-c)},
\qquad
\pi_\infty^{\mathrm{cov}}=\frac c2.
$$

Plan 7 found that the tracked-$q_t$ recommendation reproduced the stored
population covariance-only oracle with mean absolute error below `.001` on
both schedules. The stationary $c/2$ approximation was useful but less
accurate, with errors of approximately `.0094` on linear and `.0120` on
sigmoid. Adding oracle population movement changed the mean recommendations
only from approximately `.135` to `.137` and `.143` to `.146` on that audited
trajectory.

These findings validate the covariance subproblem, not the complete online
controller. Historical EDR discounted the full old- and new-risk coefficients
before taking their ratio. That operation needlessly lagged the exactly known
$q_t$ contribution and made its interpretation depend on variation in the
estimated common covariance scale. Plan 8 instead preserves $q_t$ exactly and
discounts only an online estimate of the unknown movement premium:

$$
\bar\rho_t
=(1-\beta_t)\bar\rho_{t-1}+\beta_t\widehat\rho_t,
\qquad
\widehat\pi_t^{\mathrm{D\text{-}EDR}}
=\frac{\bar\rho_t+m_tq_t}{\bar\rho_t+m_tq_t+1}.
$$

This is a testable repair, not an established estimator. Plan 7 did not prove
that the online trend statistic estimates population movement, and one
trajectory cannot identify the distribution of learner-anchor error. The
experiment must therefore isolate the validated covariance term from the
still-uncertain movement term and judge the resulting policy by both
mechanistic and predictive evidence.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Estimand, inheritance, and comparison contracts | Complete |
| 1 | Decomposed controller and source-backed runner | Complete |
| 2 | Paired single-lap rechallenge | Complete |
| 3 | Double-lap reversal stress | Complete |
| 4 | Canonical repeated-path and applied Hybrid gate | Skipped by gate |
| 5 | Fresh independent confirmation | Skipped by gate |
| 6 | Artifact-only findings and evidence review | Complete |
| 7 | Theory and project-findings integration | Pending |

## Scientific Contract

### Recommendation hierarchy

Keep four objects distinct throughout implementation and analysis:

1. **Tracked-$q$ covariance recommendation**

   $$
   \widehat\pi_t^{\mathrm{cov},q}
   =\frac{m_tq_t}{1+m_tq_t}.
   $$

   This is the Plan 7-validated covariance component under locally matching
   covariance shapes. It is also a closed-loop ablation when actuated without
   a movement premium. In a locally stationary environment it is expected to
   decrease as information accumulates and may eventually reach
   $\pi_{\min}$.

2. **Stationary covariance approximation**

   $$
   \widehat\pi_\infty^{\mathrm{cov}}=c/2.
   $$

   This is a descriptive approximation when a constant historical action $c$
   has already generated a stationary $q_t$. It is not a controller recursion.
   Never implement $c_{t+1}=c_t/2$ as a treatment.

3. **Decomposed EDR recommendation**

   $$
   \widehat\pi_t^{\mathrm{D\text{-}EDR}}
   =\frac{\bar\rho_t+m_tq_t}{\bar\rho_t+m_tq_t+1}.
   $$

   This is Plan 8's principal treatment. The known covariance state enters
   without smoothing, while only the uncertain normalized movement premium is
   exponentially discounted.

4. **Population marginal oracle**

   $$
   \pi_t^{\mathrm{marg}}
   =\frac{\rho_t+m_tq_t}{\rho_t+m_tq_t+1}.
   $$

   This is an offline diagnostic based on the assumed one-step risk. It may
   use compatible Plan 6 reference quantities and the $q_t$ realized by each
   new policy. It is not a globally optimal closed-loop policy and must not
   determine an online action.

### Online movement estimate

Retain Plan 5's predictable trend and Fisher-weighted residual machinery so
the rechallenge changes one controller idea at a time. Immediately before
transition $t$, use only state produced by previously accepted updates:

$$
\widehat S_t=\widehat d_t^TG_t\widehat d_t,
\qquad
\widehat D_t
=\frac{\text{discounted Fisher residual energy}}
{\text{discounted residual scale}},
$$

$$
\widehat\rho_t
=\frac{m_t\widehat S_t}{\max(\widehat D_t,\varepsilon)}.
$$

The principal treatment smooths $\widehat\rho_t$, not
$\widehat S_t+q_t\widehat D_t$ and $\widehat D_t/m_t$ separately. Initialize
$\bar\rho_0=0$. Preserve the source experiment's trend half-life, movement
half-life, eight accepted-update cold start at $\pi=.05$, and
$[\pi_{\min},\pi_{\max}]=[.01,.95]$ bounds. Accumulate predictable movement
diagnostics during cold start but do not actuate them until cold start ends.
If the covariance-scale estimate is unsupported or nonfinite, use the
tracked-$q$ recommendation after cold start and record the fallback.

After accepting the update, advance the trend, residual moments, movement
moment, and

$$
q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m_t}
$$

for use by transition $t+1$. The batch weighted by $\pi_t$ must not enter the
decision for $\pi_t$. One realized $\pi_t$ controls both the EWC objective and
the direct-EMA Fisher update.

Do not add trend debiasing, a change-point detector, an uncertainty throttle,
or a new half-life in the principal treatment. Such changes would prevent the
experiment from isolating the decomposition discovered in Plan 7. Store any
already available debiased statistic as a non-actuating diagnostic only.

### Experimental conditions

Use concise, schema-stable names and record whether a condition was loaded
from a completed source artifact or recomputed:

- `current_only`: contextual Plan 5 control, loaded where compatible;
- `fixed_pi0025`: post-facto research benchmark, loaded where available and
  recomputed when required for a new paired block;
- `fixed_pi005`: prospective untuned baseline and compatibility sentinel;
- `legacy_edr`: historical Plan 5 EDR result, always loaded read-only;
- `tracked_q_covariance`: newly actuated covariance-only ablation;
- `decomposed_edr`: newly actuated principal Plan 8 treatment.

The main scientific contrast is `decomposed_edr` versus `legacy_edr`, which
tests the proposed repair. Comparisons with fixed $.05$ assess prospective
utility. Fixed $.025$ is a post-facto benchmark discovered through prior
experimentation and must not be described as an available online oracle or a
globally optimal policy.

The tracked-$q$ condition identifies what is supplied by the validated
covariance term alone. Its predictive success is informative, but its expected
decline toward the floor in low-movement regions is not itself a controller
failure.

### Predictive outcomes

Rotation affects all ten classes while preserving their empirical frequency.
The primary outcomes are therefore:

- current-environment multiclass accuracy and NLL trajectories;
- exposure-normalized accuracy and NLL AUC over the whole path and each leg;
- fixed-panel accuracy and NLL at $0^\circ$, $15^\circ$, and $30^\circ$;
- per-class recall and worst-class recall as retention diagnostics;
- expected calibration error;
- return, reversal, and matched-angle revisit contrasts; and
- learner time, Fisher time, optimizer evaluations, score-gradient count,
  peak memory, and artifact size.

Digit-9 one-versus-rest metrics may remain supplemental consistency checks but
are not transfer estimands in the fixed-class-composition rotation experiment.
Every trajectory plot must display angle or cumulative online observations on
the horizontal axis and identify schedule direction. Keep principal figures
to two conditions where possible and no more than four.

### Controller diagnostics

Record enough scalar state to reproduce every action without model execution:

- pre-transition $q_t$, $m_tq_t$, and effective size $q_t^{-1}$;
- $\widehat S_t$, $\widehat D_t$, instantaneous $\widehat\rho_t$, and
  discounted $\bar\rho_t$;
- covariance-only, decomposed, unclipped, and applied recommendations;
- cold-start, lower-bound, upper-bound, and unsupported-scale indicators;
- action total variation, span, floor occupancy, and maximum;
- lagged speed/action and oracle/action relationships where defined;
- Fisher traces, rank, Lanczos diagnostics, and residual-scale support; and
- exact action-reconstruction and $q_t$-recursion residuals.

Compatible Plan 6 population quantities may be used to calculate a diagnostic
$\pi_t^{\mathrm{marg}}$ after substituting the new policy's own $q_t$. Do not
interpolate missing reference angles, reuse a conditional oracle tied to the
historical learner anchor, or present oracle agreement as predictive success.

### Decision language

Classify evidence at each check-in without forcing adaptive control into a
single win/loss statement:

- **mechanistic repair:** the decomposed action is exactly reconstructable,
  removes the historical covariance-lag distortion, and avoids the legacy
  weak-EWC/large-step feedback pattern;
- **dynamic value:** schedule-dependent actions respond predictably to
  movement and improve the corresponding path regions beyond tested fixed
  policies without compensating harm elsewhere;
- **automatic-selection value:** decomposed EDR approaches the post-facto best
  tested fixed policy within the predeclared practical margin without using
  repeated policy trials;
- **prospective value:** decomposed EDR improves on untuned fixed $.05$ and
  remains acceptably close to the best tested fixed policy;
- **diagnostic improvement:** decomposition produces coherent and safer
  actions or improves substantially over legacy EDR, but predictive outcomes
  remain materially behind fixed policies; or
- **stop:** the revised policy is unresponsive, unstable, non-predictable, or
  materially harmful despite correct implementation.

A mechanistic repair is not predictive evidence. A result from the inherited
development replica is not statistical confirmation.

### Frozen materiality thresholds

Use the Plan 5 thresholds without inspecting Plan 8 outcomes:

- a practical accuracy-AUC tie is a paired difference of at least `-.01`;
- a material NLL-AUC regression is an increase greater than `.10`;
- a material final upright-accuracy or worst-class-recall regression is a
  decrease greater than `.02`;
- automatic-selection value requires an accuracy-AUC gap no worse than `-.01`
  from the best tested fixed policy;
- a material repair over legacy EDR requires an accuracy-AUC increase of at
  least `.05` on each tested schedule;
- a dynamic action signal requires a fast-minus-slow action difference of at
  least `.005`, positive lagged speed/action association, and a favorable
  schedule-by-policy interaction; and
- unsupported-scale fallback in more than `10%` of post-cold decisions makes
  predictive classification ineligible, even when the run remains a useful
  diagnostic.

Floor occupancy is interpreted separately for `tracked_q_covariance`, whose
action should decline when movement is omitted. These thresholds classify
practical evidence; they do not define statistical significance.

## Artifact Inheritance Contract

### Authoritative read-only sources

The initial source inventory should include:

- canonical repeated path:
  `cache/mnist_experiment/rotated_mnist/phase4/rotated_mnist_phase4_repeated_path_memory_screen__replica-0001__b85ea66a5617ad67`;
- double-lap challenge:
  `cache/mnist_experiment/rotated_mnist/phase5/double_lap/rotated_mnist_phase5_double_lap_development__replica-0001__e8d49d4599638493`;
- single-lap retry:
  `cache/mnist_experiment/rotated_mnist/phase5/single_lap/rotated_mnist_phase5_slow_single_lap_development__replica-0001__30f2161a06eb8d50`;
- full Plan 6 oracle:
  `cache/mnist_experiment/rotated_mnist/phase6/oracle/rotated_mnist_phase6_oracle_full_path__replica-0001__7c7dc5936d08fb91`; and
- Plan 7 coefficient audit:
  `cache/mnist_experiment/rotated_mnist/phase7/coefficient_audit/rotated_mnist_phase7_anchor_audit_primary_v3__replica-0001__4406a8e7634c0ad8`.

Every source must have a valid `COMPLETED` marker and compatible manifest.
Record source run IDs, configuration hashes, relevant file hashes, schedule
hashes, stream-plan hashes, partition hash, initial-state hash, parameter
layout, Fisher representation, dtype, and runtime metadata in each derived
Plan 8 artifact.

### Permitted reuse

For a source-backed development trajectory, Plan 8 may load:

- the exact initial model state;
- the initial rank-8-plus-diagonal Fisher representation;
- stream tensors, labels, observation identities, and schedule;
- partition and evaluation-panel identities;
- fixed-policy and historical EDR scalar trajectories; and
- exogenous Plan 6 reference quantities at exactly matched angles.

This reuse is experimental blocking, not a substitute for learner execution.
It gives the new treatment the same starting point and observations as its
historical controls.

### Prohibited reuse

For `tracked_q_covariance` and `decomposed_edr`, never load a historical
post-initialization:

- model or optimizer state;
- parameter or displacement trajectory;
- Fisher summary;
- $q_t$, trend, residual, discounted-risk, or controller state;
- action; or
- predictive metric.

Both new conditions must execute every update from the common initial state.
Once their first action differs, all downstream learner and auxiliary-process
state is endogenous and must be recomputed.

### New immutable outputs

Keep Plan 8 code under `mnist_experiment/rotated_mnist/`, tests under
`test/unit/rotated_mnist/`, and artifacts under
`cache/mnist_experiment/rotated_mnist/phase8/`. Use Plan 8-specific schema
versions and run kinds. Write through an incomplete directory and publish a
`COMPLETED` marker only after every required artifact has been validated and
flushed. A completed source or Plan 8 artifact is never repaired or mutated.

The results notebook must load lightweight completed artifacts only. It must
not train, calculate scores or Fishers, download data, resume runs, or repair
missing outputs.

## Frozen Handoff

Unless a phase check-in explicitly reopens an item, preserve:

- the canonical 512-parameter CNN and complete trainable-network likelihood
  Fisher;
- upright all-class initialization with 30,000 observations;
- the 20,000-observation initial Fisher estimate;
- $m=4$ on the single- and double-lap EDR rechallenges, and $m=8$ on
  Plan 5's canonical repeated path;
- 50 L-BFGS inner iterations with the Plan 5 optimizer settings;
- rank-8-plus-diagonal Fisher summaries;
- direct EMA Fisher updates with no LFU or HVP calculation;
- pre-update evaluation and Plan 5 exposure semantics;
- linear and normalized-logistic schedules with their existing angle grids;
- eight cold-start updates at $\pi=.05$;
- $\pi_{\min}=.01$ and $\pi_{\max}=.95$;
- the source-specific trend half-life and eight-update movement half-life;
- $q_0=1/30000$ for source-compatible runs; and
- one realized $\pi_t$ shared by EWC weighting and Fisher updating.

Do not tune half-lives, bounds, optimizer budgets, rank, or batch size during
the principal rechallenge. Any later sensitivity study must be a separately
named treatment selected at a check-in.

## Phase 0: Estimand, Inheritance, And Comparison Contracts

### Goal

Freeze exactly what changes relative to Plan 5 and prove that the proposed
source reuse preserves, rather than weakens, the paired experiment.

### Scope

1. Audit every authoritative source path, schema, hash, required tensor, and
   parameter layout needed by the new runner.
2. Freeze the recommendation hierarchy, temporal indexing, $q_0$, cold start,
   fallback, clipping, and movement-premium equations above.
3. Identify each source condition as contextual, prospective, post-facto, or
   historical and freeze the principal contrasts before seeing new outcomes.
4. Specify Plan 8 configuration, metric, artifact, and source-reference
   schemas without altering a Plan 5--7 loader.
5. Define the exact single-lap, double-lap, and canonical repeated-path source
   compatibility checks.
6. Estimate wall-clock cost from source timings and separate development,
   stress, and confirmatory compute budgets.
7. Add Plan 8 to the rotated-MNIST agent notes without changing the accepted
   Plan 5 schedule or findings.

### Verification

- Every inherited file is read-only and content-hashed.
- The initial model and Fisher can be reconstructed without selecting a
  historical condition's final state.
- New and source schedules have identical point counts, angles, directions,
  streams, labels, and evaluation indexing.
- The plan names no $c/2$ actuation condition.
- The principal contrast and stopping rules are fixed before implementation.

### Check-in

Confirm that the inherited development block is exactly paired and decide
whether the source artifacts are sufficiently complete for a source-backed
runner. Do not implement the controller until this audit passes.

### Execution Record

**Status:** Complete.

All five authoritative paths load as completed artifacts through their strict
historical schemas. The three learner sources expose the exact initial model,
initial rank-8-plus-diagonal Fisher, partitions, evaluation identity, stream
tensors, schedules, scalar metrics, and final provenance needed by a
source-backed runner. Every source uses the canonical 512-parameter CNN and a
20,000-score initial Fisher. The single- and double-lap EDR sources use $m=4$
with 80 and 120 transitions per schedule, respectively; the canonical
six-knot path uses $m=8$ with 100 transitions. These regimes must remain
separate.

The inherited model-state artifacts contain a shared `initial` state for every
historical condition, so Plan 8 need not select or resume a historical learner
branch. Source-backed runs will content-hash the initial state, Fisher,
streams, plans, and evaluation identities. All revised conditions will create
new model, optimizer, Fisher, trend, residual, movement, and $q_t$ state from
that initial boundary. Fixed controls and legacy EDR are read-only comparison
products.

The source-backed development gate is accepted. Numerical compatibility uses
exact discrete identities and initial hashes, plus tolerance-based trajectory
agreement because CUDA adaptive-pooling backward is warning-only rather than
bitwise deterministic. Fresh confirmation remains responsible for independent
initializations and Fishers.

## Phase 1: Decomposed Controller And Source-Backed Runner

### Goal

Implement the two revised treatments without changing historical controller
semantics or any completed runner.

### Scope

1. Add pure functions for the tracked-$q$ recommendation, normalized movement
   premium, movement-only EMA, and decomposed recommendation.
2. Add a separately named decomposed-controller state. Do not reinterpret
   `DiscountedRiskState` or historical `discounted_risk` artifacts.
3. Enforce pre-transition decision timing and post-acceptance updates for
   trend, covariance scale, movement state, and $q_{t+1}$.
4. Implement `tracked_q_covariance` and `decomposed_edr` behind explicit
   condition dispatch, with shared action bounds and cold-start behavior.
5. Build a Plan 8 source-backed runner that loads only permitted Plan 5 assets
   and computes complete new trajectories.
6. Store scalar action reconstructions, full parameter/displacement
   trajectories, final model states, controller states, Fisher/Lanczos
   diagnostics, predictive metrics, costs, and source provenance.
7. Add progress reporting, resumable incomplete runs, finite-value guards, and
   immutable completion behavior.
8. Add a fixed-$\pi=.05$ compatibility sentinel. On the source stream it must
   reproduce the historical algorithm within declared numerical tolerances;
   initial hashes and all discrete identities must match exactly.

### Verification

- Unit tests reproduce $q_{t+1}$, $m_tq_t/(1+m_tq_t)$, the complete
  decomposed formula, and the stationary $c/2$ identity.
- Zero movement reduces decomposed EDR to the tracked-$q$ recommendation.
- Positive movement increases the recommendation monotonically.
- No-current-batch and off-by-one tests prove predictability.
- Cold start, unsupported covariance scale, clipping, and floor behavior are
  explicit and tested.
- Legacy controller tests and artifact loaders remain unchanged and passing.
- A tiny CPU source-backed smoke completes both new conditions and refuses an
  incomplete or incompatible source artifact.

### Check-in

Inspect the first complete action tables and compatibility sentinel. Confirm
that the new runner recomputed every endogenous state and that any difference
from fixed $.05$ is attributable to the declared treatment rather than source
or optimizer drift.

### Execution Record

**Status:** Complete.

The implementation adds pure tracked-$q$, decomposed-risk, and
movement-discount functions while leaving historical `DiscountedRiskState`
and Plan 5 runners unchanged. A separately named movement state initializes at
zero, updates only from predictable pre-transition controller state, and
falls back to the tracked-$q$ action when the covariance scale is unsupported.
The Plan 8 runner loads only the source initial model, initial Fisher, streams,
partitions, and evaluation identities; every condition receives a fresh model,
optimizer, Fisher, controller, movement, and trajectory state.

The immutable CPU smoke artifact is
`cache/mnist_experiment/rotated_mnist/phase8/smoke/rotated_mnist_phase8_source_backed_smoke__replica-0001__91371ae6e8dee7a6`.
It executed three updates for both source schedules and all three development
conditions. Predictability, shared initialization, rank-eight resolution,
finite values, source completion, and no-endogenous-reuse checks passed.
Stored actions and $q_t$ updates reconstructed with zero numerical error. The
partial smoke deliberately did not evaluate the full fixed-policy
compatibility sentinel.

The focused controller and Plan 8 configuration tests report 38 passes. The
smoke's one-update cold start leaves both revised actions at the `.01` floor;
that integration behavior is not used as scientific evidence. The Phase 1
gate accepts the implementation for the complete single-lap rechallenge.

## Phase 2: Paired Single-Lap Rechallenge

### Goal

Test the repaired decomposition on the exact trajectory that exposed the
historical EDR timing and feedback failure.

### Scope

1. Use the completed Plan 5 single-lap artifact as the immutable source for
   initialization, Fisher, streams, schedules, and historical controls.
2. Execute complete 80-update `tracked_q_covariance` and `decomposed_edr`
   trajectories for both linear and normalized-logistic schedules.
3. Execute the fixed $.05$ compatibility sentinel over the complete source
   trajectory unless Phase 1 establishes an equally strong full-trajectory
   equivalence check.
4. Compare new trajectories with loaded `legacy_edr`, fixed $.05$, fixed
   $.025$, and current-only results. Use at most four conditions in any plot.
5. Decompose each new action into $m_tq_t$, instantaneous movement premium,
   discounted movement premium, clipping, and fallback contributions.
6. Compare against compatible Plan 6 and Plan 7 population diagnostics only
   at exact angle and indexing matches.
7. Report whole-path, outward-leg, return-leg, reversal-window, fixed-panel,
   calibration, worst-class, and resource metrics.

### Gate

- **Proceed:** all operational checks pass, decomposition materially reduces
  the legacy action inflation or lag pathology, and `decomposed_edr` improves
  materially over `legacy_edr` without a material regression relative to
  prospective fixed $.05$.
- **Proceed as diagnostic:** the controller is stable and mechanistically
  repaired but remains predictively behind fixed $.05$. Continue only if the
  remaining discrepancy is localized enough for the double-lap stress to
  answer a specific question without retuning.
- **Stop:** the new policy recreates positive feedback, depends primarily on
  clipping or fallback, violates predictability, or remains materially harmful
  with no interpretable improvement over legacy EDR.

This phase is an exactly paired development experiment with one inherited
replica. It cannot establish statistical significance.

### Verification

- Every new condition has 81 parameter points and 80 exact displacement
  identities per schedule.
- Stored actions reproduce exactly from stored pre-transition state.
- No historical post-initialization state enters a new trajectory.
- Evaluation and stream identities match the source artifact.
- The notebook-facing artifact contains every scalar used by the gate.

### Check-in

Classify the result using the decision language above. Decide whether the
double-lap test is scientifically earned and whether both new treatments or
only decomposed EDR should continue.

### Execution Record

**Status:** Complete.

The immutable run is
`cache/mnist_experiment/rotated_mnist/phase8/single_lap/rotated_mnist_phase8_single_lap_rechallenge__replica-0001__6445ab53bb10cd25`.
All trajectories were complete, finite, predictable, rank eight, and free of
historical endogenous state. Stored actions and $q_t$ reconstructed to
floating-point precision. The fixed-$.05$ sentinel reproduced both historical
schedules exactly in the recorded scalar metrics.

The decomposition is a mechanistic repair but not a competitive predictive
policy on this development replica. Relative to legacy EDR, decomposed EDR
improved environmental-accuracy AUC by `.0825` on the linear schedule and
`.0859` on the sigmoid schedule, satisfying the frozen repair threshold. It
nevertheless trailed prospective fixed $.05$ by `.1202` and `.0368`, with NLL
AUC regressions of `6.50` and `3.92`; the principal damage was concentrated on
the return leg. Its mean actions, `.1706` and `.1448`, remain too large despite
being substantially below legacy EDR's `.2765` and `.3000`.

The covariance-only ablation was schedule-dependent: it trailed fixed $.05$
on the linear schedule but exceeded it by `.1504` accuracy-AUC on the sigmoid
schedule while using actions near `.011`. This is informative rather than a
claim of dynamic value. Phase 2 therefore takes the predeclared **proceed as
diagnostic** branch. Phase 3 retains both revised treatments to test whether
the observed return-leg failure worsens under repeated reversals; no
hyperparameter is changed.

## Phase 3: Double-Lap Reversal Stress

### Goal

Determine whether any single-lap improvement survives faster repeated
reversals without recreating the historical delayed-release failure.

### Scope

1. Use the completed Plan 5 double-lap artifact as the read-only source.
2. Recompute the promoted Plan 8 condition from the common initial state over
   all 120 updates of each schedule. Retain `tracked_q_covariance` only if its
   Phase 2 behavior remains scientifically informative.
3. Compare with loaded legacy EDR and fixed
   $\pi\in\{.025,.05,.075,.10\}$ trajectories.
4. Report action, movement, accuracy, NLL, calibration, fixed-panel retention,
   and worst-class trajectories separately around each reversal.
5. Measure action lag, carryover across reversal, time above the tested fixed
   bracket, and weak-EWC/large-displacement feedback.
6. Do not change the source trend half-life or movement half-life to rescue a
   failed stress response.

### Gate

- **Generalize:** the repaired controller remains finite, predictable, and
  materially safer than legacy EDR, with no reversal-region collapse relative
  to fixed $.05$.
- **Slow-controller boundary:** single-lap value survives but double-lap value
  does not. Retain the method as a slow online tuner and do not claim rapid
  change-point response.
- **Stop:** the decomposition does not prevent the historical feedback and
  reversal pathology.

### Check-in

Decide whether Plan 8 supports a slow-tuning claim, a dynamic-controller
claim, only a diagnostic improvement, or no useful adaptive result. Do not
move to the canonical applied path merely because the stress run completed.

### Execution Record

**Status:** Complete.

The immutable stress run is
`cache/mnist_experiment/rotated_mnist/phase8/double_lap/rotated_mnist_phase8_double_lap_reversal_stress__replica-0001__f035d4b8cdf6d894`.
Both 120-update schedules were complete, finite, predictable, rank eight, and
exactly reconstructable from stored pre-transition controller state.

The decomposed controller did not survive repeated reversals. On the linear
schedule its environmental-accuracy AUC was `.1916`, below legacy EDR's
`.2728` and fixed $.05$'s `.6314`, while NLL AUC rose to `361.30`. On sigmoid
it improved legacy EDR by `.2067` accuracy-AUC, but remained `.0738` below
fixed $.05$ with an NLL-AUC regression of `2.14`. Its lagged speed/action
association was absent on linear and negative on sigmoid. Decomposed EDR thus
fails the generalization gate and recreates harmful high-action behavior under
the linear stress.

The tracked-$q$ covariance ablation remained stable near the `.01` floor. It
exceeded fixed $.05$ by `.0298` accuracy-AUC on linear and trailed it by
`.0493` on sigmoid. This supports its role as a low-noise covariance-risk
component, but its negligible action variation cannot establish movement
adaptation. The result is classified as a validated decomposition with a
failed online movement controller, not as a dynamic-policy success.

## Phase 4: Canonical Repeated-Path And Applied Hybrid Gate

### Goal

Evaluate a promoted decomposed controller on Plan 5's principal
$0\to15\to30\to0\to15\to30$ path and determine whether it deserves an applied
Hybrid test.

### Scope

1. Use the completed Plan 5 Phase 4 repeated-path artifact as the immutable
   source for initialization, stream, fixed $.05$, current-only, Replay B32,
   Hybrid B32, and unbounded-replay context.
2. Run a new fixed-$\pi=.025$ EWC condition on the same source because that
   condition is absent from the original repeated-path artifact.
3. Run the promoted decomposed EDR condition as EWC-only over all 100 updates.
4. Preserve the original six-knot, five-leg schedule, angular progression,
   $m=8$,
   evaluation frequency, optimizer budget, and Fisher representation.
5. Compare first exposure, return, second ascent, matched-angle revisit,
   fixed-panel retention, calibration, and realized compute.
6. Only if EWC-only decomposed EDR has prospective or automatic-selection
   value, add a separately named Hybrid B32 condition in which the same
   realized action controls the EWC objective and archive Fisher update while
   replay follows the accepted Plan 3 clean recursion.
7. Do not treat Replay or unbounded replay as alternative estimators of
   $\pi_t$; they remain applied resource controls.

### Gate

- **Promote EWC-only:** decomposed EDR improves prospective fixed $.05$ or
  approaches the post-facto fixed benchmark without material retention,
  calibration, or cost regression.
- **Promote Hybrid:** the EWC-only gate passes and adaptive Hybrid adds useful
  predictive value over fixed Hybrid B32 without destabilizing replay/archive
  semantics.
- **Diagnostic only:** action information is coherent but does not improve the
  applied quality-cost frontier.
- **Stop:** revised EDR remains materially worse than fixed EWC on the
  canonical path.

### Check-in

Choose the smallest scientifically sufficient set of conditions and schedules
for independent confirmation. Freeze all confirmatory configurations before
adding replicas.

### Execution Record

**Status:** Skipped by the Phase 3 gate.

The decomposed EWC-only controller was materially worse than fixed EWC under
the required double-lap stress and showed no favorable speed response. The
canonical repeated-path run could not promote it, and adaptive Hybrid was
therefore not implemented or executed. The accepted Plan 3 fixed-policy
Hybrid result remains unchanged.

## Phase 5: Fresh Independent Confirmation

### Goal

Replace one-replica development evidence with paired uncertainty across fresh,
independent learning trajectories.

### Scope

1. Generate a fresh early-stopped upright fit and high-quality initial Fisher
   for every replica. Do not reuse the Plan 5 initializer or Fisher across
   independent replicas.
2. Within each replica, pair all promoted conditions on initialization,
   Fisher, stream identities, schedule, optimizer settings, evaluation panel,
   and non-treatment seeds.
3. At minimum rerun prospective fixed $.05$, post-facto fixed $.025$, legacy
   EDR, and decomposed EDR. Retain the covariance-only ablation only if Phase 2
   identifies a confirmatory scientific contrast.
4. Use 30,000 initialization observations, a 20,000-score initial Fisher,
   50 L-BFGS iterations, and rank-8-plus-diagonal direct-EMA summaries for
   every replica. Preserve the selected source design's batch size: $m=4$ for
   a single- or double-lap confirmation and $m=8$ for the canonical repeated
   path. Do not pool contrasts across different batch sizes.
5. Begin with eight independent replicas. Extend in blocks of four through
   `--resume` until the two-sided 95% paired interval for the principal
   environmental-accuracy AUC contrast has half-width at most `.02`, or until
   32 replicas complete. The stopping rule depends on precision, not the sign
   or significance of the observed effect.
6. Preserve all per-replica trajectories and calculate paired intervals for
   whole-path and leg-specific accuracy, NLL, calibration, worst-class recall,
   fixed-panel retention, and cost.
7. Record initialization quality and initial-Fisher diagnostics per replica.
   A failed replica remains an immutable failed artifact and must not be
   silently replaced under the same replica ID.
8. Provide resumable command-center execution with `tqdm`, elapsed time,
   projected completion, and completed/remaining replica counts.

### Verification

- Replica seeds produce fresh initial fits, Fishers, partitions, and streams.
- Conditions within each replica remain exactly paired.
- Confidence intervals use independent replicas as the sampling units, never
  individual steps.
- Sequential extension follows the precision-only rule.
- No development artifact is included as an independent confirmatory replica.
- The final run ledger accounts for every completed, failed, and excluded
  configuration.

### Check-in

Decide whether decomposed EDR has predictive, automatic-selection,
prospective, diagnostic-only, or negative evidence. Separate conclusions about
the covariance decomposition from conclusions about online movement
estimation.

### Execution Record

**Status:** Skipped by the Phase 3 gate.

No adaptive treatment qualified for confirmatory promotion, so spending fresh
initializations and Fishers could not answer the predeclared confirmatory
question. Plan 8 makes no population-level predictive claim. The two paired
development artifacts remain explicitly one-replica evidence.

## Phase 6: Artifact-Only Findings And Evidence Review

### Goal

Make the corrected experiment legible while preserving negative results and
the historical chain of evidence. Stop for user review before changing any
mathematical theory or project-level conclusion.

### Scope

1. Create
   `mnist_experiment/rotated_mnist/decomposed_edr_results.ipynb` as a strictly
   artifact-only notebook.
2. Begin with the Plan 7 coefficient result: tracked $q_t$ is the validated
   covariance estimator, while $c/2$ is its stationary approximation rather
   than an actuated policy.
3. Show simple paired plots for legacy versus decomposed EDR actions, exact-$q$
   versus movement contributions, predictive trajectories, reversal behavior,
   fixed-panel retention, and realized cost.
4. Keep development and confirmatory evidence visibly separate. Display
   paired means and uncertainty only when independent replicas exist.
5. Link every conclusion to immutable run IDs and a compact table of source
   hashes, condition semantics, sample sizes, and runtime.
6. State plainly whether the experiment validates the covariance repair,
   movement estimator, predictive controller, applied Hybrid strategy, or only
   a diagnostic decomposition.
7. End with an explicit evidence table listing supported, unsupported, and
   unresolved claims, together with the artifact and metric behind each
   classification.
8. Propose, but do not make, any changes to `mathematical_overview.ipynb` or
   `mnist-findings.ipynb` warranted by the results.

### Verification

- Every notebook cell performs lightweight artifact loading, arithmetic, or
  plotting only.
- Every displayed action is reconstructable from stored pre-transition state.
- Principal plots contain no more than four conditions and use uncertainty
  across replicas where available.
- Fixed $.025$ is consistently labelled post-facto and fixed $.05$
  prospective.
- Negative and boundary results remain visible rather than being overwritten
  by the revised treatment.
- Neither `mathematical_overview.ipynb` nor `mnist-findings.ipynb` changes in
  this phase.

### Check-in

Review the complete evidence and decide whether decomposed EDR should be
promoted as an online recommendation, retained as a diagnostic research
direction, restricted to slow environments, or retired. Approve, revise, or
reject the proposed mathematical and project-level interpretation before
Phase 7 begins.

### Execution Record

**Status:** Complete.

The artifact-only review is
`mnist_experiment/rotated_mnist/decomposed_edr_results.ipynb`. It loads strict
completed Plan 5, Plan 7, and Plan 8 artifacts and performs only lightweight
tabulation and plotting. It includes provenance, the Plan 7 tracked-$q$
coefficient result, legacy/decomposed/covariance-only action trajectories,
movement decomposition, environmental-accuracy trajectories, fixed upright
panel retention, costs, operational identities, and an explicit evidence
ledger. No independent-replica uncertainty is displayed because Phase 5 was
correctly skipped.

The notebook classifies the evidence as **validated covariance decomposition,
failed tested movement controller**. It proposes no promotion of decomposed
EDR or adaptive Hybrid. It leaves adaptive composition and better predictable
movement estimation open, and it makes no changes to
`mathematical_overview.ipynb` or `mnist-findings.ipynb`. All nine code cells
pass the artifact-only validator in under four seconds in the development
environment, and all four figures were inspected from a noninteractive render.
Phase 7 remains pending user review.

## Phase 7: Theory And Project-Findings Integration

### Goal

Integrate only the interpretation explicitly approved at the Phase 6
check-in. Preserve a clean boundary between empirical evidence, mathematical
assumptions, and conclusions that remain open.

### Scope

1. Update `mathematical_overview.ipynb` only if the reviewed evidence changes
   the status of decomposed EDR or clarifies the relationship among
   $\pi_t^{\mathrm{cov},q}$, $c/2$, the movement premium, and historical EDR.
2. Preserve the historical EDR construction and identify decomposed EDR as a
   separately named applied estimator. Do not retroactively reinterpret old
   algorithms or artifacts.
3. Put the concise estimand and any experimentally supported conclusion in the
   main body. Keep controller initialization, smoothing, clipping, fallback,
   and numerical diagnostics in the appendix.
4. Label covariance calibration, equal covariance shape, and online movement
   estimation as assumptions unless Plan 8 directly validates them.
5. Update `mnist-findings.ipynb` only if independent confirmation supports an
   applied finding. Keep development-only improvements in the Plan 8 results
   notebook.
6. Record negative, scope-limited, and unresolved results explicitly. Do not
   convert mechanistic repair into a predictive or optimal-control claim.
7. Amend the rotated-MNIST agent notes only where needed to preserve the final
   accepted semantics for future experiments.

### Verification

- Every new theoretical or applied statement links back to the Phase 6
  evidence table and an immutable artifact.
- Theoretical identities remain distinguishable from empirical estimators and
  implementation heuristics.
- Historical notebook claims and artifact names retain their original
  meanings.
- All edited notebooks parse as strict JSON and execute their lightweight
  cells without training, score calculation, or artifact repair.

### Final Check-in

Review the integrated theory and project-level findings. Any subsequent
half-life, bound, batch-size, rank, or change-point experiment requires a new
versioned plan rather than retrospective tuning of Plan 8.
