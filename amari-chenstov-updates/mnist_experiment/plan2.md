# Implementation Plan 2: Low-Data Regime Discovery

This plan identifies a statistically and computationally meaningful small-data
regime before the project launches a broader replay, LFU, and resource-efficiency
study. It follows the phase-gate style of [plan1.md](plan1.md): each phase
produces working software or scientific evidence, then pauses for review before
the next phase begins.

Plan 2 is intentionally preparatory. Its final product is a justified target
regime and a frozen comparison package for a subsequent Plan 3. It is not
supposed to prove the full continual-learning proposal by itself.

## Motivation

The intended application is continual adaptation of a high-quality pretrained
model using small batches and modest hardware. The compressed EWC summary is
only valuable when current observations alone are insufficient. If the
current-batch objective with $\pi_t=1$ learns the new task reliably, the
experiment has not yet entered the regime that motivates retaining old
information through

$$
(\widehat\theta_t,\widehat{\mathcal I}_t).
$$

The Phase 9 adaptation screen used $m=128$ observations at each of 100 points
along the path $p=0,\ldots,1$. Its no-EWC condition learned digit 9 faster than
the EWC conditions. It crossed 90% expected accuracy conditioned on digit 9
after about 249 digit-9 presentations, while also receiving thousands of
non-nine observations over the complete gradual trajectory. Although its NLL
and Fisher tracking were much worse, it remained an effective classifier.

The principal conclusion is therefore not that $\pi=1$ is desirable. It is
that $m=128$ supplies too much current information, including substantial
implicit rehearsal of digits 0 through 8, to isolate the intended use case.

Plan 2 will answer the narrower question:

> At what online data budget does a learner using only the current batch cease
> to adapt adequately, while an EWC learner can still exploit its compressed
> historical information?

Only after locating that regime should the project spend substantial compute
on LFU corrections, replay buffers, memory matching, or deployment claims.

## Phase-gate protocol

Every phase ends with a check-in containing:

1. a concise implementation or experiment summary;
2. the exact tests and commands run;
3. links to representative immutable artifacts or notebook views;
4. observed numerical and predictive behavior;
5. deviations from this plan;
6. unresolved risks and proposed adjustments;
7. an explicit recommendation to continue, revise, repeat, branch, or stop.

Do not begin the next phase until the current phase is reviewed and explicitly
accepted. Long GPU runs remain user-controlled and should be launched with a
documented command suitable for `tmux`.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Regime and measurement contracts | Pending |
| 1 | Nested low-data streams and pipeline hardening | Pending |
| 2 | Coarse sample-size boundary screen | Pending |
| 3 | Boundary selection and focused bracketing | Pending |
| 4 | Controller recalibration in the selected regime | Pending |
| 5 | Confirmatory target-regime replication | Pending |
| 6 | Plan 3 handoff package | Pending |

## Governing experimental design

### Processes and estimand

Preserve the scientific contract from [AGENTS.md](AGENTS.md). The environmental
path moves the unique likelihood Fisher through

$$
p_t\longmapsto\theta^\star(p_t)
\longmapsto\mathcal I(\theta^\star(p_t)).
$$

The original process chooses the model displacement through optimization. The
auxiliary process maintains the Fisher summary using that realized
displacement. Varying the online batch size $m$ changes available data; it does
not redefine the Fisher estimand.

### Initial conditions

Every Plan 2 condition begins with:

- the same high-quality, independently fitted $p=0$ model within a replica;
- the same high-quality initial Fisher summary within a replica;
- the same MNIST partitions and holdout observations within a replica;
- the same optimizer settings except where an explicit budget check changes
  them;
- separately named and recorded random seeds.

The high-quality $p=0$ state represents the expensive pretrained model in the
motivating application. Plan 2 studies low-data adaptation after that state; it
does not weaken pretraining merely to manufacture a treatment effect.

### Principal conditions

The first sample-size screen uses three rank-8-plus-diagonal, $K=50$ conditions.
All use the no-LFU Fisher recursion so the screen isolates the value of old
information and the controller rather than the quality of a derivative
correction.

| Condition | Applied $\pi_t$ | EWC penalty | Purpose |
|---|---:|---|---|
| `no-ewc-pi100` | $1$ | Zero | Primary current-batch-only control |
| `fixed-ewc-pi010` | $0.10$ | Fixed old-to-new odds | EWC mechanism control without controller-estimation risk |
| `adaptive-ewc-h010` | Plug-in, clipped to $[0.05,0.95]$ | Adaptive | Applied controller treatment using the best Phase 9 default |

The fixed value $0.10$ is a diagnostic control, not a claim of optimality. It
separates a failure of the EWC summary from a failure of the plug-in controller.
The controller half-life $h=0.10$ is also only a starting value: Phase 9 chose
it at $m=128$, whereas the preferred memory may change when observation noise
increases.

All three conditions continue to calculate the same diagnostics during the
screen. No-EWC runtime is therefore experimental runtime, not an optimized
deployment benchmark.

### Sample-size axis

Hold the 100-point linear $p$ path fixed during the initial boundary search and
screen

$$
m\in\{1,2,4,8,16,32,64\}.
$$

Reuse the completed $m=128$ screen as the easy-regime anchor for every metric
its schema supports. Do not rerun it solely to populate newly added optional
diagnostics.

Fixing the path while varying $m$ intentionally changes both observations per
update and total online observations. With 100 path points, a condition sees
$100m$ observations in total and, in expectation under the linear mixture,
approximately $50m$ digit-9 and $50m$ non-nine observations. This is the data
scarcity intervention of interest.

The coarse screen begins with three paired replicas per cell. Add replicas only
after weak values of $m$ have been discarded. This is a boundary search, not a
large confirmatory factorial experiment.

### Nested stream contract

Within a replica, all Plan 2 values of $m$ should be deterministic prefixes of
one common master batch at every $p$ step. The existing 128-observation Phase 9
stream can serve as that master when compatible. For example, the `m=8` batch
contains the first eight ordered observations from the corresponding master
batch.

This construction provides:

- exact condition pairing within each $m$;
- paired comparisons across values of $m$;
- one shared initialization rather than a new early-stopped $p=0$ fit for
  every data budget;
- a transparent relationship between added data and changed outcomes.

Derived stream artifacts must record the parent stream hash, prefix rule,
derived content hash, and requested $m$. They remain immutable. The prefix is
statistically valid because each master batch is generated as an ordered IID
mixture sample.

### Primary outcomes

Treat replicas, not trajectory steps, as the statistical units. Preserve each
trajectory and summarize expected trajectories across replicas.

Primary adaptation outcomes are:

1. expected accuracy conditioned on true digit 9 versus cumulative total
   observations;
2. expected accuracy conditioned on true digit 9 versus cumulative digit-9
   presentations and unique digit-9 observations;
3. the first durable crossing of 90% by the expected digit-9 accuracy
   trajectory, when it exists;
4. digit-9 accuracy AUC over $p$ and over cumulative observations;
5. expected non-nine accuracy and forgetting at matched observation budgets;
6. expected environment-weighted accuracy.

"Durable" means that the expected trajectory remains at or above the threshold
for all later recorded points. If the threshold is never reached, report that
fact and compare accuracy at common fixed exposure budgets. Do not average
per-replica stopping times as the principal estimand.

Secondary outcomes are:

- digit-9, non-nine, and environment-weighted NLL;
- balanced accuracy and calibration;
- pointwise trajectory variability and paired uncertainty;
- Fisher tracking and controller diagnostics;
- optimizer iterations and function evaluations;
- wall time by operation, score-gradient count, HVP count, peak memory, and
  artifact size.

Record actual per-step and cumulative class counts and unique observation
counts. Do not substitute $mp_t$ for observed digit-9 exposure in reported
sample-efficiency results.

Because the $m=128$ screen exposed disagreement between accuracy and NLL, add
a lightweight multiclass calibration summary, preferably Brier score and a
fixed-bin expected calibration error, when logits are already evaluated. Keep
NLL as the primary proper scoring rule. Existing runs without the new fields
must remain readable.

Process wall time is not automatically "GPU-seconds." Label resource metrics
according to what is actually measured; add CUDA-event or sampled-utilization
instrumentation before making accelerator-consumption claims.

### Operational definition of the target regime

A value of $m$ is a target-regime candidate only when the paired expected
trajectories support both statements:

1. `no-ewc-pi100` no longer provides adequate digit-9 acquisition or stability
   at the available observation budget; and
2. at least one EWC condition obtains a practically meaningful advantage from
   its compressed historical summary without unacceptable loss of old-task
   performance.

Poor no-EWC NLL alone does not satisfy the first statement if the no-EWC
classifier still learns digit 9 effectively. Conversely, a setting where all
conditions fail is merely too data-starved; it does not demonstrate useful
information compression.

Use effect sizes, expected trajectories, paired differences, and uncertainty
together. Do not select $m$ from a single noisy threshold crossing or from a
significance test alone. The Phase 2 and Phase 3 check-ins make the practical
decision with the user rather than encoding a hidden automatic winner rule.

### Planned branches

The boundary search must be allowed to falsify its own premise:

- If no-EWC succeeds at high $m$ but fails while an EWC condition succeeds at
  lower $m$, bracket that transition.
- If fixed EWC succeeds but adaptive EWC fails, retain the candidate $m$ and
  treat controller calibration as the next problem.
- If both EWC conditions and no-EWC fail, move upward in $m$ or increase total
  observations before changing Fisher mechanics.
- If no-EWC remains effective even at $m=1$, batch size cannot move the current
  design further into the target regime. Open a corrective branch that changes
  trajectory geometry or implicit rehearsal, such as fewer $p$ steps, a
  steeper transition, or an abrupt removal of old-digit observations.
- If conclusions depend primarily on optimizer budget, resolve that compute
  confound before selecting a statistical regime.

Do not launch a full replay or LFU grid from an unresolved branch.

## Phase 0: Regime and measurement contracts

### Goal

Make the target-regime decision auditable before adding new computational
conditions.

### Scope

1. Add a concise Plan 2 conditions document or a clearly separated section in
   `EXPERIMENTAL_CONDITIONS.md` defining the three principal conditions, nested
   stream design, and target-regime gate.
2. Add artifact-only analysis helpers for:
   - cumulative total, digit-9, non-nine, and unique observation counts;
   - expected post-update trajectories aligned by step, $p$, and exposure;
   - durable expected-trajectory threshold crossings;
   - paired AUC and fixed-budget contrasts;
   - pointwise standard deviations or confidence intervals across replicas.
3. Add calibration metrics only where they can be computed during the existing
   holdout forward pass without storing logits in notebook artifacts.
4. Version any affected metric or artifact schemas while preserving strict
   readers for completed Plan 1 runs.
5. Add notebook panels that state the number of complete replicas and fail
   clearly when a requested condition or metric is absent.
6. Document which quantities are measured compute costs and which remain rough
   proxies.

### Verification

- Unit tests cover cumulative counts, repeated observation IDs, missing
  crossings, durable crossings, nonmonotone expected trajectories, and paired
  aggregation.
- An old Phase 9 bundle still loads and contributes all compatible metrics.
- Notebook validation performs no training, data download, model loading, or
  Fisher calculation.
- The written target-regime rule is understandable without reading source
  code.

### Check-in decision

Approve the outcome hierarchy and the operational meaning of "no-EWC works"
before constructing the low-data streams.

## Phase 1: Nested low-data streams and pipeline hardening

### Goal

Support very small online batches without changing initialization, pairing, or
the immutable-run contract.

### Scope

1. Implement a versioned derivation path that creates an immutable low-$m$
   replica bundle from a validated master replica bundle by:
   - reusing the exact model state, partitions, initialization metrics, and
     initial Fisher provenance;
   - truncating each ordered stream batch to the requested prefix length;
   - validating labels against retained observation IDs;
   - recording parent and derived hashes.
2. Ensure one master initialization can be referenced by every Plan 2 value of
   $m$ without refitting the $p=0$ model.
3. Make the command center prepare the three principal conditions for an
   explicit list of $m$ values without constructing an unintended Cartesian
   product.
4. Reuse one validated per-replica reference-optimum path across all values of
   $m$ when its mathematical inputs are unchanged. Validate the $p$ grid,
   partition, model state, and path hash explicitly.
5. Harden the optimization and derivative pipeline for $m\in\{1,2\}$:
   - singleton per-sample gradients retain their sample dimension;
   - NLL means and Fisher estimates have consistent scaling;
   - L-BFGS guards remain finite;
   - diagonal and rank-8-plus-diagonal updates remain PSD;
   - controller moment updates do not divide by zero.
6. Add a tiny CPU smoke bundle and a short CUDA smoke command covering all
   three conditions at `m=1` and `m=2`.
7. Update cost previews using observed smoke timing rather than scaling linearly
   from $m=128$ when fixed overhead dominates.

### Verification

- Derived streams are exact prefixes of their recorded parent streams.
- Every value of $m$ shares the same model-state and partition hashes within a
  replica.
- All three conditions at one $m$ share the same stream hash.
- Different $m$ values have the documented nested relationship.
- Completed parent bundles and runs remain unchanged.
- Unit tests and the tiny CPU smoke pass; the short CUDA smoke has finite
  losses, parameters, controller values, and Fisher diagnostics.

### Check-in decision

Inspect numerical behavior and measured per-run cost. Approve or revise the
coarse $m$ grid and initial replica count before starting the long screen.

## Phase 2: Coarse sample-size boundary screen

### Goal

Locate where the no-EWC learner begins to lose practical effectiveness.

### Scope

1. Prepare immutable paired runs for the three principal conditions at
   $m\in\{1,2,4,8,16,32,64\}$ using replicas 1 through 3.
2. Reuse compatible $m=128$ evidence as the easy endpoint.
3. Preview expected GPU time and disk use, then provide one resumable `tmux`
   command. The user controls execution.
4. After completion, produce artifact-only summaries of:
   - expected digit-9 acquisition trajectories;
   - actual total and digit-9 exposure;
   - non-nine retention;
   - accuracy/NLL/calibration disagreement;
   - controller $\pi_t$ behavior;
   - Fisher error and numerical interventions;
   - optimizer and resource costs.
5. Compare conditions only within paired replica and $m$ cells. Across $m$,
   exploit nested streams but keep replica-level uncertainty visible.

### Verification

- Every requested run is complete or explicitly identified as incomplete.
- No condition has silently changed representation, optimizer budget, path
  grid, initialization, or Fisher-update method.
- Expected-trajectory calculations use post-update accuracy for acquisition and
  actual observed class counts for exposure.
- Before-update trajectory metrics remain available for online predictive
  evaluation.
- Any optimizer failure, bound saturation, or nonfinite diagnostic is shown
  before scientific ranking.

### Check-in decision

Classify the result as one of:

- a visible no-EWC/EWC transition worth bracketing;
- universal failure at the smallest budgets;
- continued no-EWC success down to `m=1`;
- an optimizer or controller confound that prevents interpretation.

Do not tune $h$, $\pi_{\min}$, LFUs, or replay before making this decision.

## Phase 3: Boundary selection and focused bracketing

### Goal

Choose one defensible target $m$, or establish that batch size alone cannot
produce the target regime.

### Scope

1. If Phase 2 shows a transition, add only the missing neighboring batch sizes
   needed to bracket it. Avoid rerunning completed powers-of-two cells.
2. Increase the transition bracket to at least five paired replicas.
3. At each candidate $m$, compare:
   - expected digit-9 accuracy at common exposure budgets;
   - durable 90% crossing or failure to cross;
   - digit-9 and environment AUC;
   - non-nine retention;
   - paired variability and worst-replica behavior;
   - NLL and calibration;
   - compute consumed before matched predictive quality.
4. Run a focused $K\in\{20,50,100\}$ budget check only if optimization
   diagnostics or rankings indicate that $K=50$ controls the conclusion.
5. If no-EWC still works at `m=1`, replace the focused $m$ bracket with a small
   corrective path-geometry screen. Hold the total observation accounting
   explicit and vary one of:
   - number of $p$ steps;
   - transition steepness;
   - delayed or abrupt removal of old-digit observations.

Do not vary these axes together. The check-in should choose the smallest
scientifically interpretable intervention.

### Verification

- The selected candidate is not based solely on NLL degradation.
- At least one EWC condition remains useful where no-EWC becomes inadequate.
- The candidate is not simply a setting where every learner fails.
- Optimizer budget does not reverse the principal conclusion, or its role is
  explicitly incorporated into the selected regime.
- Selection rationale, uncertainty, and rejected candidates are recorded.

### Check-in decision

Choose exactly one of:

- accept a target $m$ and proceed to controller recalibration;
- accept a target path-geometry regime and proceed;
- gather additional replicas in a narrow bracket;
- revise the original learning process because no useful EWC regime was found;
- stop Plan 2.

## Phase 4: Controller recalibration in the selected regime

### Goal

Determine whether the plug-in controller can use the selected low-data regime
as effectively as a simple fixed EWC weight.

### Scope

1. Freeze the selected data regime and retain no-LFU Fisher updates.
2. Use `fixed-ewc-pi010` and `no-ewc-pi100` as controls.
3. Screen a compact controller trend half-life set centered on the Phase 9
   default, initially

   $$
   h\in\{0.05,0.10,0.20,0.40\}.
   $$

4. Treat the current implementation honestly: changing $h$ changes both trend
   responsiveness and cold-start duration. Do not claim that this screen
   isolates those mechanisms. Do not add a separate cold-start hyperparameter
   unless the check-in identifies that coupling as a material blocker.
5. Inspect $\pi_t$ distributions, bound activation, effective size, trend
   coherence, trace calibration, acquisition, retention, and variability.
6. Add a minimal $\pi_{\min}$ sensitivity only if the selected controller is
   pinned to a bound often enough to obscure the half-life comparison.
7. Preserve fixed-$\pi$ performance as the mechanism benchmark. An adaptive
   controller is not promoted merely because its formula is more elegant.

### Verification

- Controller comparisons use identical streams, initialization, Fisher method,
  representation, and optimizer budget.
- The chosen controller improves or closely matches the fixed EWC control on
  the agreed expected-trajectory outcomes.
- Trend and covariance diagnostics are sufficiently stable to interpret
  $\pi_t$; failures remain visible.
- Added controller factors remain axial rather than Cartesian.

### Check-in decision

Select one applied EWC controller for confirmation. If no adaptive condition
matches the fixed control, carry the fixed policy into Phase 5 and record the
plug-in controller as unresolved rather than forcing it into Plan 3.

## Phase 5: Confirmatory target-regime replication

### Goal

Verify that the selected low-data regime and EWC advantage are reproducible
enough to justify the broader Plan 3 experiment.

### Scope

1. Freeze a minimal confirmatory package containing:
   - `no-ewc-pi100`;
   - `fixed-ewc-pi010`;
   - the selected applied EWC controller, if distinct;
   - no LFU;
   - rank-8 plus diagonal;
   - the selected $m$, path design, and optimizer budget.
2. Accumulate at least five paired replicas, adding more only when uncertainty
   prevents a practical decision.
3. Report mean trajectories and paired effects with uncertainty, not just
   endpoints.
4. Quantify:
   - observations and unique digit-9 examples needed for useful adaptation;
   - retained non-nine performance at matched exposure;
   - accuracy, NLL, and calibration trade-offs;
   - optimizer work and measured resource costs;
   - Fisher-summary quality even though LFUs remain disabled.
5. Run one artifact-only reproducibility audit linking every aggregate to its
   immutable run, configuration hash, replica seed, initialization hash, and
   stream provenance.

### Verification

- The no-EWC control is genuinely inadequate under the agreed practical
  criterion.
- At least one EWC condition provides a repeatable practical advantage.
- The advantage is not created by different data, initialization, optimizer
  effort, or oracle actuation.
- Predictive conclusions survive inspection of NLL, calibration, and retention.
- The notebook remains lightweight and reports incomplete or incompatible
  artifacts explicitly.

### Check-in decision

Either accept the regime as the experimental foundation for Plan 3, request a
targeted additional replication, revise the regime, or conclude that the
current EWC paradigm has not earned a broader experiment.

## Phase 6: Plan 3 handoff package

### Goal

Turn the accepted regime into a concise, executable specification for the main
application-motivated comparison.

### Scope

1. Record the accepted:
   - data regime and path geometry;
   - controller or fixed $\pi$ policy;
   - optimizer budget;
   - rank-8-plus-diagonal representation;
   - replica and uncertainty policy;
   - primary predictive and cost outcomes.
2. Freeze reusable initialization, nested stream, and control-run identities.
3. Document failed or rejected Plan 2 conditions so Plan 3 does not reopen
   them without new evidence.
4. Propose a reduced Plan 3 condition set covering:
   - no EWC;
   - selected EWC without LFU;
   - full LFU and any justified AC-only diagnostic;
   - replay ceilings;
   - memory-matched replay;
   - EWC plus bounded replay, where observations leaving the replay buffer are
     compressed into the EWC summary;
   - realistic deployment diagnostics without oracle-path assistance;
   - compute, latency, accelerator, and memory accounting.
5. Keep Plan 3 factors staged. In particular, select replay budget before
   crossing it with LFU, controller, representation, and optimizer axes.
6. Update the command-center documentation with only the accepted commands and
   a short human-readable condition table. Put detailed implementation notes in
   an explicitly agent-facing document if needed.

### Verification

- A fresh preview can calculate Plan 3's proposed run count, GPU time, and disk
  use without creating artifacts.
- Every proposed Plan 3 comparison has a named control and a clear estimand.
- The handoff distinguishes measured findings from assumptions and deferred
  questions.
- No long Plan 3 experiment starts as part of this phase.

### Final check-in

Review and approve the Plan 3 scientific question, reduced condition sequence,
compute envelope, and first stopping gate before authoring or executing Plan 3.

## Explicitly deferred to Plan 3

Unless a Plan 2 check-in finds one necessary for regime identification, defer:

- full LFU and Amari-Chentsov-only treatment grids;
- replay-buffer and EWC-plus-replay implementations;
- memory-matched comparisons;
- broad Fisher-rank or representation searches;
- dense-Fisher confirmation;
- deployment-mode removal of expensive diagnostics;
- alternative neural architectures;
- robotics, language-model, or reinforcement-learning demonstrations;
- claims about optimal sample, memory, or compute efficiency.

Plan 2 succeeds by finding and validating the right experimental neighborhood,
not by exhausting every mechanism inside it.
