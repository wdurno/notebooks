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
| 0 | Regime and measurement contracts | Complete |
| 1 | Nested low-data streams and pipeline hardening | Complete |
| 2 | Coarse sample-size boundary screen | Complete |
| 3 | Boundary selection and focused bracketing | Complete |
| 4 | Controller recalibration in the selected regime | Complete |
| 5 | Confirmatory target-regime replication | Complete |
| 6 | Plan 3 handoff package | Complete |

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

| Condition | Original learning process | Auxiliary Fisher process | Purpose |
|---|---|---|---|
| `no-ewc-pi100` | Current batch only; $\pi=1$ removes EWC | Instantaneous empirical Fisher $Z_t$ after initialization | Primary control |
| `fixed-ewc-pi010` | Current batch plus fixed EWC, $\pi=.10$ | Recursive Fisher summary | EWC mechanism control without controller-estimation risk |
| `adaptive-ewc-h010` | Current batch plus adaptive EWC, $\pi_t\in[.05,.95]$ | Recursive Fisher summary | Applied controller treatment using the best Phase 9 default |

The fixed value $0.10$ is a diagnostic control, not a claim of optimality. It
separates a failure of the EWC summary from a failure of the plug-in controller.
The controller half-life $h=0.10$ is also only a starting value: Phase 9 chose
it at $m=128$, whereas the preferred memory may change when observation noise
increases.

All three conditions continue to calculate the same diagnostics during the
screen. No-EWC runtime is therefore experimental runtime, not an optimized
deployment benchmark. The no-EWC learner and its instantaneous empirical
Fisher diagnostic are two distinct memoryless processes. Its rank-one
$m=1$ diagnostic must not be interpreted as the learner retaining only one
parameter direction or as a failed rank-8 approximation.

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

Primary predictive outcomes are the following four trajectories:

1. digit-9 one-vs-rest accuracy $A_{9,\mathrm{OvR}}(p_t)$;
2. digit-9 precision at the environmental prevalence $p_t$;
3. digit-9 recall, equivalently accuracy conditioned on a true digit 9; and
4. environmental multiclass accuracy.

Calculate every metric once per replica and $p_t$, then average pointwise over
replicas at the same $p_t$. Preserve pointwise standard deviations or confidence
intervals. Precision and one-vs-rest accuracy must use the environmental
prevalence rather than the fixed holdout prevalence. If $r_9$ is digit-9 recall,
$f_9$ is the false-positive rate among non-nine observations, and $s_9=1-f_9$,
then

$$
A_{9,\mathrm{OvR}}(p_t)=p_t r_9+(1-p_t)s_9,
$$

$$
P_9(p_t)=\frac{p_t r_9}{p_t r_9+(1-p_t)f_9},
$$

whenever the precision denominator is nonzero. Environmental multiclass
accuracy remains

$$
A_{\mathrm{env}}(p_t)
=p_t r_9+(1-p_t)A_{\mathrm{non9}},
$$

where $A_{\mathrm{non9}}$ requires the exact multiclass prediction for a
non-nine observation. The distinction between $s_9$ and
$A_{\mathrm{non9}}$ is essential: a model may avoid false digit-9 predictions
while still confusing the old digits with one another.

Compare these trajectories against $p_t$, cumulative total observations,
digit-9 presentations, and unique digit-9 observations. Report their AUCs and
fixed-exposure contrasts. Any durable threshold criterion adopted at a phase
gate must consider false positives; a digit-9 recall crossing alone is not
evidence of adequate adaptation.

"Durable" means that the expected trajectory remains at or above the threshold
for all later recorded points. If the threshold is never reached, report that
fact and compare accuracy at common fixed exposure budgets. Do not average
per-replica stopping times as the principal estimand.

Secondary outcomes are:

- digit-9, non-nine, and environment-weighted NLL;
- non-nine multiclass retention and calibration;
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

### Finite-sample relative efficiency

Report sample efficiency as a finite-sample relative efficiency, presented as
the **Equivalent Data Multiplier** (EDM). At a fixed environmental state $p$,
let treatment $T$ use $n_T$ unique digit-9 observations and attain expected
primary-metric vector $\boldsymbol\mu_T(p)$. For baseline $B$, define

$$
n_B^{\mathrm{eq}}(p)
=\inf\left\{n:
\mu_{B,k}(n,p)\geq \mu_{T,k}(n_T,p)-\delta_k
\text{ for every requested metric }k\right\},
$$

and

$$
\operatorname{EDM}_{T:B}(p)=\frac{n_B^{\mathrm{eq}}(p)}{n_T}.
$$

The joint EDM requests all four primary predictive metrics, preventing recall
from hiding false-positive or old-task failure. Also report one EDM per metric
to identify the binding outcome. Practical-equivalence margins $\delta_k$ must
be declared; zero is the default until a phase gate approves nonzero margins.

Use the tested no-EWC data-budget envelope rather than assuming a smooth or
monotone realized learner. Sparse screens report discrete brackets: $(2,4]$
means twice the treatment exposure failed to match it and four times succeeded.
If the largest baseline budget still fails, report a right-censored result such
as EDM $>4$. Interpolation may be descriptive but is not the principal estimate.

Plot the EDM over $p_t$ and preserve the state-specific bracket. For an agreed
joint acceptable region, also report the expected trajectory's durable
observations-to-criterion. Keep data, optimizer work, elapsed compute, and
storage as separate efficiency denominators; never combine them into an opaque
single score. Use expected exposure for design calculations and actual unique
and presented digit-9 counts for experimental reporting. Phase 3 should attach
paired-replica uncertainty to these quantities once five replicas are present.

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

### Completion record

**Status:** Complete (2026-08-13)

- Added exact optimizer-consumed exposure accounting from recorded stream IDs
  and labels, including repeated and unique digit-9/non-nine observations.
- Added opt-in multiclass Brier score and 15-bin maximum-probability ECE to the
  existing holdout forward pass. New controller runs use config schema 11 and
  scalar-metric schema 7; unchanged tensor artifacts remain at schema 4.
- Added artifact-only expected-trajectory, durable-crossing, exposure-AUC,
  fixed-budget, uncertainty, pairing, and metric-availability helpers.
- Added a Plan 2 readiness panel to `results.ipynb` and the concise human
  contract to `EXPERIMENTAL_CONDITIONS.md`.
- Verified that completed Plan 1/Phase 9 bundles still load. Their exact
  exposure coordinates are derived from immutable stream artifacts; calibration
  fields correctly remain unavailable when their older schemas did not record
  them.
- `python -m pytest -q test/unit`: 184 passed in 3.83 seconds.
- `python -m mnist_experiment.validate_results_notebook --notebook
  mnist_experiment/results.ipynb --max-seconds 60`: valid, 17 code cells in
  45.70 seconds, with no training, data download, model loading, or Fisher
  calculation.

**Gate recommendation:** review and approve the documented target-regime rule,
then proceed to Phase 1's immutable nested-stream construction.

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

### Completion record

**Status:** Complete (2026-08-14)

- Added self-contained immutable prefix bundles with derivation schema 1.
  Metadata records parent bundle/design/model/partition/stream hashes, the
  requested $m$, the exact prefix rule, and the derived stream hash. Existing
  completed masters remain unchanged.
- Added `plan2_command_center.py` and explicit production/CPU-smoke/CUDA-smoke
  specifications. The production preview creates exactly 63 runs from three
  conditions, seven requested $m$ values, and three replicas, with 21 derived
  bundles and zero new initialization fits.
- Strengthened external reference-path validation to check the source bundle
  (or a derived bundle's parent), initial parameter hash, parameter count,
  partition, $p$ grid, and oracle content hash.
- Added singleton `vmap`, Fisher-scaling, controller-finiteness, derivation,
  pairing, and no-Cartesian-expansion tests. `python -m pytest -q test/unit`
  passed all 196 tests in 3.41 seconds.
- Completed all six CPU and six CUDA smoke trajectories at $m\in\{1,2\}$.
  Every loss, parameter tensor, displacement, controller value, optimizer
  diagnostic, and represented PSD component was finite. All conditions shared
  oracle hash `32552156d0ac...` and initial-Fisher cache digest
  `6eb19716f989...`.
- CUDA condition time after warm-up was approximately 0.53 seconds for both
  $m=1$ and $m=2$, confirming dominant fixed overhead at tiny batches. The
  production proxy now combines an intercept with observation-dependent cost
  and is anchored by the 283.68-second median of 85 completed production-shaped
  $m=128$ runs. The full proposed screen previews at 4.01 wall-hours and 3.52
  GiB; these remain planning proxies.
- Prepared an artifact-only production check for replica 1 at $m\in\{1,2\}$. Both
  streams are exact prefixes of its retained $m=128$ master and share model hash
  `8acc1ee09a5e...` and partition hash `ce28156d60b3...`.
- Deviation: the first CPU smoke found a schema-v7 manifest `NameError` after
  trajectory calculation. The calibration-bin constant was centralized and
  the incomplete run resumed successfully; no completed artifact was changed.
- Residual risk: the CUDA adaptive-average-pooling backward remains warning-only
  nondeterministic. The smoke was finite, but exact bitwise CUDA replication is
  not claimed. Production cost at low $m$ is also still an estimate until the
  first coarse-screen runs are measured.

**Gate recommendation:** retain the planned grid
$m\in\{1,2,4,8,16,32,64\}$ and three paired replicas for Phase 2. Use early
status and timing from the first completed runs to revise the remaining-time
estimate, but do not narrow the statistical grid before observing outcomes.

## Phase 2: Coarse sample-size boundary screen

### Goal

Locate where the no-EWC learner begins to lose practical effectiveness.

### Scope

1. Prepare immutable paired runs for the three principal conditions at
   $m\in\{1,2,4,8,16,32,64\}$ using replicas 1 through 3.
2. Reuse compatible $m=128$ evidence as the easy endpoint.
3. Preview expected wall time and disk use, then provide one resumable `tmux`
   command. The user controls execution.
4. After completion, produce artifact-only summaries of:
   - expected digit-9 OvR accuracy, precision, and recall trajectories;
   - expected environmental multiclass accuracy trajectories;
   - actual total and digit-9 exposure;
   - non-nine retention;
   - accuracy/NLL/calibration disagreement;
   - controller $\pi_t$ behavior;
   - Fisher error and numerical interventions;
   - optimizer and resource costs.
5. Compare conditions only within paired replica and $m$ cells. Across $m$,
   exploit nested streams but keep replica-level uncertainty visible.

### Pre-launch record

**Status:** User-controlled execution in progress (2026-08-15)

- Prepared immutable bundle
  `plan2-low-data__r0001-r0003__913b7f56a535`: 63 run intentions from three
  conditions, seven values of $m$, and three paired replicas.
- All 21 derived stream bundles and reference paths validate as complete; no
  new initialization fits are required. The six intentions also present in the
  Phase 1 subset bundle deduplicate by run ID rather than creating extra runs.
- The artifact-only analysis maps the compatible Phase 9 conditions for
  replicas 1 through 3 into the $m=128$ endpoint, checks their aligned $p$
  grids, and pairs both EWC treatments to the no-EWC control.
- `results.ipynb` now reports Plan 2 launch/completion state and merges completed
  nested-stream trajectories with that $m=128$ anchor. It executes without
  training and validated in 45.36 seconds against the prepared inventory.
- The calibrated planning proxy is 4.01 wall-hours and 3.52 GiB. It is not a
  measurement of active GPU time.
- The first production launch exposed two rank-deficiency failures in the
  `m=1`, no-EWC Fisher diagnostics: CUDA `eigvalsh` nonconvergence and a
  legacy Lanczos recurrence continuing beyond the candidate's numerical rank.
  The update itself remained finite. The tracker now retries only failed
  spectral diagnostics on CPU float64, reports a scale-aware
  material-negative count, caps Krylov depth at numerical rank plus one
  null-space direction, and lets the wrapper step down to a finite Krylov
  prefix if needed. The pinned copied `lanczos.py` source was not modified.
- The exact failed run subsequently completed all 100 steps in 257.85 seconds.
  Two spectral diagnostics used the CPU fallback; no candidate was materially
  negative, no finite-prefix Lanczos retry was required after rank limiting,
  and maximum represented-diagonal relative error was
  $2.63\times10^{-5}$.
- Plan 2 analysis derives separate original-learning and auxiliary-Fisher
  process labels from the named condition. Existing artifacts are not mutated:
  `no-ewc-pi100` is reported as current-batch learning with an instantaneous
  empirical Fisher diagnostic after initialization, while both EWC conditions
  retain recursive Fisher summaries.
- The artifact-only expected-trajectory, durable-crossing, and metric-coverage
  summaries now stratify by `samples_per_step`. This prevents completed
  low-data trajectories from colliding with or being averaged into the reused
  $m=128$ anchors during partial execution.
- Metric-contract amendment (2026-08-15): the original scalar artifacts called
  accuracy conditioned on a true 9 `nine_accuracy`; that quantity is 9 recall
  and cannot detect a classifier that predicts 9 indiscriminately. Plan 2 now
  uses 9 OvR accuracy, environmental-prevalence-adjusted 9 precision, 9 recall,
  and environmental multiclass accuracy as its four primary predictive
  trajectories. Completed schema-7 runs remain immutable and will receive a
  separate inference-only classification summary derived from their saved
  parameter trajectories. Future runs record the required confusion rates
  directly under scalar-metric schema 8.

### Completion record

**Status:** Complete (2026-08-16)

- All 63 requested low-data trajectories completed, and inference-only
  classification sidecars supplied the four primary metrics for those runs and
  the nine compatible $m=128$ anchors without mutating their source artifacts.
- The screen found the intended transition. $m\in\{1,2\}$ was general failure,
  $m=4$ was noisy and borderline, $m=8$ was the clearest strongly
  data-constrained EWC advantage, $m=16$ retained a material EWC advantage,
  and no-EWC became competitive by $m=32$. The methods were near parity at
  $m=64$, while no-EWC won the easy $m=128$ anchor.
- At $m=8$ and the last state below $p=.5$, fixed EWC obtained 9 OvR accuracy
  $.822$, 9 precision $.790$, 9 recall $.873$, and environmental multiclass
  accuracy $.738$. No-EWC obtained $.577$, $.551$, $.992$, and $.547$.
  Its high recall therefore reflected excessive digit-9 prediction rather than
  a superior joint classifier.
- Reaching that state used 400 online observations per trajectory, with 99
  expected digit-9 presentations; the empirical means were 97.3 presentations
  and 95.7 unique nines.
- No-EWC at $m=16$ did not jointly match fixed EWC at $m=8$, while no-EWC at
  $m=32$ did. The principal finite-sample statement is therefore a tested EDM
  bracket $(2,4]$. A roughly threefold interpolation is useful motivation, not
  yet a precise estimate.
- Fixed $\pi=.10$ was slightly better and substantially less variable than the
  adaptive controller at $m=8$. The adaptive controller averaged $\pi_t=.060$
  over the first half of the path, suggesting excess retention in this regime.
  It remains the portability candidate rather than being promoted as the
  within-path winner.
- All Plan 2 conditions used no LFU. The screen identifies a data regime and
  does not provide evidence for or against the derivative correction.

**Gate decision:** accept $m=8$ as the provisional target and advance to a
focused Phase 3 replication. Preserve the $m=16$ and $m=32$ upper bracket so
the EWC advantage and no-EWC-equivalent data boundary are estimated with the
same five paired replicas.

### Verification

- Every requested run is complete or explicitly identified as incomplete.
- No condition has silently changed representation, optimizer budget, path
  grid, initialization, or Fisher-update method.
- Expected-trajectory calculations use post-update 9 OvR accuracy, precision,
  recall, and environmental multiclass accuracy for acquisition and actual
  observed class counts for exposure.
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
   - expected 9 OvR accuracy, precision, and recall at common exposure budgets;
   - any jointly defined durable acquisition criterion or failure to meet it;
   - 9-centric and environmental multiclass AUC;
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

### Pre-launch decision

**Status:** Complete (2026-08-16)

- Retain $m\in\{8,16,32\}$ and add replicas 4 and 5, producing 18 new paired
  trajectories across the three principal conditions. Existing replicas 1
  through 3 remain immutable and are not rerun.
- Keep the 100-point path, rank-8-plus-diagonal representation, no-LFU Fisher
  recursion, and $K=50$ L-BFGS budget fixed. Existing diagnostics do not
  motivate the optional optimizer-budget screen.
- Treat $m=8$ as the provisional application target, $m=16$ as its upper
  neighbor, and $m=32$ as the tested no-EWC matching boundary used by the EDM.
- Report the four expected predictive trajectories, discrete per-metric and
  joint EDM brackets, exposure counts, retention, controller behavior,
  numerical diagnostics, and separately labelled resource costs. Do not use
  an interpolated EDM as the confirmatory estimate.
- Prepared immutable bundle
  `plan2-low-data__r0004-r0005__ebf1dd35123e` contains the intended 18 runs,
  six completed derived streams, completed reference paths, and zero new
  initialization fits. Its planning estimate is 1.15 wall-hours and 1.01 GiB.

### Compute record

- All 18 new trajectories completed without run failures, numerical rejection,
  hard freezes, negative-eigenvalue events, or CPU spectral fallbacks. Their
  summed sequential trajectory time was 1.44 hours.
- At $m=8$ and the last state below $p=.5$, adaptive EWC averaged 9 OvR
  accuracy $.827$, precision $.817$, recall $.853$, and environmental accuracy
  $.740$. Fixed EWC averaged $.813$, $.817$, $.811$, and $.716$; no-EWC
  averaged $.701$, $.684$, $.954$, and $.624$ with substantially greater
  between-replica variability.
- Adaptive EWC also had the best midpoint NLL, Brier score, ECE, and non-nine
  accuracy. Its paired environmental-accuracy advantage over no-EWC was $.115$
  with a 95% interval $[.006,.225]$; the corresponding first-half trajectory
  AUC advantage was $.211$ with interval $[.032,.390]$.
- Adaptive and fixed EWC are not decisively separated. Adaptive leads on the
  mean joint classifier and environmental trajectory, while fixed remains a
  useful mechanism control. The adaptive controller averaged $\pi_t=.058$ in
  the first half of the $m=8$ path and occupied $\pi_{\min}=.05$ on 48% of
  those steps.
- The strict joint point-mean EDM lies in the empirical unique-nine exposure
  bracket $(1.96,3.79]$ for both EWC treatments. Exact paired bootstrap
  matching probabilities show why this must remain a bracket: no-EWC $m=16$
  matched only 47% of adaptive and 51% of fixed resamples, whereas no-EWC
  $m=32$ matched 82% and 95%, respectively.
- No-EWC's instantaneous rank-8 Fisher diagnostic remained numerically weak at
  some low-data steps. This is an expected rank-limited diagnostic and is not
  evidence that its learner failed. The EWC recursive summaries remained
  numerically stable.

**Recommended gate decision:** accept $m=8$ and proceed to Phase 4 controller
recalibration. Preserve the EDM as an interval rather than claiming a precise
threefold gain; narrow it later only if that precision changes an application
decision.

**Gate decision:** accepted $m=8$ and advanced to Phase 4. The strict joint
EDM remains an interval, and no extra Phase 3 replicas are requested.

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

### Pre-launch decision

**Status:** Complete (2026-08-16)

- Phase 4A screens $h\in\{.05,.10,.20,.40\}$ at $m=8$ with five paired
  replicas. Reuse the completed $h=.10$, fixed-EWC, and no-EWC trajectories;
  execute only the 15 missing adaptive trajectories.
- The initial expanded bundle `plan2-low-data__r0001-r0005__967e09afc995` was
  superseded before execution because schema-8 regeneration would have
  needlessly duplicated nine schema-7-plus-sidecar controls. It remains an
  immutable intention but is excluded from execution. The replacement bundle
  contains only the three missing adaptive conditions and pairs them during
  analysis by replica and stream provenance.
- Keep the 100-point path, rank-8-plus-diagonal representation, no-LFU Fisher
  recursion, $K=50$ optimizer budget, $\pi_{\min}=.05$, and
  $\pi_{\max}=.95$ fixed.
- Interpret each $h$ as an applied policy package because it controls both
  trend responsiveness and cold-start duration.
- Select $h$ using the four primary expected trajectories, proper scoring and
  calibration, non-nine retention, controller diagnostics, and paired
  variability. No single endpoint or recall trajectory decides the screen.
- Phase 4B is conditional. If the selected controller remains materially
  pinned to $\pi_{\min}$, screen $\pi_{\min}\in\{.01,.05,.10\}$ only at the
  selected $h$, reusing the center value and executing ten missing trajectories.

### Half-life compute record

- Immutable bundle `plan2-low-data__r0001-r0005__292a833830f0` completed all
  15 missing trajectories without run failures, numerical rejection, hard
  freezes, spectral fallbacks, or materially negative eigenvalues.
- At the last state below $p=.5$, $h=.20$ obtained 9 OvR accuracy $.831$,
  precision $.824$, recall $.848$, environmental accuracy $.752$, NLL $.901$,
  Brier score $.485$, ECE $.162$, and non-nine accuracy $.657$.
- $h=.20$ and $h=.40$ were operationally indistinguishable over the first half
  of the path. Their paired midpoint differences were below $.003$ on every
  reported predictive or proper-scoring metric. Select $h=.20$ because it has
  the shorter cold start and therefore preserves more responsiveness without a
  measured loss.
- $h=.05$ was materially worse, while $h=.10$ remained competitive but had
  weaker midpoint environmental accuracy, proper scores, calibration, and
  retention than $h=.20$.
- The selected $h=.20$ controller occupied $\pi_{\min}=.05$ on 99.2% of
  first-half points; $h=.40$ occupied it on 100%. The lower bound therefore
  obscures the controller formula and triggers Phase 4B.

**Phase 4B decision:** hold $h=.20$ fixed and add only
$\pi_{\min}\in\{.01,.10\}$ around the completed $.05$ center, using the same
five replicas and all other Phase 4 factors unchanged.

### Lower-bound compute record

- Immutable bundle `plan2-low-data__r0001-r0005__0c4eebbf40db` completed all
  ten requested trajectories without numerical or artifact failures.
- Lowering $\pi_{\min}$ to $.01$ destabilized the feedback loop: first-half
  applied $\pi_t$ averaged $.121$ with standard deviation $.160$, trace error
  increased sharply, and midpoint environmental accuracy fell to $.589$.
- Raising $\pi_{\min}$ to $.10$ pinned every point to $.10$ and reproduced the
  fixed-$\pi=.10$ learner exactly. This is a useful implementation audit, not
  evidence for successful adaptation.
- The $.05$ center remained best, but it was bound-active on 99.6% of all path
  points. The $h=.40$ condition was bound-active at $.05$ on every point and is
  therefore behaviorally fixed, despite retaining an adaptive configuration.

**Begged control:** run one explicit fixed-$\pi=.05$ five-replica condition.
It should reproduce the realized $h=.40$ learner exactly. Use this audit to
separate the value of the $.05$ composition from the value of the plug-in
controller before selecting the Phase 5 policy.

### Fixed-policy audit and decision

- Immutable bundle `plan2-low-data__r0001-r0005__69aea5f9bdc3` completed all
  five explicit fixed-$\pi=.05$ trajectories without numerical or artifact
  failures.
- The fixed policy reproduced `adaptive-ewc-h040` exactly at all 500 paired
  replica-step points. Parameter hashes, parameter norms, applied EWC and
  Fisher weights, predictive metrics, proper scores, calibration, Fisher
  error, and storage diagnostics all matched.
- At the last state below $p=.5$, fixed $\pi=.05$ obtained 9 OvR accuracy
  $.831$, precision $.824$, recall $.846$, environmental accuracy $.750$,
  NLL $.904$, Brier score $.487$, ECE $.163$, and non-nine accuracy $.655$.
- The $h=.20$ plug-in policy was nearly fixed as well: only two of 500 points
  rose above $.05$, with a maximum applied value of $.0516$. Lowering the
  bound did not uncover a stable controller, while raising it reproduced the
  corresponding fixed policy.

**Gate decision:** carry explicit fixed $\pi=.05$ into Phase 5 as the selected
applied policy, retain fixed $\pi=.10$ as a mechanism comparator, and classify
the plug-in controller as unresolved in this low-data regime. The adaptive
conditions show that $.05$ is a useful composition weight here; they do not
establish that the current trend-and-trace formula selected it.

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

### Pre-launch decision

**Status:** Complete (2026-08-17)

- Use five fresh replicas, 6 through 10. Each receives an independent
  early-stopped $p=0$ fit, initial Fisher summary, stream, and reference path.
  Replicas 1 through 5 selected the Phase 4 settings and are not reused as
  confirmatory statistical units.
- Hold $m=8$, the 100-point path, rank-8-plus-diagonal representation,
  no-LFU Fisher recursion, and optimizer budget $K=50$ fixed.
- Confirm four paired conditions: no EWC, fixed $\pi=.05$, adaptive
  $h=.20$ with $\pi_{\min}=.05$, and fixed $\pi=.10$.
- Treat fixed $\pi=.05$ as the selected practical policy. Retain adaptive
  $\pi$ as a forward-looking policy whose value may emerge on more dynamic
  manifolds; it need not beat fixed $.05$ on this path, but material harm must
  remain visible. Fixed $.10$ measures sensitivity to the composition weight.
- Generate only one Phase 9 oracle anchor per fresh replica. Do not execute the
  seven unrelated controller-screen cells used solely to define the reusable
  source configuration.
- Estimated sequential wall time is approximately 6.25 hours: 4.2 hours for
  five reference paths, about 32 minutes for initialization and source-anchor
  trajectories, and about 1.6 hours for the 20 confirmatory trajectories.

### Compute and result record

- Source bundle `phase9-initial__r0006-r0010__065c98b1061d` completed exactly
  five requested oracle anchors, one per fresh replica. The 35 unrelated
  controller-screen cells were intentionally not executed.
- Confirmation bundle `plan2-low-data__r0006-r0010__7bb69d3a9424` completed all
  20 trajectories with exit status zero. Every derived stream and reference
  dependency is complete; no run failed or required numerical recovery.
- Over $p<.5$, fixed $\pi=.05$ and adaptive $h=.20$ obtained mean
  environmental-accuracy AUC $.682$ versus $.532$ for no EWC. The paired
  improvement was positive in every fresh replica, averaging $.151$ with
  standard deviation $.155$.
- Their corresponding 9 OvR, precision, and recall AUCs were $.837$, $.579$,
  and $.573$, versus $.798$, $.553$, and $.570$ for no EWC. These 9-specific
  paired effects were positive on average but varied across replicas.
- Proper scoring and retention replicated cleanly. Relative to no EWC, the
  paired first-half AUC differences were $-6.929$ for environmental NLL,
  $-.328$ for Brier score, $-.215$ for ECE, and $+.187$ for non-nine accuracy;
  every fresh replica improved in the favorable direction on all four.
- At the last state below $p=.5$, the $.05$ policies obtained 9 OvR accuracy
  $.821$, precision $.779$, recall $.899$, environmental accuracy $.763$, NLL
  $.824$, Brier score $.504$, ECE $.152$, and non-nine accuracy $.630$.
- Adaptive $h=.20$ occupied $\pi_{\min}=.05$ on every point and reproduced
  fixed $\pi=.05$ exactly at all 500 paired replica-step points. This confirms
  the selected EWC weight, but still treats adaptive $\pi$ as a diagnostic on
  the current path rather than evidence of successful variable actuation.
- Fixed $\pi=.10$ improved NLL and calibration over no EWC but was materially
  weaker than $.05$ on acquisition, environmental accuracy, and retention.

**Gate decision:** accept the $m=8$, rank-8-plus-diagonal, no-LFU regime as the
foundation for Plan 3. Carry both fixed $\pi=.05$ and adaptive $h=.20$ with
$\pi_{\min}=.05$: fixed $.05$ is the confirmed practical policy, while the
adaptive condition remains a forward-looking diagnostic for later paths with
more variable speed or curvature.

### Scope

1. Freeze a minimal confirmatory package containing:
   - `no-ewc-pi100`;
   - `fixed-ewc-pi005` as the selected policy;
   - `fixed-ewc-pi010`;
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

- A fresh preview can calculate Plan 3's proposed run count, wall time, and disk
  use without creating artifacts.
- Every proposed Plan 3 comparison has a named control and a clear estimand.
- The handoff distinguishes measured findings from assumptions and deferred
  questions.
- No long Plan 3 experiment starts as part of this phase.

### Completion record

**Status:** Complete (2026-08-17)

- Added [plan3.md](plan3.md), with eight gated phases sized around replay,
  hybrid-archive, LFU, and deployment decisions. Tricky replay and archive
  implementations each receive a smoke-and-review phase before a long run.
- Froze replicas 6 through 10 through source bundle
  `phase9-initial__r0006-r0010__065c98b1061d` and confirmation bundle
  `plan2-low-data__r0006-r0010__7bb69d3a9424`. New Plan 3 conditions reuse
  those initialization, stream, partition, holdout, and reference identities
  while writing new immutable runs.
- Recorded the accepted $m=8$, 100-point path, 50-iteration optimizer,
  rank-8-plus-diagonal summary, fixed $\pi=.05$, adaptive $h=.20$ diagnostic,
  and no-LFU baseline. Rejected and non-promoted Plan 2 regions are listed so
  later work does not reopen them accidentally.
- Defined a first replay screen with FIFO capacities 8, 32, 128, and an
  unbounded unconstrained-memory control. The unbounded condition retains at
  most 800 observations here and is distinct from full retraining on the
  original initialization data.
- Defined memory-matched replay and a hybrid whose active buffer is disjoint
  from the EWC archive. Evicted observations enter the archive once. The exact
  memory-matched capacity will be derived from measured persistent bytes after
  implementation rather than selected for predictive performance.
- Added a validated planning-only specification and
  `python -m mnist_experiment.plan3_command_center preview`. It has no
  `prepare` or `run` command and creates no artifacts. The replay screen
  contains 20 new runs and currently previews at 1.64 sequential wall-hours
  and 1.12 GiB. All contingent later stages total at most 75 new runs, 5.50
  planning hours, and 4.19 GiB before gate-based pruning.
- The cost model is explicitly provisional: replay timing, LFU incremental
  cost, deployment diagnostic savings, and the selected-budget placeholder
  must be recalibrated during Plan 3. Memory matching is now derived separately
  from the canonical persistent-byte contract.
- Added concise human condition/command documentation and separate
  [AGENTS_PLAN3.md](AGENTS_PLAN3.md) implementation invariants.
- `python -m pytest -q test/unit/test_plan3.py`: 5 passed.
- `python -m pytest -q test/unit/test_plan2.py test/unit/test_phase9.py`: 19
  passed.
- `python -m pytest -q test/unit`: 231 passed.
- `python -m mnist_experiment.plan3_command_center preview --stages
  replay-screen --details`: completed without writing run artifacts.

**Gate decision:** Plan 2 is complete. Begin Plan 3 only after reviewing the
replay timing and hybrid weighting contracts in its Phase 0; no Plan 3 compute
experiment was launched here.

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
