# Implementation Plan 16: Paired Diagonal Fisher Updates

**Motivation:** improve continual learning when observations, retained data, and
compute are scarce. The practical question is whether a cheaper way of keeping
EWC curvature current improves learning of rare digit 9s while preserving digit
0--8 recognition. Fisher-estimation accuracy is a supporting diagnostic, not
the objective or sufficient evidence of useful learning.

Earlier LFU studies estimated noisy contracted AC/residual corrections and did
not establish applied value. Plan 15 established useful predictive improvements
for a different intervention, FD KL on STM; its recent EWC archives were
rank-64-plus-diagonal, not purely diagonal. This plan does not assume that those
results establish the adequacy of diagonal EWC. It deliberately makes diagonal
EWC the common, economical background for a new paired comparison.

> **Question:** can a matched-input difference of model-Fisher diagonals at the
> actual pre/post-update parameters improve acquisition and retention compared
> with direct-EMA EWC, at an acceptable measured computational cost?

**Status:** completed on 2026-11-01. All phases, 15 development trajectories and
36 fresh trajectories finished with common $c=.01$, $r=.1$ in the corrected
`production_v3` source revision. Fresh cohorts contain eight main and four
stationary replicas, each with all three methods. Incomplete v1/v2 pilots are
retained with repairs and common optimizer rescaling documented in the notebook.
Recorded cumulative active time was about 2.43 hours, plus a few minutes of final
reporting/verification, below the 20-hour ceiling. Plan 17 remains unexecuted.
Results and completion state appear in
`plan16_results.ipynb` and the concise `PLAN16_HANDOFF.md`.

**Roles:** Astra authors the mathematical design and this plan. After execution
is authorized, Sol implements, validates, calibrates, executes, repairs, and
closes out the study without requiring further model switches or user reviews.
These are the user's operating roles, not inferred model identities.

**Allocation:** at most **20 cumulative active wall-clock compute hours for
Plan 16**, including acceptance tests, failed runs, repairs, analysis, reporting,
and restart overhead. Plan 17 has its own proposed 20-hour allocation and needs
separate authorization; these are not permission for an automatic 40-hour run.
Source-writing time is estimated separately below. Reboots and resume do not
reset the allocation.

**Report:** `mnist_experiment/plan16_results.ipynb`. Create it during Phase 0,
update it throughout execution, and keep it concise. Detailed logs and tables
belong in artifacts, not in the notebook narrative.

## Exploratory Contract

The user explicitly prefers exploration over a predeclared success criterion.
Do not impose an efficacy gate, minimum lift, significance threshold, or
noninferiority margin that determines whether this study counts as a success.
Show acquisition, retention, uncertainty, and cost together; describe gains,
losses, and unresolved trade-offs honestly. Improvement in both accuracies is
desirable but not required before reporting something useful.

Fix measurement definitions, seeds, data isolation, and protocol identities for
each computational iteration. These are reproducibility protections, not a
pre-registration of a winning outcome. Outcome-informed revisions are allowed
within the budget, but become named development versions. Preserve all earlier
results and distinguish fresh replication from reused development evidence.
Neither a promising p-value nor a disappointing curve is a stopping rule.

## Scope And Shared Learner

- Use digit-9 mixtures first. Rotation is a bounded, documented fallback only;
  it is not a mandatory second dataset or automatic Cartesian expansion.
- All maintained Fisher estimates, including initialization and references,
  are strictly diagonal in one fixed parameter chart. No low-rank Fisher
  factors, eigendecompositions, fixed directional bases, or primary random
  probes are required.
- Keep the action weight fixed within each primary trajectory. Adaptive
  actions belong to [Plan 17](plan17.md), which must not be crossed with LFU
  variants here.
- Use the existing gauge-fixed, smooth 487-coordinate MNIST CNN from
  `rotated_mnist/plan12/gauge.py`; keep all 487 coordinates trainable. This
  explicitly replaces the 512-coordinate redundant chart in the generic
  experiment defaults. Diagonality is chart-dependent: never change or
  rebalance this chart during a run.
- The primary learner is batch likelihood plus diagonal EWC and a shared
  isotropic ridge. No STM, replay training, or FD-KL auxiliary penalty is
  included. This isolates the Fisher-update question and is not a replication
  of Plan 15's memory architecture. Adding that applied background later needs
  a separately versioned extension, not a silent change of controls.
- A fresh incoming batch can supply both the auxiliary observation of an
  already-realized displacement and the subsequent learner update. Held-out
  observations never enter either process.

Reuse model, dataset, optimizer, and lifecycle helpers without changing frozen
historical experiment behavior. Prefer a small shared implementation with
Plan 17 over copying an old large scheduler wholesale. Do not inherit Plan 13's
class-importance weighting: the present training likelihood is ordinary
categorical cross-entropy, without class weights or an extra softmax.

## Scientific Contract

### One Fisher And Two Processes

For negative log likelihood $L$, let $g=\nabla L=-s$. Maintain an approximation
to the same likelihood Fisher throughout:

$$
\mathcal I(\theta)=\mathbb E_\theta[s s^T],\qquad
d(\theta)=\operatorname{diag}\mathcal I(\theta).
$$

Digit-9 prevalence changes the implicit true-parameter path; do not introduce
a second Fisher estimand or an independent prevalence argument to $\mathcal I$.
The learner chooses $u_{t-1}=\theta_t-\theta_{t-1}$; the auxiliary process
observes this realized displacement. It must not choose an optimizer proposal
and then incorrectly treat it as an independent historical displacement.

With $H=\nabla^2L$, the diagonal derivative operator is

$$
J_d(\theta)_{ik}=C_{iik}+R_{iik},\qquad
J_d(\theta)=\mathbb E_\theta\left[-(g\odot g)g^T+
2\operatorname{diag}(g)H\right].
$$

It is a generally nonsymmetric matrix even though the Fisher is diagonal in
our representation. A diagonal Fisher does not imply a diagonal derivative
operator. Do not project $J_d$ to PSD or assume it shares Fisher eigenvectors.
The full uncontracted operator has quadratic rather than cubic storage, but
this study need not construct it. The principal treatment estimates its
path-integrated action, not all its entries or the separate AC and residual
tensors.

### Integrate Model Labels Before Differencing

For an input $x$, compute the diagonal contribution

$$
h_i(x;\theta)=\sum_{k=0}^{9}p_\theta(k\mid x)
\left[\partial_i\log p_\theta(k\mid x)\right]^2,
\qquad z_B(\theta)=\frac1{|B|}\sum_{x\in B}h(x;\theta).
$$

Sum all ten model labels exactly, chunking inputs and labels as necessary.
Accumulate directly into vectors; never build a dense Fisher and discard its
off-diagonal entries afterward. Observed-label gradients remain appropriate
for training, not substitutes for this model-label Fisher calculation. The
derivative must include the derivative of the model-label probabilities; a
detached probability weight would omit part of $C+R$.

Accumulate and subtract Fisher observations in float64. A float32 derivative
pass is allowed only after small-step cancellation tests; otherwise evaluate
the curvature pass in float64 and charge its measured cost. Keep the selected
precision policy common across treatments and distinguish roundoff from
statistical instability. Training precision may differ if recorded.

Use identical input IDs, transforms, normalization, and weights at both
endpoints. No dropout or independently randomized augmentation is allowed
between them. Record the input-sampling contract. Label integration removes
conditional label noise, not input-sampling error or misspecification. In a
changing stream, using current inputs to represent both nearby parameter
locations is a local sampling approximation; do not claim that it exactly
reconstructs a discarded historical input population.

### Paired Endpoint Correction

After arriving at $\theta_t$ and before learning from the new batch $B_t$, set

$$
z_t^+=z_{B_t}(\theta_t),\qquad z_t^-=z_{B_t}(\theta_{t-1}),\qquad
\widehat\Delta_t=z_t^+-z_t^-.
$$

At the population level,

$$
d(\theta_t)-d(\theta_{t-1})
=\int_0^1 J_d(\theta_{t-1}+a u_{t-1})u_{t-1}\,da.
$$

Thus this treatment estimates the finite Fisher change rather than a
pointwise derivative with Taylor truncation. Small steps support local EWC
and paired variance reduction; they are not required for the endpoint identity.
Use the name **Paired FD** in plots and distinguish it from Plan 15's FD KL.

With the same fixed action $c$ used by EWC, form the raw update

$$
\widetilde d_t=(1-c)(\widehat d_{t-1}+\widehat\Delta_t)+c z_t^+
=z_t^+ +(1-c)(\widehat d_{t-1}-z_t^-).
$$

This has the recursive correction structure used in SPIDER/STORM [3,4], now
applied to diagonal Fisher measurements. Their smoothness/sampling results
motivate the estimator; they are not predictive-performance guarantees for
nonstationary EWC. Keep fresh-observation assimilation: repeatedly adding
differences without it can accumulate error.

Do not subtract independently sampled successive batches and call the result
paired. Under a valid common sampling law and mean-square smoothness, paired
difference noise can shrink with $\|u\|^2/|B|$; independent endpoint noise need
not vanish as the step shrinks. Do not use the maintained estimate as $z_t^-$:
that would turn the correction into replacement by a fresh noisy measurement.
Reuse a computed endpoint only when all its parameters and batch IDs match.

### Projection And Learner Timing

Record negative-coordinate fraction, minimum value, and negative mass of
$\widetilde d_t$ before clipping. Then use
$\widehat d_t=\max(\widetilde d_t,0)$ elementwise. Ridge is separate from this
Fisher vector. No spectral selector or hidden large diagonal floor is allowed.

For a candidate $\vartheta$, use

$$
J_t(\vartheta)=\overline L(B_t;\vartheta)
+\frac{1-c}{2c}\sum_i(\widehat d_{t,i}+\kappa_t)
(\vartheta_i-\theta_{t,i})^2,
\qquad
\kappa_t=r\,\operatorname{mean}_i\widehat d_{t,i}.
$$

Freeze curvature, ridge, action, and anchor within the solve. Start with the
existing reset-per-arrival strong-Wolfe L-BFGS implementation, at most 50
iterations/75 closure calls, and its recorded tolerances. Record convergence,
gradient norm, closure count, and accepted displacement. Do not accept silent
NaNs or redefine a failed solve as a completed trajectory. On the first online
arrival, use the shared high-quality initial diagonal and no unavailable LFU.

**Numerical revision in production v3:** pilot solves frequently reached the
evaluation cap. Use the same objective in diagonally scaled optimizer
coordinates, $\vartheta=\theta_t+b\odot z$, with
$b_i=[\omega(\widehat d_{t,i}+\kappa_t)+\widehat d_{t,i}]^{-1/2}$.
The implementation floors only this numerical scaling denominator at $10^{-12}$;
it does not add that floor to the Fisher or statistical regularizer. Apply this
to all arms, keep the fixed physical chart for Fisher estimation, and report
physical-coordinate gradients. The saved-solve audit found a roughly $2\times
10^{-7}$ objective gap with unchanged holdout accuracies and substantially fewer
evaluations after scaling. This motivates the runtime revision, not an efficacy
claim. Preserve earlier raw-coordinate pilots as separate development evidence.

At each later arrival: choose/freeze the action from the prescribed policy;
consume $B_t$ to evaluate the previous displacement; update the Fisher;
solve the learner objective; record holdout predictions; atomically checkpoint
the entire accepted transition. Current-batch information must not retroactively
select the action that weights that same batch.

Updating model Fisher is not an exact update of a fixed old-data Hessian. The
EWC use remains a local adaptive-curvature approximation. Its practical value
is assessed through acquisition and retention, not inferred from the identity.

## Conditions And Bounded Calibration

| Condition | Correction before the common direct observation | Role |
| --- | --- | --- |
| **EMA control** | $\widehat\Delta_t=0$ | Practical diagonal EWC control |
| **Paired FD** | $z_{B_t}(\theta_t)-z_{B_t}(\theta_{t-1})$ | Main economical treatment |
| **Analytic LFU** | $J_{d,B_t}(\theta_{t-1})u_{t-1}$ | Differentiated, label-integrated first-order comparison |
| Frozen initialization | No updates; evaluate the shared initial model | Inference-only context, not a tuned learning control |

Use JVPs/HVPs through the label-integrated diagonal for Analytic LFU. Never
materialize a full per-sample Hessian. A central stencil on a few saved
development states is an implementation check, not a fourth full treatment.
Compare analytic and endpoint corrections on shrinking displacements in tests.

Calibrate a **common** action and ridge on development trajectories, starting
with $c\in\{.01,.025,.05\}$ and $r=.1$. Choose using the two accuracy curves
and cost, recording the acquisition/retention trade-off rather than imposing
a pass threshold. If ridge dominates learning or solves fail, allow one
bounded common check of $r\in\{0,.01\}$ at the best-supported action. Keep all
earlier choices visible. Re-run every retained treatment under a changed
common setting; never give a deliberately weak control to a favored method.

For unattended scheduling, the default development tie-breaking/ranking rule
is the equally weighted mean of observation-averaged $A_9$ and $A_{0:8}$,
averaged over the allocated development replicas; exact ties prefer lower
measured cost. This is only a transparent way to nominate a setting, not a
success threshold or headline KPI. Show both underlying accuracies and the
Pareto alternatives. Sol may replace that heuristic with a documented trade-off
judgment before fresh replication; the executable default must not wait for
an agent response or user approval.

Do not run the entire Cartesian grid for every condition. Time one trajectory
of each condition early; spend development predominantly on the common control
and paired treatment comparison. One documented half/double correction-strength
or stencil repair is allowed if a numerical diagnosis supports it, under a
new treatment identity. Outcome-driven tuning is exploratory and its compute
is charged. Random probes are a fallback only after a recorded reason that the
principal implementations cannot answer a useful question, not an automatic
extra arm. No fixed directional bases are authorized.

## Streams, Initialization, And Holdouts

Use the same experimental substrate in Plan 17, with separately named seeds.
Prevalence $p$ is an environmental control, not the learner action $c$ or $\pi$.
Preserve fixed empirical relative frequencies of digits 0--8 conditional on
being non-9, and fixed within-class image sampling. Start with $m=4$ observations
per arrival, $p_{\rm low}=.005$, and $p_{\rm high}=.10$.

The main schedule is **low hold, smooth rise, high hold, smooth return, low
hold**. Initial lengths are 352 arrivals per hold and 128 per ramp. Use a
half-cosine interpolation on each ramp, with zero slope at its endpoints.
Materialize the exact 1,312-arrival vector in an immutable stream artifact;
record segment boundaries, batch counts, and expected/realized 9 exposures.
The controller never receives the boundaries or true $p_t$.

These holds exceed five fixed-weight half-lives for $c=.01$:

$$H_{1/2}(c)=\log(1/2)/\log(1-c).$$

If a smaller action is added, extend holds accordingly or label that schedule
as a different, insufficiently separated probe. Never shorten regimes merely
to make a replica target fit. Include a stationary low-prevalence schedule of
the same length as a smaller negative-control cohort. Continuous drift without
holds is optional, not required. Do not pool different schedules as replicas
of one condition. Rotations are a fallback only after digit-9 evidence and its
limitations have been reported; they require new versioned streams and metrics.

Initialize each replica on a disjoint 2,048-image sample at $p_{\rm low}$,
using the existing supervised fitting helper and a fixed validation-based
checkpoint rule shared by every condition. Audit whether this supplies good
0--8 recognition and remaining 9-learning headroom; ceiling/floor effects are
reported, not hidden. Development may change initialization exposure once for
all arms if the task is uninformative, retaining both versions. Do not select
initializations for favorable treatment effects.

Estimate the initial diagonal by exact label integration on 1,024 independent
reference inputs at the initialization environment. This is an initialization
cost, available identically to all arms, not an uncharged online oracle. Clone
the exact initial parameters, diagonal, and nominal information mass. Record
the real initialization labels/counts; they do not disappear from data accounting.

Use separate initialization, stream, Fisher-reference, development-evaluation,
and fresh-replication evaluation partitions. A given image does not cross these
roles. Sampling with replacement inside an online partition is permitted if
required for long plateaus, but count arrivals and unique image IDs separately.
All conditions in a replica share the same entire online stream and holdouts.

## Predictive Reporting

Headline measurements use ordinary ten-class argmax predictions on independent
stratified holdouts, initially 200 images per class. Evaluate before adaptation
and every 16 arrivals, with extra segment-end evaluations. Sol may reduce only
evaluation frequency, not silently remove entire phases, if timing demands it.

$$
A_9=\Pr(\widehat Y=9\mid Y=9),\qquad
A_{0:8}=\frac19\sum_{k=0}^{8}\Pr(\widehat Y=k\mid Y=k).
$$

Call these **9 acquisition accuracy (recall)** and **0--8 retention accuracy**.
They are class-specific proxies for acquisition/retention, not distinct visual
domains: prevalence drift leaves within-class image distributions unchanged.
Also report false positives and 9 precision standardized to evaluation
prevalence $p_{\rm ref}=.1$:

$$
P_9^{\rm ref}=\frac{.1\,A_9}{.1\,A_9+.9\,\mathrm{FPR}_9}.
$$

Use the agreed non-9 conditional frequencies for FPR and report them. If there
are no predicted positives, mark precision undefined and the count explicitly;
do not draw a misleading interpolated improvement. A display zero may be used
only with an explicit convention/flag. No evaluation prevalence affects training.

Show endpoint and observation-averaged accuracies, separately for each hold
and ramp and for the full trajectory. Report paired percentage-point differences
and nominal 95% intervals across whole replica trajectories. Include cumulative
9 exposures, initial 9 exposure, arrival count, unique IDs, and active compute.
NLL, prequential NLL, and accuracy-weighting decompositions are secondary.
Training losses containing regularizers never enter predictive NLL tables.

Bootstrap complete paired replicas with a persisted seed, not time points or
individual images as independent learner replicas. Start with 2,000 bootstrap
draws in a bounded analysis job. With fewer than four complete pairs, show
individual differences and mark interval estimates as unreliable. Separate
development, fresh replication, revisions, and incomplete pairs. Publish all
specified contrasts, not only favorable ones; make multiple comparisons and
interim inspection explicit. No familywise significance claim is planned.
No numeric success gate or equivalence/noninferiority declaration is imposed.

Fisher error, correction magnitudes, clipping, and solver health get a compact
supporting panel on a few development states. They must not become another
large matrix-estimation study. Equal-arrival comparisons are not compute-matched
comparisons: report measured overhead rather than claiming free efficiency.
Sample-equivalence inversion is optional only where curves genuinely overlap
and its additional assumptions are clearly stated; it is not required to close.

## Execution Phases

### Phase 0: Sol Implementation And Acceptance

Create a small versioned package, proposed as `mnist_experiment/diagonal_lfu/`,
with the common model/stream/metric helpers reusable by Plan 17. Implement the
three conditions, scheduler, budgets, checkpoint lifecycle, and report builder.
Create `plan16_results.ipynb` immediately with pending progress, glossary, and
empty-but-explicit result sections. It must not train or silently repair data.

Acceptance covers score signs, ten-label diagonal calculation, differentiation
through label weights, endpoint identity, zero-step paired cancellation,
analytic/stencil agreement, no dense Fisher allocations, pre-projection metrics,
gauge/model counts, lagged ordering, pure predictive loss, paired streams,
metric denominators, budget accounting, and immutable completion. Use tiny
categorical models for analytic tests, not the retired Gaussian auxiliary-score
experiment. Run focused unit tests and explicit tiny CPU/CUDA resume drills.

### Phase 1: Initial Assets And Resource Pilot

Generate development assets and run the three conditions on one complete main
schedule. These are reusable development trajectories, not disposable timing
work. Measure initialization, ten-label passes, closures, evaluation, checkpoint,
and reporting time separately. Replace the provisional full-study forecast with
measured timings and storage. Inspect the first accuracy plots and check timing,
data leakage, saturation, and numerical validity before substantial allocation.

Correct defects autonomously. Poor predictive results alone do not stop the
study. If no valid trajectory fits the remaining budget, report the concrete
resource issue rather than launching work that cannot be checkpointed.

### Phase 2: Bounded Development

Complete the common action/ridge screen and at least two paired development
replicas of the retained three-arm comparison where affordable. Include one
stationary paired block. Publish choices, rejected numerical implementations,
and trade-offs in a compact table. Freeze the next allocation's settings and
cohort size based on measured cost before inspecting that cohort's outcomes.
No high-sample oracle policy or Fisher-error threshold gates learning experiments.

### Phase 3: Fresh Paired Exploration

Target **eight fresh main-schedule paired replicas** across all three methods
and **four fresh stationary pairs** across them. These are budget-calibrated
targets, not a promise of power or a mandatory count that overrides 20 hours.
Run round-robin by replica/condition; publish only matched comparisons and
clearly label unpaired completed arms. Give both schedules early visible data.

Before this phase, reduce the planned number of complete pairs if required by
measured costs plus 25% contingency and a reserved closeout budget. Protect the
EMA/Paired-FD contrast and complete horizons first. Analytic LFU can be reported
as development-only if its higher cost makes full replication unaffordable;
record that decision before choosing fresh replicas. Never stop at a favorable
interval or extend solely because significance is close. Resource-limited or
mixed evidence is a legitimate exploratory outcome.

### Phase 4: Interpretation And Closeout

Finish paired summaries and inspect all notebook plots. Explain whether Paired
FD changes acquisition, retention, both, neither, or their trade-off, and at
what actual overhead. State the limits of diagonal curvature and the local
input-sampling approximation. Report numerical failures and missing pairs.
Leave a concise final handoff with completed work, budget, artifacts, and any
unexecuted options. Do not automatically launch Plan 17 or reopen Plan 15.

| Phase | Operator | Provisional active compute | Output |
| --- | --- | ---: | --- |
| 0: acceptance | Sol | 0.5--1 h | Tested pipeline, smoke/recovery, pending notebook |
| 1: resource pilot | Sol | 1--2 h | Initial assets and measured full-trajectory costs |
| 2: development | Sol | 2.5--4 h | Common settings and transparent pilot evidence |
| 3: fresh pairs | Sol | 7--10 h | Acquisition/retention/cost comparisons |
| 4: closeout | Sol | 0.5--1 h | Final concise report and restart/closure record |
| **Planning range** | **Sol** | **11.5--18 h** | Subject to measured pilot costs |
| **Hard allocation** | **Sol** | **20 h** | Includes all retries and overhead |

**Measured execution supersedes the provisional range above:** the 15 production
development trajectories used 0.50 h and the 36 fresh trajectories 1.58 h.
Initialization, reports, acceptance, abandoned revisions and restart overhead
bring the persisted cumulative allocation clock to 2.425 h; final post-run checks
add only a few minutes. The diagonal pipeline and common optimizer rescaling
made this substantially cheaper than forecast. All planned pairs completed;
unused allocation was not spent on significance-driven extensions. Accounted
storage stayed below 0.7 GB, including shared references. The focused suite
passed 104 tests, all 71 completed units passed full checksum verification, and
production `--resume` returned `already_completed` without rerunning training.

Allow approximately 4--8 hours of initial engineering effort separately; this
is not a benchmark or an end-to-end delivery promise. Tests and computational
work during engineering count against the active allocation. Do not double
charge a reusable Phase 1 trajectory as a second Phase 2 run. Optional work must
fit after reserving closeout time and storage. More than 20 hours requires a
new user allocation, not a unilateral forecast revision.

## Human-Friendly Notebook

The canonical path is **`mnist_experiment/plan16_results.ipynb`**. Do not place
the only report inside an implementation subdirectory or append this study to
Plan 15. Keep older phases above newer phases. Main content should remain
roughly six multi-panel figures and three compact tables, regardless of replicas.

1. Title, motivation, and a near-top progress table: phase, completed/planned
   paired blocks and trajectories, active unit/checkpoint, elapsed/remaining
   forecast, cumulative allocation, heartbeat, revision, and storage headroom.
2. Short glossary and settings, explaining diagonal EWC, LFU, Paired FD versus
   FD KL, matched inputs, and the two accuracy denominators.
3. Development accuracy/cost curves and a concise settings/changes table.
4. Main and stationary accuracy trajectories with uncertainty, regime bands,
   frozen reference, exposure counts, and consistent legend/axis conventions.
5. Precision/recall, paired accuracy differences, and measured cost; one small
   numerical-health panel linked to full machine-readable diagnostics.
6. Bottom Conclusion: useful observations, trade-offs, limitations, and next
   questions. Do not manufacture a binary winner or fill paragraphs with numbers.

Perform inference, bootstrapping, and expensive summaries in Python entry
points. The notebook loads compact sealed artifacts and renders lightweight
figures only. Explicit partial-study status is allowed; corrupt/incompatible
artifacts must fail clearly. Refresh atomically at phase boundaries, safe stops,
and at most every 15 active minutes; avoid per-skipped-unit report replay.

## Recovery, Autonomy, And Storage

- Use `~/.venv/bin/python`; record Python/PyTorch/CUDA, device, dtype, git commit,
  dirty status, schemas, hashes, and named component seeds. Do not install into
  the system interpreter. CUDA availability and competing GPU work are checked
  before launch; do not terminate unrelated processes.
- Every computational revision writes under
  `cache/mnist_experiment/plan16/<study-id>/`. Completed units are immutable;
  manifests bind configuration, source, schedules, seeds, and dependencies.
  Temporary files are atomically sealed only after required writes/checks finish.
- Checkpoint parameters at both displacement endpoints, Fisher state, optimizer
  state if needed, pending batch, RNGs, metric offsets, q/history accounting,
  solver status, and cumulative time/storage at accepted-arrival boundaries.
  Initialization and analysis use bounded restartable chunks. A hard kill should
  lose no more than one arrival or one explicitly measured bounded chunk.
- Require `--resume`, `--allocation-hours`, `--max-wall-seconds`, `--max-units`,
  `--clear-stop`, `status`, and a stop-file/signal interface. Persistent active
  time includes retries and resume validation; conservatively account for an
  interrupted invocation. Never reset it with a new process.
- Target a graceful stop within one minute. Check stop requests during training,
  hashing, validation, report generation, and bootstrap work. Test actual bounds;
  disclose unavoidable kernels. Resume uses a validated completion index, not
  an hours-long scan of every old checkpoint or notebook reconstruction.
- One foreground Python scheduler must execute all authorized phases without
  conversational polling. It can be launched in `tmux` for remote operation;
  do not require `tmux` locally or imply that a reboot auto-resumes. Prevent
  duplicate writers with a recoverable lease and source/config checks.
- Sol may repair mathematics or implementation and make bounded experimental
  revisions without Astra/user check-ins. Record material changes promptly in
  the notebook and a linked change ledger, including equations, evidence,
  affected artifacts, validation, and revised forecasts. New scientific behavior
  gets a new identity; old evidence remains intact. Continue compatible work.
- Target **under 10 GB** of new artifacts, revised from actual measurements.
  Enforce a **shared 500,000,000,000-byte peak ceiling for Plans 16 and 17**,
  counting referenced assets once and including staging/checkpoint copies and
  finalization reserves. Also retain at least 20 GB of actual filesystem free
  space. Reserve bytes before admission; stop before exhausting space. Store
  metric vectors, seeds/IDs, selected checkpoints, and aggregate diagnostics,
  not every per-example gradient or intermediate derivative graph.
- Preserve all existing experiments and archival jobs. No pruning, external
  migration, old notebook rewrites, or automatic budget extension is authorized.

Tested interfaces (the active production revision is `production_v3`):

```bash
~/.venv/bin/python -u -m mnist_experiment.diagonal_lfu.run \
  --config mnist_experiment/diagonal_lfu/configs/default.json \
  --root cache/mnist_experiment/plan16/production_v3 \
  --allocation-hours 20 --resume
~/.venv/bin/python -m mnist_experiment.diagonal_lfu.control status --root <study-root>
~/.venv/bin/python -m mnist_experiment.diagonal_lfu.control stop --root <study-root>
~/.venv/bin/python -m mnist_experiment.diagonal_lfu.report --root <study-root>
```

Sol supplies the tested concrete launch/resume commands and keeps
`mnist_experiment/PLAN16_HANDOFF.md` current once implementation begins. Smoke
and production configurations must have different identities. Report generation
must never initiate experiments, download data, or repair incomplete runs.

## References

1. S. Amari, [Differential Geometry of Curved Exponential Families](https://bsi-ni.brain.riken.jp/database/file/85/076.pdf), 1982. Natural-coordinate structure does not remove the residual in general neural coordinates.
2. J. Martens, [New Insights and Perspectives on the Natural Gradient Method](https://www.jmlr.org/papers/volume21/17-678/17-678.pdf), 2020, Section 9. Model-label integration and the output-Jacobian Fisher factorization.
3. C. Fang et al., [SPIDER](https://proceedings.neurips.cc/paper_files/paper/2018/file/1543843a4723ed2ab08e18053ae6dc5b-Paper.pdf), 2018, Section 2. Paired differences for tracking vector-valued quantities under smoothness and sampling assumptions.
4. A. Cutkosky and F. Orabona, [Momentum-Based Variance Reduction in Non-Convex SGD](https://proceedings.neurips.cc/paper/9659-momentum-based-variance-reduction-in-non-convex-sgd.pdf), 2019. STORM's recursive correction motivates the algebra, not a continual-learning accuracy guarantee.
5. R. Karakida et al., [Universal Statistics of Fisher Information in Deep Neural Networks](https://proceedings.mlr.press/v89/karakida19a.html), 2019. Skewed Fisher spectra do not establish matching singular directions for $J_d$.
6. J. Spall, [Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf), 1992. Theoretical background for a possible random-probe fallback, not a primary treatment here.
