# Implementation Plan 15: Fisher-Regularized Short-Term Memory

**Motivation:** make continual learning practical when both new observations
and computational resources are scarce. The aim is useful acquisition of new
information without sacrificing predictions on data already evicted from replay.
Statistical efficiency means useful learning per unique observation;
computational efficiency means useful learning per unit of time, computation,
and memory. Report these separately: a method can improve one while worsening
another, and a successful application needs an acceptable balance.

Fisher-regularized memory is the proposed means, not the ultimate objective.
Later corrections and design revisions should serve this limited-data,
limited-compute continual-learning goal, rather than optimize a curvature
diagnostic for its own sake. Predictive performance on both new and evicted
data, alongside resource cost, remains the basis for judging practical value.

> **Hypothesis:** age-weighted reduction of predictive Fisher curvature in a
> small, LoRA-shaped short-term memory can improve acquisition of new data
> while retaining information from observations already evicted from replay.
> Replay supplies predictive pressure during consolidation; long-term-memory
> EWC protects the evicted information approximately, not perfectly.

**Status:** Draft for review. No Plan 15 implementation or experiment execution
is authorized by writing this document. The numerical defaults below are
prospective choices for this plan, not results or previously validated facts.

**Scope:** paired LoRA-shaped STM and trainable LTM, three curvature strategies,
an otherwise identical replay/EWC control, and an explicit distillation control.
Both rotated digits and low-prevalence digit-9 mixtures are required. Linear
and sigmoid rotations are separate conditions. This is not another architecture,
SubGD, ASG, spectral-ridge, or optimal-$\pi$ search.

**Operational contract:** Astra develops the mathematical paradigm and plan,
then writes and validates the initial experiment codebase. After one user-managed
switch, Sol owns calibration, pilots, confirmation, maintenance, and closeout.
The user may be unavailable throughout the run. Sol may independently correct
implementation and mathematical defects and revise the experiment within its
research scope; no return to Astra or intermediate user review is required.
Material changes are recorded in the notebook and versioned with their evidence.
Most experimental compute belongs after that initial handoff.
The notebook follows Plan 13's readable, progressive format, not Plan 14's
large diagnostic transcript. Peak Plan 15 artifact storage must remain below
**500 GB, interpreted as 500,000,000,000 bytes**.

## Questions And Interpretation

Separate four questions:

1. Can we optimize the intended curvature quantity, with reliable gradients
   into both memory blocks, at an acceptable cost?
2. Does old-data sensitivity to STM actually decline, beyond changes caused by
   factor rescaling, poor predictions, or a learner that has stopped adapting?
3. Does the method improve acquisition and retention relative to the same
   model, replay policy, EWC, and ridge without curvature reduction?
4. Does implicit consolidation compare favorably with an explicit
   distillation-based alternative under data and compute constraints?

Improved predictive performance is useful even when the proposed mechanism
does not explain it. Reduced trace alone is not a positive learning result.
Safe local STM movement without safe removal of the entire STM is a possible
success, not a contradiction. No metric is interpreted as a percentage of an
observation's knowledge stored in one parameter block.

## Scientific Contract

### Likelihood And Curvature

Keep the project's likelihood Fisher and loss-gradient convention:

$$
g_i=\nabla_\theta L_i=-s_i,
\qquad
\mathcal I(\theta)=\mathbb E_\theta[s s^T],
\qquad
\theta=(\theta_L,\theta_S).
$$

For a fixed classification input $x_i$, use the model-label contribution

$$
F_i(\theta)=\sum_{k=0}^{9}p_\theta(k\mid x_i)
\nabla_\theta\log p_\theta(k\mid x_i)
\nabla_\theta\log p_\theta(k\mid x_i)^T.
$$

Its STM and LTM principal blocks are $F_{SS,i}$ and $F_{LL,i}$. These are
contributions to the same likelihood Fisher, not new Fisher estimands. Changing
the digit-9 prevalence moves the environment and estimated parameter; it does
not add an independent prevalence argument to $\mathcal I$. Do not substitute
the outer product of an observed-label gradient or a batch-mean gradient for
model-label integration. Observed-label gradients remain legitimate training
gradients and separately named diagnostics.

The clean local identity is predictive KL, not a general quadratic expansion
of an individual observation's NLL without its linear term:

$$
\operatorname{KL}\!\left(p_\theta(\cdot\mid x_i)
\Vert p_{\theta+(0,\delta_S)}(\cdot\mid x_i)\right)
=\tfrac12\delta_S^TF_{SS,i}(\theta)\delta_S
+o(\|\delta_S\|^2).
$$

Define $T_{S,i}=\operatorname{tr}F_{SS,i}$ and
$T_{L,i}=\operatorname{tr}F_{LL,i}$. Their raw and per-coordinate values are
mechanistic diagnostics, not memory shares. Neither is required to increase
when the other decreases. Confidence and parameterization affect both.

This plan explicitly uses direct-EMA EWC updates without LFU, following the
practical baseline of Plans 12 and 13. It does not modify or reinterpret the
full LFU identity in `mathematical_overview.ipynb`.

### LoRA-Shaped Memory

Use Plan 13's smooth, gauge-fixed 487-coordinate CNN as the base network. Add
an STM residual to each of its three affine maps, including the two convolution
kernels viewed as output-channel-by-flattened-input matrices:

$$
W_{\ell,\mathrm{eff}}=W_{\ell,L}+M_\ell,
\qquad M_\ell=\frac{\alpha_\ell}{r_\ell}B_{\ell,S}A_{\ell,S}.
$$

The branches sum before the existing SiLU or output transformation. Preserve
average pooling and the orthonormal sum-to-zero output chart. All biases belong
to LTM. No additional nonlinear adapter or architecture sweep is included.

Start with rank two and $\alpha_\ell/r_\ell=1$ in every map. This gives 176
STM coordinates and 663 total trainable coordinates; assert both counts. Draw
$A$ from a recorded fan-in-scaled initialization and initialize $B=0$. Never
initialize both factors to zero. Verify identical initial predictions across
conditions and against the base network.

Both LTM and STM remain trainable. Ordinary replay gradients and curvature
penalty gradients reach both blocks. LTM alone receives EWC protection. This
does not inherit the backward-cost savings of a frozen-base LoRA deployment;
measure the actual cost. Initial preferential learning in STM is a hypothesis,
not an assumption about what the optimizer must do.

Fix the factor chart, initialization scale, and LoRA multiplier before method
comparison. Do not silently rebalance, merge, reset, or rotate factors during
training. Such operations can change optimizer behavior and curvature. Record
factor norms, singular values of the small effective residuals, and effective
update magnitudes. Tests must demonstrate that $A\mapsto cA$, $B\mapsto B/c$
can alter raw factor traces while leaving functional dependence diagnostics
unchanged. The latter, not a trace-only improvement, protects interpretation.

## Replay And Eviction

### Objective And Age Clock

Use a FIFO replay capacity of $C=32$ unique observations after each accepted
arrival. Let $B_t$ be the incoming batch and $R_t$ the pre-arrival replay set.
The fitting working set has at most $C+|B_t|$ observations; report this peak
explicitly instead of calling the entire training memory 32 observations.

With nonempty replay, the proposed common data loss is

$$
L_{\mathrm{data},t}
=\tfrac12\overline L(B_t)+\tfrac12\overline L(R_t).
$$

With empty replay, use $\overline L(B_t)$. New observations are not counted
again in the replay term on their arrival. Replays count toward computation,
not toward unique observations or digit-9 exposure.

Let $n_i$ be the cumulative arrival count when observation $i$ entered, and
$n_t$ the count after the current arrival. Define age weight

$$
a_{t,i}=\min\!\left(1,\frac{n_t-n_i}{C}\right)^2,
\qquad a_{t,i}=0\text{ for newly arriving observations}.
$$

This is an observation-age clock, not a count of optimizer or replay passes.
Use the unweighted per-observation Fisher values in descriptive age plots.

The curvature treatment minimizes

$$
J_t=L_{\mathrm{data},t}+R_{\mathrm{EWC},t}+R_{\mathrm{ridge},t}
+\lambda_F\frac{1}{\max(1,|R_t|)}
\sum_{i\in R_t}a_{t,i}\Phi_i.
$$

Do not normalize by the sum of age weights: that would remove the intended
gentle onset when all replay observations are young. The three definitions of
$\Phi_i$ are specified below. Empty replay gives zero auxiliary penalty.

### Retention Policy

For an LTM anchor $\mu_t$ and compressed mean Fisher $\widehat F_{LL,t}$,

$$
R_{\mathrm{EWC},t}
=\frac{\omega}{2}(\theta_L-\mu_t)^T\widehat F_{LL,t}
(\theta_L-\mu_t),\qquad \omega=\frac{1-\pi}{\pi}.
$$

Starting defaults are $\pi=.025$ for rotation and $.05$ for mixtures. Use
rank-eight-plus-diagonal compression of the LTM block, without adding ridge
to the Fisher estimate. The proposed common ridge is

$$
R_{\mathrm{ridge},t}
=\frac{\omega\kappa_t}{2}
\left(\|\theta_L-\mu_t\|^2+
\|\theta_S-\theta_{S,t}^{\mathrm{start}}\|^2\right),
\qquad
\kappa_t=.1\frac{\operatorname{tr}\widehat F_{LL,t}}{d_L}.
$$

The STM ridge is step-anchored numerical/statistical damping, not an archive
of observations. Its anchor and all other penalty inputs stay fixed inside a
solve. This is a new replay/STM retention protocol, not an exact reproduction
of Plan 13, whose mixture conditions had no ridge. Phase 1 may apply the
bounded common-policy calibration below; treatment-specific EWC tuning is
not permitted.

After fitting, remove the oldest examples needed to restore capacity $C$.
Only those evicted exposure IDs enter the online EWC update. At the accepted
full-model state, form their mean model-label Fisher contribution, restrict
to LTM, and update

$$
\widehat F_{LL,t+1}
=\mathcal C_8\left[(1-\eta_t)\widehat F_{LL,t}
+\eta_t\overline F_{LL,\mathrm{evicted},t}\right],
\qquad
\eta_t=1-(1-\pi)^{e_t/m},
\qquad
\mu_{t+1}=\theta_{L,t}^{\mathrm{accepted}}.
$$

Here $e_t$ is the number evicted and $m$ the environment's fixed arrival size.
If $e_t=0$, neither archive nor EWC anchor changes. This is an explicitly
approximate online-EWC policy; it is not exact Bayesian accumulation. The
initial pretraining archive is separate and does not contain online exposures.
Do not force a terminal flush and call it observed post-eviction retention.

The eviction transaction checkpoints the model, archive, anchor, replay IDs,
unique-consolidation ledger, counters, and RNG state together. Resumption must
neither double-consolidate an observation nor lose an evicted observation's
archive contribution. Its last replay fit precedes its archive insertion.

### Solver

Start from the established reset-per-arrival strong-Wolfe L-BFGS solver with
50 iterations and at most 75 closure evaluations. Hold probes, perturbation
steps, teacher predictions, age weights, direction proposals, and EWC inputs
fixed throughout every closure evaluation. Recompute differentiable model
quantities at the candidate parameters; do not freeze their gradients.

Record objective decrease, final residual, closure count, failures, and total
cost. A Phase 1 numerical calibration may raise the common ceiling to 100
iterations or select a tested first-order fallback if higher-order line-search
optimization is unusable. The fallback must pass both acquisition and retention
sanity checks and use a new solver contract. No known under-optimized common
learner may be described as evidence against consolidation, repeating Plan 13's
original Phase 5 problem. If the implemented choices prove inadequate, Sol may
implement and test a repair or replacement without waiting for a model switch.
Apply the autonomous correction protocol below, including a new solver version
and compatible paired controls rather than silently changing an ongoing cohort.

## Curvature Strategies And Controls

### A: JVP Trace

For a Rademacher vector $v$ on STM coordinates, logits $z$, probabilities $p$,
and $u=J_Sz\,v$,

$$
q_A(v)=\sum_k p_k u_k^2-\left(\sum_kp_ku_k\right)^2
=v^TF_{SS,i}v,
\qquad \mathbb E_v q_A(v)=T_{S,i}.
$$

Use $\Phi_i=K^{-1}\sum_jq_A(v_j)/d_S$. Division by $d_S$ sets a mean-direction
scale for comparing penalty strengths; raw trace remains a reported target.
Differentiate through the JVP and probabilities into both memory blocks.

### B: Symmetric Finite-Difference KL

For the same probes and $p_\pm=p_{\theta_L,\theta_S\pm hv/2}(\cdot\mid x_i)$,

$$
q_B(v)=\frac{\operatorname{KL}(p_+\Vert p_-)
+\operatorname{KL}(p_-\Vert p_+)}{h^2}
=v^TF_{SS,i}v+O(h^2).
$$

Use the same probe average and $d_S$ normalization as A. Differentiate both
distributions, not just one branch. Calibrate $h$ against A or an exact small
reference using $h/2,h,2h$ and gradient checks in float64. Freeze the numerical
rule before predictive selection; $h$ is not chosen to maximize test accuracy.
Record the full perturbation norm $h\|v\|$, not only the coordinate step.

### C: Optimizer-Direction KL

On a disposable copy of the current state, make one bounded ordinary-optimizer
proposal with the auxiliary penalty disabled. Its closure budget is at most
five evaluations. Discard all shadow state changes and detach the resulting
STM displacement $\Delta_S$. Count this proposal's cost. Define

$$
\Phi_i=
\frac{2\operatorname{KL}(p_\theta(\cdot\mid x_i)
\Vert p_{\theta+(0,\Delta_S)}(\cdot\mid x_i))}{\|\Delta_S\|^2}.
$$

Use a calibrated maximum perturbation radius and a recorded zero-direction
rule: a negligible proposal produces a zero penalty and an unsupported-direction
flag, not an unstable division. The detached proposal is fixed within the
solve; gradients still pass through both predictive distributions. This is
directional curvature, not a trace estimator. Evaluate it against future
updates and independent directions, never only its training direction.

### Paired Controls

1. `replay_ewc`: the same trainable LTM/STM model, replay, ridge, and eviction
   archive, with no Fisher-reduction penalty. This is the primary comparator.
2. `distillation`: add age-weighted KL from a frozen pre-solve full-model
   teacher to the current LTM-only student, while retaining the ordinary
   full-model data objective, EWC, and ridge.
3. `replay_compute_matched`: in confirmation, give the baseline additional
   optimization on the same observations up to the selected method's calibrated
   compute allowance. Do not give it additional unique data.

For the distillation control, the auxiliary term is

$$
\lambda_D\frac{1}{\max(1,|R_t|)}\sum_{i\in R_t}a_{t,i}
\operatorname{KL}\left(
\operatorname{stopgrad}[p_{\mathrm{teacher}}(\cdot\mid x_i)]
\Vert p_{\theta_L,M=0}(\cdot\mid x_i)\right).
$$

Only this auxiliary term updates LTM alone; the ordinary full-model replay
loss still updates both blocks. Joint fitting must account for any double
contribution from the existing STM, rather than distilling and blindly
restoring a residual afterward. No automatic STM reset or factor merge is
performed. This is a continual joint-distillation control inspired by Progress
& Compress, not a claim to reproduce its task-based algorithm exactly.

Freeze compute-matched allowances from development timing, before confirmation
outcomes. Measure full end-to-end overhead as well as adaptation-only cost.
The controller cannot improve a baseline's budget by reading its comparator's
future results. All curvature implementations remain available as explicit,
versioned configuration choices after selection.

## Environments And Pairing

### Required Conditions

| Condition | Online stream | Purpose |
| --- | --- | --- |
| Rotation, linear | $0\to30\to0\to30$ degrees; 40 arrivals per arrow; four observations per arrival | Smooth continual change and revisitation |
| Rotation, sigmoid | Same endpoints, counts, and normalized logistic convention with slope eight | Concentrated change; separate evidence |
| Digit-9 mixture | $p_j=.001j$, $j=1,\ldots,100$; two observations per arrival | Low-prevalence acquisition with enough FIFO turnovers |

Use the established bilinear rotation, fill, image dimensions, and class
sampling. Record resolved angles and schedule hashes. Do not pool the two
rotation schedules for primary inference or choose only the favorable one.

For mixtures, preserve the existing conditional frequencies among digits
zero through eight. The 200-observation online stream has

$$
\mathbb E[N_9]=2\sum_{j=1}^{100}.001j=10.1.
$$

Precede it by one common two-observation update at $p=0$, before treatment
assignment. This is the only online burn-in. The finer arrival schedule is
deliberate: Plan 13's ten treatment arrivals are too short for this memory
lifecycle study at a useful replay size. This is not the identical Plan 13
ten-positive experiment. Never condition replicas on realized positive counts.

A hundred-positive mixture extension may use the same 100 arrival coordinates
with 20 observations each, giving 101 expected positives. It is a separately
versioned optional condition, not automatically part of initial confirmation.
Keep the 32-observation resident replay capacity and disclose its shorter
arrival-count lifetime. Do not present that extension as isolating positive
count from batching, optimization, or turnover effects.

### Initialization And Evaluation Assets

Reuse tested data/model code, not treatment outcomes or hidden Plan 14 data
dependencies. Build fresh paired initializations. Follow the established
pretraining regime at zero rotation or $p=0$; record every pretraining example,
epoch, and compute cost separately from online statistical efficiency. There
are no digit-9 training examples in mixture initialization. A large pretrained
base is a prior resource, not evidence of learning the entire task from ten
examples.

Use disjoint initialization, online, reference, development-evaluation, and
confirmation-evaluation samplers. Start initial Fisher estimation with 1,024
reference inputs and exact ten-label integration, streamed in chunks; use the
same immutable initial asset for all treatments in a replica. Calibrate its
numerical adequacy before freezing, without a large Fisher-accuracy factorial.

Use fixed stratified holdout panels, initially 100 examples per class, with
class weights appropriate to each environment. All paired conditions see the
same panels. Evaluate at a fixed cadence of four arrivals plus endpoints and
selected eviction checkpoints; compute prequential loss on every incoming batch
before fitting. The protocol records this grid so AUC reconstruction is exact
for the declared sampling grid, not an implied every-step evaluation.

Component seeds separately identify initialization, stream order, references,
evaluation panels, factors, training probes, independent diagnostic probes,
and bootstrap resampling. Development and confirmation replica families are
disjoint. Cache source images once and reference immutable IDs/transforms.

## Predictive Performance And Functional Memory

### New And Evicted Data Are Both Headline Outcomes

The main notebook must show both acquisition and post-eviction retention for
every environment, not only Fisher traces or aggregate current accuracy.

**New/current data:** report prequential NLL, independent current-environment
NLL and accuracy trajectories, and improvement versus unique observations and
wall-clock compute. Distinguish the immediate pre-update score from the
post-update holdout score.

**Evicted data:** report NLL and accuracy on preassigned training cohorts after
their actual eviction, plus independent holdouts from the historical
environments associated with those cohorts. A holdout itself was never in
replay; label these two measurements differently. Preserve the initial-task
holdout as a separate retention reference.

Choose four audit cohorts, each comprising eight consecutive arrival batches
ending at trajectory fractions $0.1,0.3,0.5,0.7$, rounded by a documented
deterministic rule. These windows are chosen before labels or outcomes are seen.
Track individual arrival/eviction times, predictions before the first member's
eviction, and predictions after the entire cohort has been evicted. Follow at
fixed exposure horizons, including one and two replay-capacity lengths where
available. Right-censor unavailable horizons rather than claiming unobserved
retention. Use an observation-based age grid and equal cohort weights within
each replica. Independent historical panels sample the cohort's eight resolved
environments with fixed equal weights, not a retrospectively favorable angle
or prevalence. Preserve class strata for rare-digit diagnostics.

Evaluation-only access to archived IDs cannot feed gradients, teacher targets,
EWC, selection during confirmation, or replay reinsertion. Save provenance
identifying what was evicted, when, and which independent panel corresponds
to it. Audit data are not hidden additional training memory.

For mixtures, prominently show digit-9 precision and recall on current and
historical evaluation panels at fixed evaluation prevalence $p_{\mathrm{ref}}=.1$:

$$
Q_{.1}=\frac{.1\,\mathrm{TPR}_9}
{.1\,\mathrm{TPR}_9+.9\,\mathrm{FPR}_9}.
$$

This prevalence affects evaluation only. Retain the ordinary argmax operating
point, threshold-swept PR curves, false positives per 1,000 non-9 examples,
and conditional digit-9/non-9 NLL. No predicted positives gives precision zero
with an explicit flag. A training cohort with no positives has undefined
recall, not zero recall; retain its count and missingness reason. Plot expected
and realized digit-9 exposure and keep precision and recall separate.

Mixture drift changes class prevalence, not the conditional image distribution
within each class. Standardized current and historical PR therefore need not
differ systematically; they are not different recognition tasks. Historical
mixture NLL also reflects changed weighting. Report this limitation rather
than inventing a distinct task to make retention appear more demanding.

### STM Reliance

On evaluation copies only, replace all $M_\ell$ by $\tau M_\ell$ for
$\tau\in\{1,.9,.5,0\}$ and compute

$$
R_i(\tau)=L_i(\theta_L,\tau M)-L_i(\theta_L,M).
$$

Positive values mean attenuation hurts, negative values mean STM is hurting
that prediction. Report absolute full-model and LTM-only losses alongside the
difference. Calculate predictive KL from the same forward outputs. These
measurements are invariant to factor transformations preserving $BA$ but are
not unique allocations of knowledge across interacting memory blocks.

### STM Update Vulnerability

At selected saved states and exposure horizons evaluate

$$
V_i(t,h)=L_i(\theta_{L,t},M_{t+h})-L_i(\theta_{L,t},M_t).
$$

This swaps in realized later STM while keeping earlier LTM fixed on a
diagnostic copy. It is a counterfactual intervention, not a second trained
trajectory. Record actual joint-model retention, new-data acquisition, and
effective update size beside it; no movement is not successful plasticity.
Independent random-probe curvature remains necessary because one realized
future displacement does not test every available direction.

### Mechanism Classification

| Evidence | Interpretation |
| --- | --- |
| Lower reliance, better LTM-only performance, preserved full performance | Evidence consistent with consolidation into LTM |
| Lower update vulnerability and useful acquisition, but STM removal still hurts | More robust distributed memory; compatible with the local hypothesis |
| Lower trace only | Curvature optimization without demonstrated functional benefit |
| Better prediction without independent trace reduction | Useful alternative mechanism, especially plausible for directional KL |
| Lower reliance accompanied by worse full predictions | Forgetting, not successful consolidation |
| Low vulnerability with negligible acquisition or updates | Lost plasticity, not released capacity |

These patterns support or challenge a mechanism; they do not by themselves
identify a formal causal mediation effect. The paired treatment/control
comparison supplies the intervention on the regularizer.

## Experimental Phases

### Phase 0: Astra Implementation, Validation, And Handoff

Implement every planned production path before the model switch: A/B/C, controls,
both environment adapters, numerical calibration, pilot selection, confirmation,
checkpointing, resource admission, reporting, and artifact-only closeout.
The experiment must be executable by a Python runner, not by future agent
tool calls that manually choose the next unit. This is an initial readiness
requirement, not a claim that unforeseen repairs can only be made by Astra.

Required acceptance checks include:

- Exact small-model Fisher traces and directional values, with finite-difference
  checks of penalty gradients into STM and LTM; no dense per-sample Hessians.
- Nonzero-gradient and learning checks from the LoRA initialization; logits
  equivalent to the base model at zero residual; gauge and parameter counts.
- A/B consistency, both-branch KL differentiation, frozen probes per solve,
  detached C proposal, and explicit zero-direction behavior.
- Distillation teacher detachment and student-only auxiliary gradients, while
  ordinary replay gradients still reach both blocks; no hidden reset/merge.
- FIFO ages, exactly-once eviction, checkpoint transaction recovery, and
  isolation of evicted evaluation examples from all training inputs.
- Functional reliance invariance under factor rescaling, signed ablation
  differences, vulnerability swaps, right-censoring, PR weighting, and AUCs.
- CPU smoke, CUDA smoke through at least two FIFO turnovers, and resumed versus
  uninterrupted equivalence on both environments and every condition.
- Hard-stop and SIGTERM drills during fitting, artifact writes, reporting,
  and validation; cap admission and disk-full tests using an artificially tiny
  budget; no corruption or mutation of completed units.
- A fresh-process resume with no reliance on conversation history, an
  operational live notebook, bounded log/report size, and a serializable
  selection/priority ledger.

Smoke runs are not development or confirmation replicas. Limit Astra's compute
to acceptance and throughput/storage measurements. Do not run the scientific
parameter sweep or long confirmation before handing over merely because the
software is already loaded.

Create `PLAN15_HANDOFF.md` with commands, acceptance results, source/config
hashes, current status, forecast, defaults awaiting calibration, fallback rules,
stop/resume procedure, autonomous correction authority, and known risks. Record
one compact machine-readable
acceptance manifest under the Plan 15 cache. The user controls the model switch;
the runner does not claim to switch Codex models automatically. Phase 0 ends at
an explicit handoff checkpoint with no unattended GPU job still running unless
the user separately asked to start it.

### Phase 1: Sol Numerical And Common-Policy Calibration

Measure complete arrival, trajectory, reporting, validation, and restart costs.
Build shared development checkpoints covering young replay, impending eviction,
and post-eviction learning in all three required environment conditions.

Compare A/B at identical states and probes against exact trace and gradient
references; compare C against its own exact directional reference. Independently
measure trace with exact label integration or a larger held-out probe bank.
Use $K\in\{1,2,4,8\}$ only as a numerical convergence ladder, not a predictive
factorial. Starting numerical targets are relative value error below 2% and
gradient cosine above .99 on nondegenerate float64 checks; log absolute errors
and flag near-zero reference norms separately. These targets establish correct
implementation, not low variance from a single training probe.

If the default control cannot acquire new information while retaining old
predictions, allow one bounded common-policy calibration over replay weights
$\{.25,.5\}$ and ridge multipliers $\{.01,.1\}$ on development seeds. Freeze the
chosen policy for every treatment within that environment. Do not tune EWC,
rank, initialization, buffer capacity, and architecture simultaneously. Diagnose
numerical under-solving before attributing poor behavior to the memory idea.

Publish numerical validity, actual cost, and a simple outcome-independent
storage/compute forecast. A method with unsupported derivatives or reproducible
nonfinite numerics is marked unavailable, not silently assigned zero curvature.
All valid strategies receive Phase 2 predictive probing.

### Phase 2: Sol Paired Strategy And Strength Pilot

Use two paired development replicas per required environment for an initial
three-strength screen of each valid method and the distillation control.
Initialize strength brackets by matching auxiliary/data gradient-norm ratios
near $.03,.1,.3$ on shared nondegenerate checkpoints; this only sets a bracket.
Freeze actual coefficients for each trajectory, record ratios for both blocks,
and do not continually retune them to current outcomes.

If the best response lies at a bracket edge, or all strengths visibly suppress
learning, permit one tenfold extension in the indicated direction. Publish the
reason and all failed candidates. Calibration occurs only on development data.
Matched numerical strengths are not presumed scientifically equivalent.

Run complete trajectories, not merely same-state penalty optimization. Every
candidate must encounter real eviction and subsequent learning. Then take one
best strength per strategy plus the baseline and distillation winner to two
fresh development replicas per environment. This guards against choosing solely
from an unusually favorable first pair without pretending to remove selection
bias completely.

Select using separate current and post-eviction NLL AUC, mixture PR tradeoffs,
functional diagnostics, and cost. Use a development retention tolerance of
0.01 NLL per observation as a proposed reviewable margin, not a discovered
significance threshold. Among candidates inside that margin, prefer better
current NLL; differences smaller than development uncertainty favor lower cost.
Record acquisition/retention Pareto alternatives rather than collapsing them
into an invented universal score.

Choose one primary strategy per environment before confirmation. Carry at most
one additional, mechanistically distinct strategy when its tradeoff warrants
the cost. If no strategy looks favorable, still carry the best valid diagnostic
candidate with a visible failure classification. Poor early efficacy alone is
not a stop rule. A broken likelihood, optimizer, pairing, or artifact contract
triggers isolation and repair of affected work under Sol's authority, not a
mandatory wait for the user or Astra. Unaffected valid work should continue.

### Phase 3: Sol Fresh Paired Evaluation

Freeze the selected configurations, contrasts, independent seed family,
evaluation panels, compute allowances, and replica count before examining any
confirmation outcomes. The initial target is 12 paired replicas per condition;
Phase 1 timing may set a common target of 8 or 16 instead, based only on available
compute and measured throughput. Record that decision before this phase starts.

Each of the three environments includes the ordinary replay/EWC control, its
selected curvature method, distillation, and the compute-matched replay control.
A Phase 2-selected second curvature candidate adds one condition, not another
whole factorial. Schedule round-robin by replica and environment, completing paired
condition sets promptly so interruptions leave interpretable evidence for both
rotations and mixtures. Do not finish all rotations before starting mixtures.

Report running results, but do not stop for significance, change thresholds,
replace a losing condition, or extend a cohort because its p-value is close.
An interruption leaves a clearly labeled incomplete cohort. Additional compute
adds predeclared replica batches or a separate extension, never overwrites old
trajectories. A low-power unresolved result is not proof of no useful effect.
Necessary scientific or implementation corrections remain permitted during this
phase. Sol versions the correction, identifies affected evidence, and restarts
affected comparisons or opens a new cohort as appropriate. It does not silently
pool changed protocols or retain a confirmation label for outcome-informed
redesign; fresh confirmation can follow within the remaining allocation.

### Phase 4: Artifact Synthesis And Closeout

Reconcile the ledger, complete bounded integrity checks, refresh the concise
notebook, and write a short conclusion distinguishing numerical correctness,
functional mechanism, acquisition, retention, and practical cost. List failed
and unresolved conditions, realized positive counts, censoring, and remaining
uncertainty. No expensive reanalysis is silently launched from the notebook.

### Optional Phase 5: Explicit Extensions

The software supports later execution of an unselected A/B/C method, additional
fixed replica batches, or the 101-positive mixture scale. Each requires a new
ledger/version and a declared compute allocation; none is automatically triggered
by a nearly significant result. The four-day travel window can fund these or
larger precommitted cohorts if authorized, but is not permission to consume
96 hours or exceed the storage cap merely because the node is idle.

## Statistical Reporting

Primary predictive outcomes are exposure-normalized current holdout NLL AUC
and post-eviction historical holdout NLL AUC. For retention, time zero is the
last member's eviction. Integrate each cohort through
$H_c=\min(2C,n_{\mathrm{end}}-n_{\mathrm{eviction},c})$ additional observations,
normalize by $H_c$, and average cohorts within a replica. Use the same horizon
across paired conditions; a zero-length horizon is unavailable, not a zero AUC.
Also report trained-cohort retention, prequential NLL,
accuracy, PR, and initial-task retention without conflating their populations.

For each environment separately, define gain as control minus treatment NLL
AUC, so positive is favorable. Report mean paired effects, 95% intervals,
medians, and favorable-replica counts. Resample entire replicas, not timepoints
or individual examples, for paired bootstrap intervals. With very small cohorts,
show individual paired values and acknowledge unstable uncertainty estimates.

The selected curvature method versus ordinary replay/EWC is the primary
contrast. Distillation and compute-matched replay are declared secondary
comparators. A practical acquisition claim requires improved current prediction
without exceeding the prespecified retention margin, with both endpoints shown.
Do not infer noninferiority from a nonsignificant retention difference. Report
nominal per-condition inference transparently; there is no pooled or
familywise-significant claim across schedules, methods, and metrics.

Development evidence is labeled exploratory and never pooled into the fresh
confirmation cohort. Material repairs and outcome-informed design changes have
explicit validity and version labels. Reusing already inspected data for repair
is allowed as development; independent confirmation requires fresh replicas and
evaluation assets. Repeated notebook inspection does not license optional
significance stopping. Fixed cohort completion, not a pleasing chart, determines
when its planned inferential summary is released.

## Human-Friendly Progressive Notebook

Create `mnist_experiment/fisher_memory_results.ipynb` during Phase 0. It reads
small artifacts only: no training, inference, downloads, matrix estimation,
artifact repair, or automatic resume. An empty or partial study has an explicit
status panel; malformed, incompatible, or falsely completed units fail clearly.

Use this main reading order, with about six to eight multi-panel figures total:

1. **Execution progress:** phase, environment, completed/planned paired units,
   active compute, time in reporting/validation/restart, projected remaining
   time, disk used/reserved/forecast, heartbeat, active protocol revision, and
   development versus confirmation.
2. **Question and glossary:** a short hypothesis and definitions of STM, LTM,
   eviction, curvature, reliance, vulnerability, and fixed evaluation prevalence.
3. **Calibration and pilot:** one compact method-quality/cost table and one
   acquisition-retention plot; selected choices and failures in plain language.
4. **Rotation results:** linear and sigmoid facets showing current/new and
   evicted-data NLL and accuracy, with visible uncertainty and matched axes.
5. **Mixture results:** current/new and evicted-data predictive panels, standardized
   precision-recall, exposure/count annotations, and rare-event limitations.
6. **Memory dependence:** age/cohort reliance and LTM-only performance, realized
   update vulnerability, and a compact independent-curvature diagnostic.
7. **Conclusion:** one table giving acquisition, eviction retention, mechanism,
   cost, and uncertainty; name the best supported balance only when justified.

Add a compact **Significant changes** table near the progress panel when needed.
For each material correction, give the date/revision, problem and evidence,
essential before/after change, affected results and validity, validation outcome,
and what runs next. Mathematical changes include the corrected equation or a
link to a short derivation. Link the detailed change record and artifact lineage
instead of pasting code, long logs, or a repair diary into the notebook. Changes
must be visible as work proceeds, not disclosed only in the final conclusion.

Keep paragraphs short, legend names readable, and favorable signs explicit.
No per-replica figure dump, full configuration listing, thousands of metric rows,
or repeated historical status diary in the main notebook. Put detailed numerical
checks and full tables in a separate diagnostic report or small machine-readable
artifacts linked from the notebook. Do not add pages automatically as runs grow.

Refresh atomically after each completed paired unit or at most every 15 minutes
of active work. Reuse incremental summaries and fixed-size figures; never rebuild
the entire historical report after each skipped completed unit during resume.
Only sealed metric chunks enter visible partial results, with their partial
trajectory/cohort status made explicit.

## Robust Execution And Storage

### Immutable Units And Resume

Use `cache/mnist_experiment/fisher_memory/<study-id>/` with configuration-driven
identities containing protocol, source, environment, schedule, replica, condition,
seeds, coordinate layout, and schema hashes. Completed units are append-only.
Temporary writes become completed only after required artifacts are flushed and
verified. Preserve all metadata required by the repository's `AGENTS.md`.

Checkpoint at accepted arrival boundaries and before potentially long analysis
work. Persist parameters, optimizer state where applicable, EWC, anchors, FIFO,
eviction ledger, all RNG states, teacher/probe provenance, evaluation schedule,
metric-chunk offsets, and cumulative compute/storage counters. The maximum lost
work on a hard kill should be one arrival or one bounded analysis chunk, not a
whole trajectory. Resume skips completed units and never double-appends metrics.

Provide `--resume`, `--max-units`, `--max-wall-seconds`, persistent cumulative
compute limits, a stop-file control, and signal handling. Validation, hashing,
compression, and report generation must check for a stop request between bounded
chunks. Target graceful stop within a minute on this small model; smoke-test the
actual bound and report a slower unavoidable kernel explicitly. No uninterruptible
hours-long archive replay as a prerequisite to stopping.

Validate dependency checksums before consuming them. Keep an incremental
completion/validation index so resume does not deserialize all prior checkpoints
or re-run completed reports; full audits remain available, bounded, and
stop-aware. Reject changed scientific code/configurations rather than silently
continuing under an old identity. Authorized autonomous corrections create new
identities and explicit lineage; identity checks must not require a user approval
step. Version reporting-only changes separately.

Run the scheduler as a foreground Python process in the user's `tmux` session
or equivalent persistent session. It must continue its predeclared phases and
fallbacks without an agent connection. Loss of credits or network access must
not determine scientific scheduling. A reboot still requires restarting with
`--resume`; no automatic machine startup is assumed. Sol monitors rather than
being a required dependency for every GPU work unit.

### Hard 500 GB Admission Limit

Budget the entire Plan 15 peak footprint: shared assets, completed and incomplete
runs, checkpoints, metric chunks, logs, reports, compressed files and their
uncompressed temporary counterparts, and staging/scratch copies. Put temporary
files under tracked study storage; moving them elsewhere does not exempt them.

Maintain a byte ledger and reserve conservative upper bounds before each write
or work unit, including simultaneous old/new checkpoints during atomic replacement.
Reject unbounded writes. Admission requires both

$$
\text{used bytes}+\text{reserved bytes}+\text{next-unit upper bound}
\le 500{,}000{,}000{,}000
$$

and enough actual filesystem free space for the reservation plus an OS safety
reserve. Existing unrelated files reduce free space even though they are not
Plan 15 artifacts. Reconcile the ledger periodically against the filesystem;
do not rely solely on a check after a large file has already been written.

Target a footprint far below the ceiling, preferably below 50 GB for the initial
study. Forecast bytes per trajectory and the full ledger during acceptance and
calibration. Store compact metrics and selected audit checkpoints, not complete
per-example score banks, every probe tensor, or every full Fisher matrix at every
step. Stream transient matrices/probes and retain reproducible seed/ID manifests.
Checkpoint models are small; do not multiply large image/reference assets across
conditions. Completed artifacts are not deleted to make a failing forecast fit.

Reserve final checkpoint/report space before launching work and stop cleanly
before exhausting it. Never silently drop required diagnostics, move data onto
the external drive, change precision, or exceed 500 GB to finish. The ongoing
Plan 14 archival job and its source/output directories are outside this task;
do not interrupt, edit, prune, or require them.

## Astra-To-Sol Responsibilities

| Astra before handoff | Sol after handoff |
| --- | --- |
| Develop the mathematical paradigm and initial plan | Reassess assumptions and correct mathematical defects when evidence warrants |
| Implement the planned methods, controls, and initial fallbacks | Run, debug, maintain, and extend the implementation as necessary |
| Validate mathematics, gradients, pairing, and lifecycle | Validate repairs and preserve interpretable comparisons |
| Test interruption, restart, reporting, and budget enforcement | Resume work, allocate remaining authorized compute, and monitor resources |
| Supply executable commands and a concise acceptance manifest | Keep results, significant changes, and continuation notes current |
| Version the initial scientific code and protocol | Version subsequent corrections and track affected evidence |

The user expects to be away on a business trip and must not be a recurring
model-switch or review dependency. Once execution is authorized, Sol has
authority to make implementation repairs, mathematical corrections, solver
replacements, and justified experimental revisions within the stated research
objective. This authority is not limited to prewritten fallbacks. Prefer the
planned rules when adequate, but do not leave the node idle awaiting Astra
because a productive correction was not anticipated.
The correction protocol governs necessary departures from phase-specific
defaults and frozen choices. Their purpose is to protect comparison provenance,
not to require external permission for repairs.

### Autonomous Correction Protocol

1. Diagnose the problem and preserve the affected artifacts. Stop or quarantine
   only work that cannot currently produce valid evidence; continue unaffected
   conditions or useful diagnostics where possible.
2. Implement the smallest justified correction, including a revised mathematical
   formulation or experimental choice when necessary. Add focused tests and an
   appropriately small numerical/smoke validation before substantial reruns.
3. Record material changes promptly in the notebook's significant-changes table
   and a linked versioned change record. State the reason, evidence, changed
   equations or behavior, validation results, effect on the estimand/comparison,
   affected artifact IDs, and updated time/storage forecast.
4. Preserve old completed units unchanged. Mark their interpretation in a new
   validity/lineage record as unaffected, invalid for a particular claim,
   exploratory, or superseded. A correction never turns an old result into a
   retrospectively corrected measurement.
5. Create a new scientific source/configuration identity for behavior-changing
   repairs. Resume from a checkpoint only when its mathematical and state
   compatibility has been established; otherwise restart affected paired units
   from valid shared inputs. Apply common changes to the relevant controls too.
6. Continue automatically within the remaining authorized resources. Reallocate
   remaining work toward valid, informative comparisons and record departures
   from the planned ledger. Material outcome-informed redesign is development;
   use fresh cohorts for any subsequent confirmation. Do not mix versions or
   select a stopping point merely to obtain significance.

Material changes include likelihood/Fisher corrections, gradient routing,
optimizer or retention behavior, memory lifecycle, data/evaluation protocols,
selection rules, and changes to the estimand. Pure presentation fixes need only
ordinary tested code history unless they alter reported interpretation. This
keeps the notebook useful without requiring an entry for every minor edit.

No intermediate user approval or return to Astra is required for this protocol.
The user can inspect the record, intervene later, or request reruns. Autonomy
does not authorize fabricated results, hidden data leakage, overwriting completed
artifacts, interference with unrelated work, exceeding the 500 GB cap, or silently
expanding the allocated compute budget. If no valid productive work remains
within those constraints, checkpoint and report the concrete blocker rather
than continuing known-invalid computation. This is a last-resort resource or
integrity stop, not a routine review gate.

Use Astra and Sol as the user's selected operating roles, not inferred model
capability guarantees. Record the actual model identifiers when available.
OpenAI's model guidance recommends testing the least costly setting that meets
the task's quality bar; the acceptance suite is the relevant test here [4].

## Proposed Code And Operator Interface

Keep all scripts and small configurations in the codebase:

- `mnist_experiment/fisher_memory/`: model adapters, curvature, replay/EWC,
  distillation, environment adapters, metrics, protocol, runner, and reporting.
- `mnist_experiment/fisher_memory/configs/`: smoke and production JSON contracts.
- `mnist_experiment/fisher_memory_results.ipynb`: concise artifact-only report.
- `mnist_experiment/PLAN15_HANDOFF.md`: short operator handoff, maintained during
  implementation and execution rather than embedded as a long plan status diary.
- `test/unit/test_fisher_memory_*.py`: focused mathematical/execution tests.
- `test/integration/test_fisher_memory_resume.py`: explicit CPU/CUDA pipeline drills.

Reuse tested local helpers without changing frozen Plan 14 scientific sources
or reviving its costly orchestration. The implemented interface is:

```bash
~/.venv/bin/python -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/smoke.json \
  --root cache/mnist_experiment/plan15/smoke_v1 --mode acceptance
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.run \
  --config mnist_experiment/fisher_memory/configs/default.json \
  --root cache/mnist_experiment/plan15/production_v1 \
  --allocation-hours <cumulative-hours> --resume
~/.venv/bin/python -m mnist_experiment.fisher_memory.control status --root <study-root>
~/.venv/bin/python -m mnist_experiment.fisher_memory.control stop --root <study-root>
~/.venv/bin/python -m mnist_experiment.fisher_memory.report --root <study-root>
```

Production commands use the tested CUDA environment in `~/.venv`; dataset and
artifact paths remain arguments, not personal absolute paths in source files.
Switching the assistant model does not itself start or authorize an experiment.
Omit `--resume` on the first invocation. Add `--clear-stop` when deliberately
resuming after a STOP request. The initial default allocation is zero, so a
production run cannot start accidentally. See [PLAN15_HANDOFF.md](PLAN15_HANDOFF.md)
for acceptance results, tested commands, and the current execution state.

## Provisional Wall-Clock Envelope

These are revised planning ranges informed by Phase 0's production-shaped
timing probes, not measurements of complete production trajectories. Account
for initialization, training, diagnostics, reports, validation, and restart
overhead, not just time in optimization kernels. Publish revised forecasts
after Phase 0 and Phase 1; the forecast is not permission to expand a budget.

| Phase | Operator | Expected active compute | Deliverable |
| --- | --- | --- | --- |
| 0 | Astra | Complete; final verification tasks total roughly 10 minutes of process time, partly overlapping, excluding implementation | Tested runnable pipeline and handoff |
| 1 | Sol | 0.5--2 h | Numerical/common-policy calibration and measured forecast |
| 2 | Sol | 4--8 h | Fair A/B/C and distillation pilots in all environments |
| 3 | Sol | 5--13 h | Fresh paired cohorts, including compute control |
| 4 | Sol | 0.1--0.5 h | Artifact synthesis and concise conclusion |
| Optional 5 | Sol | Measured and allocated separately | Additional replicas/method or positive-count extension |

The remaining study is provisionally **10--24 active compute hours**, revised
from the original 14--40-hour estimate. See the handoff for measured solve costs
and the limits of this extrapolation. Twenty hours remains the preferred scale,
but it is neither a promised runtime nor a reason to skip eviction follow-up
or give a method only a token trial.
Before production launch, record an explicit cumulative compute allocation with
the user; this draft does not interpret the travel discussion as automatic
authorization for four days. Once allocated, planned phases proceed without
intermediate user approval. Use runtime-based cohort sizing before confirmation
outcomes, not significance-based extensions afterward.

Most compute should occur under Sol: Phase 0 is deliberately small. A validated
autonomous runner matters more than agent polling frequency. Poll sparingly,
publish bounded health updates, and let the GPU run between substantive checks.

### Phase 0 Acceptance Checkpoint

Completed 2026-10-17. All A/B/C and control paths, both rotation schedules, and
the digit-9 mixture passed CPU/CUDA acceptance and interruption recovery. The
miniature scheduler completed calibration, screening, frozen selection, and
closeout using smoke configurations only. The results notebook is deliberately
concise and prominently labels these as software checks, not scientific results.

The implementation uses mathematically equivalent fixed 7x7 average pooling for
the canonical 28x28 inputs because PyTorch's adaptive-pooling CUDA backward is not
deterministic. Numerical stop/resource exceptions propagate without being
misclassified as invalid methods. Pre-eviction diagnostics, explicit one/two-buffer
follow-up, and nested initial-Fisher sensitivity checks are recorded.

The acceptance manifest and operator instructions are in
[PLAN15_HANDOFF.md](PLAN15_HANDOFF.md). No production study was launched and no
unattended Phase 0 compute remains running.

## References

1. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
   Plan 15 uses the additive factorization, but trains the base under EWC.
2. Schwarz et al., [Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html).
   Motivation for explicit transfer into protected long-term parameters; our
   replay-age distillation control differs from its task-based procedure.
3. Dinh et al., [Sharp Minima Can Generalize For Deep Nets](https://proceedings.mlr.press/v70/dinh17b.html),
   and Kunstner et al., [Limitations of the Empirical Fisher Approximation](https://papers.nips.cc/paper_files/paper/2019/hash/46a558d97954d0692411c861cf78ef79-Abstract.html).
   Parameterization and empirical-Fisher cautions, not validation of the proposed
   consolidation mechanism.
4. OpenAI, [Model selection](https://developers.openai.com/api/docs/guides/model-selection),
   consulted 2026-10-17. Model-role assignment is an operational choice; numerical
   acceptance and reproducible execution remain the scientific safeguards.
