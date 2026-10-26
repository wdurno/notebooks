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

**Status:** phases through 8 have completed execution records below. The user
authorized Phase 8.1 implementation and execution on 2026-10-25; recovery and
CPU/CUDA acceptance precede production. Unexecuted numerical settings are
prospective choices, not results. Live state is in the notebook and handoff.

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

The optional JVP/directional extension has its own entry point and run root.
Choose the method and environment lists **before** the first invocation; the
completed extension ledger freezes those lists, replica count, pilot-selected
strengths, reference-control seals, and paired asset-plan hashes. The source
`production_v4` root is read-only. Extension outcomes are secondary,
post-selection evidence and are not pooled into Phase 3 confirmation.

Launch from the repository root in `tmux` after choosing a scope and cumulative
compute allocation (omit `--resume` on the first invocation):

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.extension \
  --source-root cache/mnist_experiment/plan15/production_v4 \
  --root cache/mnist_experiment/plan15/extension_v1 \
  --methods <selected-methods> --environments <selected-environments> \
  --replicas 12 --allocation-hours <cumulative-hours>
```

Allowed methods are `jvp` and `directional`; allowed environments are
`rotation_linear`, `rotation_sigmoid`, and `digit9_mixture`. Keep the same scope,
source, and root on resume and add `--resume`. `--max-units` or
`--max-wall-seconds` can bound an invocation without changing the cohort;
`--clear-stop --resume` deliberately clears a cooperative STOP. The live
artifact-only notebook is `mnist_experiment/fisher_memory_extension_results.ipynb`.
The extension uses the shared 500 GB budget lock and checkpoints every accepted
step. A VS Code SSH disconnect does not terminate a runner launched inside a
remote `tmux` session; a machine reboot still requires `--resume`.

### Phase 5A: Sigmoid JVP Strength Reassessment (completed)

The first Phase 5 extension used JVP strength 0.0268905 for sigmoid rotation,
while the selected finite-difference KL coefficient was 26.8896597. Its weak
sigmoid result therefore does not establish that a sensibly penalized JVP is
inferior. The coefficient scales different penalties and is not an equivalence
calibration; the matched numeric value is a diagnostic anchor, not an assertion
of equal effective curvature. Do not use the earlier extension outcomes as
pilot data or pool them with this versioned repair.

Freeze a sigmoid-only development grid at factors
$(1/30,1/10,1/3,1,3)$ times the sealed finite-difference coefficient from
`production_v4` (approximately 0.8963, 2.6890, 8.9632, **26.8897**, and
80.6690). Run three new paired replicas per grid point, each sharing its own
fresh replay/EWC control. Select the coefficient with the largest mean paired
current-holdout NLL-AUC gain among points whose mean paired post-eviction
historical NLL-AUC harm is at most 0.01. The three-replica pilot makes a
choice, not a significance claim. If none meet retention, select the best
joint diagnostic fallback and label it as such.

Use a disjoint 12-replica confirmation seed family, again with fresh paired
replay/EWC controls. Confirm the pilot-selected JVP coefficient as the primary
repair comparison. Also confirm the fixed 26.8897 anchor as a secondary
diagnostic if it was not selected. Report both predictive endpoints, favorable
replica counts, nominal paired intervals, exact STM Fisher trace, solver health,
and measured compute. Retain the earlier finite-difference study as historical
context, not a same-seed paired comparison. No STM-to-LTM transfer claim follows
from a predictive gain alone.

The repair uses its own immutable root, a cumulative active-time allocation,
the shared 500 GB budget lock, step checkpoints, and `--resume`. A compact
artifact-only notebook at
`mnist_experiment/fisher_memory_jvp_repair_results.ipynb` publishes pilot and
confirmation progress while trajectories complete. Start in `tmux` from the
repository root (omit `--resume` initially):

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.jvp_repair \
  --source-root cache/mnist_experiment/plan15/production_v4 \
  --root cache/mnist_experiment/plan15/jvp_repair_v1 \
  --allocation-hours 6
```

For a later invocation, use the same arguments plus `--resume`; the allocation
is cumulative. `--max-units` and `--max-wall-seconds` bound an invocation.
The allocation of six active hours leaves headroom over the approximately
3.5--5 hour wall-clock forecast; it is a ceiling, not a target to consume.

Phase 5A subsequently completed all 42 trajectories in 2.92 active hours.
The pilot selected JVP strength 26.8897. In 12 fresh paired confirmation
replicas, replay/EWC minus JVP mean NLL-AUC gain was 0.1145 on the current
sigmoid holdout and 0.1359 on the post-eviction holdout, favorable in all 12
replicas on both endpoints. This repairs the weak-strength JVP result against
replay/EWC but does not provide a paired JVP-versus-finite-difference comparison.
All 59 units passed full artifact validation. The separate repair notebook
contains the complete readout; the earlier forecast above is retained as the
pre-run estimate.

### Phase 5B: Rare Rotated-Domain Drip Development (completed)

The question is whether finite-difference STM Fisher shrinkage improves
prediction of a rare new domain while novel labeled images are scarce. This
extends the continual-learning, limited-data and limited-compute motivation of
Plan 15. The development phase checks that the task has measurable acquisition,
calibrates one finite-difference coefficient, and measures its actual runtime.
Development results never enter Phase 5C inference.

Keep the Plan 15 architecture, 30,000-image all-digit unrotated initialization,
rank-two STM, FIFO capacity 32, Ridge, eviction-only LTM EWC, common repaired
optimizer ceiling, and rotation archive rate $\pi=0.025$. Freeze two domains:
$D_0$ contains unrotated MNIST images of all ten digits and $D_R$ contains the
same class distribution rotated by $30^\circ$. The previous frozen model had
mean NLL about 0.20 on $D_0$ and 1.39 on $D_R$; this motivates the fixed angle,
not a new angle search. A domain flag guides sampling and reporting but is never
an input to the model. Within each replica, use unique stream source-image IDs
disjoint from initialization and Fisher-reference training IDs. Draw evaluation
IDs only from the MNIST test split. Partition its per-class test IDs into
disjoint development and confirmation pools; replicas may draw from their own
phase's pool, and all four conditions in a replica use identical fixed panels.

Generate each replica's observation stream before fitting any condition. With
batch size four, set the rotated draw probability for arrival batch $j$ to
$p_j=0.1\min((j-1)/39,1)$: a 40-batch ramp from zero followed by a 0.10
plateau. Draw domain independently of digit label, retain the original ten-class
distribution, and stop **at the 100th unique rotated image**. Truncate the
final batch at that image, so every completed mixed trajectory has exactly 100
novel exposures. It has about ten per digit in expectation, without imposing
ten-per-class quotas. This schedule reaches 100 events around batch 270 on
average; a cheap 20,000-stream simulation gave 5th/95th percentiles near
232/311 batches (random seed 1515). Record the complete domain and label
sequence, realized per-class counts, base-image counts, batch sizes, source IDs,
and seeds.

The four conditions share initialization, source IDs, labels, arrival order,
optimizer settings unrelated to treatment, evaluation panels, and event times:

| Domain stream | Replay/EWC control | Finite-difference KL |
| --- | --- | --- |
| Rare $D_R$ mixed into $D_0$ | Main acquisition control | Main treatment |
| Every stream image left in $D_0$ | Base-only continuation control | Base-only penalty control |

For the base-only arms, use the **same** image IDs as their mixed partners and
leave images unrotated, including at the 100 scheduled rotated-event positions.
Those arms see zero rotated training images. The event positions supply only a
matched reporting clock. Their curves must say "paired rotated-event index",
never "rotated images observed". Use one selected KL coefficient in both
penalized arms. All replay and archive mechanics remain otherwise identical.

For each condition $m$, let $R_m^R(n)$ be mean NLL on one fixed, all-digit
$30^\circ$ test panel after the batch containing the $n$th rotated event in
the paired mixed stream. If a batch crosses several event counts, use its one
post-batch evaluation for each crossed count. Record $R_m^R(0)$ before online
training, then evaluate at all event counts $n=1,\ldots,100$. The primary
per-replica acquisition score is

$$
A_m^R(100)=\frac{1}{100}\sum_{n=1}^{100}R_m^R(n).
$$

Here "test panel" means a fixed, class-stratified sample from the official
MNIST test split: 100 images per digit per replica in the production
configuration. Every image is rotated by $30^\circ$ for the new-domain panel.
Within a replica, the frozen reference and all four adaptive arms use identical
test images, labels, rotations, and weights at every checkpoint. Each digit's
total weight is its frequency in the full MNIST training split, divided equally
among that digit's test examples; weights do not change with the stream.
Test examples never enter pretraining, online optimization, replay, Fisher
estimation, or curvature penalties. Confirmation test IDs are disjoint from
the Phase 5B development panels used for coefficient selection. Panels may
overlap between replicas; replicas, not repeated evaluations, are the
uncertainty units.

"Post-batch" means after finishing optimization on the entire incoming batch
of up to four training examples plus the resident replay examples, followed by
inference on the separate test panel. It does not mean evaluating loss on the
training batch. These NLLs exclude EWC, Ridge, and KL penalties. The frozen
reference is the initial model's measured NLL on that same **rotated test
panel**, not its unrotated pretraining loss. Its predictive NLL remains constant
without online updates, so its acquisition score equals its endpoint NLL.
For an adaptive model, endpoint NLL and acquisition score differ only in time
aggregation. Report equal-weight means of these per-replica quantities.

This is simply the average new-domain test NLL after seeing one through 100
novel images. Lower values reward good prediction throughout the scarce-data
period, even when final NLLs coincide. Use the same event checkpoints for the
fixed unrotated test panel $R_m^0(n)$ and its retention AUC $A_m^0(100)$.
The base-only conditions use their partner's event times for both panels;
their $n$ is a paired index, not an actual novel-exposure count. Track incoming
rotated-image prequential NLL, per-domain accuracy, solver health, replay and
eviction counts, STM diagnostics, optimizer closures, and wall-clock time as
secondary observations. For prequential NLL, predict each scheduled rotated
image before its update. A base-only arm may predict that rotated form for
this audit but trains on the unrotated form. Do not collapse acquisition,
retention, and compute into an unmotivated weighted score.

Implement a tiny cached-MNIST smoke and interruption/recovery check first.
Then run three control-only development replicas with both replay/EWC stream
types. Display the 30-degree frozen-to-final response, how much base-only
continuation changes 30-degree NLL, event-count distribution, runtime, and
storage in the live notebook. A flat response is reported as a limitation;
poor efficacy alone does not stop the predeclared experiment.

On four additional development replicas, run mixed replay/EWC and a broad
finite-difference coefficient grid
$\{0.00178,0.0219,0.269,2.689,26.89,53.63\}$. Its endpoints come from the
prior digit-9 and rotated-domain selections, with intermediate log-scale
probes. Select the coefficient with the largest mean paired gain in $A^R(100)$
among candidates with finite metrics, no recorded solver failures, and mean
harm to $A^0(100)$ at most 0.01 NLL relative to mixed replay/EWC. Record
under-solved updates without treating them as failed solves. Break exact ties
by lower measured online time, then lower coefficient. If all candidates miss
the retention margin or fail to improve acquisition, carry the best acquisition
candidate into confirmation and label the pilot warning; do not hide the
failure. If the entire grid is numerically invalid, repair and version the
pilot before confirmation. If the best candidate is at a grid boundary, test
exactly one further coefficient at one tenth of the lower edge or three times
the upper edge on the same four development replicas, then apply the same
selection rule. Finally, run the two base-only arms on those four replicas
using the selected coefficient. Seal the selection and all decisions before
Phase 5C starts.

### Phase 5C: Fresh Drip Confirmation And Reporting (completed)

Run 12 new paired replicas of all four conditions, with a seed family and
test-image pool disjoint from Phase 5B. Freeze the angle, schedule, event
budget, archive policy, selected coefficient, evaluation panels, and statistic
before viewing these outcomes. Every replica must complete its 100th rotated
event; a machine interruption leaves a resumable partial trajectory, not a
scientifically censored sample-efficiency value. Completed trajectories are
immutable and are never extended because an interval approaches zero.

The one primary paired contrast is
$D_r=A^R_{r,\mathrm{mixed,replay}}(100)-
A^R_{r,\mathrm{mixed,FD}}(100)$, positive when finite-difference KL predicts
better on the new domain during acquisition. Test $E[D_r]>0$ at one-sided
$\alpha=0.05$ with a prespecified paired sign-flip test; show the mean,
replica bootstrap 95% interval, individual differences, favorable count, and
standardized effect. The sign-flip result requires exchangeable signs under
the null; disclose material asymmetry rather than presenting the test as exact
under any data distribution. Pilot replicas are excluded from this test.

Show mixed-condition base-domain retention separately as
$A^0_{\mathrm{FD}}-A^0_{\mathrm{replay}}$ (positive means harm), with a
prespecified descriptive tolerance of 0.01 NLL and its 95% interval. A
nonsignificant difference is not a noninferiority result. As a secondary
diagnostic, compare the treatment gain in the mixed stream with its gain in
the base-only stream:

$$
\bigl(A^R_{\mathrm{mixed,replay}}-A^R_{\mathrm{mixed,FD}}\bigr)
-\bigl(A^R_{\mathrm{base,replay}}-A^R_{\mathrm{base,FD}}\bigr).
$$

Report this interaction and both constituent gains without claiming it proves
STM-to-LTM transfer. Include NLL and accuracy trajectories for both fixed test
domains, rotated-event prequential NLL, per-class event counts, final NLL, and
time/closure cost. New-domain acquisition is the primary result; retention and
cost determine whether a positive result is useful in the intended setting.

Add **one concise section** to `fisher_memory_results.ipynb`, after earlier phases,
with progress in its header. Explain $A^R(100)$ in plain language as "average rotated-domain test
NLL after each of the first 100 rotated training images" and state that lower
is better. Include one acquisition plot versus actual rotated-event count,
one base-retention plot at the matched checkpoints, and a small paired-effect
table with the 95% interval and measured compute. Label the base-only curves'
pseudo-clock explicitly. Publish sealed partial results during both phases;
the notebook only reads artifacts and never trains or repairs them. Keep
detailed checks in separate JSON. Do not make samples-to-target a primary
result or display only trajectories that reached a threshold. If a threshold
view is later added, show attainment fraction by event count and the fraction
still unreached at 100.

Use new immutable roots under `cache/mnist_experiment/plan15/` for development
and confirmation, the shared 500 GB admission limit, a frozen ledger, accepted
batch checkpoints, `--resume`, stop-file and signal handling, cumulative time
limits, and bounded report updates. A new protocol version isolates this
outcome-informed follow-up from the completed Plan 15 cohorts. The runner
continues without an assistant connection and records any material correction
in the live notebook. Phase 5B uses `fisher_memory.drip` under the separate
`cache/mnist_experiment/plan15/drip_development_v1` root. Its cached-MNIST
accepted-step pause/resume smoke and focused unit suite passed before launch.
Phase 5C is authorized for execution with the predeclared ten-active-hour cap.
Its separate `fisher_memory.drip_confirmation` entry point reads and verifies
the sealed Phase 5B selection, freezes a 12-replica ledger, and reuses the same
trajectory mathematics. Confirmation metadata and orchestration are versioned;
the numerical learner and initialization are unchanged. Run from the repository
root using:

```bash
~/.venv/bin/python -u -m mnist_experiment.fisher_memory.drip_confirmation \
  --source-root cache/mnist_experiment/plan15/drip_development_v1 \
  --root cache/mnist_experiment/plan15/drip_confirmation_v1 \
  --allocation-hours 10
```

Add `--resume` after interruption, and `--clear-stop` when deliberately resuming
after a STOP request. Checkpoints, the frozen cohort, and the compute allocation
survive restart. The notebook publishes descriptive progress; the final inference
is released after all 48 trajectories are sealed. No Phase 6 condition enters
this confirmation.

Phase 5B completed all 46 trajectories (including four predeclared boundary
probes) in 5.21 active hours. All 55 ledger, asset, selection, and trajectory
units passed full content-hash validation. The pilot selected
$\lambda_F=160.89$, the upper edge after its one allowed extension. On the four
screen replicas, mixed-stream FD KL gained 0.500 mean 30-degree NLL-AUC over
mixed replay/EWC, but base-only FD KL gained 0.413 on that same 30-degree
holdout without any rotated training images. The mixed-specific descriptive
difference was 0.087 NLL-AUC (3/4 replicas positive). These selected-on-pilot
effects are not confirmatory evidence; most of the treatment lift may reflect
protection or regularization rather than rare-domain learning. The concise
notebook and sealed selection artifact contain the full development readout.

Measured sigmoid trajectories with 120 batches took about 114 seconds for
replay/EWC and 206 seconds for finite-difference KL; their per-batch solve
times were nearly flat across the run. Scaling to roughly 270 batches gives
about 4.3 and 7.7 minutes per trajectory before asset, reporting, and restart
overhead. The new event-aligned score requires up to 100 evaluation checkpoints
per trajectory, more than the old every-fourth-batch schedule; its inference
cost is not isolated in those timings. The pilot will measure this cost and
replace the extrapolation with observed drip times.

| New phase | Forecast wall-clock compute | Basis |
| --- | --- | --- |
| 5B: development and strength pilot | 5--7 h | Three control-only pairs; four strength-screen replicas with six coefficients, then their selected-coefficient base-only pairs; includes smoke, assets, reporting, and bounded validation |
| 5C: fresh confirmation | 5--8 h | Twelve replicas x four conditions, roughly 270 batches per condition; includes assets, reports, and integrity audit |
| **Both phases** | **10--15 h** | GPU execution and validation; code-writing time excluded |

Proposed per-root active-time ceilings are eight hours for 5B and ten hours
for 5C, for at most 18 hours across the two new roots. This leaves room for
ordinary timing variation, not for extra replicas or a new outcome-driven
condition. Reforecast Phase 5C after 5B measures the
actual event counts and per-condition costs. If the forecast exceeds the
available allocation, preserve completed pilot artifacts and seek a separate
compute decision; do not shorten the 100-event confirmation post hoc.

Phase 5C completed on 2026-10-23: all 48 trajectories across twelve fresh paired
replicas finished in 4.59 active hours. Full content-hash validation passed all
62 artifact units, the sealed analysis reproduced exactly, and the frozen
scientific source hash remained unchanged. The root occupies about 284 MB.
There were no recorded solver failures or nonfinite trajectories; 8,318
under-solved updates remain disclosed diagnostics under the common ceiling.

The primary mean acquisition gain was **0.4619 NLL-AUC**, with a paired
bootstrap 95% interval **[0.3472, 0.5783]** and the prespecified one-sided
sign-flip **p = 0.00024414**; all twelve replicas favored FD. Mean acquisition
NLL was 2.1161 for bounded replay/EWC versus 1.6542 for FD. Mean base-retention
harm was -0.1566, with interval [-0.1898, -0.1238], indicating lower base-domain
NLL under FD. The secondary interaction was +0.1027, with nominal interval
[0.0512, 0.1585]. These results do not identify a memory-transfer mechanism.

Final rotated-domain accuracy averaged 61.98% versus 65.62% (descriptive).
Measured mixed-stream online time averaged 3.88 versus 6.88 minutes, a 1.77x
ratio. Thus the new-domain predictive advantage is confirmed at the same scarce
observation budget and resident replay capacity; the equal-compute comparison
and useful replay/FD allocation remain Phase 6 design questions. The concise
Phase 5C notebook section contains trajectories and the final readout. The
earlier wall-clock ranges are retained as pre-run forecasts. GPU work has stopped.

### Phase 6: Compute Allocation After Replay (provisional questions)

These are guiding questions for further experimental design; they are not
themselves an experimental design, a frozen protocol, or an additional compute
allocation. Phase 6 remains deferred, is not a prerequisite for Phase 7, and
must not start automatically when Phase 7 finishes. Keep Phase 5C unchanged.
Its acquisition result need not establish STM-to-LTM transfer or a positive
mixed-versus-base interaction to be
practically useful. Phase 6 asks how to learn effectively from scarce incoming
labeled observations when compute and resident replay are limited, including
after those observations leave the buffer.

The four paired Phase 5B screen replicas provide an initial cost reference:

| Mixed-stream treatment | Mean online active minutes | Seconds per incoming observation |
| --- | ---: | ---: |
| Replay/EWC | 3.974 | 0.224 |
| FD KL, $\lambda_F=160.89$ | 7.174 | 0.404 |

Both means cover the same 1,065.25 incoming observations per trajectory on
average, including exactly 100 rotated images. The ratio of aggregate online
times is 1.805: roughly 81% additional time, not a universal kernel-cost ratio.
These summary timers include fitting, eviction processing, and online evaluation,
but exclude shared initialization, checkpoint/report writes, and final validation.
Fit-only means were 3.832 and 7.030 minutes. The estimates are hardware- and
configuration-specific development measurements, not equal-compute comparisons.

Prioritize the following questions in order:

1. **What does FD cost, and where?** Measure per-arrival training, curvature,
   eviction/archive, and evaluation costs separately, with end-to-end experiment
   time reported alongside them. Report time per incoming observation and the
   actual replay/probe work; never treat optimizer iterations as equal-cost
   across methods. Establish a feasible resource envelope before selecting a
   predictive experiment grid.
2. **Does FD earn its cost at equal compute?** With the same stream, architecture,
   buffer capacity, and EWC policy, compare replay/EWC with FD under common
   per-arrival compute allowances. Give the baseline a genuine opportunity to
   use additional optimization on resident observations. A higher iteration
   ceiling alone is not evidence of matching actual expenditure: record early
   convergence and unused allowance, and do not manufacture useful work by
   forcing iterations after convergence. Resolve the budget-enforcement and
   accepted-update policy in development before confirmation.
3. **What replay/FD allocation works best?** Compare no FD, the current FD
   schedule, and a small set of intermittent schedules, for example one FD
   arrival in four. Spend saved allowance on replay optimization where useful.
   Calibrate penalty strength separately for each schedule; changing strength
   alone does not meaningfully allocate computation. Start with FD and one
   buffer capacity instead of crossing every method, probe count, cadence, and
   strength. Keep ordinary replay, EWC, and incoming-data access common.
4. **Where does the useful trade-off change?** After selecting promising
   allocations, vary compute allowance and a small number of buffer capacities
   separately. Ask whether FD can sustain useful new-domain prediction with
   less resident memory, or whether replay is preferable at a given budget.
   Larger buffers do not receive free replay computation. Only after this
   comparison consider another curvature method or a focused no-STM control;
   neither architecture attribution nor a broad method ranking is the first
   Phase 6 objective.

Retain new-domain NLL versus unique novel observations and its acquisition AUC
as the main statistical-efficiency view, with accuracy alongside it. Compare
performance at fixed compute allowances and report realized time and resident
bytes; avoid an arbitrary NLL/time ratio. Base-domain retention is a guardrail,
not a competing primary story. Comparable prediction with fewer resources is
a useful outcome, but a nonsignificant difference alone does not establish
comparability: freeze a practical degradation margin before such a confirmation.

Use bounded development followed by fresh paired confirmation, not a full
factorial search or extensions driven by approaching significance. Preserve
immutable resumable artifacts, the shared 500 GB cap, and a concise notebook
with a compact progress summary and results ordered oldest to newest. The intended output is a measured map of useful
replay/FD allocations at the tested budgets, not a universal optimum. Specify
the exact conditions, cohort, and phase-by-phase wall-clock forecast after
Phase 5C and the cost calibration; prefer a roughly 20-hour follow-up scope
unless a separate allocation is justified and agreed.

### Phase 7: Regularization, Numerical Adequacy, And Useful Acquisition (7A--7E completed)

**Status:** the user authorized implementation and execution of 7A--7D on
2026-10-23, using the proposed initial 26-hour aggregate active-time allowance.
Sol completed implementation and CPU/CUDA recovery acceptance (389 tests plus
focused reporting follow-ups), then executed all four sub-phases on CUDA.
Completion was 2026-10-24 12:42 UTC, with final validation/report refresh
afterward: about 9.44 aggregate active hours, 128 sealed trajectories, 48
sealed frozen-objective audits, and about 0.78 GB of artifacts. Full checksum
validation and all-stage completed-work `--resume` passed with all 201
development/confirmation unit seals unchanged. The final 43 reporting tests
also passed. Phase 6 remains deferred and has not started. Original forecasts
below are retained; results and tested restart commands live in
`PLAN15_HANDOFF.md` and the chronological Phase 7 notebook section.

Phase 7E below is an approved artifact-only interpretation follow-up. Its
implementation, CPU analysis, visual fit review, and notebook conclusion
completed on 2026-10-24. It does not reopen 7D's confirmation, authorize more
trajectories, or activate the deferred Phase 6.

**Outcome:** the audit did not trigger a solver repair (2/48 material objective
gaps, versus the predeclared 12/48 trigger). Both methods selected the strongest
tested ridge, $c=1$; the optimum is not bracketed beyond the grid. FD retained
$\lambda_F=160.89$ after the three-coefficient study; no edge extension ran.
In twelve fresh paired replicas, its primary acquisition gain over same-ridge
replay was $0.1302$ (bootstrap 95% interval $[0.1046,0.1571]$; one-sided
sign-flip $p=0.00024414$), favorable in 12/12 replicas. Its nominal acquisition
gain over freezing was $0.1200$ ($[0.0589,0.1880]$), favorable in 11/12.
Base-domain acquisition NLL still worsened by $0.0646$ from initialization
($[0.0588,0.0709]$), although less than replay. Mixed-stream FD cost about
1.98 times replay's online time. These are equal-data predictive results in
this tested regime, not equal-compute superiority, a samples-saved estimate,
or proof of STM-to-LTM transfer. The confirmation recorded no solver failures
or nonfinite trajectories, but 7,583 residual-based under-solved update flags;
the audit does not establish exact stationarity for every later solve.

**Motivation:** improve learning from scarce new-domain observations while
retaining useful predictions on older data, using the existing objective's
regularizers before introducing update damping or an early-stopping policy.
The completed Phase 5C comparison favored FD over replay/EWC, but its mean
acquisition NLL was 1.6542 versus 1.6068 for the frozen initial model. FD's final
rotated NLL was lower, 1.5596. In the base-only continuation controls, mean
base-domain acquisition NLL was 0.4371 for replay/EWC and 0.3240 for FD, versus
0.1822 initially; both deteriorated in all twelve replicas. These observations
motivate the investigation but do not prove overfitting, overconfidence, or an
STM-to-LTM transfer mechanism.

The drip experiment held ridge at $c=0.1$. Earlier common-policy fallback
calibration considered only $c\in\{0.01,0.1\}$, on the earlier environments.
Neither establishes a well-calibrated stronger-ridge baseline for this task.
Phase 7 is explicitly an outcome-informed follow-up, not a replacement analysis
of Phase 5C or a retroactive change to its confirmatory claim.

#### Shared Contract And Scope

Inherit the Phase 5B/5C drip protocol: a fixed $30^\circ$ novel domain, arrivals
of at most four images, novelty probability ramping from zero to 0.1 over forty
batches, and termination at exactly 100 unique rotated images. Keep the paired
base-only stream, whose corresponding images stay unrotated. It has zero novel
training exposures; its event index is only a matched reporting clock.
This phase does not rerun digit-9 mixtures or linear/sigmoid rotation schedules.

Keep 487 LTM and 176 STM coordinates, replay capacity 32, replay weight 0.5,
eviction-only model-label Fisher updates, rank-eight-plus-diagonal compression,
and $\pi=0.025$ fixed. There is still one likelihood Fisher. Use the existing
initialization/reference protocol and FD stencil $h=0.01$ with two fixed probes
per solve. The starting FD coefficient is $\lambda_0=160.89$. Controls have
$\lambda_F=0$ and exactly the same trainable coordinates and ordinary losses.
Do not search replay weights, $\pi$, architecture, buffer size, learning rates,
FD probe counts, or perturbation radii in this phase.

The varied ridge multiplier is $c$, not an absolute penalty coefficient:

$$
s_t=\frac{\operatorname{tr}\widehat F_{LL,t}}{d_L},\qquad
\kappa_t=c\,s_t,\qquad \omega=\frac{1-\pi}{\pi}=39,
$$

$$
R_{\mathrm{ridge},t}(c)=\frac{\omega\kappa_t}{2}
\left(\|\theta_L-\mu_t\|^2+
\|\theta_S-\theta_{S,t}^{\mathrm{start}}\|^2\right).
$$

The EWC term, age-weighted FD term, and anchor-update rules remain as specified
above. In particular, neither ridge anchor is a permanent initial-model anchor.
Hold both anchors fixed inside each solve; update the archive anchor only at
the existing eviction transaction. Record $s_t$, $\kappa_t$, and
$\omega\kappa_t$, because trajectories with the same $c$ can develop different
Fisher scales. Holding $\pi$ fixed avoids simultaneously changing EWC odds and
archive forgetting. A permanent-anchor experiment would be a separate design,
not an automatic extension of this grid.

Use four newly seeded development replicas shared throughout 7A--7C, and twelve
fresh paired confirmation replicas in 7D. Within a replica, every candidate
shares initialization, reference inputs, stream IDs/order, event times, test
panels, and named random-probe/compression seeds. A treatment must not change
the random stream that generates those common assets. Reuse completed 7A units
in 7B only when all scientific identities, including the solver policy, match.
Historical Phase 5 trajectories are context, not additional Phase 7 replicas.

Preserve the existing per-class development/confirmation test-pool split and
explicitly assign the new families to their intended partitions. The current
`data.py` chooses the confirmation half using a family-name allowlist; merely
inventing a new family name is insufficient. Test partition membership and
disjointness. Do not evaluate Phase 7 development candidates on its confirmation
partition. That partition has already been used in earlier studies, including
5C: describe 7D as fresh stochastic replicas on a reused MNIST benchmark, not
as a never-inspected external dataset. Record overlap with previous test panels.

Use the same fixed, class-weighted held-out panels and event-aligned definitions
$A_m^R(100)$ and $A_m^0(100)$ as 5B/5C. These are predictive NLLs without any
regularization terms. All candidates and the frozen reference in a replica
use the same images, transformations, labels, and weights. Initial evaluation
is excluded from the 100-event average. Multiple events in one batch share
its post-batch held-out prediction. No evaluated panel or evicted audit example
returns to optimization, Fisher estimation, or replay.

#### Phase 7A: Frozen-Objective Solve-Sensitivity Audit

**Question:** are the ordinary accepted updates practically close enough to a
solution that objective calibration, rather than more optimization, is the
appropriate next intervention?

Implement the Phase 7 runner, audit hook, selection logic, tests, and progressive
report before production. Run the four development replicas at $c=0.1$ with
replay/EWC and FD at $\lambda_0$, in both streams: **16 trajectories**. Start
with the actual drip policy, 100 L-BFGS iterations and 150 closure evaluations,
not the original 50/75 defaults. During each trajectory capture the update
containing paired event counts 10, 50, and 90: **48 diagnostic branches**.
The event rule depends only on the shared stream, never on observed loss.

For each captured update:

1. Preserve its pre-arrival learner, incoming/replay IDs and labels, ages, Fisher,
   both original anchors, FD probes/seeds, configuration, and ordinary accepted
   parameters. Preserve the usual solver diagnostics and objective components.
2. Reconstruct the same $J_t$ using `prepare_objective` with the **pre-arrival**
   parameters. Verify its value and gradient at the ordinary solution against
   the original closure before branching. Starting a new ordinary `fit_step`
   from the accepted parameters would reset the STM anchor and is not this test.
3. Starting at the ordinary solution, perform a diagnostic re-solve with fresh
   L-BFGS history and at most 200 additional iterations / 300 closures. Keep all
   objective inputs fixed. This is a bounded numerical probe, not a guarantee
   of global optimality. Report the history reset and the actual work used.
4. Evaluate ordinary and extended parameters on the same two **development**
   holdouts. Save total/component objective changes, gradient infinity norms,
   LTM/STM displacement norms, held-out NLL/accuracy changes, and mean predictive
   total variation $\tfrac12\sum_k|p_k^{\mathrm{extended}}-p_k^{\mathrm{ordinary}}|$.
   Average this last diagnostic using the fixed panel's observation weights.
   Distinguish fitted-batch/replay losses from these held-out quantities.
5. Discard the branch from the live learner: commit only the ordinary accepted
   update. Do not consolidate branch data, alter the FIFO, consume the live RNG,
   or count audit evaluations as extra training observations. Audit time is
   separately measured and excluded from normal online-method timing.

A small prediction change supports practical adequacy even if parameter changes
or raw gradient norms are large. Lower training objective with worse holdout
NLL supports an objective/generalization problem; improvement in both suggests
numerical work could matter. Neither interpretation is proof from four seeds.
Show all prespecified snapshots, not only ones that support a diagnosis.

Use one bounded, common numerical safeguard rather than a solver grid. Define

$$
g_J=\frac{J_t(\theta^{\mathrm{ordinary}})-
J_t(\theta^{\mathrm{extended}})}
{\max(1,|J_t(\theta^{\mathrm{ordinary}})|)}.
$$

If at least 12 of the 48 valid branches have $g_J>10^{-3}$, use the existing
larger-ceiling option, **200 iterations / 300 closures for both methods**, and
repeat the sixteen $c=0.1$ trajectories and their audits under a new policy
identity. This threshold is a numerical triage convention, not a significance
test or a claimed universal convergence tolerance. It never uses held-out NLL
to choose the ceiling. Preserve the first sixteen as diagnostic evidence and
exclude them from the subsequent grid. Freeze the resulting common policy
before 7B; do not keep increasing ceilings to get a favorable chart.

If material gaps remain after that one repair, continue finite, otherwise valid
comparisons with an explicit optimization-limited label. Do not claim that the
intended objective has been optimized to stationarity or disproved. Nonfinite
objectives, wrong gradients, or corrupt branches require the autonomous defect
correction protocol, not a numerical-adequacy label. Reforecast costs immediately
if the common ceiling changes; resource limits still apply.

**Deliverables:** sealed audit table, common solver-policy decision, one concise
objective-change versus held-out-change figure, and measured cost/storage
forecast. Missing audit snapshots must be regenerated in an incomplete/new unit,
not invented from the old runs' rolling checkpoints.

#### Phase 7B: Paired Ridge-Response Study

**Question:** can calibrating the existing ridge improve acquisition and base
retention, and how much of FD's relative advantage survives that calibration?

On the same four development replicas, cross

$$
c\in\{0.03,0.1,0.3,1\},\qquad
m\in\{\mathrm{replay/EWC},\mathrm{FD}(\lambda_0)\},\qquad
\text{stream}\in\{\mathrm{mixed},\mathrm{base\ only}\}.
$$

This is **64 trajectories total**, including the sixteen compatible 7A
trajectories, hence **48 new trajectories**. The grid gives a weaker reference,
the current setting, and two stronger settings. Do not automatically expand
the ridge grid: a boundary preference is a reported limitation and possible
future extension, not an invitation to an unbounded search.

Archive each trajectory's acquisition/retention scores, endpoint NLL/accuracy,
prequential rotated-image NLL, initial/frozen references, realized exposure
counts, solver health, time, and ridge-scale/parameter-motion summaries. Separate
post-fit incoming and replay losses from held-out losses. Audit these summaries
on both streams; base-only deterioration is a diagnostic, not the selection
objective. Do not choose a model merely because it moves least.

Use the following deterministic development selection rule in both 7B and 7C.
A candidate must complete all four replicas with finite metrics and no recorded
solver failures. Under-solving alone does not invalidate it. Define retention
harm relative to the **same** mixed-stream reference, replay/EWC at $c=0.1$
under the frozen Phase 7 solver policy:

$$
H(c,\lambda)=\frac14\sum_{r=1}^4
\left[A^0_{r,\mathrm{mixed},c,\lambda}(100)
-A^0_{r,\mathrm{mixed,replay},0.1}(100)\right].
$$

Among valid candidates with $H\le0.01$, choose the lowest mean mixed-stream
$A^R(100)$. Break exact ties by lower measured online time, then lower $c$,
then lower $\lambda$. If none meets the margin, choose the valid candidate with
lowest acquisition NLL using the same tie-breaks, label the retention warning,
and continue. If none is numerically valid, repair/version before selection;
never drop failed replicas to improve a candidate's mean. This inherited margin
only prevents further deterioration relative to the existing control. It is
not evidence that absolute retention is good or that freezing has been beaten.

Apply the rule separately to replay and FD. Let $c_R$ be replay's selected
ridge and $c_{F,0}$ FD's provisional ridge. Retain their union, at most two
ridge settings, for 7C. This gives FD its own promising setting while ensuring
it is also tried at the best calibrated replay setting. Show the entire grid
and paired differences, not only selected points. Four-replica intervals are
exploratory; selection does not turn them into confirmatory evidence.

**Deliverables:** sealed grid and shortlist, plus an acquisition-versus-retention
plot with frozen reference and measured time visible.

#### Phase 7C: Bounded FD Retuning At The Shortlisted Ridges

**Question:** does FD still help when its coefficient is calibrated for the
changed ridge, rather than inherited from a different objective balance?

At each shortlisted ridge, run

$$
\lambda_F\in\{\lambda_0/3,\lambda_0,3\lambda_0\}
=\{53.63,160.89,482.67\}
$$

on all four development replicas and both streams. The middle coefficient
already exists in 7B. This adds **16 or 32 FD trajectories** for one or two
shortlisted ridges. Use the mixed-stream selection rule above, never the
base-only response, to rank coefficients at each ridge.

If the selected coefficient at a ridge is the lower or upper boundary, allow
exactly one additional coefficient there: $\lambda_0/9$ or $9\lambda_0$,
respectively, on the same replicas and both streams. This adds eight FD
trajectories per affected ridge, at most sixteen overall. Do not extend again,
including when the new edge wins. Seal the extension decision before running
it; resumption must not rediscover and repeat it.

Select the final FD pair $(c_F,\lambda_F)$ from all completed FD candidates in
7B/7C using the same rule. Include the nonshortlisted original-coefficient grid
points so retuning cannot accidentally discard the earlier best feasible FD
candidate. The final replay setting remains $c_R$: replay has no FD coefficient
to tune. Both methods have been calibrated, but FD has a larger search space;
report the candidate counts and development compute rather than claiming equal
tuning expenditure.

Freeze the final recipes, common solver, twelve confirmation seeds, evaluation
partition, analysis, and treatment ledger before any 7D outcomes. A weak or
unfavorable development result is reported and carried into the fixed
confirmation, not used for significance-based stopping. The one exception is
an all-invalid comparison, which needs a recorded implementation repair.

**Deliverables:** sealed selection with all rejected/selected candidates and
warnings, extension history, confirmation ledger, and updated remaining-time
forecast. No further method, anchor, or hyperparameter search is implied.

#### Phase 7D: Fresh Confirmation And Learning-Efficiency Interpretation

**Question:** after calibration, do we obtain useful prediction from scarce
new observations, and is FD beneficial beyond an adequately calibrated ridge?

Run twelve fresh replicas in both mixed and base-only streams with these
recipes, all under the same frozen solver policy:

| Recipe | Purpose |
| --- | --- |
| Replay/EWC at $c_R$ | Independently calibrated practical control |
| FD at $(c_F,\lambda_F)$ | Independently calibrated treatment |
| Replay/EWC at $c_F$, only if $c_F\ne c_R$ | Same-ridge control to separate FD from the ridge-setting difference |

This is **48 trajectories if the selected ridges coincide, otherwise 72**.
Deduplicate an identical replay recipe; do not manufacture another stochastic
replica for the same arm. Include the shared frozen initial model as an
evaluation-only reference, with no training trajectory or fictitious compute.
No 7A--7C or Phase 5 replica enters these estimates.

The single primary contrast is the mean paired acquisition gain

$$
D_r=A^R_{r,\mathrm{mixed,replay}(c_R)}(100)
-A^R_{r,\mathrm{mixed,FD}(c_F,\lambda_F)}(100).
$$

It compares calibrated **recipes**, not a pure FD intervention when their
ridge values differ. Report mean, paired replica-bootstrap 95% interval,
individual differences/favorable count, and the predeclared one-sided sign-flip
test at $\alpha=0.05$, with its sign-exchangeability limitation as in 5C. Use
2,000 bootstrap draws and a separately named analysis seed. Release inference
only after the complete fixed cohort; no optional significance stopping or
additional replicas because an interval is near zero.

The same-ridge secondary contrast replaces $c_R$ in the control with $c_F$.
It isolates the effect of enabling FD under that common ridge/solver policy,
but still does not identify a transfer mechanism. When $c_R=c_F$, it is the
primary contrast, not a second independent finding. Show the retention harm
$A^0_{\mathrm{FD}}-A^0_{\mathrm{replay}}$ and its paired interval separately for
each relevant comparison. A nonsignificant harm is not proof of noninferiority.

Also show the absolute acquisition gain over freezing,
$R_r^R(0)-A^R_{r,m}(100)$, and base-domain deterioration relative to the same
initial model, $A^0_{r,m}(100)-R_r^0(0)$. These prevent a relative win over a
poor learner, or a model that has essentially stopped learning, from becoming
a claim of useful adaptation. They are secondary, nominal comparisons, not
extra opportunities to declare the primary hypothesis confirmed. An interval
overlapping zero means uncertainty, not equivalence. Report endpoint NLL and
accuracy as well as acquisition AUC; the primary score can worsen even when
the final model improves. Do not report an NLL percentage as samples saved.

Keep the notebook to **one chronological Phase 7 section**, approximately six
to eight cells and three multi-panel figures, with progress visible from 7A:

1. Numerical audit: extra objective reduction against held-out NLL change,
   marking stream/method and any common-ceiling repair.
2. Development: acquisition versus retention, with ridge labels, frozen origin
   (subtract each replica's own initial scores before averaging), and the small
   FD-retuning comparison. Keep the full candidate table in a linked artifact.
3. Confirmation: new-domain NLL and running $A^R(n)$ versus actual novel count,
   base-domain NLL, both domains' accuracy, paired gains/95% intervals, and
   measured online time. Put base-only controls in separate facets explicitly
   labeled with their paired-event clock. Do not overlay every pilot trajectory.

Explain the two acquisition comparisons in plain language: "better than
calibrated replay" and "better than learning nothing." Include post-fit versus
held-out loss and ridge/motion summaries in a compact expandable diagnostic,
not a wall of per-arrival output. Preserve the earlier phase sections unchanged.

Use these interpretations rather than an automatic winner label:

| Pattern | Supported next conclusion |
| --- | --- |
| Calibrated replay improves and FD's advantage shrinks | The original FD lift may partly reflect inadequate ordinary regularization; compare fresh common-ridge results before attributing it |
| FD retains an acquisition gain at the common ridge, with acceptable retention | Evidence of incremental predictive value in the tested regime, not proof of STM-to-LTM transfer |
| Both improve relative to replay but do not beat freezing on acquisition AUC | Relative protection remains useful evidence, but efficient acquisition is not established by this KPI |
| Stronger ridge mostly approaches freezing | Stability alone has not solved learning; an anchor/objective redesign is a later question |
| Numerical gaps remain material | Results concern bounded numerical implementations; do not make stationary-optimum claims |

Report per-method fitting/eviction/evaluation time and total study time, including
calibration, audits, initialization, checkpoints, reports, and validation.
Equal-data comparisons are not equal-compute comparisons. Do not automatically
start the deferred Phase 6 allocation experiments or a permanent-anchor study.

#### Phase 7E: Budget-Averaged Efficiency And Exploratory Sample Equivalence (completed)

**Purpose:** conclude the notebook's Phase 7 section with a confident statement
of budget-averaged efficiency and a clearly separate, preliminary estimate of
the observations needed to match the current model's predictive capability.
This is an outcome-informed, post hoc analysis of completed 7D artifacts, not a
new confirmation or training experiment. Approval of this phase freezes the
choices below; Sol should not need further design decisions to execute it.

**Data and scope.** Require the complete sealed 7D ledger/results and all twelve
paired replicas. Use only the mixed-training FD and replay/EWC recipes selected
in 7C, both at $c=1$, with FD strength $\lambda_F=160.89$. Preserve their fixed,
class-weighted $30^\circ$ test panel and predictive NLL without regularizers.
Do not pool development replicas or earlier phases. The unrotated-domain and
unrotated-only-control results remain retention/context evidence, not the
trajectories fitted for new-domain sample equivalence. There are no new model
evaluations, label acquisitions, hyperparameter searches, or GPU jobs.

**Established budget-averaged result.** Reuse the original sealed contrast and
interval without refitting $A(n)$, choosing a burn-in, or recalculating a primary
test. For $N=100$, explain

$$
A_m^R(N)=\frac1N\sum_{n=1}^{N}R_m^R(n),\qquad
\mathbb E_K[R_m^R(K)]=A_m^R(N),\quad
K\sim\operatorname{Uniform}\{1,\ldots,N\}.
$$

The random budget is independent of the learning trajectory. Averaging across
replicas gives the analogous population-budget interpretation. No monotonic
trend or i.i.d. continual-training stream is required for this identity.
Use a concise declaration such as:

> Under the tested mixed-stream protocol and 32-image replay budget, FD improves
> budget-averaged rotated-domain predictive NLL over calibrated replay/EWC by
> 0.1302 nats (paired 95% interval [0.1046, 0.1571]), with favorable differences
> in all twelve fresh replicas. This establishes a budget-averaged efficiency
> advantage over the first 100 new rotated observations in this tested regime.

Keep the positive nominal comparison against freezing visible. Also retain the
unrotated retention cost, approximately 1.98-times online compute, and the
best-tested ridge-grid boundary qualification. The 9.1% reduction in acquisition
NLL is not a percentage of observations saved or equal-compute superiority.

**One linear fit per condition.** Fit the two mean trajectories in **Mixed
training: Rotated test NLL**, not **Running A(n): mixed training**. Specifically,

$$
\overline R_m^R(n)=\frac1{12}\sum_{r=1}^{12}R_{r,m}^R(n),\qquad
f_m(n)=\alpha_m+\beta_m n,\qquad n\in\{1,\ldots,100\}.
$$

Use ordinary least squares with an intercept and equal weights across those
100 post-observation checkpoints, separately for FD and replay. Show $n=0$ as
the initial reference, but do not include it as an extra fitted observation.
There is **no $n>10$ restriction**, alternative fit window, spline, power law,
forced negative slope, or automatic model-selection procedure. Preserve the
existing event alignment: several novel images in one batch can share one
post-batch held-out prediction. The 100 checkpoints are not independent data
replicas, and conventional i.i.d.-residual regression intervals are inappropriate.

**Fixed matching target.** Anchor the comparison at replay's
fitted current-model test NLL after 100 new observations:

$$
q_{100}=f_{\mathrm{replay}}(100),\qquad
n_{\mathrm{FD}}^{\mathrm{eq}}
=\frac{q_{100}-\alpha_{\mathrm{FD}}}{\beta_{\mathrm{FD}}},\qquad
S=100-n_{\mathrm{FD}}^{\mathrm{eq}},\qquad
P=100\left(1-\frac{n_{\mathrm{FD}}^{\mathrm{eq}}}{100}\right)\%.
$$

Here $S$ is the model-implied number of new rotated observations saved, and $P$
is its percentage relative to replay's 100-observation budget. This is one
endpoint-anchored comparison, not a search over performance targets. Use the
**fitted**, not observed final-checkpoint, replay NLL consistently. Fractional
counts are linear-model interpolation; calculate before rounding and describe
them as approximate. This targets the inverse of the mean learning curve, not
the average of individual replicas' first-passage times or a guaranteed
per-trajectory sample requirement.

Check the shared observed mean initial rotated-test NLL against $q_{100}$.
If initialization already meets or beats this target, label it **already met
without adaptation** and withhold a new-learning sample-saving claim. Do not
replace it with a more favorable target. The comparison is between fitted
post-arrival trends, not proof that replay actually needed 100 observations
to reach that NLL for the first time.

Only interpret the point estimate if both fitted slopes are strictly negative,
the fitted NLLs are finite and nonnegative on the fit interval, and the matching
FD budget lies in $[1,100]$. A nonnegative/degenerate slope or out-of-range
crossing is a reportable limitation, not a reason to clip a value, assume a
saving, extrapolate beyond 100, or choose another cutoff. A negative fitted
slope is a descriptive trend under the linear approximation, not proof of
monotonicity of the underlying expected risk or of a good fit.

**Uncertainty and qualifications.** Use 2,000 bootstrap draws of the twelve
whole paired replicas with a new, independently named analysis seed. Recompute
both mean trajectories, both fits, $q_{100}$, and the crossing within every
draw; do not hold the estimated replay target fixed. Recompute the shared
initial reference as well, and apply the same slope, finite/nonnegative-loss,
and already-met-target checks to each draw. Keep pairing and the full
within-trajectory dependence. Do not bootstrap checkpoints independently,
report slope-significance tests, or create a new confirmatory p-value.

Retain every draw and its status; failed checks make a draw inversion-invalid.
If all inversions are well-defined, form
nominal percentile intervals without discarding finite out-of-range crossings.
If any draw is inversion-invalid, report its count and mark the ordinary inversion
interval unavailable; never calculate a 95% interval conditional on successful
draws alone. If the point or interval extends outside the observed budget range,
label the sample-saving claim range-limited rather than clipping bounds or
presenting extrapolation as observed evidence. Preserve numerical outputs and
failure categories in the analysis artifact even when a headline estimate is
withheld. Bootstrap uncertainty describes replica variability conditional on
the fit, not uncertainty from linear-model misspecification or post hoc design.

Interpretation assumes a fixed target holdout population, independently sampled
paired replicas, and adequate moments of predictive log loss for uncertainty
estimation; NLL is not automatically bounded. The quantity counts **new rotated
observations**, in addition to shared pretraining and the prescribed unrotated
background stream. Do not describe it as total training data saved, guaranteed
future performance, or proof of STM-to-LTM transfer.

**Human-facing conclusion and diagnostic.** Append a compact **Conclusion** at
the end of the existing Phase 7 notebook section, retaining earlier phase order.
Allow at most three additional cells: the established statement, one diagnostic
figure, and a short preliminary interpretation. The figure overlays the two
observed mean $R(n)$ trajectories with their two straight-line fits over
$1\le n\le100$. Label actual new rotated observations on the horizontal axis
and rotated-test NLL in nats on the vertical axis. Identify the methods, means,
and fits in the legend; mark the matching target and within-range crossing
without adding a wall of numbers or bootstrap-path overlays.

The diagnostic is the intended visual check of the linear approximation. Do
not add automatic goodness-of-fit gates or alternative model searches. If the
fit is visibly poor, flag that issue and withhold a persuasive samples-saved
claim while still publishing the figure and established budget-averaged result.
Complete this reporting phase without waiting for user review of the plot;
any corrective model/window change is a separately recorded follow-up, not a
silent improvement of an unfavorable fit. Label any supported numerical result
**exploratory linear-model sample equivalence**, with a nominal interval when
available, motivating further work rather than validating actual future savings.

**Implementation, recovery, and forecast.** Put fitting/bootstrap work in a
small standalone CPU entry point, not notebook execution. Prefer a module such
as `mnist_experiment.plan15_efficiency` outside the frozen `fisher_memory`
scientific-source fingerprint; reuse its artifact and report helpers without
editing the completed learner, scheduler, or scientific-source exclusions.
Write to a new analysis-only root such as
`cache/mnist_experiment/plan15/regularization_efficiency_v1`, with its own
source/configuration hashes, schema/protocol version, named seed, and dependency
completion seals. This preserves 7A--7D source identity and completed evidence.

Provide `--resume`, atomic completion, the shared storage-budget lock/reservations,
and cooperative interruption. Persist the resolved analysis and bootstrap
indices/seed so an interruption reproduces the same draws; repeating an
unfinished small analysis unit is sufficient. Skip completed analysis on resume.
The notebook only renders sealed scalar/fit artifacts and fails clearly on
incomplete or incompatible dependencies; it never trains or performs bootstrap
inference. Charge runtime/storage to the study and retain the 500 GB shared cap.

Before publication, test exact linear recovery and target inversion, the full
fit window with initialization excluded, paired deterministic resampling and
target recomputation, invalid-slope/crossing handling, resume equivalence, and
unchanged scientific completion seals. Verify chronological compact rendering,
no training/inference during notebook rendering, and preservation of all
existing plots and the original 7D inference.

**Expected cost:** approximately **1--5 minutes of CPU wall-clock compute** for
validation, fits, bootstrap, tests, and reporting; **zero GPU training time** and
no new trajectories. Allow roughly **20--45 minutes of implementation/checking**
separately. Reserve at most 100 MB of additional working/report artifacts within
the shared cap. The user approved execution on 2026-10-24. Implementation and
110 focused unit/recovery/reporting tests passed before analysis execution.

**Execution record (2026-10-24).** Completed the CPU-only entry point
`mnist_experiment.plan15_efficiency` and its separate artifact-only renderer;
the final focused suite passes 111 tests. The new analysis root is
`cache/mnist_experiment/plan15/regularization_efficiency_v1`. It seals the
resolved configuration, named seed and all 2,000 paired bootstrap index sets,
dependency seals, both point fits, every draw/status, and a separate visual
fit review. No training, model evaluation, or original inference was repeated.
All 201 scientific dependency completion seals remain unchanged.

Replay's fitted target is $q_{100}=1.3190592$ nats. The FD fitted mean crosses
at approximately **43.25** new rotated observations, a model-implied saving of
**56.75 observations (56.75%)** relative to replay's 100. Both point slopes are
negative and the crossing is inside the observed range. Visual inspection
supports a coarse overall linear trend, with early overshoot and local
reversals, not a monotonic or first-passage model. These numbers are exploratory
sample equivalence, not validated future observations saved.

**No ordinary 95% inversion interval is reported:** two of 2,000 draws already
meet the fitted replay target at initialization and therefore fail the agreed
new-learning guard. All draws are retained; none are selectively removed.
There are no out-of-range finite crossings. This does not alter 7D's established
budget-averaged gain or its original interval. The notebook adds exactly three
cells after the existing Phase 7 results; every pre-existing cell and original
inference is preserved. Analysis/report execution takes seconds of CPU time;
the focused test invocations take approximately two minutes altogether, with
zero GPU training and far less than the 100 MB additional artifact allowance.
See `PLAN15_HANDOFF.md` for the tested resume command and final accounting.

#### Sol Implementation And Recovery Checklist (Phases 7A--7D)

Keep the implementation small: extend the existing `DripRunner`, `Treatment`,
`Store`, and report helpers where compatible, with a separately versioned Phase 7
scheduler and report module rather than copying an entire experiment framework.
Suggested entry point: `mnist_experiment.fisher_memory.regularization`, with
`--stage all|development|confirmation` and a sealed `--source-root` for standalone
confirmation.
These names are a proposed interface, **not commands that exist yet**. Development
must advance 7A--7C autonomously; the confirmation command implements 7D and
verifies its development dependency before execution. The all-stage entry point
advances through 7D only after successful development completion, without an
assistant connection; a pause/failure must not launch the next stage.
Document tested commands
in the handoff after implementation, using `~/.venv/bin/python`.

Use new roots such as `plan15/regularization_development_v1` and
`plan15/regularization_confirmation_v1` under `cache/mnist_experiment/`, with
new seed families, schema/protocol identities, and the common Plan 15 budget
lock. Never resume Phase 5 artifacts under changed scientific sources. Seal
the common-policy, shortlist, extension, and final-selection decisions so a
restart cannot reselect them. Source-hash changes require new scientific roots;
record compatibility when importing unchanged dependency assets read-only.

Save selected frozen-objective snapshots before running their branches. Audit
branches are separate immutable units with parent-state hashes and their own
RNG/timing metadata. Checkpoint between the bounded branch solves; an interrupt
may repeat one unfinished diagnostic solve, not its parent trajectory. Persist
ordinary accepted updates independently so a pending audit cannot change replay
or cause duplicate consolidation. Retain only these selected snapshots and the
normal rolling recovery checkpoints, not every historical Fisher or input bank.

Require `--resume`, cooperative signals/STOP, persistent cumulative active-time
limits, per-invocation limits, and completion-skipping. Freeze prospective jobs
and run them round-robin by replica, alternating methods and stream types, so
partial progress is useful on both streams. Deduplicate scientifically identical
jobs across sub-phases. Report completed/planned units, audit branches, pending
decisions, active and estimated remaining time, storage, and numerical validity.
Refresh the artifact-only notebook on paired completions and a bounded
15-minute heartbeat; skip redundant historical report rebuilds on resume.

Before production, tests must cover:

- the exact $c\,s_t$ ridge and unchanged EWC/anchor semantics;
- identical frozen objective values/gradients before and after reconstruction;
- unchanged main-trajectory scientific states/predictive metrics with audits
  enabled versus disabled (timers are necessarily different);
- independently checkpointed audits and no duplicate replay eviction on restart;
- development/confirmation partition assignment and paired seeds/assets;
- deterministic selection, ties, warnings, one-edge extensions, and recipe deduplication;
- immutable completion seals, resource/STOP handling, and exact CPU/CUDA
  interrupted-versus-uninterrupted smoke results with cached MNIST;
- chronological, bounded notebook output, no training during rendering, and
  refusal to publish final inference from an incomplete confirmation ledger.

Sol may fix genuine defects under the existing autonomous-correction rules,
without waiting for Astra. Record material changes and isolate their evidence.
Poor efficacy alone does not license redesigning a completed confirmation.
The hard shared **500 GB** peak-storage cap and filesystem reserve remain in
force. These small-model runs should require only a few additional GB; reserve
a conservative 5--10 GB initially and replace that estimate with 7A measurements,
including branch snapshots and temporary/report copies. Do not delete old evidence.

#### Phase 7 Wall-Clock Forecast

Phase 5C measured 3.88 online minutes per mixed replay trajectory and 6.88 per
mixed FD trajectory; its complete 48-trajectory study took 4.59 active hours.
Use these as proxies, not guarantees for stronger ridge or larger FD strengths.
For example, 7B's 48 additional trajectories contain 24 of each method, about
4.30 online hours before overhead. 7C adds 16--48 FD trajectories, about
1.83--5.50 online hours. All ranges below include initialization, checkpoints,
reporting, audit/validation, and modest timing variation rather than GPU kernels
alone. Shared assets and compatible 7A trajectories are counted only once.

| Sub-phase | New production trajectories | Expected active wall-clock compute |
| --- | ---: | ---: |
| 7A: numerical audit and smoke acceptance | 16, plus 48 short diagnostic branches | 1.5--2.5 h |
| 7B: ridge-response study | 48 beyond compatible 7A runs | 4.5--6.5 h |
| 7C: FD retuning and at most one edge extension per shortlisted ridge | 16--48 | 2--6.5 h |
| 7D: twelve-replica confirmation and synthesis | 48 or 72 | 5--8 h |
| **Ordinary-policy total** | **128--184, plus audit branches** | **About 14--24 h** |
| 7E: artifact-only efficiency conclusion (additional) | 0 | 1--5 min CPU; no GPU training |

The sum of unrounded lower bounds is 13 hours; quote 14--24 hours as the practical
planning range, with roughly 18--20 hours as a central estimate. Programming is
separate: provisionally 3--6 hours of engineering elapsed time for Sol, with
smoke/test process time included in 7A above. Neither estimate is a measurement
of the not-yet-built Phase 7 pipeline or a promise of statistical significance.

The common-ceiling repair is a real contingency, not hidden in that range.
It adds sixteen replacement development trajectories and their audits, and may
increase subsequent solve costs. A provisional repaired-policy envelope is
**20--40 active compute hours**; replace it with measured timing immediately
after the repeat audit rather than assuming twice the ceiling means twice the
runtime. Do not spend extra iterations just to exhaust an allowance.

The initial **26-hour aggregate active-time allowance** across the two
Phase 7 roots was authorized with execution on 2026-10-23. Sub-root CLI
ceilings must share or partition that allowance, never each receive 26 hours.
Include superseded audit/repair roots in cumulative Phase 7 expenditure.
If the measured repair/grid forecast cannot fit, preserve progress and report
the additional time needed; do not silently consume the contingency, reduce the
confirmation cohort, shorten trajectories, or drop a control. Halt/resume time
while the node is repurposed is not active compute. The original plan amendment
itself did not launch computation; the later execution request authorizes it.

### Phase 8: Regression, Resource Allocation, And Sample Efficiency (completed)

**Completed 2026-10-24.** Phases 8A--8E are complete under the repaired v2
protocol, including twelve fresh paired confirmation replicas. Actual aggregate
active compute was approximately **2.93 hours**, including superseded work,
repairs and validation, below the 20-hour ceiling. The execution record below
supersedes the initial timing forecast; no additional cohort was launched.

**Motivation:** find useful continual-learning policies when new labeled
observations and computation are both scarce. First select how much information
to keep replayable and how much replay to perform; then measure the selected
policies' acquisition per new observation while checking retention. The target
is good enough prediction within an explicit resource envelope, not full-network
performance, minimum curvature, or proof that information moved between blocks.

Phase 7 motivates this study but does not answer its resource-allocation question.
Its approximately 1.98-times online cost and exploratory sample equivalence
belong to a categorical, fixed-angle experiment. Neither ratio is assumed here.
Regression makes errors in the prediction itself visible: a lower Gaussian NLL
with fixed variance necessarily means lower squared angle error. It does not
guarantee that the new task is easier or that the earlier advantage will recur.

**Scope and authority.** Astra authors the paradigm and implements/tests it in
8A. After the user-managed model switch, Sol owns 8B--8E, including calibration,
execution, corrections, reporting, and closeout, without intermediate review
gates. Target at most **20 aggregate active wall-clock compute hours**, including
8A test/smoke processes and all later development, repairs, validation, and
reporting. Programming time is estimated separately below. This is a new Phase 8
allocation, not a renewal of Phase 7's 26 hours. The existing **500 GB shared
Plan 15 peak-artifact cap** remains in force.

This phase operationalizes selected Phase 6 questions; it does not activate
that provisional phase wholesale. Do not automatically rerun categorical digit-9,
fixed-angle, linear/sigmoid, directional-KL, JVP-treatment, or distillation studies.
The following phase-specific regression contract supersedes classification
defaults above and in the older package instructions, only for new Phase 8 roots.
Preserve all existing results and implementations.

#### Regression And Data Contract

Predict the **applied rotation angle**, not digit identity or an annotation of
the handwriting's intrinsic orientation. All ten digits remain included, with
the same fixed training-set class proportions in both domains. Ambiguous shapes
and natural handwriting slant create irreducible uncertainty; neither removing
difficult digits nor clipping bad predictions is an automatic response to it.

The base domain is near-upright throughout pretraining **and** online learning:

$$
Q_0=\operatorname{Uniform}[-10^\circ,10^\circ],\qquad
\alpha_t\sim(1-p_t)Q_0+p_tQ_R.
$$

For $Q_R$, draw a fair sign independently of $Z\sim\mathcal N(0,1)$, set
$M=40^\circ\exp(0.25Z)$, reject and redraw magnitudes exceeding $90^\circ$,
and use $\alpha=\pm M$. Reject rather than clip: there must be no artificial
probability atom at either endpoint. The modes are approximately
$\pm37.6^\circ$, the central 95% magnitude interval is approximately
$[24.5^\circ,65.1^\circ]$, and upper truncation removes about 0.059% of the
untruncated magnitude distribution. The small overlap with $Q_0$ is allowed.
Count novelty by the recorded generating component, not by an angle threshold.

With angles expressed numerically in degrees, its density is

$$
q_R(a)=\frac{\mathbf 1\{0<|a|\le90\}}
{2|a|s\sqrt{2\pi}\,\Phi(z_{\max})}
\exp\!\left[-\frac{\log^2(|a|/40)}{2s^2}\right],
\qquad s=0.25,\quad z_{\max}=\frac{\log(90/40)}s.
$$

Here $s$ is the lognormal shape, not the Fisher trace scale used below.
Publish the two component densities on one labeled figure, including the
near-upright band and truncation annotation. Use one frozen rotation renderer,
interpolation, padding, normalization, and sign convention for all arms; no
angle/domain flag or digit label is a model input. Include example transformed
digits in the linked diagnostics to check border/resampling artifacts.

Generate source IDs, domain flags, digit strata, angles, and arrival order before
training. Start with the earlier rare-domain schedule on nominal groups of four:
$p_j=0.1\min((j-1)/39,1)$, followed by the 0.1 plateau. Stop exactly at
$N=100$ unique novel source images, approximately ten per digit in expectation,
not a class quota. Keep all intervening near-upright arrivals. Record realized
counts rather than substituting expectations in efficiency plots.

**Exact sample-budget checkpoints.** Split each nominal four-image group after
every novel arrival, also retaining any trailing base-only segment. Train on
each resulting segment before proceeding. Thus every novel event ends an
accepted update and its evaluation has seen exactly $n$ novel observations,
never $n+1$ hidden in the same batch. Do not change $p_j$ when a group is split;
it belongs to the original pre-generated arrival slots. All conditions share
these segments. This changes Phase 7's post-batch alignment deliberately, and
its extra solver cost belongs in calibration. Save cumulative total arrivals,
novel counts, base counts, and work at each checkpoint.

Use disjoint source-image pools for pretraining, initialization validation,
initial Fisher inputs, online learning, and held-out evaluation. One source ID
appears at most once in a trajectory, even with different rotations. Partition
the official test IDs into explicit development and confirmation pools; never
select the pool implicitly from an unfamiliar family name. Within a replica,
use the same fixed source IDs, angles, and class weights for every method and
every checkpoint, with separate $Q_0$ and $Q_R$ panels. A default of 100 test
images per digit per panel is adequate to start; any pilot revision is common
and frozen before confirmation. Earlier phases inspected MNIST test data, so
describe this as fresh stochastic confirmation on a reused benchmark, not a
never-inspected external validation set.

#### Model, Likelihood, And Consolidation

Use the existing smooth convolutional trunk, SiLU, average pooling, and LoRA
residuals, but a scalar linear regression head from initialization onward.
There is no categorical pretraining, head swap, softmax gauge, or categorical
Fisher reuse. Keep rank two for convolution adapters and rank one for the
one-output head, with multiplier $\alpha_\ell/r_\ell=1$, random $A$ and zero
$B$. This gives **287 LTM + 135 STM = 422 coordinates**; assert the counts and
record the new parameter ordering. Both blocks train online in every adaptive
arm, including controls. No architecture search is planned.

Let $f_\theta(x)$ predict the angle in units of $a_*=45^\circ$. Use

$$
\alpha\mid x,\theta\sim\mathcal N(a_*f_\theta(x),a_*^2),\qquad
L_i=\tfrac12(\alpha_i/a_*-f_\theta(x_i))^2
+\log(a_*\sqrt{2\pi}).
$$

The fixed scale is a declared likelihood convention, not an estimated noise
standard deviation. Do not learn variance or tune units separately by method.
Output means remain unbounded; report errors in degrees without clipping.
Model misspecification, including bounded true angles, is a limitation rather
than a reason to reinterpret observed-label gradient products as the Fisher.

For $J_i=\nabla_\theta f_\theta(x_i)$, exact integration over this likelihood's
model targets gives the per-input contribution $F_i=J_iJ_i^T$. Its LTM and STM
principal blocks belong to the single likelihood Fisher. In particular,
$\nabla L_i\nabla L_i^T=(f_i-\alpha_i/a_*)^2J_iJ_i^T$ is **not** an interchangeable
Fisher contribution. Use exact model-target integration, not sampled labels,
batch-mean outer products, or old classification artifacts.

Pretrain LTM under this regression likelihood on $Q_0$, keeping STM at its
initial zero residual. Start from 30,000 unique images with one fixed sampled
angle each, Adam at $10^{-3}$, batch 256, and 15 epochs. Assess initialization
on a separate near-upright validation pool against constant-zero prediction;
Sol may repair training or increase its adequacy within the common budget.
Do not select initializations using novel-domain confirmation performance.
Freeze the common recipe after 8B and repeat it with each replica's named seeds.
Estimate the initial Fisher on independent near-upright inputs, starting with
1,024 and a nested-prefix sensitivity check. Initial assets are shared within
each paired replica. Report pretraining observations and compute separately:
the later sample-efficiency claim concerns marginal novel observations, not
training the entire predictor from 100 images.

FD uses the existing symmetric perturbation convention, now with Gaussian KL.
For $f_\pm=f_{\theta_L,\theta_S\pm hv/2}(x_i)$,

$$
q_{\mathrm{FD},i}(v)
=\frac{\operatorname{KL}(p_+\Vert p_-)+\operatorname{KL}(p_-\Vert p_+)}{h^2}
=\frac{(f_+-f_-)^2}{h^2},\qquad
\Phi_i=\frac{1}{K d_S}\sum_{k=1}^{K}q_{\mathrm{FD},i}(v_k).
$$

Differentiate both branches into both memory blocks. Start with two Rademacher
probes and $h=0.01$; numerically calibrate $h/2,h,2h$ against
$(J_{S,i}^Tv)^2$ and float64 gradient checks. This JVP is a small numerical
reference, not a competing treatment. Freeze probes inside a solve and record
full perturbation norms. The old classification coefficient 160.89 is context,
not a calibrated regression coefficient.

Let $C$ be FIFO capacity and $b$ the maximum number of resident examples sampled
uniformly without replacement **once per accepted-update solve**. Require
$1\le b\le C$ in replay arms and use $S_t$ of size $\min(b,|R_t|)$. Freeze
that subset throughout line-search closures. Both ordinary replay and the FD
penalty use $S_t$, not an unmetered sweep over the remaining buffer:

$$
J_t=\tfrac12\overline L(B_t)+\tfrac12\overline L(S_t)
+R_{\mathrm{EWC},t}+R_{\mathrm{ridge},t}
+\frac{\lambda_F}{|S_t|}\sum_{i\in S_t}a_{t,i}\Phi_i.
$$

With empty replay, use only $\overline L(B_t)$ for the data term and zero FD.
Retain the arrival-based age rule $a_{t,i}=\min(1,(n_t-n_i)/C)^2$; $n_t$ here
counts **all** arrivals, not just novel ones. Uniform subsampling preserves the
full-buffer mean objective in expectation at fixed parameters; it does not
make the optimized random-subset solution identical to a full-buffer solve.
Varying $b$ changes evidence per solve and work, not the 0.5 replay weight.
Record repeated closure presentations: $b$ is not a count of total passes.
Use an independently named replay-sampling seed: identical-$C,b$ pairs receive
identical subsets, and different $b$ values at the same $C$ use prefixes of the
same random resident ordering. Replay, probes, compression, and solver behavior
must not advance the random generators that define the common data stream.

Keep eviction-only, rank-eight-plus-diagonal LTM EWC, $\pi=.025$, its anchor
update, and the step-anchored STM ridge. Write the ridge coefficient as
$\kappa_t=c\,\operatorname{tr}(\widehat F_{LL,t})/d_L$ and report its realized
value. Calibrate the shared multiplier $c$ below. Retired observations enter the
archive once at accepted parameters, whether or not selected in the final
replay subset. For the archive update use
$\eta_t=1-(1-\pi)^{e_t/4}$: four is the fixed nominal exposure unit, not the
variable segment length. No eviction means no archive/anchor change; no final
forced flush. All retirement Fisher work is charged. Offloading an eligible
example is not retirement.

The no-replay boundary has $C=b=0$, no age division, no FD, and immediate
retirement after fitting each new segment. It retains the same LTM/STM model,
ridge, and initial/online EWC. Do not interpret it as an unregularized learner.

#### Controls And Resource Envelope

| Condition | Allocation | Role |
| --- | --- | --- |
| `fd_replay_ewc` | Selected $C,b$, calibrated FD strength | Proposed policy |
| `replay_ewc` | Independently selected $C,b$, FD disabled | Primary resource-constrained comparator |
| `ewc_only` | $C=b=0$ | Scarce-resource boundary |
| `frozen_regression` | No online fitting | Common initial capability reference |

All adaptive conditions receive the identical mixed stream. There are no
base-only continuation arms. Same-$(C,b)$ development pairs isolate the added
FD term under the shared regularization recipe; the independently allocated
confirmation compares complete policies under the same constraints. Do not
call the former compute-matched or the latter an isolated penalty effect.

Fix FIFO, uniform sampling, archive rate/representation, FD cadence, and probe
count after calibration. Search only $C,b$ in 8C. Logical capacity constrains
eligibility for replay, not necessarily VRAM. Assume input supply can overlap
compute and omit disk streaming/prefetch engineering; report this optimistic
input-supply assumption explicitly. Measure actual input preparation/transfer,
peak VRAM and host memory anyway. Do not extrapolate tiny-model timings to a
large deployment or treat a retained image as permanently GPU-resident.

Measure deployed learning work as fitting, replay selection/materialization,
FD, eviction/Fisher, and compression, divided by unique incoming observations.
Also report end-to-end experiment time including shared initialization, held-out
evaluation, checkpoints, reporting, and validation. Meter forward/backward and
replay/probe presentations and closure counts; do not call those counts exact
FLOPs or assume an iteration has equal cost in both methods.

Freeze one principal average-learning-time envelope from 8B timing, before
looking at 8C predictive results. Its provisional rule is 1.25 times the larger
measured mean seconds per arrival of the adequately solved reference policies
at $(C,b)=(64,16)$, with and without FD. Use the same allowance for both methods,
the same logical capacity ceiling $C\le256$, and the same available-device
memory ceiling. This intentionally permits both reference policies and gives
replay room to buy additional rehearsal; it is a calibrated experimental budget,
not a universal deployment deadline. Publish measured risk/cost frontiers as
exploratory context at other allowances, without additional confirmation grids.

Resource feasibility is based on measured total learning cost, not prescribed
iterations. Record mean and high-quantile update latency and unused allowance.
Do not force useless iterations after convergence, change replay samples inside
L-BFGS, or call an adequately solved policy infeasible merely because it stops
early. Conversely, do not quietly undersolve to meet the allowance. A policy
whose fresh measured cost exceeds the envelope keeps its predictive results
but loses the resource-feasible claim. This is an average-throughput study, not
a guaranteed hard real-time scheduler.

#### Phase 8A: Astra Implementation, Validation, And Handoff

Implement the complete 8B--8E pipeline, not only its first pilot, before the
model switch. Prefer a separately versioned regression package, for example
`mnist_experiment.fisher_memory_regression`, reusing compatible artifact,
parameter, compression, and reporting helpers. Give it local instructions
covering its Gaussian likelihood and scalar-head counts; do not weaken old
categorical assertions to make new runs fit old schemas. The implemented
interfaces and operator commands are recorded in `PLAN15_HANDOFF.md`.

Provide configuration-driven stages `calibrate`, `allocate`, `confirm`,
`analyze`, and `all`, with `--resume`, persistent study-wide active-time limits,
per-invocation limits, cooperative STOP/signals, explicit stage dependencies,
and completion-skipping. The all-stage scheduler must progress without an
assistant connection. Its confirmation ledger and selection artifacts are
sealed before any confirmation outcome can influence scheduling.

Required acceptance includes:

- Gaussian NLL/degree-MSE equivalence, score sign, exact model-target Fisher,
  and Gaussian symmetric-KL factors; FD values and gradients against references.
- Scalar-head parameter counts, identical paired initial predictions, no
  categorical loss/Fisher calls, and gradients into both memory blocks.
- Truncated distribution normalization, symmetry, deterministic sampling,
  correct renderer sign, and no endpoint clipping or hidden angle features.
- Disjoint data roles, named/component seeds, identical paired streams/panels,
  exact novel-count checkpoints, and no synthetic or replay exposure inflation.
- Fixed replay subsets/probes within closures, mean-loss normalization for
  different $b$, capacity/empty-buffer boundaries, and correct age/exposure clocks.
- Exactly-once retirement with subsampling and $C=0$, checkpoint transactions,
  immutable completion, source/configuration mismatch refusal, and budget locks.
- CPU and cached-MNIST CUDA smoke interruption/recovery, including pretraining,
  mid-solve and eviction boundaries, plus stop-aware validation/reporting.
- Deterministic selection, fresh-cohort separation, paired uncertainty,
  exact-budget contrasts, incomplete-run reporting, and no training in notebooks.

Use the user's `~/.venv/bin/python`. Limit this stage to unit tests and tiny
smokes, not development selection. Deliver tested foreground/resume commands,
acceptance manifests, timing/storage smoke measurements, and a short update to
`PLAN15_HANDOFF.md`. Begin the chronological Phase 8 notebook section with its
question, density plot, concise glossary, and explicit pending status. Historical
cells and sealed scientific artifacts must remain intact.

**8A implementation checkpoint (2026-10-24).** The separate regression package
implements all subsequent stages. Acceptance covers tiny CPU and CUDA runs,
production-dtype FD checks, interruption recovery and exact prefix continuation.
The focused CPU suite passes 276 tests; CUDA recovery is also checked in both
float64 and float32. Detailed counts, source hashes, immutable smoke seals and
charged test/smoke time live in
`cache/mnist_experiment/plan15/regression_8a_acceptance_v1/units/acceptance/result.json`.
Production 8B--8E remains pending the subsequent execution instruction.
Acceptance charged approximately **7.8 minutes** against the shared 20-hour
allowance, with about **45.5 MB** of artifacts including superseded smokes.
All 41 historical notebook cells are unchanged; the new section adds eight
concise cells including its progress panel. These are software checks, not a
production throughput forecast or scientific result.

Implementation uses one versioned study root with sealed stage units, rather
than requiring four roots. It retains one shared Phase 8 time allocation and
the existing Plan 15 storage ceiling. Numerical acceptance now explicitly
requires both float64 and actual-training-dtype FD value/gradient agreement.
The shared solver may receive at most two automatic ceiling doublings, with
revalidation of selected full-budget policies and separate immutable decisions
for each attempt. Gaps are normalized after removing the Gaussian additive
constant; failed audits are not accepted as evidence of convergence. Remaining
defects require Sol's documented, versioned repair rather than weakening an
integrity check. Tiny smoke outcomes are not development evidence.

**Execution repair (2026-10-24).** Protocol v1 is superseded after a confirmed
archive-compression defect. Legacy Lanczos can duplicate nearly rank-one Ritz
directions, inflate the Fisher trace and hence the ridge coefficient, and cause
nonfinite solver trials. A fixed PSD target with trace 1 was represented with
trace 4; repeated compression grew its trace above 63,000 in ten updates.
The sealed v1 `compression-failure-evidence` unit retains the reproduction and
actual asset audit. All v1 development and partial confirmation are preserved
for diagnosis and excluded from final inference; no individual failed replica
is silently removed from a cohort.

Protocol v2 replaces only this regression compressor with stable exact eigh
on the small dense LTM block. For the supplied matrix
$M=(1-\eta)\widehat F+\eta F_{\mathrm{retired}}$, take its top eight PSD modes
$A=U_8\operatorname{diag}(\sqrt{\lambda_1},\ldots,\sqrt{\lambda_8})$ and store
$AA^T+\operatorname{diag}(\operatorname{diag}(M-AA^T))$. Only roundoff-sized
negative residuals may be clipped, with explicit diagnostics and a diagonal
conservation check. Its trace therefore follows the intended convex update,
without manufactured curvature. Likelihood, Fisher estimand, eviction clock,
anchors and ridge equations remain unchanged. Failed numerical fits stop
before committing an arrival. Fresh v2 component seeds and a new run identity
restart all stages; superseded compute still counts against 20 hours. Repairs
remain within Sol's existing authority and are visible in the notebook.

#### Phase 8B: Sol Numerical, Initialization, And Cost Calibration

Use development-only seeds/assets. Check near-upright initialization against
zero prediction and document novel-domain starting error, per-digit ambiguity,
and solver health. A weak learning curve triggers diagnosis and best-bet
calibration, not an efficacy-based early stop. Do not pretrain on the novel
distribution to manufacture a stronger starting model.

Start with reset-per-update strong-Wolfe L-BFGS, 100 iterations/150 closures.
On a small fixed set of early, middle, and late development snapshots, compare
the accepted solution with a higher-ceiling solve of exactly the same objective,
subsample, probes, and anchors. Report objective gaps, residuals, predictions,
and cost. Repair a materially inadequate shared solver before interpreting
method differences; a larger ceiling is a diagnostic, not a new retention knob.
Freeze the numerical FD step from approximation/gradient agreement, not from
held-out predictive wins.

Give regularization a bounded regression-specific calibration at $(64,16)$ on
two paired development replicas, initially stopping at 30 novel observations:

1. Compare replay/EWC with shared-ridge candidates $c\in\{0.1,1,10\}$. Select
   the best acquisition candidate satisfying the base-retention guard below;
   use the same selected $c$ for FD and the no-replay boundary. If none satisfy
   it, retain the least-violating numerically valid candidate and make the
   warning explicit. If none are numerically valid, repair before selection.
2. At representative nonempty, aged-buffer snapshots from that selected
   replay/EWC trajectory, measure the norm ratio of ordinary data-loss gradient
   to unit-strength FD-penalty gradient. Let
   $\lambda_0$ be its median over finite, nondegenerate snapshots; freeze it.
   Probe $\lambda_F\in\{0.03,0.3,3\}\lambda_0$ on the same two streams.
   Gradient scale seeds a search; it is not a predictive optimality claim.
   Zero/unstable denominator gradients require a numerical diagnosis, not an
   enormous coefficient hidden behind an epsilon.
3. Carry the two best admissible FD strengths, the replay control, and the
   no-replay boundary through the full 100-event trajectory on these replicas.
   Select FD strength on full-budget acquisition/retention. Allow one additional
   adjacent strength, divided or multiplied by ten, if the best tested value
   is at the original boundary and the time reserve permits. Never rerun a
   scientifically identical prefix. Seal the 30-event prefix as its own unit;
   a selected continuation is a new unit referencing that immutable checkpoint,
   not an append to a completed run. Its planned 100-event summary binds both
   units under the declared staged ledger.

Start with a base-retention guard of **one degree** additional budget-averaged
RMSE relative to each replica's frozen base-domain reference. Define that RMSE
as the square root of its budget-averaged MSE, not the mean of per-step RMSEs.
Development feasibility uses the mean of these paired per-replica differences,
not a requirement that every individual trajectory improve or satisfy the guard.
This is an explicit practical tolerance for this illustrative task, not a
clinical/application requirement or a fraction of samples saved. A changed
tolerance must be recorded as outcome-informed development and frozen before
confirmation; it cannot be relaxed to rescue confirmation results.

Publish a compact calibration summary: initialization adequacy, numerical
validity, $c$, FD strength/step, the measured FD/control cost ratio, bytes per
trajectory, and the remaining-time forecast. Seal the common solver, likelihood,
pretraining, strengths, retention guard, throughput allowance, and planned 8C
grid. Calibration records remain exploratory, even if their contrasts look
significant. Reserve most remaining time for complete fresh paired trajectories.

#### Phase 8C: Sol Allocation Map And Prospective Efficiency Design

Start with $C\in\{16,64,256\}$ and $b\in\{4,16,64\}$, masking $b>C$ rather
than silently converting it to another cell. This is eight valid cells per
method. Screen both methods on the same two new development replicas, plus the
no-replay boundary. The frozen reference is evaluated once per shared asset.
The $(64,16)$ reference remains visible from calibration, distinctly labeled
when it has different development replicas. Do not pool unequal replica sets
and describe a heatmap as a fully paired comparison.

Before the map starts, Sol may shrink its breadth using measured runtime and
the 20-hour forecast. Preserve at least two capacities and two replay amounts
when feasible. Show unmeasured cells as unmeasured, not zero or poor outcomes.
Deepen up to two promising or ambiguous cells on two additional paired
development replicas if affordable. Use those same replicas for the relevant
control comparisons. No full factorial over solver, strength, ridge, and grid
is planned. A scientifically justified adjustment is versioned and noted,
not blocked awaiting the user.

**Map and selection.** Render shared-scale maps of mean novel-domain
budget-averaged MSE for each method, a paired control-minus-FD difference map,
and a compact base-retention/cost feasibility view. Annotate replica counts and
link uncertainty and numerical diagnostics. These are held-out predictive
losses without regularizers. Sol must inspect exported figures and record a
short review for isolated extrema, noisy neighborhoods, missing cells, boundary
optima, and solver problems. Inspection is a diagnostic, not an additional
unrecorded objective or a substitute for uncertainty.

Inspect while development runs when an agent is available. The unattended
scheduler uses the recorded numerical rule and marks a missing visual review
as pending; it does not wait for user acknowledgement or an attached model.
A later review may flag a defect under the correction protocol, but cannot
quietly reselect a policy using confirmation outcomes.

Within each method, shortlist numerically valid, resource-feasible candidates
meeting the development retention guard. Let the acquisition tie tolerance be
1% of the common development frozen novel-domain MSE. Among candidates within
that absolute tolerance of the smallest mean acquisition MSE, prefer lower
measured learning cost, then smaller $C$, then smaller $b$. Publish this
practical tie rule, the selected cells, and whether an optimum was bracketed.
This selects a best-tested policy, not a global optimum. If none qualify,
select the least retention-violating resource-feasible candidate for an openly
flagged confirmation; do not label it practically successful. If no tested
candidate is resource-feasible, resolve the grid/timing problem in development
before freezing confirmation, or close out the infeasibility within budget.

**Prospective sample-equivalence choices.** Before confirmation, freeze one
smaller novel-observation budget $n_F$ and the replay budget $n_R=N=100$.
Using only development curves, choose the smallest candidate in
$\{25,50,75\}$ for which mean FD novel-domain MSE is no worse than replay at
100 and the candidate represents improvement over initialization. If none
qualify, retain $n_F=75$ as an explicit unsupported development probe; do not
invent an expected saving. Record uncertainty around this selection rather than
describing a two-replica crossing as established.

For the secondary descriptive plot, freeze
$q_{\mathrm{dev}}=\overline R^R_{\mathrm{replay,dev}}(100)$ as a target and
inspect whether a straight line over $n=1,\ldots,100$ adequately represents
each mean MSE curve. If it does, register that one fit per method, its fitting
range, and diagnostic before confirmation. Otherwise use the observed curves and
the direct budget comparison without a fitted samples-saved estimate; do not
force monotonicity or launch a curve-family search. The fixed target is not
recomputed from whichever confirmation endpoint looks attractive.
Default to no fitted inversion if no development fit-review record is available
at the transition. This preserves autonomous progression and the primary/direct
efficiency analyses without inventing a visual adequacy decision.

Seal all selected recipes, resource/tolerance choices, cohort size, seeds,
sample-budget contrast, target, and fit decision in a confirmation manifest.
No confirmation result participates in any of these choices.

#### Phase 8D: Sol Fresh Paired Confirmation

Default to **twelve fresh paired replicas**, each with the three adaptive
policies and its shared frozen reference: 36 online trajectories. All adaptive
arms run to $N=100$, retaining the earlier exact-budget checkpoint $n_F$; the
smaller-budget comparison needs no additional training run. The selected
control can use a different $C,b$ from FD, under the same frozen envelope.
Use new initialization/stream/probe seeds and the confirmation-only test pool.

Choose the final replica count before the first confirmation trajectory from
measured upper-cost forecasts and the remaining study budget, preferably
8--12. Sol may reduce it further if necessary to complete a valid comparison
under 20 hours, with a visible precision warning and revised frozen ledger.
Do not increase/decrease it after inspecting significance. Poor efficacy alone
does not stop the cohort. Alternate methods within replica and progress paired
replicas together so interruptions leave useful comparisons. A resource halt
preserves partial evidence and pending work; it is not a completed cohort.

Let $\theta_{r,m}(n)$ denote the accepted state immediately after the segment
ending at novel event $n$. Define predictive risk on the fixed held-out
population, implemented by the frozen weighted panel, as

$$
R^D_{r,m}(n)=\mathbb E_{(X,\alpha)\sim D}
[(a_* f_{\theta_{r,m}(n)}(X)-\alpha)^2],\qquad
A^D_{r,m}(N)=\frac1N\sum_{n=1}^{N}R^D_{r,m}(n),
\quad D\in\{Q_0,Q_R\}.
$$

Here a domain denotes its joint image/angle population, not just the angle
marginal. Initial prediction at $n=0$ is a reference and is not included in
$A(N)$. Held-out panels never enter fitting, replay, FD, Fisher estimation, or
hyperparameter selection in confirmation. Evaluation is model inference on
these separate panels, not training-batch loss. Gaussian predictive NLL equals
$R/(2a_*^2)+\log(a_*\sqrt{2\pi})$ and contains no regularizers.

The primary paired effect is
$\Delta_A=\mathbb E_r[A^R_{r,\mathrm{replay}}(100)-A^R_{r,\mathrm{FD}}(100)]$;
positive favors FD. It measures performance averaged over a uniformly chosen
novel-observation budget, independently of the trajectory. The identity does
not require monotonicity or an i.i.d. training stream. Practical success also
requires the frozen base-retention and measured resource constraints, and a
visible comparison with freezing. Do not equate beating a learning control
with useful acquisition if both are worse than initialization.

The key prespecified sample-budget contrast is

$$
D_r=R^R_{r,\mathrm{FD}}(n_F)-R^R_{r,\mathrm{replay}}(100).
$$

Use a **zero harm margin** as the conservative default: a two-sided paired
95% interval with upper bound below zero supports better mean prediction at
the smaller novel-data budget. This avoids inventing a noninferiority margin
after seeing results. A nonzero practical margin would need an explicit,
prospective development amendment with its degree-error interpretation. A
nonsignificant difference is not equivalence. Even a favorable contrast means
FD used fewer observations for the tested comparison; it does not establish
that replay first reached this capability at 100 or that exactly $100-n_F$
observations were necessary savings on every trajectory.

Report paired whole-replica bootstrap 95% intervals (2,000 independently seeded
draws), mean differences, and favorable-replica counts. Retain all timepoints
within each resampled replica; never bootstrap them as independent samples.
Primary acquisition, sample-budget comparison, and retention intervals are
separately labeled nominal inference, not a familywise claim. Assumptions for
uncertainty include independent replica generation and adequate moments of
squared prediction error; bounded true angles do not bound an unbounded
predictor's loss. State that inference is conditional on the chosen benchmark
panels/recipe. Development evidence is never added to confirmation intervals.

For the retention claim, show the paired distribution and upper interval bound
for $\sqrt{A^0_{r,m}}-\sqrt{A^0_{r,\mathrm{frozen}}}$ against the one-degree
guard. Show both budget-averaged and final base risk and per-digit diagnostics.
Also retain small, matched exposure-cohort prediction audits after retirement,
with actual retirement ages and counts. Those are training-cohort retention
diagnostics, not held-out generalization. Across differing capacities, never
pretend different retirement times/cohorts form an identical retention test;
the common held-out $Q_0$ outcome is the principal retention constraint.

#### Phase 8E: Sol Efficiency Interpretation And Notebook Closeout

The main conclusion has three distinct statements: budget-averaged new-data
performance, the direct smaller-versus-larger sample-budget comparison, and
retention/resource feasibility. Report unfavorable or imprecise results just
as explicitly as favorable ones. No success claim depends on establishing
STM-to-LTM information transfer.

Produce a **sample-equivalence figure** with held-out novel-domain RMSE versus
actual unique novel observations, paired-replica uncertainty, the frozen
reference, the fixed development target in degrees, and the two prespecified
comparison budgets. Plot $\sqrt{\overline R(n)}$, consistently, rather than
silently averaging RMSEs. Label accompanying total arrivals and learning cost
at those budgets in a small companion panel or caption. More novel observations
come with more background observations under this drip schedule; the result
is not an isolated causal effect of novel labels or total data saved.

If the registered linear MSE approximation remains visually adequate, overlay
it after transforming to RMSE for display. Its target crossings may provide
an explicitly model-dependent secondary sample-equivalence estimate. Fit each
condition separately, show fit versus observed means, and report only crossings
within observed budgets with negative slopes and a target not already attained
at initialization. No extrapolation, revised fit window, or new target chosen
from confirmation is permitted within this analysis. Poor fit, an already-met
target, or an unreached target is a limitation, not a reason to replace the
primary metric. Keep model-derived counts separate from the direct observed
budget contrast. Do not require an inverse-curve CI for the study to conclude;
the paired interval on $D_r$ supplies its planned inferential statement.

Keep the Phase 8 section concise and chronological after Phase 7E. Its main
story should occupy roughly five figures and two short tables:

1. Near-upright/novel angle densities, one sentence explaining the drip.
2. Calibration quality/cost table and the selected numerical recipe.
3. Allocation heatmaps with resource/retention feasibility and a brief visual
   review, updating in place rather than accumulating successive maps.
4. Fresh new-domain and base-domain RMSE trajectories with consistent legends,
   uncertainty, the frozen reference, and actual paired-replica counts.
5. Budget-averaged paired effects and a compact retirement-retention view.
6. Sample-equivalence figure and a short conclusion table stating what is and
   is not supported, including actual learning cost and peak memory.

Use the existing compact progress panel near the notebook top: phase/stage,
completed/planned pairs, active hours used out of 20, forecast remaining time,
bytes used/reserved, last heartbeat, revision, and development/confirmation
status. Update on paired completions and at most every 15 active minutes.
Show partial results as descriptive; publish final inference only against the
sealed cohort ledger. Record material corrections in the existing short
significant-changes table, linking detailed diagnostics instead of printing
configuration dumps, every replica, or solver logs. The notebook loads small
artifacts only and never trains, evaluates models, repairs data, or runs the
bootstrap; compute analysis in a resumable Python stage.

**Execution record (2026-10-24).** The valid root is
`cache/mnist_experiment/plan15/regression_phase8_v2`. It contains 20 calibration
trajectory units (including prefixes/continuations), 44 allocation trajectories,
and 36 fresh confirmation trajectories: twelve complete paired replicas. V1
evidence remains separate and superseded. Calibration selected $c=10$,
$\lambda_F=263.5643$, and $h=0.01$. Development selected FD at $(C,b)=(256,16)$
and replay/EWC at $(256,4)$, with the $C=b=0$ EWC-only control. These are policy
comparisons under common constraints, not isolated equal-allocation FD effects.

| Fresh-cohort contrast | Mean [nominal paired 95% interval] | Interpretation |
| --- | --- | --- |
| Replay minus FD new-domain $A(100)$, degrees squared | -21.91 [-28.44, -14.49] | Selected replay policy is better |
| FD $R(75)$ minus replay $R(100)$, degrees squared | 79.80 [59.75, 101.02] | No smaller-budget advantage; FD is worse |
| Frozen minus FD new-domain $A(100)$, degrees squared | 293.92 [267.64, 316.07] | FD still learns usefully versus freezing |
| FD base RMSE increase, degrees | 1.334 [1.162, 1.485] | Exceeds the one-degree guard |
| Replay base RMSE increase, degrees | 1.591 [1.408, 1.769] | Also exceeds the guard |

Only one of twelve acquisition comparisons favors FD. Its smaller-budget probe
was the prospectively recorded unsupported-development fallback, not an
established sample equivalence. The full-range linear inversion was rejected
before confirmation because it misses early and late plateaus. No fitted
samples-saved claim is made. Both learning policies improve on freezing, but
neither meets the full practical-success contract. Lower FD retention harm is
descriptive here; it does not rescue its worse new-data efficiency.

Selected FD uses 47.77 mean learning seconds per trajectory versus 27.61 for
replay (1.73 times), with 100 novel observations and 1048.5 mean total arrivals.
Both fit the frozen measured time/memory envelope; this is not a large-network
timing extrapolation. The final v2 root occupies about 0.235 GB, and shared
artifacts/dependencies about 4.45 GB of the 500 GB cap. Final acceptance passes
305 CPU checks and nine CUDA recovery checks; all 100 trajectory units are
numerically valid, with zero failed solves. Full content checks preserve prior
scientific seals and all 41 historical notebook cells. Final archive trace
error is below $3.1\times10^{-16}$ relative. Evidence is sealed in
`closeout-qa-v1` and `closeout-qa-v2`; the latter also checks the report-label
rendering repair. Supplementary report v2 preserves the earlier immutable v1
report. The concise results and interpretation are in
[fisher_memory_results.ipynb](fisher_memory_results.ipynb), with operational
details in [PLAN15_HANDOFF.md](PLAN15_HANDOFF.md). No Phase 8 job remains running.

#### Phase 8 Recovery, Autonomy, And Budget

Use new, source/configuration-versioned roots under
`cache/mnist_experiment/plan15/`, for example `regression_calibration_v1`,
`regression_allocation_v1`, `regression_confirmation_v1`, and
`regression_analysis_v1`, with a **single** Phase 8 compute ledger. Include
superseded repair roots, acceptance tests, and failed work in its expenditure.
Reusing an unchanged asset saves work; it does not erase its charged cost.

Accepted-update checkpoints atomically bind parameters, archive/anchors,
FIFO and sampled-subset provenance, unique retirement ledger, data cursor,
pending nominal-group segments, RNG state, metric offsets, and timing counters.
Pretraining checkpoints bound replay of unfinished work to at most sixteen
minibatches; online recovery repeats at most one uncommitted solve. Selection,
stage transitions, timing calibration, cohort admission, and bootstrap chunks
are independently resumable. Preserve already completed units byte-for-byte.
Use regular durable timing heartbeats so repeated hard kills cannot reset the
allocation; conservatively charge the bounded uncommitted interval on recovery.

Validation/reporting must remain incremental and stop-aware, with no full
historical archive replay on every restart. Target graceful stop within a
minute and verify the bound in smoke acceptance. Resume after a node restart
is an explicit command, not a promise of surviving machine shutdown. The
scheduler can run in a foreground shell or `tmux`; it must not require an
ongoing model/remote connection to advance its agreed work.

Forecast compact artifacts at roughly 2--10 GB initially and replace that with
smoke/pilot measurements, counting temporary copies and report staging. Store
immutable small trajectories, shared inputs, a rolling recovery checkpoint,
and selected audits, not every per-example matrix or historical checkpoint.
Admission uses the existing shared **500,000,000,000-byte** ledger plus actual
filesystem reserve. No deletion of old evidence, external-drive dependency,
or silent cap increase is authorized.

Sol may repair mathematics/software, adjust development settings, reduce grid
breadth, and choose a smaller fresh cohort to finish within the allocated
compute. Keep a paired, useful comparison ahead of exhaustive coverage. Log
changed equations, reasons, affected evidence, validation, and the revised
forecast; create new scientific identities and fresh confirmation when a
repair changes the comparison. Do not mix incompatible versions or modify a
completed cohort in pursuit of significance. Poor performance alone is not a
fast-fail trigger. Integrity/resource exhaustion can require a clean stop with
an explicit partial conclusion, not invented completeness or a blocking request
for routine scientific judgment.

| Sub-phase | Owner | Initial active-compute allocation | Deliverable |
| --- | --- | ---: | --- |
| 8A: implementation acceptance | Astra | 0.5 h | Tested pipeline, tiny smokes, notebook scaffold, handoff |
| 8B: calibration | Sol | 2.5 h | Regression/solver/FD validation, measured costs, frozen common policy |
| 8C: allocation and efficiency registration | Sol | 4 h | Paired maps, selected policies, frozen targets/budgets/cohort |
| 8D: fresh confirmation | Sol | 10 h | Preferably 8--12 complete paired replicas, with all controls |
| 8E: analysis and reporting | Sol | 0.5 h | Sealed effects, efficiency figure, concise conclusion |
| Shared correction/timing reserve | Primarily Sol | 2.5 h | Numerical repairs or forecast error within scope |
| **Total ceiling** | | **20 h** | One aggregate allocation, not per invocation or root |

These are provisional allocations, not measurements of an implemented pipeline.
Expect approximately **14--20 active compute hours** if calibration is healthy;
8B must replace that forecast using measured complete-unit costs, including
initialization and reporting. Astra programming/review is provisionally another
**4--8 hours of engineering elapsed time**, separate from process compute.
Unused stage allocations may be reassigned prospectively within the total;
unused time does not authorize an extra significance-seeking cohort. Reduce
development breadth before shortening the 100-novel-observation learning window
or dropping a control. A necessary scope change must be explicit and versioned.
Reserve final analysis/checkpoint space and time before admitting more jobs.
Paused/node-repurposing time is excluded, but all active experiment overhead
counts. This draft starts no job; implementation/execution require the user's
subsequent instruction.

### Phase 8.1: Retention-Matched Regression Follow-Up

**Authorized for implementation and execution on 2026-10-25.** The reviewed
design below is the execution contract, with a 20-hour aggregate ceiling. This is
an outcome-informed extension of Phase 8, not a replacement of its evidence or
a repair that retroactively invalidates its selected-policy comparison.
Its question is: **which method acquires useful predictions from scarce new
observations at comparable old-domain deterioration and bounded compute?**
EWC-only is a genuine candidate, not a control that must lose. No success claim
requires FD, replay, or a proposed STM-to-LTM mechanism to be the winner.

Phase 8's twelve-replica selected policies ordered acquisition and retention
differently. Mean deterioration was 1.3342351463 degrees for FD, 1.5906512523 for
replay/EWC, and 3.4114979497 for EWC-only. FD's new-domain budget-averaged MSE
was 21.91 square degrees worse than replay (nominal paired 95% interval
[14.49,28.44]), while its retention was descriptively better. All missed the
original one-degree guard. Phase 8 measured this trade-off but did not establish
retention matching. Its short, shared-ridge calibration also left method-specific
regularization interactions unresolved. These observations motivate a better
comparison, not a predetermined explanation or a favorable reinterpretation.

**Execution record:** 8.1A acceptance passed 319 CPU and ten CUDA checks on
2026-10-25, including hard-kill recovery and immutable completed replay.
Production began at approximately 15:46 UTC in tmux session `plan15-phase81`,
under `cache/mnist_experiment/plan15/regression_phase8_1_v1`. The learning engine
and original Phase 8 artifacts are unchanged; the new scientific source hash is
`0cdc9f1dba59112c8b520cdab0a9849dd3df5a46983108b6614fb1bc7898dad8`.
**Completed 2026-10-25.** All 272 trajectories finished: 144 initial-grid,
80 bounded-refinement, 12 additional finalist-development and 36 fresh
confirmation trajectories. No solver failures occurred. The 12-replica cohort
and three policies were frozen before confirmation; no outcome-driven extension
or retuning followed. Aggregate charged active compute was approximately
4.93 hours out of 20, with about 0.50 GB additional artifacts. All 318 completed
units and 2,884 sealed file checksums passed post-run checks; all earlier
notebook cells and the frozen Phase 8 notebook remain unchanged.

Primary replay-minus-FD acquisition gain was **18.8931 square degrees**,
nominal paired bootstrap 95% interval **[15.6257,22.3599]**, favorable in 12/12
replicas. This is a **1.40% reduction in budget-averaged novel MSE** for the
selected FD policy. FD-minus-replay deterioration was 0.04904 degrees
[-0.00253,0.09847], supporting retention equivalence within the registered
0.20-degree tolerance. All methods supported the common retention ceiling by
their one-sided 95% upper bounds and met measured resource limits; FD's upper
bound was 1.31994 degrees versus the 1.33424-degree ceiling.

Mean learning time was 41.39 seconds for FD versus 25.02 for replay/EWC,
or 1.65 times as much. This compares independently calibrated policies,
including different replay counts/ridge strengths, not the isolated FD penalty
or equal-compute efficiency. It does not estimate actual samples saved.
The notebook distinguishes the secondary FD/EWC-only comparison, for which
retention equivalence was not established. Its linked, content-addressed HTML
supplement contains same-allocation development contrasts and calibration/
retired-training-cohort diagnostics, without additional confirmation tests.
Final reporting/archive checks passed 19 tests; four final figures were
visually inspected. Detailed results, source identities, metadata interpretation
notes and artifact-only republishing instructions are in
[PLAN15_HANDOFF.md](PLAN15_HANDOFF.md).

#### Inherited Contract And Explicit Changes

Inherit the **repaired Phase 8 v2** model, Gaussian likelihood, angle mixtures,
renderer, pretraining, disjoint source roles, exact novel-event alignment,
$N=100$ stopping rule, paired streams/panels, trace-preserving Fisher compression,
retirement-only archive, and metric definitions. Keep all three adaptive arms
and the shared frozen-model reference. Keep $\pi=.025$, rank eight, the
nominal-four-observation archive clock, replay weight .5, two FD probes,
$h=.01$, and the validated 100-iteration/150-closure solver policy. Numerical
repairs require new identities and a common adequacy audit, not covert tuning.

Change only the development search, selection/retention contract, fresh replica
families, and reporting. Unlike Phase 8, each method receives its own ridge
search; FD strength and ridge are crossed at each investigated allocation;
every efficacy candidate runs through all 100 novel observations. No 30-event
screen chooses a shared ridge and no FD coefficient is transferred to a new
allocation without testing that combination.

Keep $C=256$ in replay arms and give **both** FD and replay/EWC the same
$b\in\{4,16\}$ options. EWC-only retains $C=b=0$. This is a deliberately
restricted follow-up near Phase 8's selected allocations, not a new exhaustive
$C\times b$ optimization. Report same-$(C,b)$ development contrasts separately
from final independently selected policy comparisons. A winner is best among
tested policies, not a globally optimal algorithm.

Preserve the original objective normalization: replay arms use
$\tfrac12\overline L_{\rm new}+\tfrac12\overline L_{\rm replay}$, whereas the
empty-buffer boundary uses $\overline L_{\rm new}$. EWC-only therefore differs
in new-loss weight relative to regularizers and immediate consolidation, as
well as absence of replay. Independent calibration accommodates these policies;
it does not turn their comparison into an isolated causal replay ablation.
Do not add normalization variants or online retention controllers here.

#### Retention And Acquisition Targets

For replica $r$, method $m$, settings $\eta$, and $D\in\{0,R\}$, retain
predictive holdout MSE $R^D_{m,r}(n;\eta)$ after exactly $n$ novel observations.
Neither predictive loss contains EWC, ridge, or FD penalties. Define

$$
A^D_{m,r}(N;\eta)=\frac1N\sum_{n=1}^{N}R^D_{m,r}(n;\eta),\qquad
H_{m,r}(\eta)=\sqrt{A^0_{m,r}(100;\eta)}-\sqrt{R^0_r(0)}.
$$

$A^R$ is new-domain budget-averaged error in square degrees; $H$ is old-domain
deterioration in degrees. Take the square roots **within each replica before
averaging $H$**. It is neither endpoint error nor mean per-step RMSE. Shared
initialization/panels make the frozen reference identical across paired arms.
Show endpoint and temporal old-domain errors as diagnostics: equal $H$ need not
mean equal forgetting at every checkpoint, nor on every individual example.

Propose the frozen ceiling and matching tolerance

$$
\delta=1.3342351463^\circ,\qquad \epsilon=0.20^\circ.
$$

The ceiling is Phase 8 FD's observed mean, recorded as a **benchmark-informed
design choice**, not an application-derived safety requirement. The tolerance
is a prospective practical resolution for this illustrative comparison, not
estimated from desired significance. Both are fixed for this authorized run;
neither may be relaxed after new confirmation. The desired development matching band is
$[\delta-\epsilon,\delta]$. A common ceiling alone is not retention matching.

At fixed $N=100$ and resource bounds, compare low $\mathbb E[A^R_m]$ at
comparable $\mathbb E[H_m]$. Retention is a treatment outcome, not an external
covariate to regress away. Do not condition confirmation on realized per-replica
forgetting, discard unmatched replicas, rescale observed losses, or manufacture
performance by interpolating parameters or predictions between trained policies.

Inherit the measured Phase 8 mean-learning-cost ceiling of
**0.0585204394 seconds per total arriving observation** and its peak-memory
limit. This is a common allowance, not equal time consumed. A hardware/timing
change can motivate a common rescaling using reference timing alone in 8.1A,
before inspecting new predictive results; seal its rationale. Report actual
learning, eviction, evaluation, initialization, tuning, and total costs
separately. Do not select a slow policy as practically feasible by omitting FD
or consolidation work. Streaming/offloading and large-network extrapolation
remain outside scope.

#### Phase 8.1A: Implementation And Recovery Acceptance

Extend the existing regression pipeline with a separately identified protocol
and CLI accepting `--resume`. Keep Phase 8 v2 runnable/recoverable from its
recorded source revision and configurations; do not mutate its completed units.
Use new names such as `regression_phase8_1_v1` under the shared Plan 15 cache,
explicit development/confirmation partitions, and independently named seeds.
Do not use the old v2 `--resume` command to start this follow-up.

Add focused tests for the $H$ definition, pairing, selection-band/ceiling
distinction, deterministic candidate expansion, no confirmation leakage,
resource eligibility, immutable collisions, and recovery across decisions.
Run tiny cached-MNIST CPU/CUDA recovery smokes, including hard interruption
before/after a committed update and a graceful stop. These bounded integration
runs are included in the execution authorization. Confirm old scientific hashes/artifacts
are unchanged and old report publishers cannot overwrite the historical
notebook or replace the new main section.

Seal the initial grid, metric definitions, target, matching tolerance, resource
limit, component seeds, expansion rules, and shared compute/storage ledger
before production. Publish progress and a concise glossary before the first
development trajectory. After execution is authorized, Sol owns implementation,
numerical corrections, scheduling and closeout without requiring a model swap
or intermediate user review; record material changes and preserve affected
versions. Use the user's `~/.venv/bin/python`.

#### Phase 8.1B: Full-Budget Trade-Off Map

Use **four new paired development replicas**, sharing initialization, exact
streams and panels across all candidates. Independently calibrate replay/EWC
and EWC-only over

$$
c\in\{1,3,10,30\}.
$$

At both replay allocations, cross the same ridge grid for FD with

$$
\lambda_F\in\{0.01,0.1,1\}\lambda_{\rm historical},\qquad
\lambda_{\rm historical}=263.5643254076337.
$$

Thus FD initially covers approximately 2.636, 26.36, and 263.56. This scale
comes from the completed regression study, not the old categorical coefficient.
Zero FD is the corresponding replay/EWC control and must reuse that unit rather
than retrain a duplicate. The initial grid is **24 FD + 8 replay + 4 EWC-only
settings, or 144 complete trajectories**. Recheck numerical/solver health at
weak and strong regularization; report finite under-solving rather than hiding
it. No candidate is removed for unfavorable predictive efficacy.

After each paired completion update one acquisition-retention plot: horizontal
$\overline H$, vertical $\overline A^R(100)$, method colors, allocation marker,
target band, uncertainty where available, and visible resource violations.
Show every measured setting; connect only observed nondominated settings within
comparable replica sets. Connections guide the eye, not a guarantee of attainable
interpolated performance or a monotone ridge response. Display smaller facet
panels by $b$ if the combined map becomes crowded. Keep exact settings and
individual replicas in linked diagnostics, not a wall of notebook numbers.

Allow **at most 20 additional distinct settings**, each on the same four
replicas, for targeted refinement. Allocate at least one quarter of available
refinement slots to each adaptive method before borrowing unused slots.
Prefer unresolved retention matching to polishing an apparent acquisition win:

1. If a method/allocation cannot reach the band, extend $c$ outward by a factor
   of three, at most twice per allocation. At most two promising FD-strength
   slices per allocation may be followed, chosen by proximity to the band and
   then acquisition, using the full four-replica means.
2. Within a measured bracket of the target-band midpoint, try geometric
   midpoints in $c$, at most twice per slice. Check the measured response;
   nonmonotonic behavior is reported and handled by observed local brackets,
   not a forced global bisection or linear extrapolation.
3. When a promising near-band FD setting sits at a strength boundary, allow one
   adjacent decade in $\lambda_F$ there, with ridge refinement only within the
   remaining shared setting cap. No blanket factorial expansion is authorized.

Seal each proposed addition and its reason before running it so resumption
cannot repeatedly expand the grid. Reforecast complete-unit time, counting
pretraining and all overhead; reserve confirmation and analysis before adding
points. If a target remains unreachable within the tested range/resources,
retain the evidence and continue with the closest valid candidate visibly
marked unmatched. Do not silently move $\delta$ or omit that method.

#### Phase 8.1C: Selection And Fresh Confirmation

For each method, shortlist at most two distinct resource-feasible settings:
the lowest-acquisition setting inside the matching band and the best remaining
ceiling-feasible acquisition alternative (including below-band settings). If
the band is empty, retain the setting closest to its midpoint as the matching
candidate. If no setting is ceiling-feasible, keep the least-violating candidate
plus the next closest distinct candidate. If no numerically valid setting meets
the resource limits, keep at most two closest-to-target settings as diagnostic
finalists and flag that method's resource failure. Numerical invalidity is never a
favorable endpoint; repair/version it rather than dropping failed replicas.
Break exact acquisition ties by lower measured online cost, then $(b,c,\lambda_F)$.

Run those at most six settings on **two additional paired development
replicas**, yielding six-replica estimates for every finalist. Reuse the earlier
four immutable trajectories. Reassess resource eligibility on those six replicas
and apply selection among eligible finalists. Select the lowest-acquisition
finalist in the matching band. If a below-band finalist has both lower $H$ and lower acquisition
error than that choice, retain the dominating finalist instead: never worsen
retention just to obtain equality. Label it a potential joint improvement,
not a matched-forgetting result. If neither finalist enters the band, select
the closest valid finalist and explicitly carry the matching limitation forward.
If both are resource-infeasible, carry the cheaper one as diagnostic-only;
there is no resource-feasible selected policy for that method.

Seal the final three recipes, all final matching/feasibility warnings, analysis,
fresh confirmation seeds and cohort size **before any confirmation outcomes**.
Use **12 fresh paired replicas** by default: 36 complete trajectories through
100 novel observations. A timing-only reduction to 8 or 10 replicas is allowed
before confirmation if necessary to fit the total ceiling; never choose a
cohort size from interim significance or a hoped-for equivalence interval.
If eight do not fit, close with development/partial status rather than imply
the confirmatory experiment was completed. Preserve every planned valid pair;
unannounced node interruptions change timing, not inclusion criteria.

The primary acquisition contrast is

$$
G_r=A^R_{\mathrm{replay},r}(100)-A^R_{\mathrm{FD},r}(100).
$$

Positive is favorable to FD, negative to replay. Include FD versus EWC-only and
replay versus EWC-only as named secondary comparisons with nominal paired 95%
intervals; do not imply familywise significance for selecting the best-looking
comparison. Use 2,000 paired whole-replica bootstrap draws, preserving joint
acquisition/retention measurements. Never bootstrap checkpoints as independent
observations. Frozen-model gains and actual cost remain visible for every arm.

For each comparison, report the paired interval for $\mathbb E[H_m-H_{m'}]$.
Call retention equivalence supported only if its entire nominal 95% interval
lies within $[-\epsilon,\epsilon]$; a nonsignificant difference is not enough.
Separately report each mean $H$ and its upper one-sided 95% bound against
$\delta$. A supported ceiling requires that bound to lie below the ceiling;
point feasibility alone is labeled as such. Thus a valid acquisition advantage
can coexist with an unresolved retention match or unmet practical ceiling.
If a method improves both outcomes, report the paired evidence for both rather
than withholding the joint improvement because equality was unnecessary.
No new replicas or confirmation-based retuning follow an inconclusive match.

Development uses novel training observations and separate old/novel validation
panels. Confirmation panels never select settings, tune coefficients, supply
gradients, replay, or Fisher inputs. Use fresh stochastic families and preserve
the explicit development/confirmation source partition. MNIST has already been
inspected, including the results used to choose $\delta$: this is confirmation
conditional on a reused benchmark, not pristine external validation. Old Phase 8
replicas remain context only and never enter new confidence intervals.
Report development novel exposures, unique validation images/labels, repeated
evaluations and tuning compute separately. The 100-observation claim concerns
online learning **after calibration**, not total label cost or zero-shot tuning
on an unseen novel distribution.

#### Phase 8.1D: Human-Readable Conclusion

Keep progress near the top and results chronological after the short Phase 8
historical summary. Update on paired completion and at most every 15 active
minutes; partial results are descriptive. Limit the main section to roughly
four figures and two compact tables:

1. Development acquisition-retention map with the ceiling and matching band,
   followed by a short visual assessment of coverage and noisy/nonmonotone regions.
2. Fresh old- and novel-domain predictive trajectories with frozen references,
   paired-replica uncertainty, unique novel counts and accompanying total arrivals.
3. Paired acquisition effects alongside retention differences and matching
   bounds; show resource eligibility and complete-pair counts.
4. Running $A^R(n)$ versus novel observations, with measured learning cost in a
   companion panel. This is the main sample-efficiency story at the stated
   retention/resource conditions, not an automatic number of samples saved.

Explain $H$ and $A$ before showing them. Use all three methods, not just the
favorite contrast. If unmatched, say so prominently. Preserve the scalar
retirement-cohort diagnostics inherited from Phase 8 in a linked supplement;
do not confuse training-cohort fit with the common held-out retention target.
Do not inherit Phase 8's unsupported 75-versus-100 probe or resurrect the rejected
linear inversion. Actual samples saved require a separately registered future
analysis; budget-averaged efficiency needs no fitted curve. The subsequently
authorized Phase 8.1E below supplies a separate exploratory mean-curve analysis,
without changing this phase's original confirmation or claiming guaranteed savings.

#### Phase 8.1E: Exploratory Local Sample Equivalence

**Authorized 2026-10-25 after inspection of the completed confirmation curves.**
Estimate the number of novel observations needed by the selected FD policy to
match replay/EWC's observed mean predictive error at 100 novel observations,
then measure the compute/retention trade-off at those respective budgets.
This is post hoc and conditional on the reused benchmark and selected policies,
not a new confirmation, policy search, or estimate of total labels including
pretraining/calibration. It does not replace the original budget-averaged result.

Use only the twelve complete paired confirmation replicas from
`regression_phase8_1_v1`. Validate the complete three-arm source cohort, its
frozen manifest, summaries, pairing, and all dependency seals before analysis.
Do not pool development or older studies, alter policies, evaluate new model
predictions, or train. Preserve the original analysis and computational units.

Define the observed mean-risk target and the piecewise-linear interpolation
of FD's mean predictive MSE by

$$
q=\overline R^R_{\mathrm{replay}}(100),\qquad
f_{\mathrm{FD}}(k)=\overline R^R_{\mathrm{FD}}(k),\quad k=0,\ldots,100.
$$

Interpolate only between adjacent observed checkpoints, with no global line,
alternative fit window, smoothing, monotonicity constraint or extrapolation.
Retain every downward target crossing. The reported matching budget is the
**last entry into the at-or-below-target region that persists through the
observed endpoint**. This conservative finite-window rule avoids reporting a
temporary dip as a saving; it is not an assertion about performance after 100.
When initialization is above target and FD's endpoint reaches target, let

$$
j=\max\{k\in\{0,\ldots,99\}:f_{\mathrm{FD}}(k)>q\},\qquad
\widehat n_{\mathrm{FD}}=j+
\frac{f_{\mathrm{FD}}(j)-q}{f_{\mathrm{FD}}(j)-f_{\mathrm{FD}}(j+1)}.
$$

Report $S=100-\widehat n_{\mathrm{FD}}$ and $100S/100$ percent, together with
the earlier crossing(s) as a sensitivity diagnostic. Fractional budgets are
interpolated cohort-mean equivalence, not fractional training images or mean
individual-trajectory first-passage times. If the target is already met at
initialization, FD's endpoint does not reach it, or the crossing lies outside
$[1,100]$, record the reason and withhold an in-range savings claim. Do not
assign zero savings, censor at 100, or choose a different target to repair it.

Use 2,000 whole-paired-replica bootstrap draws with a new named seed. Recompute
both means, the replay target, all crossings, the terminal crossing and its
guards within every draw. Preserve all draws, including unresolved outcomes.
If every draw is valid, report nominal percentile 95% intervals; otherwise
report the failure counts and mark the ordinary inversion interval unavailable,
never condition it on successful draws. Record multiple-crossing frequencies.
These intervals assume independent paired replicas, adequate squared-error
moments and a stable local crossing; they omit post hoc design uncertainty and
unobserved between-checkpoint behavior. Flat or reversing curves can make
inverse-risk uncertainty fragile even when the original mean contrast is clear.

At each point/bootstrap matching budget, interpolate mean cumulative learning
seconds and total arrivals at the same adjacent checkpoints. Compare FD at
$\widehat n_{\mathrm{FD}}$ with replay at 100, including an equal-capability
compute ratio, rather than automatically multiplying the original 1.65 ratio
by the estimated sample fraction. State that the novel-image budget is coupled
to the prescribed background stream, not a pure isolated-label intervention.
Also report $\sqrt{\overline R^0_{\mathrm{FD}}(\widehat n_{\mathrm{FD}})}$ and
$\sqrt{\overline R^0_{\mathrm{replay}}(100)}$, with their paired bootstrap
difference. These are descriptive old-domain endpoint errors, not $H$ or an
extension of its original matching/ceiling certificate to unequal budgets.

Implement a separate CPU-only entry point outside the frozen scientific-source
fingerprint, with a new analysis root such as `retention_efficiency_v1`.
Seal the design, named seed, input dependency seals, source snapshot, bootstrap
blocks and final analysis. Support `--resume`, stop files/signals, bounded
recomputation and the existing cumulative Phase 8.1 20-hour/Plan 15 500-GB
ledgers. A completed bootstrap block is immutable. Verify dependencies again
before publication. Include fast tests for interpolation/reversals, invalid
crossings, complete-cohort pairing, bootstrap-target recomputation, stop/resume,
immutability and notebook preservation. No GPU training is authorized or needed.

Append at most three concise cells to the Phase 8.1 notebook section: a clear
definition, one two-panel figure (all measured budgets and a last-quarter-budget
zoom), and an explicitly exploratory conclusion comparing samples, actual time
and old-domain error. Plot the observed means, replay target and matching point;
do not present an unobserved straight-line fit. Preserve all existing cells and
keep future parent-report refreshes from deleting or reordering the follow-up.
Visually inspect the final figure. Expected artifact-only compute is **1--6
minutes**, with a **10-active-minute cap** for this follow-up inside the existing
20-hour ceiling; initial implementation is estimated at **0.5--1.5 hours**.

**Execution record:** completed October 25 local time (October 26 UTC), with
39 focused unit/CPU CLI checks passing. The separately sealed artifact-only
analysis is under `cache/mnist_experiment/plan15/retention_efficiency_v1`.
All twelve source replicas and 318 original computational-unit seals were
validated; no training or source-result changes occurred. The new analysis
contains 20 immutable bootstrap blocks and supports `--resume`.

At replay's observed endpoint target of 1159.3112 square degrees, the terminal
FD mean-curve crossing is **91.4087 novel images**. Estimated savings are
**8.5913 images (8.59%)**, nominal paired-bootstrap 95% interval
**[1.8083,21.1794]** images. All 2,000 draws are in-range and evaluable; **959
draws (47.95%) have multiple downward entries**, a prominent sensitivity warning,
not a reason to discard those draws. The point mean curve has one downward
entry. The result is exploratory mean-curve equivalence under the recorded
terminal-crossing rule, not validated future observations saved.

At these respective budgets, measured mean learning time is **38.4630 seconds
for FD versus 25.0247 for replay**, a ratio of **1.5370 [1.3287,1.6402]**.
Mean total arrivals are 983.9636 versus 1050.7500. Old-domain endpoint RMSE is
7.3221 versus 7.6818 degrees; FD-minus-replay difference is
-0.3596 [-0.8591,-0.0075] degrees (nominal paired-bootstrap interval).
These endpoint diagnostics do not reuse H's equivalence margin or constitute
a new matched-H certificate. The notebook appends three concise cells with
one full-range/local-zoom figure and a two-policy comparison table. The
analysis uses roughly one charged active minute including its conservative
60-second acceptance allowance, inside the existing 20-hour aggregate ceiling.
Final closeout verified all 50 pre-existing notebook cells unchanged, the frozen
Phase 8 archive checksum, both source identities and all original unit seals.
The new analysis has 23 completed units, and its final figure was visually
inspected. Charged follow-up time including report refresh was about 68 seconds.

#### Preservation, Recovery, And Budget

The original Phase 8 protocol and execution record above stay intact. Its
completed section, including existing attachments and corrections, is copied
unchanged to
[fisher_memory_phase8_initial_results.ipynb](fisher_memory_phase8_initial_results.ipynb).
The archive records the original notebook checksum, preserved-cell checksum,
source revision, run identity and scientific source hash. The main notebook
keeps its unfavorable result and an archive link, then the new pending section.
Do not refresh the snapshot in place or hide negative evidence under a new title.
Keep all original code/configuration provenance and computational artifacts;
an additional historical analysis needs a new report, not a changed old result.

Use one authorized **20-hour aggregate active-compute ceiling for Phase 8.1**,
not per resume or method. This is a new allocation, not reuse of Phase 8's
unused time. Shared Plan 15 storage remains strictly below
500,000,000,000 bytes, including originals, temporary files and reserves.
Expect roughly 1--3 GB additional compact artifacts, to be checked in acceptance.
No deletion of evidence or external-drive requirement is authorized.

| Sub-phase | Expected active wall-clock compute | Main deliverable |
| --- | ---: | --- |
| 8.1A: acceptance and timing | 0.3--0.8 h | Tested recovery, selection contract, initial notebook and ledger |
| 8.1B: grid and bounded refinement | 3--6 h | 144 initial and at most 80 additional trajectories; trade-off map |
| 8.1C: finalist refinement and confirmation | 1.5--3 h | At most 12 new development and 36 confirmation trajectories |
| 8.1D: analysis and closeout | 0.2--0.6 h | Paired effects, matching assessment, concise conclusion |
| **Expected total** | **5--11 h** | Before pauses; initial implementation is separately estimated at 2--4 h |
| **Aggregate ceiling** | **20 h** | Includes failed work, tests, repairs, restarts, reporting and I/O |

Forecasts use Phase 8's roughly 22--48 seconds per selected trajectory, with
allowance for weaker-ridge solves, more candidates and shared initialization.
Those selected-policy times are not guarantees for the grid; replace the
forecast after the first complete paired candidates. The tested-grid maximum
is 272 trajectories, not a requirement to spend every refinement slot.
Preserve full 100-event windows and all controls before spending on extra points.

Inherit atomic accepted-update checkpoints, immutable completed units, bounded
restart loss, decision checkpoints, durable compute accounting, free-space
admission and single-writer locks. The scheduler must progress without an active
assistant session and accept `--resume` after a reboot. Stop-aware validation
and incremental notebook updates must not rebuild the entire historical archive
on each restart. Publish compact handoff notes at pauses and record significant
scientific corrections, with new roots when meaning changes. Poor efficacy is
not a fast-fail trigger; integrity/resource limits can require an honest partial
closeout. The user subsequently authorized this phase's implementation and execution;
the notebook and handoff report its current state.

## Statistical Reporting

For Phases 5B/5C, 7, 8, and 8.1, their phase-specific acquisition scores and primary
contrasts above supersede the earlier phases' outcome defaults below. Phase 8
uses Gaussian regression, MSE/RMSE, and its prospectively selected sample-budget
comparison; it does not inherit categorical accuracy or Phase 7E's post hoc fit.
Phase 8.1 retains regression $A^R(100)$ and explicitly evaluates retention
matching; it does not inherit Phase 8's smaller-budget contrast.

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

Keep the title and study overview first. Order experimental result sections by
phase from oldest to newest, keeping plots and explanations within their phase.
Append new phases at the bottom rather than inserting them ahead of earlier
results. A compact progress overview can remain at the front.

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

This envelope describes the original Phases 0--4, not the later extensions.
Phases 7 and 8 have their own forecasts above; Phase 8 proposes a separate
20-hour aggregate ceiling. Phase 6 remains deferred.

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

### Production Execution Notes

Sol began phases 1--4 on 2026-10-17 with an initial 20-hour active-time allocation.
Early Phase 1 diagnostics exposed two numerical handling omissions, repaired
under the autonomous-correction policy. Most under-solved default updates now
trigger the planned common 100-iteration/150-closure ceiling repair; a repaired
policy search cannot reselect the original short ceiling. For line-search
increases within 16 parameter-dtype machine epsilons times
$\max(1,|L|)$, the learner retains its previous accepted iterate and records
rounding stagnation rather than a solver failure. Larger increases remain
failures. Rejected increases and their tolerances are recorded explicitly.

The first two roots are separate Phase 1 diagnostic evidence. Version 3 finished
Phase 1 but exposed finite-stencil truncation in one mixture directional check:
the local radius .001 produced 4.3% gradient-magnitude error despite a passing
independent derivative check. A radius of .00025 reduced the error to 1.08%,
with gradient cosine above .99997, on the same saved state and direction.
The mathematical audit now tries the bounded radii
$\{.001,.0005,.00025,.0001\}$ until its original tolerances pass, retaining
every attempted result. Failure at all radii still makes the method unavailable.
The actual training direction and perturbation cap are unchanged.

Production resumes under `production_v4`, with no confirmation outcomes used
in these corrections. Version 3's early development remains separate.
Details and the current restart command are in the handoff; significant changes
also appear in the notebook and survive resumption.

Phases 1--4 completed on 2026-10-18 (America/Los_Angeles), with 140 development
and 144 fresh confirmation trajectories in version 4. The fixed cohort used
12 replicas per environment and four conditions. Cumulative active time was
about 14.5 hours, below the 20-hour allowance, and artifacts plus budgeted
dependencies occupy about 1.9 GB. Full content validation passed all 630 units;
there are no incomplete units or recorded numerical failures. No optional
extension was launched.

Selected finite-difference KL improved current and historical NLL AUC in all
12 confirmation replicas of each rotation schedule, with nominal paired 95%
intervals above zero. The digit-9 mixture showed no clear predictive gain.
These results do not establish consolidation: rotation STM-removal effects,
the frozen-initial reference, and persistent under-solving leave regularization
or capacity suppression as alternatives. The calibrated larger-ceiling replay
control consumed less actual time than the selected method and is not a
realized equal-time comparison. See the notebook's concise conclusion and the
handoff for effects, uncertainty, actual costs, and remaining limitations.

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
