# Implementation Plan 13: Continual SubGD Adaptation Geometry

> **Hypothesis:** low-data continual learning may become more statistically
> efficient when optimization is restricted or preferentially directed toward
> a learned covariance of historically useful adaptation directions. A causal
> online update of that covariance may retain the benefit when the useful
> tangent space changes, while an innovation-controlled full-rank channel may
> permit recovery when the current basis is stale.

**Status:** Production execution began on 2026-10-04 after the CPU smoke,
interruption/resumption drill, and CUDA smoke passed. Phases 0 through 5 ran
under the immutable, resumable, progressively reported contract below. A
post-execution optimizer diagnostic found that the original Phase 5 learner
did not solve the EWC subproblem adequately before SubGD treatment assignment.
The original Phase 5 artifacts remain immutable negative evidence about that
optimizer, but they are not evidence about SubGD efficacy. Phase 5R completed
on 2026-10-05: all 98 units were sealed, the burn-in viability gate passed,
and all 80 paired treatment trajectories entered the final analysis. The live
artifact-only report is `continual_subgd_results.ipynb`.

Phase 6 completed on 2026-10-05 with all 64 fresh paired replicas and 577
immutable units. Phase 6A completed on 2026-10-06 with all 64 fresh paired
replicas and 449 immutable units. It increased the low-prevalence batch size
to target approximately ten observed digit-9 examples per trajectory while
preserving the exploratory contract specified below. Phase 6B completed on
2026-10-06 as a final 64-replica closeout study targeting approximately 100
observed digit-9 examples per trajectory. All 449 immutable units and the
artifact-only report are complete.

**Scope:** Plan 13 studies adaptation geometry. It does not redefine the
likelihood Fisher, replace the EWC archive, or resume the search for an
optimal $\pi$. Rotated MNIST is the first controlled environment, not the
definition of the method. The original execution ended with a small transport
cohort on the older digit-9-mixture trajectory; Phase 5R adds a bounded repair
of that cohort without reopening rotation development. Phase 6 starts a fresh
low-prevalence mixture study on $p\in[0,.1]$ and does not alter any prior
artifact or conclusion. Proposed Phase 6A retains that range while increasing
the fixed per-step batch size. Phase 6B repeats that exploratory scaling design
at a 100-positive target without changing its conditions or learning rules.
The shared implementation therefore uses an environment adapter rather than
copying or reinterpreting rotation code.

## Motivation

Plans 11 and 12 did not produce a general controller for choosing $\pi$, and
statistically sized ridge regularization produced a real but modest benefit.
Those studies nevertheless sharpened the problem. The 512-parameter raw
network has a 25-dimensional exact likelihood gauge, and the remaining
487-dimensional identifiable chart contains many directions that are weakly
constrained by four-observation updates. Isotropic ridge suppresses wandering,
but it does not identify which parameter movements are repeatedly useful.

SubGD suggests a different source of structure: estimate a low-dimensional
second moment of adaptation displacements across related tasks, then use it as
an optimization geometry for a new task. The original construction estimates
that geometry offline and holds it fixed during deployment. Plan 13 asks
whether the idea survives a causal continual setting in which:

1. only an initial burn-in period is available for calibration;
2. the useful adaptation basis may rotate as the environment changes; and
3. a stale basis must not suppress the observations needed to repair itself.

The experiment separates four claims:

1. A learned low-dimensional geometry is better than an equally sized random
   geometry and practical head-only tuning.
2. A fixed learned geometry improves sample efficiency relative to ordinary
   full-space optimization under the same retention objective.
3. Updating the geometry online improves on a fixed geometry.
4. Innovation-controlled relaxation toward full-rank optimization improves
   recovery when the online geometry is stale.

No claim later in this sequence is inferred merely because an earlier one
looks plausible.

## Scientific Contract

### Fisher And Adaptation Covariance Are Distinct

The likelihood continues to induce the unique Fisher information matrix

$$
\mathcal I(\theta)
=\mathbb E_\theta[s(X;\theta)s(X;\theta)^T].
$$

The EWC process maintains an estimate $\widehat F_t$ of that object along the
realized learner path. Plan 13 additionally maintains an uncentered adaptation
second moment

$$
C_t=\mathbb E[z_tz_t^T],
$$

where $z_t$ is defined below. $C_t$ is not a Fisher matrix, an empirical
Fisher, a Hessian, or a natural-gradient metric. It describes directions in
which an unrestricted local learner would have adapted under the declared
retention objective.

The symbols and artifacts for $\widehat F_t$ and $C_t$ must remain separate.
No code may reuse the Fisher representation type while silently changing its
estimand to $C_t$.

### Identifiable Parameter Chart

Use Plan 12's functionally equivalent sum-to-zero classifier chart. Every
Plan 13 geometry acts on the resulting $p=487$ trainable coordinates. The
25-dimensional common-logit gauge is not a candidate adaptation subspace and
must not re-enter through a raw-model control.

The full network remains trainable. "Low dimensional" means a restriction on
the update geometry, not a change to the canonical CNN architecture.

### Common Retention Objective

Within an environment and replica, every post-burn-in learning condition uses
the same data loss, shared archive initialization, Fisher-update rule, ridge
rule, anchor-update rule, batch, number of inner optimization steps, and
scalar learning-rate schedule. Only the adaptation geometry differs at
treatment assignment. Thereafter, each condition's parameters, anchor, and
Fisher archive may diverge as path-mediated consequences of that treatment;
they are never forced to share a Fisher estimated at another condition's
parameter value.

For the principal rotation study, preserve the Plan 12 ordering and objective

$$
J_t(\theta)
=
\frac{1}{m}\sum_{i=1}^m L(X_{t,i};\theta)
+\frac{\omega}{2}
(\theta-a_t)^T
(\widehat F_t+\kappa_t I)
(\theta-a_t),
$$

with

$$
\omega=\frac{1-.025}{.025},
\qquad
\kappa_t=.1s_t,
\qquad
s_t=\frac{\operatorname{tr}(\widehat F_t)}{487}.
$$

Thus the principal retention mechanism is the empirically best-balanced
isotropic Plan 12 ridge, not the overconstraining spectral selector. Direct
EMA Fisher updating, rank-eight-plus-diagonal archive compression, and the
four-observation stream remain unchanged. Phase 0 must reproduce the relevant
Plan 12 control before Plan 13 treatment data are generated.

"Full-space fine-tuning" below means ordinary full-space optimization of this
retained objective. It does not mean current-only fitting. No-update and
current-only trajectories may be reported as boundary context, but they are
not substitutes for the full-space retained control.

The original digit-9-mixture transport phases use the accepted Plan 3
environment and retention contract: 100 evenly spaced values of $p$ from zero
to one, eight new observations per step, fixed $\pi=.05$, a direct-EMA
rank-eight-plus-diagonal Fisher archive, and no LFU. Phase 5 used Plan 13's
functionally equivalent gauge-fixed chart and fixed-step geometry optimizer.
Phase 5R preserves the chart and replaces only the failed mixture optimizer
as specified in its amendment. Phase 6 retains the same likelihood,
importance weight, Fisher archive, and no-ridge objective but replaces the
full-range schedule through its explicit low-prevalence amendment. None of
these versions adds Plan 12 ridge:

$$
\omega_{9}=\frac{1-.05}{.05},
\qquad
\kappa_{9,t}=0.
$$

All Plan 13 conditions within a mixture phase share this objective exactly.
Rotation evidence selects the geometry algorithm but cannot tune that phase's
frozen mixture retention policy, $p$ schedule, sample count, or evaluation
panels.

## Geometry Observation $z_t$

The geometry-update statistic must be defined in parameter-displacement units.
The principal statistic is the **unrestricted shadow displacement**. Starting
from the condition's pre-update state $\theta_t$, clone the model and run the
ordinary full-space reference optimizer for the fixed inner-step budget on
the same objective $J_t$ and current batch:

$$
\widetilde\theta_{t+1}^{\mathrm{full}}
=\operatorname{Opt}^{H}_{I}(J_t,\theta_t),
\qquad
z_t
=\widetilde\theta_{t+1}^{\mathrm{full}}-\theta_t.
$$

For the rotation study and the original Phase 5, the reference optimizer is
Plan 13's fixed-budget first-order method with $P_t=I$, not the historical
L-BFGS solver run to approximate convergence. Phase 5R supersedes this choice
only for repaired digit-9-mixture trajectories: its shadow displacement is the
endpoint of the declared full-space L-BFGS solve on the same local objective.
Rotation artifacts and their definition of $z_t$ are unchanged.

The clone is a measurement branch. It cannot update the deployed parameters,
Fisher archive, EWC anchor, optimizer state, evaluation counters, or random
stream. For the full-space control, its realized update already supplies
$z_t$, so no duplicate branch is required.

This choice is deliberately closer to classical SubGD endpoint displacements
than a raw gradient outer product. It also prevents a self-confirming loop in
which the current basis suppresses every direction from which its successor
could learn.

A cheaper, separately named diagnostic may use

$$
z_t^{\mathrm{grad}}
=-\eta_{\mathrm{ref}}\nabla J_t(\theta_t).
$$

It is a gradient-covariance proxy, not the same estimand. Phase 1 compares its
subspace, innovation signal, and cost with the shadow-displacement statistic.
It may replace the principal statistic in a later production protocol only
through a versioned amendment; it cannot do so silently because it is faster.

For either statistic:

- calculate it before any Plan 13 preconditioning;
- use the complete gauge-fixed parameter vector;
- retain the uncentered outer product $z_tz_t^T$;
- record $\|z_t\|$, objective decrease, optimizer diagnostics, and source;
- treat a numerically negligible $z_t$ as an explicit zero-information event;
  and
- never construct it from the accepted preconditioned displacement.

## Burn-In And Calibration

Every paired condition in one replica shares a common full-space burn-in:

$$
\theta_0\longrightarrow\theta_1\longrightarrow\cdots
\longrightarrow\theta_K.
$$

At each burn-in transition, use the ordinary full-space optimizer and retain
the resulting $z_t$. Define

$$
C_K
=\frac{1}{K}\sum_{t=0}^{K-1}z_tz_t^T
=B_K\Lambda_KB_K^T+E_K,
$$

where $B_K\in\mathbb R^{487\times r}$ contains the leading orthonormal
eigenvectors and $\Lambda_K$ their eigenvalues. All learned-geometry
conditions begin at the identical $\theta_K$, EWC state, $B_K$, and
$\Lambda_K$.

Burn-in observations are excluded from the principal post-calibration
estimand, but their sample and compute cost must be reported. An end-to-end
metric including burn-in is mandatory so calibration cannot be advertised as
free.

Phase 1 examines the nested candidates

$$
K\in\{8,16,32\},
\qquad
r\in\{4,8,16\},
\qquad r\le K.
$$

It selects one $(K,r)$ pair from held-out displacement reconstruction,
subspace stability, and resource cost without using downstream predictive
outcomes. Prefer the smallest rank within one standard error of the best
held-out reconstruction score. If no candidate explains more displacement
energy than matched random subspaces, classify low-rank calibration as
unsupported before closed-loop claims are made.

Because linear and sigmoid schedules cover different environmental distances
during the same $K$ transitions, calibrate and report them as separate
experimental conditions. They share candidate rules, not realized bases.

## Online Low-Rank Update

For $t\ge K$, maintain

$$
C_t\approx B_t\Lambda_tB_t^T,
$$

and update it only after the step-$t$ geometry has been chosen. For fixed
update rate $\beta$,

$$
C_{t+1}=(1-\beta)C_t+\beta z_tz_t^T.
$$

Given the current rank-$r$ representation, calculate

$$
a_t=B_t^Tz_t,
\qquad
e_t=z_t-B_ta_t,
\qquad
q_t=\frac{e_t}{\|e_t\|}
$$

when $\|e_t\|$ is numerically nonzero. Eigendecompose the $(r+1)$-dimensional
matrix

$$
M_t
=
(1-\beta)
\begin{bmatrix}
\Lambda_t&0\\
0&0
\end{bmatrix}
+\beta
\begin{bmatrix}
a_t\\
\|e_t\|
\end{bmatrix}
\begin{bmatrix}
a_t\\
\|e_t\|
\end{bmatrix}^{T},
$$

retain its leading $r$ eigenpairs $U_{t,r}D_{t,r}U_{t,r}^T$, and set

$$
B_{t+1}=[B_t\;q_t]U_{t,r},
\qquad
\Lambda_{t+1}=D_{t,r}.
$$

The zero-residual case is handled by the corresponding $r$-dimensional
update, not by inventing an arbitrary $q_t$. Reorthogonalize when a declared
orthogonality tolerance is exceeded and record that event.

The 487-dimensional experiment can afford a dense $C_t$ oracle in Phase 1.
Use it to measure cumulative error from repeated rank truncation, but never
present the dense oracle as the deployable method.

Parameterize a fixed $\beta$ by an interpretable half-life

$$
\beta=1-2^{-1/h_C}.
$$

Phase 2 screens a small frozen set of half-lives rather than tuning an
unbounded real-valued rate.

## Innovation Controller

Before applying the step-$t$ geometry, measure the fraction of unrestricted
shadow movement not represented by the lagged basis:

$$
v_t
=
\frac{\|(I-B_tB_t^T)z_t\|^2}
{\|z_t\|^2+\delta}.
$$

Smooth it with

$$
\bar v_t=(1-\tau)\bar v_{t-1}+\tau v_t,
\qquad
\tau=1-2^{-1/h_v}.
$$

The current $v_t$ may reduce trust for the current update because it is
computed from the current data before the deployed update. The current
$z_t$ may update $B_{t+1}$ only; it cannot rotate $B_t$ and then receive
credit for explaining itself.

Define

$$
\alpha_t=\frac{1}{1+k_\alpha\bar v_t},
$$

and an innovation-dependent geometry rate

$$
\beta_t
=\beta_{\min}
+(\beta_{\max}-\beta_{\min})
\frac{k_\beta\bar v_t}{1+k_\beta\bar v_t}.
$$

Thus high innovation decreases confidence in the current geometry and speeds
future basis adaptation. Controller constants are selected only in the
development phase and then frozen.

## Applied Optimization Geometry

Normalize the retained spectrum so its mean gain inside the learned subspace
is one:

$$
\widetilde\Lambda_t
=\frac{r\Lambda_t}
{\operatorname{tr}(\Lambda_t)+\delta}.
$$

For the pure SubGD conditions,

$$
P_t=B_t\widetilde\Lambda_tB_t^T.
$$

For adaptive conditions with orthogonal floor $\epsilon\ge0$,

$$
\widetilde C_t
=B_t\widetilde\Lambda_tB_t^T
+\epsilon(I-B_tB_t^T),
$$

$$
P_t=(1-\alpha_t)I+\alpha_t\widetilde C_t.
$$

The corresponding gains are

$$
\gamma_{t,j}^{\parallel}
=(1-\alpha_t)+\alpha_t\widetilde\lambda_{t,j},
\qquad
\gamma_t^\perp
=(1-\alpha_t)+\alpha_t\epsilon.
$$

Use a fixed number $H$ of first-order steps for every condition:

$$
\theta_t^{(\ell+1)}
=\theta_t^{(\ell)}
-\eta_\ell P_t\nabla J_t(\theta_t^{(\ell)}),
\qquad \ell=0,\ldots,H-1.
$$

$P_t$ remains fixed within one batch update. The scalar learning-rate
schedule and $H$ are calibrated using only the full-space control and frozen
before treatment outcomes are generated.

This finite compute contract is essential. If an invertible $P_t$ were used
only as a coordinate change and every branch optimized $J_t$ to convergence,
all full-rank conditions would approach the same minimizer and the adaptive
controller would become scientifically vacuous. Plan 13 therefore studies
joint sample and bounded-compute efficiency and reports objective residuals
so incomplete optimization is visible.

## Experimental Conditions

The complete vocabulary is:

1. **No update:** retain $\theta_K$; a preservation boundary, not a learner.
2. **Full space:** $P_t=I$ under the common retained objective.
3. **Head only:** update only the 225 identifiable classifier-head
   coordinates under the same objective.
4. **Random rank matched:** use a deterministic Haar basis with rank $r$ and
   the learned $\widetilde\Lambda_K$ spectrum. This separates learned
   orientation from dimensionality and spectral scaling.
5. **Static learned projector:** $P_t=B_KB_K^T$. This separates basis quality
   from covariance eigenvalue weighting.
6. **Static SubGD:** use fixed $B_K,\Lambda_K$ for all $t>K$.
7. **Online SubGD:** update $B_t,\Lambda_t$ with a fixed $\beta$, with
   $\alpha_t=1$ and $\epsilon=0$.
8. **Adaptive online SubGD:** update the geometry with $\beta_t$ and use the
   innovation controller with $\epsilon=0$.
9. **Adaptive online SubGD plus floor:** as above with one frozen
   $\epsilon>0$.

Do not launch this full list at confirmation scale. Conditions 1, 3, 4, and 5
are controls or development ablations. The prospective comparison sequence is

$$
\text{full space}
\longrightarrow
\text{static SubGD}
\longrightarrow
\text{online SubGD}
\longrightarrow
\text{adaptive online SubGD}
\longrightarrow
\text{adaptive plus floor}.
$$

## Outcomes And Estimands

### Primary Statistical-Efficiency Outcome

For each schedule, the primary outcome is post-burn-in current-environment
NLL area under the learning curve, normalized by post-burn-in exposure. Lower
is better. The principal estimand for treatment $a$ against comparator $b$ is
the mean paired replica difference

$$
\Delta_{a,b}^{\mathrm{NLL}}
=\mathbb E
\left[
\operatorname{NLLAUC}_b-
\operatorname{NLLAUC}_a
\right],
$$

so positive values favor treatment $a$. Linear and sigmoid schedules are
separate experimental conditions and are never pooled into one nominal
replica count.

Also report the end-to-end NLL AUC including burn-in and samples required to
cross frozen NLL and accuracy thresholds. Thresholds are selected from
development controls before confirmation and remain secondary because a
single threshold discards most of the trajectory.

### Retention And Final Performance

At every evaluation point, measure NLL and accuracy on the current
environment, the initialization environment, every frozen retention panel,
and the worst panel. Report retention NLL AUC as a coequal tradeoff rather
than hiding it inside current-environment performance.

Final NLL, accuracy, Brier score, calibration error, classwise recall, and
confusion matrices distinguish faster learning from a different terminal
solution. A method is a "best balance" only if the current-versus-retention
Pareto comparison supports that description.

### Mechanistic Diagnostics

Record at every step where defined:

- $\|z_t\|$, shadow objective decrease, and shadow optimizer work;
- $v_t$, $\bar v_t$, $\alpha_t$, $\beta_t$, and $\gamma_t^\perp$;
- the eigenvalues of $\Lambda_t$ and their effective rank;
- raw and relative residual energy $\|(I-B_tB_t^T)z_t\|^2$;
- principal angles and projector distance between $B_t$ and $B_{t+1}$;
- overlap with the head subspace and leading Fisher eigenspace;
- alignment of the accepted update with $z_t$;
- dense-versus-streaming covariance and subspace error where available;
- data loss, EWC penalty, total objective, gradient norm, and objective
  residual after the fixed compute budget; and
- displacement energy in the learned, orthogonal, Fisher-resolved, and
  Fisher-unresolved subspaces.

The expected mechanism is

$$
\text{regime mismatch}
\Rightarrow v_t\uparrow
\Rightarrow \alpha_t\downarrow
\Rightarrow \gamma_t^\perp\uparrow
\Rightarrow \beta_t\uparrow
\Rightarrow B_{t+1}\text{ rotates}
\Rightarrow v_{t+j}\downarrow.
$$

Predictive lift without the corresponding diagnostic sequence remains an
empirical optimizer result, not evidence for this mechanism.

### Resource Outcomes

Report burn-in and post-burn-in costs separately:

- wall-clock and GPU time;
- peak host and GPU memory;
- objective and gradient evaluations;
- trainable coordinates per condition;
- bytes for $B_t$, $\Lambda_t$, controller state, and any shadow branch;
- streaming eigensystem time; and
- overhead relative to full-space and head-only controls.

The shadow displacement is part of the deployed cost of the principal online
method. It may not be omitted from efficiency comparisons.

## Experimental Phases

### Phase 0: Contracts, Reproduction, And Smoke Timing

1. Reproduce the gauge-fixed Plan 12 isotropic-ridge rotation control from its
   immutable configuration and verify NLL, accuracy, displacement, and archive
   ordering against completed artifacts.
2. Implement the environment-neutral geometry, optimizer, artifact, and
   analysis interfaces plus rotation and digit-9-mixture adapters.
3. Verify that every condition receives an identical objective before its
   geometry is applied.
4. Unit-test the streaming rank-one update against dense EMA covariance,
   including zero and nearly dependent residuals.
5. Verify that shadow branches leave the learner, archive, optimizer, random
   streams, and evaluation counters unchanged.
6. Run tiny CPU trajectories in both environments, spanning all condition
   types and deliberate interruption/resumption.
7. Benchmark full, static, online, and adaptive steps on CUDA. Replace all
   provisional compute estimates before production.

Hard blockers are a failed Plan 12 reproduction, broken gauge equivalence,
condition-dependent objective, noncausal geometry update, failed pairing, or
artifact mutation. Ordinary lack of predictive promise is not an integrity
failure.

### Phase 1: Geometry Identification On Shared Full-Space Trajectories

Generate a small development cohort of full-space rotation trajectories on
both schedules. Store every burn-in and post-burn-in unrestricted shadow
displacement. This cohort is used only to answer geometry questions:

1. Compare the frozen $(K,r)$ candidates by held-out displacement energy,
   directional reconstruction, basis stability, and memory.
2. Compare displacement and gradient-proxy bases by principal angles,
   held-out innovation, and computational cost.
3. Compare incremental rank-$r$ updates with the dense EMA oracle across the
   candidate covariance half-lives.
4. Measure overlap among the learned basis, classifier head, Fisher-leading
   subspace, and random rank-matched bases.
5. Freeze one $K$, $r$, the numerical contract for the principal shadow
   displacement, and a small set of covariance half-lives before Phase 2
   outcomes are generated. The gradient proxy remains diagnostic unless a
   separately reviewed amendment promotes it.

Use the same full-space artifacts to construct an **open-loop innovation
calibration** before any adaptive condition runs. Replay each candidate
$(K,r,h_C)$ geometry over the stored $z_t$ sequence without changing the
learner path. Predeclare familiar within-leg windows and short windows after
route knots, then estimate the corresponding distributions of $v_t$. Use
those distributions to generate a small controller set with meaningful
operating points rather than an arbitrary Cartesian grid:

- familiar observations should ordinarily retain high SubGD trust;
- knot innovations should produce a material, but not necessarily complete,
  relaxation toward full space;
- slow and fast covariance half-lives must be commensurate with the number of
  transitions between route events; and
- reject mappings whose $\alpha_t$ or $\beta_t$ is effectively constant,
  saturated, or driven by numerical zero displacements.

This replay can calibrate the controller's dynamic range and timing. It cannot
predict its closed-loop NLL effect because the replayed path is fixed.

Predictive outcomes from this cohort are displayed only as control health and
cannot select geometry hyperparameters.

### Phase 2: Foundational Rotation Pilot

Use fresh paired replicas on both schedules to compare:

- no update;
- full space;
- head only;
- random rank matched;
- static learned projector;
- static SubGD; and
- online SubGD over the frozen covariance half-life candidates.

The initial development size is 16 replicas per schedule. It estimates effect
directions and paired variances; it does not support a definitive significance
claim. Select one online half-life lexicographically:

1. reject numerical failures and material retention regressions;
2. minimize mean current-environment NLL AUC;
3. within a practically indistinguishable band, prefer lower shadow and
   eigensystem cost; and
4. break remaining ties in favor of the longer covariance half-life.

Interpretation must distinguish:

- learned versus random orientation;
- low rank versus head-only restriction;
- learned basis versus learned eigenvalue weighting; and
- online rotation versus a fixed basis.

If learned bases do not beat matched random bases, prominently classify the
SubGD mechanism as unsupported. Phase 3 may still run its fixed small
diagnostic ledger to determine whether innovation-controlled full-rank
relaxation rescues a stale low-rank method, but it is not called a promising
continuation.

### Phase 3: Adaptive Controller Probing And Development

Carry forward full space, static SubGD, the selected fixed-rate online SubGD,
adaptive online SubGD with $\epsilon=0$, and a small frozen floor set. The
adaptive conditions receive two levels of probing before confirmation.

#### Phase 3A: Closed-Loop Mechanism Probe

Run the controller candidates produced by Phase 1's open-loop calibration on
four fresh paired replicas per schedule. This probe eliminates mechanically
dysfunctional settings; it does not choose a winner from noisy endpoint
effects. Require:

- finite objectives, updates, bases, spectra, and controller states;
- a nontrivial but nonsaturated range of $\alpha_t$ and $\beta_t$;
- increased orthogonal gain around at least one predeclared route event;
- measurable subsequent basis rotation without loss of orthogonality;
- no catastrophic NLL, retention, or objective-residual excursion relative to
  full space; and
- wall-clock and memory overhead inside the Phase 0 feasibility envelope.

If every candidate fails, generate at most one revised controller set from the
observed mechanistic failure, document the revision in the notebook, and
repeat the probe on four new replicas per schedule. This is the only
adaptive-grid repair allowed before fresh development. A second universal
failure carries the least pathological candidate forward as a labeled
negative diagnostic rather than starting an open-ended tuning loop.

#### Phase 3B: Predictive Development

Use 16 new paired replicas per schedule for the surviving candidates. Develop
the controller sequentially rather than as a large Cartesian grid:

1. choose $h_v$ and $k_\alpha$ from innovation response and NLL/retention
   tradeoffs with the geometry rate held fixed;
2. choose the slow and fast covariance half-lives and $k_\beta$ with the trust
   mapping frozen; and
3. compare $\epsilon\in\{0,.01,.05,.10\}$ with every earlier choice frozen.

Report the complete response surface inspected at each sequential step. A
selected controller must show the proposed diagnostic sequence around at
least one predeclared route event; a favorable endpoint alone is insufficient
for a mechanistic claim. This combination of open-loop calibration, tiny
closed-loop probing, and fresh predictive development gives adaptive SubGD a
serious opportunity without spending confirmation-scale compute on arbitrary
gains.

At the end of Phase 3, freeze at most one adaptive controller and one
nonadaptive SubGD comparator. Estimate the fresh-replica count needed for
Phase 4 from conservative paired variance and a predeclared smallest effect
of interest, not the raw selected mean alone.

### Phase 4: Independent Rotation Confirmation

Before creating fresh artifacts, freeze the replica count, schedule-specific
estimands, smallest effects of interest, retention margin, and condition
ledger. The default compact comparison is:

1. full space;
2. head only;
3. selected static or fixed-rate online SubGD;
4. selected adaptive SubGD; and
5. random rank matched.

Analyze linear and sigmoid schedules separately with paired replica-level
means, confidence intervals, effect sizes, medians, and win counts. No
cumulative significance testing or outcome-driven stopping is allowed.

For purposes of classifying the method carried into the cross-environment
study, a Plan 13 method is **promising** only if fresh Phase 4 evidence shows
one of the following:

- at least a 1% relative current-NLL-AUC improvement over full space with no
  greater than 1% relative retention-NLL-AUC harm; or
- at least a 1% relative retention-NLL-AUC improvement with no greater than
  1% relative current-NLL-AUC harm.

The corresponding paired confidence interval must exclude zero in at least
one schedule, the other schedule must not show material harm, and the method
must beat the matched random-basis control in the schedule supporting the
claim. These are promotion rules, not universal definitions of scientific
importance.

### Phase 5: Digit-9-Mixture Transport

This section records the frozen protocol used by the original Phase 5. Its
artifacts are retained, but the optimizer diagnostic in Phase 5R invalidates
their use as evidence for or against SubGD.

Phase 5 is part of the default execution and starts automatically after the
Phase 4 integrity checks and notebook refresh. It runs even if rotation is
unfavorable, because the mixture path may expose a different adaptation
geometry and the bounded transport cohort is useful negative evidence.

Freeze 16 new mixture replicas before the phase begins. The environment
adapter emits the accepted Plan 3 stream, evaluation panels, coordinate
$p_t$, and schedule metadata. Keep the Plan 13 gauge chart, selected $K$ and
$r$, geometry estimator, fixed-step optimizer budget, and controller
hyperparameters unchanged. Use the historical within-mixture pairing
contract, but never reuse old outcomes as new replicas.

Compare exactly:

1. full space;
2. head only;
3. matched random basis;
4. static SubGD; and
5. the dynamic method frozen before mixture outcomes exist.

If Phase 4 opens the promotion gate, item 5 is the promoted rotation method.
Otherwise, carry the Phase 3-selected adaptive controller as a labeled
**diagnostic best bet**, even if Phase 4 was negative. Do not reselect a
different method from the Phase 4 result merely because it appears more
likely to succeed on mixture.

The 16-replica mixture cohort is a transport study, not independent
confirmation and not a new hyperparameter search. Report its paired effect
estimates and uncertainty without adding its tests to a claim that Plan 13
succeeded somewhere. Rotation and mixture are separate experiments and are
never pooled as replications. Any mixture-specific tuning or confirmatory
expansion requires a later amendment and fresh replicas.

### Phase 5R: Digit-9 Optimizer Repair Amendment

Phase 5R is a post-diagnostic development repair. It does not overwrite Phase
5, count the reused streams as fresh confirmation, or change any rotation
result. Its purpose is to determine whether the frozen Phase 5 geometry
conditions are viable when the common retained objective is actually solved.

#### Evidence Requiring Repair

A three-replica, pre-treatment $2\times2$ diagnostic reused the immutable
Phase 5 assets and crossed optimizer with EWC presence. Every condition used
the Plan 13 proposal ordering in which the learner sees the prior Fisher
archive and the current direct Fisher observation enters the archive after the
proposal. At the first treatment coordinate, $p=32/99$, the means were:

| Optimizer | Retention | 9 recall | 9 specificity | Current accuracy | $p=0$ accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| first-order Armijo | EWC | .105 | .973 | .586 | .816 |
| first-order Armijo | none | .813 | .804 | .690 | .631 |
| strong-Wolfe L-BFGS | EWC | **.818** | **.889** | **.760** | **.732** |
| strong-Wolfe L-BFGS | none | .801 | .870 | .662 | .596 |

The paired L-BFGS-minus-Armijo EWC recall differences were $.771$, $.868$,
and $.500$. Median displacement norm increased from $.049$ to $.455$, while
the median relative final gradient norm fell from $.583$ to $.056$. Because
Armijo learned digit 9 without EWC and L-BFGS learned it with EWC under the
same prior-archive ordering, the supported diagnosis is an interaction between
the stiff EWC objective and the first-order solver. Fisher-update ordering is
not needed to explain the failure.

These three reused replicas are diagnostic, not an inferential sample. The
large within-replica separation and restoration of the historical recall
level justify repairing the implementation; they do not establish a new
performance claim.

The immutable aggregate is
`cache/mnist_experiment/continual_subgd/default/digit9_mixture/phase5_diagnostic/`
`digit9_mixture__phase5_diagnostic__optimizer_parity_analysis__00000__9dfba557cb7f25d6/summary.json`.
Its schema is `plan13-phase5-optimizer-parity-v1`; the twelve contributing
cells and their source-asset hashes must remain available to the notebook.

#### Frozen Scope

Reuse the 16 immutable Phase 5 initialization, stream, evaluation, and initial
Fisher assets. Do not regenerate or alter them. Create a new protocol version,
ledger, burn-in namespace, trajectory namespace, metric schema, and notebook
section named `phase5r`. The source Phase 5 asset hash is part of every Phase
5R unit identity.

Preserve all non-optimizer mixture choices:

- $p_t=t/99$ and eight observations per step;
- fixed $\pi=.05$, hence $\omega_9=19$;
- $\kappa_{9,t}=0$;
- direct EMA and rank-eight-plus-diagonal Fisher compression;
- the prior-archive proposal ordering diagnosed above;
- the selected $K$, rank, geometry estimator, controller, and floor values;
- the gauge-fixed 487-coordinate model; and
- the five frozen conditions: full space, head only, matched random basis,
  static SubGD, and the previously selected dynamic diagnostic best bet.

No Phase 5R result may select a different rank, controller, floor, or dynamic
method. Any such choice requires another development amendment and cannot be
evaluated on the same 16 streams as fresh evidence.

#### Coordinate L-BFGS Estimand

At outer step $t$, freeze the treatment geometry for the duration of the
local solve. Let $M_t$ denote the positive-semidefinite preconditioner declared
by that condition and choose an explicit factor $A_t$ satisfying

$$
A_tA_t^T=M_t.
$$

Optimize the local coordinates $q$, initialized at zero:

$$
\phi_t(q)=J_t(\theta_t+A_tq),
\qquad
q_0=0,
\qquad
\theta_{t+1}=\theta_t+A_t\widehat q_t.
$$

Use differentiable functional model evaluation; copying a detached coordinate
vector into the model inside the objective is not an acceptable substitute.
Reset L-BFGS history at every outer step so changing bases cannot transport
curvature state across incompatible coordinate systems.

The factors are condition-specific:

- full space uses $A_t=I$;
- head only uses the coordinate-selection matrix for the 225 gauge-fixed head
  coordinates;
- a pure learned or random rank-$r$ condition uses
  $A_t=B_tD_t^{1/2}$, with $D_t=I$ for a projector-only condition and the
  frozen normalized adaptation eigenvalues for spectral SubGD; and
- a relaxed adaptive condition factors its complete declared geometry

$$
M_t
=(1-\alpha_t)I
+\alpha_t\left[
B_tD_tB_t^T+\epsilon(I-B_tB_t^T)
\right].
$$

For the last case, the parallel eigenvalues are

$$
\mu_{t,j}=(1-\alpha_t)+\alpha_tD_{t,j},
$$

and the orthogonal eigenvalue is

$$
\mu_{t,\perp}=(1-\alpha_t)+\alpha_t\epsilon.
$$

The implementation may apply the analytic square-root operator without
materializing a dense $487\times487$ factor. Numerically zero eigenvalues are
removed from the coordinate chart and recorded; they are not lifted by an
undeclared ridge.

For $A_t=I$, use the accepted direct model-parameter L-BFGS implementation.
This is the identity coordinate chart and reproduces the diagnostic baseline
without the avoidable float32 roundoff introduced by repeatedly forming
$\theta_t+q$. Every nonidentity factor uses differentiable affine-coordinate
functional evaluation. Both branches optimize the same declared objective
with the same L-BFGS settings and reset state at every outer step.

Every local solve uses the accepted historical settings: strong-Wolfe L-BFGS,
learning rate $1$, at most 50 iterations, at most 75 objective evaluations,
history size 20, gradient tolerance $10^{-5}$, and change tolerance $10^{-9}$.
These are common caps, not a promise of equal realized function evaluations.
Record realized iterations, evaluations, objective decrease, data-loss
decrease, penalty, displacement norm, and initial and final gradient norms.

For every non-full-space condition, calculate the unrestricted shadow
displacement by a separate full-space L-BFGS solve from exactly the same
pre-update state, objective, Fisher archive, and batch. Restore the deployed
state before solving the treatment coordinates. The shadow branch cannot
change model state, archive state, controller state, random state, or counters
used by the deployed condition.

#### Burn-In Viability Gate

Regenerate every burn-in trajectory and every adaptation basis with
full-space L-BFGS. The original Armijo burn-in displacements and all bases
derived from them are scientifically invalid for Phase 5R and must not be
loaded, projected, or warm-started.

After all repaired burn-ins reach $K=32$, evaluate the common post-burn state
at $p=32/99$. Before launching any treatment trajectory, require:

1. all artifact, pairing, displacement-identity, and finite-value checks pass;
2. mean digit-9 recall across the 16 replicas is at least $.60$;
3. mean non-9 accuracy is at least $.65$; and
4. mean balanced digit-9 one-vs-rest accuracy is at least $.70$.

These broad thresholds are an engineering validity gate, not significance
tests or success criteria. They were frozen after the diagnostic and are below
both its L-BFGS means and the historical recall reference. If any gate fails,
do not launch SubGD treatments. Refresh the notebook with the failed gate and
retain the completed burn-ins for analysis.

#### Interpretation Boundary

The repaired cohort estimates treatment differences under coordinate
L-BFGS. It may rehabilitate Phase 5 as a useful development study, but reuse of
the same streams after inspecting the failed outcomes prevents confirmatory
language. A favorable result can motivate a separately powered experiment
with fresh replicas. An unfavorable result, after the viability gate passes,
is meaningful evidence against these frozen SubGD geometries on the mixture
path.

Changing the optimizer only for Phase 5R means it is not a literal transport
of the rotation solver. Report it as transport of the frozen geometry
algorithm under an environment-appropriate common solver. Do not revise or
pool the rotation estimates. A later amendment would be required to rerun the
rotation study with coordinate L-BFGS.

#### Execution Result (2026-10-05)

All 16 repaired burn-ins passed their artifact checks. At $p=32/99$, the
viability-gate means were $.7673$ digit-9 recall, $.8288$ balanced digit-9
one-vs-rest accuracy, and $.7345$ non-9 accuracy, so treatment execution was
unlocked. All 80 treatment trajectories and the paired aggregate then
completed.

Against repaired full-space learning, head-only tuning reduced current NLL
AUC by $.02479$ (paired 95% CI $[.01072,.03886]$) and retention NLL AUC by
$.28531$, but reduced digit-9 recall AUC by $.01512$. Static SubGD's current
NLL gain was $.00552$ (95% CI $[-.01078,.02182]$), with lower current accuracy
and digit-9 recall. The selected adaptive condition was effectively tied with
full space: current NLL gain $-.00233$ (95% CI $[-.00755,.00289]$), current
accuracy gain $-.00030$, and balanced digit-9 gain $.00007$.

This post-diagnostic development cohort therefore validates the optimizer
repair but provides no evidence that either frozen SubGD geometry outperforms
full-space coordinate L-BFGS on the digit-9-mixture path. Head-only tuning is
the strongest practical retention/current-NLL tradeoff in this cohort, with
its recall cost kept explicit. These reused streams are not fresh
confirmation.

### Phase 6: Low-Prevalence Few-Shot Digit-9 Study

Phase 5R placed treatment assignment at $p=32/99\approx.323$, outside the
region now judged scientifically interesting. Its rising current accuracy was
also dominated by the changing class prevalence: at $p=1$, current ten-class
accuracy is exactly digit-9 recall. Phase 6 is a new prospective experiment,
not a reinterpretation or continuation of Phase 5R. It asks how adaptation
geometries behave under the severe and application-relevant constraint
$0\le p\le.1$ with essentially no calibration period.

Phase 6 requires fresh initialization, stream, and evaluation seeds. No Phase
5, Phase 5R, optimizer-diagnostic, or Phase 6 smoke outcome counts as a Phase
6 production replica. Existing code and frozen hyperparameters may be reused;
observed trajectories may not.

#### Low-Prevalence Schedule And Burn-In

Use the explicit versioned schedule

$$
\mathcal P_6=\{0,.01,.02,\ldots,.10\},
$$

with eight new observations at every coordinate. Store the resolved decimal
values and schedule hash in each stream artifact. Generate no training or
primary evaluation coordinate above $.1$.

Apply exactly one common full-space L-BFGS update at $p=0$. This is the entire
burn-in and gives $K=1$. Evaluate and archive the state both before and after
that update. Assign treatments before the $p=.01$ batch, so every treatment
receives 80 post-burn-in observations over ten steps. The expected number of
digit-9 observations over that complete treatment window is only

$$
8\sum_{j=1}^{10}\frac{j}{100}=4.4.
$$

This scarcity is part of the estimand, not a failed viability gate. Record the
realized digit-9 count in every batch and cumulatively, but do not condition,
resample, reject, or stratify replicas by that realized count.

The one burn-in shadow displacement defines

$$
C_1=z_0z_0^T.
$$

Its effective rank is at most one. Do not pad it to the historical rank 16,
inject synthetic directions, borrow a basis from another replica, or use a
Phase 5R basis. If $z_0$ is numerically negligible, record a rank-zero basis
and execute the declared rank-zero behavior. This intentionally exposes the
calibration requirement of SubGD.

#### Frozen Objective And Solver

Preserve the repaired mixture objective and optimizer from Phase 5R:

$$
\omega_9=\frac{1-.05}{.05}=19,
\qquad
\kappa_{9,t}=0,
$$

with direct-EMA rank-eight-plus-diagonal Fisher updating, prior-archive
proposal ordering, the gauge-fixed 487-coordinate model, and reset-per-step
strong-Wolfe coordinate L-BFGS. All optimizing conditions receive the same
batch, initial model, Fisher archive, anchor, solver tolerances, and objective.
Only the declared coordinate geometry differs.

The unrestricted shadow branch remains the geometry observation $z_t$. It is
computed before the deployed update from the same pre-update state and cannot
alter the learner, archive, controller, counters, or random state. Online
bases update only after the current treatment geometry has acted.

#### Frozen Conditions

Run exactly these seven paired conditions:

1. **No update:** retain the common post-burn-in state. This is a boundary
   control, not a competitive optimizer.
2. **Full space:** direct full-parameter L-BFGS with $A_t=I$.
3. **Digit-9 bias only:** optimize the one-dimensional logit contrast that
   adds $\delta$ to the digit-9 logit and $-\delta/9$ to every non-9 logit.
   Map this functionally sum-to-zero direction into the gauge-fixed chart and
   verify it by raw-model functional equivalence.
4. **Head only:** optimize the established 225-coordinate gauge-fixed output
   head.
5. **Matched random rank one:** use one deterministic random identifiable
   direction with the same normalized one-dimensional spectrum as the burn-in
   basis. Freeze its orientation after treatment assignment.
6. **Static tiny-burn-in SubGD:** freeze the rank-zero-or-one basis and
   spectrum obtained from $z_0$.
7. **Adaptive online SubGD plus floor:** start from the same $C_1$, use the
   Phase 3-selected controller and $epsilon=.1$, and update causally from
   unrestricted shadows with rank cap 16. Its realized rank can grow by at
   most one after each informative observation and must be reported.

Conditions 3 and 4 encode stable structural priors and do not receive an
estimated adaptation basis. Condition 5 tests whether any static SubGD result
comes from the learned orientation rather than one-dimensional restriction.
Condition 7 is the previously selected adaptive best bet; Phase 6 does not
retune its floor, controller, rank cap, or rates.

For a rank-zero static or random factor, the accepted displacement is exactly
zero. For adaptive SubGD, use the declared full-rank floor/controller fallback
without manufacturing low-rank eigenvectors. Record the parallel and
orthogonal gains so early full-space-like behavior is visible rather than
described as learned SubGD.

#### Precision And Recall Estimands

Use the fixed holdout panel to estimate digit-9 recall and non-9 specificity
at every model state:

$$
R_{9,t}=\Pr(\widehat Y=9\mid Y=9),
\qquad
S_{9,t}=\Pr(\widehat Y\ne9\mid Y\ne9).
$$

The principal precision metric standardizes every state to the fixed
reference prevalence $p_{\mathrm{ref}}=.1$:

$$
Q_{.1,t}
=
\frac{.1R_{9,t}}
{.1R_{9,t}+.9(1-S_{9,t})}.
$$

This reference prevalence affects evaluation only. It never changes the
training stream, loss, Fisher estimate, model, or classification threshold.
If both the numerator and denominator are zero because the model predicts no
positives, set reported precision to zero and record a separate
`no_predicted_positive` indicator.

The two co-primary trajectory outcomes are exposure-normalized
fixed-$.1$-precision AUC and digit-9-recall AUC over
$p\in[.01,.1]$. Higher is better. Do not collapse them into F1 or another
single score. Report paired effects jointly and show the precision-recall
tradeoff explicitly.

The point precision and recall metrics use the ordinary argmax decision. For
treatment $a$ against comparator $b$, define favorable paired effects as

$$
\Delta_{a,b}^{Q}
=
\mathbb E[\operatorname{AUC}(Q_{.1}^{a})
-\operatorname{AUC}(Q_{.1}^{b})],
$$

$$
\Delta_{a,b}^{R}
=
\mathbb E[\operatorname{AUC}(R_9^{a})
-\operatorname{AUC}(R_9^{b})].
$$

Thus positive values favor treatment $a$ for both outcomes. Preserve the two
components even when one treatment is Pareto-incomparable with another.

The single prespecified primary comparison is head only versus full space.
The SubGD-specific comparison is adaptive online SubGD plus floor versus full
space; head only versus adaptive SubGD is the prespecified practical contrast.
Bias only versus head only tests whether the changing task is effectively
one-dimensional. Static SubGD and matched random rank one are mechanistic
controls, not additional primary claims.

Separately calculate a threshold-swept precision-recall curve from the
digit-9 softmax score at $p_{\mathrm{ref}}=.1$. At each threshold, estimate
recall and false-positive rate separately on the two holdout strata, then
apply the same fixed-prevalence formula. Store the compact curve and
standardized average precision in the computational artifact; the notebook
must not rerun model inference to construct it.

Secondary outcomes are:

- false positives per 1,000 non-9 observations, $1000(1-S_{9,t})$;
- current-mixture and fixed-$.1$ precision, shown with distinct labels;
- actual-current-mixture NLL AUC on $p\in[.01,.1]$;
- $p=0$ retention NLL and ten-class accuracy;
- Brier score and calibration error at fixed prevalence $.1$;
- worst-stratum NLL, objective residuals, and compute cost; and
- current ten-class and binary OvR accuracy as de-emphasized diagnostics.

No headline Phase 6 plot may use raw current accuracy without displaying its
prevalence decomposition. No metric evaluated above $p=.1$ enters a Phase 6
claim.

#### Replicas And Interpretation

Freeze 64 fresh paired production replicas before execution. This larger
initial cohort reflects the roughly 4.4 expected digit-9 observations per
trajectory and is not a retrospective power calculation from Phase 5R.
Report mean paired effects, 95% confidence intervals, standardized paired
effects, medians, and favorable-replica counts. The head-only-versus-full
comparison is primary; all other intervals retain their descriptive meaning
without selecting a winner by the smallest observed $p$-value.

There is no performance viability gate. A rank-zero basis, zero observed
nines in an early batch, poor SubGD learning, or disappointing precision is a
valid result and cannot trigger replacement, delayed treatment assignment,
or reuse of another basis. Halt only for artifact corruption, broken pairing,
nonfinite optimization, functional-contrast failure, or another hard contract
violation.

Phase 6 is prospectively specified after inspecting Phase 5R but uses a new
low-prevalence estimand and fresh replicas. It may support inference for that
estimand after all 64 replicas complete. It is not confirmation of the
rotation study, and its evidence is never pooled with Phase 5 or Phase 5R.

#### Execution And Notebook Contract

Implement a new immutable `phase6_low_prevalence` ledger and run namespace.
The runner must support `--resume`, `--max-units`, and `--max-wall-seconds`,
validate every completed source hash, and refresh the artifact-only notebook
after every bounded batch. A CPU smoke and one-replica CUDA smoke validate
the schedule, one-dimensional bias factor, rank-zero/rank-one behavior,
precision reconstruction, PR-curve artifact, and interruption recovery; they
cannot alter frozen production choices.

The Phase 6 notebook section must show, as artifacts accumulate:

- realized and cumulative digit-9 counts;
- fixed-$.1$ precision and recall trajectories with uncertainty;
- false positives per 1,000 non-9 observations;
- fixed-$.1$ precision-recall curves at model states $p=.01,.05,.10$;
- $p=0$ retention and actual-current NLL trajectories;
- realized SubGD rank, innovation, floor gain, and shadow alignment;
- paired effect distributions for every prespecified contrast; and
- an explicit incomplete-ledger banner until all 64 replicas finish.

Phase 6 requires separate user authorization after review of this amendment.

#### Phase 6 Execution Result

Phase 6 completed on October 5, 2026 with all 64 fresh paired replicas, 448
treatment trajectories, and the fixed-cohort analysis immutable. The full
577-unit ledger completed without an efficacy stop, and a subsequent
`--resume` invocation performed zero work. Production computation from the
first asset start through analysis completion took approximately 35.4 minutes
on the available CUDA device. The mean realized treatment count was 4.1719
digit-9 observations per replica, close to the design expectation of 4.4.

For the primary head-only-versus-full-space comparison, fixed-$.1$ precision
AUC gain was $.00700$ with paired 95% CI $[-.00314,.01715]$ and 34 of 64
favorable replicas. Digit-9 recall AUC gain was $.01352$ with paired 95% CI
$[.00247,.02456]$ and 42 of 64 favorable replicas. Head-only tuning also
improved actual-current NLL AUC by $.19070$ with paired 95% CI
$[.13837,.24302]$ and improved $p=0$ retention NLL AUC by $.09133$ with paired
95% CI $[.05827,.12438]$. It is therefore the strongest practical condition,
but not a joint co-primary win because its precision interval includes zero.

Adaptive SubGD with floor $.1$ was effectively tied with full space on both
co-primary outcomes: precision AUC gain $.00145$ with paired 95% CI
$[-.00198,.00489]$, and recall AUC gain $.00034$ with paired 95% CI
$[-.00307,.00375]$. Its actual-current NLL AUC gain was $-.00957$ with paired
95% CI $[-.01608,-.00305]$, favoring full space. Its realized basis grew from
rank one to rank 11 as specified, so this null result is not explained by a
failure to execute online rank growth.

The static tiny-burn-in and matched random rank-one conditions both had zero
argmax digit-9 recall and therefore zero fixed-$.1$ precision throughout the
exposure-normalized trajectory. The learned static orientation nevertheless
improved current NLL AUC over the random orientation by $.15121$ with paired
95% CI $[.11454,.18787]$; this is evidence of orientation-specific soft-loss
structure, not usable rare-class detection. Bias-only tuning preserved the
$p=0$ environment and reduced false positives relative to head-only, but was
substantially worse on precision, recall, and current NLL. The immutable
notebook reports the complete trajectories, PR curves, geometry diagnostics,
and paired effect distributions.

### Phase 6A: Ten-Positive Low-Prevalence Scaling Study

Phase 6 intentionally tested an extreme few-shot regime and succeeded in
estimating the behavior of that regime. Its mean treatment stream contained
only 4.1719 realized digit-9 observations against a design expectation of
4.4. Phase 6A does not repair, replace, or relabel that result. It asks a new
development question: what precision-recall tradeoff is available when the
same low-prevalence path receives a still-small but less severe data budget of
approximately ten expected digit-9 examples?

This amendment was written after inspecting Phase 6. Its condition pruning,
sample size, comparisons, and plots are therefore transparently exploratory.
Phase 6A can motivate the separate head-focused plan, but it is not an
independent confirmation of head-only tuning and cannot be pooled with Phase
6 to manufacture a larger confirmatory cohort.

#### Fixed Schedule And Exposure

Retain the exact Phase 6 prevalence schedule

$$
\mathcal P_{6A}=\{0,.01,.02,\ldots,.10\}.
$$

Increase the fixed batch size from 8 to 18 new observations at every
coordinate. Apply one common full-space L-BFGS burn-in update to the 18
observations at $p=0$, preserve $K=1$, and assign treatments before the
$p=.01$ batch. Every condition then receives 180 post-burn-in observations.
The expected number of observed 9s in the treatment window is

$$
\mathbb E[N_9]
=
18\sum_{j=1}^{10}\frac{j}{100}
=9.9.
$$

The value 18 is frozen by this arithmetic. The runner does not stop when ten
9s have appeared, add observations after an unlucky stream, reject replicas,
or balance batches by class. Record the realized batch and cumulative counts
exactly as in Phase 6. The resulting estimand is the fixed 18-observation
workflow, not performance conditional on observing ten positives.

The larger $p=0$ batch also changes the common burn-in state and its one-step
geometry observation. Consequently, cross-phase Phase 6 versus Phase 6A
differences describe the complete change in batch regime; they do not isolate
a causal effect of adding exactly 5.5 expected positives. Within Phase 6A,
all treatment comparisons remain paired.

#### Frozen Practical Conditions

Run these five paired conditions:

1. **No update:** retain the common post-burn-in state as the boundary
   control.
2. **Full space:** use direct full-parameter coordinate L-BFGS with $A_t=I$.
3. **Digit-9 bias only:** use the same functionally sum-to-zero logit contrast
   as Phase 6.
4. **Head only:** optimize the established 225-coordinate gauge-fixed output
   head.
5. **Adaptive online SubGD plus floor:** use the Phase 6 controller, rank cap
   16, and $\epsilon=.1$ without retuning.

The static tiny-burn-in and matched-random rank-one conditions are not rerun.
Their zero argmax recall in Phase 6 already answered the mechanistic question
needed here, while Phase 6A is intended to compare practical learning choices
at a larger batch size. This outcome-informed pruning is another reason to
label Phase 6A as development evidence. It does not erase those controls from
the Phase 6 notebook.

Reuse the Phase 6 likelihood, class weighting, direct-EMA
rank-eight-plus-diagonal Fisher archive, proposal ordering, EWC objective,
coordinate strong-Wolfe L-BFGS tolerances, gauge-fixed chart, shadow branch,
and causal online-basis update. Do not tune a learning rate, floor, rank,
threshold, or stopping rule from Phase 6A outcomes.

#### Estimands And Reporting

Use the same independently sampled stratified holdout construction,
$p_{\mathrm{ref}}=.1$ precision standardization, argmax operating point, and
threshold-swept digit-9 score as Phase 6. The paired co-primary trajectory
outcomes remain fixed-$.1$ precision AUC and digit-9 recall AUC over
$p\in[.01,.1]$. The prespecified practical comparison is head only versus
full space. Head only versus no update establishes absolute learning, while
adaptive SubGD versus full space remains a labeled secondary comparison.

Report all five conditions. For readability, the notebook may additionally
show a focused practical precision-recall panel containing no update, full
space, head only, and adaptive SubGD, provided the adjacent complete panel and
table retain digit-9 bias only. A focused panel is presentation, not condition
selection.

At minimum, the Phase 6A notebook section shows:

- mean cumulative realized digit-9 count with the 9.9 design expectation;
- fixed-$.1$ precision and recall trajectories with uncertainty;
- false positives per 1,000 non-9 observations;
- threshold-swept precision-recall curves at $p=.01,.05,.10$, including an
  enlarged endpoint panel at $p=.10$;
- standardized average precision and the argmax operating point;
- actual-current and $p=0$ retention NLL trajectories;
- adaptive SubGD rank and innovation diagnostics; and
- paired effect distributions and confidence intervals for every declared
  comparison.

Do not choose a deployment threshold on the fixed evaluation panel. Any later
threshold selection requires a separate validation sample and belongs to the
following plan. Do not treat a visually smoother or higher Phase 6A PR curve
as evidence that the Phase 6 estimate was defective.

#### Replicas, Artifacts, And Authorization

Freeze 64 fresh paired production replicas. Use new initialization, stream,
evaluation-panel, and numerical-algorithm seeds; no Phase 6 production stream
or model state may become a Phase 6A treatment input. Implement an immutable
`phase6a_ten_positive` ledger and run namespace with a new protocol and schema
version. Preserve per-replica trajectories and compact PR artifacts.

The runner must support `--resume`, `--max-units`, and `--max-wall-seconds`,
refresh the artifact-only notebook after every bounded batch, and never mutate
a completed Phase 6 or Phase 6A unit. A CPU smoke and one-replica CUDA smoke
verify the 18-observation schedule, 9.9 expected count, pairing, resumption,
metric reconstruction, and notebook rendering. Smoke outcomes cannot change
the frozen design.

Phase 6A has no efficacy or appearance gate. Poor precision, poor recall, or
an unattractive PR curve is a valid result and does not trigger more data,
condition replacement, threshold tuning, or replica rejection. Halt only for
a hard integrity failure. Phase 6A required separate user authorization after
review of this amendment; that authorization was given before execution.

#### Phase 6A Execution Result

Phase 6A completed on October 6, 2026 with all 64 fresh paired replicas, 320
treatment trajectories, and the fixed-cohort analysis immutable. The full
449-unit ledger passed its integrity checks, and a subsequent `--resume`
invocation performed zero experimental work. Production computation from the
first asset start through analysis completion took approximately 28 minutes
49 seconds on the available RTX 4070; final notebook rendering brought the
end-to-end production invocation to approximately 30 minutes. The mean
realized treatment count was 9.7969 digit-9 observations per replica, close to
the design expectation of 9.9.

At the ordinary argmax operating point after the $p=.10$ batch, full space,
head only, and adaptive SubGD respectively reached mean precision/recall pairs
of $(.5241,.5183)$, $(.5166,.4854)$, and $(.5222,.5021)$. Their descriptive
threshold-swept standardized average precisions were $.5075$, $.4817$, and
$.4995$. Full space therefore has the strongest endpoint PR curve in this
exploratory cohort, although the three practical conditions occupy a similar
operating region.

For the prespecified head-only-versus-full-space comparison, fixed-$.1$
precision AUC gain was $.00738$ with paired 95% CI $[-.00270,.01746]$ and 35
of 64 favorable replicas. Digit-9 recall AUC gain was $-.00399$ with paired
95% CI $[-.01431,.00632]$ and 31 of 64 favorable replicas. Phase 6A therefore
does not establish a head-only precision-recall advantage over full-space
updating.

Head-only tuning did provide the strongest learning-retention balance on the
loss scale. Relative to full space, it improved actual-current NLL AUC by
$.03869$ with paired 95% CI $[.02481,.05258]$ and improved $p=0$ retention NLL
AUC by $.02905$ with paired 95% CI $[.01873,.03938]$. It also improved
actual-current accuracy AUC by $.00445$ with paired 95% CI
$[.00214,.00675]$. Adaptive SubGD was slightly worse than full space on
actual-current NLL AUC by $.00606$ with paired 95% CI
$[.00083,.01129]$ in full-space-favoring orientation, and did not improve the
co-primary precision or recall outcomes. The immutable notebook reports the
complete trajectories, threshold-swept PR curves, operating points, geometry
diagnostics, and paired effect distributions.

### Phase 6B: Hundred-Positive Low-Prevalence Closeout Study

Phase 6B closes the Plan 13 SubGD experiments with a direct scaling variant of
Phase 6A. It asks what precision-recall tradeoff the same practical learners
reach with approximately 100 expected observed digit-9 examples, while keeping
the difficult $p\leq.1$ path and every treatment definition fixed. It is not a
new hyperparameter search and does not attempt to rescue a condition selected
from the earlier plots.

This amendment was written after inspecting Phases 6 and 6A. Phase 6B is
therefore exploratory closeout evidence, not independent confirmation. The
hope of seeing 80--90% precision motivates a more legible operating regime but
is not an efficacy gate, stopping rule, threshold-selection criterion, or
license to change the experiment after results begin arriving.

#### Fixed Schedule And Exposure

Retain the exact prevalence schedule

$$
\mathcal P_{6B}=\{0,.01,.02,\ldots,.10\}.
$$

Use a fixed batch size of 182 observations at every coordinate. Apply the same
one-step common full-space L-BFGS burn-in to the 182 observations at $p=0$ and
assign treatments before the $p=.01$ batch. Each condition then receives 1,820
post-burn-in observations. The expected treatment count is

$$
\mathbb E[N_9]
=
182\sum_{j=1}^{10}\frac{j}{100}
=100.1.
$$

The fixed batch size is the nearest simple integer design to 100 expected 9s.
Do not stop at the hundredth observed 9, add samples to a low-count replica,
balance a batch by class, or reject a stream. Record realized counts. The
estimand is the fixed 182-observation workflow, not performance conditional on
observing exactly 100 positive examples.

As in Phase 6A, the larger $p=0$ batch changes the shared burn-in state and its
geometry observation. Cross-phase differences among Phases 6, 6A, and 6B are
descriptive dose-response evidence for complete batch regimes, not a causal
effect of positive-count alone. Treatment comparisons within Phase 6B remain
paired.

#### Frozen Conditions And Estimands

Run the five Phase 6A conditions without retuning:

1. **No update.**
2. **Full space.**
3. **Digit-9 bias only.**
4. **Head only.**
5. **Adaptive online SubGD plus floor** with the Phase 6 controller, rank cap
   16, and $\epsilon=.1$.

Reuse the Phase 6A likelihood, class weighting, direct-EMA
rank-eight-plus-diagonal Fisher archive, EWC objective, coordinate
strong-Wolfe L-BFGS tolerances, gauge-fixed chart, shadow branch, causal basis
update, holdout construction, $p_{\mathrm{ref}}=.1$ precision
standardization, argmax operating point, and threshold-swept score. Do not
tune the floor, rank, optimizer, controller, threshold, or stopping rule.

The paired co-primary trajectory outcomes remain fixed-$.1$ precision AUC and
digit-9 recall AUC. Head only versus full space remains the practical
comparison, head only versus no update establishes absolute learning, and
adaptive SubGD versus full space remains secondary. Report current and
retention NLL and accuracy, false positives, standardized average precision,
argmax precision and recall, geometry health, and every declared paired
effect. Keep all five conditions visible even if the focused practical panel
omits bias only.

Every Phase 6, 6A, and 6B low-prevalence notebook section and precision-recall
figure must explicitly state its design expectation for observed digit-9
examples per trajectory: 4.4, 9.9, and 100.1, respectively. Show the realized
cumulative count beside the expectation so readers can compare the regimes
without inferring exposure from a phase name.

#### Replicas, Artifacts, And Authorization

Freeze 64 fresh paired replicas. Use new initialization, stream,
evaluation-panel, and numerical-algorithm seeds; do not continue a Phase 6 or
6A model, stream, or trajectory. Implement a new immutable
`phase6b_hundred_positive` namespace, schema version, and 449-unit ledger.
Preserve per-replica metric trajectories and compact threshold-swept PR
artifacts.

The runner supports `--resume`, `--max-units`, and `--max-wall-seconds`,
refreshes the artifact-only notebook after bounded batches, and never mutates
a completed unit. A CPU smoke, interruption/resumption drill, one-replica CUDA
smoke, and zero-work resume must pass before or alongside production. Phase 6B
has no efficacy or appearance gate. Only a hard integrity failure can halt
execution. The user authorized implementation and execution with this
amendment.

#### Phase 6B Execution Result

Phase 6B completed on October 6, 2026 with all 64 fresh paired replicas, 320
treatment trajectories, and the fixed-cohort analysis immutable. All 449
ledger units passed their integrity checks. An interrupted three-unit CPU
smoke resumed to completion, the one-replica CUDA smoke passed, and both smoke
and production zero-work `--resume` invocations reported no new work. Mean
realized exposure was 101.78125 digit-9 observations per replica against the
100.1 design expectation.

At the ordinary argmax operating point after the $p=.10$ batch, the practical
conditions produced:

| Condition | Mean precision | Mean recall | Mean standardized AP |
| --- | ---: | ---: | ---: |
| Full space | .6785 | .6248 | .6890 |
| Head only | .6732 | .5838 | .6587 |
| Adaptive SubGD plus $.1$ floor | .6774 | .6276 | .6899 |

Adaptive SubGD therefore reached essentially the same endpoint
precision-recall region as full-space learning. For adaptive SubGD versus full
space, fixed-$.1$ precision AUC gain was $-.00218$ with paired 95% CI
$[-.00568,.00131]$, while recall AUC gain was $.00137$ with paired 95% CI
$[-.00004,.00278]$. Neither co-primary classification outcome establishes an
adaptive advantage.

The secondary loss and accuracy outcomes contain a small, consistent adaptive
effect. Relative to full space, adaptive SubGD improved actual-current NLL AUC
by $.00059$ with paired 95% CI $[.00006,.00113]$, actual-current accuracy AUC
by $.00017$ with CI $[.00003,.00032]$, and $p=0$ retention NLL AUC by $.00058$
with CI $[.00018,.00099]$. Its $p=0$ accuracy AUC gain was $.00007$ with CI
$[-.00007,.00020]$. These are statistically resolved but practically tiny
development effects; they do not establish a precision-recall gain.

Head only did not retain its low-exposure appeal at this data budget. Against
full space, its precision AUC gain was $-.00307$ with CI
$[-.01093,.00480]$, recall AUC gain was $-.01703$ with CI
$[-.02270,-.01136]$, and actual-current NLL AUC gain was $-.01487$ with CI
$[-.01767,-.01207]$. This closeout supports a dose-dependent interpretation:
structural restriction can be useful when positives are extremely scarce,
but full-space learning catches up as evidence accumulates; the $.1$-floor
adaptive method mostly behaves like full space while retaining a very small
regularization benefit.

Production ran from 17:11:21--17:40:41 UTC, approximately 29 minutes 20
seconds from the first asset start through analysis completion on the RTX
4070. Final report rendering brought the initial end-to-end invocation to
approximately 32 minutes. The 5--7 hour estimate was conservative because it
scaled Phase 6A time linearly with sample count; the 182-observation batches
used the GPU much more efficiently.

## Environment-Neutral Interface

The Plan 13 core may assume only that an environment adapter provides:

- a versioned high-quality initialization asset;
- a deterministic resolved environment schedule and coordinate name;
- paired online batches and independent evaluation panels;
- the environment-specific incumbent retention contract;
- pre-update and post-update evaluation semantics;
- route-event labels for mechanistic summaries; and
- stable content hashes for streams and panels.

The rotation adapter provides $\varphi_t$ and the accepted
$0^\circ\to30^\circ\to0^\circ\to30^\circ$ linear or sigmoid paths. The
mixture adapter provides $p_t=\mathbb P(M_i=1)$ and preserves the fixed
conditional distribution of digits 0 through 8. Neither adapter may rename
its coordinate to resemble the other, and neither changes the Fisher
estimand.

## Statistical Reporting

- The independent unit is a complete paired trajectory replica.
- Steps, classes, route legs, hyperparameters, and schedules do not multiply
  the replica count.
- Report means, confidence intervals, standardized paired effects, medians,
  and favorable-replica counts.
- For digit-9-mixture trajectories, report digit 9 one-vs-rest accuracy
  $p_t\,\mathrm{TPR}_{9,t}+(1-p_t)\,\mathrm{TNR}_{9,t}$ and balanced
  one-vs-rest accuracy alongside general ten-class accuracy. Keep class-9
  recall and non-9 specificity visible so prevalence cannot hide a degenerate
  classifier.
- For Phases 6, 6A, and 6B, fixed-$.1$ precision, recall, and false-positive
  rate replace binary OvR accuracy as the headline classification metrics.
  Keep current-mixture precision visibly separate from fixed-reference
  precision and label every plot with its expected observed digit-9 count.
- Keep development, selection, confirmation, and transport evidence visibly
  separate.
- Label original Phase 5, the optimizer diagnostic, and repaired Phase 5R as
  three distinct evidence layers. Phase 5R is post-diagnostic development,
  never fresh confirmation.
- Keep linear and sigmoid rotation schedules separate.
- Show all inspected controller and floor candidates, including failures.
- Label shadow-displacement and gradient-proxy results separately.
- Do not infer statistical efficiency from final accuracy alone.
- Do not call preservation "learning" without improvement away from the
  initialization environment and against the no-update control.
- Do not call a basis adaptive merely because $\beta_t$ changed; show
  principal-angle movement and subsequent innovation reduction.

## Immutable Execution And Resumption

Every expensive phase uses configuration-driven entry points and immutable
units under
`cache/mnist_experiment/continual_subgd/<environment>/`. A unit identity
includes the Plan 13 contract version, phase, environment, schedule, replica,
condition, statistic definition, $K$, rank, controller, and all geometry
hyperparameters.

All long runners must support:

- `--resume` with validation of completed units;
- `--max-units` and `--max-wall-seconds`;
- deterministic priority ordering;
- atomic temporary directories and final `COMPLETED` markers;
- restart of incomplete units without mutation of completed units; and
- explicit failure artifacts rather than silently dropped conditions.

The shared burn-in is a content-addressed immutable asset. Conditions reference
it by hash. Shadow displacements, accepted displacements, bases, spectra,
controller state, archive state, and metric rows are stored with schema
versions sufficient for artifact-only reanalysis.

Phase 5R must use new unit identities and may reference original Phase 5 only
through immutable source-asset hashes. It must reject any original Phase 5
burn-in, basis, optimizer state, or trajectory artifact as an input.

Phase 6 must use a new protocol hash, ledger, stream family, initialization
seeds, evaluation-panel seeds, and run namespace. It may import tested Phase
5R implementations, but must reject every Phase 5 or Phase 5R burn-in, basis,
stream, model state, and treatment trajectory as a production input.

Phase 6A must likewise use a new protocol hash, ledger, seed families, and run
namespace. It may share tested implementation code with Phase 6 but must not
mutate or continue a completed Phase 6 unit. Its manifests record the fixed
18-observation batch size and 9.9 expected treatment count.

Phase 6B must use another protocol hash, ledger, seed family, and run
namespace. It may share the Phase 6A implementation, but may not continue or
mutate a Phase 6A unit. Its manifests record the fixed 182-observation batch
size and 100.1 expected treatment count.

## Autonomous Execution

Once the user authorizes Plan 13 execution, Phases 0 through 5 may proceed
without mandatory review between phases. Scientific disappointment does not
halt the runner: it records the failure classification in the notebook and
uses the predeclared fallback or diagnostic best bet. Execution halts only for
a hard integrity failure such as broken pairing, incompatible normalization,
noncausal updates, corrupt required artifacts, failed gauge equivalence, or an
undefined required estimand without a declared fallback.

Phase 5R requires separate user authorization after review of this amendment.
Once authorized, implementation validation, the 16 repaired burn-ins, the
viability gate, and treatment trajectories may proceed without intermediate
review. A failed viability gate is a hard scientific stop for treatment
execution, not a disappointing result eligible for an automatic fallback.

Phase 6 also requires separate user authorization after review. Once
authorized, its implementation tests, smoke runs, 64 production replicas,
analysis, and progressive notebook refresh may proceed without intermediate
review. It has no efficacy gate and does not stop for poor precision, absent
early digit-9 observations, or rank-zero SubGD. Smoke failures may repair only
implementation defects; they cannot tune a scientific condition.

Phase 6A required its own authorization after review. After that authorization
was given, its implementation tests, smoke runs, 64 fresh production replicas,
analysis, and progressive notebook refresh proceeded without intermediate
review. It had no efficacy or appearance gate and did not enlarge batches or
change conditions in response to rolling plots.

Phase 6B was separately authorized as the Plan 13 closeout study. Its tests,
smoke runs, 64 fresh replicas, analysis, and notebook refresh may proceed
without intermediate review. It has no efficacy or appearance gate, and the
80--90% precision hope cannot alter its sample count, conditions, threshold,
or stopping rule.

Every bounded session refreshes the notebook before starting the next batch.
An unexpected process stop, reboot, driver reset, or user retasking may leave
only the current temporary unit incomplete. A later `--resume` validates and
skips every completed unit, restarts that incomplete unit, and continues the
deterministic ledger. No scientific result depends on a process surviving for
the duration of a phase.

## Progressive Notebook

Create `mnist_experiment/continual_subgd_results.ipynb`. It reads artifacts
only and never trains, downloads data, resumes work, computes large matrices,
or repairs an incomplete unit. Refresh it atomically after every bounded
compute batch.

Its first section always reports planned, complete, failed, and incomplete
units; elapsed compute; integrity; current phase; environment; schedule; and
whether displayed evidence is development or confirmation.

As artifacts become available, render:

- burn-in displacement spectra and held-out explained-energy curves;
- rank, calibration-length, and gradient-proxy comparisons;
- learned, random, head, and Fisher subspace overlaps;
- current and retention NLL and accuracy trajectories;
- digit-9 one-vs-rest and balanced one-vs-rest accuracy trajectories for the
  mixture transport study;
- the optimizer-diagnostic table, per-replica endpoint comparison, and
  optimization-residual diagnostics;
- an explicit invalid-for-SubGD-efficacy banner on original Phase 5 results;
- Phase 5R gate status and repaired trajectories in a separate section, with
  no visual pooling with original Phase 5;
- Phase 6 fixed-$.1$ precision, recall, false-positive, PR-curve, cumulative
  digit-9-count, and realized-rank diagnostics in a separate low-prevalence
  section, with no visual pooling with prior mixture phases;
- Phase 6A ten-positive diagnostics immediately after Phase 6, with a separate
  incomplete-ledger banner and explicit 9.9 expected-count annotation;
- Phase 6B hundred-positive diagnostics immediately after Phase 6A, with a
  separate incomplete-ledger banner and explicit 100.1 expected-count
  annotation;
- running exposure-normalized AUC from the common post-burn-in origin;
- paired effect distributions and confidence intervals;
- innovation, trust, geometry-rate, orthogonal-gain, and subspace-rotation
  trajectories around route events;
- objective residual and compute-cost comparisons;
- dense-versus-incremental covariance diagnostics; and
- a conclusion table naming the best statistical/retention balance, the
  cheapest acceptable method, failure classifications, the rotation promotion
  classification, and Phase 5 transport progress.

Partial ledgers remain visibly incomplete and never unlock fixed-size
inferential language.

## Proposed Implementation Layout

- `mnist_experiment/continual_subgd/config.py`: versioned contracts,
  condition ledgers, and component seeds.
- `mnist_experiment/continual_subgd/geometry.py`: dense and streaming
  adaptation-covariance representations.
- `mnist_experiment/continual_subgd/optimizer.py`: fixed-budget full,
  projected, and preconditioned updates.
- `mnist_experiment/continual_subgd/phase5_optimizer_diagnostic.py`: immutable
  optimizer-by-EWC diagnostic and aggregate classification.
- `mnist_experiment/continual_subgd/coordinate_lbfgs.py`: differentiable
  affine-coordinate model evaluation, geometry square roots, and reset-per-step
  strong-Wolfe L-BFGS.
- `mnist_experiment/continual_subgd/phase5r.py`: repaired burn-in, viability
  gate, treatment ledger, and resumable execution.
- `mnist_experiment/continual_subgd/phase6_low_prevalence.py`: explicit
  low-prevalence schedule, one-step burn-in, seven-condition ledger,
  fixed-reference precision artifacts, analysis, and resumable execution.
- `mnist_experiment/continual_subgd/phase6a_ten_positive.py`: fixed
  18-observation schedule, five-condition exploratory ledger, Phase 6A
  analysis, and resumable execution.
- `mnist_experiment/continual_subgd/phase6b_hundred_positive.py`: fixed
  182-observation schedule, five-condition exploratory closeout ledger, Phase
  6B analysis, and resumable execution.
- `mnist_experiment/continual_subgd/controller.py`: innovation smoothing and
  causal $\alpha_t,\beta_t$ decisions.
- `mnist_experiment/continual_subgd/environments/base.py`: narrow environment
  protocol.
- `mnist_experiment/continual_subgd/environments/rotation.py`: adapter over
  the accepted rotated-MNIST assets.
- `mnist_experiment/continual_subgd/environments/digit9_mixture.py`: Phase 5
  adapter over the accepted Plan 3 stream and retention contract.
- `mnist_experiment/continual_subgd/artifacts.py`: immutable units, ledgers,
  hashes, and resumption.
- `mnist_experiment/continual_subgd/run.py`: bounded phase orchestration.
- `mnist_experiment/continual_subgd/analysis.py`: artifact-only summaries.
- `mnist_experiment/continual_subgd/refresh_notebook.py`: atomic notebook
  refresh.
- `test/unit/continual_subgd/`: deterministic geometry, controller,
  optimizer, artifact, and environment-adapter tests.

Existing mixture and rotation runners may be imported as libraries or have
narrow tested helpers promoted into `src/`; they must not import Plan 13 or
change historical outputs.

## Testing Requirements

Add fast deterministic tests for:

- separation of $C_t$ from the Fisher representation and metadata;
- dense uncentered covariance and the streaming rank-one update;
- zero residual, rank-deficient burn-in, repeated eigenvalues, sign changes,
  and reorthogonalization;
- projector, spectral, head-only, random-basis, and adaptive preconditioner
  matrix-vector products;
- spectrum normalization and analytic parallel/orthogonal gains;
- innovation, half-life, $\alpha_t$, and $\beta_t$ calculations;
- causal ordering: $B_t$ acts on step $t$ and $z_t$ first enters $B_{t+1}$;
- shadow-branch noninterference with learner and random state;
- equality of every condition's pre-geometry objective;
- fixed optimization-step accounting and objective diagnostics;
- equivalence of full-space coordinate L-BFGS and direct model-parameter
  L-BFGS under $A_t=I$;
- coordinate gradients against finite differences for rectangular and
  full-rank $A_t$;
- $A_tA_t^T=M_t$ for head, projector, spectral, random, and relaxed adaptive
  geometries, including rank-deficient and zero-eigenvalue cases;
- exact confinement of accepted displacements to the declared coordinate
  image when $M_t$ is singular;
- reset of L-BFGS history after every outer step and basis change;
- Phase 5R rejection of original burn-in and basis artifacts;
- deterministic Phase 5R gate calculation and treatment refusal after a
  failed gate;
- exact gauge-fixed digit-9 bias contrast and raw-model functional
  equivalence;
- Phase 6 schedule identity, one-step $p=0$ burn-in, and treatment assignment
  before the $p=.01$ batch;
- rank-zero and rank-one static, random, and adaptive bootstrap behavior
  without eigenvector padding;
- fixed-reference precision against a hand-calculated confusion table,
  including the no-predicted-positive convention;
- fixed-$.1$ PR-curve reconstruction from class-conditional threshold rates;
- Phase 6 rejection of all Phase 5 and Phase 5R treatment inputs;
- Phase 6A schedule identity, fixed 18-observation batches, 9.9 expected
  treatment count, five-condition ledger, and rejection of count-based stops;
- Phase 6A rejection of Phase 6 production streams, model states, and
  trajectory artifacts as mutable continuation inputs;
- Phase 6B schedule identity, fixed 182-observation batches, 100.1 expected
  treatment count, five-condition ledger, and a distinct immutable namespace;
- Phase 6B rejection of Phase 6A streams, model states, and trajectory
  artifacts as mutable continuation inputs;
- deterministic paired seeds and random orthonormal bases;
- immutable collision, completion, interruption, and resume behavior;
- a tiny CPU rotation trajectory spanning every condition; and
- notebook rejection of training, downloads, and artifact repair.

The mixture adapter requires its own stream identity, conditional
digit-distribution, fixed-$\pi=.05$ objective, exact resolved-schedule hash,
and no-coordinate-reinterpretation tests. Preserve a regression test for the
historical 100-point schedule and add distinct tests for the Phase 6 11-point
low-prevalence schedule, Phase 6A's 18-observation variant, and Phase 6B's
182-observation variant.

## Failure Classifications

Record, rather than soften, any of the following:

1. Burn-in displacements have no stable low-rank structure beyond random
   baselines.
2. The learned basis reconstructs shadow movement but does not improve held-out
   prediction or retention.
3. A random basis performs as well as the learned basis.
4. Head-only tuning dominates learned low-rank geometry.
5. Static SubGD helps, but online updates add truncation noise or rotate in
   response to minibatch noise rather than regime change.
6. Innovation relaxation merely approaches the full-space control and adds no
   benefit beyond it.
7. Apparent gains are scalar step-size effects, incomplete optimization, or
   preservation of $\theta_K$ rather than new learning.
8. Shadow displacement is useful but too computationally expensive, while the
   gradient proxy fails to reproduce it.
9. Rotation evidence fails to transport to the digit-9-mixture path.
10. A common optimizer fails to solve the retained objective before treatment
    assignment, making downstream geometry comparisons uninterpretable.
11. A tiny burn-in cannot identify a useful SubGD basis before the
    low-prevalence region ends, while structural head or bias restrictions
    remain effective.

These are scientifically useful outcomes. They do not authorize replacing
the failed condition after viewing confirmation results.

The original Phase 5 is classified under item 10. Phase 5R repairs that common
failure before revisiting items 1 through 9; it does not erase the failure.
Phase 6 is designed to expose item 11 rather than protect SubGD from it.

## Provisional Compute Envelope

Historical Plan 12 trajectories took roughly one minute each on the RTX 4070,
but Plan 13 changes the optimizer and adds shadow branches. Phase 0 timing is
therefore authoritative. Before those measurements, budget only the following
wide envelope:

| Phase | Default work | Provisional wall time |
| --- | --- | ---: |
| 0 | Implementation validation, two-environment CPU smoke, CUDA profiling | 2--5 h |
| 1 | Shared full-space geometry cohort and artifact analysis | 1--3 h |
| 2 | 16 replicas $\times$ 2 schedules $\times$ development conditions | 5--10 h |
| 3 | Open-loop calibration, 4-replica probes, and 16-replica controller development | 6--14 h |
| 4 | Independent confirmation; count frozen after Phase 3 | 8--16 h |
| 5 | 16 mixture replicas $\times$ 5 transport conditions | 4--10 h |
| 5R | Reused assets, 16 repaired burn-ins, gate, and 5 coordinate-L-BFGS conditions | 1--3 h |
| 6 | 64 fresh low-prevalence replicas $\times$ 7 conditions, plus smoke and analysis | 2--6 h |
| 6A | 64 fresh ten-positive replicas $\times$ 5 conditions, plus smoke and analysis | 1--3 h |
| 6B | 64 fresh hundred-positive replicas $\times$ 5 conditions, plus smoke and analysis | 5--7 h planned; 0.5 h observed |
| Analysis | Progressive notebook and integrity scans | 1--2 h |
| **Plan 13 total through original Phase 5** | Rotation development/confirmation plus mixture transport | **27--60 h** |
| **Plan 13 including Phase 5R** | Original execution plus post-diagnostic repair | **28--63 h** |
| **Plan 13 including Phase 6** | All completed work plus low-prevalence study | **30--69 h** |
| **Plan 13 including Phase 6A** | All completed work plus ten-positive scaling study | **31--72 h** |
| **Plan 13 including Phase 6B** | All completed work plus hundred-positive closeout | **36--79 h** |

The Phase 5 row is the requested small transport cohort. It does not include
an independent mixture confirmation, which would require a later power and
timing calculation. All estimates remain provisional until Phase 0 measures
the fixed-step optimizer and shadow-branch overhead.

The Phase 5R envelope excludes asset generation but includes full-space shadow
solves. It incorporates the coordinate-L-BFGS CPU tests, an 8.9-second
one-replica CUDA burn-in, and successful CUDA probes of all four non-full-space
factor types. The completed 98-unit production ledger ran from
20:42:38--21:29:24 UTC on the RTX 4070, approximately 46 minutes 46 seconds of
wall-clock time including notebook refreshes and final analysis.

Phase 6 ultimately completed in approximately 35.4 minutes of production wall
time on the available CUDA device. Scaling that observation by the Phase 6A
batch and condition ratios gives a crude production estimate of

$$
35.4\text{ min}\times\frac{18}{8}\times\frac{5}{7}
\approx57\text{ min}.
$$

Budget 1--3 hours for Phase 6A including implementation validation, fresh
assets, smoke runs, notebook refreshes, and analysis. Its one-replica CUDA
smoke replaces this planning estimate before production begins; it cannot
change the scientific design.

The completed production cohort ran from 13:50:13--14:19:02 UTC on October 6,
2026, approximately 28 minutes 49 seconds from the first asset start through
analysis completion. Final notebook rendering completed within approximately
30 minutes of the production invocation. CPU and CUDA smoke validation were
performed separately before production.

Phase 6B preserves Phase 6A's five-condition and 64-replica design while
increasing each batch from 18 to 182 observations. Linear extrapolation from
Phase 6A's completed production runtime gives

$$
28.8\text{ min}\times\frac{182}{18}\approx291\text{ min}=4.85\text{ h}.
$$

The pre-execution budget was 5--7 hours including smoke validation,
progressive notebook rendering, and final analysis. Actual production through
the analysis artifact took approximately 0.5 hours because the larger batches
used the GPU efficiently. The CUDA smoke was an execution-health check and did
not change the frozen design.
