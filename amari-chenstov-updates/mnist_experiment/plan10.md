# Plan 10: Fixed-Composition Pace-Control Hypothesis

> **Hypothesis:** when instantaneous optimal composition cannot be identified
> reliably from one low-data trajectory, fix an interpretable composition
> action $\bar\pi$ and control the rate at which the encountered population
> moves through Fisher distance so that $\bar\pi$ is locally risk-consistent.

**Status:** Stopped at the Phase 1d gate. Multi-start refinement in Phase 1c
did not identify a credible local population-risk path, and the post-gate
direct finite-EWC response surface then rejected the oracle mechanism without
using that path. The 128-replicate expansion and Phase 2 were not run.

**Dependency:** Satisfied by the rejected Plan 9 E9.14 structured-gain gate
and the subsequent user-approved pivot.  
**Scope:** A modular theoretical alternative, not a reinterpretation of the
immutable adaptive-composition experiments.

## Motivation

The applied fixed-batch controller asks a four-observation, single-trajectory
process to estimate a small population movement premium. LFU derivatives,
trend smoothing, decomposed EDR, anchor cancellation, and lag-separated
cross-moments have not produced a calibrated prospective controller. E9.9 also
shows that realized EWC optimization is not generally the scalar affine update
used by the local risk model.

Plan 9 gave one lower-order alternative a final gate: estimate the predictable
matrix gain from accumulated second-order curvature. Its directional solves
were numerically accurate but amplified weakly identified directions by orders
of magnitude in every path group. Plan 10 therefore changes the control problem
rather than adding another estimator of the same weakly identified quantity.

## World, Model, and Learner

The model does not create truth. Let $P_t$ denote the encountered world
distribution and define its model-relative pseudo-true point by

$$
\theta^\star(P_t):=\arg\min_\theta\mathbb E_{P_t}L(X;\theta).
$$

A pace control $a_t$ acts on a curriculum, policy-induced sampling law, or
other controllable transition of the encountered environment,

$$
P_{t+1}=\mathcal T_{a_t}(P_t).
$$

For a smooth induced path, its local parameter representation is

$$
d\Theta_t^\star=a_t b(\Theta_t^\star)dt.
$$

Thus $a_t$ reparameterizes progress along a family of encountered
distributions. It does not choose an arbitrary solution point. This
interpretation is natural for curriculum pacing and policy-dependent RL, but
not for a wholly exogenous environment that the learner cannot pause or
influence.

Keep the states distinct:

- $\Theta_t^\star$ is the population or pseudo-true path induced by the world;
- $\widehat\Theta_t$ is the learner;
- $e_t=\widehat\Theta_t-\Theta_t^\star$ is tracking error;
- $q_t$ is the concentration of retained information weights;
- the auxiliary Fisher process estimates curvature along the realized path.

The factor $\bar\pi$ scales the learner's local response. It does not belong in
the definition of the true path $d\Theta_t^\star$.

## Applied Fixed-Batch Inversion

The source model receives a fixed batch of $m_t$ new observations and has
marginal local risk

$$
R_{B,t}^{\mathrm{marg}}(\pi)
=(1-\pi)^2\left(S_t+q_tD_t^{\mathrm{old}}\right)
+\pi^2\frac{D_t^{\mathrm{new}}}{m_t},
$$

where

$$
S_t=\|d\theta_t-\mu_t\|_{\mathsf M_t}^2.
$$

Its scalar-affine minimizer is

$$
\pi_{B,t}^{\mathrm{marg}}
=\frac{S_t+q_tD_t^{\mathrm{old}}}
{S_t+q_tD_t^{\mathrm{old}}+D_t^{\mathrm{new}}/m_t}.
$$

Fix $\pi_t\equiv\bar\pi\in(0,1)$ and invert this relation. The movement energy
for which $\bar\pi$ is locally optimal is

$$
\boxed{
S_t^\dagger(\bar\pi)
=\frac{\bar\pi}{1-\bar\pi}\frac{D_t^{\mathrm{new}}}{m_t}
-q_tD_t^{\mathrm{old}}.
}
$$

This is a feasibility relation, not a universal proof that $\bar\pi$ is
optimal. If $S_t^\dagger<0$, no nonnegative movement energy can make the chosen
action optimal under the surrogate.

When covariance shapes match, $D_t^{\mathrm{old}}=D_t^{\mathrm{new}}=D_t$,

$$
S_t^\dagger
=D_t\left[\frac{\bar\pi}{m_t(1-\bar\pi)}-q_t\right].
$$

For constant $m$ and fixed $\bar\pi$, the exact composition recursion is

$$
q_{t+1}=(1-\bar\pi)^2q_t+\frac{\bar\pi^2}{m},
$$

with equilibrium

$$
q_\infty=\frac{\bar\pi}{m(2-\bar\pi)}.
$$

The stationary movement target is therefore

$$
\boxed{
S_\infty^\dagger
=\frac{D\bar\pi}{m(1-\bar\pi)(2-\bar\pi)}.
}
$$

If the controllable local direction is $v_t$ and
$d\theta_t=a_tv_t$, the centered model gives

$$
a_t^\dagger
=\sqrt{\frac{[S_t^\dagger]_+}{v_t^T\mathsf M_tv_t}}.
$$

This calculation requires a Fisher quadratic form but no Fisher inverse, LFU,
or rank-three derivative estimate. Nonzero $\mu_t$ changes the equation to
$\|a_tv_t-\mu_t\|_{\mathsf M_t}^2=S_t^\dagger$ and may yield two or no feasible
roots; the centered model must not be imported silently.

## Sampling Procedure

The proposed applied transition is:

1. Before seeing batch $t+1$, observe the deployable state
   $(\widehat\theta_t,q_t,G_t)$ and choose a predictable pace $a_t$.
2. Advance the controllable encountered distribution through
   $P_{t+1}=\mathcal T_{a_t}(P_t)$, inducing a local population displacement
   $d\theta_t=a_tv_t$.
3. Draw exactly $m_t$ observations from $P_{t+1}$.
4. Combine their mean likelihood with the compressed old likelihood using the
   fixed action $\bar\pi$.
5. Accept the EWC optimizer result, update the auxiliary Fisher summary, and
   apply the fixed-$\bar\pi$ concentration recursion.

The sample count and information weights are fixed. The control changes the
transition kernel of the next observations by changing progression through the
encountered distribution family.

## Theoretical Bernoulli Companion

The fixed-total theory model retains $N$ local contributions and independently
assigns each one to the new point with
$M_i\sim\operatorname{Bernoulli}(\bar\pi)$. Its risk is

$$
R_{A,t}(\bar\pi)
=(1-\bar\pi)^2S_t+\frac{\bar\pi}{N}V_t.
$$

Inverting its interior minimizer gives

$$
\boxed{S_{A,t}^\dagger=\frac{V_t}{2N(1-\bar\pi)}.}
$$

Here $N$ and the Bernoulli law remain fixed while pace controls the separation
between the old and new local population points. This model is the cleaner
asymptotic theory; the fixed-batch model is the intended application.

## Candidate Small-Noise Limits

Let $h_m=\varepsilon_m=m^{-1/2}$ and impose the LAN-scale true displacement

$$
d\theta_{k,m}=a_{k,m}b(\theta_{k,m}^\star)h_m.
$$

Under the scalar-affine, oracle-recentered fixed-batch approximation, the
learner increment is

$$
\Delta\widehat\theta_{k,m}
=\bar\pi a_{k,m}b(\theta_{k,m}^\star)h_m
+\bar\pi\mathcal I(\theta_{k,m}^\star)^{-1/2}h_m\xi_{k+1,m}
+r_{k,m}.
$$

For bounded predictable controls converging to $a_t$, the candidate
finite-information interpolation is

$$
\boxed{
d\widehat\Theta_t^{(\varepsilon_m)}
=\bar\pi a_t b(\Theta_t^\star)dt
+\sqrt{\varepsilon_m}\,\bar\pi
\mathcal I(\Theta_t^\star)^{-1/2}dW_t.
}
$$

The corresponding population path satisfies

$$
d\Theta_t^\star=a_tb(\Theta_t^\star)dt.
$$

The Bernoulli companion replaces the fixed-batch noise coefficient
$\bar\pi$ by $\sqrt{\bar\pi}$. These displays are hypotheses awaiting a proper
triangular-array derivation. The finite tracking-error recursion should remain
the principal applied object because it retains random anchor error and the
fast composition-state transient.

## Excluded Reinterpretation

Plan 10 does not define pace by scaling an already computed learner update,

$$
\widehat\theta_{t+1}
=\widehat\theta_t+\alpha_tu_t^{\mathrm{EWC}}.
$$

Under the scalar-affine model this merely changes the effective composition to
$\alpha_t\bar\pi$, requiring the same effective action in the Fisher and
$q_t$ recursions. Outside that model it becomes a different stochastic-
optimization problem. Either case requires a new derivation and risks
reintroducing adaptive $\pi$ under another name.

## Assumptions Requiring Proof or Audit

1. The pace action is predictable and genuinely controls the next encountered
   distribution rather than reacting to the same batch it changes.
2. The induced pseudo-true path has a smooth, locally identifiable vector field
   $b$ in the chosen parameter chart.
3. The controlled displacement remains at the LAN scale and the target
   $S_t^\dagger$ is feasible.
4. The retained summary is covariance calibrated and its Fisher-risk quadratic
   is meaningful in the trainable subspace.
5. The local scalar-affine EWC approximation is accurate enough for the
   inverted risk relation. Plan 9 E9.9 makes this the most serious applied
   concern.
6. The direction $v_t$ is predictable or separately estimable without using
   information from the batch whose distribution is being controlled.
7. The augmented state $(\Theta_t^\star,\widehat\Theta_t,q_t)$ admits the
   claimed controlled weak limit. With fixed $\bar\pi$, the rescaled
   concentration state has a fast transient that may collapse to
   $q_\infty$ on the diffusion clock.

## Falsifiable Consequences

The phases below test, in order:

- whether fixed $\bar\pi$ induces the predicted stationary $q_\infty$;
- whether controlled Fisher displacement tracks $S_t^\dagger$ without using an
  inverse Fisher;
- whether local one-step parameter risk is minimized near $\bar\pi$ when the
  pace condition is met;
- whether the result survives finite-batch EWC optimization rather than only
  the affine oracle;
- whether pace control improves cumulative predictive performance relative to
  fixed-step and unconstrained controls at matched observations and compute.

Negative feasibility or affine-fidelity results should stop the program before
large predictive experiments.

## Early Experimental Notes

- Rotated MNIST can expose a pace controller by changing angular progression,
  but it is only a mechanism test because an experimenter controls the
  curriculum directly.
- A more compelling application would let a policy alter its own future
  sampling distribution while preserving a fixed information-composition
  budget.
- Preserve direct-EMA rank-8-plus-diagonal Fisher summaries and no LFU as the
  incumbent numerical baseline.
- Keep all Plan 9 and earlier artifacts immutable. Plan 10 requires new schema
  names and run roots if it is eventually authorized.
- New source belongs under `rotated_mnist/pace_control/`, new artifacts under
  `cache/mnist_experiment/rotated_mnist/plan10/`, and analysis in an
  artifact-only `rotated_mnist/pace_control_results.ipynb`. Removing that
  package must leave Plans 5--9 executable and interpretable.
- Reuse validated initialization, stream partitions, and population references
  when their estimands are unchanged. Every pace-controlled post-initialization
  learner trajectory is new computation and receives a new immutable identity.
- No phase below is authorized merely by being present. Each check-in controls
  whether the next phase remains scientifically warranted.

## Status

| Phase | Name | Status | Check-in decision |
|---|---|---|---|
| 0 | Mathematical and comparison contract | Complete | Yes: pace changes $P_{t+1}$; fixed composition remains unchanged. |
| 1 | Oracle Fisher-speed feasibility map | Complete; gate rejected | No: the retained local population path does not identify stable Fisher speed. |
| 1b | Finite excess-risk pace map | Complete; gate rejected | The metric is globally coherent, but the retained checkpoints are not credible local population optima. |
| 1c | Multi-start population-reference repair | Complete; gate rejected | No: refinement lowers NLL but does not identify angle-specific local optima. |
| 1d | Direct finite-EWC response surface | Complete; gate rejected | No: the empirical optimum remains below `.05` throughout the pace range. |
| 2 | Finite-EWC one-step fidelity pilot | Not run | The repaired finite-risk prerequisite failed. |
| 3 | Pace scheduler and immutable pipeline | Pending | Is the implementation causal and exactly paired? |
| 4 | Oracle-paced development trajectories | Pending | Does information pacing improve allocation? |
| 5 | Deployable information-source decision | Pending | What may an application know before acting? |
| 6 | Causal pace observer and smoke challenge | Pending | Is online pace estimation stable enough to actuate? |
| 7 | Independent confirmation | Pending | Does the selected policy survive fresh replicas? |
| 8 | Findings and integration review | Pending | What, if anything, belongs in the main theory? |

## Frozen Starting Point

Unless Phase 0 uncovers a mathematical contradiction, use:

- fixed composition $\bar\pi=.05$ as the primary action, with `.025` and `.10`
  used only for predeclared local-risk sensitivity;
- exactly $m=4$ fresh observations per accepted update;
- the direct-EMA rank-8-plus-diagonal Fisher summary, with no LFU;
- the double-lap route $0\to15\to30\to0\to15\to30$ degrees as the primary
  mechanism challenge; and
- environmental multiclass accuracy and NLL as primary predictive outcomes,
  with fixed-angle panel accuracy, ECE, and worst-class recall as context.

The pace action is an angular increment in rotated MNIST. It changes which
distribution generates the next batch. It never rescales an already optimized
parameter update, changes $m$, or changes the EWC action $\bar\pi$.

## Comparison Contract

Pace changes the number and placement of updates needed to traverse a fixed
route, so one comparison cannot answer every question. Preserve two views:

1. **Matched route and observations.** After an oracle-paced route determines
   its update count, compare it with a uniform schedule using the same route,
   number of batches, observations, optimizer budget, and initialization. This
   isolates allocation of observations along the route.
2. **Matched online budget.** At fixed numbers of observed samples, report
   tracking quality and cumulative angular progress. This exposes controllers
   that obtain good accuracy merely by refusing to advance.

Always plot outcomes against both cumulative observations and cumulative
angular distance. Report route-completion observations, learner time, score
gradient count, optimizer evaluations, pace-bound occupancy, and Fisher-target
error. A controller that does not finish within the predeclared step cap is a
failure, not a high-accuracy trajectory.

## Phase 0: Mathematical and Comparison Contract

### Goal

Close the discrete-time mathematics before building a scheduler.

### Work

1. Derive the inverted fixed-batch target from the exact frozen risk surrogate
   and verify its first- and second-order optimality conditions symbolically and
   numerically.
2. Fix indexing and predictability: $a_t$ is chosen from $\mathcal F_t$, changes
   $P_{t+1}$, and cannot use scores from batch $t+1$.
3. Prove the fixed-$\bar\pi$ $q_t$ recursion and stationary target, including
   the transient from the high-quality initialization.
4. State the centered pace solution and derive the full quadratic feasibility
   condition for nonzero $\mu_t$. Keep the centered case primary unless data
   later require the additional trend term.
5. Give the candidate triangular-array and small-noise limits a proper
   remainder and tightness checklist. Do not promote them to
   `mathematical_overview.ipynb` yet.
6. Freeze the two comparison views above, route-completion semantics, primary
   fixed action, sensitivity actions, and stop conditions.
7. Add deterministic unit tests for the inversion, $q_t$ transient and
   equilibrium, infeasible negative targets, pace roots, and the prohibition on
   post-update rescaling.

### Gate And Check-In

Proceed only if pace acts on the next distribution, the fixed action appears
unchanged in both EWC and $q_t$, and the comparison cannot reward stalling. The
check-in reviews the mathematical contract and may close Plan 10 without any
learner computation.

### Execution Record

**Complete.** The detachable contract is implemented in
`rotated_mnist/pace_control/theory.py` and summarized in
`rotated_mnist/pace_control/CONTRACT.md`. The exact risk minimizer, inverse
target, closed-form $q_t$ transient and equilibrium, centered and noncentered
pace roots, and causal action target are covered by deterministic tests.

The gate passes. Pace is defined only on the next encountered distribution;
$\bar\pi$ enters the EWC objective and concentration recursion unchanged; and
the paired comparison contract counts non-completion as failure. The
small-noise displays remain candidate limits subject to the seven-item proof
checklist in the contract rather than being promoted prematurely.

## Phase 1: Oracle Fisher-Speed Feasibility Map

### Goal

Determine whether rotated MNIST contains an attainable pace-control problem
before fitting new continual learners.

### Work

1. Audit existing Plan 6 population parameters and high-sample Fisher
   references for sufficient angular resolution. Reuse them only where
   interpolation can be validated; compute new immutable reference points when
   the old grid is too coarse.
2. Along both directions of the double-lap route, estimate the local Fisher
   speed

   $$
   J(\varphi)=
   \left(\frac{d\theta^\star}{d\varphi}\right)^T
   \mathsf M(\varphi)
   \left(\frac{d\theta^\star}{d\varphi}\right),
   $$

   and verify the finite-step relation
   $\|\theta^\star(\varphi+\delta)-\theta^\star(\varphi)\|_{\mathsf M}^2
   \approx J(\varphi)\delta^2$ over candidate increments.
3. Propagate the exact $q_t$ transient and solve
   $\delta_t^\dagger=\sqrt{[S_t^\dagger]_+/J(\varphi_t)}$ without a Fisher
   inverse. Record infeasible targets and apply only predeclared physical pace
   bounds.
4. Produce an immutable artifact and a lightweight feasibility notebook showing
   $J(\varphi)$, $S_t^\dagger$, proposed increments, local approximation error,
   bound occupancy, and implied route-completion observations.

### Gate And Check-In

Proceed only if the target is nonnegative after the initialization transient,
finite-step Fisher energy is locally monotone in pace, the quadratic
approximation is useful at the proposed increments, and the route finishes
without spending most updates at a pace bound. Failure means this rotated-MNIST
mechanism cannot test Plan 10; it does not justify tuning bounds until it passes.

### Execution Record

**Complete; gate rejected.** The artifact-only audit is implemented under
`rotated_mnist/pace_control/` and recorded immutably at
`cache/mnist_experiment/rotated_mnist/plan10/phase1/rotated_mnist_plan10_phase1_fisher_speed_feasibility__replica-0001__34f1eda2cc232ef6`.
The companion `rotated_mnist/pace_control_feasibility.ipynb` loads only that
completed artifact.

The 79-angle Plan 6 source is dense enough to contain the complete uniform
$.75^\circ$ grid, but its fitted Euclidean population path is not locally
smooth enough for the pace estimand. On the 41-point uniform map:

- 9 debiased Fisher-speed estimates are exactly zero because the neighboring
  retained reference parameter vectors are identical;
- only `0.731` of the larger finite secants have energy at least as large as
  their nested smaller secants, below the frozen `.95` monotonicity gate;
- median relative error of $J(\varphi)\delta^2$ is `0.506`, just beyond the
  frozen `.50` gate;
- the implied 86-update route is on a predeclared pace bound for `0.965` of
  updates, including 32 infeasible zero-speed steps; and
- route completion itself succeeds in 344 observations only because clipping
  supplies nearly every action.

The failure is not caused by Fisher inversion: all calculations use rank-16
quadratic forms. Paired local-MLE clouds debias finite-reference movement, and
no parameter interpolation is used. The gate therefore rejects this retained
rotated-MNIST population path as a basis for local pace control. It does not
falsify the algebraic inversion.

## Phase 1b: Finite Excess-Risk Pace Map

### Goal

Replace the unstable derivative of approximate neural-network optima with a
finite, functionally meaningful population-risk quantity while preserving the
same local Fisher-risk estimand to second order.

### Frozen Estimand And Design

For retained reference fits at angles $\varphi$ and $\varphi+\delta$, define

$$
S_{\mathrm{NLL}}(\varphi,\delta)
=2\left[
\mathcal L_{\varphi+\delta}(\theta^\star(\varphi))
-\mathcal L_{\varphi+\delta}(\theta^\star(\varphi+\delta))
\right].
$$

At an interior optimum under the information identity,
$S_{\mathrm{NLL}}=\|d\theta\|_{\mathcal I}^2+O(\|d\theta\|^3)$. Estimate both
losses on the same 8,000-observation test panel excluded from the Plan 6
reference checkpoint selection. This common random panel controls comparison
noise and avoids validation-selection optimism.

Use every retained angle on the uniform $.75^\circ$ grid. At each route state,
evaluate all grid-aligned candidate endpoints remaining in the current leg and
choose the smallest increment whose finite energy attains its corresponding
$S_t^\dagger$. If none attains the target, advance to the leg endpoint and
record the miss. Do not interpolate parameter vectors, smooth energies, impose
an outcome-selected pace cap, or clamp negative excess risk.

Before execution, freeze these gates:

1. at least `.80` of directed finite energies are nonnegative;
2. at least `.80` of nested candidate increments are energy-monotone;
3. at least `.80` of route updates attain their movement target;
4. median relative target error among selected updates is at most `.50`;
5. every target is nonnegative and the full double-lap route completes within
   400 updates; and
6. the finite energy has positive rank correlation with the rank-16 Fisher
   quadratic where that quadratic is nonzero.

### Gate And Check-In

Proceed to Phase 2 only if all six checks pass. A failure means neither the
derivative nor this finite-risk measurement can support pace control on the
retained rotated-MNIST reference path. Passing supports a finite-step applied
controller, not an infinitesimal Fisher-speed theorem.

### Execution Record

**Complete; gate rejected.** The immutable artifact is
`cache/mnist_experiment/rotated_mnist/plan10/phase1b/rotated_mnist_plan10_phase1b_finite_excess_risk__replica-0001__6c442a2a4d3dad03`.
It cross-evaluates all 41 retained uniform-grid reference states on 8,000
paired test observations excluded from Plan 6 checkpoint selection, producing
1,640 directed finite-risk comparisons in 15.75 seconds on CUDA.

The finite measurement improves global coherence but not the required local
identification:

- global finite excess risk and debiased Fisher energy have Spearman
  correlation `0.699`, and `0.853` of nested finite energies are monotone;
- local correlation for increments at most $3^\circ$ is only `0.088`;
- `0.770` of directed excess risks are nonnegative, below the frozen `.80`
  gate, and the local nonnegative fraction is only `0.617`;
- only 2 of 41 nominal target-angle reference states minimize held-out NLL
  among the retained checkpoints at their own angle; median regret to the best
  retained state is `0.0149` NLL; and
- the discrete route uses 12 large moves with median pace $6.375^\circ$, but
  only `0.583` attain their movement target. Its median selected-target error
  is nevertheless a reasonable `0.121`.

The negative risks are not clamped, and no smoothing or parameter
interpolation is used. These results support finite excess NLL as a more robust
global movement diagnostic, while showing that the inherited continuation
checkpoints are not sufficiently optimized local estimates of
$\theta^\star(\varphi)$ for pace control. Repair now requires a new reference-
estimation protocol, not threshold or pace-bound tuning.

## Phase 1c: Multi-Start Population-Reference Repair

### Goal

Test whether the Phase 1b failure belongs to its finite-risk measurement or to
the inherited early-stopped continuation checkpoints. Preserve the finite-risk
estimand and all earlier artifacts.

### Frozen Reference Protocol

1. Use the same 41-angle uniform $.75^\circ$ grid, 10,000-observation Plan 6
   fitting sample, 2,000-observation checkpoint-selection panel, and disjoint
   8,000-observation held-out panel.
2. At each target angle, cross-evaluate the retained states on the selection
   panel and choose the three lowest-NLL states with distinct state hashes.
   These are starting points only; selection-panel outcomes never enter the
   held-out audit.
3. Independently refine all three starts on the target-angle fitting sample for
   exactly 12 Adam epochs at the inherited learning rate. Run every epoch even
   when validation stops improving, retain each start's best selection-panel
   epoch including epoch zero, then select the candidate with the lowest
   selection-panel NLL.
4. Pair shuffle order across the three starts at an angle, reset model and
   optimizer state for every fit, and record all start identities, seeds,
   histories, selected epochs, and parameter hashes.
5. Cross-evaluate the 41 selected states only after selection is complete.
   Construct finite excess risks on held-out observations without clamping,
   smoothing, parameter interpolation, or Fisher recomputation.

Before execution, freeze the inexpensive screen gates:

- at least `.80` of selected angle-specific references are best among the 41
  repaired states on their own held-out angle;
- median held-out regret to the best repaired state is at most `.002` NLL;
- at least `.80` of all directed and `.80` of at-most-$3^\circ$ finite excess
  risks are nonnegative; and
- at least `.80` of nested finite energies are monotone.

### Conditional Expensive Work

Only after the screen passes, recompute rank-8 and rank-16 score Fishers and 64
paired local-MLE clouds of size 2,048 around the selected states. Re-run the
Phase 1b route with the same target-attainment and target-error gates. Only a
passing rebuilt route may unlock Phase 2.

### Gate And Check-In

Failure of the inexpensive screen stops before Fisher and covariance work.
Failure after rebuilding those quantities stops before Phase 2. Do not rescue
the result by adding starts, epochs, smoothing, or relaxed gates after observing
the outcome; those would be new experiments requiring review.

### Execution Record

**Complete; inexpensive gate rejected.** The immutable artifact is
`cache/mnist_experiment/rotated_mnist/plan10/phase1c/screen/rotated_mnist_plan10_phase1c_reference_repair_screen__replica-0001__26787e9eeaccf60d`.
It contains 123 complete 12-epoch refinements, all selection histories, the 41
selected states, and a disjoint 8,000-observation held-out cross-evaluation.
The screen finished in 197.8 seconds on CUDA.

Refinement materially lowered each state's own-angle loss: median held-out NLL
improvement over the inherited references is `0.0221`, and no selected fit used
epoch zero. That optimization gain does not recover the needed local estimand:

- only 4 of 41 (`0.0976`) selected states minimize held-out NLL at their own
  angle, far below the frozen `.80` gate;
- median held-out regret is `0.00353` NLL, above the frozen `.002` limit, with
  maximum regret `0.0238`;
- `0.901` of all directed finite energies are nonnegative and `0.917` of
  nested energies are monotone, so their global structure improves; but
- for increments at most $3^\circ$, only `0.614` of finite energies are
  nonnegative, essentially unchanged from Phase 1b's `0.617` and below the
  frozen `.80` local gate.

The selected held-out minimizer is typically offset from the nominal angle and
only 16 selected states minimize any of the 41 held-out angle losses. The
failure therefore survives substantially better optimization: rotated MNIST's
loss surface does not identify the required fine angle-indexed population path
at this model and data resolution. The conditional Fisher and local-MLE rebuild
was not run, preserving the predeclared compute gate.

## Amendment: Phase 1d Direct Finite-EWC Response Surface

### Rationale And Scope

This amendment was written after observing the Phase 1--1c failures. It is a
new, explicitly post-gate experiment rather than a reinterpretation of those
results. The earlier phases asked whether retained neural-network references
identify a sufficiently smooth map from angular pace to local Fisher movement.
They do not. Phase 1d asks the narrower decision-level question that remains:

> As the next-distribution angular increment increases, does the empirical
> finite-EWC risk-minimizing composition move through
> $\bar\pi=.05$ at an attainable pace?

The screen evaluates actual finite-EWC updates and their next-distribution
predictive NLL. It does not estimate $\theta^\star(\varphi)$, differentiate a
reference path, fit a Fisher-speed map, or use a reference Fisher to define
parameter error. Predictive NLL is used here to test the finite learner's
decision consequence directly; it does not replace the Fisher-risk estimand in
the inverted theory or establish the formula for $S_t^\dagger$.

### Empirical Object

Let

$$
\mathcal A_j=
(\widehat\theta_j,\widehat{\mathcal I}_j,q_j,\varphi_j,h_j)
$$

be a frozen deployed anchor state, including its route history $h_j$ and next
travel direction $\sigma_j\in\{-1,+1\}$. For candidate pace magnitude
$\delta\geq0$, paired batch replicate $r$, and diagnostic composition $\pi$,
define

$$
\widehat\theta^+_{j,r}(\delta,\pi)
=\operatorname{EWC}\!\left(
\mathcal A_j,
B_{j,r}^{(m=4)}(\varphi_j+\sigma_j\delta);
\pi
\right).
$$

Every value of $\pi$ receives the same four observations, initialization,
Fisher summary, optimizer budget, and random state within a paired
$(j,r,\delta)$ block. The population decision risk is

$$
\mathcal R_j(\delta,\pi)
=\mathbb E_{B^{(m=4)}}\mathbb E_{Z\sim
P_{\varphi_j+\sigma_j\delta}}
L\!\left(Z;\widehat\theta^+_j(\delta,\pi)\right),
$$

estimated on a common high-sample panel excluded from anchor selection and
all four-observation update batches. Define the diagnostic discrete optimum
and regret of the proposed fixed action by

$$
\widehat\pi_j^\star(\delta)
\in\arg\min_{\pi\in\Pi}\widehat{\mathcal R}_j(\delta,\pi),
\qquad
\Delta_j(\delta)
=\max_{\pi\in\Pi}
\left[
\mathcal R_j(\delta,.05)-\mathcal R_j(\delta,\pi)
\right].
$$

Offline variation of $\pi$ is a diagnostic intervention only. A successful
screen would still deploy `.05` unchanged; it would not authorize an adaptive
$\pi$ controller.

### Frozen Screen Design

1. Freeze the `linear/fixed_pi005` states from the immutable Plan 5 double-lap
   development run at accepted steps 20, 40, 60, and 100. These are,
   respectively, the first $15^\circ$ ascent with next direction positive, the
   $30^\circ$ reversal with next direction negative, the $15^\circ$ return leg
   with next direction negative, and the matched second-ascent $15^\circ$
   revisitation with next direction positive. The source run is
   `rotated_mnist_phase5_double_lap_development__replica-0001__e8d49d4599638493`.
   Record the model, Fisher, concentration, and source hashes before fitting;
   no predictive outcome selects an anchor.
2. Use pace magnitudes
   $\delta\in\{0,.375,.75,1.125,1.5,3.0\}$ degrees. Zero is a no-movement
   diagnostic. Only `.375` through `1.5` degrees belong to the original
   physical action range and may satisfy the viability gate. The `3.0` degree
   point characterizes a missed crossing but can never rescue the gate.
3. Use the diagnostic grid
   $\Pi=\{.0125,.025,.0375,.05,.075,.10,.15\}$. This brackets the frozen
   action and its original `.025` and `.10` sensitivities without searching a
   dense outcome-adapted grid.
4. Draw exactly $m=4$ fresh observations for every update replicate. Pair each
   batch across all values of $\pi$ and keep batches independent across
   replicates. Name and record separate anchor, batch, optimizer, and
   evaluation seeds.
5. Reset model and optimizer state for every fit. Use the historical EWC
   objective and optimizer budget unchanged except for the diagnostic value of
   $\pi$. Do not warm-start neighboring response-surface cells.
6. Make next-distribution mean NLL the sole gate metric. Report accuracy,
   old-angle NLL, parameter displacement, optimizer convergence, and compute
   only as diagnostics. No downstream trajectory metric may select an anchor,
   pace, action, or threshold.
7. Use a fixed high-sample evaluation panel that was not used to fit or select
   the anchor states. If no existing panel satisfies that contract, create and
   hash a deterministic panel in the new Phase 1d namespace before any EWC
   fit. Reuse the same endpoint panel across $\pi$ values.
8. Do not use repaired reference states, local-MLE clouds, Fisher inversion,
   parameter interpolation, or smoothing across angles. The direct surface is
   the artifact; a fitted curve may be shown only as a visual aid.

### Fail-Fast Execution

Run a tiny schema smoke first. Then run a coarse screen with 16 paired batch
replicates at one predeclared anchor in each direction. Proceed to the full
four-anchor pilot only if each direction has at least one in-bound pace at
which `.05` is not already demonstrably worse than the best diagnostic action.
A one-sided paired simultaneous interval whose lower endpoint exceeds `.002`
NLL at every in-bound pace is an immediate rejection for that direction.

The full pilot uses 32 new paired batch replicates per
$(\text{anchor},\delta)$ block; coarse-screen replicates are excluded from its
confirmatory summaries. Bootstrap whole four-observation batch replicates and
form simultaneous intervals over $\Pi$ within an anchor and pace. A
deterministic expansion to 128 new replicates is permitted only when all
qualitative gates below pass but the `.002`-NLL equivalence decision remains
uncertain. The expansion rule, seeds, and work-unit identities must be in the
configuration before the coarse screen begins.

The `.002` threshold is inherited as the predeclared practically meaningful
local NLL scale from Phase 1c. It must not be changed after inspecting this
response surface.

### Gate And Interpretation

Phase 1d passes only if all of the following hold:

1. Every anchor has at least one pace in the original `.375`--`1.5` degree
   range for which the simultaneous 95% upper confidence bound for
   $\Delta_j(\delta)$ is at most `.002` NLL.
2. The response surface contains a resolved low-to-high composition shift as
   pace increases in both travel directions. A surface on which every tested
   $\pi$ is practically indistinguishable does not validate the inversion.
3. The empirical minimizing action is nondecreasing with pace in at least
   `.80` of adjacent in-bound comparisons after ties within `.002` NLL are
   treated as ties rather than ordered evidence.
4. At least three of the four anchor-specific `.05`-competitive paces are
   strictly inside the physical pace interval. A crossing found only at zero,
   `3.0` degrees, or a physical bound is a failure, not a clipping policy.
5. The result does not collapse by travel direction or revisitation history,
   and numerical or optimizer failures affect fewer than `.01` of fits.

Failure means the current inverted optimal-$\pi$ pace strategy is not viable
for this Rotated-MNIST learner, even as an oracle mechanism, and Plans 2--8
remain closed. A mixed, flat, boundary-only, or direction-specific result is
also a rejection; it is not a prompt to tune the action or pace grids.

Passing establishes only the existence of an empirical finite-EWC crossing.
It does not rehabilitate the rejected Fisher-speed map, validate
$S_t^\dagger$, identify a causal pace estimator, or demonstrate trajectory
benefit. A passing artifact therefore requires a new check-in to decide
whether to replace the blocked reference-based Phase 2 with an explicitly
empirical calibration problem. It does not automatically authorize Phase 3.

### Artifact Contract

Any implementation belongs under `rotated_mnist/pace_control/` and writes only
to a new immutable `cache/mnist_experiment/rotated_mnist/plan10/phase1d/`
namespace. Store the resolved configuration, source anchor hashes, all seeds,
per-fit optimizer diagnostics, per-replicate NLLs, paired contrasts,
simultaneous intervals, compute counts, completion status, and the predeclared
gate decision. The results notebook remains artifact-only. No Phase 1--1c
artifact or historical learner run may be modified.

### Execution Record

**Complete; gate rejected.** The implementation, frozen stage configurations,
paired-bootstrap analysis, resumable work units, and immutable artifact writer
live under `rotated_mnist/pace_control/`. Source reconstruction checks every
stored parameter hash and pre-update Fisher trace through the latest requested
anchor before running a disposable fit.

The schema smoke completed 168 fits in 49.4 seconds with no failures. The
16-replicate coarse screen then completed 1,344 fits in 356.7 seconds. It
formally opened the full pilot because the simultaneous lower regret bounds
did not reject `.05` at every pace in either direction, even though `.05` was
not point-competitive at either anchor. The completed coarse artifact is
`rotated_mnist_plan10_phase1d_coarse_response_surface__replica-0001__8eaf26f3187d51b9`.

The decisive full artifact is
`rotated_mnist_plan10_phase1d_full_response_surface__replica-0001__2917fd3e4ea0fb0b`.
It ran 5,376 fresh finite-EWC fits at four anchors and 32 paired batch
replicates per anchor and pace. All fits succeeded, and the run completed in
1,390.5 seconds on CUDA. The direct response surface rejects the mechanism:

- among the 16 in-bound anchor/pace cells, the empirical minimizing action was
  `.0125` in 10 cells, `.025` in 5, and `.0375` in 1; it was never `.05` or
  larger;
- `.05` was not point-competitive within `.002` NLL at any anchor or pace. Its
  in-bound point regret ranged from `0.0113` to `0.1447` NLL, with median
  `0.0612`;
- no anchor had an in-bound pace whose simultaneous 95% upper regret bound was
  at most `.002`, and therefore no interior crossing satisfied the equivalence
  gate;
- the contrast
  $\mathcal R(.025)-\mathcal R(.075)$ remained negative over the complete
  in-bound grid. On the negative direction its simultaneous upper bounds were
  below zero at every pace; on the positive direction `.075` was never
  significantly preferred. Thus neither direction exhibited the required
  low-to-high composition crossing; and
- the tie-aware empirical minimizer was nondecreasing in `0.917` of adjacent
  comparisons, but this weak ordering cannot rescue a surface that remains
  entirely below `.05`. Even the diagnostic `3.0`-degree points selected only
  `.0125` or `.025`.

The frozen gate fails fixed-action competitiveness, a resolved crossing in
both directions, and sufficient interior crossings. Numerical reliability and
the `.80` monotonicity check pass. Because the failure is qualitative rather
than equivalence uncertainty, the predeclared 128-replicate expansion is not
authorized. This result bypasses the disputed population-reference path and
shows that the current inverted optimal-$\pi$ pace strategy is not viable for
this Rotated-MNIST learner even as an oracle mechanism. It does not falsify the
algebraic inversion in a model where the scalar-affine assumptions hold.

## Phase 2: Finite-EWC One-Step Fidelity Pilot

### Goal

Test the most vulnerable assumption: whether the inverted scalar-affine risk
surrogate predicts actual low-batch EWC behavior.

### Work

1. Select representative low-, medium-, and high-$J(\varphi)$ anchors plus both
   travel directions without looking at downstream accuracy.
2. At each anchor, use the oracle pace from Phase 1 and paired $m=4$ batches to
   run fresh one-step EWC fits at $\pi\in\{.025,.05,.10\}$. Reset model,
   optimizer, Fisher, and random state for each paired fit.
3. Estimate parameter risk to the next high-sample population point with its
   reference Fisher. Report the empirical minimizing action, the risk gap at
   $\bar\pi=.05$, optimizer fidelity, and dependence on local curvature and
   direction.
4. Start with a cheap smoke and a modest Monte Carlo pilot. Predeclare a
   resumable expansion only if uncertainty overlaps a practically meaningful
   neighborhood of `.05`.

### Gate And Check-In

Continue if `.05` is competitive with the paired empirical optimum across the
representative anchors and the affine prediction has useful rank correlation
with realized risk. A clear optimum elsewhere or direction-dependent collapse
rejects the current inversion before trajectory experiments.

### Execution Record

**Not run.** Phases 1 and 1b rejected the inherited population-geometry
prerequisite. Phase 1c then improved the references' held-out NLL but failed
the frozen local-identification screen. Phase 1d subsequently tested fresh EWC
fits over pace and composition directly and found no empirical crossing at
`.05`, so this phase remains closed for a second, reference-independent reason.
No Phase 2 learner artifact was created and no historical artifact was
modified.

## Phase 3: Pace Scheduler and Immutable Pipeline

### Goal

Build a detachable, causal runner after the mathematical and optimizer gates
have passed.

### Work

1. Implement a monotone route-progress state with direction, remaining angular
   distance, pace bounds, completion tolerance, and maximum accepted updates.
2. Separate `choose_pace(state)` from `observe_batch(...)`. Assert that the
   chosen pace and fixed $\bar\pi$ are recorded before the next batch is
   materialized.
3. Generate paired pace-controlled and uniform schedules before learner
   training. The uniform comparator inherits the realized pace schedule's
   update count but not its nonuniform placement.
4. Add immutable configuration, manifests, source hashes, checkpoints,
   cooperative stop handling, and `--resume` work units using Plan 9's command
   center conventions.
5. Store scalar trajectories needed by notebooks: angle, angular increment,
   $q_t$, target and realized Fisher energy, bound status, predictive metrics,
   observations, and compute costs.
6. Run deterministic synthetic scheduler tests and a tiny CPU trajectory smoke.

### Gate And Check-In

The smoke must establish causality, exact fixed-$\bar\pi$ recursion, paired
streams, route completion, immutable resumption, and no imports from Plans 5--9
into their historical runners.

## Phase 4: Oracle-Paced Development Trajectories

### Goal

Ask whether allocating a fixed number of low-data updates according to Fisher
speed has any predictive value when pace geometry is known accurately.

### Conditions

1. Oracle-paced EWC with fixed $\bar\pi=.05$.
2. Uniform-paced EWC with fixed `.05`, matched to condition 1's route and
   observation count.
3. A predeclared fixed-increment EWC baseline using the established Plan 5
   schedule, shown separately when exact matching is impossible.
4. Current-batch-only learning on the matched oracle-generated schedule as a
   contextual control, not as a pace-policy competitor.

### Work And Metrics

Run a paired development set first. Plot mean trajectories with uncertainty
against observations and angular progress, keeping each figure to at most four
conditions. Emphasize environmental multiclass accuracy, mean NLL, fixed-panel
retention, Fisher-target error, completion observations, and learner compute.
Diagnose whether gains arise from allocating extra updates near high curvature
rather than merely moving more slowly everywhere.

### Gate And Check-In

Oracle pacing must either improve predictive risk at matched observations,
reduce observations needed for matched route quality, or materially reduce
Fisher-target error without stalling. If none occur, no deployable pace
estimator is warranted.

## Phase 5: Deployable Information-Source Decision

### Goal

Choose what information a real application may use to estimate Fisher speed.
This is a scientific decision phase, not an implementation formality.

### Candidate Contracts

1. **Calibrated curriculum geometry:** an offline profile maps a known task
   coordinate to Fisher speed. This is strongest for engineered curricula but
   requires calibration data.
2. **Causal secant observer:** past accepted transitions estimate Fisher speed.
   This is cheapest but risks recreating Plan 9's anchor-contaminated movement
   problem.
3. **Candidate-distribution probe:** a small explicitly charged probe estimates
   the next distribution's score geometry before commitment. This broadens the
   filtration and spends data and compute, so probes must count in every budget.

### Gate And Check-In

Use Phase 4 evidence to decide whether deployment value justifies one of these
contracts. Do not quietly call oracle geometry deployable. The selected contract
must state its data, memory, latency, and predictability costs and explain why it
does not estimate adaptive $\pi$ under another name.

## Phase 6: Causal Pace Observer and Smoke Challenge

### Goal

Implement only the information contract selected in Phase 5 and establish that
it can control pace without oracle leakage.

### Work

1. Add the smallest estimator consistent with the selected contract; avoid
   directional Fisher solves and treatment-specific rescue tuning.
2. Record the oracle pace only in sealed retrospective scoring.
3. Compare estimated versus oracle Fisher speed, pace, target energy, completion
   behavior, and cumulative data/compute cost on paired smoke trajectories.
4. Add sensitivity only for parameters with an interpretable physical or
   statistical meaning. Stop if actuation is primarily clipping or if lag makes
   route completion unstable.

### Gate And Check-In

Proceed only if the causal observer tracks the oracle well enough to preserve
the Phase 4 mechanism without excessive bound occupancy, probe cost, or lag.

## Phase 7: Independent Confirmation

### Goal

Estimate population-level effects only after the mechanism and deployable
observer have passed their respective gates.

### Design

Run fresh independent initializations and streams for the selected causal pace
policy, its oracle ceiling, matched uniform fixed-`.05` EWC, established
fixed-increment EWC, and the necessary contextual control. Use paired seeds
within replicas and immutable replica additions so `--resume` can increase
power without changing earlier evidence.

Predeclare the primary contrast and minimum practically important effect after
the Phase 6 check-in. Report paired confidence regions, normalized
environmental-accuracy and mean-NLL AUCs over observed samples, explicit
compute-cost curves, route-completion distributions, and sensitivity across
route legs. Keep mechanism, deployment, and predictive claims visibly
separate.

### Gate And Check-In

Classify the result as supported, mixed, null, or harmful. A null predictive
result may still support Fisher-distance pacing as a diagnostic; a harmful or
stalling result closes the controller.

## Phase 8: Findings and Integration Review

### Goal

Produce a concise human-readable record without rewriting prior negative
results.

### Work

1. Finalize `pace_control_results.ipynb` with simple paired plots, uncertainty,
   equations for the target and pace, and direct links to immutable evidence.
2. Record costs and limitations, especially controllability of the world,
   calibration requirements, affine-EWC fidelity, and the distinction between
   oracle and deployable pacing.
3. Update this plan with execution records and decisions.
4. Propose edits to `mathematical_overview.ipynb` only after user review. Keep
   numerical estimation details in an appendix and the elegant inverted control
   model in the main body if the evidence warrants integration.
5. Leave Plan 9's EDR and structured-gain findings intact. Plan 10 is a new
   control problem, not a retroactive repair.
