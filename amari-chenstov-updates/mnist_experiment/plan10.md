# Plan 10: Fixed-Composition Pace-Control Hypothesis

> **Hypothesis:** when instantaneous optimal composition cannot be identified
> reliably from one low-data trajectory, fix an interpretable composition
> action $\bar\pi$ and control the rate at which the encountered population
> moves through Fisher distance so that $\bar\pi$ is locally risk-consistent.

**Status:** Draft phased plan. No phase has been executed.  
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
| 0 | Mathematical and comparison contract | Pending | Is pace genuinely distinct from adaptive composition? |
| 1 | Oracle Fisher-speed feasibility map | Pending | Is the inverted target attainable on rotated MNIST? |
| 2 | Finite-EWC one-step fidelity pilot | Pending | Does the surrogate survive the actual optimizer? |
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
