# Scientific Ledger 9: Markov Movement Estimation

Plan 9 follows the instantaneous-oracle work in [plan6.md](plan6.md), the
coefficient audits in [plan7.md](plan7.md), and the decomposed-controller
rechallenge in [plan8.md](plan8.md). It is an optional rotated-MNIST research
extension, not a revision of those completed experiments.

The ledger asks one high-level question:

> Can controlled Markov dynamics and a smooth population drift field turn
> accepted learner updates into anchor-cancelling observations that provide a
> calibrated, single-trajectory estimate of the population movement premium?

Plan 7 Phase 5 found that the covariance-only recommendation remains well
calibrated after substituting each Plan 8 path's endogenous $q_t$, while the
online movement energy is inflated by several orders of magnitude. Replacing
only the movement numerator with its population value reduced recommendation
MAE to approximately `.003`; replacing only the covariance scale did not.
Plan 9 therefore focuses on identifying population movement. It does not
reopen the $q_t$ recursion, LFU, Fisher representation, or fixed-policy
baseline.

This document is a **living scientific ledger**. Its charter and frozen source
contracts should remain stable. New studies may be appended when earlier
evidence raises a specific question. Completed study records must not be
rewritten to make later results appear predeclared.

## High-Level Goal

Develop and test an online-compatible estimator for

$$
S_t=(d_t^\star)^T\mathcal I_t d_t^\star,
\qquad
\rho_t^\star=\frac{m_tS_t}{D_t^{\mathrm{new}}},
$$

where $d_t^\star=\theta_{t+1}^\star-\theta_t^\star$ is population movement
and $D_t^{\mathrm{new}}=\operatorname{tr}(\mathcal I_tK_{t+1})$ is the new
estimator's Fisher-weighted covariance shape. The estimator should use one
continual-learning trajectory, remain compatible with rank-plus-diagonal
Fisher summaries, and avoid empirical Fisher inversion.

Success in this ledger means progressively stronger evidence:

1. an algebraically valid anchor cancellation;
2. retrospective calibration under population reference quantities;
3. prospective calibration using only deployable online quantities;
4. useful closed-loop behavior in a paired development experiment; and
5. independent predictive confirmation.

Evidence at one level must not be described as evidence at a later level.

## Ledger Conventions

### Study records

Each study receives a stable identifier `E9.n` and records:

- the question and hypothesis fixed before execution;
- evidence level and source artifact IDs;
- estimand, estimator, timing, and assumptions;
- configuration and implementation changes;
- primary metrics and decision rule;
- immutable output artifact and source commit;
- observation, interpretation, and decision as separate entries; and
- any follow-up opened by the result.

Allowed statuses are:

- **Proposed:** specified but not implemented;
- **Ready:** implementation and smoke verification passed;
- **Running:** an immutable computation is active or resumable;
- **Supported:** the predeclared mechanistic or predictive gate passed;
- **Rejected:** the predeclared gate failed clearly;
- **Inconclusive:** the study completed but cannot distinguish its hypotheses;
- **Superseded:** a later study replaces an estimator without changing the
  historical result; and
- **Integrated:** reviewed evidence has been promoted outside this optional
  module.

Amendments are appended below the original study record with a date and
rationale. They never mutate completed artifacts or silently change an
estimand.

### Evidence levels

Use the following labels in every table and notebook:

| Level | Meaning |
|---|---|
| Algebraic | Identity or synthetic-model verification only |
| Oracle retrospective | Completed learner path evaluated with unavailable population quantities |
| Online observer | Causal deployable estimator recorded without controlling the learner |
| Closed-loop development | Estimator controls one paired development trajectory |
| Independent confirmation | Fresh initializations and trajectories support population-level comparison |
| Integration | Reviewed evidence promoted beyond the optional module |

### Modularity boundary

Until an integration decision is approved:

- keep Plan 9 source under
  `mnist_experiment/rotated_mnist/markov_movement/`;
- keep small immutable configurations beside that package;
- write generated artifacts under
  `cache/mnist_experiment/rotated_mnist/plan9/`;
- keep source-controlled code, configurations, notebook cells, and documented
  commands free of machine-specific absolute paths; discover the repository
  root at runtime and store in-repository references as relative paths;
- do not make existing Plan 5--8 runners import Plan 9 code;
- do not change `src/controller.py`, `mathematical_overview.ipynb`, or
  `mnist-findings.ipynb`;
- reuse stable loaders and representation operations when their semantics are
  unchanged, but copy no completed artifact into a mutable location; and
- make removal of the Plan 9 package leave all prior experiments executable
  and interpretable.

### Batched execution, interruption, and shutdown

Long Plan 9 computations must be divided into stable work units, such as one
replica, condition, oracle block, or analysis block. A source-controlled
configuration determines every work-unit ID before execution. The command
center writes a manifest containing dependencies and status, skips validated
completed units on `--resume`, and never changes a completed unit in place.

Run unattended batches in a named detached `tmux` session. Cancellation in
the VS Code interface controls the active agent interaction and must not be
treated as a guaranteed way to terminate either a foreground process tree or
a detached `tmux` job. Provide commands to inspect and reattach to the named
session. Provide a cooperative stop command that writes a stop request; the
worker checks it between work units, finishes the current atomic unit, records
its state, and exits without scheduling another unit. Also support a maximum
wall-time budget with the same between-unit behavior.

A normal node shutdown terminates the worker and the `tmux` server; the
session itself does not survive reboot. After restart, invoking the same
bundle with `--resume` must recover all validated completed work and resume a
valid checkpoint or restart only the interrupted unit. A hard power loss is
not a normal stopping mechanism and may lose the current unit, but must not
invalidate earlier completed units.

Write checkpoints and final files through temporary paths on the same
filesystem, flush them before atomic rename, and create `COMPLETED` only after
all required artifact files and hashes validate. Handle `SIGINT` and `SIGTERM`
by recording an interrupted state when possible; never promote a partial file
to completed status. Resume must validate hashes and schemas rather than
trusting file presence. Commands, manifests, and stored repository references
must remain relative to the discovered repository root.

## Status

| Study | Name | Evidence level | Status | Dependency |
|---|---|---|---|---|
| E9.0 | Model, timing, and artifact contracts | Algebraic | Supported | None |
| E9.1 | Oracle-assisted anchor-cancellation audit | Oracle retrospective | Inconclusive | E9.0 |
| E9.2 | Smooth-drift filtering and variance correction | Oracle retrospective | Inconclusive | E9.1 informative mixed result |
| E9.3 | Prospective online observer | Online observer | Blocked | E9.2 calibration gate not met |
| E9.4 | Closed-loop development challenge | Closed-loop development | Blocked | E9.3 deployment gate |
| E9.5 | Hybrid and independent confirmation | Independent confirmation | Blocked | E9.4 predictive gate |
| E9.6 | Mathematical and project integration review | Integration | Blocked | User review of supported evidence |
| E9.7 | Movement-opportunity and conditioning audit | Oracle retrospective | Supported | E9.1--E9.2 findings |
| E9.8 | Lag-separated cross-moment audit | Oracle retrospective | Inconclusive | E9.7 |
| E9.9 | EWC-affinity remainder attribution | Oracle retrospective | Supported | E9.8 diagnosis |
| E9.10 | Opportunity-qualified path design | Oracle retrospective | Rejected | E9.7 |
| E9.11 | Prospective cross-moment observer | Online observer | Blocked | E9.8 and E9.10 gates |
| E9.12 | Predictable gain contract and artifact audit | Algebraic / artifact-only | Supported | E9.9 |
| E9.13 | Dual-timescale curvature estimator | Numerical smoke | Supported | E9.12 |
| E9.14 | Structured-gain calibration and pivot gate | Oracle retrospective | Rejected | E9.13 |

Append additional studies after E9.6 rather than renumbering completed work.

## Scientific Charter

### Filtrations and predictability

Keep two filtrations distinct:

- $\mathcal G_t$ is the full analytical filtration immediately before batch
  $t+1$. It contains the population and learner state needed to state the
  probability model.
- $\mathcal F_t\subseteq\mathcal G_t$ is the learner's deployable filtration.
  It contains observed batches, accepted parameters, controller state, and the
  predictable Fisher summary, but not $\theta_t^\star$ or population oracle
  quantities.

The action $\pi_t$ must be $\mathcal F_t$-measurable. A statistic formed after
accepting transition $t$ may update the decision for transition $t+1$ but may
not retroactively determine $\pi_t$.

Markovianity does **not** imply
$\mathbb E[e_t\mid\mathcal F_t]=0$. If the relevant filtration contains the
current state, then the realized anchor error
$e_t=\widehat\theta_t-\theta_t^\star$ is already fixed. Plan 9 does not require
it to be centered. Centering across hypothetical learner replicas remains a
separate marginal calibration assumption.

### Smooth population dynamics

Treat the full controlled state as locally Markov. For environmental step
$\Delta_t$, assume

$$
\theta_{t+1}^\star
=\theta_t^\star+\Delta_t b(\theta_t^\star)+r_{t+1}^{\theta},
$$

where $b$ is locally $C^1$ and
$\|r_{t+1}^{\theta}\|=O(\Delta_t^2)$ in the deterministic rotated-MNIST
experiment. Consequently,

$$
d_{t+1}^\star-d_t^\star=O(\Delta_t^2)
$$

for equal small steps on a compact local neighborhood. This supports local
smoothing but does not by itself identify $d_t^\star$ from learner updates.
In a later stochastic-environment extension, distinguish predictable drift
from realized process noise before reusing this estimand.

### Local estimator and EWC dynamics

Let the estimator obtained from the fresh batch satisfy

$$
\widetilde\theta_{t+1}
=\theta_{t+1}^\star+\xi_{t+1},
$$

with the local martingale and covariance assumptions

$$
\mathbb E[\xi_{t+1}\mid\mathcal G_t]=0,
\qquad
\operatorname{Cov}(\xi_{t+1}\mid\mathcal G_t)
=\frac{K_{t+1}}{m_t}+o(m_t^{-1}).
$$

Conditionally adjacent batch innovations are initially assumed uncorrelated.
This is testable in simulation and must be revisited for replay or overlapping
batches.

Model the optimized local EWC update as

$$
\widehat\theta_{t+1}
=(1-\pi_t)\widehat\theta_t
+\pi_t\widetilde\theta_{t+1}
+a_{t+1},
$$

where $a_{t+1}$ is the EWC-affinity remainder. It includes unequal local
curvature, nonquadratic likelihood behavior, optimizer truncation, and any
failure of the accepted parameter update to behave like an affine statistical
combination. Do not assume this remainder is centered merely for analytical
convenience. Record or bound its observable implications wherever possible.

### Anchor-cancelling observation

Define

$$
u_t=\widehat\theta_{t+1}-\widehat\theta_t,
\qquad
Y_t=\frac{u_t}{\pi_t}.
$$

The positive action floor makes the normalization well-defined. The local
model gives

$$
Y_t
=d_t^\star-e_t+\xi_{t+1}+\frac{a_{t+1}}{\pi_t}.
$$

The historical EMA trend smooths $Y_t$ and therefore need not remove the
persistent $-e_t$ term. Instead define, after transition $t$ has been
accepted,

$$
Z_t
:=Y_t-(1-\pi_{t-1})Y_{t-1}.
$$

For $t\geq1$, direct substitution yields

$$
Z_t
=d_t^\star
+(\xi_{t+1}-\xi_t)
+\frac{a_{t+1}}{\pi_t}
-\frac{a_t}{\pi_{t-1}}.
$$

Thus the realized anchor error cancels algebraically. Under negligible or
appropriately controlled EWC remainder,

$$
\mathbb E[Z_t\mid\mathcal G_{t-1}]
\approx
\mathbb E[d_t^\star\mid\mathcal G_{t-1}].
$$

This is a one-transition-lagged observation. A Plan 9 controller may use it
only after acceptance to update a recommendation for the next transition.

### Instantaneous movement energy

With a common local metric $G_t$ approximating $\mathcal I_t$, define

$$
S_{Z,t}^{\mathrm{raw}}=Z_t^TG_tZ_t.
$$

For independent adjacent innovations, its leading noise contribution is

$$
V_{Z,t}
=\operatorname{tr}\!\left[
G_t\left(
\frac{K_t}{m_{t-1}}+
\frac{K_{t+1}}{m_t}
\right)
\right].
$$

Retain both the signed calibration statistic and deployable nonnegative form:

$$
\widetilde S_{Z,t}=S_{Z,t}^{\mathrm{raw}}-\widehat V_{Z,t},
\qquad
\widehat S_{Z,t}=[\widetilde S_{Z,t}]_+.
$$

Clipping introduces positive bias and must be reported, not hidden. The
normalized movement estimate is

$$
\widehat\rho_{Z,t}
=\frac{m_t\widehat S_{Z,t}}
{\max(\widehat D_t^{\mathrm{new}},\varepsilon)}.
$$

When old and new covariance shapes differ, evaluate the general marginal
recommendation

$$
\widehat\pi_{B,t}^{\mathrm{marg}}
=\frac{
\widehat S_{Z,t}+q_t\widehat D_t^{\mathrm{old}}
}{
\widehat S_{Z,t}+q_t\widehat D_t^{\mathrm{old}}
+\widehat D_t^{\mathrm{new}}/m_t
}
$$

rather than forcing both shapes into one scalar $D_t$.

### Smoothed movement energy

Smoothness of $b$ permits a local weighted estimate

$$
\overline Z_t=\sum_{k\leq t}w_{t,k}Z_k,
\qquad
\sum_{k\leq t}w_{t,k}=1.
$$

The differenced errors
$\eta_k=\xi_{k+1}-\xi_k$ are not independent: adjacent terms share
$\xi_k$ with opposite signs. For locally independent batch innovations with
covariances $\Sigma_k$, the exact finite-window covariance is

$$
\operatorname{Cov}\!\left(\sum_kw_k\eta_k\right)
=\sum_kw_k^2(\Sigma_{k+1}+\Sigma_k)
-2\sum_kw_k w_{k-1}\Sigma_k.
$$

Any quadratic debiasing of $\overline Z_t$ must include the negative adjacent
cross-covariance. Treating smoothed $Z_t$ values as independent is prohibited.
This telescoping structure may reduce variance materially and is one of Plan
9's principal hypotheses.

## Frozen Evidence Baseline

The first studies use these completed artifacts read-only:

- Plan 6 population oracle:
  `rotated_mnist_phase6_oracle_full_path__replica-0001__7c7dc5936d08fb91`;
- Plan 7 movement audit:
  `rotated_mnist_phase7_movement_audit_primary_v2__replica-0001__c24262eac6ec1625`;
- Plan 8 single lap:
  `rotated_mnist_phase8_single_lap_rechallenge__replica-0001__6445ab53bb10cd25`;
- Plan 8 double lap:
  `rotated_mnist_phase8_double_lap_reversal_stress__replica-0001__f035d4b8cdf6d894`.

The Plan 8 artifacts retain all accepted 512-dimensional parameter vectors and
displacements. Plan 6 retains high-sample local parameter and Fisher
references. E9.1 and E9.2 should therefore require no learner training. Every
source completion marker, configuration hash, file hash, parameter layout,
transition identity, schedule direction, and data partition must be validated
before quantities are combined.

## E9.0: Model, Timing, and Artifact Contracts

**Evidence level:** Algebraic  
**Status:** Proposed

### Question

Can the Markov model, anchor cancellation, variance correction, and causal
timing be implemented without changing an existing runner or estimand?

### Work

1. Create the detachable `markov_movement` package with strict configuration,
   artifact, estimator, and analysis modules.
2. Freeze exact source IDs and hashes for the retrospective studies.
3. Confirm vector, Fisher, condition, schedule, and transition alignment.
4. Implement pure operations for $Y_t$, $Z_t$, instantaneous noise
   correction, general marginal recommendations, and finite weighted
   covariance correction.
5. Preserve score-versus-loss-gradient signs and ordinary Euclidean parameter
   coordinates. No Fisher inverse, pseudoinverse, or coordinate conversion is
   permitted.
6. Add unit tests for exact anchor cancellation under an affine synthetic
   learner, nonzero anchor error, EWC remainder propagation, time alignment,
   no-look-ahead decisions, unequal batch sizes, and zero population movement.
7. Verify the weighted correction by Monte Carlo, including the negative
   covariance between adjacent differenced innovations.
8. Implement the resumable command-center contract: stable bundles and work
   IDs, dependency-aware unit scheduling, `--resume`, cooperative stop
   requests, maximum wall-time limits, and concise progress/status output.
9. Add interruption tests covering a stop between units, `SIGINT` during an
   active unit, an invalid partial checkpoint, idempotent resume, and reuse of
   completed dependencies. Document portable detached-`tmux`, reattach,
   graceful-stop, and post-reboot resume commands emitted by the command
   center.

### Gate

Proceed only if the synthetic identities close to floating-point tolerance,
the Monte Carlo correction agrees with its theoretical expectation, and every
required historical vector is present. The interruption smoke test must also
show that resume reuses completed units, rejects invalid partial state, and
restarts no more than the interrupted unit. Otherwise mark E9.0 `Rejected` or
`Inconclusive` before running a learner.

### Execution record: 2026-09-05

Implemented the detachable estimator, immutable artifact store, resumable work
units, cooperative stopping, wall-time limits, and portable `tmux` command
generation under `rotated_mnist/markov_movement/`. Unit tests verify affine
anchor cancellation with nonzero anchor error, the exact adjacent-innovation
covariance, Monte Carlo agreement, immutable completion, and interrupted-unit
behavior. Frozen Plan 6 and Plan 8 vectors and transition ancestry validated.
The E9.0 gate passed.

## E9.1: Oracle-Assisted Anchor-Cancellation Audit

**Evidence level:** Oracle retrospective  
**Status:** Inconclusive

### Question

Does $Z_t$ estimate population movement more accurately than the historical
EMA of $Y_t$ when both are evaluated in the same high-sample population
Fisher metric?

### Design

1. Reconstruct $Y_t$ and $Z_t$ from completed Plan 8 parameter displacements
   and realized actions. Do not optimize a model or resume controller state.
2. Use exact Plan 6 transition matching to obtain population
   $d_t^\star$, $S_t$, $D_t^{\mathrm{old}}$, $D_t^{\mathrm{new}}$, and the
   reference Fisher representation.
3. Use decomposed EDR as the primary observed path. Include fixed $.05$ and
   tracked-$q_t$ Plan 8 paths as action-dependence sensitivities when their
   vector artifacts and transition ancestry match exactly.
4. Evaluate raw and oracle-debiased $Z_t$ separately. The oracle correction
   diagnoses the Markov estimator; it is not deployable evidence.
5. Compare with the stored historical trend using the same metric, transition
   subset, and population target.
6. Stratify by single/double lap, linear/sigmoid schedule, leg, movement
   direction, cold start, and the first eight post-reversal transitions.

### Primary metrics

- Fisher-metric squared error and cosine alignment between $Z_t$ and
  $d_t^\star$;
- signed error, MAE, and robust log ratio for movement energy;
- signed and clipped $\widehat S_t$ calibration and clipping frequency;
- $\widehat\rho_t$ error;
- resulting marginal-action MAE against the Plan 6 population recommendation;
- improvement over the historical trend estimator; and
- sensitivity to small population movement, oracle precision, and realized
  action size.

### Decision rule

- **Supported:** the anchor-cancelling estimator lowers marginal-action MAE in
  all four principal design/schedule groups and lowers the median group MAE by
  at least 25%, without obtaining the result primarily through clipping.
- **Inconclusive:** direction improves but quadratic energy remains too noisy,
  or gains depend materially on schedule, action, or low-precision oracle
  points. Open E9.2 with the ambiguity recorded.
- **Rejected:** it improves fewer than three principal groups, or its residual
  behavior remains incompatible with the predicted innovation covariance and
  smooth-drift approximation on high-precision oracle points. Do not assign
  that incompatibility uniquely to the affine-EWC remainder from this study.

Write an immutable scalar/vector audit and a small artifact-only notebook.
Do not change an online controller.

### Execution record: 2026-09-05

Artifact
`rotated_mnist_plan9_e9_1_anchor_cancellation__replica-0001__b1234a5779ce2e42`
reproduced population covariance coefficients within `3.6e-15`. The
oracle-debiased $Z_t$ action improved over the historical trend in only one of
four design/schedule groups. Median relative action-MAE reduction was `-6.36`,
and mean Fisher cosine was near zero in every group. Because the linear
double-lap group improved while the remaining quadratic estimates were very
noisy, classify E9.1 as **Inconclusive** and open E9.2 through the predeclared
informative-mixed-result route.

## E9.2: Smooth-Drift Filtering and Variance Correction

**Evidence level:** Oracle retrospective  
**Status:** Inconclusive

### Question

Can smoothness of $b(\theta)$ reduce the variance of the anchor-cancelling
observation without recreating the historical trend's anchor bias or excessive
reversal lag?

### Design

1. Predeclare a small smoothing family before viewing predictive outcomes:
   no smoothing, the nearest historical angular half-life, and one half/double
   sensitivity on each side.
2. Smooth vectors before taking their Fisher norm. Also retain a scalar-energy
   EMA as a clearly separate estimator.
3. Compute the exact finite-window innovation covariance from realized weights,
   including adjacent negative cross-covariance and unequal local covariance
   shapes.
4. Retain signed and clipped energy estimates. Do not choose a half-life by
   downstream accuracy or NLL.
5. Quantify local-linearity bias by comparing filtered population movement
   with the unsmoothed Plan 6 path.
6. Report response lag and overshoot around every reversal. A low whole-path
   error may not hide failure during regime changes.

### Gate

Promote one estimator to E9.3 only if it improves oracle movement-energy and
action calibration over both unsmoothed $Z_t$ and the historical trend in at
least three of four principal groups, has no catastrophic reversal window,
and remains stable under the neighboring half-life sensitivities. Selection
uses calibration only, never learner predictive outcomes.

### Execution record: 2026-09-05

The first completed smoothing artifact used absolute half-life labels and is
retained as a superseded bookkeeping pilot. Artifact
`rotated_mnist_plan9_e9_2_smooth_drift_v2__replica-0001__b4c838dcc0b327bf`
compared the intended common `0.5x`, `1x`, and `2x` angular half-life choices
across all four groups. The selected `2x` vector smoother reduced
movement-energy MAE relative to instantaneous $Z_t$ in all four groups and
did not catastrophically fail a reversal window. It beat both instantaneous
$Z_t$ and the historical trend in action MAE in only one group, however; the
gate requires three. E9.2 is therefore **Inconclusive**, and no estimator is
promoted to E9.3.

## E9.3: Prospective Online Observer

**Evidence level:** Online observer  
**Status:** Blocked on E9.2

The 2026-09-05 E9.2 calibration gate was not met. No prospective learner was
run.

### Question

Does the selected estimator remain calibrated when it uses only quantities
available from the learner's predictable rank-8-plus-diagonal Fisher summary
and residual-risk state?

### Design

1. Add Plan 9 instrumentation to a newly versioned runner; never mutate a Plan
   8 artifact.
2. Use fixed $\pi=.05$ as the primary action so the new estimator cannot
   alter its own evidence. Pair a tracked-$q_t$ sensitivity if E9.1 identifies
   meaningful action dependence.
3. Reuse the validated source initialization and stream only for exact
   development pairing. Recompute the complete learner and Fisher trajectory.
4. Calculate $Z_t$, its weighted correction, $\widehat S_t$,
   $\widehat\rho_t$, and the implied recommendation after each accepted update.
   Record them without actuation.
5. Retain population references strictly for offline scoring. No population
   quantity may enter the observer state.
6. Record Fisher quadratic products, correction components, clipping,
   prediction timing, optimizer diagnostics, and cost sufficient to reproduce
   every recommendation.

### Gate

Proceed to closed loop only if the online observer preserves the direction of
the E9.2 improvement, improves marginal-action MAE over historical movement
EDR, and does not rely on unavailable covariance shapes or unstable clipping.
Failure of online $D_t$ calibration must remain distinguishable from failure
of the $Z_t$ numerator.

## E9.4: Closed-Loop Development Challenge

**Evidence level:** Closed-loop development  
**Status:** Blocked on E9.3

E9.3 produced no deployable estimator, so no closed-loop Plan 9 condition was
run.

### Question

Does the calibrated recommendation remain useful after its actions alter
$q_t$, EWC weighting, accepted parameters, and subsequent observations?

### Design

1. Recompute paired fixed $.05$, tracked-$q_t$, historical decomposed EDR, and
   Markov-movement conditions from common initial state and streams.
2. Use one realized $\pi_t$ for both EWC and Fisher-memory weighting. Preserve
   the exact $q_t$ recursion and the established action bounds.
3. Keep EWC-only as the primary controller-isolation setting. Do not add replay
   until movement-aware actuation demonstrates value.
4. Use the canonical single and double laps with both schedule shapes. Add no
   newly tuned environment merely to favor the estimator.
5. Evaluate action calibration, environmental accuracy/NLL, fixed-panel
   retention, worst-class recall, reversal behavior, and learner/Fisher cost.
6. Treat fixed $.05$ as the prospective practical baseline and fixed $.025$
   only as a post-facto research benchmark if included.

### Gate

Classify mechanistic and predictive evidence separately. A well-calibrated
recommendation with neutral predictive outcomes remains scientifically useful;
a predictive gain without calibrated internal quantities is not evidence for
the proposed Markov estimator. Independent confirmation is justified only
when the new condition is operationally stable and improves or closely tracks
the prospective fixed baseline across both schedules without sacrificing
retention materially.

## E9.5: Hybrid and Independent Confirmation

**Evidence level:** Independent confirmation  
**Status:** Blocked on E9.4

E9.4 produced no qualifying closed-loop treatment. The replica contrast and
confidence-width amendment therefore remain intentionally unspecified, and no
independent computation was run.

If approved after the E9.4 check-in:

1. Introduce Hybrid B32 as a separate applied treatment without changing the
   EWC-only controller result.
2. Generate fresh early-stopped initialization and high-quality initial Fisher
   per replica. Pair conditions within, never across, replicas.
3. Begin with eight replicas and extend in predeclared blocks using a
   precision-only stopping rule capped at 32 replicas.
4. Preserve complete metric and action trajectories. Independent replicas,
   not transitions, are the inferential units.
5. Report calibration, predictive utility, data exposure, compute, and memory.
6. Keep an unbounded-replay ceiling contextual rather than requiring the
   adaptive controller to beat an out-of-scope hardware regime.

The exact contrast and confidence-width target must be written as an amendment
and approved before execution.

## E9.6: Mathematical and Project Integration Review

**Evidence level:** Integration  
**Status:** Blocked on user review

Only reviewed, supported evidence may leave the optional module.

1. If only the algebra and oracle audit survive, document the construction in
   the Plan 9 notebook and ledger; do not change the project theory.
2. If the online observer is calibrated, propose the controlled Markov model,
   filtration distinction, smooth-drift assumption, anchor cancellation, and
   variance correction for the appendix of `mathematical_overview.ipynb`.
3. Promote only the concise population estimand and any genuinely general
   identity to the main body. Keep filtering, clipping, half-lives, and online
   diagnostics in the appendix.
4. If closed-loop or independent evidence supports an applied conclusion,
   separately propose updates to `mnist-findings.ipynb`, README guidance, and
   shared controller code.
5. Preserve the result that Markovianity alone does not center realized anchor
   error and the distinction between instantaneous recommendation calibration
   and path-optimal policy performance.

### Final review

Decide whether the Markov movement estimator should be integrated, retained as
an oracle-assisted diagnostic, revised through another ledger study, or closed
as a productive negative result. No integration is automatic.

## E9.7: Movement-Opportunity and Conditioning Audit

**Date opened:** 2026-09-07  
**Status:** Supported  
**Evidence level:** Oracle retrospective  
**Depends on:** E9.1--E9.2 findings

### Question

Did E9.1--E9.2 fail chiefly because movement was estimated poorly, or because
the underlying movement premium was too small to create a materially different
action even under an oracle?

### Frozen hypothesis and estimands

Write

$$
A_t=S_t+q_tD_t^{\mathrm{old}},
\qquad
B_t=\frac{D_t^{\mathrm{new}}}{m_t},
\qquad
\pi_t^{\mathrm{marg}}=\frac{A_t}{A_t+B_t}.
$$

With

$$
\rho_t=\frac{m_tS_t}{D_t^{\mathrm{new}}},
\qquad
r_{D,t}=\frac{D_t^{\mathrm{old}}}{D_t^{\mathrm{new}}},
$$

the action odds satisfy

$$
\frac{\pi_t^{\mathrm{marg}}}{1-\pi_t^{\mathrm{marg}}}
=\rho_t+m_tq_t r_{D,t}.
$$

Keep three conditioning targets distinct:

$$
S_t^{\mathrm{pop}}=\|d_t^\star\|_{G_t}^2,
\qquad
S_t^{\mathrm{marg}}=\|d_t^\star-\mu_t\|_{G_t}^2,
\qquad
S_t^{\mathrm{cond}}=\|d_t^\star-e_t\|_{G_t}^2.
$$

The population target equals the marginal target only under the centering
assumption $\mu_t=0$. The conditional target describes the realized learner
but is not deployable because $e_t$ requires the population optimum. Do not
silently use one target as evidence about another.

### Sources and design

1. Reuse the validated Plan 6--9 artifacts read-only.
2. Reconstruct $\rho_t$, covariance-only odds, marginal oracle odds, and
   $\Delta\pi_t=\pi_t^{\mathrm{marg}}-\pi_t^{\mathrm{cov}}$ at every aligned
   transition.
3. Evaluate the population and conditional targets wherever the retained
   population parameters permit them. Treat the marginal target as
   unidentified unless enough independent replicas exist to estimate $\mu_t$.
4. Plot opportunity over path position and report how long it remains above
   practically meaningful action differences. Use $|\Delta\pi_t|\geq .02$
   for at least eight consecutive transitions as the primary opportunity
   threshold, with `.01` and `.05` as sensitivities.
5. Separate absence of oracle opportunity from estimator error. No online
   estimator is evaluated in this study.

### Decision rule

- **Opportunity present:** the primary threshold is met in at least one
  pre-existing path without violating the local-linearity checks.
- **Opportunity weak:** only a sensitivity threshold is met.
- **Opportunity absent:** even the `.01` sensitivity is not sustained.

An absent opportunity does not reject adaptive control. It requires E9.10 to
construct a mechanism challenge before further prospective computation.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_7_movement_opportunity__replica-0001__6109e38ad6876642`
validated 950 aligned transitions and classified population opportunity as
**absent**. Across the four principal groups, mean population $\rho_t$ was
`.00286`--`.00330`, mean oracle action lift over covariance-only was
`.00252`--`.00282`, and the maximum lift was `.01314`. No group sustained even
the `.01` sensitivity threshold for eight transitions.

The realized conditional target was qualitatively different: mean conditional
$\rho_t$ ranged from `6.48` to `63.83` and implied much larger actions. This is
oracle-only evidence about learner anchor error, not identification of the
population or centered marginal recommendation. E9.7 is **Supported** as an
opportunity audit; its scientific result is absence of material population
opportunity on the retained paths.

## E9.8: Lag-Separated Cross-Moment Audit

**Date opened:** 2026-09-07  
**Status:** Inconclusive  
**Evidence level:** Oracle retrospective  
**Depends on:** E9.7

### Question

Can population movement energy be estimated from cross-products of locally
repeated anchor-cancelling observations without subtracting a large estimated
self-noise energy?

### Frozen hypothesis and estimator

Write $Z_i=d_i^\star+\eta_i+r_i$, where $\eta_i$ is the differenced batch
innovation and $r_i$ contains the scaled EWC-affinity remainder. For two local
observations,

$$
\mathbb E[Z_i^TG_tZ_j]
=d_i^{\star T}G_td_j^\star
+\operatorname{tr}\!\left(G_t\operatorname{Cov}(\eta_i,\eta_j)\right)
+\text{remainder terms}.
$$

The cross-product is therefore a movement-energy estimator, not an
autocorrelation estimand. Dependence is a nuisance term that determines which
cross-lags are admissible. Because adjacent differenced innovations share one
batch innovation, the primary estimator excludes $|i-j|\leq1$:

$$
\widehat S_{t}^{\mathrm{cross}}
=
\frac{
\sum_{i<j}w_{t,ij}\mathbf 1\{|i-j|>1\}Z_i^TG_tZ_j
}{
\sum_{i<j}w_{t,ij}\mathbf 1\{|i-j|>1\}
}.
$$

Local smoothness supplies
$d_i^{\star T}G_td_j^\star\approx\|d_t^\star\|_{G_t}^2$ inside the window.
Unlike a debiased self-product, this estimator does not obtain a small signal
by subtracting two large positive quadratic estimates.

The primary deterministic pair weight is
$w_{t,ij}=a_{t,i}a_{t,j}$, where $a_{t,i}$ is the frozen exponential angular
weight. Each unordered pair $i<j$ is included once and the pair weights are
renormalized after lag exclusion. Windows reset at recorded leg boundaries so
the common-drift approximation never intentionally mixes opposite movement
directions. Shared observations make pair estimates dependent but do not
double-count the point estimand; report effective pair count rather than
treating all pairs as independent observations.

### Sources and design

1. Use only completed E9.1--E9.2 vectors and population references. No learner
   training is permitted.
2. Freeze the primary local window from the E9.2 `2x` angular support. Compare
   half/double window sensitivities without selecting by predictive outcomes.
3. Use lag exclusion $L=1$ as primary. Report $L=2$ and $L=4$ sensitivities;
   these diagnose longer dependence rather than provide unrestricted tuning.
4. Estimate Fisher-weighted residual cross-covariance by lag using oracle
   $Z_t-d_t^\star$. Report signs, uncertainty across eligible pairs, and the
   effective number of cross-pairs.
5. Retain signed $\widehat S_t^{\mathrm{cross}}$ for calibration. Clip only
   when mapping it into a feasible action, and report clipping frequency.
6. Compare movement-energy bias/MAE, $\rho_t$ error, and action MAE with
   instantaneous $Z_t$, the E9.2 vector smoother, and the historical trend.
7. Stratify reversal windows separately so a long-window average cannot hide
   lag at a change in direction.

### Decision rule

- **Supported:** cross-moments improve movement-energy MAE over the E9.2
  smoother in at least three of four principal groups, improve or preserve
  action MAE where E9.7 shows oracle opportunity, and remain directionally
  stable under neighboring window and lag exclusions.
- **Inconclusive:** movement calibration improves but residual dependence or
  weak oracle opportunity prevents an action conclusion.
- **Rejected:** improvement requires outcome-selected lags, clipping, or a
  dependence pattern incompatible with the assumed cross-moment cancellation.

Only a supported result may enter E9.11. A dependence-related inconclusive or
rejected result opens E9.9.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_8_lagged_cross_moment__replica-0001__3a5f6c720b52b9f4`
evaluated 3,264 estimator rows and 48 lag diagnostics in 16.1 seconds. The
primary cross-moment estimator improved signed movement-energy MAE over the
E9.2 vector smoother in zero of four groups. Its sensitivity behavior was
stable, but that stability preserved the wrong quantity.

Clipped actions sometimes appeared substantially better because negative
cross-moments collapsed to the already well-calibrated covariance-only
recommendation. That is not evidence of movement identification. Oracle
residual cross-moments remained large and alternating beyond lag one,
especially on sigmoid paths, contradicting the simple adjacent-innovation
model. E9.8 is **Inconclusive** rather than promoted and opens E9.9.

## E9.9: EWC-Affinity Remainder Attribution

**Date opened:** 2026-09-07  
**Status:** Supported  
**Evidence level:** Oracle retrospective  
**Depends on:** E9.8 diagnosis

### Question

If lag-separated cross-moments remain biased, is the dominant cause serial
batch innovation or failure of the optimized EWC update to behave as an
affine statistical combination?

### Sources and design

1. Run this study only if E9.8 exposes residual dependence inconsistent with
   the adjacent-innovation model.
2. At predeclared representative transitions, reproduce the exact fresh batch
   and fit a disposable unregularized local estimator under calibrated fixed
   optimizer budgets. This fit is diagnostic and must never control a learner.
3. Measure

   $$
   a_{t+1}
   =\widehat\theta_{t+1}
   -(1-\pi_t)\widehat\theta_t
   -\pi_t\widetilde\theta_{t+1},
   $$

   and its contribution
   $a_{t+1}/\pi_t-a_t/\pi_{t-1}$ to $Z_t$.
4. Report Fisher energy, direction, lag dependence, and optimizer-budget
   sensitivity separately for innovation and affinity components.
5. Do not call $a_t$ intrinsic model error when it changes materially with the
   operational definition or optimizer budget of $\widetilde\theta_t$.

### Decision and next action

This is an attribution study, not a controller gate. It must conclude whether
the present estimator needs a better observation model, a robust dependence
correction, or abandonment under the small-batch operating regime.

### Execution record: 2026-09-07

The first completed attribution artifact lacked a direct paired optimizer-
budget statistic and remains immutable as a superseded pilot. Versioned
artifact
`rotated_mnist_plan9_e9_9_affinity_attribution_v2__replica-0001__c40471e46aa06d8f`
ran 448 CUDA fresh-batch fits in 56.9 seconds. Every primary fit started from
the exact stored pre-transition parameter, used the exact four-observation
batch, removed EWC, and used fresh strong-Wolfe L-BFGS state.

Removing the measured affinity-remainder contribution reduced mean residual
Fisher energy by more than half in three of four groups. Their mean
residual/remainder cosine was `.866`--`.907`, and median energy reduction was
`.854`--`.931`. Double-lap linear was the genuine exception: cosine was `.030`
and removal increased residual energy. All 48 paired 50/100-step sensitivity
fits produced the same float32 parameter result, so this attribution is not an
optimizer-budget artifact within the tested procedure. E9.9 is **Supported**:
the affine-EWC remainder is the dominant omitted term in three groups, but not
a universal explanation.

## E9.10: Opportunity-Qualified Path Design

**Date opened:** 2026-09-07  
**Status:** Rejected  
**Evidence level:** Oracle retrospective  
**Depends on:** E9.7

### Question

Can a rotated-MNIST mechanism challenge exhibit a sustained oracle movement
premium while preserving small batches, local Fisher validity, and the
modularity of Plan 9?

### Design

1. Screen candidate schedule speeds, rotation ranges, and plateau lengths with
   population references before running a continual learner.
2. Keep $m$ fixed in the low-data regime. Do not create apparent opportunity
   merely by increasing fresh-batch size.
3. Require the E9.7 primary $\Delta\pi$ threshold and retain local-linearity,
   Fisher-reference precision, and task-learnability diagnostics.
4. Select no path using downstream learner accuracy or NLL. Label the selected
   path a mechanism challenge rather than representative applied evidence.
5. Preserve the canonical Plan 5 path as an external reference; do not mutate
   its artifacts or reinterpret its negative result.

### Gate

No new prospective learner may be launched unless at least one candidate has
material sustained oracle opportunity and passes the numerical and local-model
checks. If no candidate qualifies, close this rotated-MNIST route rather than
escalating data and compute indefinitely.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_10_opportunity_path_screen__replica-0001__2051e6774c0e6333`
screened 185 population-reference transitions without learner training or
predictive-outcome selection. Increasing angular speed raised mean oracle
action lift from `.00393` to `.07937`, confirming the expected movement effect.
It simultaneously shortened the useful regime and increased within-step drift
variation. No candidate sustained a `.02` lift for eight transitions while
meeting the local common-drift criterion. E9.10 is **Rejected** under its
predeclared gate; no opportunity-qualified rotated-MNIST path is promoted.

## E9.11: Prospective Cross-Moment Observer

**Date opened:** 2026-09-07  
**Status:** Blocked  
**Evidence level:** Online observer  
**Depends on:** Supported E9.8 and qualified E9.10 path

Recompute a complete fixed-action learner and Fisher trajectory on the
qualified path. Record the cross-moment recommendation causally without
actuation, use population quantities only for sealed offline scoring, and
compare calibration with the historical EDR observer. A separate reviewed
amendment must specify any later closed-loop or predictive experiment; E9.11
does not authorize one automatically.

### Execution record: 2026-09-07

E9.11 was not run. E9.8 did not support the cross-moment estimator and E9.10
found no qualifying path. Running a prospective observer would therefore
convert two failed mechanistic gates into an uninterpretable predictive
experiment. The study remains **Blocked** without an artifact.

The artifact-only notebook
`rotated_mnist/markov_movement_extensions_results.ipynb` presents the E9.7--E9.10
evidence and the resulting E9.11 gate decision. It performs no training,
resumption, or artifact repair.

## E9.12: Predictable Gain Contract and Artifact Audit

**Date opened:** 2026-09-07  
**Status:** Supported  
**Evidence level:** Algebraic / artifact-only  
**Depends on:** E9.9

### Question

Can the predictable component of the EWC-affinity remainder be represented by
a local matrix gain estimated from accumulated second-order information, rather
than by the unsuccessful LFU derivative estimator?

### Frozen local model

Let $G_t^{\mathrm{old}}$ be the pre-transition EWC curvature summary and let
$H_{t\mid t-1}^{\mathrm{new}}$ be a causal estimate of current new-data
curvature formed only from batches observed before transition $t$. Under the
local quadratic objective, define

$$
\mathsf K_{t\mid t-1}^{\mathrm{gain}}(\pi)
=\left[(1-\pi)G_t^{\mathrm{old}}
+\pi H_{t\mid t-1}^{\mathrm{new}}\right]^{-1}
\pi H_{t\mid t-1}^{\mathrm{new}}.
$$

This is a directional linear solve, not a request to materialize or
pseudoinvert a Fisher matrix. The superscript distinguishes this gain from the
covariance-shape matrices $K_t^{\mathrm{old/new}}$ used elsewhere in this
ledger. The scalar-affine model is the restricted case
$\mathsf K_t^{\mathrm{gain}}(\pi)=\pi I$. The corresponding local tracking
recursion is

$$
e_{t+1}=(I-\mathsf K_t^{\mathrm{gain}})(e_t-d\theta_t)
+\mathsf K_t^{\mathrm{gain}}\epsilon_{t+1}.
$$

The experiment estimates expected likelihood curvature with score outer
products. It does not claim that an EMA Fisher equals each realized
four-observation loss Hessian. Any resulting discrepancy remains part of the
observation remainder.

### Artifact and estimand audit

1. Verify that the authoritative E9.1 and E9.9 artifacts identify the realized
   update $u_t$, action $\pi_t$, operational fresh-batch displacement
   $\widetilde u_t$, pre-transition parameter, incoming batch, and source
   information needed to reproduce EWC curvature without mutating an earlier
   run.
2. Reconstruct $\widetilde u_t=(u_t-a_{t+1})/\pi_t$ from the retained E9.9
   affinity vectors and verify it against retained scalar fit diagnostics.
3. Record the scalar Fisher projection

   $$
   k_t^{\mathrm{oracle}}
   =\frac{\langle u_t,\widetilde u_t\rangle_{G_t}}
   {\|\widetilde u_t\|_{G_t}^2}
   $$

   only as an attributive ceiling. Construct causal scalar-EMA baselines using
   past $k_s$ values; they must not select matrix-estimator hyperparameters.
4. Freeze the score convention, parameter chart, update timing, curvature
   representation, solver tolerances, and source hashes before implementation.
5. Confirm that no current-batch curvature enters
   $H_{t\mid t-1}^{\mathrm{new}}$. Current scores may update only the next
   state $H_{t+1\mid t}^{\mathrm{new}}$.

The authoritative trajectory files retain parameters and displacements but
not every pre-transition Fisher representation. E9.12 must therefore freeze a
deterministic replay of the auxiliary direct-EMA rank-8-plus-diagonal recursion
from the retained initialization and immutable stream. Replay is curvature
reconstruction, not learner retraining, and must reproduce the original scalar
Fisher diagnostics before it can become a source for E9.13.

### Gate

Proceed only if every required vector and predictable curvature state can be
reconstructed or recomputed from immutable sources without learner retraining.
An operational fresh-batch fit may remain an oracle-assisted evaluation target;
it must not be introduced into a deployable controller.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_12_predictable_gain_contract__replica-0001__1a7d0f76ca07aefe`
replayed all 400 auxiliary Fisher transitions from the immutable Plan 8
parameters and streams without fitting a learner. The replay reproduced every
retained candidate trace exactly. Reconstructed operational fresh-fit norms
agreed with E9.9 to a maximum relative error of $7.71\times10^{-15}$.

The timing audit records two distinct summaries. $G_t^{\mathrm{old}}$ is frozen
before the transition batch and is the strictly predictable curvature used by
this study. The historical Plan 8 implementation observes that batch, performs
its direct-EMA Fisher update, and supplies the resulting **penalty Fisher** to
the EWC optimizer. Both representations are retained so this pipeline detail
is explicit rather than silently folded into the gain. E9.12 is **Supported**:
all vectors and curvature states were recovered with no learner retraining or
mutation of source artifacts.

## E9.13: Dual-Timescale Curvature Estimator

**Date opened:** 2026-09-07  
**Status:** Supported  
**Evidence level:** Numerical smoke  
**Depends on:** Supported E9.12

### Question

Can recent score batches provide a numerically stable, causal estimate of the
new-data curvature entering $\mathsf K_t^{\mathrm{gain}}$, while the existing
EWC summary represents the slower historical curvature?

### Design

1. Reproduce $G_t^{\mathrm{old}}$ by the deterministic auxiliary-process replay
   frozen in E9.12. Use the original initialization, batches, actions, direct
   EMA, rank budget, and diagonal policy exactly; validate against retained
   diagnostics and do not reconstruct it from a different history.
2. Build a separate fast EMA from per-observation score outer products. Screen
   half-lives $H_K\in\{4,8,16\}$ accepted updates, with $H_K=8$ primary. For
   $m=4$, their stationary weight-based supports are approximately 46, 92, and
   185 observations. These are sensitivity settings, not outcome-selected
   candidates. Initialize every fast filter from the paired high-quality
   initial Fisher. Evaluate each arriving batch at its original pre-update
   parameter, and incorporate it only after the gain for that transition has
   been frozen.
3. Use rank 8 plus a nonnegative residual diagonal, matching the practical
   Fisher representation supported by earlier experiments. Record numerical
   rank, discarded energy, diagonal floor, and representation memory.
4. Apply $\mathsf K_{t\mid t-1}^{\mathrm{gain}}(\pi_t)$ to the retained
   $\widetilde u_t$ by solving

   $$
   \left[(1-\pi_t)G_t^{\mathrm{old}}
   +\pi_tH_{t\mid t-1}^{\mathrm{new}}\right]x_t
   =\pi_tH_{t\mid t-1}^{\mathrm{new}}\widetilde u_t.
   $$

   Use a structured or matrix-free solve with explicit residual and iteration
   diagnostics. Do not form an inverse or pseudoinverse.
5. Preserve the filter through reversals. Report reversal windows separately;
   resetting from known schedule boundaries would leak experimental structure.
6. Add deterministic unit tests for EMA timing, effective support, structured
   products, solver residuals, scalar equal-curvature reduction
   $\mathsf K^{\mathrm{gain}}=\pi I$, and immutable artifact handling.
7. Run a tiny CPU float64 smoke before processing the authoritative paths.

### Numerical gate

Continue only if every primary solve is finite, its normalized residual is at
most $10^{-6}$, representation diagnostics remain valid, and no current-batch
information leaks into the predicted gain. Numerical regularization must be
frozen and reported; treatment-specific rescue tuning rejects the estimator.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_13_dual_timescale_gain__replica-0001__6fe2754caee9fc0e`
performed 1,200 strictly lagged structured solves in 3.54 seconds. The
$H_K=4,8,16$ filters have stationary weight-based supports of approximately
46.3, 92.4, and 184.7 observations at $m=4$. Every prediction was finite,
every filter respected the rank-8 limit, and the largest primary normalized
solve residual was $6.48\times10^{-7}$, below the frozen $10^{-6}$ gate.
Current-batch gradients updated only the next fast-Fisher state. E9.13 is
**Supported** as a numerical and causal implementation result.

## E9.14: Structured-Gain Calibration and Pivot Gate

**Date opened:** 2026-09-07  
**Status:** Rejected  
**Evidence level:** Oracle retrospective  
**Depends on:** Supported E9.13

### Question

Does the predictable rank-8-plus-diagonal gain explain realized EWC actuation
better than the scalar-affine model strongly enough to justify revisiting the
movement observer?

### Comparators and metrics

At every retained E9.9 transition compare:

1. the incumbent scalar prediction $\pi_t\widetilde u_t$;
2. the causal scalar-gain EMA frozen by E9.12;
3. the primary predictable structured prediction
   $\mathsf K_{t\mid t-1}^{\mathrm{gain}}(\pi_t)\widetilde u_t$;
4. the contemporaneous oracle scalar projection as an attributive ceiling,
   never as a deployable method.

Report mean and median Fisher residual-energy ratios relative to the incumbent,
Fisher cosine with $u_t$, norm calibration, solver diagnostics, half-life
sensitivity, and results inside versus outside reversal windows. Keep
predictive accuracy and NLL sealed: this study tests the observation model, not
a learner policy.

### Decision rule

- **Supported:** the primary structured gain lowers mean Fisher residual energy
  by at least 25% in at least three of four principal path groups, improves on
  the causal scalar baseline in at least three groups, remains directionally
  stable under neighboring half-lives, and passes the reversal and numerical
  checks.
- **Inconclusive:** it predicts ordinary stretches but loses calibration near
  reversals, or the primary and neighboring half-lives disagree materially.
- **Rejected:** it fails the 25% gate, requires noncausal curvature, or depends
  on unstable/treatment-specific solves.

A supported result authorizes only a reviewed amendment for a gain-aware
movement observer. An inconclusive or rejected result closes this estimator
family under the present low-data contract and releases the fixed-composition
pace-control hypothesis in `plan10.md` for formal planning.

### Execution record: 2026-09-07

Artifact
`rotated_mnist_plan9_e9_14_structured_gain_calibration__replica-0001__b97ed267c1bc27da`
scored all predictions with the independent high-sample rank-16 population
Fisher. The primary structured estimator passed the 25% reduction gate in zero
of four path groups and beat the causal scalar EMA in zero. No neighboring
half-life improved on the scalar-affine incumbent in any group. Primary
aggregate residual-energy ratios were approximately 11,436, 1,498, 69,124,
and 18,280 for single-linear, single-sigmoid, double-linear, and
double-sigmoid respectively.

This is broad miscalibration rather than a small number of reversals: primary
median stepwise ratios ranged from roughly 146 to 8,244, and the largest
reversal-window aggregate ratio exceeded $10^5$. The directional solve is
numerically accurate, but weakly identified directions in two compressed,
differently smoothed curvature summaries produce enormous parameter-space
actions. E9.14 is **Rejected** under the frozen low-data contract. This does not
reject the exact local quadratic identity with known nonsingular curvatures;
it rejects this practical estimator of that identity and releases Plan 10 for
formal planning. Predictive accuracy and NLL remained sealed.

The artifact-only notebook `rotated_mnist/markov_gain_results.ipynb` presents
the numerical contract, primary comparisons, and gate interpretation.

## Extension Register

Append future entries here using the following template:

```markdown
## E9.n: Short Study Name

**Date opened:** YYYY-MM-DD
**Status:** Proposed
**Evidence level:** ...
**Depends on:** ...

### Question
### Frozen hypothesis and estimand
### Sources and design
### Metrics and decision rule
### Execution record
### Observation
### Interpretation
### Decision and next action
```

Potential extensions should be motivated by an earlier result. Candidate
topics include explicit EWC-remainder measurement, robust innovation filters,
change-point-aware half-lives, overlapping/replay batch covariance, and
stochastic environmental drift. Their presence here is not approval to run
them.

## Evidence for the Next Pivot

The cumulative evidence makes fixed-composition pace control a warranted next
pivot and no longer supports another direct estimator of the movement premium
under the same four-observation, single-trajectory contract:

1. The direct LFU identity remained valid, but its local third-moment and HVP
   estimator was noisier than a direct Fisher EMA in every tested cell and
   accumulated damaging indefinite corrections.
2. EDR and the E9.2 smooth-drift filters reduced some observation variance but
   did not recover a calibrated population movement premium or outperform the
   fixed-composition controls prospectively.
3. E9.7 found mean population action opportunity of only `.00252`--`.00282`,
   with no path sustaining even a `.01` lift for eight transitions. The much
   larger conditional quantity depended on unavailable realized anchor error.
4. E9.8's lag-separated cross-moment improved signed movement-energy MAE in
   zero of four groups. Large alternating residual cross-moments remained
   beyond the excluded adjacent lag.
5. E9.9 attributed most of that residual energy to failure of the scalar
   affine EWC observation model in three of four groups. It also exposed one
   previously untested alternative: estimate the predictable gain from pooled
   second-order curvature rather than an instantaneous LFU derivative.
6. E9.10 increased theoretical opportunity by accelerating the path, but no
   candidate sustained the required opportunity while preserving the local
   common-drift approximation.

This is not an impossibility theorem for adaptive composition. E9.12--E9.14
give the pooled-curvature gain model one final, predeclared gate because it uses
the same lower-order EMA machinery that succeeded for Fisher tracking. If that
gate fails, the stopping decision applies to the present estimator family under
the intended data and compute constraints. Plan 10 then fixes an interpretable
composition action and inverts the local risk relation to study a controllable
movement budget. That redesign changes the control variable and sampling law;
it requires a fresh mathematical contract before implementation.
