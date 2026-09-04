# Implementation Plan 7: Anchor-Aware Coefficient Audit

Plan 7 follows the completed instantaneous-oracle study in
[plan6.md](plan6.md). It does not rerun or revise that experiment. It uses the
stored Plan 5 learner trajectories and Plan 6 reference quantities to answer
one narrower question:

> Does the apparent upward bias in EDR come from effective-information
> feedback, systematic learner-anchor error, or excess online trend energy?

The audit is artifact-only. It must not train a model, calculate scores, update
a Fisher matrix, download data, or alter a completed artifact.

## Motivation

Plan 6 compared an online trend statistic with the centered marginal target

$$
A_t^{\mathrm{marg}}
=\|d\theta_t^\star\|_{\mathcal I_t}^2+q_tD_t,
\qquad
D_t=\operatorname{tr}(\mathcal I_tK_t).
$$

That target assumes the current learner error
$e_t=\widehat\theta_t-\theta_t^\star$ is centered across repeated learner
trajectories. The observed normalized displacement instead satisfies the local
fixed-batch approximation

$$
\frac{u_t}{\pi_t}
\approx d\theta_t^\star+\epsilon_{t+1}-e_t.
$$

Conditionally on the realized anchor, its mean is therefore
$d\theta_t^\star-e_t$. If the trend estimate follows that conditional
catch-up direction, adding $q_tD_t$ to its squared norm may count old-anchor
uncertainty twice. If the trend estimate averages anchor error away and targets
population motion, the marginal term remains appropriate. The current
artifacts can distinguish much of this ambiguity without another compute run.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Estimand and artifact contracts | Complete |
| 1 | Coefficient decomposition implementation | Complete |
| 2 | Artifact-only audit execution | Complete |
| 3 | Audit notebook and evidence classification | Complete |
| 4 | Mathematical reintegration | Complete |

## Scientific Contract

### Weight concentration and covariance calibration

Let $w_{t,i}$ be the normalized historical observation weights represented by
the old summary and define

$$
q_t:=\sum_i w_{t,i}^2,
\qquad
N_{\mathrm{eff},t}:=q_t^{-1}.
$$

Replacing every old weight by $(1-\pi_t)w_{t,i}$ and assigning each of the
$m_t$ new observations weight $\pi_t/m_t$ gives the exact scalar identity

$$
q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m_t}.
$$

Giving $q_t$ a covariance interpretation requires a separate **local
calibrated-summary assumption**. Across repeated learner trajectories sharing
the same local population state, assume

$$
\mathbb E[e_t\mid\mathcal E_t]=\mu_t,
\qquad
\operatorname{Cov}(e_t\mid\mathcal E_t)
=q_tK_t+o(q_t),
$$

and for a fresh local estimator based on $m_t$ observations,

$$
\mathbb E[\epsilon_{t+1}\mid\mathcal E_t]=0,
\qquad
\operatorname{Cov}(\epsilon_{t+1}\mid\mathcal E_t)
=\frac{K_{t+1}}{m_t}+o(m_t^{-1}).
$$

Centered old and new errors are conditionally uncorrelated. The efficient-MLE
special case is $K_t=\mathcal I_t^{-1}$; the experiment need not assume that
special case when it directly estimates $K_t$. Local constancy of covariance
shape is required before the scalar $q_t$ recursion can stand in for the full
covariance recursion.

Here $\mathcal E_t$ fixes the environment and experimental design but not the
realized learner. Under the operational history $\mathcal F_t$, the current
anchor and $e_t$ are fixed oracle quantities rather than random centered
errors. Keeping these conditioning statements separate is mandatory.

### Three related targets

Using the same Fisher metric and
$B_t=\operatorname{tr}(\mathcal I_tK_{t+1})/m_t$, define:

1. The covariance-only recommendation

   $$
   \pi_t^{\mathrm{cov}}
   =\frac{q_tD_t}{q_tD_t+B_t}.
   $$

2. The centered marginal recommendation used by Plan 6

   $$
   \pi_t^{\mathrm{marg}}
   =\frac{\|d\theta_t^\star\|_{\mathcal I_t}^2+q_tD_t}
   {\|d\theta_t^\star\|_{\mathcal I_t}^2+q_tD_t+B_t}.
   $$

3. The realized-anchor conditional recommendation

   $$
   \delta_t^\star
   :=d\theta_t^\star-e_t
   =\theta_{t+1}^\star-\widehat\theta_t,
   \qquad
   \pi_t^{\mathrm{cond}}
   =\frac{\|\delta_t^\star\|_{\mathcal I_t}^2}
   {\|\delta_t^\star\|_{\mathcal I_t}^2+B_t}.
   $$

The conditional target omits $q_tD_t$ because $e_t$ is fixed after
conditioning on the realized anchor. The marginal target with nonzero mean
anchor error is instead

$$
A_t^{\mathrm{marg}}
=\|d\theta_t^\star-\mu_t\|_{\mathcal I_t}^2+q_tD_t.
$$

The online quantity $\widehat d_t^TG_t\widehat d_t$ is not a fourth population
objective. It is a predictable plug-in statistic whose interpretation depends
on whether $\widehat d_t$ tracks population motion $d\theta_t^\star$ or the
realized catch-up direction $\delta_t^\star$.

### Descriptive boundary

One stored trajectory reveals realized anchor error but cannot identify the
replica mean $\mu_t$ or its covariance independently. Plan 7 may diagnose an
estimand mismatch, but it must not claim to estimate the bias distribution or
prove that either local recommendation improves closed-loop predictive loss.

## Artifact Contract

1. Read the completed Plan 5 single-lap source artifact and completed Plan 6
   debias and oracle artifacts without mutation.
2. Validate source run ID, parameter layout, initial-state hashes, schedule
   hashes, transition indexing, Fisher representation, and parameter dtype.
3. Write a new immutable audit artifact beneath
   `cache/mnist_experiment/rotated_mnist/phase7/coefficient_audit/`.
4. Record exact input paths, content hashes, git state, schema versions, and all
   decomposition identities needed to reproduce each scalar.
5. Keep the notebook artifact-only and fail clearly on missing, incomplete, or
   incompatible inputs.
6. Use no Fisher inverse, pseudoinverse, new optimization, score calculation,
   or candidate-$\pi$ grid.

## Phase 0: Estimand And Artifact Contracts

### Goal

Freeze the conditioning and comparison rules before calculating anchor error.

### Scope

1. Adopt $q_t$ as the pre-transition weight concentration for the current
   anchor, with $q_{t+1}$ produced by action $\pi_t$.
2. Freeze the centered marginal, realized conditional, and covariance-only
   recommendations above.
3. Confirm that Plan 5 trajectory parameters and Plan 6 reference parameters
   share architecture, parameter ordering, initialization ancestry, and the
   same $t\to t+1$ angle schedule.
4. Determine whether fixed-$\pi=.025$ and fixed-$\pi=.05$ trajectories share
   enough of that contract to serve as matched anchor-error diagnostics. Keep
   EDR as the primary target.
5. Record the extra high-sample optimization performed by the reference branch
   so initial reference displacement cannot be mistaken for policy-induced
   lag.

### Check-in

Confirm that parameter-space anchor comparisons are meaningful on the shared
branch. Stop before Phase 1 if parameter identities or initialization ancestry
cannot be validated.

### Execution Record

Phase 0 is complete. All four Plan 5 conditions and both schedules share
initializer state hash
`8a2fe31b3e1a2dec7e53316ea3c465fad699307e64b10bdaeca9228ae9aae1ff`,
the same 512-parameter layout, 81 parameter states, and the same schedule
hashes. The Plan 6 source and reference contracts identify that exact Plan 5
run, so the learner and reference parameters occupy the same fixed chart and
share initialization ancestry. The fixed `.025` and `.05` trajectories are
therefore admissible matched diagnostics.

The gate passes with a named limitation. Before following the angle path, the
reference branch performed 25 additional selected epochs at $0^\circ$ and
improved validation NLL from `.4525` to `.3100`. Consequently, realized anchor
distance is mathematically well-defined but includes a pre-policy
reference-optimization offset. Phase 2 must report that offset separately and
must not attribute all anchor energy to continual-learning lag.

## Phase 1: Coefficient Decomposition Implementation

### Goal

Implement an exact artifact-only decomposition of the available local risks.

### Scope

For every compatible transition, calculate with the Plan 6 pre-step Fisher:

$$
S_t^{\mathrm{pop}}
=(d\theta_t^\star)^T\mathcal I_td\theta_t^\star,
$$

$$
E_t^{\mathrm{anchor}}=e_t^T\mathcal I_te_t,
\qquad
X_t^{\mathrm{cross}}=-2e_t^T\mathcal I_td\theta_t^\star,
$$

and verify numerically that

$$
\|d\theta_t^\star-e_t\|_{\mathcal I_t}^2
=S_t^{\mathrm{pop}}+E_t^{\mathrm{anchor}}+X_t^{\mathrm{cross}}.
$$

Also record:

- $q_tD_t$, $B_t$, and the three recommendations in the scientific contract;
- realized EDR, instantaneous online, and debiased online recommendations;
- population, anchor, cross, and online signal contributions normalized by the
  corresponding covariance scale;
- the stationary-weight diagnostic
  $q_\infty(c)=c/[m(2-c)]$ and its covariance-only consequence
  $\pi^{\mathrm{cov}}=c/2$;
- rank-8 versus rank-16 sensitivity where both are already available.

Do not compare absolute quadratic scales across different Fisher
representations without also reporting their within-representation normalized
ratios.

### Verification

- Synthetic weighted observations recover the $q_t$ recursion and Kish size.
- Synthetic centered anchors recover the marginal risk from conditional risks.
- Nonzero anchor means recover
  $\|d\theta-\mu\|_{\mathcal I}^2+qD$.
- Conditional quadratic decomposition closes to numerical tolerance.
- Common positive rescaling of the Fisher leaves every recommendation
  unchanged.
- Artifact indexing uses only the pre-transition state and next population
  target.

### Check-in

Review the implementation and one-transition hand calculation before producing
the complete audit artifact.

### Execution Record

Phase 1 is complete. Strict configuration and artifact contracts live in
`phase7_config.py` and `phase7_artifacts.py`; the pure quadratic and weight
algebra lives in `phase7_audit.py`; and `run_phase7_audit.py` performs only
validated artifact loading and CPU float64 arithmetic. Eight focused tests
cover the exact normalized-weight recursion, Kish size, the stationary $c/2$
identity, population/anchor/cross decomposition, centered and noncentered
marginal risks, strict configuration, and common Fisher-scale invariance.

The implementation preserves both raw and finite-reference-corrected signals.
It also reports the regular efficient-MLE sensitivity
$\operatorname{tr}(\mathcal I\mathcal I^{-1})=p$ separately from the measured
local-fit covariance shape; it never relabels that sensitivity as an empirical
estimate.

## Phase 2: Artifact-Only Audit Execution

### Goal

Measure which term explains the apparent high recommendation on the stored
linear and sigmoid paths.

### Scope

1. Execute the decomposition over all 160 Plan 6 transitions.
2. Keep cold-start and live-controller intervals separate.
3. Compare EDR with the compatible fixed policies to determine whether anchor
   energy is path-wide, policy-specific, or already present at initialization.
4. Report how much of each recommendation comes from covariance concentration,
   population movement, realized anchor displacement, and excess online trend
   energy.
5. Preserve reversal and local-speed annotations without treating path points
   as independent replicas.

### Gate

- **Centered marginal supported:** realized-anchor terms fluctuate around the
  centered marginal scale and online excess remains after accounting for them.
- **Conditional target supported:** realized-anchor risk explains the online
  signal materially better than population movement plus $q_tD_t$.
- **Mixed:** initialization mismatch, branch movement, and estimator error
  cannot be separated with the stored trajectory.
- **Stop:** parameter comparisons fail the shared-branch contract.

### Check-in

Decide whether the next intervention belongs in the online estimator, the
reference construction, the controller estimand, or nowhere until independent
replicas exist.

### Execution Record

Phase 2 is complete. The authoritative immutable artifact is
`cache/mnist_experiment/rotated_mnist/phase7/coefficient_audit/rotated_mnist_phase7_anchor_audit_primary_v3__replica-0001__4406a8e7634c0ad8`.
It contains 960 rows: 80 transitions, two schedules, three matched policies,
and ranks 8 and 16. Two earlier completed development artifacts remain
immutable and are superseded by this explicit efficient-MLE sensitivity.

The gate is **Mixed**. For rank 16, the EDR covariance-only recommendation is
`.135` on linear and `.143` on sigmoid, within MAE `.0094` and `.0120` of half
the realized action. Adding population movement changes these to only `.137`
and `.146`. The online instantaneous recommendations are `.262` and `.298`,
while debiasing lowers them only to `.244` and `.274`.

Realized anchor energy is not negligible: its live mean is `162.6` on linear
and `93.0` on sigmoid, compared with population movement energies `.0134` and
`.0142`. The empirical local-fit covariance shape is only about `15.8`, or
`3.1%` of the regular $p=512$ efficient-MLE benchmark. It consequently makes
the raw conditional oracle extremely aggressive (`.926` and `.890`). Replacing
only that covariance scale by the declared efficient-MLE sensitivity reduces
the conditional means to `.495` and `.318`; this is informative but not an
estimate, because regular efficiency is itself unverified for the deep net.

The matched controls show that anchor energy grows with the realized learning
policy rather than coming only from the initial reference offset. Rank-16 live
means on linear are `17.2` for fixed `.025`, `34.3` for fixed `.05`, and
`162.6` for EDR; sigmoid gives `18.2`, `38.0`, and `93.0`. Nonetheless, the
reference branch's additional zero-degree optimization and the locally
underdispersed covariance estimate prevent the stored paths from selecting a
clean conditional or marginal controller target. The evidence identifies both
state feedback and model-assumption failure, not a scalar correction to
$\pi_t$.

## Phase 3: Audit Notebook And Evidence Classification

### Goal

Make the decomposition visually legible without overstating one-trajectory
evidence.

### Scope

Create
`mnist_experiment/rotated_mnist/anchor_coefficient_audit.ipynb` with:

1. a concise derivation of $q_t$ from observation weights;
2. separate linear and sigmoid panels for covariance-only, centered marginal,
   realized conditional, and online recommendations, with no more than four
   conditions per panel;
3. a population/anchor/cross energy decomposition;
4. a view of anchor energy across EDR and compatible fixed-policy controls;
5. a compact table quantifying how much current debiasing removes;
6. explicit reference-branch and single-trajectory limitations;
7. a final classification using the Phase 2 gate language.

Every cell must load completed artifacts and perform lightweight arithmetic or
plotting only.

### Check-in

Confirm that every claimed discrepancy is visible in a plot and traceable to a
stored scalar before changing theory or controller code.

### Execution Record

Phase 3 is complete. The artifact-only analysis is in
`mnist_experiment/rotated_mnist/anchor_coefficient_audit.ipynb`. Its seven code
cells load only the authoritative completed Phase 7 artifact, perform
lightweight tabular arithmetic, and render four focused figures. A clean
headless execution reproduced all tables and figures without training,
optimization, score evaluation, or artifact repair.

The notebook makes each part of the **Mixed** gate classification visible:

- the tracked-$q_t$, equal-shape recommendation reproduces the population
  covariance oracle to mean absolute error below `.001` on both schedules,
  while the stationary $c_t/2$ limit explains the factor-of-two pattern;
- population movement barely separates the centered marginal curve from the
  covariance-only curve;
- measured-$K$ and regular-efficient conditional sensitivities disagree
  sharply, exposing unresolved covariance calibration;
- the population/anchor/cross decomposition shows anchor error overwhelming
  population step energy; and
- matched fixed-policy trajectories show that anchor energy is policy
  dependent, while the recorded zero-degree reference optimization prevents
  interpreting it as pure online lag.

Every oracle input is stored directly in `coefficient_rows.json`; the notebook
adds only the deterministic scalar transformation
$m_tq_t/(1+m_tq_t)$. Each headline artifact mean is reproduced in
`audit_summary.json`. The covariance-only subproblem therefore has both a
population-risk derivation and strong artifact reproduction. At the Phase 3
check-in, this did not yet select a complete controller target because movement
and anchor conditioning remained unresolved; mathematical reintegration was
deferred to Phase 4 for user review.

## Phase 4: Mathematical Reintegration

### Goal

Integrate the covariance-only population estimand and its matrix-free scalar
realization into the applied fixed-batch EWC model. Preserve the distinct
Bernoulli-composition experiment and avoid promoting the result into a complete
closed-loop controller claim.

### Scope

1. Keep the theoretical Bernoulli-composition model and its linear-in-$\pi$
   variance law unchanged. State explicitly that the new result belongs to the
   applied fixed-batch model, whose weighted fresh-batch variance is quadratic
   in $\pi$.
2. Define $q_t$ as the pre-transition concentration of normalized historical
   weights and use the unambiguous timing

   $$
   q_{t+1}=(1-\pi_t)^2q_t+\frac{\pi_t^2}{m_t}.
   $$

   Separate this exact weight identity from the calibrated-summary assumption
   that gives it a parameter-covariance interpretation.
3. Present the fixed-batch centered marginal and realized-anchor conditional
   risks with explicit conditioning. Correct language that calls a risk
   conditional while integrating over random old-anchor error.
4. Define the covariance-only population estimand and derive

   $$
   \pi_t^{\mathrm{cov}}
   =\frac{q_tD_t^{\mathrm{old}}}
   {q_tD_t^{\mathrm{old}}+D_t^{\mathrm{new}}/m_t}.
   $$

   Under locally matching covariance shapes, derive the matrix-free
   recommendation $m_tq_t/(1+m_tq_t)$ and, for constant action $c$ and batch
   size $m$, its stationary limit $c/2$.
5. Quantify the small-movement approximation exactly by displaying the
   nonnegative gap between the complete marginal and covariance-only
   recommendations. Reparameterize the equal-shape rule as a known covariance
   baseline plus a normalized movement premium.
6. In Appendix B, present the population oracle, tracked-$q_t$ recommendation,
   and stationary $c/2$ approximation as distinct levels. Clarify that $q_t$ is
   calculated from known weights, while covariance calibration and equal shape
   are statistical assumptions. Warn that the update $c_{t+1}=c_t/2$ is not a
   controller and would collapse toward zero.
7. Reframe the existing plug-in and EDR construction as estimating the movement
   contribution beyond the covariance-only baseline. Preserve the historical
   algorithm and artifact names, and do not silently reinterpret its realized
   trend as population movement.
8. Record the Plan 7 evidence: tracked $q_t$ reproduces the stored covariance
   oracle with MAE below `.001` on both schedules; $c/2$ is a useful but less
   accurate stationary approximation; and movement contributes little on this
   one MNIST trajectory.
9. Change no deployed controller code or completed artifact. Any decomposed
   controller using the new baseline remains a separately named future
   treatment requiring validation.

### Final Check-in

Review whether the revised theory cleanly establishes the covariance-only
subproblem while leaving movement and realized-anchor conditioning open. Decide
separately whether a future experiment should combine the tracked-$q_t$
baseline with a new estimator of the normalized movement premium.

### Execution Record

Phase 4 is complete. `mathematical_overview.ipynb` now places the new result
only in the applied fixed-batch EWC model and explicitly preserves the
theoretical Bernoulli-composition model's different, linear-in-$\pi$ variance
law. The main text now:

- defines $q_t$ as the pre-transition concentration of normalized historical
  weights and makes $q_{t+1}$ the result of action $\pi_t$;
- separates the exact weight identity from a local asymptotically linear
  covariance-calibration assumption;
- distinguishes marginal repeated-learner risk from realized-anchor
  conditional risk;
- derives the population covariance-only estimand, the matrix-free
  $m_tq_t/(1+m_tq_t)$ recommendation, and its stationary $c/2$ limit; and
- writes the complete equal-shape marginal rule as a known covariance state
  $m_tq_t$ plus an unknown normalized movement premium $\rho_t$.

Appendix A now identifies $K_t=\mathcal I_t^{-1}$ as the efficient special
case and recovers the scalar covariance-only recommendation from the
fixed-batch LAN construction. Appendix B presents the population, tracked-$q_t$,
and stationary recommendation levels separately. It retains historical EDR
unchanged, explains that its difficult contribution is movement and anchor
behavior, and marks smoothing only $\rho_t$ as an untested future treatment.
The Plan 7 empirical errors are recorded in the experimental-status section.

No controller implementation or completed artifact changed. Strict JSON
loading and delimiter checks passed for all seven notebook cells; symbolic
checks reproduced the risk minimizer, the exact small-movement gap, and the
stationary $c/2$ identity. The final check-in therefore accepts the
covariance-only estimation paradigm while leaving a decomposed adaptive
controller for a separately planned experiment.

A readability follow-up moved the canonical $q_t$ definition out of the
single-observation LFU discussion and into **Applied fixed-batch EWC model**.
That section now begins with a compact notation table and an explicit
pre-decision $q_t\to q_{t+1}$ timeline. It defines $c$ at the same point as the
constant historical action that generates $q_\infty(c)$, not as another
controller parameter. Appendix B now references these definitions instead of
reintroducing them.
