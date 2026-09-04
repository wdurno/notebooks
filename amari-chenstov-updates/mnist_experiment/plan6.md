# Implementation Plan 6: Instantaneous Oracle Calibration for EDR

Plan 6 is a detachable diagnostic follow-up to the stopped Plan 5 EDR
challenge. It does not reopen or rewrite that result. Its purpose is to answer
the unresolved question exposed by the realized EDR trajectories:

> Is EDR primarily failing because its online coefficients are noisy or
> biased, because exponential discounting lags a valid instantaneous target,
> or because the assumed local risk surrogate is itself inaccurate?

The primary object is the instantaneous local oracle from the applied
fixed-batch theory. An empirical grid search over composition weights is not
the oracle in this plan. Such a search could later validate the surrogate, but
it cannot define the theoretical target.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Estimand, indexing, and artifact contracts | Complete |
| 1 | Artifact-only trend-variance audit | Complete |
| 2 | Local-oracle estimator implementation | Complete |
| 3 | Frozen-state convergence pilot | Complete |
| 4 | Full instantaneous-oracle path | Complete |
| 5 | Tracking analysis and next-regime decision | Complete |

## Scientific Contract

### Target estimand

For the transition from environment state $t$ to $t+1$, retain the applied
fixed-batch risk

$$
R_t(\pi)=(1-\pi)^2A_t^\star+\pi^2B_t^\star,
$$

$$
A_t^\star=S_t^\star+V_{\mathrm{old},t}^\star,
\qquad
B_t^\star=V_{\mathrm{new},t}^\star,
$$

with

$$
S_t^\star
=(d\theta_t^\star)^T\mathcal I_t d\theta_t^\star,
\qquad
d\theta_t^\star=\theta_{t+1}^\star-\theta_t^\star.
$$

The target is

$$
\boxed{
\pi_t^\star
=\frac{A_t^\star}{A_t^\star+B_t^\star}.
}
$$

This is the exact minimizer of the assumed conditional one-step surrogate. It
is not a globally optimal closed-loop policy.

### Covariance realization

Let $M$ denote an oracle-only local MLE sample size and define

$$
K_t:=M\operatorname{Cov}(\widehat\theta_{t,M}).
$$

Keep the deployed batch size fixed at $m=4$. Under the calibrated local model,

$$
V_{\mathrm{old},t}^\star
=q_t\operatorname{tr}(\mathcal I_tK_t),
\qquad
V_{\mathrm{new},t}^\star
=\frac{1}{m}\operatorname{tr}(\mathcal I_tK_{t+1}).
$$

Increasing $M$ improves oracle precision; it must never replace $m$ in the
target formula. The principal target conditions on the $q_t$ realized by the
stored EDR trajectory so that the oracle and applied action describe the same
pre-step state. A recursively actuated oracle would be a different experiment.

The covariance matrix need not be materialized. For $B$ independent local MLE
fits and
$\delta_t^{(b)}=\widehat\theta_{t,M}^{(b)}-\overline\theta_{t,M}$, estimate

$$
\operatorname{tr}(\mathcal I_tK_t)
\approx
\frac{M}{B-1}\sum_{b=1}^B
(\delta_t^{(b)})^T\widehat{\mathcal I}_t\delta_t^{(b)}.
$$

Use streaming score outer products and a rank-plus-diagonal Fisher
representation. Never invert or pseudoinvert a Fisher matrix.

### Temporal indexing

Every displayed $\pi_t$ must weight the batch and transition from $t$ to
$t+1$. The realized EDR action, online debiased recommendation, oracle
coefficients, and oracle recommendation must be aligned before plotting.
Oracle quantities may use future truth offline; the deployed EDR decision must
remain predictable and use no current-batch outcome.

### Primary visual comparison

The controller-tracking plot compares only the realized EDR action with the
instantaneous oracle $\widehat\pi_t^\star$ and its uncertainty band. The raw
historical `plugin_pi` remains available for debugging but is not a scientific
target and does not appear in the primary plot.

A separate estimator-validation panel compares the debiased instantaneous
online recommendation with the same instantaneous oracle. An oracle-smoothed
sequence may be calculated privately to attribute smoothing lag, but it is a
secondary diagnostic rather than the target.

## Isolation And Artifact Contract

1. Keep Plan 6 code, configurations, tests, and notebooks under
   `mnist_experiment/rotated_mnist/` and
   `test/unit/rotated_mnist/` using Plan 6-specific names.
2. Reuse completed Plan 5 artifacts read-only. Do not relabel, migrate, or
   mutate them.
3. Write every pilot and full oracle run as a new immutable artifact beneath
   `cache/mnist_experiment/rotated_mnist/phase6/`.
4. Cache reference fits, score-based Fisher representations, covariance-risk
   replicates, and convergence diagnostics by angle so the linear and sigmoid
   schedules can share identical local estimates.
5. Record oracle sample identities, fit seeds, score seeds, optimizer
   diagnostics, ranks, dtypes, devices, and stopping decisions.
6. Keep notebooks artifact-only. They must never fit models, estimate Fishers,
   or repair incomplete artifacts.
7. Do not amend `mathematical_overview.ipynb` or promote EDR in
   `mnist-findings.ipynb` until the final gate supports a precise statement.
8. Preserve `single_lap_edr_results.ipynb` as the Plan 5 narrative. Publish
   Plan 6 results separately in `instantaneous_oracle_results.ipynb` so the
   exploratory oracle analysis can be revised or retired without changing the
   completed closed-loop account.

## Phase 0: Estimand, Indexing, And Artifact Contracts

### Goal

Freeze what the oracle means before introducing an estimator.

### Scope

1. Audit the Plan 5 single-lap artifact for every scalar needed to reconstruct
   $q_t$, trend gains, signal energy, uncertainty scale, and realized EDR.
2. Freeze the authoritative linear and normalized-logistic paths from
   `rotated_mnist_phase5_slow_single_lap_development__replica-0001__30f2161a06eb8d50`.
3. Define the reference-angle union shared across those paths and the exact
   $t\to t+1$ indexing contract.
4. Specify schemas for reference fits, Fisher representations, local-MLE
   replicates, scalar oracle coefficients, uncertainty, and completion.
5. Add configuration validation that distinguishes $M$, $B$, and deployed
   $m$ and forbids a candidate-$\pi$ grid as the primary oracle.

### Check-in

Confirm that the conditional one-step oracle on the realized EDR state is the
desired target before implementing local MLE estimation.

### Execution Record

Phase 0 is complete. The authoritative Plan 5 source contains all 81 parameter
vectors and 80 displacements for each condition and schedule, the complete
transformed streams, final model/controller states, the initial Fisher, and
the predictable scalar controller state at every transition. It does not
contain per-step Fisher representations or intermediate model state dicts;
Plan 6 may reconstruct a model from each stored parameter vector, while any
new score calculation remains explicit compute rather than artifact-only
analysis.

After canonicalizing roundoff-equivalent reverse-leg angles, the linear and
sigmoid paths each contain 41 distinct angles and their union contains 79.
Only $0^\circ$, $15^\circ$, and $30^\circ$ are shared. The source condition,
deployed $m=4$, schedule names, source run ID, and $t\to t+1$ indexing are now
strict configuration fields. Candidate-$\pi$ optimization is absent from the
oracle contract.

## Phase 1: Artifact-Only Trend-Variance Audit

### Goal

Determine whether an inexpensive correction can plausibly explain the
excessive online recommendations before spending compute on reference fits.

### Scope

1. Reconstruct the approximate Fisher-energy variance recursion

   $$
   v_{t+1}
   =(1-\gamma_t)^2v_t
   +\gamma_t^2(q_t+m^{-1})\widehat D_t.
   $$

2. Form

   $$
   \widehat S_t^{\mathrm{db}}
   =\max(0,\widehat d_t^TG_t\widehat d_t-v_t)
   $$

   and its implied instantaneous recommendation.
3. Preserve both unclipped and clipped values, correction magnitude, boundary
   occupancy, and sensitivity to locally plausible variance scaling.
4. State explicitly that changing $G_t$, trend curvature, and contamination of
   $\widehat D_t$ make this an assumption check rather than a proven correction.
5. Add unit tests for zero-noise, stationary-noise, variable-gain, and
   nonnegative-boundary cases.

### Gate

- **Continue:** the reconstruction is finite and identifies a measurable,
  interpretable bias range.
- **Continue with caution:** the correction is small, leaving lag or oracle
  mismatch as the main possibilities.
- **Stop and redesign:** the available artifact cannot support even the stated
  approximation without silently inventing missing state.

### Check-in

Review whether the cheap correction changes the likely explanation and
whether the frozen-state pilot remains worth its cost.

### Execution Record

Phase 1 is complete. The immutable artifact is
`cache/mnist_experiment/rotated_mnist/phase6/debias/rotated_mnist_phase6_debias_primary__replica-0001__e1c9d975a0351e2e`.
It reconstructs the primary variance correction and `.5`/`2.0` scaling
sensitivities from Plan 5 artifacts without training or score calculation.

The gate is **continue with caution**. At the primary scale, correction removes
an average capped `.310` of recorded signal energy on linear and `.524` on
sigmoid, zeroing `.097` and `.389` of live signal estimates respectively.
Nevertheless, mean instantaneous recommendations only fall from the realized
EDR means `.277` to `.244` on linear and `.300` to `.274` on sigmoid. Doubling
the correction still leaves maxima `.497` and `.801`. Trend-quadratic variance
is material but cannot by itself explain the high-action regime. The named
limits remain changing Fisher geometry, trend curvature, and possible trend
error inside the residual uncertainty estimate.

## Phase 2: Local-Oracle Estimator Implementation

### Goal

Implement the theoretical matrix-term estimator independently of the online
controller.

### Scope

1. Implement high-sample, continuation-based reference fits at requested
   angles while retaining one local parameter branch.
2. Estimate each population Fisher from an independent score stream and
   compress it to rank 8 plus diagonal without constructing a dense
   parameter-by-parameter matrix.
3. Implement repeated local MLE fits and direct Monte Carlo estimation of
   $\operatorname{tr}(\widehat{\mathcal I}_tK_t)$ through Fisher quadratic
   products.
4. Estimate $S_t^\star$ from adjacent reference fits and record its
   finite-reference noise correction. Preserve the uncorrected value as a
   diagnostic.
   Use observation- and optimizer-paired local fits at adjacent angles to
   estimate $M\operatorname{Cov}(\widehat\theta_{t+1,M}-
   \widehat\theta_{t,M})$ directly; do not erase the cross-covariance by
   treating the two reference fits as independent.
5. Propagate uncertainty in signal and covariance risk into
   $\widehat\pi_t^\star$ using independent replicate or block-bootstrap units,
   never trajectory steps as independent observations.
6. Add checkpointable, immutable entry points and a tiny CPU smoke.

### Verification

- A regular synthetic likelihood with known Fisher and MLE covariance recovers
  its analytic $S_t$, covariance risks, and $\pi_t^\star$.
- Scaling $M\operatorname{Cov}(\widehat\theta_M)$ is stable across a small
  sample-size ladder in the synthetic test.
- Parameter ordering matches the stored model and Fisher representation.
- Rank-plus-diagonal trace products agree with dense calculations in a tiny
  model.
- No Fisher inverse, pseudoinverse, or candidate-$\pi$ optimization appears in
  the oracle path.

### Check-in

Review the estimator and smoke artifact before authorizing repeated GPU fits.

### Execution Record

The matrix-free estimator, strict configurations, checkpointable immutable
runner, and CPU smoke are implemented. Local-MLE replicate $b$ reuses the
same sampled observation identities and loader seed at every angle. The
finite-reference correction therefore estimates
$M\operatorname{Cov}(\widehat\theta_{t+1,M}-\widehat\theta_{t,M})$ from paired
displacements, retaining the adjacent-fit cross-covariance instead of assuming
independence. Bootstrap units preserve that pairing and include a scaled
reference-displacement perturbation.

The current immutable smoke artifact is
`cache/mnist_experiment/rotated_mnist/phase6/oracle/rotated_mnist_phase6_oracle_smoke__replica-0001__a431dc81287c974a`.
It exercised source validation, rotated fitting, score collection, matrix-free
Lanczos compression, paired local fits, uncertainty propagation, and atomic
completion. Its intentionally tiny $B=2$ estimate failed the precision gate,
as expected. Focused algebraic and Gaussian-location tests verify paired-noise
cancellation, dense quadratic agreement, the analytic risk minimizer, and
stability of $M\operatorname{Cov}(\widehat\theta_M)$ across $M$.

## Phase 3: Frozen-State Convergence Pilot

### Goal

Measure estimator cost, variance, local-branch stability, and rank sensitivity
at a few scientifically distinct states.

### Pilot Revision

The first pilot was stopped after 15 of 79 reference angles. Validation NLL
fell from `.399` at $0^\circ$ to `.267` by roughly $2^\circ$, showing that the
six-epoch continuation budget was still completing the underfit initializer
rather than isolating angular movement. Its incomplete artifact is retained at
`cache/mnist_experiment/rotated_mnist/phase6/oracle/.incomplete/rotated_mnist_phase6_oracle_frozen_state_pilot__replica-0001__8699ca539b29b781`.

The revised pilot gives only the first canonical reference angle an 80-epoch
ceiling with validation patience. Subsequent angles retain the six-epoch
continuation ceiling. Per-local-fit NLL, accuracy, predictive KL from the
reference model, and parameter displacement now make branch failures visible.

### Scope

1. Select frozen states covering cold-start release, mid-ramp movement,
   approach to $30^\circ$, reversal, and mid-return on both schedule shapes.
2. Use nested oracle sample sizes and local-fit replicate counts so added
   compute extends rather than replaces earlier estimates.
3. Start with ranks 8 and 16. Treat their difference as representation
   sensitivity, not independent replication.
4. Monitor local-MLE optimizer diagnostics, parameter displacement, predictive
   KL, and label metrics to detect failed fits or departures from the intended
   local branch.
5. Stop an estimate when its six-standard-error half-width for
   $\widehat\pi_t^\star$ is at most `.01`, subject to a fixed maximum of 64
   valid local fits.
6. Produce a conservative full-path time and storage projection.

### Gate

- **Go:** representative points are finite, locally stable, materially more
  precise as samples accumulate, and project below three hours.
- **Revise:** one named component fails while the remaining oracle construction
  is coherent and a bounded repair is evident.
- **Stop:** parameter-branch ambiguity, optimizer error, or unresolved rank
  sensitivity dominates the statistical uncertainty.

### Check-in

Freeze the full-run sample ladder, rank, replicate cap, and convergence rule.

### Execution Record

The revised immutable pilot is
`cache/mnist_experiment/rotated_mnist/phase6/oracle/rotated_mnist_phase6_oracle_frozen_state_pilot__replica-0001__f4571e24ac61eb43`.
It completed 79 shared reference angles and 1,792 paired local fits in 415
seconds. The gate is **Go**.

At $M=2048$, rank 16, and $B=64$, 11 of 12 representative transitions meet
the six-standard-error `.01` target; the unresolved sigmoid mid-return point
has width `.0129` and remains explicitly flagged. Rank 8 versus rank 16 changes
$\widehat\pi_t^\star$ by `.00055` on average and `.00414` at most. Changing
$M$ from 512 to 2048 changes it by `.00163` on average and `.00502` at most.
The primary $M=2048$ local fits have small predictive changes from their
reference states, while all rank-8 and rank-16 Lanczos fits attain their
requested ranks.

The full run therefore freezes $M=2048$, $B\in\{16,32,64\}$, ranks 8 and 16,
and the original 64-fit cap. Rank 16 is the primary oracle and rank 8 remains
a representation-sensitivity diagnostic. The full configuration reuses the
pilot's immutable reference states and Fishers exactly.

## Phase 4: Full Instantaneous-Oracle Path

### Goal

Estimate the instantaneous theoretical oracle along the completed linear and
sigmoid EDR paths.

### Scope

1. Run the union of required reference angles once and reuse those immutable
   estimates across schedules.
2. Resume local fits until each point reaches the frozen precision rule or its
   replicate cap.
3. Calculate schedule-specific $S_t^\star$, old and new covariance risks,
   $q_t$, risk curvature, and $\widehat\pi_t^\star$ for every transition.
4. Preserve uncertainty intervals and explicit nonconvergence flags; never
   hide an unstable point through interpolation.
5. Use checkpointed `tmux` execution with a three-hour wall-clock cap.

### Verification

- Shared angles have identical reference artifact hashes across schedules.
- Every oracle row points to the exact pre-step EDR state and transition.
- Increasing oracle sample size changes precision but not deployed $m$.
- The run is resumable while incomplete and immutable after completion.

### Check-in

Decide whether the oracle line is sufficiently precise to attribute EDR error.

### Execution Record

The full immutable artifact is
`cache/mnist_experiment/rotated_mnist/phase6/oracle/rotated_mnist_phase6_oracle_full_path__replica-0001__7c7dc5936d08fb91`.
It completed 5,056 paired local fits over all 79 shared angles and 160
schedule-specific transitions in 1,400 seconds, including bootstrap
postprocessing. The reloaded reference states and Fisher representations are
tensor-identical to the pilot artifacts and share reference-contract hash
`158dc2e39a76dae3daf0d903b532147e697a8fd5e6d2310589b8d7b211f47f34`.

For the primary $M=2048$, rank-16, $B=64$ estimate, 140 of 160 transitions
meet the conservative six-SE `.01` target. The 20 flagged transitions are
retained without interpolation and cluster mainly on the sigmoid return, where
$q_t$ and the old-risk contribution are largest. The ordinary 95% intervals
remain narrower than the deliberately strict six-SE gate.

## Phase 5: Tracking Analysis And Next-Regime Decision

### Goal

Separate instantaneous estimation error, intentional smoothing lag, and local
surrogate limitations without overstating a one-replica diagnostic.

### Scope

1. Produce
   `mnist_experiment/rotated_mnist/instantaneous_oracle_results.ipynb` as an
   artifact-only notebook. It must identify the authoritative Plan 5 source
   trajectory and every Plan 6 oracle artifact it consumes, report incomplete
   or incompatible inputs clearly, and perform no hidden reconstruction.
2. Open with a concise human-readable statement of the estimand, the
   distinction between deployed $m$ and oracle $M$, and why the oracle is local
   rather than a globally optimal policy.
3. Include two primary panels:
   - debiased instantaneous recommendation versus instantaneous oracle;
   - realized EDR action versus instantaneous oracle with uncertainty.
4. Keep linear and sigmoid schedules separate. Mark cold start, direction
   reversal, angle, and realized angular speed without crowding the action
   curves.
5. Report ordinary tracking error and consequence-weighted error

   $$
   \operatorname{MAE}(\pi^{\mathrm{EDR}},\widehat\pi^\star),
   \qquad
   \sum_t(A_t^\star+B_t^\star)
   (\pi_t^{\mathrm{EDR}}-\widehat\pi_t^\star)^2.
   $$

6. Calculate the oracle-smoothed sequence only as a secondary attribution
   diagnostic. Do not substitute it for the instantaneous target in the
   controller-tracking plot.
7. Add compact convergence views for oracle sample size, local-fit replicate
   count, Fisher rank sensitivity, and unresolved points. Keep implementation
   diagnostics out of the primary scientific plots.
8. Classify the evidence:
   - **estimator bias/variance** when the debiased instantaneous estimate fails
     to track the oracle;
   - **smoothing lag** when the instantaneous estimate tracks but realized EDR
     does not;
   - **surrogate concern** when tracking is good but the realized learner still
     suffers material predictive harm; or
   - **mixed/inconclusive** when uncertainty cannot separate these mechanisms.
9. Only after classification, propose a predeclared hold-ramp-hold-return
   schedule in which low and high oracle regimes persist long enough to test
   adaptive value. Do not tune that schedule against realized EDR outcomes.

### Notebook Verification

- Every plotted value is traceable to an immutable artifact and aligned to the
  same $t\to t+1$ transition.
- The controller-tracking panel uses the instantaneous oracle, never the
  oracle-smoothed diagnostic, as its target line.
- Uncertainty bands come from independent local-fit or score-block units, not
  from treating path steps as independent observations.
- Executing all cells performs no training, score calculation, Fisher update,
  local MLE fit, artifact repair, or network access.

### Final Check-in

Decide whether to repair the online estimator, alter the smoothing model,
validate the local surrogate empirically, construct a sustained-regime
experiment, or stop the EDR branch.

### Execution Record

Phase 5 is complete. The artifact-only report is
`mnist_experiment/rotated_mnist/instantaneous_oracle_results.ipynb`. Its
primary figures compare the instantaneous rank-16 oracle with the debiased
online recommendation and realized EDR action; the oracle-smoothed sequence is
kept in the secondary attribution panel. All plotted values load strictly from
the completed Plan 5 source, Phase 6 debias artifact, pilot, and full oracle
artifact.

After the eight-step cold start, the linear schedule's mean absolute errors are
`.107` for the debiased instantaneous recommendation, `.139` for realized EDR,
and `.0227` for coefficient-wise oracle smoothing. The corresponding sigmoid
errors are `.132`, `.154`, and `.0378`. Consequence-weighted mean squared error
is `.0931` on linear and `.208` on sigmoid. The realized EDR means, `.277` and
`.300`, materially exceed the oracle means, `.137` and `.146`. The principal
classification is therefore **estimator bias/variance**, with a smaller but
visible smoothing contribution; ordinary EDR discounting is not the dominant
source of tracking error in these paths.

This is a diagnostic result, not proof that the one-step surrogate or oracle is
globally optimal. The finite local optimization resolves the corrected signal
at zero on `.425` of linear and `.500` of sigmoid transitions, so the bootstrap
intervals describe local-fit sampling uncertainty but cannot remove possible
optimizer-resolution bias. The strong pathwise association between realized
EDR and this conditional oracle is also partly mechanical because the oracle
uses EDR's realized $q_t$.

The EDR branch remains open, but the next experiment should not tune another
schedule against these outcomes. The predeclared candidate is a
hold-ramp-hold-return path: hold $0^\circ$ for 20 updates, ramp to $30^\circ$
over 8, hold for 32, return over 8, and hold $0^\circ$ for 32, retaining $m=4$
and the eight-step action half-life. It should be attempted only after improving
the local-reference optimization or validating a coarser signal estimator;
the present evidence prioritizes estimator calibration over longer smoothing.
