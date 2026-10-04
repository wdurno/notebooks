# Implementation Plan 12: Statistically Sized Ridge And Estimator Health

> **Hypothesis:** the continual learner is statistically unstable because its
> likelihood supplies very little curvature in a large parameter subspace.
> A statistically material ridge penalty may suppress weak-direction
> wandering, reduce estimator variance, and improve predictive behavior
> without materially biasing well-identified directions.

**Status:** Authorized for autonomous execution on 2026-09-30. Phases 0--4 are
complete. The post hoc MP $q_{.01}$ probe in Phase 5 was authorized and
completed on 2026-10-03 under the immutable/resumable contract below. Plan 11
Phase 3B and the proposed post-Phase 3A factorial study remain unlaunched.

**Scope:** This is an exploratory estimator-health study. It is not primarily
a test of an adaptive $\pi$ policy, and it does not assume that stabilizing the
learner will rescue $\pi^\star$. Parameter stability, bias, variance, excess
predictive risk, calibration, optimizer health, and retention are first-class
outcomes. $\pi^\star$ stability is a secondary diagnostic.

## Motivation

The canonical network has 512 trainable parameters, while each online update
contains only four observations. Plan 11 represents the Fisher with rank eight
plus a nonnegative residual diagonal. An exploratory spot check of eleven
evenly spaced fixed-$\pi=.025$ linear Phase 3 artifacts found a median spectral
entropy effective rank of about 25, a median top-eight trace fraction of about
$.73$, and a typical resolved condition ratio on the order of $10^7$. These
numbers are motivation, not frozen Plan 12 evidence; Phase 0 must reproduce
them systematically from validated artifacts.

The model also has an exact 25-dimensional likelihood gauge. Adding the same
24-vector to every classifier weight row and the same scalar to every
classifier bias adds only an input-dependent common shift to all ten logits.
Softmax probabilities and NLL remain unchanged. The population Fisher is
therefore singular in at least these 25 directions. This exact
non-identifiability must be separated from statistically weak directions that
can change the learned function.

The Phase 3A result gives a compatible but non-causal symptom: adaptive and
fixed learners differed detectably in linear-schedule NLL without a resolved
accuracy difference. Weakly constrained fitting of four-observation batches
could change probability quality while leaving most argmax decisions intact.
Ridge regularization is therefore a plausible upstream intervention, but the
existing evidence does not establish that weak-space wandering caused the
observed result.

## Scientific Contract

### Fisher And Penalized Estimator

The likelihood still induces one unregularized Fisher information matrix,

$$
\mathcal I(\theta)
=\mathbb E_\theta[s(X;\theta)s(X;\theta)^T].
$$

Ridge does not redefine $\mathcal I$, manufacture information, or turn an
unidentified direction into a likelihood-identified one. It defines a
penalized M-estimator. The unregularized Fisher remains the target for spectral
diagnostics and Fisher-risk evaluation.

Using the project's mean-loss and mean-Fisher normalization, the first ridge
intervention is

$$
\widehat\theta_t(\kappa)
=\arg\min_\theta
\left\{
L_t(\theta)
+\frac{\beta_t}{2}
(\theta-\widehat\theta_{t-1})^T
(\widehat F_{t-1}+\kappa_t I)
(\theta-\widehat\theta_{t-1})
\right\}.
$$

For the mixture-normalized EWC objective,

$$
\beta_t=\frac{1-\pi_t}{\pi_t}.
$$

The effective Euclidean penalty coefficient is consequently

$$
\tau_t=\beta_t\kappa_t.
$$

Every artifact and plot must expose $\kappa_t$, $\beta_t$, and $\tau_t$.
The initial causal study fixes $\pi_t=.025$, so the distinction between a
Fisher-internal floor and a separately parameterized proximal coefficient
cannot be confounded with a changing $\pi_t$. Before any adaptive-$\pi$
study, a versioned amendment must decide whether $\kappa_t$ or $\tau_t$ is the
primitive regularization action.

All Fisher eigenvalues, ridge values, and loss terms must use compatible
average-observation units. Summed-loss and mean-loss conventions may not be
mixed silently.

### Gauge Handling

Let $G_0$ denote the 25-dimensional common-classifier gauge and verify
numerically that model probabilities and per-sample NLL are invariant along
every basis direction. Plan 12 must implement a canonical chart satisfying

$$
\sum_{k=0}^{9}W_k=0,
\qquad
\sum_{k=0}^{9}b_k=0,
$$

or an equivalent orthonormal nine-class contrast parameterization. The
gauge-fixed model has 487 trainable coordinates and must be functionally
identical to the original model at matched states.

The raw 512-parameter model is retained only as a diagnostic control. The
gauge-fixed trainable dimension, not 512, is used in spectral aspect ratios and
regularization calculations. Report movement in the exact gauge separately;
do not count its removal as evidence that statistically meaningful wandering
was fixed.

### Operational Subspaces

At an anchor state, let $U_r$ contain a declared set of orthonormal,
numerically validated leading Fisher directions and define

$$
P_r=U_rU_r^T,
\qquad
P_r^\perp=I-P_r.
$$

These are operational resolved and unresolved subspaces, not claims about a
true finite-dimensional manifold. Report sensitivity to several predeclared
ranks or spectral thresholds. Do not select a rank because it makes ridge look
favorable.

Alongside isotropic ridge, the local study may evaluate the targeted penalty

$$
\widehat F_{t-1}+\kappa_tP_r^\perp.
$$

This condition directly tests the submanifold hypothesis while leaving the
resolved subspace unchanged. A deployed tail-only rule must use a predictable,
lagged subspace; a same-batch eigenspace is allowed only for an explicitly
oracle diagnostic.

## Questions And Estimands

Plan 12 asks five ordered questions:

1. How much apparent instability is exact gauge motion, numerical compression
   error, or movement in statistically weak but functionally relevant
   directions?
2. At fixed $\pi=.025$, does nonzero ridge reduce conditional and propagated
   estimator variance?
3. What bias does that stabilization introduce in parameter, Fisher-risk, and
   predictive coordinates?
4. Is there a reproducible interval of ridge strengths with better overall
   estimator health than the unregularized controls?
5. Only if such an interval exists, can a predictable spectral rule select a
   useful $\kappa_t$ without oracle outcome tuning?

Let $a$ index a frozen anchor state, $r$ an independently sampled local batch,
and $k$ a ridge condition. Write the resulting estimator as
$\widehat\theta_{a,r,k}$ and define its batch-replication mean

$$
\overline\theta_{a,k}
=\mathbb E_r[\widehat\theta_{a,r,k}\mid a].
$$

Use disjoint high-sample data to construct two references. Let
$\theta_a^\star$ be the unregularized current-environment pseudo-true target,
and let $\theta_{a,k}^{\mathrm{pop}}$ minimize the population analogue of
condition $k$'s penalized objective in the same local basin. Estimate three
distinct biases in the gauge-fixed chart:

$$
B_{\mathrm{est},a,k}
=\|\overline\theta_{a,k}-\theta_{a,k}^{\mathrm{pop}}\|_2^2,
$$

$$
B_{\mathrm{reg},a,k}
=\|\theta_{a,k}^{\mathrm{pop}}-\theta_a^\star\|_2^2,
$$

$$
B_{\mathrm{total},a,k}
=\|\overline\theta_{a,k}-\theta_a^\star\|_2^2,
$$

where estimator and regularization bias vectors need not be orthogonal, so
their squared norms are not assumed to add. Conditional variance is

$$
V_{E,a,k}
=\mathbb E_r
\|\widehat\theta_{a,r,k}-\overline\theta_{a,k}\|_2^2.
$$

Evaluate every bias and variance component in resolved, unresolved, and
reference-Fisher-weighted coordinates. Because neural parameter bias is
chart- and basin-dependent, report functional analogues in logits,
probabilities, NLL, Brier score, and calibration. Both high-sample references
are local benchmarks, not assertions of a unique global parameter truth.

For full trajectories, decompose both conditional innovation variance and
unconditional across-replica variation. Random-anchor propagation is part of
the estimand rather than a footnote.

## Outcome Families

### Primary Estimator-Health Outcomes

- Euclidean and reference-Fisher-weighted squared error relative to the local
  high-sample reference.
- Bias, variance, and total mean squared error in the full, resolved, and
  unresolved parameter spaces.
- Held-out current-environment NLL and normalized NLL AUC.
- Retention-panel NLL and worst-panel NLL.
- Across-batch and across-replica variance of predicted logits and
  probabilities after gauge alignment.

### Secondary Predictive Outcomes

- Accuracy, Brier score, expected calibration error, classwise recall, and
  confusion matrices.
- Digit-9 and non-9 metrics, reported as diagnostics rather than selectable
  success endpoints.
- Excess training-to-evaluation NLL and the frequency of extreme predictive
  probabilities.

### Optimization And Numerical Health

- Initial and final gradient norms, objective decrease, line-search events,
  optimizer iterations, and convergence reason.
- Total, resolved, unresolved, and exact-gauge displacement norms.
- EWC penalty, Fisher-weighted displacement, and effective isotropic penalty.
- Fisher trace, effective rank, spectral entropy, condition diagnostics,
  Lanczos residuals, residual-diagonal summaries, and compression error
  against dense checkpoints.
- Dense and compressed curvature assigned to the exact gauge, including
  $\|\widehat F G_0\|$ and gauge-restricted quadratic forms.
- Sensitivity to batch identity, numerical randomization, and allowable
  optimizer tolerances.

### Secondary $\pi$ Diagnostics

- Mean, variance, and seed reproducibility of $\widehat\pi_t^\star$ and any
  deployable recommendation.
- Sensitivity of $\widehat\pi_t^\star$ to the ridge strength, Fisher
  representation, and batch identity.
- Agreement between any covariance formula used by the recommendation and the
  empirical penalized-estimator covariance.

No phase may be declared successful solely because a $\pi$ trajectory becomes
smoother or moves toward a preferred fixed value.

## Experimental Controls

The complete control vocabulary is:

1. **No-update anchor:** zero conditional innovation variance and maximal
   local tracking bias; this is a bias-variance boundary, not a deployable
   learner.
2. **Current-only control:** fit the four-observation batch without an old-data
   EWC penalty; this is the opposite bias-variance boundary.
3. **Legacy raw control:** the existing 512-parameter model and current
   rank-eight-plus-diagonal Fisher, with no new EWC ridge.
4. **Gauge-fixed control:** the functionally equivalent identifiable chart,
   with no new EWC ridge.
5. **Dense-archive control:** a gauge-fixed branch using the dense archive
   Fisher propagated with the same observations and weights as the deployed
   compressed archive, with no new EWC ridge.
6. **Isotropic ridge grid:** gauge-fixed branches using
   $\widehat F+\kappa I$.
7. **Tail-only ridge grid:** gauge-fixed branches using
   $\widehat F+\kappa P_r^\perp$.
8. **Simple heuristic:** for later comparison,
   $\kappa=.01\widehat\lambda_1$.
9. **Spectral selector:** theoretical and, if justified, calibrated
   deformed-MP candidates.
10. **Oracle ridge:** the best member of the frozen grid under a declared
   estimator-health loss, used only as a benchmark and never deployed on the
   same data used to select it.

The phases below deliberately avoid running this entire Cartesian product at
full-trajectory scale.

## Phase 0: Identifiability, Artifact, And Cost Audit

### Phase 0A: Existing-Artifact Audit

Read only immutable initialization, Plan 11 development, and focused Phase 3
artifacts. Validate their contracts and hashes before analysis. Reproduce the
motivating spectral summaries over all eligible artifacts or a frozen,
uniformly selected subset when dense eigendecomposition cost requires it.

At minimum, report:

- represented spectral distributions, effective ranks, and condition
  diagnostics by schedule, policy, step, and replica;
- residual-diagonal zeros and near-zeros;
- spurious curvature assigned by compressed representations to the exact
  likelihood gauge;
- variation in leading eigenvalues and subspaces across numerical seeds;
- which requested displacement and reference quantities cannot be recovered
  from existing artifacts;
- the exact locations where existing numerical damping is used and proof that
  it is distinct from a statistically sized EWC ridge.

This phase may motivate the study but cannot establish a causal ridge benefit.

### Phase 0B: Gauge And Dense-Fisher Validation

Implement fast deterministic tests for the common-logit gauge, the
sum-to-zero chart, and round-trip parameter mappings. On tiny CPU fixtures,
verify equal logits up to their common shift, equal probabilities, equal NLL,
and the removal of exactly 25 trainable gauge dimensions.

At one smoke anchor, calculate both a dense high-sample measurement Fisher and
a dense archive counterpart propagated with the same history as the deployed
low-rank-plus-diagonal archive. Compare each compressed representation with
its matched dense object. Validate eigenspaces, quadratic forms, isotropic
ridge, tail-only ridge, and matrix-free penalty evaluation.

Benchmark separately:

- construction of one shared anchor and reference Fisher;
- one local EWC branch per ridge value;
- one complete fixed-$\pi$ trajectory;
- artifact size and notebook load time.

Phase 0 ends with measured wall-clock projections for Phases 1--4. Once the
overall Plan 12 launch has been authorized, Phase 1 starts automatically after
the Phase 0 integrity and feasibility checks pass.

## Phase 1: Conditional Local Ridge Response

Phase 1 is the cheapest causal test of the anti-wandering hypothesis. It does
not run an adaptive controller.

### Frozen Anchors

Generate a small, predeclared set of fresh canonical anchor states spanning
distinct path locations, including low rotation, first ascent, reversal, and
second ascent. Linear and sigmoid anchors are separate conditions. Each anchor
stores the model, optimizer-independent parameter order, deployed archive
Fisher, matched dense archive counterpart, controller-independent metadata,
and disjoint sampling pools.

For each anchor:

1. Construct a high-sample dense measurement Fisher in the gauge-fixed chart.
2. Construct the disjoint unregularized pseudo-true target and each
   condition-specific population penalized target by warm-started optimization
   in the same local basin.
3. Freeze independent four-observation batch identities before fitting any
   ridge condition.
4. Branch every batch through every authorized local condition using common
   optimizer settings and common numerical randomization.
5. Evaluate every branch on the same held-out current and retention panels.

Every high-sample target must pass predeclared convergence, repeated-fit, and
held-out objective checks. If a local target cannot be reproduced to the
required tolerance, parameter-space bias relative to that target is marked
unavailable rather than reported with false precision; functional held-out
risk and pairwise estimator dispersion remain valid outcomes.

Before production data are generated, freeze the following validity gates:

- repeated target fits must be within Euclidean parameter distance $10^{-2}$;
- both target fits must finish with relative gradient norm at most $.05$;
- both target fits must weakly decrease their respective objectives.

The relative gradient norm is the final gradient norm divided by the larger
of the initial gradient norm and the numerical floor used by the optimizer.
The intentionally tiny smoke run is allowed to fail these gates so that the
unavailable-target analysis path is tested before production.

The statistical unit is an independently sampled batch branch conditional on
the anchor. Anchor-level summaries remain visible so a large number of batches
at one anchor is not misrepresented as broad path replication.

### Initial Ridge Grid

Let

$$
s_a=\frac{\operatorname{tr}(\widehat F_a)}{p_a}
$$

in the gauge-fixed trainable space. Subject to Phase 0 numerical review, the
initial isotropic and tail-only grid is

$$
\frac{\kappa}{s_a}
\in
\{0,10^{-4},10^{-3},10^{-2},10^{-1},1\}.
$$

Include the no-update, current-only, legacy raw, gauge-fixed, and matched
dense-archive controls. The heuristic $.01\widehat\lambda_1$ may be evaluated
as a labeled external rule; it does not replace the scale grid. Record all
realized ratios
$\kappa/\widehat\lambda_j$ and $\tau/\widehat\lambda_j$.

### Phase 1 Interpretation

Report complete response curves and paired uncertainty intervals. Seek a
contiguous ridge region rather than a single best grid point. A promising
region must:

- reduce unresolved, non-gauge estimator variance;
- reduce total parameter or functional mean squared error, not variance alone;
- preserve or improve held-out probability quality relative to the
  gauge-fixed control;
- avoid material deterioration in retention or optimizer convergence;
- occur at more than one anchor and not depend on one selected class or
  exposure window.

Classify the anti-wandering hypothesis as unsupported when variance is not
concentrated in weak directions, ridge only removes exact gauge motion, every
variance reduction is offset by larger bias or predictive risk, or useful
values occur only as isolated outcome-selected grid points.

Phase 1 is exploratory development evidence. Any selected ridge region is
optimistically biased and requires fresh trajectory data.

## Phase 2: Propagated Estimator Health

Phase 2 starts automatically after Phase 1 artifacts and the progressive
notebook pass integrity checks. Freeze at most two nonzero ridge rules before
generating fresh trajectory replicas: ordinarily one isotropic rule and one
tail-only rule from a contiguous Phase 1 region. No user review is required
between these phases.

Select each geometry's rule lexicographically across the predeclared anchors:

1. Exclude conditions with invalid artifacts, nonfinite optimization, or a
   failed normalization contract.
2. When reproducible population targets exist, minimize mean
   reference-Fisher-weighted total mean squared error.
3. When those targets are unavailable, minimize held-out current and retention
   NLL under a frozen aggregate loss.
4. Break practically indistinguishable ties in favor of smaller $\kappa$.

For this selection, a nonzero rule is a practical improvement only when its
aggregate objective is at least $.5\%$ below the gauge-fixed no-ridge control.
Differences within $.5\%$ are treated as practically indistinguishable and the
smaller ridge is preferred. This threshold is frozen before production and is
not adjusted after inspecting response curves.

If no nonzero value improves on the gauge-fixed control, label the geometry
**likely unfavorable**, select its least harmful nonzero value, report the
failure prominently in the notebook, and continue. Advancing that condition
is a diagnostic stress test, not evidence that it is promising.

Use fixed $\pi=.025$ for all conditions and preserve the Plan 11 double-lap
environment, four-observation batches, model, optimizer, and evaluation
panels. Treat linear and sigmoid schedules as separate experimental
conditions. The minimum comparison is:

1. current-only, retained as a high-variance boundary control;
2. gauge-fixed fixed-$\pi$, no ridge;
3. gauge-fixed fixed-$\pi$, selected isotropic ridge;
4. gauge-fixed fixed-$\pi$, selected tail-only ridge;
5. legacy raw fixed-$\pi$, no ridge, retained to quantify the effect of gauge
   fixing and representation compatibility.

Every within-replica condition shares initialization, observations, reference
panels, and numerical seeds. Lanczos seeds are keyed by replica, schedule, and
step, never by treatment. Parameter hashes must remain identical until a
treatment actually differs.

Primary trajectory analyses estimate schedule-specific means and paired
differences in NLL-AUC, reference-Fisher risk, estimator variance, squared
bias proxies, weak-space movement, and retention. Accuracy and $\pi$ metrics
remain secondary. Report complete trajectories, not only terminal summaries.

The default 64-replica Phase 2 size is exploratory and is frozen before Phase
1 outcomes are inspected. Phase 1 reports the precision that this size is
expected to attain but does not change the count in response to a favorable or
unfavorable ridge effect. Do not reuse the earlier 352-replica target, and do
not repeatedly test accumulating replicas until significance appears.

## Phase 3: Spectral $\kappa$ Calibration

Phase 3 starts automatically after Phase 2 integrity and analysis complete.
An empirically useful ridge region opens the full selector-development track.
If ridge appears unfavorable, Phase 3 still executes the empirical-resampling,
heuristic, and theoretical-edge comparisons at the frozen checkpoints, labels
the premise as likely failed, and treats calibration as a falsification audit.
Rolling PIT calibration may be fit and reported, but no selector is promoted
as deployable merely because it predicts an edge when ridge itself is harmful.

### Empirical Calibration Benchmark

At frozen, independently selected checkpoints, retain or recompute disjoint
per-sample scores in the gauge-fixed chart. Use dense high-sample Fishers and
blocked or otherwise justified score resampling to estimate empirical
distributions of unresolved sample-spectrum maxima under the actual weighting
scheme.

The benchmark compares:

- fixed ridge values from the Phase 1 response region;
- $.01\widehat\lambda_1$;
- an empirical resampling quantile;
- a theoretical deformed-MP edge;
- a rolling calibrated edge, only after adequate calibration history;
- the frozen oracle grid benchmark.

Evaluate edge coverage, quantile error, temporal stability, selected-$\kappa$
error, and regret under the declared estimator-health loss. Good unresolved
edge coverage is not sufficient if the selected ridge has poor bias-risk
tradeoffs.

### Requirements For MP Use

The MP implementation must not substitute $n_{\mathrm{eff}}$ into an
ordinary unweighted law without validating the exponentially weighted
spectral model. It must document the stationarity, dependence, moment, aspect
ratio, edge-regularity, and outlier assumptions used at each checkpoint.

Fit the population spectral model to lagged or sample-split projected-score
curvatures. Raw in-sample Lanczos Ritz values may be logged but must not be
silently treated as population eigenvalues and then passed through a second
sample-spectrum map.

For the frozen production calibration, order checkpoints by their predeclared
replica and path indices, fit the spectral model on the first half, and reserve
the second half for empirical edge calibration and coverage assessment. No
checkpoint may contribute to both halves. If any required empirical checkpoint
fails its calibration contract, use the theoretical deformed-MP fallback and
label the empirical selector unavailable rather than selecting from the
remaining favorable checkpoints.

With approximately eight resolved directions, power-law fit windows,
pseudo-tail holdouts, and rank offsets are fragile. Freeze them before
evaluation and expose sensitivity analyses. Overlapping EMA checkpoints do
not provide independent PIT observations; calibration uncertainty must use a
dependence-aware effective history and must not claim nominal $.99$ coverage
without empirical validation.

Only a selector that predicts held-out edge behavior and lands reproducibly
inside the useful ridge region may be described as calibrated or deployable.
Failed selectors still advance to Phase 4 as explicitly labeled diagnostic
controls under the frozen fallback rule.

## Phase 4: Independent Confirmation And $\pi^\star$ Health

Phase 4 starts automatically after Phase 3 and evaluates $\pi^\star$ health
regardless of whether fixed-$\pi$ ridge improved general estimator health.
Freeze one gauge contract, the selected isotropic and tail-only ridge rules,
the best calibrated spectral rule or its theoretical fallback, and fresh
replica identities before Phase 4 outcomes are generated.

The no-ridge gauge-fixed estimator remains the primary control. If all ridge
conditions were unfavorable, carry forward the least harmful nonzero
isotropic and tail-only values and label them negative-control best bets. If
the MP selector failed calibration, carry its frozen theoretical value as a
model-failure diagnostic rather than silently replacing it after seeing
$\pi^\star$ outcomes.

At fresh checkpoints and along fresh shadow trajectories, estimate:

- conditional and across-replica variance of $\widehat\pi_t^\star$;
- bias against a high-sample local $\pi_t^\star$ reference when that reference
  passes convergence and uncertainty checks;
- mean squared error, interval coverage, seed reproducibility, temporal
  stability, and sensitivity to batch identity;
- covariance-model calibration under the penalized sandwich approximation;
- relationships between $\pi^\star$ error and general estimator-health
  outcomes without treating correlation as causation;
- the frequency and persistence of boundary, nonfinite, or numerically
  unsupported recommendations.

Both linear and sigmoid schedules remain separate experimental conditions.
Recommendations are evaluated in shadow mode by default so this phase measures
$\pi^\star$ health without changing the learner path. A small paired actuated
sensitivity may be included only if its condition ledger and fixed size were
frozen before Phase 4 began; it remains exploratory and cannot overwrite the
shadow-estimation conclusion.

The analysis must use the penalized sandwich covariance, account for
random-anchor propagation and ridge bias, and state whether each reference
targets the penalized estimator or the unregularized likelihood optimum.
An unfavorable fixed-$\pi$ result does not cancel this phase; it becomes
context for interpreting whether ridge stabilizes $\pi^\star$ while harming
the learner.

Plan 12 does not authorize retroactively changing the estimand or conclusion
of any Plan 11 experiment.

## Phase 5: Exploratory MP $q_{.01}$ Probe

Phase 4 found that the frozen empirical MP $q_{.99}$ selector was useful as a
strong retention condition but suppressed more than $99.96\%$ of the squared
parameter displacement observed under `gauge_no_ridge`. Its advantage near
upright digits and disadvantage near maximal rotation are consistent with an
anti-forgetting trust region that permits little adaptation. This motivates a
lower-quantile probe, not a retroactive repair of the failed Phase 3 selector
contract.

Use the eight completed Phase 3 `full_maxima` bootstrap distributions and the
same conservative `higher` order-statistic convention used by the frozen
$q_{.99}$ selector. At checkpoint $a$, define

$$
r_{a,.01}
=
\frac{Q_{.01}^{\mathrm{higher}}(M_a^*)}{s_a},
\qquad
s_a=\frac{\operatorname{tr}(\widehat F_a)}{487},
$$

and freeze the production ratio at the median across checkpoints,

$$
r_{.01}=\operatorname{median}_{a=1,\ldots,8}r_{a,.01}
=11.818529434984821.
$$

The exploratory treatment uses the same isotropic form as the Phase 4
spectral condition,

$$
R_t=I,
\qquad
\kappa_t=r_{.01}s_t,
$$

with fixed $\pi=.025$ and all other learner, estimator, evaluation, and shadow
$\pi^\star$ settings unchanged. Call this condition `mp_q01_isotropic`; do not
call it calibrated or selected. It is an outcome-informed, post hoc probe.

Run 16 paired replicas for each of the linear and sigmoid schedules. Reuse the
exact immutable Phase 4 replica assets, component seeds, streams, and
evaluation panels for replica indices 1 through 16. Compare the new condition
with the matching completed Phase 4 `gauge_no_ridge` and
`spectral_selector` trajectories. Existing Phase 4 artifacts remain immutable
and are referenced by hash; they are not copied, rewritten, or relabeled.

The primary diagnostic question is whether $q_{.01}$ restores material
adaptation while preserving some of the $q_{.99}$ retention benefit. Report:

- NLL and accuracy trajectories separately for each schedule;
- angle-stratified performance near $0$--$5^\circ$ and $25$--$30^\circ$;
- first ascent, return, and second-ascent summaries;
- total squared displacement relative to both controls;
- NLL-AUC, accuracy-AUC, retention, optimizer, and Fisher-health diagnostics;
- descriptive paired means, medians, replica counts, and uncertainty
  intervals, without significance or selector-calibration claims.

The initial 16-replica sample is a fixed exploratory health check, not an
early-stopping boundary. Its notebook section must remain labeled incomplete
for inference. Expansion to 64 replicas per schedule requires a separate
decision after review; such an expansion appends replicas 17 through 64 and
does not alter the first 16 artifacts or promote the probe to confirmatory
evidence.

Phase 5 uses its own versioned ledger and supports `--resume`, `--max-units`,
and `--max-wall-seconds`. Each trajectory is an immutable unit. Refresh the
artifact-only notebook after every bounded batch and show progress by schedule
even when only one schedule is partially complete.

## Statistical Reporting

The development phases emphasize estimation rather than binary declarations:

- report means, medians, standard deviations, paired effect sizes, and
  uncertainty intervals;
- show bias and variance separately before combining them as mean squared
  error;
- keep linear and sigmoid schedules separate;
- distinguish batches, anchors, steps, schedules, and replicas as statistical
  units;
- label oracle selection, post hoc ranks, and inspected grid regions clearly;
- never count correlated steps or ridge values as independent replication;
- retain unfavorable conditions and complete response curves.

Intermediate notebook inspection may assess execution health and descriptive
progress. It may not select a stopping time because a preferred ridge value
crosses a significance threshold. Partial-sample intervals and trajectories
are labeled descriptive and incomplete; fixed-size inferential summaries stay
locked until their ledger is complete.

## Immutable Execution And Resumption

All expensive work uses configuration-driven Python entry points under
`mnist_experiment/rotated_mnist/plan12/`. Notebooks never train, calculate
large Fishers, repair runs, or download data.

Each study freezes a condition ledger before its first non-smoke unit. Unit
identities include the Plan 12 contract version, phase, anchor or replica,
schedule, gauge condition, Fisher representation, ridge geometry, ridge rule,
and ridge value. Completed units are immutable.

Every long-running entry point must support:

- `--resume` to skip validated completed units and restart incomplete units;
- `--max-units` to bound work by immutable units;
- `--max-wall-seconds` to stop before beginning work unlikely to finish;
- deterministic work ordering recorded in the ledger;
- atomic temporary directories and a final `COMPLETED` marker;
- explicit failure records without silently dropping a branch.

The original top-level orchestrator proceeds through Phases 0--4 without
requiring an interactive approval between phases. The separately authorized
Phase 5 extension follows its frozen ledger. Scientific failure classifications do
not halt it: they trigger the frozen fallback conditions, an analysis record,
and a prominent notebook warning. It halts only when artifact integrity,
normalization, pairing, or a required estimand is undefined and no declared
fallback exists.

At the end of every bounded compute batch and every phase transition, run the
artifact-only analysis and atomically refresh the executed output notebook.
A notebook-refresh failure is recorded and retried before the next compute
batch; it does not mutate completed scientific artifacts.

Per-sample scores, dense Fishers, model states, and run artifacts remain under
`cache/`. Large shared anchor assets are content-addressed and referenced by
hash from branch artifacts. A completed shared asset is never modified.

## Proposed Implementation Layout

- `mnist_experiment/rotated_mnist/plan12/config.py`: frozen contracts,
  condition generation, hashes, and component seeds.
- `mnist_experiment/rotated_mnist/plan12/gauge.py`: identifiable classifier
  chart and parameter mappings.
- `mnist_experiment/rotated_mnist/plan12/ridge.py`: isotropic and tail-only
  matrix-free penalties.
- `mnist_experiment/rotated_mnist/plan12/spectral.py`: dense and Lanczos
  diagnostics, projected-score estimates, and effective sample accounting.
- `mnist_experiment/rotated_mnist/plan12/local_response.py`: Phase 1 anchor
  construction and paired branches.
- `mnist_experiment/rotated_mnist/plan12/trajectory.py`: restartable fixed-$\pi$
  Phase 2 trajectories.
- `mnist_experiment/rotated_mnist/plan12/mp_calibration.py`: Phase 3
  spectral calibration.
- `mnist_experiment/rotated_mnist/plan12/artifacts.py`: immutable units,
  ledgers, integrity hashes, and schema validation.
- `mnist_experiment/rotated_mnist/plan12/analysis.py`: artifact-only summaries
  and bias-variance decompositions.
- `mnist_experiment/rotated_mnist/plan12/refresh_notebook.py`: atomic,
  artifact-only execution of the progressive notebook.
- `mnist_experiment/rotated_mnist/ridge_estimator_health.ipynb`: lightweight
  results notebook.
- `mnist_experiment/rotated_mnist/plan12/PHASE*_FINDINGS.md`: concise phase
  records with decisions and limitations.

## Notebook Requirements

The results notebook must validate contracts, ledgers, artifact hashes,
pairing, normalization, and schema versions before calculating summaries. It
must fail clearly on corrupt required units. An incomplete ledger is rendered
as an explicit progress state rather than an error or a silently reduced
sample.

The notebook is useful from the first completed Phase 0 unit onward. Its first
section always reports, by phase and condition, planned units, completed units,
failed or incomplete units, elapsed compute, integrity status, current
scientific classification, selected fallback rules, and the timestamp and
analysis-source hash of the refresh. Later sections render every currently
available result while clearly distinguishing partial development evidence,
complete development evidence, and fresh confirmation evidence.

At minimum, render:

- dense and represented spectra with ridge levels and effective ranks;
- exact-gauge, resolved, and unresolved displacement distributions;
- estimator bias, variance, and mean squared error response curves;
- predictive bias-variance and calibration summaries;
- optimization-health diagnostics across ridge values;
- Pareto views of variance reduction against bias and held-out NLL;
- full fixed-$\pi$ trajectory comparisons by schedule;
- MP or empirical-edge calibration as soon as Phase 3 artifacts exist;
- secondary $\pi^\star$ stability plots visibly separated from primary health
  conclusions.

## Testing Requirements

Add fast deterministic tests for:

- common-logit invariance and the sum-to-zero parameter chart;
- score, Hessian-vector, and Fisher transforms under gauge fixing;
- isotropic and tail-only quadratic forms and gradients;
- equality of dense and matrix-free ridge penalties;
- resolved/unresolved orthogonal decomposition;
- average-Fisher normalization and the identity $\tau=\beta\kappa$;
- deterministic paired condition and seed generation;
- bias-variance decomposition on synthetic estimators with known moments;
- immutable collision, completion, interruption, and resumption behavior;
- a tiny CPU local-response smoke study;
- a tiny CPU trajectory smoke study;
- notebook rejection of training and artifact repair operations.

Numerical MP tests in Phase 3 must include spherical cases with known
Marchenko-Pastur edges, finite discrete population spectra, pole avoidance,
multiple support intervals, regular-edge checks, and simulation-based
coverage fixtures.

## Failure Classification And Continuation

The following are scientific failure classifications, not automatic stop
conditions:

1. Nearly all apparent weak-space movement is exact gauge motion or a known
   representation artifact.
2. Ridge reduces variance but increases total estimator mean squared error and
   held-out predictive risk throughout the frozen grid.
3. No contiguous ridge interval behaves consistently across anchors.
4. The useful ridge scale appears outside the initial grid or is too large to
   preserve resolved-direction behavior.
5. Dense-archive controls show that the apparent benefit only compensates for a
   correctable compression defect.
6. The deformed-MP selector is empirically miscalibrated or selects harmful
   ridge values.

Record any such classification in the phase findings and the first notebook
section, then continue under the declared fallback. Gauge-only findings retain
the identifiable controls and least harmful nonzero ridge conditions. A
uniformly unfavorable grid advances its least harmful nonzero isotropic and
tail-only values. An apparent boundary optimum may trigger one predeclared
one-decade grid extension, after which no further outcome-driven expansion is
allowed. A compression defect creates a new versioned corrected condition and
preserves the original artifacts. MP failure uses the frozen theoretical rule
as a labeled negative control in Phase 4.

Execution halts only for a hard scientific-integrity blocker: unrecoverable
artifact corruption, incompatible normalization, broken pairing, a failed
gauge equivalence test, or an undefined required estimand with no declared
fallback. Resource interruptions use `--resume` and are not scientific stop
conditions.

## Autonomous Defaults

Once Plan 12 execution is authorized, Phases 0--4 may run without intermediate
user review. The default design, subject only to Phase 0 feasibility checks,
is:

- the full 25-dimensional classifier gauge is removed;
- Phase 1 uses eight anchors, four path locations on each schedule, and 32
  independent four-observation batch branches per anchor;
- the initial ridge grid and all boundary-extension rules above are frozen;
- Phase 2 uses 64 fresh replicas, two schedules, and the five listed
  trajectory conditions, unless Phase 0 freezes the 128-replica contingency
  using historical variance before Phase 1 outcomes exist;
- Phase 3 uses eight frozen checkpoints and 512 empirical spectral resamples
  per checkpoint;
- Phase 4 uses 64 fresh shadow-trajectory replicas per schedule for the
  no-ridge, selected isotropic, selected tail-only, and spectral-selector
  conditions, with the same Phase 0-only 128-replica contingency;
- target failure falls back from parameter MSE to the frozen functional NLL
  and retention loss; ties favor smaller $\kappa$;
- no choice may be made using favorable $\pi^\star$ behavior alone.

Phase 0 may reduce the size of an execution batch for memory or interruption
robustness, but it may not reduce the frozen total unit count. A scientifically
material change to an estimand still requires a versioned amendment; ordinary
candidate ranking and failure fallback do not.

## Provisional Compute Envelope

The estimate below excludes engineering time and assumes the historical RTX
4070 throughput of about 58.25 seconds per complete Plan 11 trajectory and
10.82 seconds per shared replica asset. Phase 0 replaces these projections
with measured Plan 12 timings in the notebook.

| Phase | Default work | Projected wall time |
| --- | --- | ---: |
| 0 | Artifact audit, gauge validation, smoke profiling | 1--2 h |
| 1 | Eight anchors, local targets, and ridge response branches | 3--6 h |
| 2 | 64 replicas $\times$ 2 schedules $\times$ 5 trajectories | 11--13 h |
| 3 | Dense checkpoints, 4,096 resamples, and MP calibration | 3--6 h |
| 4 | 64 fresh replicas $\times$ 2 schedules $\times$ 4 shadow trajectories | 9--12 h |
| 5 | 16 paired replicas $\times$ 2 schedules $\times$ 1 MP $q_{.01}$ trajectory | 40--50 min |
| Analysis | Integrity scans and progressive notebook refreshes | 1--2 h |
| **Total** | Default autonomous experiment | **28--41 h** |

The planning midpoint is approximately **34 hours** of wall-clock compute. If
Phase 0 justifies 128 rather than 64 trajectory replicas in both Phases 2 and
4, add approximately 19 hours. Driver outages, thermal throttling, and CPU-only
execution are outside this estimate; immutable units and bounded sessions
preserve useful progress through interruptions.

## References For The Spectral Work

- Dobriban, E. (2015), *Efficient Computation of Limit Spectra of Sample
  Covariance Matrices*, <https://arxiv.org/abs/1507.01649>.
- Lee, J. O. and Schnelli, K. (2016), *Tracy-Widom Distribution for the
  Largest Eigenvalue of Real Sample Covariance Matrices with General
  Population*, <https://arxiv.org/abs/1409.4979>.
- Oriol, B. (2025), *Asymptotic Spectrum of Weighted Sample Covariance:
  Another Proof of Spectrum Convergence*,
  <https://arxiv.org/abs/2410.14408>.
- Silverstein, J. W. and Choi, S. I. (1995), *Analysis of the Limiting
  Spectral Distribution of Large Dimensional Random Matrices*,
  <https://doi.org/10.1006/jmva.1995.1058>.
