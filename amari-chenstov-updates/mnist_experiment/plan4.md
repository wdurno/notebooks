# Implementation Plan 4: Adaptive Pi Under Variable Speed

Plan 4 follows the completed deployment baseline in [plan3.md](plan3.md). It
asks one narrow question:

> Can the applied controller adjust $\pi_t$ usefully when the true parameter
> moves at a nonconstant rate, without retuning one fixed $\pi$ for each
> environment?

This began as a variable-speed experiment, not a claim about intrinsic
manifold curvature. The schedule changes the parameterization in time of the
same one-dimensional path

$$
p\longmapsto\theta^\star(p).
$$

The completed Euclidean-risk design screen found no actionable controller
signal, even under severe speed shocks. The plan now contains a separately
identified Fisher-risk amendment: measure local movement and estimation error
with the already maintained Fisher summary before deciding whether the
adaptive controller itself lacks value.

Every phase ends with a check-in. Do not begin a long predictive experiment
until the preceding mathematical and software gates have passed.

## Motivation

On the 100-step linear MNIST path, the applied controller stayed almost
entirely at $\pi_{\min}=.05$. It consequently reproduced the practical fixed
policy and demonstrated no value as an adaptive mechanism. That result does
not test the controller under changing drift speed.

Let $\gamma(p)=\theta^\star(p)$ and choose a schedule $p=p(t)$. Then

$$
\dot\theta(t)=\gamma'(p(t))\dot p(t).
$$

Changing $\dot p(t)$ preserves the geometric image of $\gamma$ but changes the
per-update displacement entering the controller risk. With

$$
A_t=\|d\theta_t\|^2+\tau_{\mathrm{old},t},
\qquad
B_t=\tau_{\mathrm{new},t},
$$

the applied fixed-batch risk and minimizer are

$$
R_t(\pi)=(1-\pi)^2A_t+\pi^2B_t,
\qquad
\pi_t^\star=\frac{A_t}{A_t+B_t}.
$$

For any candidate action $\pi$ at the same fixed controller state,

$$
R_t(\pi)-R_t(\pi_t^\star)
=(A_t+B_t)(\pi-\pi_t^\star)^2.
$$

If the local coefficients $A_t$ and $B_t$ are frozen independently of the
candidate policy, their conditional best constant action is

$$
\pi_{\mathrm{fixed}}^\star
=\frac{\sum_t A_t}{\sum_t(A_t+B_t)}.
$$

and the corresponding conditional oracle opportunity is

$$
G_{\mathrm{oracle}}
=
\frac{
\sum_t R_t(\pi_{\mathrm{fixed}}^\star)
-\sum_t R_t(\pi_t^\star)
}{
\sum_t R_t(\pi_{\mathrm{fixed}}^\star)
}.
$$

This closed form is not yet closed-loop policy regret. In the applied model,

$$
q_t=(1-\pi_t)^2q_{t-1}+\frac{\pi_t^2}{m_t},
$$

so $\tau_{\mathrm{old},t}=T_tq_t$ depends on earlier actions. Phase 2 must
therefore roll each adaptive or constant policy through its own $q_t$
recursion and calculate cumulative risk on that policy's resulting state
sequence. The frozen-coefficient expression remains a transparent local
diagnostic only.

These equations also describe the unconstrained ideal. The applied feasibility
gate clips both pointwise and constant policies to
$[\pi_{\min},\pi_{\max}]$ and uses bounded closed-loop rollouts for its
principal risk opportunity. Report conditional and unconstrained values only
as diagnostics so lower-bound clipping or a frozen state cannot masquerade as
adaptation.

Variable speed therefore creates an ideal opportunity for adaptation exactly
when it makes $\pi_t^\star$ vary. It may still fail in practice if movement is
noise-dominated, the predictable plug-in estimate reacts too late, or the
controller mistakes nonstationarity for covariance.

## Status

| Phase | Name | Status |
|---|---|---|
| 0 | Mathematical and schedule contracts | Complete |
| 1 | Variable-speed infrastructure | Complete |
| 2 | Euclidean design-oracle feasibility screen | Complete - stop |
| 3 | Fisher-risk controller contract and implementation | Complete |
| 4 | Fisher-risk diagnostic screen | Complete - stop |
| 5 | Final realized-actuation challenge | Complete |
| 6 | Exponentially discounted Fisher-risk control | Complete - diagnostic characterization |
| 7 | Theory integration and historical decision | Pending |

## Fisher-Risk Amendment

Phase 2 established that the Euclidean controller is behaving consistently
with its stated objective: even very large coordinate displacement remains
small relative to its estimated Euclidean covariance cost. It does not
establish that Euclidean parameter error is the right loss for predictive
adaptation.

For a predictable Fisher summary $G_t=\widehat{\mathcal I}_{t\mid t-1}$,
define the local Fisher quadratic

$$
\|v\|_{G_t}^2=v^TG_tv.
$$

This quantity is computable for positive-semidefinite and singular estimates;
no inverse or factorization is required. For the true Fisher, it is the
second-order term in local distributional divergence,

$$
2D_{\mathrm{KL}}(P_{\theta_t}\|P_{\theta_t+v})
=v^T\mathcal I(\theta_t)v+O(\|v\|^3).
$$

The Fisher-risk controller therefore changes the controller's estimand from
coordinate parameter MSE to local predictive divergence. It is an
experimental treatment, not a silent algebraic substitution in the existing
theory.

Let

$$
S_t=\widehat d_t^TG_t\widehat d_t,
\qquad
V_{\mathrm{old},t}
=\mathbb E[e_t^TG_te_t\mid\mathcal F_{t-1}],
\qquad
V_{\mathrm{new},t}
=\mathbb E[\epsilon_t^TG_t\epsilon_t\mid\mathcal F_{t-1}].
$$

The applied one-step risk becomes

$$
R_{G,t}(\pi)
=(1-\pi)^2(S_t+V_{\mathrm{old},t})
+\pi^2V_{\mathrm{new},t},
$$

with unconstrained action

$$
\pi_{G,t}^*
=\frac{S_t+V_{\mathrm{old},t}}
{S_t+V_{\mathrm{old},t}+V_{\mathrm{new},t}}.
$$

All three terms must use the same predictable metric. Replacing only
$\|d\theta\|^2$ while retaining Euclidean trace variance would mix two loss
functions and is prohibited.

Under ideal covariance calibration,

$$
\operatorname{Cov}(e_t)\approx q_t\mathcal I_t^{-1},
\qquad
\operatorname{Cov}(\epsilon_t)\approx m_t^{-1}\mathcal I_t^{-1},
$$

the variance terms reduce to $q_td_{\mathrm{eff},t}$ and
$d_{\mathrm{eff},t}/m_t$. Here $d_{\mathrm{eff},t}$ is only a theoretical
interpretation of the Fisher-weighted covariance. The implementation must
estimate the Fisher-weighted residual energy directly and must not invert,
pseudoinvert, or numerically rank the Fisher to obtain it.

The amendment does not replace Euclidean norms used for LFU Taylor control,
optimizer diagnostics, or numerical step limits. It does not change the EWC
objective, original/auxiliary-process distinction, likelihood Fisher
estimand, or direct-EMA Fisher update. The existing rank-8-plus-diagonal EWC
summary supplies $G_tv$ and $v^TG_tv$ cheaply in the network's ordinary
Euclidean parameter chart.

## Frozen Handoff

Unless a phase check-in explicitly reopens one item, preserve:

- the canonical 512-parameter CNN and full-network likelihood Fisher;
- $m=8$ current observations and $K=100$ environmental steps;
- the application-focused path segment $p_0=0$ to $p_{K-1}=.2$;
- 50 L-BFGS inner iterations per learner update;
- rank-8-plus-diagonal Fisher summaries;
- direct EMA Fisher updates with no LFU or HVP calculation;
- fixed $\pi=.05$ as the deployment baseline;
- scale-adjusted plug-in $h=.05$, $\pi_{\min}=.05$, and
  $\pi_{\max}=.95$ for the Fisher-risk amendment;
- Hybrid B32 as the principal applied learner after controller feasibility;
- pre-update evaluation and the exposure semantics established in Plan 3.

The first controller-isolation pilot uses EWC without replay because the
fixed-batch risk was derived for that setting. Hybrid B32 is introduced only
after the controller demonstrates genuine, predictable actuation. Its replay
and archive observations retain Plan 3's clean-recursion semantics.

Retain the environmental-distance trend gain

$$
\gamma_t=1-2^{-\Delta p_t/H_p}.
$$

Do not silently replace it with a step-based EMA. A faster schedule therefore
changes both the displacement signal and the amount of trend updating per
step, while preserving the accepted half-life interpretation.

The historical $h=.20$ was selected on $p\in[0,1]$. Plan 4 observes only
$p\in[0,.2]$, for which the exact range-scaled value is $.04$. Use the nearest
previously tested value $h=.05$ as the primary Fisher-risk setting and retain
$h=.10$ only as a labeled sensitivity. This is a predeclared scale adjustment,
not predictive-outcome tuning. At $\kappa=256$, the principal jump may begin
just before the $.05$ cold-start boundary; report that timing explicitly and
do not let this extreme stress schedule overrule coherent results from
$\kappa\in\{32,64,128\}$.

## Experimental Contracts

### Schedule family

For $t=0,\ldots,K-1$, define the derived normalized step coordinate

$$
r_t=\frac{t}{K-1}.
$$

It is not an additional hyperparameter. For schedule inputs
$p_{\mathrm{start}}$, $p_{\mathrm{end}}$, temporal center $c$, and steepness
$\kappa$, define

$$
L_t=\sigma(\kappa(r_t-c)),
\qquad
S_t=\frac{L_t-L_0}{L_{K-1}-L_0},
$$

$$
p_t^{(\kappa)}
=p_{\mathrm{start}}
+(p_{\mathrm{end}}-p_{\mathrm{start}})S_t.
$$

Freeze the Plan 4 defaults

$$
p_{\mathrm{start}}=0,
\qquad
p_{\mathrm{end}}=.2,
\qquad
c=.5,
\qquad
K=100.
$$

At $c=.5$, the derived prevalence center is

$$
p_{\mathrm{center}}
=\frac{p_{\mathrm{start}}+p_{\mathrm{end}}}{2}=.1,
$$

and maximum speed occurs across the two finite-grid transitions surrounding
$r=.5$. Record `center_p` as a resolved diagnostic rather than an independent
input.

Keep the range-matched linear schedule as the known gentle control:

$$
p_t^{\mathrm{linear}}
=p_{\mathrm{start}}
+(p_{\mathrm{end}}-p_{\mathrm{start}})r_t.
$$

Candidate steepness values are frozen at
$\kappa\in\{8,16,32\}$. Phase 2 selects at most one non-linear schedule for
the applied pilot. Selection uses controller-signal diagnostics, not
predictive outcomes.

After the initial Phase 2 stop decision, the user approved one
breaking-point extension at $\kappa\in\{64,128,256\}$ under the same gate and
$K=100$ contract. This is an explicitly labeled stress extension, not part of
the original frozen candidate set. Float64 represents the negligible tails of
$\kappa=128$ and $256$ as repeated $p_t$ values; their 21 and 35 zero-speed
tail transitions are recorded rather than replaced by artificial movement.

The deterministic Phase 0 schedule audit gives:

| Schedule | Peak $\Delta p$ | Peak/linear | Half-peak transitions | Expanded event window | Expected nines before step 50 |
|---|---:|---:|---:|---:|---:|
| Linear | .00202 | 1.00 | 99 | Not localized | 19.80 |
| $\kappa=8$ | .00419 | 2.07 | 43 | 24 through 76 | 12.38 |
| $\kappa=16$ | .00808 | 4.00 | 21 | 35 through 65 | 6.83 |
| $\kappa=32$ | .01613 | 7.98 | 11 | 40 through 60 | 3.43 |

Here $\Delta p_t=p_t-p_{t-1}$ for $t\geq1$. A half-peak transition satisfies
$\Delta p_t\geq\tfrac12\max_j\Delta p_j$; the expanded event window adds five
steps on each side. The discarded $\kappa=4$ candidate reached only 1.31 times
linear peak speed and classified 87 transitions as high speed, so it was not a
meaningful change-shock treatment.

For the symmetric default,

$$
S_t+S_{K-1-t}=1,
$$

so

$$
p_t^{(\kappa)}+p_{K-1-t}^{(\kappa)}
=p_{\mathrm{start}}+p_{\mathrm{end}}=.2.
$$

Every schedule therefore has the same expected total digit-9 exposure,

$$
\mathbb E N_9
=m\sum_{t=0}^{K-1}p_t
=mK\frac{p_{\mathrm{start}}+p_{\mathrm{end}}}{2}
=80,
$$

and the same 800 total online observations. Only the timing changes. The
Plan 4 generating process ends explicitly at $p=.2$; do not add a plateau or
an unrecorded continuation to $p=1$.

For completeness, a noncentral temporal input has derived prevalence center

$$
p(c)=p_{\mathrm{start}}
+(p_{\mathrm{end}}-p_{\mathrm{start}})
\frac{
1/2-\sigma(-\kappa c)
}{
\sigma(\kappa(1-c))-\sigma(-\kappa c)
}.
$$

### Design-oracle measurement error

The design oracle depends on $\|d\theta_t\|^2$, so reference-fit noise must not
be interpreted as true movement. At selected reference values of $p$, use
independent high-sample fits $\widehat\theta^{(a)}(p)$ and
$\widehat\theta^{(b)}(p)$ to estimate

$$
\nu(p)
=\frac12\mathbb E
\left\|
\widehat\theta^{(a)}(p)-\widehat\theta^{(b)}(p)
\right\|^2.
$$

For an interpolated raw displacement, report both

$$
\widehat d_{t,\mathrm{raw}}^2
=\left\|
\widehat\theta(p_{t+1})-\widehat\theta(p_t)
\right\|^2
$$

and the nonnegative noise-corrected estimate

$$
\widehat d_{t,\mathrm{corr}}^2
=\left[
\widehat d_{t,\mathrm{raw}}^2
-\nu(p_t)-\nu(p_{t+1})
\right]_+.
$$

This correction assumes independent, locally mean-centered reference-fit
errors. Record repeated-fit variation and a bootstrap interval rather than
asserting the assumption silently. Use piecewise-linear interpolation in $p$
for the existing dense reference path and assess its numerical floor by
coarsening that grid. If feasibility depends on interpolation resolution,
generate targeted reference fits rather than choosing a schedule.

The principal Phase 2 gate uses the corrected displacement and the existing
covariance-calibrated residual trace estimate. Raw displacement supplies an
upper sensitivity bound. A schedule must retain its opportunity conclusion
across the recorded reference-fit, interpolation, and trace uncertainty
range. No Fisher inversion or pseudoinversion is permitted.

### Pairing

Within a schedule and replica, all conditions receive identical arrivals. To
couple different schedules without changing either sampling distribution,
draw common uniforms and set

$$
M_{t,b}^{(s)}=\mathbf 1\{U_{t,b}<p_t^{(s)}\}.
$$

Use common ordered digit-9 and non-nine index streams after the mixture draw.
Record schedule identity, resolved $p_t$ values, uniform-stream hash, selected
indices, and resulting content hash. A schedule change must create a new
immutable stream and run identity.

### Controller comparisons

The principal comparison is adaptive versus the frozen fixed $\pi=.05$
policy. A small fixed grid brackets the adaptive behavior and estimates the
best fixed policy in hindsight. Begin with

$$
\pi\in\{.05,.10,.20\}.
$$

Phase 2 may replace `.20` with one rounded value selected by a predeclared
oracle-quantile rule. Let $q_{.90}$ be the interior 90th percentile of bounded
oracle $\pi_t^\star$, round it to the nearest $.05$, and clip it to
$[.10,.50]$. Use the distinct set $\{.05,.10,q_{.90}\}$ as the fixed bracket.
Do not tune fixed controls from predictive metrics.

An adaptive policy earns practical value only if it:

1. improves on frozen fixed $.05$ under the selected change-shock schedule;
2. approaches the best fixed control without schedule-specific retuning;
3. does not obtain its result through materially worse digit-9 discrimination,
   calibration, or old-capability retention; and
4. remains predictable, using information available only through step $t-1$.

Movement of $\pi_t$ alone is not success. If the controller actuates only
after predictive degradation, treat it as a diagnostic rather than a policy.

EWC-only provides the principal theoretically matched controller test. If the
Fisher-risk EWC pilot passes the first part of Phase 5, fixed and adaptive
Hybrid B32 are required in the second part rather than optional. For that
transfer test, $m_t=8$ continues to count only the independent current
arrivals in the controller covariance model. Do not count the 32 reused replay
observations as independent new samples. Until a replay-aware covariance
derivation exists, describe adaptive Hybrid B32 as an applied transfer test
rather than an optimal hybrid controller.

### Outcomes

Preserve complete per-replica trajectories. Report:

- environmental multiclass accuracy as the primary predictive outcome;
- digit-9 OvR accuracy, precision, and recall;
- NLL, Brier score, ECE, and non-nine accuracy;
- expected and realized cumulative digit-9 exposure;
- applied $\pi_t$, unbounded plug-in $\widehat\pi_t^\star$, and bound flags;
- trend norm, residual moment, effective information size, and trace estimate;
- schedule speed $\Delta p_t$, parameter displacement norm, and decision lag;
- learner time, optimizer evaluations, persistent bytes, and peak memory.

Plot metrics against step, $p_t$, and expected cumulative digit-9 exposure.
Add an event-centered view around the maximum $\Delta p_t$. Cross-schedule
plots must not compare equal step numbers as though they were equal $p$ values.

The primary controller-opportunity window is the set of steps
where $\Delta p_t$ is at least half its schedule maximum, plus five updates on
each side. The complete $0\leq p\leq.2$ trajectory and expected-nine-exposure
views remain required application context rather than substitutes for this
event-centered test. Event-window environmental-accuracy AUC is primary;
full-segment AUC is the principal broader secondary outcome.

For treatment inference, use paired replica differences and 95% Student-$t$
intervals. Keep exploratory and confirmatory replicas separate.

## Phase 0: Mathematical and Schedule Contracts

### Goal

Freeze the estimand, schedule family, controller-value criterion, and stopping
rules before implementation or predictive screening.

### Scope

1. Record explicitly that schedule speed changes $\|d\theta_t\|$ but does not
   create a new manifold or Fisher estimand.
2. Derive the best constant policy under the summed one-step risk and its
   conditional excess risk relative to pointwise oracle adaptation. Separately
   define the principal closed-loop comparison by rolling each policy through
   its own $q_t$ recursion.
3. Freeze feasibility thresholds before viewing schedule-screen results. On
   the primary event window:
   - bounded oracle $\pi_t^\star$ exceeds $.07$ on at least ten transitions;
   - its bounded range is at least $.05$; and
   - its bounded closed-loop cumulative risk improves by at least 5% over the
     best bounded constant policy rolled through its own $q_t$ recursion.
   All three conclusions must survive the recorded displacement, trace, and
   interpolation uncertainty analysis.
4. Freeze the normalized logistic formula, candidate $\kappa$ values, pairing
   semantics, application window, and event-centered window
   definition.
5. Define `promote`, `diagnostic-only`, and `stop` decisions before compute.

The numerical thresholds are frozen design guards, not theoretical constants:

- **Promote:** at least one candidate passes every corrected, bounded,
  closed-loop feasibility threshold throughout the uncertainty analysis.
  Select the least steep passing candidate for the predictive pilot.
- **Diagnostic only:** an unbounded or raw-displacement calculation suggests
  adaptation, but no candidate robustly passes the corrected bounded gate.
  Preserve the diagnostic artifact and do not escalate predictive compute.
- **Stop:** no candidate displays material oracle variation or risk
  opportunity even before conservative correction, or the apparent signal is
  entirely a numerical-resolution artifact.

### Verification

- The risk comparison separates oracle opportunity from plug-in quality.
- Equal endpoint and total-exposure identities are proved for every schedule.
- Curvature, speed, and Fisher variation are not used interchangeably.
- No FIM inverse or pseudoinverse is introduced.

### Check-in

Approve the schedule family, feasibility thresholds, and interpretation of
adaptive value before changing configuration schemas.

### Completion record

**Status:** Complete (2026-08-21)

- Distinguished variable speed from path geometry and retained the unique
  likelihood Fisher estimand.
- Froze $K=100$, $m=8$, $p\in[0,.2]$, $c=.5$, and
  $\kappa\in\{8,16,32\}$ with a range-matched linear control.
- Verified equal expected exposure: 800 total arrivals and 80 expected nines
  under every schedule. No endpoint plateau or continuation is allowed.
- Rejected $\kappa=4$ as insufficiently localized; the retained candidates
  reach approximately $2.07$, $4.00$, and $7.98$ times linear peak speed.
- Made bounded closed-loop risk, with policy-specific $q_t$ recursion, the
  principal design criterion; frozen-coefficient and unbounded calculations
  remain diagnostics.
- Froze reference-fit noise subtraction, interpolation sensitivity checks,
  and a strict prohibition on Fisher inversion or pseudoinversion.
- Designated EWC-only as the theoretically matched pilot and Hybrid B32 as a
  required applied transfer test after a successful pilot. Replay samples do
  not inflate the controller's independent-current-sample count.
- Froze the feasibility thresholds and `promote`, `diagnostic-only`, and
  `stop` rules before any oracle-screen outcomes are computed.

**Gate recommendation:** proceed to Phase 1 without reopening the scientific
contracts. Phase 1 may refine storage and schema mechanics, but any change to
the schedule, exposure, controller estimand, or feasibility gate requires a
new check-in.

## Phase 1: Variable-Speed Infrastructure

### Goal

Add schedule-aware, immutable, paired experiment infrastructure while
preserving exact backward compatibility for existing linear runs.

### Scope

1. Add a named schedule specification and stable schedule hash to strict
   configuration parsing.
2. Implement linear and normalized-logistic schedules with stored resolved
   values rather than notebook-time reconstruction.
3. Implement common-uniform stream coupling and schedule-specific immutable
   stream artifacts.
4. Record stepwise $\Delta p_t$, expected exposure, realized exposure, and
   maximum-speed step in trajectory artifacts.
5. Extend command-center preview, resume, collision, and completion checks.
6. Add a tiny CPU smoke experiment containing fixed and adaptive EWC on one
   short logistic schedule.

### Verification

- Unit tests cover endpoints, strict monotonicity, complement symmetry, equal
  expected exposure, deterministic pairing, and schedule hashing.
- Existing linear configurations resolve to their previous values and hashes
  unless an explicit schema migration is requested.
- Conditions within one schedule receive identical observations.
- Controller decisions remain predictable and all smoke runs record zero
  HVPs.
- Completed artifacts remain append-only.

### Check-in

Review the resolved schedules and smoke trajectories before any design-oracle
or GPU pilot.

### Completion record

**Status:** Complete (2026-08-21)

- Added strict optional schedule configuration with unambiguous
  `center_fraction` and derived `center_p` semantics. Legacy configurations
  omit the field during serialization, preserving their exact configuration
  hashes and replica bundle IDs.
- Implemented range-matched linear and normalized-logistic schedule resolution
  with stored $p_t$, $\Delta p_t$, schedule hash, and maximum-speed transition.
- Added schema-v2 scheduled mixture streams based on common uniforms and
  treatment-independent digit-9/non-nine candidate streams. Legacy streams
  retain their schema-v1 generator and RNG consumption.
- Added a separately versioned `schedule_trajectory.json` ledger containing
  expected and realized batch/cumulative digit-9 exposure. Controller and
  Hybrid runs validate this ledger against their immutable stream.
- Added Plan 4 command-center `preview`, `status`, and `run` operations with
  explicit missing, incomplete, invalid, completed, resume, and collision
  handling.
- Completed paired fixed/adaptive EWC CPU smoke runs on the same five-step
  $\kappa=16$ schedule and replica bundle. Both runs share schedule hash
  `7daa46d19115904ee8bd97930029a8c868488af0071a4ec24fd3cf318b2fd0a7`
  and uniform-stream hash
  `bd42fdaabb185819510a09b15f53cb8d8348ff3087e0c1ac965eb5bd06e8fb0e`.
- Verified the unified-$\pi$ contract and exactly zero HVP calculations for
  the direct-EMA smoke. Re-running the command center skipped both completed
  immutable runs and re-audited their artifacts.
- Passed the complete unit suite: 273 tests passed and 2 were skipped.

**Gate recommendation:** proceed to Phase 2 only after reviewing the resolved
schedule and smoke ledgers at this check-in. Phase 2 remains diagnostic-only
and must not train or rank predictive learners.

## Phase 2: Design-Oracle Feasibility Screen

### Goal

Determine whether any candidate schedule contains enough ideal controller
signal to justify training adaptive learners.

### Scope

1. Reuse high-sample path information only as a design diagnostic. For every
   stored outer replica, interpolate each of its 32 repeated-fit parameter
   paths separately before forming adjacent displacements. Estimate and
   subtract the sampling variance of the repeated-fit mean displacement while
   retaining the empirical dependence between neighboring endpoints. Report
   the uncorrected mean-path displacement as an upper sensitivity bound and
   repeat the construction on a factor-two coarsened source grid.
2. Estimate trace terms with the existing covariance-calibrated residual
   mechanism. The primary source is the ten independent `m=8`, fixed
   $\pi=.05$, no-LFU trajectories: pool their oracle-detrended residual and
   scale moments before division, then interpolate the resulting $T(p)$ curve.
   Use plug-in residual moments as a sensitivity analysis. Do not count the
   numerically duplicate adaptive trajectories as additional replicas, and
   never invert or pseudoinvert a Fisher estimate.
3. For each candidate schedule, calculate:
   - $\|d\theta_t\|^2$ and schedule speed;
   - bounded and unbounded oracle $\pi_t^\star$;
   - lower- and upper-bound occupancy;
   - the frozen-coefficient best constant as a local diagnostic;
   - bounded closed-loop risk rollouts for adaptive and fixed policies;
   - absolute and relative closed-loop oracle risk reduction; and
   - sensitivity to uncertainty in the trace and displacement estimates.
4. Select at most one logistic schedule satisfying the Phase 0 feasibility
   gate. Use the least steep candidate that creates a clear signal, avoiding
   an unnecessarily discontinuous stress test.
5. Choose the third fixed control through the frozen oracle-quantile rule.

Use a deterministic nested bootstrap. Resample complete outer replicas so
neighboring $p$ values remain dependent; within each selected reference path,
resample complete repeated-fit trajectories. A feasibility conclusion must
survive the 95% joint bootstrap interval, the factor-two interpolation
coarsening, and the oracle-detrended versus plug-in trace sensitivity.

Do not use historical adaptive-Hybrid artifacts in the primary screen. Their
controller covariance recursion counted current plus replay observations as
$m_t$, whereas Plan 4 freezes $m_t=8$ as the independent current arrivals.
Those immutable artifacts remain valid for their original analysis but answer
a different controller question.

This phase does not rank predictive methods and cannot validate the plug-in
controller. It is artifact-only: do not train a learner or generate a new
reference path unless a targeted fit is later approved because a gate decision
changes over the stored uncertainty band. If no schedule passes, stop this
plan and retain fixed $.05$.

### Verification

- Diagnostic inputs identify their source artifacts and assumption status.
- The ten fixed-policy source trajectories are the only outer trace replicas;
  adaptive duplicates and historical Hybrid traces are explicitly excluded.
- Bootstrap units are complete outer replicas and complete repeated-fit paths,
  never individual $p$ points.
- Conclusions are stable across the recorded trace/displacement sensitivity
  range.
- The selected schedule changes speed rather than total observations,
  expected class composition, optimizer budget, or model architecture.
- An artifact-only notebook displays schedule, speed, oracle $\pi$, and risk
  opportunity without training.

### Check-in

Choose `stop` or select one variable-speed schedule and its fixed-policy
bracket for a predictable-controller pilot.

### Completion record

**Status:** Complete (2026-08-21)

- Built an artifact-only design-oracle analysis from the ten independent
  `m=8`, fixed-$\pi=.05$, no-LFU source trajectories. Numerically duplicate
  adaptive runs were not counted as replicas, and historical Hybrid traces
  were excluded because their controller used incompatible batch-size
  semantics.
- Reconstructed $T(p)=\operatorname{tr}[\mathcal I(\theta^\star(p))^{-1}]$
  by pooling oracle-detrended residual and scale moments before division. The
  plug-in residual construction was retained as a sensitivity analysis; no
  Fisher inverse or pseudoinverse was calculated.
- Interpolated each of the 32 repeated-fit paths within each outer replica
  before forming displacements. The paired correction retained adjacent-fit
  dependence and removed only `.12%` to `.33%` of median raw squared movement,
  depending on schedule. Factor-two coarsening changed median corrected
  movement by approximately `2.9%` to `4.1%`.
- Ran 1,000 deterministic nested bootstrap resamples, using complete outer
  replicas and complete repeated-fit paths as the two sampling units. Only 10
  of the 200 strict reference checkpoints through $p=.2$ met their original
  convergence gate, but the decision was unchanged under raw displacement,
  coarsened interpolation, plug-in trace, and every bootstrap draw.
- None of the seven schedules produced a bounded oracle action above `.07`, a
  nonzero bounded action range, or positive adaptive risk reduction. Every
  bounded rollout remained at $\pi_{\min}=.05$, and the best bounded constant
  was also `.05`.
- On the bounded rollout's own $q_t$ sequence, the pre-clipping oracle was
  approximately `.023` to `.025` in the original logistic screen. The
  breaking-point extension increased it only to `.0254`, `.0263`, and `.0282`
  for $\kappa=64,128,256$.
- The extreme schedules reached approximately `15.86`, `30.93`, and `56.36`
  times linear peak speed. At $\kappa=256$, one transition had
  $\Delta p\approx.114$ and corrected displacement norm about `1.20`, yet the
  median event trace remained near `3,298` and the `.05` lower bound remained
  active.
- A near-zero-resolved constant-policy grid showed no hidden unconstrained
  advantage. The myopic adaptive rollout ranged from essentially tied at low
  steepness to about `.96%` worse than the best constant at $\kappa=256$ once
  its policy-specific $q_t$ consequences were included.
- Wrote the canonical immutable schema-v5 analysis artifact
  `cache/mnist_experiment/plan4/analysis/phase2__ba21cc259451` and the
  artifact-only [plan4_oracle_results.ipynb](plan4_oracle_results.ipynb).
- Passed the full unit suite: 281 tests passed and 2 were skipped.

**Gate recommendation:** `stop`. Under the frozen Plan 4 design, changing only
the speed of this $p\in[0,.2]$ path does not create enough ideal controller
opportunity to justify the original Euclidean predictive pilot. Retain fixed
$\pi=.05$ for the practical baseline. The Fisher-risk amendment beginning in
Phase 3 is the approved new mathematical treatment; it must pass its own cheap
diagnostic gate rather than reopening or reinterpreting this completed result.

## Phase 3: Fisher-Risk Controller Contract and Implementation

### Goal

Implement a second, explicitly named controller-risk model without changing
the completed Euclidean controller, immutable artifacts, or learner update
semantics.

### Scope

1. Audit the controller timing and stored state before choosing estimator
   indices. Freeze $G_t$ from the Fisher summary available before the action
   at step $t$; neither the current observations nor their resulting Fisher
   update may influence the metric used to choose their own $\pi_t$.
2. Add an explicit controller-risk mode with at least:
   - `euclidean`, reproducing the existing controller exactly; and
   - `fisher`, using the normalized per-observation EWC Fisher summary rather
     than its accumulated precision-mass scaling.
3. Compute $v^TG_tv$ through each representation's existing matrix-vector or
   quadratic-form operation. Do not densify the rank-8-plus-diagonal summary
   and do not require positive definiteness.
4. Replace the trend signal by
   $S_t=\widehat d_t^TG_t\widehat d_t$. Preserve the accepted environmental
   half-life $h=.20$ and predictable vector-EMA trend construction.
5. Extend the existing detrended covariance-calibration recursion to
   Fisher-weighted residual energy. If the current scalar recursion identifies
   only a common covariance scale, estimate that scale first and derive
   $V_{\mathrm{old},t}$ and $V_{\mathrm{new},t}$ through the recorded $q_t$
   and $m_t$ factors. Record every intermediate scalar so the action can be
   reconstructed offline.
6. Keep all zero-information cases explicit. When Fisher energy and estimated
   uncertainty are both numerically zero, fall back to the frozen
   $\pi_{\min}$ policy and record the reason; do not create a hidden ridge in
   the scientific formula.
7. Introduce new controller-state, trajectory-metric, and artifact-schema
   versions. Existing completed artifacts and legacy configuration hashes must
   remain readable and unchanged.
8. Add a tiny paired CPU smoke containing fixed `.05`, Euclidean adaptive,
   and Fisher adaptive EWC on one scheduled stream.

The implementation may use a small numerical floor only for division and
finite-value protection. Such a floor must be recorded and must not be
presented as Fisher damping or as part of the estimand.

### Verification

- Unit tests evaluate Fisher quadratics for dense, diagonal, singular, and
  low-rank-plus-diagonal positive-semidefinite matrices.
- Rescaling or factoring a structured representation without changing its
  represented matrix leaves the controller action unchanged.
- `euclidean` mode reproduces an existing deterministic smoke trajectory and
  artifact hash under the legacy schema path.
- Fisher actions use only predictable state and can be reconstructed from
  trajectory artifacts.
- No Fisher inverse, pseudoinverse, eigendecomposition, or numerical-rank
  estimate is introduced.
- The full unit suite and the new CPU integration smoke pass.

### Check-in

Review estimator timing, Fisher-energy scales, zero-information behavior, and
the paired smoke before running a design screen. No predictive GPU run is
authorized by this phase.

### Completion Record

**Status:** Complete (2026-08-21)

- Added explicit `euclidean` and `fisher` controller-risk modes under config
  schema 17, controller artifact schema 9, and metric schema 13. Schemas 4
  through 16 continue to resolve implicitly to Euclidean risk and omit the new
  field from canonical serialization; the prior Plan 4 adaptive smoke retained
  its exact configuration hash
  `f914ad42fa5c11779ce91ed07b23b70efb12a4efb6c92793a014d9818b8267b2`.
- Exposed the normalized auxiliary Fisher available before each action. The
  current observations and their Fisher update cannot affect the metric used
  to choose their own $\pi_t$.
- Reused the accepted residual calibration with
  $r_t^TG_tr_t$ as its numerator and
  $\pi_t^2(q_t+1/m_t)$ as its scale observation. Stored the resulting signal
  energy, uncertainty scale, old/new covariance risks, and complete risk
  numerator and denominator for offline reconstruction.
- Evaluated dense, diagonal, singular, and low-rank-plus-diagonal Fisher
  quadratics through representation-native operations. No dense conversion,
  inversion, pseudoinversion, eigendecomposition, damping, or numerical-rank
  calculation was added to the controller.
- Added an explicit lower-bound fallback only when a deployable Fisher plug-in
  decision has exactly zero movement and residual information after cold
  start. Numerical floors remain limited to finite division protection and
  are recorded separately from the estimand.
- Completed paired five-step CPU smoke runs for fixed `.05`, Euclidean
  adaptive, and Fisher adaptive policies. They shared schedule hash
  `7daa46d19115904ee8bd97930029a8c868488af0071a4ec24fd3cf318b2fd0a7`
  and uniform-stream hash
  `bd42fdaabb185819510a09b15f53cb8d8348ff3087e0c1ac965eb5bd06e8fb0e`,
  and calculated exactly zero HVPs.
- Reconstructed all 15 smoke decisions from stored scalar state. At the final
  smoke point, after the intentionally short cold start, the Euclidean and
  Fisher plug-ins produced finite distinct raw actions of approximately
  `.1033` and `.1315`. This verifies treatment actuation only; it is not a
  predictive or scientific comparison.
- Passed the complete unit suite: 290 tests passed and 2 were skipped.

**Gate recommendation:** proceed to the Phase 4 artifact and diagnostic-state
audit. Existing artifacts may be reused only when they store the predictable
Fisher state required by the new risk. No predictive GPU run is authorized.

## Phase 4: Fisher-Risk Diagnostic Screen

### Goal

Determine cheaply whether the Fisher-risk objective creates coherent adaptive
signal on the same linear and variable-speed paths that defeated the
Euclidean objective.

### Scope

1. Reuse completed immutable reference, EWC, and schedule artifacts whenever
   they contain the predictable Fisher state required by the new estimator.
   Record any missing state explicitly. Never reconstruct a quantity from an
   incompatible artifact merely to avoid a small diagnostic run.
2. If existing artifacts are insufficient, run only the minimum diagnostic
   trajectories needed to produce the Fisher summary, trend, residual energy,
   and $q_t$ recursion. Do not launch a replica grid or Hybrid learner.
3. Screen the linear schedule and the existing logistic stress family. Begin
   with $\kappa\in\{32,64,128,256\}$; do not add new schedules unless all
   retained cases fail for a diagnosed resolution reason.
4. Use $h=.05$ for the principal diagnostic and $h=.10$ only as a sensitivity.
   Report the ungated predictable plug-in and the operational cold-start
   action separately. A signal visible only before the cold-start gate is
   `diagnostic-only`, not a failure of Fisher geometry and not eligible for a
   predictive pilot.
5. Compare, at every step:
   - Euclidean and Fisher movement energies;
   - Euclidean trace and Fisher-weighted uncertainty terms;
   - unbounded and clipped controller actions;
   - lower-bound occupancy and action range;
   - event-window response timing; and
   - closed-loop one-step risk against fixed `.05` and the best fixed action
     under the same Fisher-risk state recursion.
6. Check estimator dependence by comparing the predictable lagged Fisher
   metric with a separately labeled contemporaneous diagnostic. Only the
   predictable version is eligible for promotion.
7. Write an immutable diagnostic artifact and an artifact-only notebook panel
   that exposes the numerator and denominator of $\pi_{G,t}^*$ rather than
   showing only its clipped output.

### Gate

- **Promote:** Fisher-risk $\pi_t$ leaves `.05` coherently in at least one
  predeclared event window, the action responds in the expected direction,
  and its bounded closed-loop Fisher risk improves by at least 5% over fixed
  `.05` without numerical-floor dependence.
- **Diagnostic only:** the action tracks Fisher movement but is too late,
  dominated by uncertainty, or offers less than 5% conditional opportunity.
- **Stop:** the action remains pinned, becomes unstable, depends materially on
  self-coupled contemporaneous information, or obtains apparent signal only
  from numerical safeguards.

These thresholds screen controller signal, not predictive effectiveness. A
passing result authorizes only the small paired pilot in Phase 5.

### Verification

- Every diagnostic identifies its source artifact, metric timing, and Fisher
  representation.
- Singular and low-rank summaries produce finite energies without special
  rank handling.
- The Euclidean columns reproduce the completed Phase 2 conclusion.
- The notebook performs no training or Fisher estimation.

### Check-in

Choose `stop`, retain the controller as a diagnostic, or freeze one Fisher-risk
configuration and one schedule for a paired predictive pilot.

### Completion Record

**Status:** Complete (2026-08-21)

- Audited the completed Plan 2 artifacts before adding compute. They retain
  full reference-optimum parameter paths, learner displacements, oracle
  residual vectors, and the high-sample initial Fisher, but only three
  auxiliary-Fisher checkpoints. They cannot reconstruct a predictable
  per-step Fisher metric without new score observations.
- Implemented an immutable score-only diagnostic replay. It interpolates each
  source's high-sample reference path onto the linear and
  $\kappa\in\{32,64,128,256\}$ schedules, regenerates paired arrivals from the
  recorded partitions and seeds, calculates exact per-sample scores, and
  maintains the rank-8-plus-diagonal Fisher by direct EMA using the
  operational $\pi_t$. It performs no learner optimization, HVP, LFU,
  Fisher inverse, pseudoinverse, or dense numerical-rank calculation.
- Used three independent fixed-$.05$ source trajectories, $m=8$, $K=100$,
  primary $h=.05$, and sensitivity $h=.10$. The expected controller-state
  recursion propagates the interpolated true displacement and separately
  accounts for source-calibrated learner covariance; this is a diagnostic
  expectation calculation, not a predictive learner trajectory.
- Corrected the scheduled-stream validator to admit explicitly recorded
  zero-speed tail transitions. It now matches the nondecreasing schedule
  contract used by the numerically saturated $\kappa=128$ and $256$ tails;
  no artificial movement was injected.
- The operational Fisher controller remained exactly at $\pi_{\min}=.05$ for
  every source, schedule, and half-life. Removing cold start did not reveal a
  hidden response: across nonlinear event windows, the largest within-source
  ungated action range was approximately `.00209`, far below the `.05` gate.
- The bounded Fisher design oracle and the Euclidean control also remained at
  `.05`, yielding zero operational risk reduction against fixed `.05`.
  On the selected $\kappa=32$ summary, even the most favorable transition had
  Fisher movement energy only about `.00067` times the new-observation
  covariance risk. Fisher reweighting therefore did not repair the
  signal-to-uncertainty imbalance.
- A deliberately self-coupled current-batch Fisher diagnostic was noisy
  rather than coherently shock-responsive. Its nonlinear maxima stayed near
  `.025` to `.032`; one linear-path source spiked to approximately `.156`.
  This does not justify allowing current observations to select their own
  weight.
- Wrote immutable schema-v1 artifact
  `cache/mnist_experiment/plan4/fisher_analysis/phase4__571e2543d083` and the
  artifact-only [plan4_fisher_results.ipynb](plan4_fisher_results.ipynb). The
  three-source diagnostic consumed 12,000 score gradients and completed in
  approximately 13 seconds of measured schedule time.
- Passed the complete unit suite: 296 tests passed and 2 were skipped. The
  results notebook also passed the artifact-only validator in under 2 seconds.

**Gate recommendation:** `stop`. Retain fixed $\pi=.05$ for the practical
baseline. The Fisher-risk controller remains mathematically coherent and may
be useful as an offline diagnostic, but this path supplies no evidence that a
predictive adaptive-policy experiment would justify its compute cost.

## Phase 5: Final Realized-Actuation Challenge

### Goal

Give adaptive $\pi$ one direct, oracle-free challenge on realized stochastic
learner trajectories. First establish that a predictable controller actually
actuates under at least one frozen speed treatment; inspect predictive outcomes
only after schedule selection is complete.

### Scope

1. Run one paired EWC-only development replica on each of `linear` and
   $\kappa\in\{32,64,128,256\}$. Compare exactly:
   - fixed $\pi=.05$;
   - the unchanged Euclidean adaptive controller; and
   - Fisher-risk adaptive control with $h=.05$.
   Conditions within one schedule share initialization, arrivals, optimizer
   settings, and every non-treatment seed. Schedule streams use common random
   numbers under the existing threshold coupling.
2. Select a schedule using controller behavior only. Predictive accuracy,
   NLL, calibration, and resource outcomes are sealed until the actuation
   artifact and selected schedule are immutable. A schedule passes only when,
   in its predeclared event window, the predictable Fisher controller:
   - exceeds `.07` on at least ten transitions;
   - has action range at least `.05`;
   - has positive movement/action correlation; and
   - reduces its own summed one-step Fisher-risk estimate by at least 5%
     relative to applying fixed `.05` at the same pre-decision states.
   Rank passing schedules by estimated risk reduction, then action range, then
   lower $\kappa$. The linear schedule is a control and cannot win a tie over
   a passing nonlinear schedule.
3. Stop immediately when no schedule passes. This is the final
   realized-actuation falsification; noise-driven action movement or a
   predictive accident does not reopen schedule tuning.
4. If one schedule passes, freeze it and run three fresh paired EWC-only
   replicas using source replicas 2 through 4. Only then unseal and report
   predictive outcomes.
5. Preserve $m=8$, $K=100$, $p\in[0,.2]$, 50 L-BFGS iterations, direct-EMA
   Fisher updates, and rank-8-plus-diagonal summaries.
6. Use the oracle-free capacity-zero EWC recursion established in Plan 3.
   The controller metric is the archive Fisher available before its action;
   the current observations and resulting Fisher update cannot choose their
   own $\pi_t$. No reference-optimum fit is generated or consumed.
7. Report action timing and decomposition alongside environmental accuracy,
   digit-9 OvR accuracy, precision, recall, NLL, ECE, non-nine accuracy,
   optimizer evaluations, time, and memory.
8. If Fisher adaptive improves over fixed `.05` without a material secondary
   failure, run the same three-condition comparison for Hybrid B32. Replay
   observations remain excluded from $m_t$ under the existing clean-recursion
   semantics.

### Implementation Checkpoint

- Added schema 18 for oracle-free hybrid/EWC runs whose controller may use the
  predictable rank-8-plus-diagonal archive Fisher. Schema 17 remains the
  non-replay Fisher-controller schema; all earlier schema identities remain
  readable and unchanged.
- Froze immutable 15-run development bundle
  `cache/mnist_experiment/plan4/challenge/bundles/plan4-actuation-screen__r0001__b9918dae8a95`.
  It crosses five schedules with fixed `.05`, historical Euclidean adaptive
  $h=.20$, and Fisher adaptive $h=.05$ on one common outer replica.
- The selection analysis reads only controller state, controller decisions,
  acceptance diagnostics, schedule values, and provenance hashes. Predictive
  classification and resource fields remain sealed until a schedule passes.
- A five-step CPU integration run verified that the pre-update Fisher chooses
  the action and that the identical action is consumed by learner EWC and
  archive EMA. Its Fisher action reached approximately `.401`; HVP count was
  zero. This demonstrates plumbing, not scientific efficacy.
- Passed the complete unit suite before launch: 298 tests passed and 2 were
  skipped.

### Pilot Gate

- **Promote:** after passing the actuation screen, Fisher adaptive improves mean paired event-window environmental
  accuracy AUC over fixed `.05`, is not clearly dominated by Euclidean
  adaptive, and introduces no material digit-9, retention, calibration, or
  resource regression.
- **Diagnostic only:** it actuates coherently but does not improve predictive
  outcomes or transfers poorly to Hybrid B32.
- **Stop:** it is pinned, unstable, predictively worse, or reproducibly
  dominated by fixed `.05`.

### Verification

- Pilot runs use new immutable identities and are never pooled silently with
  Plan 3 or the completed Euclidean Plan 4 screen.
- The schedule-selection artifact contains no predictive metrics and records
  the hash of every still-sealed run artifact used to calculate actuation.
- Fixed, Euclidean, and Fisher decisions all identify their risk metric;
  Fisher decisions use the pre-update rank-8-plus-diagonal archive summary.
- Every predictive comparison is paired and displays individual trajectories
  as well as mean differences.
- No panel contains more than four conditions.
- The result notebook cannot train, repair, or complete artifacts.

### Check-in

Decide whether the Fisher-risk controller merits confirmatory replication,
remains a useful diagnostic, or should be retired without changing the
mathematical overview.

### Completion Record

**Status:** Complete (2026-08-21)

- Ran the immutable 15-condition CPU screen
  `cache/mnist_experiment/plan4/challenge/bundles/plan4-actuation-screen__r0001__12f3d0ce3ac9`.
  The five schedules share initialization hash
  `b6c3bd6ae8b0006527b4675b8ba362c81e26fff7860c35d6422b2b62c9569aae`
  and common-uniform hash
  `366b4acc92406093f81a246a7f21429989b5ad3eed247cc4fbd744efa394b4a5`.
- Derived each scheduled stream from the exact archived Plan 2 model and
  partitions. This avoided a device-dependent initialization refit while
  retaining immutable schedule-specific replica provenance.
- The Fisher-risk controller did actuate predictably. Its maximum action rose
  from `.128` on the linear path to `.353` at $\kappa=256$; event-window
  signal/action correlations ranged from `.49` to `.83`, with no missing-Fisher
  fallback.
- Sharper schedules produced material same-state Fisher-risk reductions versus
  fixed `.05`: `9.1%` at $\kappa=128$ and `18.4%` at $\kappa=256$. They did not
  sustain actions above `.07` for the required ten event transitions: the
  counts were 7 of 13 and 5 of 11, respectively.
- Gentler schedules supplied enough transitions but too little risk advantage.
  Linear and $\kappa=32$ reached 19 and 11 qualifying transitions, but reduced
  same-state risk by only `1.9%` and `1.5%`.
- The historical Euclidean controller remained at `.05` throughout linear,
  $\kappa=32$, and $\kappa=64$. It moved only outside the predeclared event
  window for the two sharpest schedules and supplied no passing event signal.
- Wrote immutable controller-only schema-v1 analysis
  `cache/mnist_experiment/plan4/challenge/analysis/phase5_actuation__266e6f391abb`
  and the artifact-only
  [plan4_challenge_results.ipynb](plan4_challenge_results.ipynb). Predictive
  outcomes remained sealed because schedule selection failed.
- Passed the complete unit suite before launch: 299 tests passed and 2 were
  skipped.

**Gate recommendation:** `stop`. Fisher-risk adaptation is a coherent and
responsive diagnostic under abrupt movement, but no schedule passed the
predeclared duration-and-value gate. Do not spend confirmatory or Hybrid B32
compute; retain fixed $\pi=.05$ as the practical baseline.

### Exploratory Floor Sensitivity

**Status:** Complete (2026-08-21)

After closing the confirmatory gate, run one explicitly exploratory sensitivity
study at $\pi_{\min}=.025$. This value is motivated by the realized controller:
the nonlinear schedules' ungated event minima were approximately `.0247` to
`.0253`, while the `.05` floor clipped 56% to 75% of their decisions.

Reuse the exact five schedules, initialization, common-random streams, and
fixed-`.05` incumbent. Add only fixed $\pi=.025$ and Fisher adaptive
$\pi_{\min}=.025$, $h=.05$. Report controller behavior and paired predictive
trajectories for environmental accuracy, digit-9 OvR accuracy, precision,
recall, non-nine accuracy, NLL, ECE, optimizer work, and time. One development
replica supplies descriptive evidence only; it cannot reopen the completed
promotion gate or support a confirmatory claim.

- Ran immutable bundle
  `cache/mnist_experiment/plan4/challenge/bundles/plan4-floor-sensitivity__e3757c2ac286`.
  It reused all five completed fixed-`.05` controls and added ten paired CPU
  trajectories without refitting initialization or regenerating streams.
- Lowering the floor caused material actuation. Adaptive maximum $\pi$ ranged
  from `.205` to `.422`, and its internal event-risk reduction versus fixed
  `.025` ranged from `5.1%` to `27.4%`.
- Those internal gains did not predict learner quality. Relative to fixed
  `.025`, adaptive control reduced mean environmental accuracy in every
  schedule by `1.8` to `8.6` percentage points, increased mean NLL by `.97` to
  `3.06`, and increased ECE by `4.6` to `15.7` percentage points. Digit-9 OvR
  accuracy, precision, and recall were also lower on every complete-trajectory
  mean.
- Fixed `.025` descriptively outperformed fixed `.05` across the five paired
  schedules: its schedule-averaged mean differences were `+3.86` points in
  environmental accuracy, `+0.69` in digit-9 OvR accuracy, `+5.26` in
  precision, `+2.56` in recall, and `+3.69` in non-nine accuracy. Mean NLL was
  lower by `1.53` and ECE by `9.04` percentage points. Schedules are coupled
  sensitivity treatments, not five independent replicas, so these averages
  are descriptive rather than inferential.
- Optimizer budgets and measured resource use remained effectively matched.
  Every learner performed 4,950 L-BFGS iterations; total wall time stayed near
  65 to 67 seconds per trajectory.
- Wrote immutable schema-v1 analysis
  `cache/mnist_experiment/plan4/challenge/floor_analysis/floor_sensitivity__b022b149d154`
  and extended
  [plan4_challenge_results.ipynb](plan4_challenge_results.ipynb) with paired
  action and predictive plots.

**Exploratory recommendation:** retain fixed $\pi=.025$ as a promising applied
candidate for future independent replication. Do not promote the Fisher-risk
adaptive policy: its optimized internal risk is empirically misaligned with
the predictive objective on this path.

## Phase 6: Exponentially Discounted Fisher-Risk Control

### Goal

Determine whether the Fisher-risk controller contains a useful slowly varying
signal that was obscured by harmful instantaneous action noise. Replace the
instantaneous minimizer by the minimizer of an exponentially discounted
history of estimated one-step risks, then compare it directly with historical
fixed $\pi=.05$, incumbent fixed $\pi=.025$, and the completed unsmoothed
controller.

This phase tests a new regularized adaptive controller. It does not reopen the
completed Phase 5 promotion gate or reinterpret its exploratory floor study as
confirmatory evidence.

### Motivation And Estimand

Write the predictable Fisher-risk estimate as

$$
\widehat R_t(\pi)=(1-\pi)^2A_t+\pi^2B_t,
\qquad
A_t=\widehat S_t+\widehat D_tq_t,
\qquad
B_t=\widehat D_t/m_t.
$$

The instantaneous controller uses

$$
\widetilde\pi_t=\frac{A_t}{A_t+B_t}.
$$

A plain EMA of $\widetilde\pi_t$ gives equal influence to ratios calculated at
states where the action is consequential and states where the estimated risk
is nearly flat. Instead, for an action half-life $H_\pi$ measured in accepted
updates, let

$$
\gamma_\pi=1-2^{-1/H_\pi},
$$

$$
\bar A_t=(1-\gamma_\pi)\bar A_{t-1}+\gamma_\pi A_t,
\qquad
\bar B_t=(1-\gamma_\pi)\bar B_{t-1}+\gamma_\pi B_t,
$$

and apply

$$
\pi_t^{\mathrm{EDR}}
=\operatorname{clip}\left(
\frac{\bar A_t}{\bar A_t+\bar B_t},
\pi_{\min},\pi_{\max}
\right).
$$

Conditional on the predictable coefficient history, this action minimizes the
exponentially discounted frozen-coefficient objective

$$
J_t(\pi)=\sum_{k\leq t}(1-\gamma_\pi)^{t-k}\widehat R_k(\pi).
$$

This is not an exact closed-loop optimum: applied actions change later $q_t$,
learner parameters, and Fisher summaries. The predictive experiment tests
whether the conditional approximation remains useful after that feedback is
restored.

Call this policy **exponentially discounted risk control** (`edr`). The action
half-life is distinct from the existing trend-estimation half-life: the latter
estimates $\widehat d_t$ and $\widehat D_t$, whereas $H_\pi$ controls how much
historical estimated risk informs the operational action.

### Frozen Design

- Use Fisher risk, EDR $\pi_{\min}=.01$, $\pi_{\max}=.95$, controller trend
  half-life $h=.05$, $m=8$, $K=100$, and $p\in[0,.2]$.
- Preserve no LFU, direct-EMA rank-8-plus-diagonal Fisher summaries, 50
  L-BFGS iterations, and the oracle-free capacity-zero EWC recursion.
- Use EWC-only for this mechanism test. Hybrid B32 is not authorized unless
  independent EWC confirmation later demonstrates dynamic value.
- Use $H_\pi=4$ accepted updates and $\pi_{\min}=.01$ as the predeclared
  primary treatment. $H_\pi\in\{2,8\}$ and
  $\pi_{\min}\in\{.02,.03\}$ are artifact-only sensitivity calculations and
  cannot replace the primary values using predictive outcomes.
- Reuse completed fixed-$.05$, fixed-$.025$, and unsmoothed Fisher-adaptive
  $\pi_{\min}=.025$ trajectories from the exact five Phase 5 schedules. Fixed
  `.05` is the historical baseline from before adaptive $\pi$ nominated a
  lower operating region; fixed `.025` is the best presently observed fixed
  policy. Add only the primary EDR treatment after its implementation and
  smoke gate passes.

### Scope

1. Reconstruct the available $A_t$, $B_t$, instantaneous action, and event
   windows from completed Phase 5 artifacts. Apply EDR with
   $H_\pi\in\{2,4,8\}$ and final-action floors
   $\pi_{\min}\in\{.01,.02,.03\}$ without learner training. Report unclipped
   and clipped action range, total variation, peak attenuation, event response
   area, peak delay, lower-bound occupancy, and same-state estimated risk.
2. Treat that artifact replay only as an assumption check. Filtered actions do
   not alter the archived learner, $q_t$, Fisher, or later controller states,
   so they are not counterfactual predictive trajectories and cannot establish
   treatment value.
3. Proceed with primary $H_\pi=4$, $\pi_{\min}=.01$ unless it fails to
   attenuate raw action variation, erases the sustained event response, or
   remains pathologically bound at `.01`. If any occurs, stop for a check-in
   rather than selecting another half-life or floor from predictive
   performance.
4. Add explicit EDR controller state containing $\bar A_t$ and $\bar B_t$.
   Accumulate the unbounded, nonnegative predictable coefficients without
   clipping them. During the existing cold start, apply fixed $\pi=.025$ while
   accumulating those coefficients. After cold start, clip only the final EDR
   action to $[.01,.95]$. If the accumulated denominator is exactly zero,
   apply $\pi=.025$ and record the fallback; do not introduce a pseudo-count
   or hidden prior mass.
5. Freeze $G_t$, $A_t$, and $B_t$ before the current observations. The action
   must remain predictable, and the same realized $\pi_t$ must be consumed by
   learner EWC, archive consolidation, Fisher EMA, and the $q_t$ recursion.
6. Introduce new configuration, metric, controller-state, and artifact-schema
   versions. Preserve all prior configuration hashes and immutable artifacts.
   Record instantaneous and smoothed coefficients, instantaneous proposal,
   applied EDR action, cold-start state, and numerical fallback reason at each
   transition.
7. Run a tiny paired CPU smoke before scientific compute. Fixed `.05`, fixed
   `.025`, raw adaptive, and EDR conditions must share initialization,
   observations, schedule, and all non-treatment seeds.
8. Run only the five new EDR development trajectories, one on each existing
   linear and $\kappa\in\{32,64,128,256\}$ schedule. Reuse the completed
   fixed-$.05$, fixed-$.025$, and raw-adaptive artifacts for paired
   comparisons.
9. Report complete and event-window trajectories for environmental accuracy,
   digit-9 OvR accuracy, precision, recall, non-nine accuracy, NLL, ECE,
   optimizer evaluations, learner time, persistent memory, and every action
   diagnostic. NLL is the primary predictive outcome because the Fisher-risk
   estimand is local KL divergence.

### Interpretation Gate

- **Dynamic success:** EDR moves materially and coherently around its local
  operating level, improves predictive NLL over both fixed `.025` and `.05`,
  introduces no material secondary predictive or calibration regression, and
  outperforms the unsmoothed policy. This authorizes a separately specified
  fresh-replica confirmation.
- **Automatic-calibration result:** the unclipped EDR proposal stabilizes near
  `.025` without material `.01` floor occupancy, improves over fixed `.05`,
  approximately matches fixed `.025`, and avoids the raw controller's damage.
  This supports automatic operating-level calibration but not dynamic
  adaptation.
- **Failure or floor-driven result:** EDR follows the imposed floor, retains
  harmful excursions, responds principally after the event, becomes
  numerically or feedback unstable, or predictively underperforms fixed
  `.025`. Retire the applied controller and retain only fixed `.025` for future
  independent confirmation.

The five schedules share one outer development replica and are not independent
statistical units. No outcome from this phase alone supports a confirmatory
claim.

### Verification

- Unit tests prove that coefficient EMA minimizes the explicitly accumulated
  discounted quadratic risks and that $H_\pi$ maps to the intended update-step
  half-life.
- Decision reconstruction from stored coefficients exactly reproduces every
  EDR action, including cold-start and zero-denominator behavior.
- Changing $\pi_{\min}$ changes only final action clipping; instantaneous and
  accumulated risk coefficients remain identical under artifact-only floor
  sensitivities.
- Legacy fixed and unsmoothed controller tests retain their exact outputs and
  configuration identities.
- The EDR smoke records no current-batch leakage, Fisher inversion,
  pseudoinversion, eigendecomposition, HVP, or LFU operation.
- The artifact-only notebook cannot train, resume, repair, or mutate a run and
  never presents the filter-only replay as a predictive counterfactual.

### Check-in

Review the artifact-only smoothing screen before implementing or running the
five predictive trajectories. After those trajectories, classify the result
as dynamic success, automatic calibration, or failure/floor-driven before
considering fresh replicas or theory changes.

### Completion Record

**Status:** Complete - stop (2026-08-22)

- The immutable coefficient-replay screen
  `cache/mnist_experiment/plan4/edr/screen/edr_screen__014849bf996e`
  passed its assumption gate. Primary $H_\pi=4$, $\pi_{\min}=.01$ reduced
  mean nonlinear action total variation to `34.1%` of the raw controller while
  retaining `48.5%` of its event-response area. Post-cold floor occupancy was
  zero.
- Added schema-v20 EDR configuration and schema-v11/v15 hybrid artifacts and
  metrics. The accepted transaction stores separate $\bar A_t,\bar B_t$ state
  and applies exactly one predictable $\pi_t$ to learner EWC, archive
  consolidation, Fisher EMA, and the $q_t$ recursion. Legacy schemas retain
  their prior hashes.
- The paired CPU smoke completed without current-batch leakage, inversion,
  eigendecomposition, HVP, or LFU work. Cold-start accumulation, final-action
  clipping, checkpoint state, and exact cross-consumer actuation were checked
  directly.
- Ran immutable bundle
  `cache/mnist_experiment/plan4/edr/bundles/plan4-edr__48f8dc7a8f6b` in a
  detached `tmux` session. It reused 15 completed controls and added the five
  EDR trajectories. All 20 entries validate as complete; each new 100-step
  trajectory required approximately 64 seconds.
- EDR was not floor-driven, but its operating level remained too high. Mean
  applied $\pi$ ranged from `.043` to `.088` by schedule, with maxima from
  `.074` to `.277`; post-cold `.01` floor occupancy was zero everywhere.
  Exact-zero denominator fallbacks occurred only before meaningful exposure
  (`p` no larger than $5.5\times10^{-18}$), including the long numerically
  flat prefixes of the two sharpest sigmoid schedules.
- Relative to fixed $\pi=.025$, EDR increased complete-trajectory mean NLL on
  every schedule (`+.46` to `+1.95`; schedule average `+1.45`). It also reduced
  schedule-averaged environmental accuracy by `2.93` percentage points and
  digit-9 OvR accuracy by `.82` points, while increasing ECE by `6.52` points.
- Smoothing recovered limited signal relative to the raw adaptive policy:
  schedule-averaged NLL was `.14` lower and environmental accuracy `.54`
  points higher. Those gains were path-dependent and did not overcome the
  much larger deficit to fixed `.025`.
- Wrote immutable schema-v2 analysis
  `cache/mnist_experiment/plan4/edr/analysis/edr_predictive__f83d7cbf0459`
  and artifact-only
  [plan4_edr_results.ipynb](plan4_edr_results.ipynb). The notebook executes in
  under two seconds and labels all schedule averages as descriptive evidence
  from one outer replica.
- The complete unit suite passed before launch (`307 passed`, `2 skipped`) and
  after analysis (`308 passed`, `2 skipped`). The notebook's artifact-only
  execution validator also passed in under two seconds.

**Original gate decision:** predictive failure, not floor-driven failure. EDR
is a better-behaved diagnostic than instantaneous Fisher-risk control, but it
does not merit fresh-replica or Hybrid B32 confirmation against the strong
fixed `.025` incumbent. Because that EDR treatment also used `.025` during
cold start, it did not answer automatic calibration from the historical `.05`
starting point; the amendment below supersedes only that discovery
interpretation.

### Cold-Start Discovery Amendment

**Status:** Complete - mechanical discovery only (2026-08-22)

The original EDR treatment applied fixed $\pi=.025$ throughout its trend
cold start. That design remains a valid comparison against the best known
fixed policy, but it cannot answer whether EDR would have discovered a useful
action below the historically available fixed $\pi=.05$: the candidate value
was already embedded in the treatment.

Run one explicitly exploratory, paired amendment on the linear schedule, whose
EDR action trajectory was the most interpretable:

- change only the EDR cold-start and exact-zero fallback action from `.025` to
  `.05`;
- retain $H_\pi=4$, $\pi_{\min}=.01$, $\pi_{\max}=.95$, trend half-life
  $h=.05$, $m=8$, $K=100$, $p\in[0,.2]$, EWC-only, no LFU, rank-8-plus-
  diagonal Fisher summaries, and 50 L-BFGS iterations;
- reuse the exact linear initialization, observation stream, schedule, and
  non-treatment seeds;
- use fixed $\pi=.05$ as the primary discovery comparator;
- use the completed EDR cold-`.025` treatment as a cold-start sensitivity and
  fixed $\pi=.025$ only as a clearly labelled hindsight benchmark. Neither may
  determine whether discovery occurred.

The cold-start EDR and fixed-`.05` learners must be identical through the last
cold transition. After EDR becomes live, report its action trajectory and
paired predictive outcomes separately from the cold prefix. Define
**mechanical discovery** as a sustained, unclipped action below `.05` near the
end of the path. Define **predictive discovery** as lower post-cold NLL than
fixed `.05` without a material regression in environmental accuracy, digit-9
OvR accuracy, precision, recall, non-nine accuracy, or ECE. NLL remains primary
because it matches the local-KL motivation. For this one-replica development
gate, call an absolute secondary change larger than two percentage points
material; report the unthresholded differences as well.

This is a post-hoc mechanism clarification on one development replica. Even a
positive result would justify a fresh-replica confirmation, not an applied
promotion claim.

#### Amendment Completion Record

- Added the immutable paired bundle
  `cache/mnist_experiment/plan4/edr/discovery/bundles/plan4-edr-discovery__c7e7ed9d8438`.
  It reuses fixed `.05`, fixed `.025`, and EDR cold-`.025`, and adds exactly one
  linear EDR cold-`.05` treatment. The generated scientific configuration
  differs from the original EDR configuration only in experiment identity and
  cold-start action.
- The new EDR learner and fixed-`.05` learner have identical parameter hashes
  and predictive outcomes through all 25 cold transitions. The 100-step
  treatment completed in `63.7` seconds in a detached `tmux` session.
- **Mechanical discovery passed.** The first live action was `.0572`, the
  trajectory peaked at `.0923`, and `51.4%` of live actions were below `.05`.
  Its final ten actions ranged from `.02412` to `.02641` and averaged
  `.02501`, with no `.01` floor clipping.
- **Predictive discovery failed.** Across the 74 live outcomes, EDR minus fixed
  `.05` was `+1.488` NLL, `-4.25` percentage points environmental accuracy,
  `-1.48` points digit-9 OvR accuracy, `-5.23` points precision, `-3.99`
  points recall, `-4.30` points non-nine accuracy, and `+5.68` points ECE.
  The transient controller excursion therefore caused material closed-loop
  damage even though the coefficient recursion eventually identified the
  useful `.025` neighborhood.
- Wrote immutable analysis
  `cache/mnist_experiment/plan4/edr/discovery/analysis/edr_discovery__7b14c111ed7b`
  and extended [plan4_edr_results.ipynb](plan4_edr_results.ipynb) with paired
  action, cold/live outcome, NLL, and environmental-accuracy panels.
- The complete unit suite passed (`310 passed`, `2 skipped`), and the extended
  artifact-only notebook executed in under one second.

**Amended decision:** EDR contains a meaningful offline action-calibration
signal on the linear path: it rediscovers approximately `.025` from a `.05`
cold start. Its present online actuation is not predictively useful because
the transient actions alter an irreversible optimization trajectory. Preserve
the diagnostic for theory development, but do not promote the controller or
authorize fresh predictive replication in its current form.

### Sigmoid Stress And Health-Diagnostic Amendment

**Status:** Complete - diagnostic characterization (2026-08-22)

#### Goal

Determine how the cold-`.05` EDR signal changes as the progression path bends
more sharply, and develop continuous diagnostics that distinguish a useful
but unfinished local-risk estimator from noisy, delayed, clipped, or
predictively misaligned actuation. This amendment does not classify EDR as a
generally failed controller. The completed MNIST result rejects only the
current closed-loop treatment against its predictive gate.

#### Frozen Design

- Reuse the completed linear cold-`.05` EDR trajectory and add cold-`.05` EDR
  trajectories for logistic schedules with
  $\kappa\in\{32,64,128,256\}$.
- Retain $H_\pi=4$, $\pi_{\min}=.01$, $\pi_{\max}=.95$, controller trend
  half-life $h=.05$, $m=8$, $K=100$, $p\in[0,.2]$, EWC-only, no LFU,
  direct-EMA rank-8-plus-diagonal Fisher summaries, and 50 L-BFGS iterations.
- Keep the CPU runtime used by the completed Plan 4 controls and EDR
  trajectories. Reuse each schedule's exact initialization, observation
  stream, schedule, and non-treatment seeds.
- Use fixed $\pi=.05$ as the prospective comparator available before EDR
  nominated a lower operating region. Show fixed $\pi=.025$ only as a clearly
  labelled hindsight reference; it must not determine whether EDR discovers
  useful information.
- Preserve all completed artifacts. Every added trajectory and analysis is a
  new immutable run with a distinct configuration identity.

#### Continuous Diagnostics

For the accumulated EDR coefficients, define the normalized discounted risk

$$
\overline J_t(\pi)
=(1-\pi)^2\bar A_t+\pi^2\bar B_t,
$$

the risk curvature

$$
c_t=\bar A_t+\bar B_t,
$$

and the controller's claimed risk opportunity against the prospective fixed
baseline

$$
G_t^{\mathrm{risk}}
=\overline J_t(.05)-\overline J_t(\pi_t^{\mathrm{EDR}}).
$$

Writing the unconstrained minimizer as
$\pi_t^\star=\bar A_t/c_t$, the equivalent form is

$$
G_t^{\mathrm{risk}}
=c_t\left[
(.05-\pi_t^\star)^2
-(\pi_t^{\mathrm{EDR}}-\pi_t^\star)^2
\right].
$$

For an interior live action, this reduces to
$c_t(.05-\pi_t^{\mathrm{EDR}})^2$. Direct risk subtraction is used in the
analysis so the identity remains correct during cold start and clipping.

$G_t^{\mathrm{risk}}$ is an internal, same-state estimated Fisher-risk
advantage. It is nonnegative by construction and must never be presented as a
realized predictive improvement.

Analyze each schedule with the following unthresholded diagnostics:

1. **Risk identifiability and consequence:** plot $c_t$, the decomposition
   $(\bar A_t,\bar B_t)$, and $G_t^{\mathrm{risk}}$. Action movement where
   $c_t$ is nearly zero is weak evidence because the estimated objective is
   locally flat.
2. **Input-noise attenuation:** compare the instantaneous proposal
   $\widetilde\pi_t$ with $\pi_t^{\mathrm{EDR}}$ using their difference,
   total variations, and
   $\operatorname{TV}(\pi^{\mathrm{EDR}})/
   \operatorname{TV}(\widetilde\pi)$. This separates smoothing from delayed
   transmission of genuine signal.
3. **Response lag:** report lagged association and peak delay between EDR
   action and both the controlled forcing $|\Delta p_t|$ and the deployable
   geometric movement proxy
   $\widehat d_t^T\widehat{\mathcal I}_t\widehat d_t$ recorded as
   `signal_energy`. The latter is primary because $p_t$ is unavailable in a
   real application. Report the unclipped EDR recommendation separately from
   realized post-cold actuation so a cold-start gate cannot hide whether the
   estimator recognized the event.
4. **Hysteresis:** on logistic schedules, match rising- and falling-speed
   branches around the sigmoid center and report
   $\pi_{\mathrm{exit}}(v)-\pi_{\mathrm{entry}}(v)$ over matched progression
   speed $v=|\Delta p|$, together with the signed and absolute loop areas in
   the $(v,\pi)$ plane. This measures retained action after the forcing event
   has passed. Calculate separate loops for the unclipped recommendation and
   the live applied action; mark the latter unavailable when cold start leaves
   no shared entry/exit speed support.
5. **Settling and boundary pressure:** report the final-ten action mean,
   slope, and standard deviation; peak action; integrated action area above
   and below `.05`; and mean post-cold clipping pressure
   $T^{-1}\sum_t(\pi_{\min}-\pi_{t,\mathrm{unclipped}}^{\mathrm{EDR}})_+$.
   Keep bound occupancy as context, but do not use it alone to diagnose a
   floor-driven result.
6. **Realized alignment:** against the paired fixed-`.05` learner, plot
   $G_t^{\mathrm{risk}}$ beside the predictive gain
   produced after that action,
   $G_{t+1}^{\mathrm{NLL}}=\mathrm{NLL}_{.05,t+1}-
   \mathrm{NLL}_{\mathrm{EDR},t+1}$, and their cumulative trajectories. Also
   report environmental accuracy, digit-9 OvR accuracy, precision, recall,
   non-nine accuracy, and ECE. Treat agreement as descriptive alignment, not
   proof that the same-state local risk causes later path-dependent outcomes.

#### Scope

1. Add only the scientific configurations and immutable bundle entries needed
   for the four new sigmoid cold-`.05` EDR trajectories. Confirm exact
   cold-prefix parameter and prediction parity with each fixed-`.05` control.
2. Extend the artifact-only EDR analysis and notebook with small-multiple
   plots indexed by $\kappa$. Keep each predictive contrast to EDR and fixed
   `.05`; place fixed `.025` in separate hindsight panels where useful.
3. Calculate diagnostics from stored predictable coefficients, actions,
   schedule values, movement summaries, and paired classification outcomes.
   Do not train, repair artifacts, or reconstruct unavailable quantities in
   the notebook.
4. Compare diagnostic profiles across $\kappa$ without selecting a preferred
   steepness from predictive performance. Report the linear schedule beside
   the sigmoid family as a low-curvature reference.
5. Do not define a binary healthy-controller score or tune thresholds in this
   amendment. First identify which continuous diagnostics are stable,
   interpretable, and concordant across the schedule family.

#### Verification

- New and reused pairs share schedule, stream, initialization, and all
  non-treatment seeds; cold-`.05` trajectories are identical through the last
  cold transition.
- Recomputed actions and discounted risks agree exactly with stored
  coefficients, and $G_t^{\mathrm{risk}}$ agrees with direct quadratic-risk
  subtraction within numerical tolerance.
- Linear-path diagnostics reproduce the completed cold-`.05` discovery
  artifact without mutation.
- The analysis notebook remains artifact-only and labels the five schedules
  as one paired development replica, not five independent statistical units.
- Full continuous trajectories and unthresholded scalar summaries are
  retained so later diagnostic criteria can be predeclared rather than fitted
  to these outcomes.

#### Check-in

Review which diagnostics distinguish the linear mechanical-discovery profile
from the sigmoid profiles. Decide whether the evidence motivates a
predeclared health criterion, a redesigned two-timescale actuation rule, or
retaining EDR solely as an open local-risk diagnostic before beginning Phase
7.

#### Completion Record

- Added an immutable 15-entry paired bundle at
  `cache/mnist_experiment/plan4/edr/stress/bundles/plan4-edr-stress__76be1ac9375a`.
  It reuses 11 completed controls and the linear cold-`.05` discovery run and
  adds exactly four sigmoid cold-`.05` EDR trajectories. All entries validate
  as complete with exact schedule-specific replica pairing and CPU runtime.
- The four new trajectories completed in a detached `tmux` session in
  `61.3`, `61.3`, `61.9`, and `62.1` seconds. Every schedule retained exact
  parameter and predictive parity with fixed `.05` through the outcome after
  its last cold-start action.
- Corrected the risk-opportunity shortcut used during planning. The direct
  difference $\overline J_t(.05)-\overline J_t(\pi_t^{\mathrm{EDR}})$ is used
  so cold-start and clipped actions remain valid; the simpler squared-distance
  expression applies only to an interior unconstrained minimizer. The
  post-action predictive outcome at $t+1$, rather than the pre-action outcome
  at $t$, is paired with decision $t$.
- EDR attenuated instantaneous proposal variation on every schedule. The
  live-action total-variation ratios were `.571` on the linear path and
  `.360`, `.236`, `.367`, and `.418` for
  $\kappa=32,64,128,256$, respectively. No schedule had lower-bound occupancy
  or positive clipping pressure after cold start.
- The unclipped recommendation recognized every sigmoid event but retained a
  delayed response after speed fell. Mean matched-speed hysteresis gaps rose
  from `.0416` and `.0616` at $\kappa=32,64$ to `.0671` and `.1225` at
  $\kappa=128,256$. Live applied-action hysteresis is intentionally marked
  unavailable for the two sharpest schedules because cold start covers the
  rising branch; recommendation hysteresis remains observable.
- Only the linear treatment settled cleanly near the hindsight `.025`
  neighborhood (`.02501` final-ten mean). The sigmoid final-ten means were
  `.0363`, `.0584`, `.0523`, and `.0430`, and peak actions grew as high as
  `.2261` for $\kappa=256$.
- Estimated same-state Fisher-risk opportunity was positive on every path,
  but realized cumulative NLL gain against fixed `.05` was negative on every
  path. Mean EDR-minus-fixed-`.05` NLL ranged from `+.300` at $\kappa=32$ to
  `+1.284` at $\kappa=256$; cumulative risk/outcome correlations ranged from
  `-.77` to `-.88`. This is evidence of structured but predictively misaligned
  transient actuation, not floor-driven or purely random controller behavior.
- Wrote immutable schema-v1 analysis
  `cache/mnist_experiment/plan4/edr/stress/analysis/edr_stress__0782e586b53d`
  and extended [plan4_edr_results.ipynb](plan4_edr_results.ipynb) with action,
  risk, response-lag, hysteresis, settling, boundary, and predictive-alignment
  views. The artifact-only notebook executes in approximately two seconds.
- Focused tests and the notebook validator pass. The completed schedule family
  is one paired development replica and supplies no independent-replica
  inference or post-hoc binary health threshold.

#### Single-Trajectory Prequential Calibration Amendment

**Status:** Complete (2026-08-23)

To determine whether any useful health signal survives without a paired fixed
learner, use the decision-time uncertainty estimate to predict the next
accepted residual's Fisher-risk energy. Define

$$
r_t=u_t-\pi_t\widehat d_{t|t-1},\qquad
a_t=\pi_t^2\left(q_{t-1}+m_t^{-1}\right),
$$

$$
E_t=r_t^T\widehat{\mathcal I}_{t|t-1}r_t,qquad
P_t=a_t\widehat D_{t|t-1},
$$

and the causal calibration monitor

$$
C_t=
\frac{\operatorname{EMA}_{H=4}(E_t)}
     {\operatorname{EMA}_{H=4}(P_t)}.
$$

Both $\widehat{\mathcal I}_{t|t-1}$ and $\widehat D_{t|t-1}$ are frozen before
$u_t$ is observed. Thus $C_t$ is reconstructible from, and deployable on, one
trajectory. Values near one indicate residual-risk calibration; $C_t>1$ means
the realized residual risk exceeded its forecast, while $C_t<1$ means the
forecast was too large. Use $\log C_t$ for symmetric visualization. Update the
monitor during cold start, but exclude cold-start actions from scalar health
summaries.

This is a short-horizon staleness check, not an independent goodness-of-fit
test: $\widehat D$ is itself learned from earlier residual energies. Under a
stationary process and a sufficiently long common window, the ratio should
self-calibrate toward one. The four-update monitor asks whether that slower
historical estimate still predicts residual risk at the action timescale.

- The last-ten mean $C_t$ was `.746` on the linear path and `.167`, `.0537`,
  `.0343`, and `.0172` for $\kappa=32,64,128,256`. Sharper sigmoid events
  therefore leave increasingly stale, overpredictive uncertainty scales after
  the event.
- Mean absolute $\log C_t$ ranked the four sigmoid paths in steepness order
  for monitor half-lives of 2, 4, and 8 accepted updates. The conclusion is not
  an artifact of the chosen four-update display timescale.
- As a retrospective validation only, mean absolute $\log C_t$ correlated
  `.969` with recommendation hysteresis and `.981` with mean NLL regression
  against fixed `.05` across the four sigmoid paths. Those correlations use one
  development replica and are descriptive, not inferential thresholds.
- The linear path is the important counterexample. Its residual-risk forecast
  was substantially better calibrated than the sigmoid forecasts, yet its EDR
  learner still underperformed fixed `.05`. The monitor diagnoses stale local
  risk coefficients; it cannot certify predictive benefit or account for the
  irreversible consequences of earlier actions.
- Wrote immutable schema-v2 analysis
  `cache/mnist_experiment/plan4/edr/stress/analysis/edr_stress__8f56773be109`
  and added the causal trajectories, scalar table, and half-life sensitivity to
  [plan4_edr_results.ipynb](plan4_edr_results.ipynb). No learner was rerun.

**Diagnostic decision:** EDR remains a work-in-progress local Fisher-risk
estimator with useful offline action-calibration signal. The most informative
health checks in this pilot are the deployable prequential calibration ratio,
recommendation hysteresis, tail settling, and the divergence between
accumulated internal risk opportunity and post-action predictive gain. Do not
promote the present MNIST closed-loop actuation rule; retain the continuous
diagnostics for a future predeclared controller redesign.

## Phase 7: Theory Integration and Historical Decision

### Goal

Integrate only experimentally supported changes into the mathematical account
and decide whether any older controller experiments warrant repetition.

### Scope

1. Integrate the completed Phase 5 and Phase 6 evidence without rewriting the
   mathematical overview around an unsuccessful treatment. Distinguish the
   instantaneous Fisher-risk controller, EDR controller, and accepted fixed
   policy explicitly.
2. If Phase 6 demonstrates dynamic or automatic-calibration value worth
   preserving,
   update `mathematical_overview.ipynb` to present:
   - Euclidean parameter-MSE and Fisher/local-KL risk as distinct objectives;
   - the consistently Fisher-weighted signal and covariance terms;
   - predictable metric freezing and the direct residual-energy estimator;
   - the assumed nonsingularity of the theoretical Fisher separately from
     possible singularity of finite-sample estimates; and
   - the fact that LFU remainder control and numerical optimization remain in
     the fixed Euclidean parameter chart.
3. Do not introduce intrinsic Brownian motion, geodesic optimization,
   connection corrections, or natural-gradient updates unless a later result
   specifically requires them. The local quadratic experiment does not need
   those constructions.
4. Evaluate historical reruns by decision value. Revisit only comparisons for
   which Fisher-risk actuation could change an applied conclusion; preserve
   all existing artifacts and label cross-schema comparisons explicitly.
5. If confirmation is justified, specify fresh replica counts, precision
   stopping, and Hybrid B32 as a separate follow-up plan rather than silently
   extending the development pilot.

### Verification

- Mathematical notation distinguishes the true Fisher from its stored EWC
  estimate and parameter displacement from local predictive divergence.
- No claimed invariance relies on damping, numerical rank, or an unrecorded
  coordinate transformation.
- Documentation states clearly whether instantaneous and EDR Fisher-risk
  adaptation are promoted, automatic-calibration-only, floor-driven, or
  retired.
- Historical experimental conclusions remain reproducible from their original
  immutable artifacts.

### Final Check-in

Decide whether EDR adaptive $\pi$ belongs in the practical baseline, deserves
a larger confirmation plan, remains only an automatic-calibration diagnostic,
or closes Plan 4 as a scientifically useful negative result.
