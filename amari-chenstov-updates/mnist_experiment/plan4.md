# Implementation Plan 4: Adaptive Pi Under Variable Speed

Plan 4 follows the completed deployment baseline in [plan3.md](plan3.md). It
asks one narrow question:

> Can the applied controller adjust $\pi_t$ usefully when the true parameter
> moves at a nonconstant rate, without retuning one fixed $\pi$ for each
> environment?

This is a variable-speed experiment, not yet a claim about intrinsic manifold
curvature. The schedule changes the parameterization in time of the same
one-dimensional path

$$
p\longmapsto\theta^\star(p).
$$

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
| 2 | Design-oracle feasibility screen | Complete |
| 3 | Predictable-controller pilot | Not started - Phase 2 stop gate |
| 4 | Applied confirmation and decision | Not started - Phase 2 stop gate |

## Frozen Handoff

Unless a phase check-in explicitly reopens one item, preserve:

- the canonical 512-parameter CNN and full-network likelihood Fisher;
- $m=8$ current observations and $K=100$ environmental steps;
- the application-focused path segment $p_0=0$ to $p_{K-1}=.2$;
- 50 L-BFGS inner iterations per learner update;
- rank-8-plus-diagonal Fisher summaries;
- direct EMA Fisher updates with no LFU or HVP calculation;
- fixed $\pi=.05$ as the deployment baseline;
- plug-in $h=.20$, $\pi_{\min}=.05$, and $\pi_{\max}=.95$ initially;
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

EWC-only provides the principal theoretically matched controller test. If it
passes Phase 3, fixed and adaptive Hybrid B32 are required in Phase 4 rather
than optional. For that transfer test, $m_t=8$ continues to count only the
independent current arrivals in the controller covariance model. Do not count
the 32 reused replay observations as independent new samples. Until a
replay-aware covariance derivation exists, describe adaptive Hybrid B32 as an
applied transfer test rather than an optimal hybrid controller.

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
opportunity to justify Phase 3 predictive training. Retain fixed $\pi=.05$ for
the practical baseline. Any future adaptive experiment should begin with a
new mathematical treatment that creates materially larger signal relative to
covariance, rather than escalating the present sigmoid screen.

## Phase 3: Predictable-Controller Pilot

### Goal

Test whether the deployable plug-in controller can recover the oracle
opportunity before spending compute on the Hybrid B32 application.

### Scope

1. Run paired EWC-only development replicas on linear and selected logistic
   schedules.
2. Compare adaptive $h=.20$ with the frozen `.05` policy and the Phase 2 fixed
   bracket.
3. Begin with three paired replicas. Add two only if actuation diagnostics are
   coherent but predictive uncertainty prevents a gate decision.
4. Compare plug-in and oracle $\pi_t$ trajectories for lag, RMSE, correlation,
   bound occupancy, and response around maximum schedule speed.
5. If oracle opportunity exists but $h=.20$ fails only through identifiable
   lag, run one predeclared half-life sensitivity set
   $h\in\{.05,.10,.20\}$. Do not cross it with additional schedule shapes.

### Pilot gate

Promote the controller only if:

- it leaves the lower bound no later than the second accepted update in the
  high-speed window, allowing the unavoidable lag of a predictable policy;
- its actuation direction agrees with the design oracle;
- adaptive environmental-accuracy AUC exceeds fixed `.05` in mean paired
  results on the logistic schedule; and
- no fixed bracket member clearly dominates it on both linear and logistic
  schedules.

Failure with no oracle opportunity invalidates the schedule, not the
controller. Failure despite robust oracle opportunity classifies the current
plug-in controller as `diagnostic-only` and ends predictive escalation.

### Verification

- Pilot artifacts reproduce every decision from prior state.
- Linear-path behavior remains consistent with Plan 3.
- No result is pooled with Plan 3's completed confirmation.
- The pilot notebook shows individual and mean trajectories, not only AUCs.

### Check-in

Decide whether to stop, revise only the estimator, or promote one frozen
adaptive configuration into the applied Hybrid B32 confirmation.

## Phase 4: Applied Confirmation and Decision

### Goal

Decide whether adaptive $\pi$ is a useful deployment policy rather than merely
an oracle construction or instability indicator.

### Scope

1. Freeze the schedule, controller half-life, bounds, and fixed comparison
   grid selected before confirmation.
2. Run fresh paired replicas for:
   - fixed and adaptive EWC;
   - fixed and adaptive Hybrid B32; and
   - only the fixed bracket needed to identify best-fixed regret.
3. Use blocks of five independent replicas with a predeclared minimum of ten
   and maximum of twenty. Stop on CI precision, not first significance.
4. Treat adaptive-minus-fixed Hybrid B32 event-window
   environmental-accuracy AUC on the selected logistic schedule as primary.
   Full-$p\in[0,.2]$ environmental-accuracy AUC is the principal broader
   secondary outcome. Report digit-9 OvR accuracy, precision, recall, NLL,
   ECE, retention, and resource costs as additional required outcomes.
5. Report two fixed-policy references:
   - frozen `.05`, representing no retuning from the gentle path; and
   - the best fixed policy in hindsight, representing schedule-specific
     tuning unavailable to a portable agent.
6. Build `adaptive_results.ipynb` as an artifact-only notebook with no more
   than four conditions per panel, clear confidence bands, event-centered
   views, and exposure-based low-data plots.

### Decision rule

- **Promote:** adaptive Hybrid B32 improves on frozen `.05`, remains close to
  the best fixed policy across both schedules, and introduces no material
  secondary-metric failure.
- **Diagnostic only:** $\pi_t$ detects the speed change but does not improve
  predictive outcomes or reacts too late.
- **Stop:** the oracle opportunity is absent, the plug-in estimate is unstable,
  or one fixed policy is robustly as good with lower complexity.

### Verification

- Confirmation uses fresh independent replicas and immutable run directories.
- Every claim is visible in a trajectory plot with uncertainty.
- Schedule, data, memory, and compute effects remain separate.
- Fixed and adaptive conditions share initialization, uniforms, data order,
  optimizer settings, and all non-treatment seeds within each replica.

### Final Check-in

Decide whether adaptive $\pi$ belongs in the practical baseline, remains a
research diagnostic, or should be revisited only after a genuinely turning
multi-coordinate task path is derived.
